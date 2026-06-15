r"""``KakeyaLatticeQuantizedCache`` — a ``DynamicCache`` subclass that
stores **lattice indices**, not reconstructed bf16 tensors.

This is the implementation that delivers *real* HBM savings, in
contrast to :class:`~kakeyalattice.hf.KakeyaLatticeCache` (the
reconstruction-quality probe), which round-trips K/V through the codec
but stores the reconstructed bf16 tensor.

Storage shape per layer per (batch, num_kv_heads, seq):
    q_lat  : int8 [..., head_dim]            ← head_dim integers in [-Q, Q]
    norms  : float16 [..., 1]                ← per-vector L2 norm
    qmax   : float16 [..., 1]                ← per-vector lattice scale

Total bytes per K- or V-vector at head_dim=128, Q ≤ 127:
    head_dim * 1  (int8 indices)
  + 2            (fp16 norm)
  + 2            (fp16 qmax)
  = head_dim + 4 = 132 bytes

vs bf16 baseline: head_dim * 2 = 256 bytes.

Real ratio at head_dim=128 with Q ≤ 127: 256 / 132 = **1.94×**.

This is **strictly less than** the bit-rate compression ratio
(2.4×-2.8×) advertised by the codec at the same Q, because int8
storage wastes ~2 bits per coordinate (a Q=38 value needs only
~6.3 bits but int8 uses 8). A future v1.6 bit-packed int storage
recovers the gap; the int8 implementation here trades that ~25 %
ratio for a clean PyTorch storage type and zero kernel work.

For Q > 127 the implementation falls back to int16 storage, which
gives 1.00× ratio (i.e. no savings) and is therefore not useful;
use the canonical reconstruction-only ``KakeyaLatticeCache`` for
near-lossless Q=152 operating points instead.

Design decisions
----------------
1. **Subclass DynamicCache, NOT QuantizedCache.** ``QuantizedCache``
   in transformers >= 4.45 assumes per-axis scalar quantization with
   a specific (codes, scales, zeros) protocol. Our codec has
   per-vector state (qmax, ‖x‖) that does not fit cleanly into that
   protocol without a wrapper layer; subclassing DynamicCache and
   overriding ``update`` is more direct and avoids hidden
   dequantization paths.

2. **Lazy concat.** We store new tokens' (q_lat, norms, qmax) as a
   per-call entry in a list, not a single tensor concatenated each
   call. On read we cat once. This matches DynamicCache's actual
   storage pattern (``key_cache: list[Tensor]``) and avoids
   quadratic-cost concatenation during long-prefill prompt fills.

3. **Single decode per attention read.** The attention layer calls
   ``cache.update(new_k, new_v, layer_idx)`` once per layer per
   forward. Our override:
       (a) encodes new K/V → ints,
       (b) appends to per-layer storage,
       (c) decodes ALL stored entries for that layer to bf16,
       (d) returns the decoded bf16 to attention.
   So peak HBM during step (c) does include the decoded bf16 of all
   tokens — but that bf16 tensor is what attention reads anyway. The
   *persistent* state between calls is the int8 storage, which is
   what HBM-bound deployments measure.

4. **No subclass of QuantizedCacheConfig.** We surface ``q_range``
   and ``variant`` directly on the cache constructor, mirroring
   ``KakeyaLatticeCache`` so users can swap one class for the other
   without changing their call site.

Non-goals
---------
- Bit-packed storage (saves the int8-vs-claimed gap; planned for v1.6).
- Native paged-attention integration (vLLM follow-up; this class
  works through transformers' eager / SDPA / FA path).
- Per-axis residual quantization (KIVI-style). The codec already
  handles outliers via per-vector qmax.

See also
--------
- ``kakeyalattice.hf.KakeyaLatticeCache`` — the roundtrip-only variant
  (zero real HBM savings; useful as a reconstruction-quality probe).
- ``kakeyalattice.hf.test_quantized_cache`` — empirical byte-count
  tests verifying the real HBM ratio.
"""
from __future__ import annotations

import logging
import math
import warnings
from typing import Any

import torch

logger = logging.getLogger("kakeyalattice.hf")


def _require_transformers():
    try:
        from transformers import DynamicCache  # noqa: F401
        return DynamicCache
    except ImportError as e:
        raise ImportError(
            "kakeyalattice.hf.KakeyaLatticeQuantizedCache requires the "
            "'transformers' package. Install with: pip install kakeyalattice[hf]"
        ) from e


_DynamicCache = _require_transformers()


# ---------------------------------------------------------------------------
# Codec encode / decode primitives.  These factor LatticeCodebook.roundtrip
# into two halves so we can persist the intermediate index tensor.
# ---------------------------------------------------------------------------

def encode_to_indices(codec: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the encode half of the codec.

    Mirrors ``LatticeCodebook.roundtrip`` steps 1–6, returning the
    integer lattice coordinates plus the two fp16 scalars needed to
    decode back to the original space.

    Args:
        codec: a ``V14KakeyaZamirLatticeGPU`` or ``V15KakeyaZamirE8GPU``
            instance.
        x: tensor with trailing dim = codec.D_shape; any leading
            shape is allowed.

    Returns:
        (q_lat, norms_fp16, qmax_fp16):
            q_lat: int tensor [..., D] with values in [-q_range, +q_range].
                Stored as int8 if q_range <= 127 else int16.
            norms_fp16: float16 [..., 1].
            qmax_fp16: float16 [..., 1].
    """
    assert x.shape[-1] == codec.D_shape, (
        f"expected last dim {codec.D_shape}, got {x.shape[-1]}"
    )
    leading = x.shape[:-1]
    flat = x.reshape(-1, codec.D_shape).to(torch.float32)
    eps = torch.finfo(flat.dtype).eps

    # 1. unit-normalise + fp16 norm
    norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
    norms_fp16 = norms.to(torch.float16)
    unit = flat / norms

    # 2. Hadamard rotation
    y = unit @ codec.H

    # 3. per-vector qmax + fp16
    qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
    qmax_fp16 = qmax.to(torch.float16)
    scale = qmax_fp16.to(torch.float32) / float(codec.q_range)

    # 4. scale to lattice coords
    y_scaled = y / scale

    # 5. closest-lattice-point per block
    y_blocks = y_scaled.reshape(-1, codec.K_blocks, codec.block_dim)
    q_lat = codec._closest_lattice_point(y_blocks)

    # 6. defensive clamp (parity-flip may push slightly out of range)
    q_lat = q_lat.clamp(-codec.q_range, codec.q_range)

    # E8 lattice outputs half-integer values (from the D8 + (1/2)*ones
    # coset; see Conway-Sloane Alg 5).  To preserve the half-integer
    # bit in integer storage we multiply by 2 before casting; decode
    # multiplies by 0.5 to recover.  D4 outputs are pure integers
    # (D4 = {x in Z^4 : sum even}) and need no doubling.
    #
    # Storage range after doubling for E8: 2 * q_range.  int8 fits
    # values in [-128, 127], so the int8 ceiling becomes Q <= 63 for
    # E8 (vs Q <= 127 for D4).  Beyond that we fall back to int16,
    # which doubles the storage cost and eliminates the HBM win.
    needs_half_int_doubling = (codec.block_dim == 8)  # E8 only
    if needs_half_int_doubling:
        q_int = (q_lat * 2.0).round()
        effective_max = 2 * codec.q_range
    else:
        q_int = q_lat
        effective_max = codec.q_range

    if effective_max <= 127:
        q_lat_storage = q_int.to(torch.int8)
    else:
        q_lat_storage = q_int.to(torch.int16)

    # Reshape q_lat back to flat-D + leading
    q_lat_flat = q_lat_storage.reshape(-1, codec.D_shape)
    return (
        q_lat_flat.reshape(*leading, codec.D_shape),
        norms_fp16.reshape(*leading, 1),
        qmax_fp16.reshape(*leading, 1),
    )


def decode_from_indices(
    codec: Any,
    q_lat: torch.Tensor,
    norms_fp16: torch.Tensor,
    qmax_fp16: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run the decode half of the codec.

    Mirrors ``LatticeCodebook.roundtrip`` steps 7–9.

    Args:
        codec: same codec object used in ``encode_to_indices``.
        q_lat: int tensor [..., D] of lattice coordinates.
        norms_fp16: float16 [..., 1].
        qmax_fp16: float16 [..., 1].
        out_dtype: target dtype for the reconstructed tensor
            (typically bf16 to match the model's KV dtype).

    Returns:
        Reconstructed tensor with the same leading shape as q_lat,
        trailing dim = codec.D_shape, dtype = out_dtype.
    """
    leading = q_lat.shape[:-1]
    q_lat_flat = q_lat.reshape(-1, codec.D_shape).to(torch.float32)
    norms_flat = norms_fp16.reshape(-1, 1).to(torch.float32)
    qmax_flat = qmax_fp16.reshape(-1, 1).to(torch.float32)
    scale = qmax_flat / float(codec.q_range)

    # Mirror the encode-side doubling for E8 half-integer coset.
    if codec.block_dim == 8:
        q_lat_flat = q_lat_flat * 0.5

    # 7. rescale lattice coordinates back to y space
    y_hat = q_lat_flat * scale

    # 8. inverse Hadamard
    unit_hat = y_hat @ codec.H

    # 9. restore L2 norm
    x_hat = unit_hat * norms_flat
    return x_hat.to(out_dtype).reshape(*leading, codec.D_shape)


# ---------------------------------------------------------------------------
# KakeyaLatticeQuantizedCache — DynamicCache subclass with REAL storage savings.
# ---------------------------------------------------------------------------

class KakeyaLatticeQuantizedCache(_DynamicCache):
    """A :class:`~transformers.DynamicCache` subclass that **stores
    lattice indices** instead of bf16 reconstructed K/V.

    This is the variant that produces real HBM savings between
    ``update()`` calls. For a reconstruction-only variant that keeps
    DynamicCache's bf16 storage (useful for measuring codec
    reconstruction error in isolation), use
    :class:`~kakeyalattice.hf.KakeyaLatticeCache`.

    Args:
        variant: "d4" or "e8".
        q_range: per-coordinate lattice range. Use a value <= 127
            to keep int8 storage; values > 127 fall back to int16
            (less HBM benefit).
        num_hidden_layers, head_dim: from ``model.config``.
        device: where the Sylvester-Hadamard matrix lives. Should
            match the model's device.
        boundary: number of first/last layers to skip (raw bf16 KV).
        strict: raise on head_dim incompatible with the lattice block
            dim (True, default) vs fall back to raw KV (False).
        out_dtype: dtype to decode K/V back to on read. Defaults to
            bfloat16 since that is what modern attention kernels
            expect.

    Memory accounting:
        Per K- or V-vector at head_dim=D and Q <= 127:
            int8 indices: D bytes
            fp16 norm:    2 bytes
            fp16 qmax:    2 bytes
            total:        D + 4 bytes
        vs bf16 baseline: 2D bytes.

        At D=128: 132 bytes vs 256 bytes → **1.94x compression**.
        At D=64:   68 bytes vs 128 bytes → **1.88x compression**.
        At D=256: 260 bytes vs 512 bytes → **1.97x compression**.

        The discrepancy between this real ratio and the codec's
        bit-rate ratio (~2.4x at Q=38) is the int8-vs-6-bit
        overhead; bit-packed v1.6 will close it.
    """

    _VALID_VARIANTS = ("d4", "e8")

    def __init__(
        self,
        variant: str = "e8",
        q_range: int = 38,
        num_hidden_layers: int | None = None,
        head_dim: int | None = None,
        device: str | torch.device = "cuda",
        boundary: int = 0,
        strict: bool = True,
        out_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        if variant.lower() not in self._VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {self._VALID_VARIANTS}, got {variant!r}"
            )
        if num_hidden_layers is None or head_dim is None:
            raise ValueError(
                "KakeyaLatticeQuantizedCache requires num_hidden_layers and "
                "head_dim (pass model.config.num_hidden_layers and "
                "model.config.head_dim)."
            )

        self.variant = variant.lower()
        self.q_range = int(q_range)
        self.num_hidden_layers = int(num_hidden_layers)
        self.head_dim = int(head_dim)
        self.device = torch.device(device)
        self.boundary = int(boundary)
        self.strict = bool(strict)
        self.out_dtype = out_dtype

        block_dim = 4 if self.variant == "d4" else 8
        self._block_dim = block_dim
        is_pow2 = self.head_dim > 0 and (self.head_dim & (self.head_dim - 1)) == 0
        self._supports_lattice = (
            self.head_dim % block_dim == 0 and is_pow2
        )

        if not self._supports_lattice:
            reasons = []
            if self.head_dim % block_dim != 0:
                reasons.append(f"head_dim % {block_dim} != 0")
            if not is_pow2:
                reasons.append(f"head_dim={self.head_dim} is not a power of 2")
            msg = (
                f"KakeyaLatticeQuantizedCache(variant={self.variant!r}) "
                f"constraint violated: {', '.join(reasons)}.  "
            )
            if self.strict:
                msg += (
                    "Pass strict=False to skip the codec and fall back to "
                    "raw bf16 KV (no compression on this model)."
                )
                raise ValueError(msg)
            else:
                warnings.warn(msg + " strict=False: falling back to raw KV.",
                              UserWarning, stacklevel=2)

        # int8 ceiling: D4 is Q<=127 (integer-valued lattice points);
        # E8 is Q<=63 (half-integer-valued, doubled before int8 cast).
        # Beyond the ceiling we silently fall back to int16, which
        # eliminates the HBM win.
        int8_ceiling = 63 if self.variant == "e8" else 127
        if self.q_range > int8_ceiling:
            warnings.warn(
                f"q_range={self.q_range} > {int8_ceiling} for variant "
                f"{self.variant!r}; storage falls back to int16. Real HBM "
                f"savings will be ~0%. Use q_range <= {int8_ceiling} "
                f"(typical production point is q_range=38) or use the "
                f"reconstruction-only KakeyaLatticeCache for Q=152.",
                UserWarning, stacklevel=2,
            )

        # One codec per layer.
        self._codecs: list[Any | None] = []
        if self._supports_lattice:
            self._init_codecs()

        # Per-layer per-call storage of (q_lat, norms, qmax) tuples for
        # K and V separately.  Each list grows by one entry per update().
        self._k_quant_entries: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(self.num_hidden_layers)
        ]
        self._v_quant_entries: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(self.num_hidden_layers)
        ]
        # For boundary layers and incompatible models we store raw bf16
        # like normal DynamicCache.  We re-use the parent's key_cache /
        # value_cache slots for those layers; the int-storage layers
        # don't touch them.

        # Audit counters.
        self.codec_fired_per_layer: dict[int, int] = {}
        self.skip_fired_per_layer: dict[int, int] = {}

    # ----- codec management -----

    def _init_codecs(self) -> None:
        if self.variant == "d4":
            from kakeyalattice import V14KakeyaZamirLatticeGPU as CodecCls
        else:
            from kakeyalattice import V15KakeyaZamirE8GPU as CodecCls

        self._codecs = []
        for layer_idx in range(self.num_hidden_layers):
            if self._is_boundary_layer(layer_idx):
                self._codecs.append(None)
            else:
                self._codecs.append(CodecCls(
                    D=self.head_dim,
                    q_range=self.q_range,
                    device=str(self.device),
                ))

    def _is_boundary_layer(self, layer_idx: int) -> bool:
        if self.boundary <= 0:
            return False
        return (
            layer_idx < self.boundary
            or layer_idx >= (self.num_hidden_layers - self.boundary)
        )

    # ----- HBM accounting helpers -----

    def kv_storage_bytes(self) -> int:
        """Return the total bytes currently held in quantized storage
        (int indices + fp16 norms + fp16 qmaxes) plus any raw bf16
        bytes held in fallback layers.

        Used by tests to verify real HBM savings.
        """
        total = 0
        # quantized layers
        for entries in self._k_quant_entries:
            for q_lat, n, m in entries:
                total += q_lat.element_size() * q_lat.numel()
                total += n.element_size() * n.numel()
                total += m.element_size() * m.numel()
        for entries in self._v_quant_entries:
            for q_lat, n, m in entries:
                total += q_lat.element_size() * q_lat.numel()
                total += n.element_size() * n.numel()
                total += m.element_size() * m.numel()
        # fallback bf16 layers (boundary + non-supports_lattice).
        # Transformers >=5 uses cache.layers[i].keys/.values; older
        # versions use cache.key_cache / cache.value_cache.
        if hasattr(self, "layers"):
            for layer in self.layers:
                for attr in ("keys", "values"):
                    t = getattr(layer, attr, None)
                    if t is not None:
                        total += t.element_size() * t.numel()
        else:
            for layer in getattr(self, "key_cache", []):
                if layer is not None and hasattr(layer, "element_size"):
                    total += layer.element_size() * layer.numel()
            for layer in getattr(self, "value_cache", []):
                if layer is not None and hasattr(layer, "element_size"):
                    total += layer.element_size() * layer.numel()
        return total

    # ----- DynamicCache interface override -----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode new K/V to int indices, append to per-layer storage,
        return the decoded bf16 of ALL stored entries for this layer.
        """
        # Fallback: codec disabled (incompatible model with strict=False,
        # or boundary layer).  Behave exactly like DynamicCache.
        if (
            not self._supports_lattice
            or layer_idx >= len(self._codecs)
            or self._codecs[layer_idx] is None
        ):
            self.skip_fired_per_layer[layer_idx] = (
                self.skip_fired_per_layer.get(layer_idx, 0) + 1
            )
            return super().update(
                key_states, value_states, layer_idx, *args, **kwargs
            )

        codec = self._codecs[layer_idx]
        self.codec_fired_per_layer[layer_idx] = (
            self.codec_fired_per_layer.get(layer_idx, 0) + 1
        )

        # Encode the new K/V to int indices.  We preserve the leading
        # shape (batch, num_kv_heads, new_seq) and only touch the
        # trailing head_dim.
        k_q, k_norms, k_qmax = encode_to_indices(codec, key_states)
        v_q, v_norms, v_qmax = encode_to_indices(codec, value_states)

        # Append to per-layer storage.  Each entry corresponds to one
        # update() call.  At read time we cat along the seq dim (axis -2
        # of the bf16 shape == axis -2 of (q, norms, qmax) since they
        # share the leading shape [batch, nkv, seq]).
        self._k_quant_entries[layer_idx].append((k_q, k_norms, k_qmax))
        self._v_quant_entries[layer_idx].append((v_q, v_norms, v_qmax))

        # Concat across all stored entries (cheap: views, then one cat
        # on dim -2 for each component).  Returning concatenated bf16
        # K/V matches DynamicCache.update's contract.
        k_q_all = torch.cat([e[0] for e in self._k_quant_entries[layer_idx]], dim=-2)
        k_n_all = torch.cat([e[1] for e in self._k_quant_entries[layer_idx]], dim=-2)
        k_m_all = torch.cat([e[2] for e in self._k_quant_entries[layer_idx]], dim=-2)
        v_q_all = torch.cat([e[0] for e in self._v_quant_entries[layer_idx]], dim=-2)
        v_n_all = torch.cat([e[1] for e in self._v_quant_entries[layer_idx]], dim=-2)
        v_m_all = torch.cat([e[2] for e in self._v_quant_entries[layer_idx]], dim=-2)

        k_bf = decode_from_indices(codec, k_q_all, k_n_all, k_m_all, self.out_dtype)
        v_bf = decode_from_indices(codec, v_q_all, v_n_all, v_m_all, self.out_dtype)
        return k_bf, v_bf

    # transformers' generation loop calls get_seq_length to size the
    # attention mask.  DynamicCache infers this from key_cache[layer].shape[-2].
    # We override to use our int storage instead.
    def get_seq_length(self, layer_idx: int = 0) -> int:
        if (
            self._supports_lattice
            and layer_idx < len(self._codecs)
            and self._codecs[layer_idx] is not None
            and layer_idx < len(self._k_quant_entries)
            and len(self._k_quant_entries[layer_idx]) > 0
        ):
            return sum(e[0].shape[-2] for e in self._k_quant_entries[layer_idx])
        # Fall back to parent's bf16 length (boundary / unsupported layers).
        try:
            return super().get_seq_length(layer_idx)
        except Exception:
            return 0

    # ----- diagnostics -----

    def __repr__(self) -> str:
        return (
            f"KakeyaLatticeQuantizedCache(variant={self.variant!r}, "
            f"q_range={self.q_range}, num_hidden_layers={self.num_hidden_layers}, "
            f"head_dim={self.head_dim}, boundary={self.boundary}, "
            f"supports_lattice={self._supports_lattice}, "
            f"out_dtype={self.out_dtype})"
        )


__all__ = [
    "KakeyaLatticeQuantizedCache",
    "encode_to_indices",
    "decode_from_indices",
]
