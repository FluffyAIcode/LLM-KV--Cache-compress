r"""``KakeyaLatticeCache`` — a ``DynamicCache`` subclass that applies a
per-token nested-lattice codec roundtrip on every K/V write.

Design decisions
----------------
1. **Subclass, not wrapper**. We inherit from
   ``transformers.DynamicCache`` and only override ``update``. All
   other behaviour (``reorder_cache``, ``crop``, ``get_seq_length``,
   ``batch_select_indices``, offloading, etc.) is inherited unchanged.
   This keeps us forward-compatible with transformers API changes AND
   passes ``isinstance(cache, Cache)`` checks in
   ``GenerationMixin._prepare_cache_for_generation`` and friends.

2. **Roundtrip, not store-as-index**. We run ``codec.roundtrip(k)`` and
   ``codec.roundtrip(v)`` before handing control back to the parent
   ``update``. The result is BF16/FP16/FP32 in the same dtype as the
   input, so FlashAttention / SDPA / eager attention all see a normal
   KV tensor. Memory savings at the bytes-in-HBM level are
   **nominal** unless a downstream kernel changes the storage dtype
   (the vLLM backend does this). For pure transformers usage the
   deliverable is reconstruction-accuracy, not HBM footprint.

3. **Per-token, state-independent codec**. Our V14/V15 codecs have no
   rolling per-vector state across tokens: ``roundtrip(new_token_kv)``
   produces identical output whether called on a single vector at
   decode time or on a batch of vectors at prefill. Decode-time
   overhead is therefore constant per token.

4. **Graceful fallback on non-divisible head_dim**. D4 requires
   ``head_dim % 4 == 0``; E8 requires ``% 8 == 0``. With ``strict=True``
   (default) we raise ``ValueError`` immediately with a clear message.
   With ``strict=False`` we log a warning and pass K/V through unchanged.

5. **Boundary-layer protection (optional)**. Matches the paper's
   ``k=2`` boundary-skip policy: the first and last ``boundary``
   transformer layers keep raw KV, everyone else runs the codec.
   Defaults to 0 (no skip) so transformers users get full-model
   compression out of the box.

Non-goals
---------
- We do NOT subclass ``QuantizedCache``. That class assumes a specific
  storage abstraction (store indices, dequantise on read). Our codec
  is a pure roundtrip; storing lattice indices would require a custom
  attention kernel that can dequantise at attention time, which is
  NOT what transformers' eager / SDPA / FlashAttention paths expect.
  Revisit ``QuantizedCache`` integration once a fused decode kernel
  lands.
- We do NOT intercept the attention computation. The codec only sees
  K and V tensors at write time.
"""
from __future__ import annotations

import logging
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
            "kakeyalattice.hf.KakeyaLatticeCache requires the 'transformers' "
            "package. Install with:  pip install kakeyalattice[hf]"
        ) from e


_DynamicCache = _require_transformers()


class KakeyaLatticeCache(_DynamicCache):
    """A :class:`~transformers.DynamicCache` subclass that applies a
    nested-lattice codec roundtrip on every K/V write.

    Instantiate once and pass to ``model.generate`` as
    ``past_key_values=cache``, or to forward calls directly.

    Args:
        variant: ``"d4"`` (block-dim 4) or ``"e8"`` (block-dim 8).
        q_range: per-coordinate lattice range. Canonical points for
            head_dim=128:  ``q_range=10`` (aggressive, ~3.6x CR),
            ``q_range=38`` (balanced, ~2.5x CR, **recommended**),
            ``q_range=152`` (near-lossless, ~1.9x CR). For other
            head dims the compression ratio scales linearly.
        num_hidden_layers: number of transformer layers (one codec
            instance per layer). Get from ``model.config``.
        head_dim: attention head dimension. Get from ``model.config``.
        device: where to allocate the Sylvester--Hadamard matrix.
            Should match the model's device.
        boundary: number of transformer layers at the start AND end
            that skip the codec (raw KV). Defaults to 0. Paper's
            canonical snapshot-mode setting is 2.
        strict: if True (default), raise on ``head_dim`` incompatible
            with the lattice's block dim. If False, warn and pass
            K/V through unchanged for incompatible models.
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
    ):
        # Initialise the underlying DynamicCache first so its internal
        # attributes (key_cache, value_cache, seen_tokens, …) are
        # properly set up and our inherited methods work.
        super().__init__()

        if variant.lower() not in self._VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {self._VALID_VARIANTS}, got {variant!r}"
            )
        if num_hidden_layers is None or head_dim is None:
            raise ValueError(
                "KakeyaLatticeCache requires num_hidden_layers and head_dim "
                "(pass model.config.num_hidden_layers and "
                "model.config.head_dim; these values are needed to "
                "instantiate per-layer codec objects at construction time)"
            )

        self.variant = variant.lower()
        self.q_range = int(q_range)
        self.num_hidden_layers = int(num_hidden_layers)
        self.head_dim = int(head_dim)
        self.device = torch.device(device)
        self.boundary = int(boundary)
        self.strict = bool(strict)

        block_dim = 4 if self.variant == "d4" else 8
        self._block_dim = block_dim
        # Constraints:
        #   1. head_dim % block_dim == 0 (for lattice block structure)
        #   2. head_dim is a power of 2 (for Sylvester--Hadamard rotation)
        is_pow2 = self.head_dim > 0 and (self.head_dim & (self.head_dim - 1)) == 0
        self._supports_lattice = (
            self.head_dim % block_dim == 0 and is_pow2
        )

        if not self._supports_lattice:
            reasons = []
            if self.head_dim % block_dim != 0:
                reasons.append(f"head_dim % {block_dim} != 0")
            if not is_pow2:
                reasons.append(f"head_dim={self.head_dim} is not a power of 2 "
                               "(required by Sylvester--Hadamard rotation)")
            msg = (
                f"KakeyaLatticeCache(variant={self.variant!r}) constraint "
                f"violated: {', '.join(reasons)}.  "
            )
            if self.strict:
                msg += (
                    "KakeyaLatticeCache requires a power-of-2 head_dim that is "
                    "also divisible by the lattice block dim (4 for D4, 8 for "
                    "E8).  Most LLMs use head_dim in {64, 128, 256} which "
                    "satisfy both.  Pass strict=False to skip the codec on "
                    "this model and fall back to plain DynamicCache."
                )
                raise ValueError(msg)
            else:
                msg += "strict=False: codec disabled; falling back to raw KV."
                warnings.warn(msg, UserWarning, stacklevel=2)
                logger.warning(msg)

        # Per-layer codecs are built LAZILY on first update(), keyed by the
        # head_dim actually observed at each layer — so models with
        # heterogeneous per-layer head_dim (e.g. Gemma-4 sliding=256 / full=512)
        # work drop-in. ``head_dim`` is the declared default for back-compat.
        self._codecs: list[Any | None] = [None] * self.num_hidden_layers
        self._raw_layers: set[int] = set()

        # Fire counters for sanity / audit.
        self.codec_fired_per_layer: dict[int, int] = {}
        self.skip_fired_per_layer: dict[int, int] = {}

    # ----- codec management -----

    def _codec_cls(self):
        if self.variant == "d4":
            from kakeyalattice import V14KakeyaZamirLatticeGPU as CodecCls
        else:
            from kakeyalattice import V15KakeyaZamirE8GPU as CodecCls
        return CodecCls

    def _get_codec(self, layer_idx: int, observed_dim: int):
        """Lazily build/return the per-layer codec from the observed head_dim.
        Returns None for raw bf16 (boundary / incompatible-with-strict=False)."""
        if self._is_boundary_layer(layer_idx) or layer_idx in self._raw_layers:
            return None
        codec = self._codecs[layer_idx]
        if codec is not None:
            if codec.D_shape != observed_dim:
                raise ValueError(
                    f"layer {layer_idx} head_dim changed "
                    f"{codec.D_shape} -> {observed_dim} between updates"
                )
            return codec
        bd = self._block_dim
        is_pow2 = observed_dim > 0 and (observed_dim & (observed_dim - 1)) == 0
        if (observed_dim % bd != 0) or not is_pow2:
            msg = (
                f"KakeyaLatticeCache(variant={self.variant!r}): layer "
                f"{layer_idx} head_dim={observed_dim} is incompatible "
                f"(need a power of 2 divisible by {bd})."
            )
            if self.strict:
                raise ValueError(msg + " Pass strict=False to keep raw bf16.")
            warnings.warn(msg + " strict=False: layer kept as raw bf16.",
                          UserWarning, stacklevel=2)
            self._raw_layers.add(layer_idx)
            return None
        codec = self._codec_cls()(
            D=observed_dim, q_range=self.q_range, device=str(self.device),
        )
        self._codecs[layer_idx] = codec
        return codec

    def _is_boundary_layer(self, layer_idx: int) -> bool:
        if self.boundary <= 0:
            return False
        return (
            layer_idx < self.boundary
            or layer_idx >= (self.num_hidden_layers - self.boundary)
        )

    def _roundtrip(self, kv: torch.Tensor, codec: Any) -> torch.Tensor:
        """Apply ``codec.roundtrip`` preserving shape + dtype.

        kv shape is typically ``[batch, num_kv_heads, seq, head_dim]`` but
        any trailing-``head_dim`` shape is accepted because the codec
        only touches the last dim.
        """
        if codec is None:
            return kv
        orig_dtype = kv.dtype
        kv_fp32 = kv.to(torch.float32)
        kv_out = codec.roundtrip(kv_fp32)
        return kv_out.to(orig_dtype)

    # ----- DynamicCache interface override -----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roundtrip K and V through the per-layer codec, then delegate
        to ``DynamicCache.update`` to concat with existing cache state.
        """
        # Lazily resolve the per-layer codec from the observed head_dim so
        # heterogeneous-head_dim models work drop-in. None => raw bf16.
        codec = self._get_codec(layer_idx, key_states.shape[-1])
        if codec is None:
            self.skip_fired_per_layer[layer_idx] = (
                self.skip_fired_per_layer.get(layer_idx, 0) + 1
            )
            return super().update(
                key_states, value_states, layer_idx, *args, **kwargs
            )

        k_rt = self._roundtrip(key_states, codec)
        v_rt = self._roundtrip(value_states, codec)
        self.codec_fired_per_layer[layer_idx] = (
            self.codec_fired_per_layer.get(layer_idx, 0) + 1
        )
        return super().update(k_rt, v_rt, layer_idx, *args, **kwargs)

    # ----- diagnostics -----

    def __repr__(self) -> str:
        return (
            f"KakeyaLatticeCache(variant={self.variant!r}, q_range={self.q_range}, "
            f"num_hidden_layers={self.num_hidden_layers}, head_dim={self.head_dim}, "
            f"boundary={self.boundary}, supports_lattice={self._supports_lattice})"
        )


__all__ = ["KakeyaLatticeCache"]
