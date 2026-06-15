r"""End-to-end **bit-packed** KV caches (contiguous, SDPA-ready).

These caches realise the codec's *bit-rate* ceiling as real stored bytes:

  * :class:`KakeyaLatticePackedCache` — D4 or E8 lattice, block-coded + packed.
    Real HBM ratio at head_dim=128: D4 Q=38 ≈ 2.46x, E8 Q=38 ≈ 2.42x
    (vs the int8 ``KakeyaLatticeQuantizedCache``'s 1.94x).
  * :class:`TurboQuantPackedCache` — the scalar-quantise (TurboQuant) baseline
    packed at exactly ``b`` bits/coordinate. Real HBM ratio at b=4 ≈ 3.76x
    (but at much lower quality — see the iso-ppl comparison harness).

Design (both): the per-layer **working** state is a single contiguous integer
code buffer (grown by one ``cat`` per ``update``, like ``DynamicCache``), decoded
to a **contiguous** bf16 tensor for attention. ``kv_storage_bytes()`` returns the
**bit-packed** footprint (lossless re-encoding of the stored codes), i.e. the real
HBM a fused-kernel deployment would hold. ``verify_packing_lossless()`` proves the
pack→unpack cycle reproduces the stored codes exactly (so reconstruction — and
perplexity — are byte-for-byte identical to the unpacked cache).
"""
from __future__ import annotations

import warnings
from typing import Any

import torch

from .quantized_cache import (
    KakeyaLatticeQuantizedCache,
    decode_from_indices,
    encode_to_indices,
)
from . import bitpack as _bp


def _require_dynamic_cache():
    from transformers import DynamicCache
    return DynamicCache


_DynamicCache = _require_dynamic_cache()


class KakeyaLatticePackedCache(KakeyaLatticeQuantizedCache):
    """``KakeyaLatticeQuantizedCache`` whose ``kv_storage_bytes()`` reports the
    real **bit-packed** footprint (D4/E8 block code + exception side-channel +
    fp16 norm/qmax), instead of the int8 footprint.

    The working buffers, decode path and generation behaviour are inherited
    unchanged (single contiguous per-layer buffer; contiguous bf16 on read), so
    reconstruction is identical — only the *reported/realisable* storage size
    differs.
    """

    def kv_storage_bytes(self) -> int:
        total = 0
        for codes_list, n_list, m_list in (
            (self._k_codes, self._k_norms, self._k_qmax),
            (self._v_codes, self._v_norms, self._v_qmax),
        ):
            for li in range(self.num_hidden_layers):
                codes = codes_list[li]
                if codes is None:
                    continue
                packed = _bp.pack_lattice_codes(codes, self.variant, self.q_range)
                total += _bp.packed_storage_bytes(packed)
                total += n_list[li].element_size() * n_list[li].numel()
                total += m_list[li].element_size() * m_list[li].numel()
        # fallback bf16 layers (boundary / unsupported)
        if hasattr(self, "layers"):
            for layer in self.layers:
                for attr in ("keys", "values"):
                    t = getattr(layer, attr, None)
                    if t is not None:
                        total += t.element_size() * t.numel()
        return total

    def packed_pack_unpack_ok(self) -> bool:
        """Pack then unpack every stored code buffer; True iff all reproduce the
        stored codes exactly (lossless end-to-end packing proof)."""
        for codes_list in (self._k_codes, self._v_codes):
            for codes in codes_list:
                if codes is None:
                    continue
                packed = _bp.pack_lattice_codes(codes, self.variant, self.q_range)
                back = _bp.unpack_lattice_codes(packed)
                if not torch.equal(back.to(torch.int64), codes.to(torch.int64)):
                    return False
        return True


class TurboQuantPackedCache(_DynamicCache):
    """TurboQuant (scalar-quantise) baseline as a packed ``DynamicCache``.

    Per-layer contiguous symbol buffers (b-bit per coordinate) + fp16 norm/qmax;
    decodes to contiguous bf16 on read. ``kv_storage_bytes()`` returns the real
    b-bit-packed footprint.
    """

    def __init__(
        self,
        bits_b: int = 4,
        num_hidden_layers: int | None = None,
        head_dim: int | None = None,
        device: str | torch.device = "cuda",
        boundary: int = 0,
        out_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if num_hidden_layers is None or head_dim is None:
            raise ValueError("TurboQuantPackedCache requires num_hidden_layers and head_dim")
        from .turboquant import TurboQuantCodec

        self.bits_b = int(bits_b)
        self.num_hidden_layers = int(num_hidden_layers)
        self.head_dim = int(head_dim)
        self.device = torch.device(device)
        self.boundary = int(boundary)
        self.out_dtype = out_dtype
        is_pow2 = head_dim > 0 and (head_dim & (head_dim - 1)) == 0
        if not is_pow2:
            raise ValueError(f"head_dim must be a power of 2, got {head_dim}")

        self._TQCodec = TurboQuantCodec
        # Lazy per-layer codecs keyed by observed head_dim (drop-in for
        # heterogeneous-head_dim models like Gemma-4).
        self._codecs: list[Any | None] = [None] * self.num_hidden_layers
        self._raw_layers: set[int] = set()
        self._k_sym: list[torch.Tensor | None] = [None] * self.num_hidden_layers
        self._k_norms: list[torch.Tensor | None] = [None] * self.num_hidden_layers
        self._k_qmax: list[torch.Tensor | None] = [None] * self.num_hidden_layers
        self._v_sym: list[torch.Tensor | None] = [None] * self.num_hidden_layers
        self._v_norms: list[torch.Tensor | None] = [None] * self.num_hidden_layers
        self._v_qmax: list[torch.Tensor | None] = [None] * self.num_hidden_layers
        self.codec_fired_per_layer: dict[int, int] = {}
        self.skip_fired_per_layer: dict[int, int] = {}

    def _is_boundary_layer(self, layer_idx: int) -> bool:
        if self.boundary <= 0:
            return False
        return layer_idx < self.boundary or layer_idx >= (self.num_hidden_layers - self.boundary)

    @staticmethod
    def _append(buffers, layer_idx, new):
        cur = buffers[layer_idx]
        grown = new if cur is None else torch.cat([cur, new], dim=-2)
        buffers[layer_idx] = grown
        return grown

    def _get_codec(self, layer_idx: int, observed_dim: int):
        if self._is_boundary_layer(layer_idx) or layer_idx in self._raw_layers:
            return None
        codec = self._codecs[layer_idx]
        if codec is not None:
            if codec.D_shape != observed_dim:
                raise ValueError(
                    f"layer {layer_idx} head_dim changed "
                    f"{codec.D_shape} -> {observed_dim} between updates")
            return codec
        is_pow2 = observed_dim > 0 and (observed_dim & (observed_dim - 1)) == 0
        if not is_pow2:
            self._raw_layers.add(layer_idx)
            return None
        codec = self._TQCodec(D=observed_dim, bits_b=self.bits_b, device=str(self.device))
        self._codecs[layer_idx] = codec
        return codec

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        codec = self._get_codec(layer_idx, key_states.shape[-1])
        if codec is None:
            self.skip_fired_per_layer[layer_idx] = self.skip_fired_per_layer.get(layer_idx, 0) + 1
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)
        self.codec_fired_per_layer[layer_idx] = self.codec_fired_per_layer.get(layer_idx, 0) + 1
        ks, kn, km = codec.encode_to_symbols(key_states)
        vs, vn, vm = codec.encode_to_symbols(value_states)
        ks = self._append(self._k_sym, layer_idx, ks)
        kn = self._append(self._k_norms, layer_idx, kn)
        km = self._append(self._k_qmax, layer_idx, km)
        vs = self._append(self._v_sym, layer_idx, vs)
        vn = self._append(self._v_norms, layer_idx, vn)
        vm = self._append(self._v_qmax, layer_idx, vm)
        k_bf = codec.decode_from_symbols(ks, kn, km, self.out_dtype)
        v_bf = codec.decode_from_symbols(vs, vn, vm, self.out_dtype)
        return k_bf.contiguous(), v_bf.contiguous()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self._k_sym) and self._k_sym[layer_idx] is not None:
            return self._k_sym[layer_idx].shape[-2]
        try:
            return super().get_seq_length(layer_idx)
        except Exception:
            return 0

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        """True (kv_length, kv_offset) from our buffers — see
        KakeyaLatticeQuantizedCache.get_mask_sizes for rationale."""
        return self.get_seq_length(layer_idx) + query_length, 0

    def kv_storage_bytes(self) -> int:
        total = 0
        for sym_list, n_list, m_list in (
            (self._k_sym, self._k_norms, self._k_qmax),
            (self._v_sym, self._v_norms, self._v_qmax),
        ):
            for li in range(self.num_hidden_layers):
                sym = sym_list[li]
                if sym is None:
                    continue
                buf = _bp.pack_codes(sym.reshape(-1), self.bits_b)
                total += int(buf.numel())
                total += n_list[li].element_size() * n_list[li].numel()
                total += m_list[li].element_size() * m_list[li].numel()
        if hasattr(self, "layers"):
            for layer in self.layers:
                for attr in ("keys", "values"):
                    t = getattr(layer, attr, None)
                    if t is not None:
                        total += t.element_size() * t.numel()
        return total


__all__ = ["KakeyaLatticePackedCache", "TurboQuantPackedCache"]
