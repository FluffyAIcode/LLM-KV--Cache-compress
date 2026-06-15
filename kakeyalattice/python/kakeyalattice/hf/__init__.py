"""HuggingFace ``transformers`` integration for kakeyalattice.

Two cache classes are exported, with different memory semantics:

    from kakeyalattice.hf import KakeyaLatticeCache
    # reconstruction-only: stores bf16 reconstructed tensors.
    # zero real HBM savings; useful as a codec-quality probe.

    from kakeyalattice.hf import KakeyaLatticeQuantizedCache
    # stores int8 lattice indices + fp16 norm + fp16 qmax.
    # real ~1.94x HBM savings at head_dim=128, q_range<=127.

Plus the encode/decode primitives used by the quantized cache:

    from kakeyalattice.hf import encode_to_indices, decode_from_indices

Requires ``transformers >= 4.45`` (``DynamicCache`` API).
"""
from __future__ import annotations


def __getattr__(name):
    if name == "KakeyaLatticeCache":
        from .cache import KakeyaLatticeCache
        return KakeyaLatticeCache
    if name == "KakeyaLatticeQuantizedCache":
        from .quantized_cache import KakeyaLatticeQuantizedCache
        return KakeyaLatticeQuantizedCache
    if name == "encode_to_indices":
        from .quantized_cache import encode_to_indices
        return encode_to_indices
    if name == "decode_from_indices":
        from .quantized_cache import decode_from_indices
        return decode_from_indices
    if name == "KakeyaLatticePackedCache":
        from .packed_cache import KakeyaLatticePackedCache
        return KakeyaLatticePackedCache
    if name == "TurboQuantPackedCache":
        from .packed_cache import TurboQuantPackedCache
        return TurboQuantPackedCache
    if name == "TurboQuantCodec":
        from .turboquant import TurboQuantCodec
        return TurboQuantCodec
    raise AttributeError(f"module 'kakeyalattice.hf' has no attribute {name!r}")


__all__ = [
    "KakeyaLatticeCache",
    "KakeyaLatticeQuantizedCache",
    "KakeyaLatticePackedCache",
    "TurboQuantPackedCache",
    "TurboQuantCodec",
    "encode_to_indices",
    "decode_from_indices",
]
