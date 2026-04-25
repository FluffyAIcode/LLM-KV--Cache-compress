"""HuggingFace ``transformers`` integration for kakeyalattice.

Exports:

    from kakeyalattice.hf import KakeyaLatticeCache

A drop-in replacement for ``transformers.DynamicCache`` that applies
a per-token nested-lattice roundtrip (encode + decode) to every K and V
write. The reconstruction error of the codec propagates through the
attention computation naturally; no custom attention kernel is needed.

Requires ``transformers >= 4.45`` (``DynamicCache`` API).
"""
from __future__ import annotations


def __getattr__(name):
    if name == "KakeyaLatticeCache":
        from .cache import KakeyaLatticeCache
        return KakeyaLatticeCache
    raise AttributeError(f"module 'kakeyalattice.hf' has no attribute {name!r}")


__all__ = ["KakeyaLatticeCache"]
