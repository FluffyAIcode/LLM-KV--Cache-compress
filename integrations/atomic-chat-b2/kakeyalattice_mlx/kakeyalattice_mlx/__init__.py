"""kakeyalattice_mlx — MLX port of KakeyaLattice v1.5 E8 codec.

Top-level imports are lazy: on a Linux CI machine without MLX installed,
importing this package must not fail. Sub-modules that require
`mlx.core` only import it when their functions are called.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "build_hadamard",
    "closest_d8",
    "closest_e8",
    "E8LatticeCodebookMLX",
    "KakeyaLatticeMLXCache",
    "bits_per_token_per_head",
]


def __getattr__(name):
    if name == "build_hadamard":
        from .hadamard import build_hadamard
        return build_hadamard
    if name in ("closest_d8", "closest_e8"):
        from . import closest_point
        return getattr(closest_point, name)
    if name == "E8LatticeCodebookMLX":
        from .codec import E8LatticeCodebookMLX
        return E8LatticeCodebookMLX
    if name == "KakeyaLatticeMLXCache":
        from .kv_cache import KakeyaLatticeMLXCache
        return KakeyaLatticeMLXCache
    if name == "bits_per_token_per_head":
        from .codec import bits_per_token_per_head
        return bits_per_token_per_head
    raise AttributeError(f"module 'kakeyalattice_mlx' has no attribute {name!r}")
