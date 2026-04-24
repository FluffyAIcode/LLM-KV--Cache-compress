"""KakeyaLattice reference Python codec package.

Exposes:

    from kakeyaturbo_py import V14KakeyaZamirLatticeGPU  # v1.4, D4 lattice
    from kakeyaturbo_py import V15KakeyaZamirE8GPU       # v1.5, E8 lattice

Both are Hadamard-rotated, per-vector-qmax nested-lattice codecs with
unit-normalisation + inverse rotation + rescale.  Only the lattice
differs:

  v1.4 — D4 (4-D blocks, Conway-Sloane 1982 Alg 4)
       * G(Λ) = 0.0766  →  +0.37 dB shaping gain over Z^4
       * Implementation: `D4LatticeCodebook` in `lattice_codebooks.py`

  v1.5 — E8 (8-D blocks, Conway-Sloane 1982 Alg 5)
       * G(Λ) = 0.0717  →  +0.66 dB shaping gain over Z^8
       * +0.29 dB over v1.4 at matched rate
       * Implementation: `E8LatticeCodebook` in `lattice_codebooks.py`

Both classes accept the same `(D, q_range, device)` constructor and
expose the same `.roundtrip(x)` interface.  See the canonical report
files under `reports/v1_4_release/` for measured performance.

Streaming / online semantics: neither codec has cross-token state;
both can be invoked per-decode-step on a single new token producing
the same reconstruction quality as a batched call.
"""
from __future__ import annotations


__all__ = [
    "V14KakeyaZamirLatticeGPU",
    "V15KakeyaZamirE8GPU",
]


def __getattr__(name):
    if name == "V14KakeyaZamirLatticeGPU":
        from . import v1_4_kakeya_zamir_lattice_gpu
        return v1_4_kakeya_zamir_lattice_gpu.V14KakeyaZamirLatticeGPU
    if name == "V15KakeyaZamirE8GPU":
        from . import v1_5_kakeya_zamir_e8_gpu
        return v1_5_kakeya_zamir_e8_gpu.V15KakeyaZamirE8GPU
    raise AttributeError(f"module 'kakeyaturbo_py' has no attribute {name!r}")
