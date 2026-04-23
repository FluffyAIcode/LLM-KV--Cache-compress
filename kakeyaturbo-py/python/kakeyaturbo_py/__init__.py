"""v1.4 KakeyaLattice reference Python codec package.

Exposes a single top-level symbol:

    from kakeyaturbo_py import V14KakeyaZamirLatticeGPU

`V14KakeyaZamirLatticeGPU(D: int, q_range: int, device: str = "cuda")`
is the head-of-line Zamir-Feder D4 nested-lattice codec with the full
TurboQuant-style engineering stack (Hadamard rotation + unit
normalisation + per-vector qmax + inverse Hadamard + rescale).

Implementation provenance (see `v1_4_kakeya_zamir_lattice_gpu.py` for
the full research trail that led to this codec).  The core algorithm
lives in `bridge_b2_d4_tq_style.D4TQStyleCodebook`; `V14KakeyaZamirLatticeGPU`
is a thin canonical-naming wrapper around it.

Streaming / online semantics: the codec has no cross-token state and
can be invoked per-decode-step on a single new token, producing the
same reconstruction quality as a batched call (see
`reports/v1_4_release/streaming/V14_STREAMING_REPORT.md`).
"""
from __future__ import annotations


__all__ = ["V14KakeyaZamirLatticeGPU"]


def __getattr__(name):
    if name == "V14KakeyaZamirLatticeGPU":
        from . import v1_4_kakeya_zamir_lattice_gpu
        return v1_4_kakeya_zamir_lattice_gpu.V14KakeyaZamirLatticeGPU
    raise AttributeError(f"module 'kakeyaturbo_py' has no attribute {name!r}")
