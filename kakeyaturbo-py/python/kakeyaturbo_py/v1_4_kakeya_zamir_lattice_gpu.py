"""v1.4 KakeyaLattice — canonical codec implementation.

The head-of-line KV-cache compression codec: full Zamir-Feder
nested-lattice quantiser using the D4 root lattice as the shaping
lattice, wrapped in the complete TurboQuant-style engineering stack
(Hadamard rotation + unit normalisation + per-vector qmax + inverse
Hadamard + rescale).  Strict GPU — no numpy, no CPU detour in the
codec hot path.

Internal structure: this module delegates to
`bridge_b2_d4_tq_style.D4TQStyleCodebook`, which is the research
prototype name kept for provenance.  The wrapper below is the
production-named class all public code, documentation, and
benchmarks should reference.

Measured head-to-head performance vs TurboQuant k8v4 (Qwen3-4B,
strict-GPU harness, 4 × WikiText-103 passages) at Q=152 / 1088 bits:

  K rel-MSE:  3.38 × 10⁻⁵   (TQ: 3.71 × 10⁻⁵ → 0.911× better)
  |Δppl|:     0.37 %        (TQ: 0.66 %      → 0.552× better)
  top-1 pair: 99.61 %       (TQ: 98.83 %     → +0.78 pp)
  Encode:     6.7 ms/M vec  (TQ: 10 ms/M     → 1.5× faster)

Full multi-model / multi-threshold numbers in
`reports/v1_4_release/` (iso-bit and iso-PPL comparisons).

Naming convention (enforce strictly):
  Written / spoken:      "v1.4 KakeyaLattice"  or  "KakeyaLattice"
  With parameter:        "v1.4 KakeyaLattice Q=152"
  Class name:            V14KakeyaZamirLatticeGPU
  Module name:           v1_4_kakeya_zamir_lattice_gpu
"""
from __future__ import annotations

import torch

from .bridge_b2_d4_tq_style import D4TQStyleCodebook


class V14KakeyaZamirLatticeGPU(D4TQStyleCodebook):
    """Canonical class for the v1.4 kakeya zamir lattice GPU codec.

    Bit-identical to the underlying `D4TQStyleCodebook` research
    class; exists solely to provide the canonical naming surface.
    New code should import this class, not `D4TQStyleCodebook`.

    Args:
        D: head dimension (128 for Qwen3-4B).
        q_range: per-coord lattice range.  Canonical points:
                 - Q=152 → 1088 bits (head-to-head with TQ k8v4)
                 - Q=76  →  960 bits
                 - Q=38  →  832 bits
                 - Q=19  →  704 bits
                 - Q=10  →  576 bits
                 - Q=5   →  448 bits
                 - Q=2   →  320 bits (extreme compression, 6.4×)
        device: CUDA device (strict GPU).

    See module docstring for full measured performance.
    """

    def __init__(self, D: int, q_range: int = 152, device: str = "cuda"):
        super().__init__(D=D, q_range=q_range, device=device)
        # Override the research lineage name with the canonical one so
        # downstream code that uses `codebook.name` sees v1.4 naming.
        self.name = (
            f"v1.4-kakeya-zamir-lattice-GPU-Q{q_range}"
            f"-bits{self.bits_per_token_per_head}"
        )


__all__ = ["V14KakeyaZamirLatticeGPU"]
