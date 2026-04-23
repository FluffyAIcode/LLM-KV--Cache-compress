"""v1.4 kakeya zamir lattice GPU — canonical implementation.

This is the OFFICIAL codec for all Qwen3-4B deployment work going
forward.  It strictly supersedes the v1.3 PCA+K-means+WHT+Lloyd-Max
codec family for codec-kernel purposes (the v1.3 deployment layer
— snapshot-mode, boundary skip — is orthogonal and remains valid).

Architecture: full Zamir-Feder nested-lattice quantiser using the
D4 root lattice as the shaping lattice, wrapped in the complete
TurboQuant engineering stack.  Strict GPU — no numpy, no CPU
detour in the codec hot path.

Provenance: this module was prototyped as "Bridge B2" in
`bridge_b2_d4_tq_style.py` (research-lineage name; do not re-export
from that module into new code).  The research provenance is:

  bridge_b_nested_lattice.py    : raw Zamir-Feder D4 (naive, lost
                                   to TQ by 1414× due to missing
                                   engineering; see FINDINGS_GPU.md
                                   "Three bridges from Dvir to
                                   Euclidean quantisation").
  bridge_b2_d4_tq_style.py      : D4 + Hadamard + per-vector qmax +
                                   matched bits + joint quantisation
                                   — the research prototype that
                                   first measured a win over TQ.
  v1_4_kakeya_zamir_lattice_gpu : canonical production-named class
                                   wrapping bridge_b2_d4_tq_style.
                                   This is the name all new code
                                   and documentation should use.

Measured performance (Qwen3-4B, strict-GPU harness, 4 × WikiText-103
passages):

  Q=152 (head-to-head with TQ k8v4 at 1056 bits):
    K rel-MSE:  3.38 × 10⁻⁵  (TQ: 3.71 × 10⁻⁵ → 0.911× better)
    |Δppl|:     0.37 %       (TQ: 0.66 %      → 0.552× better)
    top-1 pair: 99.61 %      (TQ: 98.83 %     → +0.78 pp)
    Encode:     6.7 ms/M vec (TQ: 10 ms/M     → 1.5× faster)

  Full Pareto (7 bit levels): B2 strictly dominates TQ on K-MSE at
  every point, ratio ranges 0.350× (at 6.4× compression) to 0.911×
  (at 1.88× compression).  See FINDINGS_GPU.md and
  SESSION_KAKEYA_RESEARCH.md section "6ter Compression-rate Pareto
  sweep" for the full data.

Naming convention (enforce strictly in all new writing):
  Spoken / written:      "v1.4 kakeya zamir lattice GPU"
  Short with parameter:  "v1.4 kakeya zamir lattice GPU Q=152"
  Class name:            V14KakeyaZamirLatticeGPU
  Module name:           v1_4_kakeya_zamir_lattice_gpu
  Do NOT use:            D4, Bridge B2, Kakeya-D4, nested lattice
                         codec (those are research lineage aliases).
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
