"""v1.5 KakeyaLattice (E8) — canonical class.

Successor to v1.4 KakeyaLattice that replaces the D4 root lattice with E8:
  * 8-D block (vs v1.4's 4-D block) → 16 blocks per head_dim=128 head
  * Voronoi region closer to a sphere
  * Shaping gain +0.66 dB over Z^8 (vs v1.4 D4's +0.37 dB over Z^4)
  * Net: +0.29 dB shaping gain over v1.4 at matched rate

Same wrapper architecture as v1.4:
  * unit-normalise + fp16 norm storage
  * Sylvester Hadamard rotation (unchanged)
  * per-vector qmax + fp16 storage
  * E8 closest-lattice-point (Conway-Sloane 1982 Alg 5) per block
  * inverse Hadamard + rescale

Storage at head_dim=128, n_kv_heads=H:
  * v1.4 D4 Q=38: 32 blocks × 25 bits + 32 overhead = 832 bits/head/token
  * v1.5 E8 Q=38: 16 blocks × 51 bits + 32 overhead = 848 bits/head/token (+1.9%)
  * v1.5 E8 Q=37: 16 blocks × 50 bits + 32 overhead = 832 bits/head/token (iso-bit)

Research hypothesis: at iso-bit, v1.5 should recover ~+0.29 dB of MSE,
which maps to ~+8-12 % abs-Delta-ppl advantage at aggressive bit budgets
(Q<=10 / b<=4) where shaping gain is not drowned by bf16 FA noise.

GPU cost: +25-30 % encode time per vector vs v1.4 (more ops per block,
but half as many blocks).  Still <1 % of end-to-end vLLM inference time.

Naming convention (enforce in all new writing):
  Written / spoken:      "v1.5 KakeyaLattice (E8)"
  Class name:            V15KakeyaZamirE8GPU
  Module name:           v1_5_kakeya_zamir_e8_gpu
  Internal impl class:   E8LatticeCodebook
  Do NOT use:            "E8-TQ-Style", "Zamir-E8", or similar
                         research-lineage aliases — the public surface
                         is V15KakeyaZamirE8GPU.
"""
from __future__ import annotations

from .lattice_codebooks import E8LatticeCodebook


class V15KakeyaZamirE8GPU(E8LatticeCodebook):
    """Canonical class for the v1.5 KakeyaLattice (E8) codec.

    Bit-identical to the underlying `E8LatticeCodebook` implementation;
    exists solely to provide the canonical naming surface analogous to
    `V14KakeyaZamirLatticeGPU`.

    Args:
        D: head dimension (128 for Qwen3-4B; must be divisible by 8 AND a
           power of 2 for the Hadamard rotation).
        q_range: per-coord lattice range.  Canonical points (for D=128):
                 - Q=152 → 1120 bits    (+32 vs v1.4 at same Q)
                 - Q=76  →  992 bits
                 - Q=38  →  848 bits
                 - Q=37  →  832 bits    (iso-bit to v1.4 Q=38)
                 - Q=19  →  720 bits
                 - Q=10  →  592 bits
                 - Q=5   →  464 bits
                 - Q=2   →  336 bits    (extreme compression, ~6.1×)
        device: CUDA device (strict GPU).

    Bits for other head_dims derive from the same formula:
        total_bits = (D / 8) · ⌈8·log₂(2Q+1)⌉ + 32
    """

    def __init__(self, D: int, q_range: int = 37, device: str = "cuda"):
        super().__init__(D=D, q_range=q_range, device=device)
        self.name = (
            f"v1.5-kakeya-zamir-E8-GPU-Q{q_range}"
            f"-bits{self.bits_per_token_per_head}"
        )


__all__ = ["V15KakeyaZamirE8GPU"]
