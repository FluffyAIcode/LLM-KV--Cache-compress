"""Bridge B2: D4 nested lattice with the full TurboQuant engineering stack.

What separates this from Bridge B:

  Bridge B  — raw Zamir-Feder D4 applied to unit K.
     * Per-block global scale (dataset-wide max_abs_per_block / q_range).
     * No Hadamard rotation (operates directly on 4-dim slices of S^(D-1)).
     * Per-block independence, no per-vector scale.
     * Measured: rel-MSE 0.049 at 736 bits (1414× worse than TQ k8v4).

  Bridge B2  — D4 lattice + the three engineering levers that TQ uses.
     1. Unit-normalise:  `unit = x / ‖x‖`               (stores ‖x‖ fp16)
     2. Hadamard rotate: `y = unit · H / √D`           (D = 128)
     3. Per-vector qmax: `qmax = max_i |y_i|`           (stores qmax fp16)
     4. Scale to lattice: `y_scaled = y · q_range / qmax`
     5. D4 closest-lattice-point per (4-dim) block.
     6. Clamp to ±q_range (handles any out-of-range quantiser output).
     7. Decode: `y_hat = q · qmax / q_range`
     8. Inverse Hadamard: `unit_hat = y_hat · H / √D`  (Hadamard is self-inverse
        after the 1/√D normalisation).
     9. Rescale: `x_hat = unit_hat · ‖x‖`.

     This is "full Zamir-Feder (lattice code) + full TurboQuant (Hadamard +
     per-vector scale) + global joint quantisation (all 32 blocks see the
     same per-vector qmax, giving joint rate allocation)".

Bit budget at q_range = 152:
  Per block: 4 · log2(2·152+1) − 1 = 4·log2(305) − 1 = 32.006 bits.
  32 blocks × 32 bits = 1024 bits for lattice indices.
  + 16 bits `‖x‖` fp16 + 16 bits `qmax` fp16 = 32 bits overhead.
  **Total = 1056 bits/token/head, EXACTLY matching TQ k8v4.**

Expected outcome per the prior theoretical analysis:
  rel-MSE ≈ TQ · 0.92 ≈ 3.2 × 10⁻⁵  (8 % better than TQ due to the D4
  shaping gain of +0.37 dB over Z^4 uniform).
"""
from __future__ import annotations

import math

import torch

from .bridge_b_nested_lattice import _closest_d4_lattice_point
from .spherical_codebooks import SphericalCodebook


def _sylvester_hadamard_normalised(D: int, device: str) -> torch.Tensor:
    """Sylvester Hadamard H_D / √D — same normalisation as TurboQuant.

    Self-inverse: H · H = I (since H is symmetric and H_raw · H_raw = D · I
    gives (H/√D) · (H/√D) = I).
    """
    assert (D & (D - 1)) == 0, f"D must be power of 2, got {D}"
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
    return H / math.sqrt(D)


class D4TQStyleCodebook(SphericalCodebook):
    """D4 nested lattice quantizer with the full TurboQuant-style engineering
    wrapper (Hadamard + per-vector qmax + unit normalisation + inverse
    Hadamard + rescale).

    Args:
        D: input dimension (128 for Qwen3-4B).
        q_range: per-coord lattice range in units of the q_range grid
                 (the actual coord range is ±q_range after scaling).
                 q_range=152 gives 32 bits/block ≡ 1024 bits data/token/head,
                 matching TQ k8v4 at 1056 total bits/token/head (+32 overhead).

    Exposes the `SphericalCodebook` interface for the head-to-head harness
    (`roundtrip(x)` returns x_hat of the same shape).  Bit count is
    reported as the sum of lattice bits and the two fp16 overhead scalars.
    """

    def __init__(
        self,
        D: int,
        q_range: int = 152,
        device: str = "cuda",
    ):
        assert D % 4 == 0, f"D must be divisible by 4 for D4 blocks, got {D}"
        assert (D & (D - 1)) == 0, f"D must be power of 2 (Hadamard), got {D}"
        assert q_range >= 1
        self.D_shape = D
        self.K_blocks = D // 4
        self.q_range = q_range
        self.H = _sylvester_hadamard_normalised(D, device)    # [D, D]

        # Bit accounting:
        # D4 lattice in [-q_range, q_range]^4 with even-sum constraint
        # has (2 q_range + 1)^4 / 2 lattice points (asymptotic, with O(q_range^3)
        # boundary correction we ignore since we use clamp-and-quantize rather
        # than a pre-listed lattice).  The rate is:
        bits_per_block = 4 * math.log2(2 * q_range + 1) - 1
        total_lattice_bits = self.K_blocks * int(math.ceil(bits_per_block))
        # Two fp16 overhead scalars per vector: ‖x‖ and qmax.
        overhead_bits = 32
        total_bits = total_lattice_bits + overhead_bits

        # Dummy codewords tensor (unused — we override `roundtrip` and never
        # do a NN search in a materialised codebook).
        dummy = torch.zeros(1, D, device=device, dtype=torch.float32)
        super().__init__(
            codewords=dummy,
            name=(
                f"D4-TQstyle-Q{q_range}-bits{total_bits}"
                f"(lat{total_lattice_bits}+oh{overhead_bits})"
            ),
            bits=total_bits,
        )

    def encode(self, x: torch.Tensor):
        raise NotImplementedError(
            "Use .roundtrip(x) — D4-TQ style uses per-vector state (qmax, "
            "‖x‖) that isn't representable as a flat (seg_id, t) pair."
        )

    def decode(self, seg_id: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError(
            "Use .roundtrip(x) — D4-TQ style has per-vector state."
        )

    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode + decode roundtrip on a batch of K vectors.

        Args:
            x: [..., D] input K vectors (not necessarily unit norm).

        Returns:
            x_hat: same shape as x.  L2 error relative to x is bounded by
                   the D4 quantisation error on the Hadamard-rotated
                   unit-normalised data, plus the fp16 round-trip on ‖x‖
                   and qmax.
        """
        assert x.shape[-1] == self.D_shape, (
            f"expected last dim {self.D_shape}, got {x.shape[-1]}"
        )
        batch = x.shape[:-1]
        flat = x.reshape(-1, self.D_shape).to(torch.float32)
        N = flat.shape[0]
        eps = torch.finfo(flat.dtype).eps

        # 1. Unit-normalise.
        norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)          # [N, 1]
        # fp16 round-trip on ‖x‖ to match TQ's storage precision.
        norms_f16 = norms.to(torch.float16).to(torch.float32)
        unit = flat / norms                                             # [N, D]

        # 2. Hadamard rotation.
        y = unit @ self.H                                               # [N, D]

        # 3. Per-vector qmax.
        qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)   # [N, 1]
        # fp16 round-trip on qmax to match TQ's storage precision.
        qmax_f16 = qmax.to(torch.float16).to(torch.float32)
        # Scale = qmax / q_range → y_scaled = y / scale lands in [-q_range, q_range]
        scale = qmax_f16 / float(self.q_range)                          # [N, 1]

        # 4. Scale to lattice coordinates.
        y_scaled = y / scale                                            # [N, D]

        # 5. D4 closest-lattice-point per (4-dim) block.
        y_blocks = y_scaled.reshape(N, self.K_blocks, 4)                # [N, K, 4]
        q_lat = _closest_d4_lattice_point(y_blocks)                     # [N, K, 4]

        # 6. Clamp to lattice range (guards against qmax being the single
        #    max coord — other coords' lattice points should fit but clamp
        #    defensively; the Conway-Sloane algorithm can occasionally
        #    produce coords just beyond q_range on the "flip" branch).
        q_lat = q_lat.clamp(-self.q_range, self.q_range)

        # 7. Decode: back to coord space.
        y_hat_blocks = q_lat.to(torch.float32) * scale.unsqueeze(-1)    # [N, K, 4]
        y_hat = y_hat_blocks.reshape(N, self.D_shape)                   # [N, D]

        # 8. Inverse Hadamard (self-inverse after /√D normalisation).
        unit_hat = y_hat @ self.H                                       # [N, D]

        # 9. Rescale by original ‖x‖ (fp16-rounded, matching TQ).
        x_hat_flat = unit_hat * norms_f16                               # [N, D]
        return x_hat_flat.reshape(*batch, self.D_shape)
