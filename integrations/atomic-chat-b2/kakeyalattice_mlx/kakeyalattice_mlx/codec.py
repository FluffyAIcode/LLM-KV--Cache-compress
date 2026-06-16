"""MLX port of ``kakeyalattice.E8LatticeCodebook`` (v1.5).

Bit-layout and numerical semantics are deliberately identical to the
PyTorch reference so that the same pre-trained model weights + same
``Q=38`` / ``Q=10`` configurations that were validated in the
``reports/v1_5_release/V15_FULL_4MODEL_REPORT.md`` benchmarks
continue to apply without re-measurement.

Pipeline (same as PyTorch ref-impl):

    x  ∈ ℝ^D
      → unit-normalise; store ‖x‖ rounded to fp16
      → Sylvester–Hadamard rotation
      → per-vector qmax; store qmax rounded to fp16
      → scale to [-Q, Q]^D; split into D/8 blocks of 8 dims
      → closest-E8-point per block
      → clamp to ±Q
      → rescale; inverse Hadamard; restore ‖x‖

The fp16 round-trips on ‖x‖ and ``qmax`` are critical — they are the
"side info" sent to the decoder, and omitting them gives a codec that
appears to work but is off-by-a-quantum from the reference at
bit-level.
"""
from __future__ import annotations

import math

import numpy as np

from .hadamard import build_hadamard
from .closest_point import closest_e8


def bits_per_token_per_head(D: int, q_range: int) -> int:
    """E8 bit count: ``(D/8) · ⌈8·log₂(2Q+1)⌉ + 32``.

    The +32 overhead is the two fp16 side-info scalars (‖x‖ + qmax).
    """
    if D <= 0 or D % 8 != 0:
        raise ValueError(f"D must be a positive multiple of 8, got {D}")
    if q_range < 1:
        raise ValueError(f"q_range must be >= 1, got {q_range}")
    bits_per_block = int(math.ceil(8.0 * math.log2(2 * q_range + 1)))
    return (D // 8) * bits_per_block + 32


def _require_mlx():
    try:
        import mlx.core as mx  # type: ignore
        return mx
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "E8LatticeCodebookMLX requires the 'mlx' package. "
            "Install with: pip install kakeyalattice-mlx[mlx]"
        ) from e


class E8LatticeCodebookMLX:
    """E8 nested-lattice codec, MLX edition.

    Args:
        D: head dimension. Must be divisible by 8 (block dim) AND a
           power of 2 (Sylvester–Hadamard). Valid head_dim's for
           mainstream LLMs: 64, 128, 256 → all OK.
        q_range: per-coord lattice range. Canonical points for D=128:
           * Q=38  (near-lossless, ~2.5× CR)
           * Q=10  (balanced, ~3.37× CR, **recommended**)
           * Q=4   (aggressive, ~4.57× CR)
           * Q=152 (ultra-near-lossless, ~1.88× CR)
        dtype: internal compute dtype. Default ``mx.float32`` for bit
           parity with the PyTorch reference. Users who want to trade
           parity for speed can pass ``mx.float16`` or ``mx.bfloat16``;
           the tests guard this by only asserting parity in float32.

    The class preserves the public shape of the PyTorch reference —
    ``codec.roundtrip(x)`` is the only method users need.
    """

    block_dim = 8
    short_name = "E8LatticeMLX"
    shaping_gain_db = 0.66

    def __init__(self, D: int, q_range: int = 38, dtype=None):
        mx = _require_mlx()

        if D <= 0 or (D & (D - 1)) != 0:
            raise ValueError(f"D must be a positive power of 2, got {D}")
        if D % self.block_dim != 0:
            raise ValueError(
                f"D must be divisible by block_dim={self.block_dim}, got D={D}"
            )
        if q_range < 1:
            raise ValueError(f"q_range must be >= 1, got {q_range}")

        self.D = D
        self.K_blocks = D // self.block_dim
        self.q_range = q_range
        self.dtype = dtype or mx.float32

        self.H = build_hadamard(D, dtype=self.dtype)
        self.bits_per_token_per_head = bits_per_token_per_head(D, q_range)

        self.name = (
            f"v1.5-kakeya-zamir-E8-MLX-Q{q_range}"
            f"-bits{self.bits_per_token_per_head}"
        )

    # ------------------------------------------------------------------

    def roundtrip(self, x):
        """Encode + decode round-trip.

        Args:
            x: ``mx.array`` of shape ``[..., D]`` or any shape whose
                trailing dim equals ``D``.
        Returns:
            ``mx.array`` of the same shape and dtype as ``x``.
        """
        mx = _require_mlx()

        if x.shape[-1] != self.D:
            raise ValueError(
                f"roundtrip expected last dim {self.D}, got {x.shape[-1]}"
            )

        orig_dtype = x.dtype
        batch_shape = x.shape[:-1]
        flat = x.reshape(-1, self.D).astype(self.dtype)
        N = flat.shape[0]

        # Match the PyTorch reference's ``torch.finfo(dtype).eps`` guard.
        # For float32 this is ~1.19e-7 which survives the fp16 round-trip
        # on norms/qmax (it lands in fp16 subnormals rather than 0), so
        # zero-input correctly returns zero instead of NaN.
        eps = mx.array(float(np.finfo(np.float32).eps), dtype=self.dtype)

        # 1. Unit-normalise and round ‖x‖ through fp16.
        sqnorm = (flat * flat).sum(axis=-1, keepdims=True)
        norms = mx.maximum(mx.sqrt(sqnorm), eps)                # [N, 1]
        norms_f16 = norms.astype(mx.float16).astype(self.dtype)
        unit = flat / norms

        # 2. Hadamard rotation.
        y = unit @ self.H                                       # [N, D]

        # 3. Per-vector qmax, rounded through fp16.
        qmax = mx.maximum(mx.max(mx.abs(y), axis=-1, keepdims=True), eps)
        qmax_f16 = qmax.astype(mx.float16).astype(self.dtype)
        scale = qmax_f16 / float(self.q_range)                  # [N, 1]

        # 4. Scale to lattice coordinates.
        y_scaled = y / scale

        # 5. Closest-lattice-point per 8-D block.
        y_blocks = y_scaled.reshape(N, self.K_blocks, self.block_dim)
        q_lat = closest_e8(y_blocks)

        # 6. Defensive clamp. Parity flip can push coords marginally
        #    outside [-Q, Q]; the reference clamps here too.
        q_lat = mx.clip(q_lat, -self.q_range, self.q_range)

        # 7. Rescale to coord space.
        y_hat = (q_lat.astype(self.dtype) * scale[..., None]).reshape(N, self.D)

        # 8. Inverse Hadamard (H/√D is self-inverse).
        unit_hat = y_hat @ self.H

        # 9. Restore original magnitude via fp16-rounded ‖x‖.
        x_hat = unit_hat * norms_f16

        return x_hat.reshape(*batch_shape, self.D).astype(orig_dtype)


__all__ = ["E8LatticeCodebookMLX", "bits_per_token_per_head"]
