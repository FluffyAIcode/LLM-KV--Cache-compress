"""Sylvester–Hadamard matrix for the Kakeya rotation step.

`build_hadamard(D)` returns the matrix `H_D / √D`, where `H_D` is the
standard Sylvester–Hadamard matrix of order D (D must be a power of 2).

The rotation has three relevant properties for this codec:

1. It redistributes per-channel energy evenly across the D dimensions
   (equipartition). Skew-tailed per-channel KV distributions become
   approximately isotropic after this step, which is what lets the
   per-vector `q_max` scaling do most of the work.
2. It is self-inverse up to the `1/√D` normalisation: `H·Hᵀ = D·I`, so
   `(H/√D)·(H/√D) = I`. We therefore use the same matrix for both
   forward rotation and inverse rotation.
3. The entries are ±1, so matmul reduces to additions / subtractions.
   On MLX + Apple Silicon the matmul itself is still the dominant cost
   for small D, but this property matters when we later swap in a
   fused shader.
"""
from __future__ import annotations

import math


def _build_hadamard_numpy(D: int):
    """Pure-NumPy Sylvester–Hadamard, used as a ref-impl and in tests.

    Returns a ``numpy.ndarray`` of shape ``(D, D)`` with dtype ``float32``
    normalised by ``1/√D``.
    """
    import numpy as np

    if D <= 0 or (D & (D - 1)) != 0:
        raise ValueError(f"D must be a positive power of 2, got {D}")

    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < D:
        top = np.concatenate([H, H], axis=1)
        bot = np.concatenate([H, -H], axis=1)
        H = np.concatenate([top, bot], axis=0)
    return H / math.sqrt(D)


def build_hadamard(D: int, dtype=None):
    """Build the normalised Sylvester–Hadamard matrix as an ``mx.array``.

    Requires MLX. The dtype defaults to ``mx.float32``.
    """
    try:
        import mlx.core as mx  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "build_hadamard() requires the 'mlx' package. "
            "Install with: pip install kakeyalattice-mlx[mlx]"
        ) from e

    if dtype is None:
        dtype = mx.float32

    H_np = _build_hadamard_numpy(D)
    return mx.array(H_np, dtype=dtype)
