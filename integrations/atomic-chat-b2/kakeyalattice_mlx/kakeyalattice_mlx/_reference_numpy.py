"""NumPy shadow reference for the E8 codec.

This module exists for two reasons:

1. **Linux CI without MLX** still needs to validate the closest-point
   and bit-accounting semantics. The NumPy shadow gives us that.
2. **Three-way parity**: PyTorch reference ↔ NumPy shadow ↔ MLX port.
   If MLX changes a rounding tiebreak we catch it by comparing
   against NumPy (same tie-breaking rules as PyTorch) rather than
   against PyTorch directly (which adds a torch dependency to
   tests).

Line-for-line port of ``kakeyalattice.lattice_codebooks`` (subset
relevant to v1.5 E8).
"""
from __future__ import annotations

import math

import numpy as np

from .hadamard import _build_hadamard_numpy


def _closest_dn_np(y: np.ndarray, n: int) -> np.ndarray:
    if y.shape[-1] != n:
        raise ValueError(f"expected last dim {n}, got {y.shape[-1]}")
    f = np.round(y)
    s = f.sum(axis=-1)
    even_mask = (s.astype(np.int64) % 2) == 0
    if even_mask.all():
        return f
    diff = y - f
    idx = np.argmax(np.abs(diff), axis=-1, keepdims=True)
    gathered = np.take_along_axis(diff, idx, axis=-1)
    sign = np.where(gathered >= 0, 1.0, -1.0).astype(y.dtype)
    adj = np.zeros_like(f)
    np.put_along_axis(adj, idx, sign, axis=-1)
    f_odd = f + adj
    return np.where(even_mask[..., None], f, f_odd)


def closest_d8_np(y: np.ndarray) -> np.ndarray:
    return _closest_dn_np(y, 8)


def closest_e8_np(y: np.ndarray) -> np.ndarray:
    if y.shape[-1] != 8:
        raise ValueError(f"closest_e8 expects last dim 8, got {y.shape[-1]}")
    a = closest_d8_np(y)
    half = np.full(y.shape[:-1] + (1,), 0.5, dtype=y.dtype)
    b_int = closest_d8_np(y - half)
    b = b_int + half
    da = ((y - a) ** 2).sum(axis=-1, keepdims=True)
    db = ((y - b) ** 2).sum(axis=-1, keepdims=True)
    return np.where(da <= db, a, b)


def roundtrip_np(x: np.ndarray, D: int, q_range: int) -> np.ndarray:
    """NumPy shadow of ``E8LatticeCodebook.roundtrip``."""
    if x.shape[-1] != D:
        raise ValueError(f"expected last dim {D}, got {x.shape[-1]}")
    if D % 8 != 0 or (D & (D - 1)) != 0:
        raise ValueError(f"D must be a power-of-2 multiple of 8, got {D}")

    H = _build_hadamard_numpy(D)
    batch_shape = x.shape[:-1]
    flat = x.reshape(-1, D).astype(np.float32)
    N = flat.shape[0]
    # Match the PyTorch reference which uses ``torch.finfo(float32).eps``
    # (~1.19e-7). This sits comfortably inside fp16's subnormal range,
    # so the subsequent fp16 round-trip doesn't collapse eps to zero —
    # zero-input therefore round-trips to zero (no NaN path).
    eps = np.finfo(np.float32).eps

    # 1. Unit-normalise + fp16 round-trip norms.
    norms = np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), eps)
    norms_f16 = norms.astype(np.float16).astype(np.float32)
    unit = flat / norms

    # 2. Hadamard rotation.
    y = unit @ H

    # 3. Per-vector qmax via fp16. Same eps guard as the norm.
    qmax = np.maximum(np.abs(y).max(axis=1, keepdims=True), eps)
    qmax_f16 = qmax.astype(np.float16).astype(np.float32)
    scale = qmax_f16 / float(q_range)

    # 4. Scale to lattice coords.
    y_scaled = y / scale

    # 5. Closest-E8 per 8-D block.
    K = D // 8
    y_blocks = y_scaled.reshape(N, K, 8)
    q_lat = closest_e8_np(y_blocks)

    # 6. Clamp.
    q_lat = np.clip(q_lat, -q_range, q_range)

    # 7. Rescale.
    y_hat = (q_lat * scale[..., None]).reshape(N, D)

    # 8. Inverse Hadamard.
    unit_hat = y_hat @ H

    # 9. Restore magnitude.
    x_hat = unit_hat * norms_f16
    return x_hat.reshape(*batch_shape, D).astype(x.dtype)


__all__ = ["closest_d8_np", "closest_e8_np", "roundtrip_np"]
