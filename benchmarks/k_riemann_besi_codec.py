"""Riemannian K-Besi codec — Python implementation for PPL testing.

Pipeline:
    K_whitened ──→ per-group: assign Haar direction in 2D
                ──→ magnitude quantized against per-(layer, group) offline scale
                ──→ reconstruct in whitened space
    harness unwhitens back to raw K space (existing Q-precond unwhiten path)

NOTE: This codec expects k_flat input to already be Q-preconditioned
(whitened) — the harness's whiten-before-codec flow does this
automatically when --q-precondition is passed.

This codec:
    - Has NO per-block skeleton (scale is offline per (layer, group))
    - Uses quantized_with_fixed_scale magnitude mode (breaks trilemma)
    - Retains block-level first-moment via block-mean subtraction

Byte cost per vector (D=128, g=2, direction_bits=db, magnitude_bits=mb):
    block_mean: D × 2 / bs bytes (amortized)
    direction ids: (D/g) × db / 8 bytes
    magnitude ids: (D/g) × mb / 8 bytes
    (no per-vector scale — that's the key win)
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfinv


_CENTROIDS_CACHE: dict[int, np.ndarray] = {}
_HAAR_CACHE: dict[int, np.ndarray] = {}


def lloyd_max_centroids(bits: int) -> np.ndarray:
    if bits in _CENTROIDS_CACHE:
        return _CENTROIDS_CACHE[bits]
    n = 1 << bits
    u = (np.arange(n) + 0.5) / n
    cents = (np.sqrt(2) * erfinv(2 * u - 1)).astype(np.float32)
    _CENTROIDS_CACHE[bits] = cents
    return cents


def haar_codebook(direction_bits: int) -> np.ndarray:
    M = 1 << direction_bits
    if M in _HAAR_CACHE:
        return _HAAR_CACHE[M]
    theta = np.pi * np.arange(M) / M
    cb = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    _HAAR_CACHE[M] = cb
    return cb


def calibrate_offline_scales(k_whitened_flat: np.ndarray, g: int = 2,
                              method: str = "sqrt_trace") -> np.ndarray:
    """Compute per-group offline scales from whitened K data.
    
    method: how to define scale per group:
        'sqrt_trace': sqrt(E[|α|^2]) via trace of 2x2 cov (RMS)
        'rms_alpha': empirical RMS of α after Haar-argmax assignment
        'pct99_alpha': 99th percentile of |α| — better for heavy-tailed data
        'pct999_alpha': 99.9th percentile — most aggressive tail coverage
    """
    D = k_whitened_flat.shape[1]
    n_groups = D // g
    scales = np.zeros(n_groups, dtype=np.float32)
    if method == "sqrt_trace":
        for k in range(n_groups):
            vec = k_whitened_flat[:, k*g:(k+1)*g]
            scales[k] = np.sqrt(max((vec**2).mean(), 1e-12))
    elif method == "rms_alpha":
        cb = haar_codebook(4)
        for k in range(n_groups):
            vec = k_whitened_flat[:, k*g:(k+1)*g]
            proj = vec @ cb.T
            alphas = proj[np.arange(vec.shape[0]), np.abs(proj).argmax(axis=1)]
            scales[k] = np.sqrt(max((alphas**2).mean(), 1e-12))
    elif method in ("pct99_alpha", "pct999_alpha", "pct95_alpha"):
        pct = {"pct95_alpha": 95.0, "pct99_alpha": 99.0,
               "pct999_alpha": 99.9}[method]
        cb = haar_codebook(4)
        for k in range(n_groups):
            vec = k_whitened_flat[:, k*g:(k+1)*g]
            proj = vec @ cb.T
            alphas = proj[np.arange(vec.shape[0]), np.abs(proj).argmax(axis=1)]
            # Scale so that the chosen percentile maps to ~3σ in unit-Gaussian
            # (the typical coverage of Lloyd-Max m=4 centroids).
            # pct99 of |α| → ≈ 2.576σ for Gaussian. So divide by 2.576 to get
            # the equivalent σ.  But for heavy-tailed data we want the
            # 99th pctile to lie near the *edge* of quantization range.
            # Lloyd-Max m=4 (16 centroids) covers roughly [-2.7, 2.7]σ.
            # So scale = pct99(|α|) / 2.7 makes pct99 land at the boundary.
            pct_val = np.percentile(np.abs(alphas), pct)
            # Divisor based on expected m_bits coverage; will pick m=4 as default.
            # For m=4: coverage ≈ 2.7σ (max centroid)
            scales[k] = max(pct_val / 2.7, 1e-12)
    else:
        raise ValueError(method)
    return scales


def encode_decode_block(
    block: np.ndarray, scales: np.ndarray, g: int,
    direction_bits: int, magnitude_bits: int,
    subtract_mean: bool = True,
) -> tuple[np.ndarray, dict]:
    """Round-trip a block through the Riemannian K-Besi codec.
    
    Input `block` should be in whitened space (harness does whiten).
    Output is in same whitened space (harness does unwhiten).
    """
    N, D = block.shape
    n_groups = D // g
    assert scales.shape == (n_groups,)
    cb = haar_codebook(direction_bits)
    centroids = lloyd_max_centroids(magnitude_bits) if magnitude_bits > 0 else None

    if subtract_mean:
        bm = block.mean(axis=0, keepdims=True).astype(np.float16).astype(np.float32)
    else:
        bm = np.zeros((1, D), dtype=np.float32)
    centered = block - bm

    rec = np.zeros_like(centered)
    total_bytes = 0
    # Block mean: D × f16
    total_bytes += D * 2 if subtract_mean else 0
    # Direction + magnitude codes
    dir_bits_per_vec = n_groups * direction_bits
    mag_bits_per_vec = n_groups * (magnitude_bits if magnitude_bits > 0 else 16)
    total_bytes += N * (dir_bits_per_vec + mag_bits_per_vec) // 8

    for k in range(n_groups):
        x_k = centered[:, k*g:(k+1)*g]
        proj = x_k @ cb.T
        ids = np.abs(proj).argmax(axis=1)
        alphas = proj[np.arange(N), ids]
        s_k = scales[k]
        if magnitude_bits == 0:
            alphas_q = alphas.astype(np.float16).astype(np.float32)
        else:
            u = alphas / s_k
            idx = np.abs(u[:, None] - centroids[None, :]).argmin(axis=1)
            alphas_q = centroids[idx] * s_k
        rec[:, k*g:(k+1)*g] = alphas_q[:, None] * cb[ids]

    rec_final = rec + bm
    mse = ((block - rec_final) ** 2).mean()
    return rec_final, {"mean_block_mse": float(mse),
                        "compressed_bytes": int(total_bytes),
                        "bytes_per_vector": float(total_bytes / N)}


def roundtrip_k_whitened(
    k_whitened_flat: np.ndarray, block_size: int, g: int,
    direction_bits: int, magnitude_bits: int,
    scales: np.ndarray, subtract_mean: bool = True,
) -> tuple[np.ndarray, dict]:
    """Round-trip K (already whitened) through Riemannian K-Besi.
    
    Returns (k_reconstructed, report) — both in WHITENED space.
    Harness then unwhitens the result.
    """
    N = k_whitened_flat.shape[0]
    out = np.zeros_like(k_whitened_flat)
    total_bytes = 0
    mse_sum = 0.0
    n_blocks = 0
    i = 0
    while i + block_size <= N:
        blk = k_whitened_flat[i:i+block_size]
        rec, rep = encode_decode_block(
            blk, scales, g, direction_bits, magnitude_bits, subtract_mean)
        out[i:i+block_size] = rec
        total_bytes += rep["compressed_bytes"]
        mse_sum += rep["mean_block_mse"]
        n_blocks += 1
        i += block_size
    if i < N:
        out[i:] = k_whitened_flat[i:]  # tail: keep original (same as Kakeya path)
    return out, {"mean_block_mse": mse_sum / max(n_blocks, 1),
                  "compressed_bytes": total_bytes,
                  "n_blocks": n_blocks}
