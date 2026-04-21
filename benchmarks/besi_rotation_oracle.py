#!/usr/bin/env python3
"""Oracle simulation: Besicovitch with optimal per-group rotation vs Haar.

This establishes the UPPER BOUND of what attention-aware Kakeya-set
construction (Perron-tree-style weighted direction density) can deliver.

Procedure:
    For each block of data (real DS-Distill K or V):
      1. Fit a per-group 2×2 covariance → eigendecomposition → rotation R_k.
      2. Apply R_k to each group: x'_k = R_k · x_k  (per-vector, per-group).
      3. Encode x' with vanilla Besicovitch (Haar codebook).
      4. Decode: x̂'_k → x̂_k = R_k^T · x̂'_k.
      5. Measure MSE vs original x, compare against Besi without rotation.

Result interpretation:
    rot_mse / haar_mse ratio shows the MSE gain from principal-axis rotation.
    - Ratio < 0.9 → rotation helps ≥10 %, worth implementing
    - Ratio 0.9-0.98 → marginal, not worth the implementation cost
    - Ratio > 0.98 → noise floor, Haar is already near-optimal

We also report the "Σ_q-weighted oracle" variant for K-stream, where R_k
is derived from Σ_q instead of the data covariance — because that's
what attention actually cares about.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import benchmarks.pre_rope_cache as prc


def haar_codebook(M: int) -> np.ndarray:
    """M × 2 unit-direction codebook on the circle, uniform angular grid."""
    theta = np.pi * np.arange(M) / M
    return np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)


def assign_haar(x: np.ndarray, cb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x: (N, 2); cb: (M, 2). Returns (ids [N], alphas [N])."""
    proj = x @ cb.T  # (N, M)
    abs_proj = np.abs(proj)
    ids = abs_proj.argmax(axis=1)
    alphas = proj[np.arange(x.shape[0]), ids]
    return ids, alphas


def reconstruct(ids: np.ndarray, alphas: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """Reconstruct x̂ = α · d_id from the codebook."""
    return alphas[:, None] * cb[ids]


def encode_group_besi(x_group: np.ndarray, M: int) -> np.ndarray:
    """Full round-trip: encode + decode a (N, 2) group via Haar-M codebook.
    Magnitude stored as float32 (this is an oracle — no magnitude quant)."""
    cb = haar_codebook(M)
    ids, alphas = assign_haar(x_group, cb)
    return reconstruct(ids, alphas, cb)


def encode_group_besi_quantized_mag(
    x_group: np.ndarray, M: int, magnitude_bits: int
) -> np.ndarray:
    """Encode with Haar directions + Lloyd-Max-style magnitude quantization
    (unit-Gaussian centroids, per-vector scale)."""
    cb = haar_codebook(M)
    ids, alphas = assign_haar(x_group, cb)
    # This group's α isn't shared scale with other groups — in real Besi
    # it is.  But for oracle upper-bound testing, treat each group
    # independently: pretend we have 1 group = 1 vector-worth of scale.
    if magnitude_bits == 0:
        # f16 precision
        alphas_q = alphas.astype(np.float16).astype(np.float32)
    else:
        # Per-vector scale = max|α|, then unit-Gaussian Lloyd-Max on α/scale.
        # For a single group there's no "per-vector" — this is per-group.
        # To match real Besi (which uses per-VECTOR scale over all groups),
        # we'd need joint handling.  As an oracle approximation we'll take
        # a global scale = 99th pctile|α|.
        scale = max(np.abs(alphas).max(), 1e-12)
        # Build Gaussian Lloyd-Max centroids approximately
        # (using inverse Gaussian CDF at uniform midpoints)
        from scipy.special import erfinv
        n_bins = 1 << magnitude_bits
        u = (np.arange(n_bins) + 0.5) / n_bins
        centroids = np.sqrt(2) * erfinv(2 * u - 1)  # unit-Gaussian quantiles
        centroids = np.sort(centroids)
        # Quantize α/scale to nearest centroid
        u_norm = alphas / scale
        # vectorized nearest-centroid
        diffs = np.abs(u_norm[:, None] - centroids[None, :])
        idx = diffs.argmin(axis=1)
        alphas_q = centroids[idx] * scale
    return reconstruct(ids, alphas_q, cb)


def besi_block_haar(block: np.ndarray, g: int, M: int,
                    magnitude_bits: int = 0) -> tuple[np.ndarray, float]:
    """Vanilla Besi (Haar) on a (N, D) block.  Returns (reconstruction, MSE)."""
    N, D = block.shape
    assert D % g == 0
    block_mean = block.mean(axis=0, keepdims=True).astype(np.float16).astype(np.float32)
    centered = block - block_mean
    out = np.zeros_like(centered)
    for k in range(D // g):
        x_k = centered[:, k*g:(k+1)*g]
        x_hat = encode_group_besi_quantized_mag(x_k, M, magnitude_bits)
        out[:, k*g:(k+1)*g] = x_hat
    rec = out + block_mean
    mse = ((block - rec) ** 2).mean()
    return rec, float(mse)


def besi_block_rotated(block: np.ndarray, g: int, M: int,
                        magnitude_bits: int = 0,
                        rotation_source: str = "data_cov") -> tuple[np.ndarray, float]:
    """Besi with per-group principal-axis rotation (ORACLE — rotation
    fit from the same block's data covariance, so this is the absolute
    upper bound on what per-(layer, group) offline calibration could
    achieve)."""
    N, D = block.shape
    assert D % g == 0
    block_mean = block.mean(axis=0, keepdims=True).astype(np.float16).astype(np.float32)
    centered = block - block_mean
    out = np.zeros_like(centered)
    for k in range(D // g):
        x_k = centered[:, k*g:(k+1)*g]
        # Fit per-group rotation
        if rotation_source == "data_cov":
            cov = np.cov(x_k.T) if N > g else np.eye(g, dtype=np.float32)
        else:
            raise ValueError(rotation_source)
        # Eigendecomp — for 2x2 we can be explicit
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort descending
        order = np.argsort(eigvals)[::-1]
        R = eigvecs[:, order].astype(np.float32)  # (g, g), columns = principal axes
        # Rotate into principal axes
        x_rot = x_k @ R  # (N, g)
        x_rot_hat = encode_group_besi_quantized_mag(x_rot, M, magnitude_bits)
        # Rotate back
        x_hat = x_rot_hat @ R.T
        out[:, k*g:(k+1)*g] = x_hat
    rec = out + block_mean
    mse = ((block - rec) ** 2).mean()
    return rec, float(mse)


def besi_block_rotated_from_calib(
    block: np.ndarray, g: int, M: int, R_per_group: np.ndarray,
    magnitude_bits: int = 0,
) -> tuple[np.ndarray, float]:
    """Besi with rotation from PRE-CALIBRATED per-group R (what the real
    Rust implementation would do — rotation doesn't depend on the block
    currently being encoded)."""
    N, D = block.shape
    assert D % g == 0
    assert R_per_group.shape == (D // g, g, g)
    block_mean = block.mean(axis=0, keepdims=True).astype(np.float16).astype(np.float32)
    centered = block - block_mean
    out = np.zeros_like(centered)
    for k in range(D // g):
        x_k = centered[:, k*g:(k+1)*g]
        R = R_per_group[k]
        x_rot = x_k @ R
        x_rot_hat = encode_group_besi_quantized_mag(x_rot, M, magnitude_bits)
        x_hat = x_rot_hat @ R.T
        out[:, k*g:(k+1)*g] = x_hat
    rec = out + block_mean
    mse = ((block - rec) ** 2).mean()
    return rec, float(mse)


def load_ds_kv_cache(model_path: str, ctx_len: int = 2048) -> DynamicCache:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager",
        trust_remote_code=True).eval()
    prc.install(model)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join(ds["text"][:200])
    ids = tok(text, return_tensors="pt").input_ids[:, :ctx_len]
    cache = DynamicCache()
    with torch.no_grad():
        model(input_ids=ids, past_key_values=cache, use_cache=True)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--stream", choices=["v", "k"], default="v")
    ap.add_argument("--direction-bits", type=int, default=3)
    ap.add_argument("--magnitude-bits", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=1024)
    args = ap.parse_args()

    print(f"Loading {args.model_path} with ctx={args.ctx_len}...")
    cache = load_ds_kv_cache(args.model_path, args.ctx_len)
    L = len(cache.layers)
    D = cache.layers[0].keys.shape[-1]
    g = 2
    M = 1 << args.direction_bits
    print(f"Model: L={L}, D={D}, g={g}, M={M} directions, m_bits={args.magnitude_bits}")
    print(f"Stream: {args.stream}\n")

    # Three baselines:
    #  (a) haar: vanilla Besi (no rotation)
    #  (b) block-oracle: rotation fit from THIS block (per-block overhead)
    #  (c) global-calib: rotation fit ONCE per layer from ALL blocks (zero per-block overhead — production-realistic)
    print(f"{'L':>3} {'haar_MSE':>11} {'blk_orac':>11} {'gl_calib':>11} "
          f"{'orac/haar':>10} {'calib/haar':>11} {'blk%':>6} {'cal%':>6}")
    print("-" * 80)

    total_haar = 0.0
    total_orac = 0.0
    total_calib = 0.0
    total_cnt = 0

    for l in range(L):
        if args.stream == "k":
            t = cache.layers[l].keys
        else:
            t = cache.layers[l].values
        arr = t.to(torch.float32).cpu().numpy()  # (1, n_kv, seq, D)
        flat = arr.reshape(-1, D)
        bs = args.block_size

        # Step 1: compute the GLOBAL per-group rotation from ALL data of this layer
        flat_centered = flat - flat.mean(axis=0, keepdims=True)
        R_global = np.zeros((D // g, g, g), dtype=np.float32)
        for k in range(D // g):
            grp = flat_centered[:, k*g:(k+1)*g]
            cov = np.cov(grp.T) if flat.shape[0] > g else np.eye(g, dtype=np.float32)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            R_global[k] = eigvecs[:, order].astype(np.float32)

        haar_mses, orac_mses, calib_mses = [], [], []
        i = 0
        while i + bs <= flat.shape[0]:
            blk = flat[i:i+bs]
            _, mse_h = besi_block_haar(blk, g, M, args.magnitude_bits)
            _, mse_o = besi_block_rotated(blk, g, M, args.magnitude_bits)
            _, mse_c = besi_block_rotated_from_calib(blk, g, M, R_global,
                                                       args.magnitude_bits)
            haar_mses.append(mse_h)
            orac_mses.append(mse_o)
            calib_mses.append(mse_c)
            i += bs
        hm = np.mean(haar_mses); om = np.mean(orac_mses); cm = np.mean(calib_mses)
        blk_gain = (1 - om/hm) * 100 if hm > 0 else 0
        cal_gain = (1 - cm/hm) * 100 if hm > 0 else 0
        print(f"{l:>3} {hm:>10.4e} {om:>10.4e} {cm:>10.4e} "
              f"{om/hm:>9.3f}x {cm/hm:>10.3f}x {blk_gain:>5.2f}% {cal_gain:>5.2f}%")
        n = len(haar_mses)
        total_haar += hm * n
        total_orac += om * n
        total_calib += cm * n
        total_cnt += n

    print("-" * 80)
    blk_ratio = total_orac / total_haar if total_haar > 0 else 1.0
    cal_ratio = total_calib / total_haar if total_haar > 0 else 1.0
    print(f"{'ALL':>3} {total_haar/total_cnt:>10.4e} {total_orac/total_cnt:>10.4e} "
          f"{total_calib/total_cnt:>10.4e} "
          f"{blk_ratio:>9.3f}x {cal_ratio:>10.3f}x "
          f"{(1-blk_ratio)*100:>5.2f}% {(1-cal_ratio)*100:>5.2f}%")
    print(f"\nOracle MSE gain summary:")
    print(f"  Per-block oracle (unrealistic, fits R to current block): {(1-blk_ratio)*100:+.2f}%")
    print(f"  Global calibrated  (realistic offline calibration):      {(1-cal_ratio)*100:+.2f}%")
    print(f"\nIf global-calib gain < 5%, Perron-tree weighted codebook won't help.")


if __name__ == "__main__":
    main()
