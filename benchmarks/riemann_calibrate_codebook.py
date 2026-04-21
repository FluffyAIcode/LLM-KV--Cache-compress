#!/usr/bin/env python3
"""Calibrate empirical Lloyd-Max centroids for Riemannian K-Besi α distribution.

Procedure:
    1. Load K-cache from calibration prefills (multiple WikiText passages)
    2. For each non-boundary layer:
       - Whiten K via Q-precond
       - Compute α = projection onto Haar-argmax direction (d=direction_bits)
       - Normalize by per-(layer, group) offline scale (pct99_alpha method)
    3. Pool normalized α from all non-boundary layers → empirical distribution
    4. Run Lloyd-Max iteration on pooled α distribution for the target bit width
    5. Save centroids as .f32 binary (1 << magnitude_bits entries)

The pooled (non-per-layer) calibration matches the observation from the
diagnostic that normalized α has P99 in 3.3-6.1 across layers — stable
enough that a single codebook captures all layers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

import benchmarks.pre_rope_cache as prc
from benchmarks.q_precondition import load as load_qp
from benchmarks.k_riemann_besi_codec import (
    calibrate_offline_scales, haar_codebook,
)


def lloyd_max_iterate(samples: np.ndarray, n_centroids: int,
                      n_iter: int = 100, tol: float = 1e-7) -> np.ndarray:
    """Standard Lloyd-Max: alternate between Voronoi-region re-centering
    and bin-boundary re-placement until convergence.

    samples: 1-D array of α values
    Returns: sorted centroid array of length n_centroids
    """
    s_sorted = np.sort(samples)
    # Initialize via quantile placement
    targets = (np.arange(n_centroids) + 0.5) / n_centroids
    centroids = np.quantile(s_sorted, targets).astype(np.float64)
    for it in range(n_iter):
        # Assign each sample to nearest centroid
        bounds = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(bounds, s_sorted)
        # Recompute centroids as means of each bin
        new_c = np.zeros_like(centroids)
        for k in range(n_centroids):
            mask = (idx == k)
            if mask.any():
                new_c[k] = s_sorted[mask].mean()
            else:
                new_c[k] = centroids[k]
        shift = np.abs(new_c - centroids).max()
        centroids = new_c
        if shift < tol:
            break
    return centroids.astype(np.float32)


def collect_alpha_distribution(model_path: str, qp_path: str,
                               skip_layers: list[int],
                               direction_bits: int,
                               scale_method: str = "pct99_alpha",
                               ctx_len: int = 2048,
                               n_passages: int = 4) -> np.ndarray:
    """Collect pooled normalized α from all non-boundary layers."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager",
        trust_remote_code=True).eval()
    prc.install(model)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    qp = load_qp(qp_path, skip_layers=skip_layers)

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    # Collect multiple passages for a larger calibration dataset
    passages = []
    cursor = 0
    # Use different chunks of the dataset for each passage
    text_blocks = [ds["text"][cursor:cursor + 200] for _ in range(n_passages)]
    for i in range(n_passages):
        text = "\n".join(ds["text"][cursor:cursor + 200])
        cursor += 200
        tokens = tok(text, return_tensors="pt").input_ids[:, :ctx_len]
        if tokens.shape[-1] < ctx_len:
            continue
        passages.append(tokens)

    D = 128; g = 2
    cb = haar_codebook(direction_bits)
    all_normalized_alphas = []

    for pi, ids in enumerate(passages):
        print(f"  passage {pi}: ctx={ids.shape[-1]}...")
        cache = DynamicCache()
        with torch.no_grad():
            model(input_ids=ids, past_key_values=cache, use_cache=True)
        for L in range(len(cache.layers)):
            if not qp.is_active(L):
                continue
            k = cache.layers[L].keys.to(torch.float32).cpu().numpy()
            k_sw = k.squeeze(0).transpose(1, 0, 2)
            k_wht = qp.whiten(k_sw, layer=L).reshape(-1, D)
            # Per-group offline scale
            scales = calibrate_offline_scales(k_wht, g=g, method=scale_method)
            # α = argmax direction projection
            for gi in range(D // g):
                vec = k_wht[:, gi*g:(gi+1)*g]
                proj = vec @ cb.T
                idx = np.abs(proj).argmax(axis=1)
                a = proj[np.arange(vec.shape[0]), idx]
                a_norm = a / max(scales[gi], 1e-9)  # normalize to unit-scale
                all_normalized_alphas.append(a_norm.astype(np.float32))

    return np.concatenate(all_normalized_alphas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--qp-path",
                    default="reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors")
    ap.add_argument("--out-path", type=Path, required=True,
                    help="Output .f32 binary file for centroids")
    ap.add_argument("--magnitude-bits", type=int, default=4)
    ap.add_argument("--direction-bits", type=int, default=6,
                    help="Direction_bits used when collecting α "
                         "(should match encoder config).")
    ap.add_argument("--skip-layers", type=int, nargs="*",
                    default=[0, 1, 26, 27])
    ap.add_argument("--scale-method", default="pct99_alpha")
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--n-lloyd-iter", type=int, default=200)
    args = ap.parse_args()

    print(f"Collecting α distribution from {args.n_passages} passages...")
    alphas = collect_alpha_distribution(
        args.model_path, args.qp_path, args.skip_layers,
        args.direction_bits, args.scale_method,
        args.ctx_len, args.n_passages)
    print(f"Collected {alphas.size:,} normalized α samples")
    print(f"  mean={alphas.mean():+.3f}  std={alphas.std():.3f}")
    print(f"  P1/P50/P99 = {np.percentile(alphas, [1, 50, 99])}")
    # Kurtosis
    z = alphas / alphas.std()
    kurt = (z**4).mean() - 3
    print(f"  kurtosis = {kurt:.2f} (unit-Gaussian = 0)")

    n_centroids = 1 << args.magnitude_bits
    print(f"\nRunning Lloyd-Max with {n_centroids} centroids, {args.n_lloyd_iter} iters...")
    centroids = lloyd_max_iterate(alphas, n_centroids, args.n_lloyd_iter)
    print(f"Converged centroids: {centroids}")

    # Compare to unit-Gaussian baseline
    from scipy.special import erfinv
    u = (np.arange(n_centroids) + 0.5) / n_centroids
    gauss_centroids = np.sqrt(2) * erfinv(2 * u - 1)
    print(f"Unit-Gaussian baseline: {gauss_centroids.astype(np.float32)}")

    # MSE of quantization on the empirical distribution
    def quant_mse(samples, centroids):
        diffs = np.abs(samples[:, None] - centroids[None, :])
        return (samples - centroids[diffs.argmin(axis=1)]) ** 2

    mse_cal = quant_mse(alphas, centroids).mean()
    mse_gauss = quant_mse(alphas, gauss_centroids.astype(np.float32)).mean()
    print(f"\nQuantization MSE on empirical α:")
    print(f"  calibrated centroids: {mse_cal:.5e}")
    print(f"  unit-Gaussian:        {mse_gauss:.5e}")
    print(f"  reduction: {(1 - mse_cal/mse_gauss)*100:.1f}%")

    # Save to .f32 binary
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    centroids.astype(np.float32).tofile(args.out_path)
    print(f"\nSaved {n_centroids} centroids to {args.out_path}")


if __name__ == "__main__":
    main()
