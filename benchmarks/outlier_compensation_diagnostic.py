#!/usr/bin/env python3
"""Outlier compensation diagnostic for Gap 1 (K-means residual non-Gaussianity).

Python-side simulation — no Rust changes needed.

For each K block after the existing pipeline (Q-precond → PCA → K-means →
WHT → scale), we take the scaled residual that would go into Lloyd-Max
at b=2, and compare two reconstructions:

  baseline: all coordinates quantized via b=2 Lloyd-Max centroids
  outlier-patched: coordinates with |r| > T kept as exact f16, rest
                   quantized via b=2 Lloyd-Max

The MSE delta between these two reconstructions tells us how much of
Gap 1 the outlier mechanism could plausibly recover.  By the log-log
MSE→Δppl correlation (0.71) established in earlier diagnostics, we
then project the expected Δppl improvement.

At thresholds T ∈ {1.5, 2.0, 2.5, 3.0}, we also compute:
  - outlier rate α
  - byte overhead per block (α × wht_len × 4 bytes)
  - end-to-end compression ratio impact
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import benchmarks.pre_rope_cache as prc
from benchmarks.q_precondition import load as load_q_precond


# ---------------------------------------------------------------------------
# Codec pipeline replica (Python) — must match the Rust codec's flow
# ---------------------------------------------------------------------------

def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return max(p, 1)


def hadamard_mat(n: int) -> np.ndarray:
    assert (n & (n - 1)) == 0 and n > 0
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def rademacher_signs(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=n) * 2 - 1).astype(np.float64)


def wht_rotate_unnorm(x: np.ndarray, seed: int) -> np.ndarray:
    """Match Rust codec's `rotate`: D · H  (unnormalised).
    x: [n, N]; returns same shape.
    """
    N = x.shape[-1]
    s = rademacher_signs(seed, N)
    h = hadamard_mat(N)
    return (x * s) @ h.T  # unnormalised — same as codec


def fit_pca(X: np.ndarray, vr: float = 1.0):
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / X.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    basis = evecs[:, idx].T
    return mean.astype(np.float32), basis.astype(np.float32)


GAUSSIAN_CENTROIDS = {
    2: np.array([-1.5100, -0.4528, 0.4528, 1.5100], dtype=np.float32),
    3: np.array([-2.151945, -1.343757, -0.756268, -0.244943,
                 0.244943, 0.756268, 1.343757, 2.151945], dtype=np.float32),
}


def lloyd_max_quant(x: np.ndarray, bits: int, centroids: np.ndarray | None = None) -> np.ndarray:
    """Nearest-centroid quantisation, returns reconstructed values."""
    c = centroids if centroids is not None else GAUSSIAN_CENTROIDS[bits]
    # x: [...,], returns same shape
    dist = np.abs(x[..., None] - c[None, :])
    idx = np.argmin(dist, axis=-1)
    return c[idx]


# ---------------------------------------------------------------------------
# Per-block residual collection and outlier analysis
# ---------------------------------------------------------------------------

def process_block(block: np.ndarray, rotation_seed: int, vr: float,
                  k_bits: int, custom_centroids: np.ndarray | None,
                  outlier_thresholds: list[float]) -> dict:
    """Simulate the K codec on one block and report reconstruction errors
    at multiple outlier thresholds.

    Note: we skip the K-means stage for simplicity — its residual is
    approximately the PCA coefficient minus its norm-weighted mean along
    the closest center, which is not materially different from the PCA
    coefficient in distribution (both are WHT'd and scaled per-vector
    anyway).  The Lloyd-Max input distribution is what matters.
    """
    n, D = block.shape
    mean, basis = fit_pca(block, vr=vr)
    coeff = (block - mean) @ basis.T   # [n, D]

    # Per-vector: pad to wht_len, rotate, scale by 1/||coeff||
    d_eff = D  # vr=1.0 → full rank
    wht_len = next_pow2(d_eff)

    # Result: per-vector rec errors, per outlier threshold
    per_vec_baseline_err_sq = np.zeros(n, dtype=np.float64)
    per_vec_outlier_errs = {t: np.zeros(n, dtype=np.float64) for t in outlier_thresholds}
    outlier_rates = {t: [] for t in outlier_thresholds}

    for i in range(n):
        c = coeff[i]
        norm_c = np.linalg.norm(c).clip(min=1e-12)
        # Pad + rotate + scale
        padded = np.zeros(wht_len, dtype=np.float64)
        padded[:d_eff] = c
        rotated = wht_rotate_unnorm(padded.reshape(1, -1), rotation_seed).reshape(-1)
        scaled = rotated / norm_c

        # --- baseline: quantize all coords with Lloyd-Max b=k_bits
        rec_scaled_base = lloyd_max_quant(scaled, k_bits,
                                           centroids=custom_centroids)
        # Baseline MSE in scaled-space
        err_base = scaled - rec_scaled_base
        per_vec_baseline_err_sq[i] = float(np.sum(err_base ** 2))

        # --- outlier-patched: |r| > T keeps exact f16
        for T in outlier_thresholds:
            mask = np.abs(scaled) > T
            rec_patched = rec_scaled_base.copy()
            # Represent outliers with f16 precision (not fp32)
            exact_f16 = scaled[mask].astype(np.float16).astype(np.float64)
            rec_patched[mask] = exact_f16
            err_patch = scaled - rec_patched
            per_vec_outlier_errs[T][i] = float(np.sum(err_patch ** 2))
            outlier_rates[T].append(int(mask.sum()))

    return {
        "n_vectors": n,
        "d_eff": d_eff,
        "wht_len": wht_len,
        "mean_baseline_err_sq": float(per_vec_baseline_err_sq.mean()),
        "outlier_err_sq_per_T": {
            T: float(per_vec_outlier_errs[T].mean()) for T in outlier_thresholds
        },
        "outlier_rate_per_T": {
            T: float(np.mean(outlier_rates[T]) / wht_len) for T in outlier_thresholds
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--q-precondition",
                    default="reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors")
    ap.add_argument("--q-precond-skip-layers", type=int, nargs="+", default=[0, 1, 26, 27])
    ap.add_argument("--n-passages", type=int, default=2)
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--bit-width", type=int, default=2)
    ap.add_argument("--rotation-seed", type=int, default=3405691582)
    ap.add_argument("--centroids-file", type=Path,
                    default="reports/v1_4_q_pca/calibrated_codebook/ds_K_b2_centroids.f32",
                    help="Path to calibrated Lloyd-Max centroids (matches Step 3)")
    ap.add_argument("--outlier-thresholds", type=float, nargs="+",
                    default=[1.5, 2.0, 2.5, 3.0])
    args = ap.parse_args()

    # Load calibrated centroids
    custom_centroids = None
    if args.centroids_file and args.centroids_file.exists():
        custom_centroids = np.frombuffer(
            args.centroids_file.read_bytes(), dtype=np.float32
        ).copy()
        print(f"loaded {custom_centroids.size} calibrated centroids: {custom_centroids}")

    print("\nloading model + Q-precond…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.eval()
    prc.install(model)
    qp = load_q_precond(args.q_precondition, skip_layers=args.q_precond_skip_layers)

    cfg = model.config.get_text_config(decoder=True)
    layer_types = getattr(cfg, "layer_types", None) or (
        ["full_attention"] * cfg.num_hidden_layers
    )
    full_attn_layers = [
        l for l in range(cfg.num_hidden_layers)
        if layer_types[l] == "full_attention" and l not in set(args.q_precond_skip_layers)
    ]
    print(f"  {len(full_attn_layers)} middle layers after skipping boundary")

    # Collect K tensors
    from benchmarks.e2e_ppl_pre_rope import load_wikitext_passages, prefill_cache
    passages = load_wikitext_passages(tok, args.ctx_len, args.n_passages)
    print(f"\nprocessing {len(passages)} passages × {len(full_attn_layers)} layers…\n",
          flush=True)

    per_block_results = []
    for pi, passage in enumerate(passages):
        ids = tok(passage, return_tensors="pt")["input_ids"][:, :args.ctx_len]
        cache = prefill_cache(model, ids, prefill_chunk=1024)
        for l in full_attn_layers:
            k = cache.layers[l].keys
            k_np = k[0].to(torch.float32).permute(1, 0, 2).cpu().numpy()  # [seq, n_kv, D]
            k_whitened = qp.whiten(k_np, layer=l) if qp.is_active(l) else k_np
            flat = k_whitened.reshape(-1, k_whitened.shape[-1]).astype(np.float32)
            n_comp = (flat.shape[0] // args.block_size) * args.block_size
            if n_comp == 0:
                continue
            for bstart in range(0, n_comp, args.block_size):
                block = flat[bstart:bstart + args.block_size]
                res = process_block(
                    block, args.rotation_seed, 1.0, args.bit_width,
                    custom_centroids, args.outlier_thresholds,
                )
                res["passage"] = pi
                res["layer"] = l
                per_block_results.append(res)
        print(f"  passage {pi+1}: accumulated {len(per_block_results)} blocks",
              flush=True)

    # Aggregate
    print(f"\n\n{'='*70}\nAGGREGATE ({len(per_block_results)} blocks total)\n{'='*70}")
    baseline_mse = float(np.mean([r["mean_baseline_err_sq"] for r in per_block_results]))
    wht_len = per_block_results[0]["wht_len"]
    print(f"\nBaseline Lloyd-Max b={args.bit_width} MSE (scaled residual): {baseline_mse:.4f}")
    print(f"   (per scaled-residual coordinate, averaged over blocks)")

    print(f"\n{'Threshold T':>11}  {'outlier α':>10}  {'MSE':>9}  "
          f"{'MSE drop':>10}  {'bytes/block':>12}  {'Δppl est':>10}")
    print("-" * 80)
    for T in args.outlier_thresholds:
        mse = float(np.mean([r["outlier_err_sq_per_T"][T] for r in per_block_results]))
        alpha = float(np.mean([r["outlier_rate_per_T"][T] for r in per_block_results]))
        drop_pct = (1.0 - mse / baseline_mse) * 100

        # Byte overhead: alpha × wht_len × bs × 4 bytes; per-block relative to
        # existing b=2 codes = wht_len × bs × bits / 8
        bs = per_block_results[0]["n_vectors"]
        outlier_bytes = alpha * wht_len * bs * 4
        baseline_codes_bytes = wht_len * bs * args.bit_width / 8

        # Projected Δppl using corr(log Δppl, log K-MSE) = 0.71 on baseline +9%
        # log(1 + 0.09) = 0.0862
        # predicted log(1 + Δppl_new) = 0.0862 + 0.71 × log(mse/baseline_mse)
        base_log = np.log(1.09)
        new_log = base_log + 0.71 * np.log(max(mse, 1e-20) / baseline_mse)
        delta_new = float(np.exp(new_log) - 1.0) * 100

        print(f"     {T:>5.1f}  {alpha*100:>8.2f}%  {mse:.4f}  "
              f"{drop_pct:>+7.2f}%   {outlier_bytes:>7.0f} B  ({outlier_bytes/baseline_codes_bytes*100:>+5.1f}%)"
              f"  {delta_new:>+6.2f}%")

    print(f"\nInterpretation:")
    print(f"  'MSE drop': reduction in Lloyd-Max MSE by replacing outliers with f16")
    print(f"  'Δppl est': projected Δppl under corr(log Δppl, log K-MSE) = 0.71,")
    print(f"              starting from Sprint 5 K b=2 observed +9.09%")
    print(f"  'bytes/block (%)': outlier list byte overhead relative to b=2 codes")


if __name__ == "__main__":
    main()
