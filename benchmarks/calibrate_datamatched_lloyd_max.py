"""Phase 2: data-matched Lloyd-Max centroid calibration on Qwen3-4B K.

Goal: replace TurboQuant's default Gaussian(0, 1)-optimal Lloyd-Max
centroids with centroids solved on the REAL distribution of Hadamard-
rotated, unit-normalised, residual-scaled Qwen3-4B K coordinates.

Why this is the minimum viable "non-Gaussian shaping":
  * Phase 1 measured K is not strictly i.i.d. Gaussian under Hadamard
    (kurtosis ~2.7, W_2/sigma ~0.3, sub-Gaussian body-shape).
  * TQ and Kakeya-v1.3 both use the Gaussian-optimal Lloyd-Max table.
  * A Lloyd-Max table solved on THE actual marginal distribution is
    optimal for that marginal in rel-MSE sense (Max 1960).
  * The distributional difference is exactly what Phase 1 measured,
    so the expected gain is bounded by those metrics (0.5-1.5 dB
    rel-MSE, ≤ 0.2 pp Δppl on Qwen3-4B).

Full Zamir-Feder nested lattice codes (coarse + fine lattice with
shaping boundary) would extend this to multi-dim vector codes but
add weeks of implementation work.  This script is the **scalar
projection** of the data-driven shaping idea — the simplest path to
test whether non-Gaussian shaping gives measurable PPL improvement
on Qwen3-4B before committing to the full nested-lattice machinery.

Output: a raw fp32 file with 2^bit_width centroids, compatible with
the existing `--k-centroids` flag in the Qwen3-4B snapshot harness
(`benchmarks/e2e_ppl_validation_vllm_snapshot_qwen3.py`).

Algorithm:
  1. Capture Qwen3-4B K via the snapshot hook (same protocol as
     Phase 1 / snapA / snapF).
  2. Run the full codec pipeline UP TO Lloyd-Max argmin for each
     non-boundary layer × each kv-head × each block:
         x ∈ R^D (K per token)
         → unit-normalise
         → Hadamard rotate
         → PCA project (per-block, d_eff=96)
         → spherical K-means (flat k=64)
         → residual = coeff − t · centre[seg_id]
         → WHT rotate
         → per-vec scale = 1/‖residual‖
         → `scaled` (this is the Lloyd-Max input)
  3. Collect all `scaled` values across all layers/heads/blocks/dims
     into a flat pool of ~10⁷ samples.
  4. Solve Lloyd-Max for the empirical distribution via Lloyd's
     iteration (assign-to-nearest-centroid + centroid-update-to-
     cluster-mean, 100 iterations or convergence).
  5. Write centroids as raw fp32 file.

The resulting table can be plugged into snapA or any flat-k Kakeya
recipe via `--k-centroids`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


# Environment setup — identical to Phase 1 / snapshot harness.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("KAKEYA_SNAPSHOT_QWEN3", "1")
os.environ.setdefault("KAKEYA_DISABLE_SIGMA_Q", "1")
os.environ.setdefault("KAKEYA_USE_M2_CENTROIDS", "0")
os.environ.setdefault("KAKEYA_OUTLIER_THRESHOLD", "0")


def capture_scaled_residuals(
    model_path: str,
    n_passages: int,
    ctx_len: int,
    gpu_mem_util: float,
    boundary_skip: set[int],
    d_eff: int = 96,
    k_means_k: int = 64,
    block_size: int = 512,
) -> np.ndarray:
    """Run Qwen3-4B prefill on `n_passages` and collect the Lloyd-Max
    INPUT samples (post-WHT, post-scale scalar coords) from all
    non-boundary layers × all kv-heads × all full blocks.

    Returns a 1-D numpy array of all collected `scaled` values.
    """
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    from datasets import load_dataset

    from kakeyaturbo_py.gpu_skeleton import fit_skeleton_batched
    from kakeyaturbo_py.gpu_encode import _wht_rotate_rows_gpu
    from kakeyaturbo_py import _core

    tok = AutoTokenizer.from_pretrained(model_path)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    joined = "\n\n".join(ds["text"])
    full_ids = tok(joined, return_tensors="pt").input_ids[0].tolist()
    passages = [
        full_ids[i * ctx_len : (i + 1) * ctx_len]
        for i in range(n_passages)
        if (i + 1) * ctx_len <= len(full_ids)
    ]
    assert len(passages) == n_passages

    llm = LLM(
        model=model_path, max_model_len=ctx_len + 1,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True, enable_prefix_caching=False,
    )
    from kakeya_v1_3_ppl.snapshot_hook import HookState
    HookState.phase = "capture"

    all_captured: dict[int, list[np.ndarray]] = {}
    for p_idx, ids in enumerate(passages):
        HookState.captured.clear()
        _ = llm.generate(
            [TokensPrompt(prompt_token_ids=ids)],
            SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1),
        )
        for lid, kv in HookState.captured.items():
            all_captured.setdefault(lid, []).append(
                np.asarray(kv["K"], dtype=np.float32),
            )
        print(f"  [capture] passage {p_idx + 1}/{n_passages}")

    # Process each non-boundary layer through the codec pipeline up to
    # Lloyd-Max input, collecting all `scaled` samples.
    all_scaled: list[np.ndarray] = []
    for lid, arrs in all_captured.items():
        if lid in boundary_skip:
            continue
        K_np = np.concatenate(arrs, axis=0)            # [N_tokens, H, D]
        N_tok, H, D = K_np.shape
        n_blocks = N_tok // block_size
        if n_blocks == 0:
            continue
        K_gpu = torch.from_numpy(K_np[: n_blocks * block_size]).cuda().float()

        layer_samples = 0
        for bi in range(n_blocks):
            lo, hi = bi * block_size, (bi + 1) * block_size
            K_block = K_gpu[lo:hi].permute(1, 0, 2).contiguous()  # [H, block, D]

            # Stage 1: PCA skeleton (flat k-means, same as snapA).
            skel = fit_skeleton_batched(
                K_block, d_eff=d_eff, k=k_means_k, seed=3405691582,
                rsvd_oversample=8, rsvd_power_iters=2,
                kmeans_max_iter=8, variance_ratio=1.0,
            )
            mean    = skel["mean"]                      # [H, D]
            basis   = skel["basis"]                     # [H, d_eff, D]
            centres = skel["centers"]                   # [H, k, d_eff]
            wht_len = int(skel["wht_len"])

            # Project to coefficient space.
            coeff = torch.einsum(
                "bnd,bkd->bnk",
                K_block - mean.unsqueeze(1),
                basis,
            )                                           # [H, block, d_eff]

            # Stage 3: K-means assignment (|<coeff, centre>|-argmax).
            cos = torch.einsum("bnc,bkc->bnk", coeff, centres)
            seg_id = cos.abs().argmax(dim=2)
            t      = cos.gather(2, seg_id.unsqueeze(-1)).squeeze(-1)
            coeff_zero = coeff.abs().max(dim=2).values <= torch.finfo(torch.float32).eps
            seg_id = torch.where(coeff_zero, torch.zeros_like(seg_id), seg_id)
            t      = torch.where(coeff_zero, torch.zeros_like(t), t)

            # Stage 4a: residual.
            chosen = centres.gather(
                1, seg_id.unsqueeze(-1).expand(H, block_size, d_eff),
            )
            residual = coeff - t.unsqueeze(-1) * chosen
            residual = torch.where(
                coeff_zero.unsqueeze(-1), torch.zeros_like(residual), residual,
            )

            # Stage 4b: pad + WHT rotate.
            if wht_len > d_eff:
                pad = torch.zeros(
                    H, block_size, wht_len - d_eff,
                    device=K_gpu.device, dtype=torch.float32,
                )
                residual_padded = torch.cat([residual, pad], dim=2)
            else:
                residual_padded = residual
            sign_np = np.asarray(
                _core.wht_sign_pattern(
                    int(skel["rotation_seed"]), int(skel["wht_len"]),
                ),
            ).reshape(-1).astype(np.float32)
            sign = torch.from_numpy(sign_np).cuda()
            flat = residual_padded.reshape(H * block_size, wht_len)
            rotated = _wht_rotate_rows_gpu(flat, sign)  # [H*block, wht_len]

            # Stage 4c: per-vec scale = 1/‖residual‖, apply to rotated.
            res_norm = residual.reshape(H * block_size, d_eff).norm(dim=1)
            eps = torch.finfo(torch.float32).eps
            scale = torch.where(
                res_norm > eps, 1.0 / res_norm, torch.ones_like(res_norm),
            )
            scaled = rotated * scale.unsqueeze(1)        # [H*block, wht_len]

            # Collect all scaled values as Lloyd-Max training samples.
            # Filter out zero-residual rows (their scaled is all ones via
            # the scale-guard).
            valid_rows = res_norm > eps
            if valid_rows.any():
                scaled_valid = scaled[valid_rows]
                all_scaled.append(scaled_valid.detach().cpu().numpy().ravel())
                layer_samples += int(scaled_valid.numel())
        print(f"    layer {lid:>2}: {layer_samples:>10,} Lloyd-Max samples")

    return np.concatenate(all_scaled, axis=0)


def solve_lloyd_max_empirical(
    samples: np.ndarray,
    n_levels: int,
    max_iter: int = 200,
    tol: float = 1e-7,
    subsample_max: int = 2_000_000,
) -> np.ndarray:
    """Lloyd's algorithm on empirical 1-D samples.

    At each iteration:
      * Partition samples by nearest current centroid.
      * Update each centroid to its cluster's mean.
      * Terminate when max |Δcentroid| < tol.

    Args:
      samples:     1-D numpy array of training points.
      n_levels:    number of centroids (2^bit_width).
      max_iter:    Lloyd iteration cap.
      tol:         convergence tolerance.
      subsample_max: cap on samples used (for wall-time).

    Returns:
      numpy fp32 array of sorted centroids, shape [n_levels].
    """
    if samples.size > subsample_max:
        rng = np.random.default_rng(0xBEEF)
        idx = rng.choice(samples.size, subsample_max, replace=False)
        samples = samples[idx]
    samples = np.ascontiguousarray(samples, dtype=np.float64)

    # Initialise centroids at quantiles of the empirical distribution —
    # best starting point for Lloyd's algorithm per Max 1960.
    quantiles = (np.arange(n_levels) + 0.5) / n_levels
    centroids = np.quantile(samples, quantiles)

    for it in range(max_iter):
        # Sort centroids; decision boundaries are midpoints.
        centroids.sort()
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        # Bucketize samples via np.searchsorted — O(N log K).
        bucket = np.searchsorted(boundaries, samples)
        # Update each centroid to its cluster's mean.
        new_centroids = np.empty_like(centroids)
        for k in range(n_levels):
            mask = bucket == k
            if mask.any():
                new_centroids[k] = samples[mask].mean()
            else:
                new_centroids[k] = centroids[k]
        delta = float(np.max(np.abs(new_centroids - centroids)))
        centroids = new_centroids
        if delta < tol:
            print(f"    Lloyd converged at iter {it + 1}, max Δ = {delta:.2e}")
            break
    else:
        print(f"    Lloyd hit max_iter={max_iter}, final Δ = {delta:.2e}")

    centroids.sort()
    return centroids.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--ctx-len",    type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--bit-width", type=int, default=4,
                    help="Bit width → 2^bit_width centroids.  Default 4 "
                         "matches snapA/snapF.")
    ap.add_argument("--boundary-skip-layers", type=int, nargs="*",
                    default=[0, 1, 2, 3, 4, 5, 6, 29, 30, 31, 32, 33, 34, 35])
    ap.add_argument("--d-eff",     type=int, default=96)
    ap.add_argument("--k-kmeans",  type=int, default=64)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--out-dir",   type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(args.boundary_skip_layers)
    n_levels = 1 << args.bit_width

    print(f"[calibrate] bit_width={args.bit_width} → {n_levels} centroids")
    print(f"[calibrate] capturing {args.n_passages} × {args.ctx_len} Qwen3-4B "
          f"prefill, collecting Lloyd-Max inputs …")
    t0 = time.perf_counter()
    samples = capture_scaled_residuals(
        args.model_path, args.n_passages, args.ctx_len, args.gpu_mem_util,
        skip, d_eff=args.d_eff, k_means_k=args.k_kmeans,
        block_size=args.block_size,
    )
    print(f"[calibrate] {samples.size:,} samples collected in "
          f"{time.perf_counter() - t0:.1f}s")

    # Sanity: training samples should have mean ≈ 0, std ≈ 1/sqrt(D)
    # (Hadamard preserves L2, then scale / ||residual|| normalises to unit).
    # Our "scaled" variable has variance per coord ≈ 1/D (D=wht_len=128),
    # so std ≈ 1/√128 ≈ 0.088.
    print(f"[calibrate] sample stats: "
          f"mean={samples.mean():.6f}  "
          f"std={samples.std():.6f}  "
          f"min={samples.min():.4f}  max={samples.max():.4f}")

    # Solve Lloyd-Max on the empirical distribution.
    print(f"[calibrate] solving Lloyd-Max with {n_levels} levels …")
    t1 = time.perf_counter()
    centroids = solve_lloyd_max_empirical(samples, n_levels)
    print(f"[calibrate] Lloyd-Max converged in "
          f"{time.perf_counter() - t1:.1f}s")
    print(f"[calibrate] centroids (sorted):")
    for k, c in enumerate(centroids):
        print(f"    [{k:>2}] {c:+.6f}")

    # Write centroids file in the format the harness expects: raw fp32
    # array of length n_levels.
    out_name = f"qwen3_4b_lloyd_max_datamatched_b{args.bit_width}.f32"
    out_path = args.out_dir / out_name
    centroids.tofile(out_path)
    print(f"[calibrate] written → {out_path}")

    # Also dump a metadata json for provenance.
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps({
        "model": args.model_path,
        "n_passages": args.n_passages,
        "ctx_len": args.ctx_len,
        "bit_width": args.bit_width,
        "n_levels": n_levels,
        "boundary_skip_layers": sorted(skip),
        "d_eff": args.d_eff,
        "k_kmeans": args.k_kmeans,
        "block_size": args.block_size,
        "n_samples": int(samples.size),
        "sample_mean": float(samples.mean()),
        "sample_std":  float(samples.std()),
        "centroids": [float(c) for c in centroids],
    }, indent=2))
    print(f"[calibrate] metadata → {meta_path}")


if __name__ == "__main__":
    main()
