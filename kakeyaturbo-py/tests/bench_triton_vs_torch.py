"""M4 Phase B wall-clock: Triton STORE kernel vs PyTorch reference.

This is not a test — a measurement script whose output lands in
`reports/v1_3_ppl/vllm_backend/M4_PHASE_B_REPORT.md`.

Runs on CUDA only (Triton requires GPU).  On H200, expect the fused
Triton kernel to dominate CPU PyTorch by a wide margin (GPU bandwidth
+ fused WHT+quant+outlier in a single launch vs three separate torch
ops on CPU).
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")

from kakeyaturbo_py import (
    encode_block_codes,
    encode_block_torch_stage2,
    encode_block_triton_stage2,
    triton_is_available,
)

if not triton_is_available():
    raise SystemExit("triton not available")


def bench(device_label: str, fn, X_np, parts, n_iter=50, warmup=5):
    for _ in range(warmup):
        _ = fn(X_np, parts)
    if device_label == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = fn(X_np, parts)
    if device_label == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000   # ms/call


def main():
    # Representative shape: the PR #15 production cell.
    n, d = 512, 128
    rng = np.random.default_rng(20260422)
    X = (rng.standard_normal((n, d)) * 0.3).astype(np.float32)

    parts = encode_block_codes(
        X,
        metric="inner_product", block_size=n, bit_width=3,
        variance_ratio=0.95, k=16, rotation_seed=3405691582,
        pca_method="randomized", rsvd_target_rank=64,
        rsvd_oversample=8, rsvd_power_iters=2,
        skeleton_dtype="fp16", share_basis=False,
        outlier_threshold=2.0,
    )
    parts = {k: (np.asarray(v) if hasattr(v, "shape") else v)
             for k, v in parts.items()}

    def run_torch(x, p):
        return encode_block_torch_stage2(
            x, p, outlier_threshold=2.0, device="cpu")

    def run_triton(x, p):
        return encode_block_triton_stage2(
            x, p, outlier_threshold=2.0, device="cuda")

    t_torch = bench("cpu", run_torch, X, parts)
    t_triton = bench("cuda", run_triton, X, parts)

    out = {
        "shape": [n, d],
        "recipe": "PR #15: inner_product, b=3, randomized PCA rank=64, outlier=2.0",
        "torch_cpu_ms_per_call":  round(t_torch, 3),
        "triton_cuda_ms_per_call": round(t_triton, 3),
        "speedup":                round(t_torch / t_triton, 2),
    }
    print(json.dumps(out, indent=2))

    # Projected per-forward cost: 28 layers × 2 streams = 56 calls/forward
    print()
    print(f"Projected per-forward-pass cost (28 layers × 2 streams = 56 calls):")
    print(f"  CPU torch : {t_torch * 56 / 1000:.2f} s")
    print(f"  Triton GPU: {t_triton * 56 / 1000:.3f} s  ({t_torch/t_triton:.1f}× faster)")


if __name__ == "__main__":
    main()
