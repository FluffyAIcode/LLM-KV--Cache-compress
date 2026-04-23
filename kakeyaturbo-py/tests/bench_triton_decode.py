"""M5 wall-clock: Triton decode kernel vs Rust decode vs PyTorch-CPU decode.

Measures `decode_block_triton_from_parts` against the Rust in-process
path (`decode_block_from_parts`) on the PR #15 production cell.
Reports absolute and relative speed-up.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")

from kakeyaturbo_py import (
    decode_block_from_parts,
    decode_block_torch_from_parts,
    decode_block_triton_from_parts,
    encode_block_codes,
    triton_is_available,
)

if not triton_is_available():
    raise SystemExit("triton not available")


def bench(device_label, fn, parts, n_iter=50, warmup=5):
    for _ in range(warmup):
        _ = fn(parts)
    if device_label == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = fn(parts)
    if device_label == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000


def main():
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

    def run_rust(p):
        return np.asarray(decode_block_from_parts(p))

    def run_torch(p):
        return decode_block_torch_from_parts(p, device="cpu")

    def run_triton(p):
        return decode_block_triton_from_parts(p, device="cuda")

    t_rust = bench("cpu", run_rust, parts)
    t_torch = bench("cpu", run_torch, parts)
    t_triton = bench("cuda", run_triton, parts)

    out = {
        "shape": [n, d],
        "recipe": "PR #15: inner_product, b=3, randomized PCA rank=64, outlier=2.0",
        "rust_cpu_ms_per_call":   round(t_rust, 3),
        "torch_cpu_ms_per_call":  round(t_torch, 3),
        "triton_cuda_ms_per_call": round(t_triton, 3),
        "speedup_vs_rust":  round(t_rust / t_triton, 2),
        "speedup_vs_torch": round(t_torch / t_triton, 2),
    }
    print(json.dumps(out, indent=2))

    print()
    print(f"Projected per-forward-pass (28 layers × 2 streams = 56 decodes):")
    print(f"  Rust CPU   : {t_rust   * 56 / 1000:.3f} s")
    print(f"  Torch CPU  : {t_torch  * 56 / 1000:.3f} s")
    print(f"  Triton GPU : {t_triton * 56 / 1000:.3f} s  "
          f"({t_rust/t_triton:.1f}× vs Rust)")


if __name__ == "__main__":
    main()
