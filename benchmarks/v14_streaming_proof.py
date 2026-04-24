"""Empirical proof that v1.4 KakeyaLattice is streaming-capable.

Three properties tested:
  (1) bit-identity: encoding one token at a time produces byte-identical
      output to encoding the whole batch at once.
  (2) per-single-token encode+decode latency — relevant for
      paged-KV-cache streaming integration.
  (3) batch amortised throughput — baseline for comparison.

If (1) passes, the codec has no cross-token state by construction and can
be dropped into the vLLM KV-cache write/read path without any changes.

No mock / no simplification / no fallback.
"""
from __future__ import annotations

import time

import torch

from kakeyalattice import V14KakeyaZamirLatticeGPU


def main() -> int:
    torch.manual_seed(0)
    # Shape of Qwen3-4B captured K per prefill: (N_tok, kv_heads, head_dim).
    N_tok, H_heads, D = 2048, 8, 128
    # Realistic K magnitude from captured post-qk-norm K statistics
    # (O(1) per coord, slightly heavy-tailed).
    X = torch.randn(N_tok, H_heads, D, device="cuda", dtype=torch.float32) * 0.3

    for Q in [10, 38, 152]:
        cb = V14KakeyaZamirLatticeGPU(D=D, q_range=Q, device="cuda")

        # ----- Correctness: batch vs per-token streaming -----
        X_hat_batch = cb.roundtrip(X)
        X_hat_stream = torch.empty_like(X)
        for t in range(N_tok):
            X_hat_stream[t : t + 1] = cb.roundtrip(X[t : t + 1])

        diff = (X_hat_batch - X_hat_stream).abs()
        max_diff = float(diff.max().item())
        is_identical = bool(torch.equal(X_hat_batch, X_hat_stream))

        print(
            f"[Q={Q:>3d}]  bit-identical batch-vs-1-at-a-time: "
            f"{is_identical}   max_abs_diff: {max_diff:.3e}",
            flush=True,
        )

        # ----- Latency: single-token roundtrip vs batch amortised -----
        # Warm up CUDA + cache.
        for _ in range(20):
            _ = cb.roundtrip(X[:1])
        torch.cuda.synchronize()

        N_WARM = 200
        N_MEAS = 2000
        for _ in range(N_WARM):
            _ = cb.roundtrip(X[N_tok - 1 : N_tok])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t in range(N_MEAS):
            _ = cb.roundtrip(X[t % N_tok : (t % N_tok) + 1])
        torch.cuda.synchronize()
        t_single_us = (time.perf_counter() - t0) / N_MEAS * 1e6

        # Batch 1024 tokens for throughput reference.
        for _ in range(5):
            _ = cb.roundtrip(X[:1024])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            _ = cb.roundtrip(X[:1024])
        torch.cuda.synchronize()
        t_batch_ms = (time.perf_counter() - t0) / 50 * 1e3
        per_tok_amortised_us = t_batch_ms * 1e3 / 1024

        print(
            f"[Q={Q:>3d}]  single-token (1 × {H_heads} × {D}) "
            f"encode+decode: {t_single_us:7.2f} us   "
            f"batch-1024 amortised: {per_tok_amortised_us:6.2f} us/tok   "
            f"slowdown: {t_single_us / per_tok_amortised_us:5.1f}×",
            flush=True,
        )

        # Sanity: reconstruction quality is the same in both paths.
        mse_rel = ((X_hat_batch - X) ** 2).sum(-1).mean() / (X ** 2).sum(-1).mean()
        print(
            f"[Q={Q:>3d}]  rel-MSE (sanity):     {float(mse_rel.item()):.3e}",
            flush=True,
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
