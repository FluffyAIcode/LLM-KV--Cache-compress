"""Diagnostic: where does the non-bit-identity come from?

Hypothesis: The v1.4 codec has NO cross-token state by construction
(pure per-vector map).  Any batch-vs-streaming difference must therefore
come from GPU non-determinism in the Hadamard GEMM (`unit @ H` uses
different cuBLAS kernels at different batch sizes, producing ULP-level
numerical drift; a few coordinates then fall to different D4 lattice
points at round() boundaries).

Three tests:
  A. Re-running the SAME token 10 times in a row  — is the output
     identical call-to-call?  (Yes ⇒ no hidden mutable state.)
  B. Batch=1 vs Batch=N  — ULP-level diff in GEMM outputs?
  C. Streaming-path rel-MSE vs batch-path rel-MSE — are the
     reconstructions statistically equivalent, even if not bit-identical?

If A passes + C shows MSE equivalence, streaming is semantically
equivalent.  (B ULP drift is a GPU-determinism caveat shared by TQ.)
"""
from __future__ import annotations

import torch

from kakeyaturbo_py import V14KakeyaZamirLatticeGPU


def main() -> int:
    torch.manual_seed(0)
    N_tok, H, D = 2048, 8, 128
    X = torch.randn(N_tok, H, D, device="cuda", dtype=torch.float32) * 0.3

    for Q in [10, 38, 152]:
        cb = V14KakeyaZamirLatticeGPU(D=D, q_range=Q, device="cuda")
        print(f"\n========== Q={Q} ==========")

        # ----- Test A: re-run same input, is output identical? -----
        x0 = X[:1]
        out0 = cb.roundtrip(x0)
        identical_repeat = True
        for _ in range(10):
            out_n = cb.roundtrip(x0)
            if not torch.equal(out0, out_n):
                identical_repeat = False
                break
        print(f"  [A] same input → same output across 10 calls: {identical_repeat}")

        # ----- Test B: batch-size-dependent numerical drift? -----
        # Compare batch-1 output for token 0 vs batch-N output indexed at 0.
        out_single = cb.roundtrip(X[:1])
        out_batch = cb.roundtrip(X)[:1]
        bit_eq_single_vs_batchN = bool(torch.equal(out_single, out_batch))
        max_diff_single_vs_batchN = float((out_single - out_batch).abs().max().item())

        # Compare batch-2 vs batch-4 vs batch-N for the same token 0.
        out_b2 = cb.roundtrip(X[:2])[:1]
        out_b4 = cb.roundtrip(X[:4])[:1]
        out_b8 = cb.roundtrip(X[:8])[:1]
        out_b32 = cb.roundtrip(X[:32])[:1]
        batch_sweep_diffs = {
            "b=1 vs b=2":  float((out_single - out_b2).abs().max().item()),
            "b=1 vs b=4":  float((out_single - out_b4).abs().max().item()),
            "b=1 vs b=8":  float((out_single - out_b8).abs().max().item()),
            "b=1 vs b=32": float((out_single - out_b32).abs().max().item()),
            "b=1 vs b=N":  max_diff_single_vs_batchN,
        }
        print(f"  [B] bit-identical single-vs-batch: {bit_eq_single_vs_batchN}")
        print(f"  [B] max abs diff vs batch size:")
        for k, v in batch_sweep_diffs.items():
            print(f"      {k:<16}  {v:.3e}")

        # ----- Test C: streaming vs batch rel-MSE equivalence -----
        # Key metric: does streaming path give statistically the same
        # reconstruction quality as batch?
        X_hat_batch  = cb.roundtrip(X)
        X_hat_stream = torch.empty_like(X)
        for t in range(N_tok):
            X_hat_stream[t : t + 1] = cb.roundtrip(X[t : t + 1])

        rel_mse_batch  = float(
            ((X_hat_batch - X) ** 2).sum(-1).mean() /
            (X ** 2).sum(-1).mean()
        )
        rel_mse_stream = float(
            ((X_hat_stream - X) ** 2).sum(-1).mean() /
            (X ** 2).sum(-1).mean()
        )
        mean_pair_diff = float((X_hat_batch - X_hat_stream).abs().mean())
        n_diff_coords = int((X_hat_batch != X_hat_stream).sum().item())
        n_total_coords = int(X.numel())
        pct_coords_differ = n_diff_coords / n_total_coords * 100

        print(f"  [C] rel-MSE batch-mode  = {rel_mse_batch:.4e}")
        print(f"  [C] rel-MSE stream-mode = {rel_mse_stream:.4e}")
        print(f"  [C] abs relative gap    = "
              f"{abs(rel_mse_batch - rel_mse_stream) / rel_mse_batch * 100:.3f}%")
        print(f"  [C] mean-abs(batch - stream) = {mean_pair_diff:.3e}")
        print(f"  [C] fraction of coords that differ: "
              f"{pct_coords_differ:.2f}% "
              f"({n_diff_coords} / {n_total_coords})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
