"""v1.5 E8 encode/decode latency benchmark vs v1.4 D4 vs TurboQuant.

Measures pure codec roundtrip wall time on H200 at realistic per-head-vector
shapes. Does NOT need vLLM or a model — it times the codec in isolation to
validate the theoretical +25-30% overhead prediction from the v1.5 scaffold.

Three codecs, matched Q/b points:
  * v1.4 D4: Q=4, 10, 38, 152
  * v1.5 E8: Q=4, 10, 38, 152
  * TQ:      b=2 (boundary-reserved; per Phase β guard), 3, 4, 6, 8

Per-codec procedure:
  1. Build the codec on CUDA.
  2. Allocate a realistic input tensor: shape (N, H, D) fp32 on GPU.
     N=2048 (prefill batch size), H=8 (Qwen3-4B KV heads), D=128.
  3. Warm up 20 iterations (JIT, kernel cache, cuBLAS plan selection).
  4. Run N_ITERS encode+decode roundtrips with torch.cuda.synchronize()
     before & after each iteration to get true wall time.
  5. Report mean, stdev, p50, p99 over N_ITERS iterations.

Also reports:
  * Effective throughput (Msamples/sec)
  * Per-vector decode cost (μs/vector)
  * Overhead vs bf16 memcpy (baseline: raw memcpy from K/V projection output
    to a freshly-allocated buffer on same device)

Compliance: real GPU, CUDA.synchronize gates timing precisely, no mock,
no fallback.  TQ b=2 deployment guardrail: the benchmark measures b=2
encode cost but the REPORT annotates that b=2 requires boundary-layer
protection to avoid Δppl divergence (per Phase β findings on Qwen3-4B
where TQ b=2 no-boundary produces 145,000% Δppl).

Usage:
  python benchmarks/e8_latency_benchmark.py --n-iters 500
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch


def _sylvester_hadamard_normalised(D: int, device) -> torch.Tensor:
    assert (D & (D - 1)) == 0, f"D must be power of 2, got {D}"
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], dim=0)
    return H / math.sqrt(D)


def make_bf16_memcpy_fn(D: int, device: str = "cuda"):
    """Baseline: raw bf16 cast + copy (simulates bf16 KV store without quant)."""
    def fn(X):
        return X.to(torch.bfloat16).to(torch.float32)
    fn.bits_per_token_per_head = 16 * D
    fn.channel_id = "bf16_memcpy"
    fn.label = "bf16 memcpy (baseline)"
    return fn


def make_v14_d4_fn(D: int, Q: int, device: str = "cuda"):
    from kakeyalattice import V14KakeyaZamirLatticeGPU
    cb = V14KakeyaZamirLatticeGPU(D=D, q_range=Q, device=device)
    def fn(X): return cb.roundtrip(X)
    fn.bits_per_token_per_head = cb.bits_per_token_per_head
    fn.channel_id = f"v14_Q{Q}"
    fn.label = f"v1.4 D4 Q={Q}"
    return fn


def make_v15_e8_fn(D: int, Q: int, device: str = "cuda"):
    from kakeyalattice import V15KakeyaZamirE8GPU
    cb = V15KakeyaZamirE8GPU(D=D, q_range=Q, device=device)
    def fn(X): return cb.roundtrip(X)
    fn.bits_per_token_per_head = cb.bits_per_token_per_head
    fn.channel_id = f"v15_Q{Q}"
    fn.label = f"v1.5 E8 Q={Q}"
    return fn


def make_tq_fn(D: int, b: int, device: str = "cuda"):
    H = _sylvester_hadamard_normalised(D, device)
    qs = (1 << (b - 1)) - 1
    eps = torch.finfo(torch.float32).eps
    bits = D * b + 32
    def fn(X):
        assert X.is_cuda and X.dtype == torch.float32
        N_tok, H_heads, _ = X.shape
        flat = X.reshape(-1, D)
        norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
        norms_f16 = norms.to(torch.float16).to(torch.float32)
        unit = flat / norms
        y = unit @ H
        qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
        qmax_f16 = qmax.to(torch.float16).to(torch.float32)
        scale = qmax_f16 / float(qs)
        q = torch.round(y / scale).clamp(-qs, qs) * scale
        return ((q @ H) * norms_f16).reshape(N_tok, H_heads, D)
    fn.bits_per_token_per_head = bits
    fn.channel_id = f"tq_b{b}"
    fn.label = f"TQ b={b}"
    return fn


def bench_one(codec_fn, X, n_iters: int, n_warmup: int = 20) -> dict:
    """Time encode+decode roundtrip. Returns stats in microseconds."""
    # Warmup
    for _ in range(n_warmup):
        _ = codec_fn(X)
    torch.cuda.synchronize()

    per_iter_us: list[float] = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = codec_fn(X)
        torch.cuda.synchronize()
        per_iter_us.append((time.perf_counter() - t0) * 1e6)

    n_tokens, n_heads, D = X.shape
    n_vecs = n_tokens * n_heads

    per_iter_us_sorted = sorted(per_iter_us)
    return {
        "channel_id": codec_fn.channel_id,
        "label": codec_fn.label,
        "bits_per_token_per_head": codec_fn.bits_per_token_per_head,
        "n_iters": n_iters,
        "shape": [n_tokens, n_heads, D],
        "total_vecs_per_call": n_vecs,
        "mean_us":  statistics.fmean(per_iter_us),
        "stdev_us": statistics.stdev(per_iter_us) if len(per_iter_us) > 1 else 0.0,
        "min_us":   min(per_iter_us),
        "p50_us":   per_iter_us_sorted[len(per_iter_us) // 2],
        "p99_us":   per_iter_us_sorted[min(len(per_iter_us) - 1,
                                           int(len(per_iter_us) * 0.99))],
        "max_us":   max(per_iter_us),
        "per_vec_mean_us": statistics.fmean(per_iter_us) / n_vecs,
        "throughput_Msamples_per_sec": (
            n_vecs / (statistics.fmean(per_iter_us) / 1e6) / 1e6
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iters", type=int, default=500)
    ap.add_argument("--n-tokens", type=int, default=2048,
                    help="Tokens per codec call (prefill batch size)")
    ap.add_argument("--n-heads", type=int, default=8,
                    help="KV heads (Qwen3-4B=8, others=2)")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("reports/v1_4_release/rigorous_eval/v15_vs_v14_vs_tq"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    D, N, H = args.head_dim, args.n_tokens, args.n_heads
    print(f"[config] D={D}, N={N}, H={H}, n_iters={args.n_iters}", flush=True)
    print(f"[device] {torch.cuda.get_device_name(0)}", flush=True)

    # Identical input across all codecs (fair comparison).
    torch.manual_seed(20260424)
    X = torch.randn(N, H, D, device="cuda", dtype=torch.float32) * 0.3

    # Build codec lineup.
    codecs = [make_bf16_memcpy_fn(D)]
    for Q in [4, 10, 38, 152]:
        codecs.append(make_v14_d4_fn(D, Q))
        codecs.append(make_v15_e8_fn(D, Q))
    for b in [2, 3, 4, 6, 8]:
        codecs.append(make_tq_fn(D, b))

    # Per-codec timing.
    results = []
    print(f"\n{'Codec':<18} {'bits':>5} {'mean μs':>10} {'p50 μs':>10} "
          f"{'p99 μs':>10} {'stdev':>9} {'Msamp/s':>10} "
          f"{'μs/vec':>10}", flush=True)
    print("-" * 88, flush=True)
    for codec in codecs:
        r = bench_one(codec, X, args.n_iters)
        r["q_or_b"] = codec.channel_id
        results.append(r)
        print(f"{r['label']:<18} {r['bits_per_token_per_head']:>5} "
              f"{r['mean_us']:>10.2f} {r['p50_us']:>10.2f} "
              f"{r['p99_us']:>10.2f} {r['stdev_us']:>9.2f} "
              f"{r['throughput_Msamples_per_sec']:>10.2f} "
              f"{r['per_vec_mean_us']:>10.3f}", flush=True)

    # Compute overhead ratios.
    bf16_mean = next(r["mean_us"] for r in results if r["channel_id"] == "bf16_memcpy")
    print(f"\nOverhead vs bf16 memcpy (baseline {bf16_mean:.2f} μs):", flush=True)
    for r in results:
        if r["channel_id"] == "bf16_memcpy":
            continue
        over = r["mean_us"] / bf16_mean
        r["overhead_vs_bf16"] = over

    # Pair v1.4 vs v1.5 at matching Q.
    print("\nv1.5 E8 vs v1.4 D4 at matched Q:", flush=True)
    print(f"  {'Q':<4} {'v14 μs':>10} {'v15 μs':>10} {'ratio v15/v14':>15} "
          f"{'v14 bits':>9} {'v15 bits':>9}", flush=True)
    for Q in [4, 10, 38, 152]:
        v14 = next((r for r in results if r["channel_id"] == f"v14_Q{Q}"), None)
        v15 = next((r for r in results if r["channel_id"] == f"v15_Q{Q}"), None)
        if v14 and v15:
            ratio = v15["mean_us"] / v14["mean_us"]
            print(f"  {Q:<4} {v14['mean_us']:>10.2f} {v15['mean_us']:>10.2f} "
                  f"{ratio:>14.3f}x "
                  f"{v14['bits_per_token_per_head']:>9} "
                  f"{v15['bits_per_token_per_head']:>9}", flush=True)

    # TQ b=2 deployment caveat
    print("\n[CAVEAT] TQ b=2 deployment guardrail:", flush=True)
    print("  TQ b=2 encode time is MEASURED here (low, similar to b=3 per the", flush=True)
    print("  Hadamard-dominated cost), but DEPLOYMENT requires boundary-layer", flush=True)
    print("  protection (first 2 + last 2 layers kept bf16) to avoid Δppl", flush=True)
    print("  divergence — see Phase β findings where TQ b=2 no-boundary on", flush=True)
    print("  Qwen3-4B produced 145,000% Δppl.  With boundary=2, b=2 is the", flush=True)
    print("  aggressive-edge TQ point and still usable.", flush=True)

    # Save raw results.
    out_path = args.out_dir / "e8_latency_benchmark.json"
    out_path.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "config": {
            "head_dim": D, "n_tokens": N, "n_heads": H,
            "n_iters": args.n_iters,
        },
        "results": results,
    }, indent=2, default=float))
    print(f"\n[done] → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
