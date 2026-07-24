r"""Stage 0.5 synthetic driver — CPU-friendly smoke + frozen reference numbers.

Runs the full DSV4 pipeline on a synthetic Gaussian hidden-state input
(no HuggingFace download needed) and reports per-stream audit +
KakeyaLattice roundtrip + FP8 baseline rel-MSE.  Serves two purposes:

  1. Quick local confidence check — no network, no weights, no CUDA.
     Catches shape/unit/dtype bugs before shipping to vast.ai.

  2. Frozen-reference numbers for CI regression.  Because the host
     hidden states are synthetic with a fixed seed, the rel-MSE values
     this script reports on Sep 24 2026 can be asserted against in a
     future PR to catch codec regressions.

The numbers reported here are NOT a claim about V4-Flash's real KV
behaviour — synthetic Gaussian inputs flow through random-init weights
producing near-Gaussian KV streams.  The real host-model run
(run_dsv4_stage0_5.py on vast.ai) is where the non-Gaussian audit
values become meaningful.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from dsv4_kv_generator import DSV4FlashArchConfig, DSV4KVGenerator
from run_dsv4_stage0_5 import (
    compute_cosine,
    compute_rel_mse,
    fp8_baseline_roundtrip,
    non_gaussian_audit,
)

from kakeyalattice import V14KakeyaZamirLatticeGPU, V15KakeyaZamirE8GPU


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[synthetic] device={device}")

    # Fixed seed synthetic hidden states.
    torch.manual_seed(20260424)
    if device == "cuda":
        torch.cuda.manual_seed(20260424)
    B, S, H = 1, 2048, 4096
    hidden = torch.randn(B, S, H, device=device, dtype=torch.bfloat16)

    cfg = DSV4FlashArchConfig(simulate_fp8=True)
    gen = DSV4KVGenerator(config=cfg, device=device, seed=20260424)
    streams = gen(hidden)
    print(f"[streams] {streams.summary()}")

    codecs = [
        ("v14_d4_Q10", V14KakeyaZamirLatticeGPU(D=512, q_range=10, device=device)),
        ("v14_d4_Q38", V14KakeyaZamirLatticeGPU(D=512, q_range=38, device=device)),
        ("v15_e8_Q10", V15KakeyaZamirE8GPU(D=512, q_range=10, device=device)),
        ("v15_e8_Q38", V15KakeyaZamirE8GPU(D=512, q_range=38, device=device)),
    ]

    results = {}
    for stream_name, kv in [
        ("sliding_window_kv", streams.sliding_window_kv),
        ("csa_pool_kv_ratio4", streams.csa_pool_kv),
        ("hca_pool_kv_ratio128", streams.hca_pool_kv),
    ]:
        stream_out = {
            "shape": list(kv.shape),
            "audit": non_gaussian_audit(kv),
            "codecs": {},
        }
        fp8 = fp8_baseline_roundtrip(kv)
        stream_out["codecs"]["fp8_baseline"] = {
            "bits_per_vector": kv.shape[-1] * 8 + (kv.shape[-1] // 64) * 16,
            "rel_mse": compute_rel_mse(kv, fp8),
            "cos_sim": compute_cosine(kv, fp8),
        }
        for name, c in codecs:
            t0 = time.perf_counter()
            kv_hat = c.roundtrip(kv.float())
            if kv.is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            stream_out["codecs"][name] = {
                "bits_per_vector": int(c.bits_per_token_per_head),
                "rel_mse": compute_rel_mse(kv, kv_hat),
                "cos_sim": compute_cosine(kv, kv_hat),
                "wall_time_sec": t1 - t0,
            }
        results[stream_name] = stream_out

    out_path = Path(__file__).parent.parent.parent / "reports" / "v1_5_release" / "dsv4_stage0_5" / "dsv4_stage0_5_synthetic_reference.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "note": (
                "Synthetic Gaussian hidden-state input + random-init DSV4 weights. "
                "These numbers are a CI smoke reference, NOT a claim about V4-Flash "
                "real KV distribution.  Real host-model runs go through "
                "run_dsv4_stage0_5.py on vast.ai."
            ),
            "config": {
                "device": device,
                "seed": 20260424,
                "hidden_shape": [B, S, H],
                "dsv4_config": streams.config_summary,
            },
            "results": results,
        }, f, indent=2)
    print(f"[out] {out_path}")

    # Print table
    print()
    print(f"{'stream':25s}  {'codec':20s}  {'bits':>6s}  {'rel-MSE':>11s}  {'cos':>7s}")
    print("-" * 80)
    for stream_name, stream_out in results.items():
        for codec_name, c in stream_out["codecs"].items():
            print(f"{stream_name:25s}  {codec_name:20s}  {c['bits_per_vector']:6d}  "
                  f"{c['rel_mse']:11.4e}  {c['cos_sim']:7.4f}")

    # Audit summary
    print()
    print(f"{'stream':25s}  {'|kurt-3|':>9s}  {'iso-var':>8s}  {'had-var':>8s}  {'W2/σ':>7s}  {'N':>6s}")
    print("-" * 75)
    for stream_name, stream_out in results.items():
        a = stream_out["audit"]
        print(f"{stream_name:25s}  {a['excess_kurtosis_abs']:9.3f}  {a['isotropy_variance_ratio']:8.2f}  "
              f"{a['hadamard_post_variance_ratio']:8.2f}  {a['rms_wasserstein2_over_sigma_per_dim']:7.3f}  "
              f"{a['num_vectors']:6d}")


if __name__ == "__main__":
    main()
