r"""Stage 0.75 — Q sweep for maximum usable compression on V4-Flash KV.

For each of V4-Flash's three KV streams (SWA layer 0, c4a-pool layer 2,
c128a-pool layer 3), sweep E8 Q across a wide range, run n=N_PASSAGES
passages per Q, and solve for the **maximum usable compression ratio**
under three progressively more permissive quality thresholds:

  - Threshold A  : E8 rel-MSE  ≤  FP8 rel-MSE           (no regression; paper-grade)
  - Threshold B  : E8 rel-MSE  ≤  1.05 · FP8 rel-MSE    (≤ +5 % MSE regression)
  - Threshold C  : E8 rel-MSE  ≤  1.20 · FP8 rel-MSE    (≤ +20 %, aggressive)

"Usable" = the lowest Q whose n=N_PASSAGES mean rel-MSE (+CI upper
bound) clears the threshold.  We report both the point-estimate answer
(mean only, single-run view) and the CI-conservative answer (use the
95 % CI upper bound so deployment does not regress on an unlucky batch).

CRs are computed vs both baselines:

  - CR_vs_bf16  =  8192 / bits_per_vec     (where 8192 = 512 · 16 bit bf16)
  - CR_vs_fp8   =  4224 / bits_per_vec     (where 4224 = 512·8 + 8·16 FP8 per-64)

Output
------
`reports/v1_5_release/dsv4_stage075/stage075_qsweep_n{N}.json` with
per-stream per-Q rel-MSE tuples (mean, std, CI95-hw, n) plus the solved
thresholds A/B/C per stream.

Running
-------
    python3 benchmarks/dsv4_stage075/run_stage075_qsweep.py \
        --host-model Qwen/Qwen2-0.5B \
        --seqlen 2048 --n-passages 8 \
        --hf-home /workspace/hf_home \
        --out reports/v1_5_release/dsv4_stage075/stage075_qsweep_n8.json

End-to-end on 2 × H200 with shards warmly cached: ~2 minutes for the
12-point sweep × n=8 = 96 codec runs + 24 FP8 baselines.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "dsv4_stage0_5"))
sys.path.insert(0, str(REPO / "benchmarks" / "dsv4_stage075"))

from dsv4_kv_generator import (  # type: ignore[import-not-found]
    DSV4Compressor, DSV4FlashArchConfig, DSV4MainKVProjection,
)
from dsv4_weight_loader import (  # type: ignore[import-not-found]
    inject_weights_into_compressor, inject_weights_into_main_kv,
    load_single_layer_weights, load_v4_shard_paths,
)
from run_dsv4_stage0_5 import (  # type: ignore[import-not-found]
    compute_rel_mse, fp8_baseline_roundtrip, non_gaussian_audit,
)
from run_stage075_n8 import (  # type: ignore[import-not-found]
    PASSAGES, build_projection_W, build_and_load_dsv4_blocks, run_trio,
    load_host_hidden_for_passage, _agg, _t95,
)

from kakeyalattice import V15KakeyaZamirE8GPU  # type: ignore


# Q sweep — 12 points covering aggressive → conservative.
# bits/vec at D=512 = 64 * ceil(8 * log2(2Q+1)) + 32.
DEFAULT_Q_VALUES: List[int] = [1, 2, 3, 4, 6, 8, 10, 14, 19, 24, 38, 76]


def e8_bits_per_vec(D: int, Q: int) -> int:
    """Same formula as in v1_5_kakeya_zamir_e8_gpu.py docstring."""
    per_block = math.ceil(8 * math.log2(2 * Q + 1))
    return (D // 8) * per_block + 32


def solve_max_cr_at_threshold(
    per_q_rel_mse: Dict[int, Dict[str, float]],
    fp8_rel_mse_mean: float,
    fp8_rel_mse_ci_hw: float,
    thr_multiplier: float,
    bits_by_q: Dict[int, int],
    bits_fp8: int,
    bits_bf16: int,
    use_ci_upper: bool,
) -> Dict:
    """Given {Q: {mean, ci95_hw, ...}} and FP8 stats, find the lowest Q
    whose E8 rel-MSE upper bound stays ≤ thr_multiplier · FP8 mean.
    If use_ci_upper, upper bound = mean + ci95_hw (conservative);
    otherwise upper bound = mean (point estimate).
    """
    budget = thr_multiplier * fp8_rel_mse_mean
    best: Tuple[int, float, float] | None = None   # (Q, bits, e8_mse_used)
    for Q in sorted(per_q_rel_mse.keys()):
        mu = per_q_rel_mse[Q]["mean"]
        hw = per_q_rel_mse[Q]["ci95_hw"]
        used = (mu + hw) if use_ci_upper else mu
        if used <= budget:
            best = (Q, bits_by_q[Q], used)
            break
    if best is None:
        return {
            "admissible": False,
            "threshold_multiplier": thr_multiplier,
            "budget_rel_mse": budget,
        }
    Q, bits, used = best
    return {
        "admissible": True,
        "threshold_multiplier": thr_multiplier,
        "budget_rel_mse": budget,
        "use_ci_upper": use_ci_upper,
        "Q_min": Q,
        "bits_per_vec": bits,
        "cr_vs_fp8": bits_fp8 / bits,
        "cr_vs_bf16": bits_bf16 / bits,
        "bit_saving_vs_fp8_pct": 100.0 * (1.0 - bits / bits_fp8),
        "bit_saving_vs_bf16_pct": 100.0 * (1.0 - bits / bits_bf16),
        "e8_rel_mse_used": used,
        "fp8_rel_mse_ref_mean": fp8_rel_mse_mean,
        "margin_pct": 100.0 * (budget - used) / budget,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host-model", default="Qwen/Qwen2-0.5B")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-passages", type=int, default=8)
    p.add_argument("--q-values", default=",".join(str(q) for q in DEFAULT_Q_VALUES))
    p.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/workspace/hf_home"))
    p.add_argument("--out", default="reports/v1_5_release/dsv4_stage075/stage075_qsweep_n8.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q_values = sorted({int(q) for q in args.q_values.split(",") if q.strip()})
    print(f"[config] host={args.host_model} seqlen={args.seqlen} batch={args.batch_size} "
          f"n_passages={args.n_passages} q_values={q_values} device={device}", flush=True)

    # 1. V4 shards
    shard_paths = load_v4_shard_paths(args.hf_home, "deepseek-ai/DeepSeek-V4-Flash")
    for needed in (2, 4, 5):
        if needed not in shard_paths:
            raise FileNotFoundError(f"Shard {needed} not found in {args.hf_home}")
    print(f"[shards] found {len(shard_paths)} V4 shards", flush=True)

    # 2. V4 blocks
    cfg = DSV4FlashArchConfig(simulate_fp8=True)
    t0 = time.perf_counter()
    blocks = build_and_load_dsv4_blocks(shard_paths, device=device, config=cfg)
    print(f"[load] V4 blocks loaded in {time.perf_counter()-t0:.2f}s", flush=True)

    # 3. Host model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.host_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.host_model, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    native_hidden = model.config.hidden_size
    W_proj = build_projection_W(native_hidden, cfg.hidden_size, device) \
        if native_hidden != cfg.hidden_size else None

    # 4. Codecs (one per Q)
    D = cfg.head_dim
    e8_codecs = {Q: V15KakeyaZamirE8GPU(D=D, q_range=Q, device=device) for Q in q_values}
    bits_by_q: Dict[int, int] = {Q: int(c.bits_per_token_per_head) for Q, c in e8_codecs.items()}
    bits_fp8 = D * 8 + (D // 64) * 16                   # 4224 at D=512 (per-64-block scale)
    bits_bf16 = D * 16                                   # 8192 at D=512
    for Q in q_values:
        print(f"[codec] E8 Q={Q:>3d}: bits/vec={bits_by_q[Q]:>4d}  "
              f"CR vs FP8={bits_fp8/bits_by_q[Q]:>5.2f}  "
              f"CR vs bf16={bits_bf16/bits_by_q[Q]:>5.2f}", flush=True)

    # 5. Iterate passages, collect per-(stream, Q) rel-MSE lists
    stream_names = ["sliding_window_kv", "csa_pool_kv_ratio4", "hca_pool_kv_ratio128"]
    rel_mse: Dict[str, Dict[int, List[float]]] = {s: {Q: [] for Q in q_values} for s in stream_names}
    fp8_mse: Dict[str, List[float]] = {s: [] for s in stream_names}
    audits: Dict[str, List[Dict]] = {s: [] for s in stream_names}

    for i in range(args.n_passages):
        print(f"\n[passage {i}/{args.n_passages}]", flush=True)
        tpp0 = time.perf_counter()
        hidden = load_host_hidden_for_passage(
            model, tok, PASSAGES[i], args.seqlen, args.batch_size,
            target_hidden_size=cfg.hidden_size, device=device, projection_W=W_proj,
        )
        streams = run_trio(blocks, hidden)
        for s in stream_names:
            kv = streams[s]
            audits[s].append(non_gaussian_audit(kv))
            # FP8 baseline once per passage per stream
            fp8_hat = fp8_baseline_roundtrip(kv)
            fp8_mse[s].append(compute_rel_mse(kv, fp8_hat))
            # E8 at each Q
            for Q in q_values:
                kv_hat = e8_codecs[Q].roundtrip(kv.float())
                if kv.is_cuda:
                    torch.cuda.synchronize()
                rel_mse[s][Q].append(compute_rel_mse(kv, kv_hat))
        tpp1 = time.perf_counter()
        print(f"  wall={tpp1-tpp0:.2f}s", flush=True)

    # 6. Aggregate
    agg_per_stream: Dict[str, Dict] = {}
    for s in stream_names:
        per_q = {Q: _agg(rel_mse[s][Q]) for Q in q_values}
        fp8_stats = _agg(fp8_mse[s])
        # Audit aggregate (per metric)
        audit_keys = list(audits[s][0].keys())
        audit_agg = {
            k: _agg([float(a[k]) for a in audits[s] if isinstance(a[k], (int, float))])
            for k in audit_keys
        }
        # Solve thresholds A / B / C at two views: point estimate AND CI-conservative
        thresholds = {}
        for name, mul in [("A_no_regression", 1.00),
                          ("B_plus5pct", 1.05),
                          ("C_plus20pct", 1.20)]:
            thresholds[f"{name}_point"] = solve_max_cr_at_threshold(
                per_q, fp8_stats["mean"], fp8_stats["ci95_hw"], mul,
                bits_by_q, bits_fp8, bits_bf16, use_ci_upper=False,
            )
            thresholds[f"{name}_ci95_conservative"] = solve_max_cr_at_threshold(
                per_q, fp8_stats["mean"], fp8_stats["ci95_hw"], mul,
                bits_by_q, bits_fp8, bits_bf16, use_ci_upper=True,
            )
        agg_per_stream[s] = {
            "fp8_rel_mse": fp8_stats,
            "e8_rel_mse_by_q": per_q,
            "audit": audit_agg,
            "thresholds": thresholds,
        }

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "host_model": args.host_model,
            "seqlen": args.seqlen,
            "batch_size": args.batch_size,
            "n_passages": args.n_passages,
            "q_values": q_values,
            "device": device,
            "head_dim": D,
            "bits_fp8_per64_baseline": bits_fp8,
            "bits_bf16_reference": bits_bf16,
            "dsv4_layers_used": {0: "SWA", 2: "c4a", 3: "c128a"},
            "threshold_definitions": {
                "A_no_regression":   "E8 rel-MSE ≤ 1.00 × FP8 rel-MSE (paper-grade, no quality regression)",
                "B_plus5pct":        "E8 rel-MSE ≤ 1.05 × FP8 rel-MSE (≤ +5 % MSE regression, deploy-cautious)",
                "C_plus20pct":       "E8 rel-MSE ≤ 1.20 × FP8 rel-MSE (≤ +20 % MSE, aggressive)",
                "_ci95_conservative_suffix": "adds CI95 half-width to E8 mean before comparison",
            },
        },
        "bits_per_vec_by_q": bits_by_q,
        "aggregate_by_stream": agg_per_stream,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[out] {out}", flush=True)

    # 7. Human-readable summary
    print("\n" + "=" * 100)
    print(f"MAX USABLE COMPRESSION — n={args.n_passages} passages, 95 % CI")
    print("=" * 100)
    for s in stream_names:
        entry = agg_per_stream[s]
        fp8 = entry["fp8_rel_mse"]
        print(f"\n[{s}]  FP8 baseline rel-MSE = {fp8['mean']:.3e} ± {fp8['ci95_hw']:.3e}")
        print(f"  {'Q':>4s} {'bits':>5s} {'CR_fp8':>7s} {'CR_bf16':>8s} {'E8 rel-MSE (mean±CI)':>30s} {'E8/FP8':>8s}")
        for Q in q_values:
            rm = entry["e8_rel_mse_by_q"][Q]
            ratio = rm["mean"] / fp8["mean"] if fp8["mean"] > 0 else float("nan")
            cr_fp8 = bits_fp8 / bits_by_q[Q]
            cr_bf16 = bits_bf16 / bits_by_q[Q]
            mark = ""
            if ratio <= 1.00:
                mark = "  [A]"
            elif ratio <= 1.05:
                mark = "  [B]"
            elif ratio <= 1.20:
                mark = "  [C]"
            print(f"  {Q:>4d} {bits_by_q[Q]:>5d} {cr_fp8:>7.3f} {cr_bf16:>8.3f}  "
                  f"{rm['mean']:>12.3e} ± {rm['ci95_hw']:>8.2e}  {ratio:>6.3f}x{mark}")

        print("  Thresholds (point estimate):")
        for tname in ("A_no_regression_point", "B_plus5pct_point", "C_plus20pct_point"):
            t = entry["thresholds"][tname]
            if t["admissible"]:
                print(f"    {tname:<30s}  Q>={t['Q_min']:>3d}  "
                      f"bits={t['bits_per_vec']}  CR vs FP8={t['cr_vs_fp8']:.2f}x  "
                      f"CR vs bf16={t['cr_vs_bf16']:.2f}x  saving vs FP8={t['bit_saving_vs_fp8_pct']:.1f}%")
            else:
                print(f"    {tname:<30s}  NOT ADMISSIBLE at any swept Q (need Q > {max(q_values)})")

        print("  Thresholds (CI95-conservative):")
        for tname in ("A_no_regression_ci95_conservative",
                      "B_plus5pct_ci95_conservative",
                      "C_plus20pct_ci95_conservative"):
            t = entry["thresholds"][tname]
            if t["admissible"]:
                print(f"    {tname:<34s}  Q>={t['Q_min']:>3d}  "
                      f"bits={t['bits_per_vec']}  CR vs FP8={t['cr_vs_fp8']:.2f}x  "
                      f"CR vs bf16={t['cr_vs_bf16']:.2f}x  saving vs FP8={t['bit_saving_vs_fp8_pct']:.1f}%")
            else:
                print(f"    {tname:<34s}  NOT ADMISSIBLE at any swept Q (need Q > {max(q_values)})")


if __name__ == "__main__":
    main()
