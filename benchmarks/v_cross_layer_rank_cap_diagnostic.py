#!/usr/bin/env python3
"""Step 1: V cross-layer rank cap diagnostic.

Does cross-layer V pool + rank cap = D/2 give enough MSE fidelity to
potentially land in ACCEPT / MARGINAL PPL regime?

Compared scenarios (all at rank_cap = D/2 = 64 on DS D=128):

  A) per-layer pool + rank cap = 64         (Sprint 3.1 equivalent, known REJECT)
  B) cross-layer pool + rank cap = 64        (proposed Scenario 3 block 3b)
  C) per-layer pool + NO rank cap (vr=1.0)  (Sprint 3.5 baseline, known ACCEPT)

Report MSE inflation ratios:
  B/C: how much worse is cross-layer + rank cap vs Sprint 3.5 baseline
  B/A: does cross-layer actually improve over per-layer at same rank cap
  A/C: how bad was the known REJECT (per-layer rank cap)

The decision rule:
  - If B/C is < 1.5x median:  Scenario 3 + b=2 is likely viable (rank cap OK under
                              cross-layer statistics)
  - If B/C is 1.5-3x median:  MARGINAL at best, push through to full PPL test
  - If B/C is > 3x median:    Abandon rank cap path, keep V at full rank in Scenario 3
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transformers import AutoModelForCausalLM, AutoTokenizer
import benchmarks.pre_rope_cache as prc
from benchmarks.cross_layer_v_pool_diagnostic import fit_pca, reconstruct_mse, collect_v_tensors


def run(model_path: str, n_passages: int = 2, ctx_len: int = 2048,
        skip_layers: list[int] | None = None, rank_cap: int | None = None):
    print(f"[{model_path}] collecting V tensors for diagnostic…", flush=True)
    v_per_layer, n_layers, layer_types = collect_v_tensors(model_path, n_passages, ctx_len)

    D = next(iter(v_per_layer.values())).shape[1]
    full_attn_layers = sorted(v_per_layer.keys())
    if skip_layers:
        skip_set = set(skip_layers)
        full_attn_layers = [l for l in full_attn_layers if l not in skip_set]
        print(f"  skip_layers={sorted(skip_set)}, pooling over {len(full_attn_layers)} interior layers")

    if rank_cap is None:
        rank_cap = D // 2  # match v1.3 tier-1 RSVD target_rank = D/2
    print(f"  D={D}, rank_cap = D/2 = {rank_cap} (matches RSVD target_rank)", flush=True)

    # ---- Scenario A: per-layer pool + rank cap ----
    per_layer_capped = {}
    for l in full_attn_layers:
        m, b, d = fit_pca(v_per_layer[l], variance_ratio=1.0, rank_cap=rank_cap)
        per_layer_capped[l] = (m, b, d)

    # ---- Scenario B: cross-layer pool + rank cap ----
    v_all = np.concatenate([v_per_layer[l] for l in full_attn_layers], axis=0)
    m_global_cap, b_global_cap, d_global_cap = fit_pca(
        v_all, variance_ratio=1.0, rank_cap=rank_cap,
    )
    print(f"  cross-layer basis d_eff = {d_global_cap}", flush=True)

    # ---- Scenario C: per-layer pool + full rank (Sprint 3.5 baseline) ----
    per_layer_full = {}
    for l in full_attn_layers:
        m, b, d = fit_pca(v_per_layer[l], variance_ratio=1.0, rank_cap=None)
        per_layer_full[l] = (m, b, d)

    # Measure per-layer MSEs under each scenario
    print(f"\n{'L':>4}  {'MSE_A (per+cap)':>17}  {'MSE_B (cross+cap)':>19}  "
          f"{'MSE_C (per+full)':>17}  {'B/A':>6}  {'B/C':>8}  {'A/C':>8}")
    print("-" * 100)
    rows = []
    for l in full_attn_layers:
        V_l = v_per_layer[l]
        mse_a = reconstruct_mse(V_l, per_layer_capped[l][0], per_layer_capped[l][1])
        mse_b = reconstruct_mse(V_l, m_global_cap, b_global_cap)
        mse_c = reconstruct_mse(V_l, per_layer_full[l][0], per_layer_full[l][1])
        ba = mse_b / max(mse_a, 1e-20)
        bc = mse_b / max(mse_c, 1e-20)
        ac = mse_a / max(mse_c, 1e-20)
        rows.append({"layer": l, "mse_a": mse_a, "mse_b": mse_b, "mse_c": mse_c,
                     "B_over_A": ba, "B_over_C": bc, "A_over_C": ac})
        print(f"  {l:>2}  {mse_a:>15.4e}  {mse_b:>17.4e}  {mse_c:>15.4e}  "
              f"{ba:>5.2f}x  {bc:>6.1f}x  {ac:>6.1f}x")

    print(f"\n{'='*100}\nSummary ({len(rows)} layers):")
    ba_arr = np.array([r["B_over_A"] for r in rows])
    bc_arr = np.array([r["B_over_C"] for r in rows])
    ac_arr = np.array([r["A_over_C"] for r in rows])
    for name, arr in [("B/A (cross-cap vs per-cap — does pooling help at same cap?)", ba_arr),
                      ("B/C (cross-cap vs Sprint 3.5 full-rank — how much worse?)", bc_arr),
                      ("A/C (per-cap vs Sprint 3.5 full-rank — known REJECT baseline)", ac_arr)]:
        print(f"\n  {name}")
        print(f"    min={arr.min():6.2f}×  median={np.median(arr):6.2f}×  "
              f"mean={arr.mean():6.2f}×  p90={np.percentile(arr, 90):6.2f}×  max={arr.max():6.2f}×")

    # Predicted PPL using corr(log Δppl, log V-MSE) = 0.55
    # Sprint 3.5 baseline Δppl = -0.0168
    base_log = np.log(1 - 0.0168)
    print(f"\nPredicted Δppl (assuming corr log-log = 0.55):")
    for name, arr, label in [("Scenario A (per-cap, known REJECT)", ac_arr, "A"),
                             ("Scenario B (cross-cap, PROPOSED)", bc_arr, "B")]:
        med = float(np.median(arr))
        predicted = float(np.exp(base_log + 0.55 * np.log(med)) - 1)
        print(f"  {name}: B/C median inflation {med:.2f}× → predicted Δppl ≈ {predicted*100:+.2f}%")

    print(f"\nVERDICT for Scenario B (cross-layer + rank cap D/2):")
    bc_med = float(np.median(bc_arr))
    if bc_med < 1.5:
        print(f"  B/C median {bc_med:.2f}× < 1.5 → LIKELY VIABLE for MARGINAL. Proceed to steps 2-5.")
    elif bc_med < 3.0:
        print(f"  B/C median {bc_med:.2f}× in 1.5-3.0 → BORDERLINE MARGINAL. Proceed with caution.")
    else:
        print(f"  B/C median {bc_med:.2f}× >= 3.0 → REJECT likely. Abandon Scenario 3 b_V rank cap.")

    return {
        "model_path": model_path,
        "D": D, "rank_cap": rank_cap,
        "n_full_attention_layers": len(rows),
        "skip_layers": list(skip_layers or []),
        "rows": rows,
        "summary": {
            "B_over_A": {"median": float(np.median(ba_arr)), "max": float(ba_arr.max())},
            "B_over_C": {"median": float(np.median(bc_arr)), "max": float(bc_arr.max())},
            "A_over_C": {"median": float(np.median(ac_arr)), "max": float(ac_arr.max())},
        },
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--n-passages", type=int, default=2)
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--skip-layers", type=int, nargs="+", default=[0, 1, 26, 27])
    ap.add_argument("--rank-cap", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = run(args.model_path, args.n_passages, args.ctx_len,
              args.skip_layers, args.rank_cap)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
