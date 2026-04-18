#!/usr/bin/env python3
"""V-channel byte-exact extrapolation for kakeyaturbo v1.2 (bit_width=2)
from ctx=4096 measured data to arbitrary longer contexts.

V-channel has two cost terms:
  1. shared_pca_bytes (one-shot, ctx-INVARIANT)
  2. per-block skeleton (K-means) + codes (linear in ctx)

So V-channel ratio grows sub-linearly toward an asymptote as ctx -> inf:
  ratio_V(C) = (bf16_per_vec * n_vecs(C)) / (shared_pca + per_vec * n_vecs(C))
             -> bf16_per_vec / per_vec    (asymptote when shared_pca << per_vec*n_vecs)

We also report K-channel for contrast (K has NO shared term in v1.2,
so K ratio is essentially ctx-independent).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def extrapolate(L: dict, ctx_target: int, ctx_meas: int):
    if L["layer_type"] != "full_attention":
        # sliding layer: passthrough, stays at bf16 regardless
        return None
    scale = ctx_target / ctx_meas
    kr = L["k_report"]
    vr = L["v_report"]

    # K side: all linear
    k_comp = kr["compressed_bytes"] * scale
    k_base = kr["baseline_bytes_bf16"] * scale

    # V side: shared_pca one-shot, rest linear
    shared = vr.get("shared_pca_bytes", 0)
    other_v = vr["compressed_bytes"] - shared
    v_comp = shared + other_v * scale
    v_base = vr["baseline_bytes_bf16"] * scale

    return {
        "k_comp": k_comp, "k_base": k_base,
        "v_comp": v_comp, "v_base": v_base,
        "v_shared": shared, "v_linear_per_vec": other_v / kr["num_vecs_encoded"],
        "v_base_per_vec": vr["baseline_bytes_bf16"] / vr["num_vecs_encoded"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", type=Path,
                    default=Path("reports/real_kakeyaturbo_bit_width_sweep"))
    ap.add_argument("--bw", type=int, default=2)
    ap.add_argument("--ctx-meas", type=int, default=4096)
    ap.add_argument("--ctx-list", type=str, default="4096,8192,16384,32768,65536,131072")
    args = ap.parse_args()

    ctx_list = [int(x) for x in args.ctx_list.split(",")]
    models = [
        "qwen2_5_0_5b",
        "qwen3_0_6b",
        "gemma4_e2b",
        "deepseek_r1_distill_qwen_1_5b",
        "glm_edge_1_5b",
        "smollm2_1_7b",
        "glm_edge_4b",
    ]

    print(f"\nV-channel (bit_width={args.bw}) — full-attn layers only, extrapolated\n")
    hdr = f"{'model':40s}" + "".join(f"{c:>9}" for c in ctx_list) + f"{'asymp':>9}{'+% @128k':>10}"
    print(hdr)
    print("-" * len(hdr))

    v_table = {}
    for m in models:
        p = args.sweep_root / f"bw{args.bw}" / m / f"ctx_{args.ctx_meas}" / "summary.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())

        row = []
        for ctx in ctx_list:
            v_comp = 0.0
            v_base = 0.0
            for L in d["per_layer"]:
                r = extrapolate(L, ctx, args.ctx_meas)
                if r is None:
                    continue
                v_comp += r["v_comp"]
                v_base += r["v_base"]
            row.append(v_base / v_comp if v_comp else 0)

        # Compute asymptote: ratio at ctx -> inf, i.e., ignore shared_pca term
        asymp_base = 0.0
        asymp_comp = 0.0
        for L in d["per_layer"]:
            r = extrapolate(L, args.ctx_meas, args.ctx_meas)
            if r is None:
                continue
            asymp_base += r["v_base_per_vec"]
            asymp_comp += r["v_linear_per_vec"]
        asymptote = asymp_base / asymp_comp if asymp_comp else 0

        # % uplift from base ctx to 128k
        uplift = (row[-1] / row[0] - 1) * 100 if row and row[0] else 0

        cells = "".join(f"{r:>8.2f}x" for r in row)
        print(f"{m:40s}{cells}{asymptote:>8.2f}x{uplift:>9.1f}%")
        v_table[m] = row + [asymptote]

    # Also show K channel for contrast
    print(f"\nK-channel (bit_width={args.bw}) for comparison — all linear, no shared term\n")
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        p = args.sweep_root / f"bw{args.bw}" / m / f"ctx_{args.ctx_meas}" / "summary.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        row = []
        for ctx in ctx_list:
            k_comp = 0.0
            k_base = 0.0
            for L in d["per_layer"]:
                r = extrapolate(L, ctx, args.ctx_meas)
                if r is None:
                    continue
                k_comp += r["k_comp"]
                k_base += r["k_base"]
            row.append(k_base / k_comp if k_comp else 0)
        uplift = (row[-1] / row[0] - 1) * 100 if row and row[0] else 0
        cells = "".join(f"{r:>8.2f}x" for r in row)
        print(f"{m:40s}{cells}{'(linear)':>9}{uplift:>9.1f}%")


if __name__ == "__main__":
    main()
