#!/usr/bin/env python3
"""Aggregate per-model K-side cross-layer ablation summaries into a
single Markdown decision report."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "reports" / "k_crosslayer_ablation"

MODELS = [
    "qwen2_5_0_5b",
    "qwen3_0_6b",
    "gemma4_e2b",
    "deepseek_r1_distill_qwen_1_5b",
    "glm_edge_1_5b",
    "smollm2_1_7b",
    "glm_edge_4b",
]


def verdict(inflate: float) -> str:
    if inflate <= 1.10:
        return "ACCEPT"
    if inflate <= 1.30:
        return "MARGINAL"
    return "REJECT"


def main():
    rows = []
    for m in MODELS:
        p = ROOT / m / "summary.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows.append({
            "model": m,
            "hd": d["meta"]["head_dim"],
            "nlay": d["num_layers"],
            "d_eff_pool": d["per_type_pooled_d_eff"],
            "strat": d["strategies"],
        })

    md = []
    md.append("# K-stream cross-layer basis-sharing ablation — results\n")
    md.append("Runs the full v1.2 monomorphic MSE codec on every full-attention")
    md.append("K stream of 7 open-source models (real HF forward passes, bf16,")
    md.append("eager attention, ctx=4096, same prompt as the d_eff/outlier ablation)")
    md.append("under three basis-sharing strategies:")
    md.append("")
    md.append("- `per_block` — one PCA per block (v1.2 K default)")
    md.append("- `per_layer_pooled` — one PCA per layer (share_basis=true)")
    md.append("- `per_type_pooled` — one PCA for ALL full-attn layers of the model")
    md.append("")
    md.append("Thresholds (consistent with the d_eff/outlier ablation):")
    md.append("- ≤ 10% MSE inflation → **ACCEPT**")
    md.append("- 10–30% → **MARGINAL**")
    md.append("- > 30% → **REJECT**")
    md.append("")

    md.append("## Per-model results\n")
    md.append("| model | hd | layers | per_layer_pooled (MSE × / byte ×) | per_type_pooled (MSE × / byte ×) | pooled d_eff | per_layer verdict | per_type verdict |")
    md.append("|---|---:|---:|---|---|---:|---|---|")
    for r in rows:
        s2 = r["strat"]["per_layer_pooled"]
        s3 = r["strat"]["per_type_pooled"]
        md.append(
            f"| `{r['model']}` | {r['hd']} | {r['nlay']} | "
            f"{s2['mse_inflation_vs_per_block']:.2f}× / {s2['bytes_vs_per_block']:.2f}× | "
            f"{s3['mse_inflation_vs_per_block']:.2f}× / {s3['bytes_vs_per_block']:.2f}× | "
            f"{r['d_eff_pool']} | "
            f"**{verdict(s2['mse_inflation_vs_per_block'])}** | "
            f"**{verdict(s3['mse_inflation_vs_per_block'])}** |"
        )

    md.append("\n## Cross-model aggregates\n")
    md.append("| strategy | mean MSE inflation | median | max | mean byte ratio | verdict (mean) |")
    md.append("|---|---:|---:|---:|---:|---|")
    for strat_key in ("per_layer_pooled", "per_type_pooled"):
        inf = [r["strat"][strat_key]["mse_inflation_vs_per_block"] for r in rows]
        byr = [r["strat"][strat_key]["bytes_vs_per_block"] for r in rows]
        mean_inf = sum(inf) / len(inf)
        med_inf = sorted(inf)[len(inf) // 2]
        max_inf = max(inf)
        mean_byr = sum(byr) / len(byr)
        md.append(
            f"| `{strat_key}` | {mean_inf:.2f}× | {med_inf:.2f}× | {max_inf:.2f}× | {mean_byr:.2f}× | **{verdict(mean_inf)}** |"
        )

    out = ROOT / "SUMMARY.md"
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
