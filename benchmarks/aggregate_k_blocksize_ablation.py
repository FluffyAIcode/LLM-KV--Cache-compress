#!/usr/bin/env python3
"""Aggregate per-model K-side block_size ablation summaries into a
single Markdown decision report."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "reports" / "k_blocksize_ablation"

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
        hd = d["meta"]["head_dim"]
        nlay = d["num_layers_measured"]
        cells = {c["block_size"]: c for c in d["aggregate"]}
        rows.append({
            "model": m,
            "hd": hd,
            "nlay": nlay,
            "cells": cells,
        })

    # Markdown.
    md = []
    md.append("# K-stream block_size ablation — results\n")
    md.append("Runs the full v1.2 monomorphic MSE codec at block_size ∈ {512, 1024, 2048}")
    md.append("on every full-attention K stream of 7 open-source models (real HF")
    md.append("forward passes, bf16, eager attention, ctx=4096, same prompt as the")
    md.append("d_eff/outlier ablation).\n")
    md.append("Verdict thresholds (same as the deff/outlier ablation):")
    md.append("- ≤ 10% MSE inflation → **ACCEPT**")
    md.append("- 10–30% → **MARGINAL**")
    md.append("- > 30% → **REJECT**\n")

    md.append("## Per-model aggregate (mean inflation / byte ratio across all full-attn layers)\n")
    md.append("| model | hd | layers | bs=512 (baseline) | bs=1024 mse / bytes | bs=2048 mse / bytes | bs=1024 verdict | bs=2048 verdict |")
    md.append("|---|---:|---:|---|---|---|---|---|")
    for r in rows:
        c512 = r["cells"].get(512)
        c1024 = r["cells"].get(1024)
        c2048 = r["cells"].get(2048)
        if not (c512 and c1024 and c2048):
            continue
        md.append(
            f"| `{r['model']}` | {r['hd']} | {r['nlay']} | "
            f"mse={c512['mean_mse_across_layers']:.3e} | "
            f"{c1024['mean_inflation_vs_baseline']:.2f}× / {c1024['mean_bytes_ratio_vs_baseline']:.2f}× | "
            f"{c2048['mean_inflation_vs_baseline']:.2f}× / {c2048['mean_bytes_ratio_vs_baseline']:.2f}× | "
            f"**{verdict(c1024['mean_inflation_vs_baseline'])}** | "
            f"**{verdict(c2048['mean_inflation_vs_baseline'])}** |"
        )

    md.append("\n## Cross-model means\n")
    cross = {512: [], 1024: [], 2048: []}
    cross_bytes = {512: [], 1024: [], 2048: []}
    for r in rows:
        for bs in (512, 1024, 2048):
            c = r["cells"].get(bs)
            if c:
                cross[bs].append(c["mean_inflation_vs_baseline"])
                cross_bytes[bs].append(c["mean_bytes_ratio_vs_baseline"])
    def avg(v): return sum(v) / max(1, len(v))
    md.append("| block_size | mean MSE inflation | mean byte ratio | max inflation | verdict |")
    md.append("|---:|---:|---:|---:|---|")
    for bs in (512, 1024, 2048):
        ms = cross[bs]; bs_ratios = cross_bytes[bs]
        md.append(
            f"| {bs} | {avg(ms):.3f}× | {avg(bs_ratios):.3f}× | {max(ms):.3f}× | **{verdict(avg(ms))}** |"
        )

    md.append("\n## Per-model per-layer worst case (max inflation at bs=1024 and bs=2048)\n")
    md.append("| model | bs=1024 max layer inflation | bs=2048 max layer inflation |")
    md.append("|---|---:|---:|")
    for r in rows:
        c1024 = r["cells"].get(1024)
        c2048 = r["cells"].get(2048)
        if not (c1024 and c2048):
            continue
        md.append(
            f"| `{r['model']}` | {c1024['max_inflation_vs_baseline']:.2f}× | "
            f"{c2048['max_inflation_vs_baseline']:.2f}× |"
        )

    out = ROOT / "SUMMARY.md"
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
