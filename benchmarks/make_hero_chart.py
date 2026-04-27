"""Generate the README / blog / Space hero chart from REAL n=8 iso-ppl data.

Source of truth:
    reports/v1_4_release/kv_128k_isoppl_n8/{qwen3_4b,glm4_9b,gemma4_e4b,deepseek_1p5b}_kv_128k.json

Produces:
    assets/hero_pareto.png    — 4-model Pareto front: |Δppl| vs compression-ratio,
                                KakeyaLattice (D4 / v1.4) vs TurboQuant (b3-b8)

Everything in the chart is pulled directly from the JSON aggregates
(the same aggregates that drive the published n=8 iso-PPL report).
No synthetic points, no interpolation, no mocking.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "reports" / "v1_4_release" / "kv_128k_isoppl_n8"
OUT = REPO / "assets" / "hero_pareto.png"

MODELS = [
    ("qwen3_4b", "Qwen3-4B"),
    ("glm4_9b", "GLM-4-9B-Chat"),
    ("gemma4_e4b", "Gemma-4-E4B"),
    ("deepseek_1p5b", "DeepSeek-R1-Distill-Qwen-1.5B"),
]


def _abs_delta_ppl_pct(agg: dict[str, Any], ppl_ref: float) -> float:
    """|Δppl| as % of bf16 ppl — same metric used in V14_VS_TQ_ISOPPL_REPORT."""
    return abs(agg["mean_abs_delta_ppl"]) / ppl_ref * 100.0


def _bf16_ppl_mean(per_passage: list[dict[str, Any]]) -> float:
    refs = [r["ppl_ref"] for r in per_passage if r["kind"] == "bf16"]
    if not refs:
        raise RuntimeError("no bf16_pass rows found — unexpected data shape")
    return sum(refs) / len(refs)


def _load_model(name: str) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]:
    """Return (kakeyalattice_points, tq_points).

    Each point = (compression_ratio, abs_delta_ppl_pct, label).
    Label = "Q=<q_range>" for KakeyaLattice (D4) and "b=<b>" for TurboQuant.

    TurboQuant's b is derived from the raw k_bits = (D * b + 32) formula
    (D = head_dim, 32-bit qmax overhead). This matches exactly what the
    benchmark harness used to sweep TQ.
    """
    path = DATA_DIR / f"{name}_kv_128k.json"
    blob = json.loads(path.read_text())
    head_dim = blob["head_dim"]
    tq_b_values = blob.get("tq_b_values", [])
    tq_bits_to_b = {head_dim * b + 32: b for b in tq_b_values}

    ppl_ref = _bf16_ppl_mean(blob["per_passage"])
    kakeya: list[tuple[float, float, str]] = []
    tq: list[tuple[float, float, str]] = []
    for agg in blob["aggregates"]:
        if agg["kind"] == "bf16":
            continue
        cr = agg["total_ratio_128k"]
        dpp = _abs_delta_ppl_pct(agg, ppl_ref)
        bits = agg["k_bits"]
        if agg["kind"] == "v14_kv":
            q = agg.get("q_range")
            label = f"Q={q}" if q is not None else f"{bits}b"
            kakeya.append((cr, dpp, label))
        elif agg["kind"] == "tq_kv":
            b = tq_bits_to_b.get(bits)
            label = f"b={b}" if b is not None else f"{bits}b"
            tq.append((cr, dpp, label))
    kakeya.sort(key=lambda p: p[0])
    tq.sort(key=lambda p: p[0])
    return kakeya, tq


def main() -> int:
    if not DATA_DIR.exists():
        print(f"ERROR: data dir missing: {DATA_DIR}", file=sys.stderr)
        return 2

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=False, sharey=False)
    axes = axes.flatten()

    kakeya_color = "#1565c0"
    tq_color = "#e07b00"

    for ax, (fname, pretty) in zip(axes, MODELS):
        kk, tq = _load_model(fname)
        kk_cr = [p[0] for p in kk]
        kk_dp = [p[1] for p in kk]
        tq_cr = [p[0] for p in tq]
        tq_dp = [p[1] for p in tq]

        ax.plot(kk_cr, kk_dp, "-o", color=kakeya_color,
                label="KakeyaLattice (D4)", markersize=6, linewidth=1.6)
        ax.plot(tq_cr, tq_dp, "-s", color=tq_color,
                label="TurboQuant", markersize=6, linewidth=1.6, alpha=0.85)

        for cr, dp, label in kk:
            ax.annotate(label, xy=(cr, dp),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=kakeya_color, alpha=0.75)
        for cr, dp, label in tq:
            ax.annotate(label, xy=(cr, dp),
                        xytext=(4, -10), textcoords="offset points",
                        fontsize=7, color=tq_color, alpha=0.75)

        ax.axhline(1.0, color="#888", linestyle=":", linewidth=0.9)
        ax.axhline(2.0, color="#bbb", linestyle=":", linewidth=0.7)
        ax.set_title(pretty)
        ax.set_xlabel("128k KV compression ratio  (higher = more savings)")
        ax.set_ylabel("|Δppl|  (% of bf16 ppl, lower = better)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")

    fig.suptitle(
        "KakeyaLattice vs TurboQuant — iso-PPL Pareto front\n"
        "Real vLLM bf16 + FlashAttention on H200 · WikiText-103 · n=8 passages × 64 eval positions",
        fontsize=11.5, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
