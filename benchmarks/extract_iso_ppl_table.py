"""Pull the iso-PPL best CR for each model, for thresholds 0.5 / 1.0 / 2.0.

Exactly matches the methodology used to produce
reports/v1_4_release/kv_128k_isoppl_n8/V14_VS_TQ_ISOPPL_REPORT.md:

  For each channel, treat aggregate `mean_abs_delta_ppl` (in raw ppl units,
  averaged across n=8 passages) as the |Δppl| metric. Winner = channel with
  the highest total_ratio_128k whose mean_abs_delta_ppl <= threshold.

This is the same convention used throughout the published reports; we carry
it forward unchanged so README numbers reconcile 1:1 with the report PDF.

Output: a markdown table fragment (README hero).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "reports" / "v1_4_release" / "kv_128k_isoppl_n8"

MODELS = [
    ("qwen3_4b", "Qwen3-4B"),
    ("glm4_9b", "GLM-4-9B-Chat"),
    ("gemma4_e4b", "Gemma-4-E4B"),
    ("deepseek_1p5b", "DeepSeek-R1-Distill-Qwen-1.5B"),
]
THRESHOLDS = [0.005, 0.010, 0.020]


def _best_cr(aggs, kind, threshold):
    best = None
    for a in aggs:
        if a["kind"] != kind:
            continue
        if abs(a["mean_abs_delta_ppl"]) > threshold:
            continue
        if best is None or a["total_ratio_128k"] > best["total_ratio_128k"]:
            best = a
    return best


def main() -> int:
    print("| Model | Target \\|Δppl\\| | KakeyaLattice CR | (Δppl) | TurboQuant CR | (Δppl) | KL advantage |")
    print("|:---|:---|---:|---:|---:|---:|---:|")
    for fname, pretty in MODELS:
        blob = json.loads((DATA_DIR / f"{fname}_kv_128k.json").read_text())
        for t in THRESHOLDS:
            kk = _best_cr(blob["aggregates"], "v14_kv", t)
            tq = _best_cr(blob["aggregates"], "tq_kv", t)
            kk_s = f"{kk['total_ratio_128k']:.2f}×" if kk else "oor"
            kk_d = f"{kk['mean_abs_delta_ppl']*100:.2f}%" if kk else "—"
            tq_s = f"{tq['total_ratio_128k']:.2f}×" if tq else "oor"
            tq_d = f"{tq['mean_abs_delta_ppl']*100:.2f}%" if tq else "—"
            if kk and tq:
                adv = (kk["total_ratio_128k"] / tq["total_ratio_128k"] - 1) * 100
                if abs(adv) < 0.5:
                    adv_s = "tied"
                elif adv > 0:
                    adv_s = f"**+{adv:.1f}%**"
                else:
                    adv_s = f"{adv:.1f}%"
            elif kk and not tq:
                adv_s = "**KL only**"
            else:
                adv_s = "—"
            print(f"| {pretty} | ≤ {t*100:.1f}% | {kk_s} | {kk_d} | {tq_s} | {tq_d} | {adv_s} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
