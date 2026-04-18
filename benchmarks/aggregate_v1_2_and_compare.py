#!/usr/bin/env python3
"""Aggregate v1.2 results and write a v1-vs-v1.2 comparison table."""

import json
from pathlib import Path

V1_ROOT = Path("/workspace/reports/real_kakeyaturbo/full")
V12_ROOT = Path("/workspace/reports/real_kakeyaturbo_v1_2/full")
OUT_MD = Path("/workspace/reports/real_kakeyaturbo_v1_2/V1_vs_V1_2_COMPARE.md")
OUT_JSON = Path("/workspace/reports/real_kakeyaturbo_v1_2/V1_vs_V1_2_COMPARE.json")

# Match the image ordering (highest Total ratio first).
ORDER = [
    ("qwen3_0_6b", "Qwen/Qwen3-0.6B"),
    ("gemma4_e2b", "google/gemma-4-E2B-it"),
    ("glm_edge_4b", "THUDM/glm-edge-4b-chat"),
    ("glm_edge_1_5b", "THUDM/glm-edge-1.5b-chat"),
    ("deepseek_r1_distill_qwen_1_5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("smollm2_1_7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("qwen2_5_0_5b", "Qwen/Qwen2.5-0.5B-Instruct"),
]
CTX = [2048, 4096, 8192]


def load_ratio(root: Path, short: str, ctx: int) -> dict | None:
    p = root / short / f"ctx_{ctx}" / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    t = s["totals"]
    # Pull mean K/V MSE across full-attn layers.
    k_mse = v_mse = None
    ks, vs = [], []
    for pl in s["per_layer"]:
        if pl.get("is_sliding") or not pl.get("k_report"):
            continue
        km = pl["k_report"]["mean_block_mse"]
        vm = pl["v_report"]["mean_block_mse"]
        if km >= 0: ks.append(km)
        if vm >= 0: vs.append(vm)
    if ks: k_mse = sum(ks) / len(ks)
    if vs: v_mse = sum(vs) / len(vs)
    return {
        "total_ratio_bf16": t["ratio_bf16"],
        "full_attn_ratio_bf16": t["full_attn_ratio_bf16"],
        "baseline_bf16_bytes": t["baseline_bf16_bytes"],
        "compressed_bytes": t["compressed_bytes"],
        "mean_k_mse": k_mse,
        "mean_v_mse": v_mse,
    }


def main():
    rows = []
    for short, full in ORDER:
        entry = {"short": short, "full_name": full, "per_ctx": {}}
        for ctx in CTX:
            v1 = load_ratio(V1_ROOT, short, ctx)
            v12 = load_ratio(V12_ROOT, short, ctx)
            if v1 is None or v12 is None:
                continue
            entry["per_ctx"][str(ctx)] = {
                "v1": v1,
                "v1_2": v12,
                "speedup": v12["total_ratio_bf16"] / v1["total_ratio_bf16"] if v1["total_ratio_bf16"] else None,
            }
        rows.append(entry)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2))

    lines = []
    lines.append("# kakeyaturbo v1.2 (A + B') vs v1.0 Real Benchmark Comparison")
    lines.append("")
    lines.append("v1.2 changes:")
    lines.append("- **A**: skeleton tensors (PCA mean / basis / K-means centres) stored as bf16 instead of f32")
    lines.append("- **B'**: V stream uses a **layer-pooled PCA basis** (one fit per layer, reused across all blocks);")
    lines.append("  K stream keeps per-block PCA (required to preserve reconstruction on RoPE-driven K distributions)")
    lines.append("")
    lines.append("Both were decided by the PCA basis-sharing ablation in PR #6: V inflation is 1.03–1.30×, K inflation would be 2–12×.")
    lines.append("")
    lines.append("**Codec params identical between v1 and v1.2**: block_size=512, variance_ratio=0.95, K=16, bit_width=3, seed=3405691582.")
    lines.append("All numbers are real end-to-end measurements via the release-built Rust binary on the same HF weights and prompt.")
    lines.append("")

    lines.append("## Total bf16 compression ratio — v1 → v1.2")
    lines.append("")
    lines.append("| Model | 2 048 (v1) | 2 048 (v1.2) | Δ | 4 096 (v1) | 4 096 (v1.2) | Δ | 8 192 (v1) | 8 192 (v1.2) | Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        cells = []
        for ctx in CTX:
            ent = r["per_ctx"].get(str(ctx))
            if ent:
                v1r = ent["v1"]["total_ratio_bf16"]
                v12r = ent["v1_2"]["total_ratio_bf16"]
                d = (v12r / v1r - 1) * 100
                cells += [f"{v1r:.3f}x", f"**{v12r:.3f}x**", f"{d:+.1f}%"]
            else:
                cells += ["—", "—", "—"]
        lines.append(f"| `{r['full_name']}` | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Full-attention-only ratio (strips sliding layers from Gemma 4)")
    lines.append("")
    lines.append("| Model | 2 048 (v1 → v1.2) | 4 096 (v1 → v1.2) | 8 192 (v1 → v1.2) |")
    lines.append("|---|---:|---:|---:|")
    for r in rows:
        cells = []
        for ctx in CTX:
            ent = r["per_ctx"].get(str(ctx))
            if ent and ent["v1"]["full_attn_ratio_bf16"] and ent["v1_2"]["full_attn_ratio_bf16"]:
                a = ent["v1"]["full_attn_ratio_bf16"]
                b = ent["v1_2"]["full_attn_ratio_bf16"]
                cells.append(f"{a:.3f}x → **{b:.3f}x** ({(b/a-1)*100:+.0f}%)")
            else:
                cells.append("—")
        lines.append(f"| `{r['full_name']}` | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Reconstruction quality change")
    lines.append("")
    lines.append("Per-layer mean K and V MSE at 4 k context. K should be unchanged (same algorithm). V should land within")
    lines.append("1.0–1.3× of v1 per the ablation prediction.")
    lines.append("")
    lines.append("| Model | K MSE v1 | K MSE v1.2 | V MSE v1 | V MSE v1.2 | V inflation |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        ent = r["per_ctx"].get("4096")
        if not ent: continue
        k1 = ent["v1"]["mean_k_mse"]; k2 = ent["v1_2"]["mean_k_mse"]
        v1 = ent["v1"]["mean_v_mse"];  v2 = ent["v1_2"]["mean_v_mse"]
        v_inf = (v2 / v1) if v1 else None
        lines.append(
            f"| `{r['full_name']}` | {k1:.3e} | {k2:.3e} | {v1:.3e} | {v2:.3e} | "
            + (f"{v_inf:.3f}×" if v_inf else "—") + " |"
        )

    lines.append("")
    lines.append("## Byte savings @ 8 192 tokens")
    lines.append("")
    lines.append("| Model | v1 compressed | v1.2 compressed | Extra saved |")
    lines.append("|---|---:|---:|---:|")
    for r in rows:
        ent = r["per_ctx"].get("8192")
        if not ent: continue
        b1 = ent["v1"]["compressed_bytes"]
        b2 = ent["v1_2"]["compressed_bytes"]
        def h(n):
            return f"{n/1024/1024:,.2f} MiB" if n < 1024**3 else f"{n/1024**3:,.2f} GiB"
        lines.append(f"| `{r['full_name']}` | {h(b1)} | {h(b2)} | {h(b1-b2)} |")

    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
