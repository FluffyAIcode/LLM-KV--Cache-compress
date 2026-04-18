#!/usr/bin/env python3
"""Byte-exact extrapolation of kakeyaturbo v1.2 (bit_width=2) vs TurboQuant turbo3
from a measured ctx=4096 run to arbitrary longer contexts.

For each layer in the measured summary.json, we decompose compressed bytes
into ctx-scaling and ctx-invariant components:

  K full-attn:  codes(C) = codes(4096) * C/4096   (share_basis=false)
                skeleton(C) = skeleton(4096) * C/4096  (one PCA + kmeans per block)
                total_K(C) = total_K(4096) * C/4096
  V full-attn:  shared_pca_bytes is ONE-SHOT (fixed)
                other bytes = (total_V(4096) - shared_pca_bytes) * C/4096
                total_V(C) = shared_pca_bytes + (total_V(4096)-shared_pca_bytes) * C/4096
  sliding:      compressed_bytes = baseline_bf16  (passthrough, capped at sw-1 tokens)
                total(C) = total(4096) unchanged  (sliding cache capped)

Baseline bf16 at ctx C for compressible layer:
  base(C) = base(4096) * C/4096

For sliding layers, baseline at ctx C in naive (uncompressed) cache:
  base(C) = 2 * min(C, sw-1) * n_kv_heads * hd * bf16
But for "v1.2 baseline" we compare to the same cache the codec SEES --
which for sliding layers is capped at sw-1 tokens because the eviction
policy upstream does that. So baseline_bf16 for sliding stays capped too.

turbo3 per-vector bytes:  K: hd*3/8 + 4 norm,  V: hd*3/8
turbo3 scales linearly with the SAME number of vectors that v1.2 sees
(both codecs run on the same cache contents).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def turbo3_ratio(hd: int, side: str) -> float:
    bits = 3
    payload = bits * hd / 8
    if side == "K":
        payload += 4  # fp32 norm for K
    baseline = 2 * hd
    return baseline / payload


def extrapolate_layer(L: dict, ctx_target: int, ctx_meas: int) -> tuple[int, int, int, int, int]:
    """Returns (v12_bytes, baseline_bf16_bytes, turbo3_bytes, k_vecs_scaled, v_vecs_scaled)
    at the target ctx.  baseline_bf16 is what an uncompressed bf16 cache would hold
    for this layer at ctx_target (respecting sliding-window caps)."""
    scale = ctx_target / ctx_meas
    if L["layer_type"] != "full_attention":
        # sliding/passthrough: capped, doesn't grow
        return (
            L["compressed_bytes"],
            L["baseline_bf16"],
            L["compressed_bytes"],  # turbo3 would also passthrough OR match kakeyaturbo; treat as bf16 here
            0, 0,
        )
    kr = L["k_report"]
    vr = L["v_report"]

    # ----- v1.2 K bytes -----
    # Everything scales linearly with num_vecs (both skeleton_per_block * num_blocks and codes)
    k_bytes = kr["compressed_bytes"] * scale

    # ----- v1.2 V bytes -----
    # shared_pca is one-shot (doesn't scale), rest scales linearly
    shared = vr.get("shared_pca_bytes", 0)
    other_v = vr["compressed_bytes"] - shared
    v_bytes = shared + other_v * scale

    v12_bytes = int(round(k_bytes + v_bytes))

    # ----- baseline bf16 -----
    base_bytes = int(round(L["baseline_bf16"] * scale))

    # ----- turbo3 bytes -----
    hd_k = kr["dim"]
    hd_v = vr["dim"]
    k_vecs = int(round(kr["num_vecs_encoded"] * scale))
    v_vecs = int(round(vr["num_vecs_encoded"] * scale))
    turbo3_k = k_vecs * (hd_k * 3 / 8 + 4)
    turbo3_v = v_vecs * (hd_v * 3 / 8)
    turbo3_bytes = int(round(turbo3_k + turbo3_v))

    return v12_bytes, base_bytes, turbo3_bytes, k_vecs, v_vecs


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

    print(f"\nv1.2 (bit_width={args.bw}) vs turbo3 — extrapolated from ctx={args.ctx_meas}\n")
    print(f"{'model':40s}{'ctx':>8}{'v1.2 total':>13}{'v1.2 FA':>11}"
          f"{'turbo3 total':>14}{'turbo3 FA':>11}{'v1.2/turbo3':>14}")
    print("-" * 111)

    for m in models:
        p = args.sweep_root / f"bw{args.bw}" / m / f"ctx_{args.ctx_meas}" / "summary.json"
        if not p.exists():
            print(f"missing: {p}")
            continue
        d = json.loads(p.read_text())
        for ctx in ctx_list:
            tot_v12 = 0
            tot_base = 0
            tot_turbo3 = 0
            fa_v12 = 0
            fa_base = 0
            fa_turbo3 = 0
            slide_bytes = 0
            for L in d["per_layer"]:
                v12, base, t3, *_ = extrapolate_layer(L, ctx, args.ctx_meas)
                tot_v12 += v12
                tot_base += base
                if L["layer_type"] == "full_attention":
                    tot_turbo3 += t3
                    fa_v12 += v12
                    fa_base += base
                    fa_turbo3 += t3
                else:
                    # sliding: both v1.2 and turbo3 end up at bf16 passthrough
                    tot_turbo3 += v12
                    slide_bytes += v12
            r_v12 = tot_base / tot_v12 if tot_v12 else 0
            r_t3 = tot_base / tot_turbo3 if tot_turbo3 else 0
            r_v12_fa = fa_base / fa_v12 if fa_v12 else 0
            r_t3_fa = fa_base / fa_turbo3 if fa_turbo3 else 0
            print(f"{m:40s}{ctx:>8}{r_v12:>12.2f}x{r_v12_fa:>10.2f}x"
                  f"{r_t3:>13.2f}x{r_t3_fa:>10.2f}x"
                  f"{r_v12/r_t3:>13.3f}x")
        print()


if __name__ == "__main__":
    main()
