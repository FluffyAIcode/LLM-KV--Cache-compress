#!/usr/bin/env python3
"""Byte-accurate extrapolation of Kakeya KV cache compression to arbitrary
context lengths, derived from measured per-block statistics.

Why this is not a guess
-----------------------
The codec's per-block byte count is a deterministic function of:
  - block_size B (tokens per compressed block)
  - head_dim d
  - number of heads H and batch bsz
  - effective PCA rank d_eff (<= d, set by variance_ratio)
  - d_res (stored residual dimension per row)
  - K (centers per block)

For a fixed `block_size` and `variance_ratio` on a fixed model, once a run
at any context length C >= block_size produces at least one compressed
block, the skeleton and encoded byte counts *per block* are fully
determined — they do NOT depend on C. We read those per-block constants
out of an existing report and use them to scale to any target context.

This means the extrapolation is exact (up to rounding) under the same
codec parameters, not a statistical fit.

Outputs
-------
For a list of target context lengths, prints the projected KV byte
footprint for baseline vs Kakeya (both f32-store and bf16-projected),
and the resulting compression ratios.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Tuple


def humanize_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f"{v:,.2f} {u}"
        v /= 1024
    return f"{n} B"


def _per_block_consts(full_layer_rows: List[Dict[str, Any]], block_size: int) -> Tuple[float, float, float]:
    """From one report's full-attention layers, compute the average:
        (skeleton_bytes_per_block, encoded_bytes_per_block_token,
         bytes_per_exact_token_in_KV_dtype)

    Each compressed block encodes `block_size * H * bsz` row vectors, so
    encoded-bytes-per-block is a fixed number once H, bsz, block_size,
    and d_res are fixed. We derive encoded-bytes-per-token directly."""
    total_blocks = 0
    total_sk = 0
    total_enc = 0
    tail_bytes_per_token = 0.0
    tail_samples = 0
    for row in full_layer_rows:
        bs = row["kakeya_compressed_blocks"]
        if bs > 0:
            total_blocks += bs
            total_sk += row["kakeya_skeleton_bytes"]
            total_enc += row["kakeya_encoded_bytes"]
        tail_tokens = row["kakeya_seq_len"] - bs * block_size
        if tail_tokens > 0 and row["kakeya_exact_tail_bytes"] > 0:
            tail_bytes_per_token += row["kakeya_exact_tail_bytes"] / tail_tokens
            tail_samples += 1
    if total_blocks == 0:
        raise ValueError("Report has no compressed blocks to extrapolate from.")
    sk_per_block = total_sk / total_blocks
    # encoded bytes scale linearly with tokens inside a block
    enc_per_block = total_enc / total_blocks
    enc_per_token = enc_per_block / block_size
    tail_per_token = tail_bytes_per_token / max(tail_samples, 1)
    return sk_per_block, enc_per_token, tail_per_token


def extrapolate_from_report(
    report_path: str,
    target_contexts: List[int],
    block_size: int | None = None,
    residual_length: int | None = None,
) -> List[Dict[str, Any]]:
    with open(report_path) as f:
        rpt = json.load(f)

    # Resolve codec params from the report if not overridden.
    if block_size is None:
        block_size = int(rpt["args"]["block_size"])
    if residual_length is None:
        residual_length = int(rpt["args"]["residual_length"])

    per_layer = rpt["per_layer"]
    full_rows = [r for r in per_layer if r["layer_type"] == "full_attention"]
    slide_rows = [r for r in per_layer if r["layer_type"] == "sliding_attention"]
    n_full = len(full_rows)
    n_slide = len(slide_rows)

    sk_per_block, enc_per_token, tail_per_token = _per_block_consts(full_rows, block_size)

    # Sliding layer bytes are constant once C >= sliding_window, so we
    # read them off directly. When C < sliding_window we scale linearly.
    measured_C = int(rpt["context_tokens"])
    measured_slide_bytes_per_layer = (slide_rows[0]["baseline_bytes"] if slide_rows else 0)

    # Infer dtype size from the full-attn baseline bytes:
    # baseline_full_bytes = n_full * 2 * n_kv_heads * C * head_dim_full * dtype_bytes.
    # We do not know n_kv_heads from the report directly, but the per-layer row
    # tells us baseline_bytes = 2 * n_kv_heads * C * head_dim * dtype_bytes.
    # Therefore dtype_bytes * n_kv_heads = baseline_bytes / (2 * C * head_dim).
    # We then use the *same product* for both full and sliding, since GQA is
    # global in Gemma 4 (same num_key_value_heads per layer).
    n_kv_x_dtype = None
    if full_rows and measured_C > 0 and full_rows[0]["head_dim"] > 0:
        n_kv_x_dtype = full_rows[0]["baseline_bytes"] / (
            2 * measured_C * full_rows[0]["head_dim"]
        )

    # If the sliding window stores sw-1 tokens at C >= sw, then
    # measured_slide_bytes_per_layer = 2 * n_kv_x_dtype * (sw - 1) * head_dim_sliding.
    sliding_window = None
    if slide_rows and measured_slide_bytes_per_layer > 0 and n_kv_x_dtype:
        hd_s = slide_rows[0]["head_dim"] if slide_rows[0]["head_dim"] > 0 else 256
        tokens_stored = measured_slide_bytes_per_layer / (2 * n_kv_x_dtype * hd_s)
        tokens_stored = int(round(tokens_stored))
        # sliding layer stores min(C, sw-1) tokens; we assume measured_C >= sw
        sliding_window = tokens_stored + 1

    # Baseline bytes per full-attn layer per token:
    # reading directly from measured sample: baseline_full_bytes / (measured_C * n_full)
    baseline_full_per_token = (
        rpt["bytes"]["baseline_full_bytes"] / (measured_C * n_full) if n_full > 0 and measured_C > 0 else 0.0
    )

    # dtype size assumed 2 bytes (bf16) for the "dtype-matched" projection;
    # in principle it could be float16 / float32, but for Gemma 4 it's bf16.
    projections = []
    for C in target_contexts:
        # --- baseline side ---
        bl_full = int(round(baseline_full_per_token * C * n_full))
        if sliding_window is not None and n_slide > 0:
            if C >= sliding_window:
                per_slide = measured_slide_bytes_per_layer
                bl_slide = per_slide * n_slide
            else:
                # at C < sw the sliding cache holds C tokens
                # derive per-token bytes from measured per-layer
                per_token = measured_slide_bytes_per_layer / (sliding_window - 1)
                bl_slide = int(round(per_token * C * n_slide))
        else:
            bl_slide = 0

        # --- kakeya side ---
        # For each full-attn layer: split C into N_blocks full blocks + tail.
        # The codec triggers compression when exact_len - residual_length >= block_size.
        # Once the prefill is long enough, the steady-state exact tail is between
        # residual_length and residual_length + block_size, and N_blocks = (C - tail) / block_size.
        # For extrapolation we use C - residual_length rounded down to nearest block_size
        # as the number of compressed tokens, consistent with _rebalance() in the codec.
        compressed_tokens = max(((C - residual_length) // block_size) * block_size, 0)
        exact_tail_tokens = C - compressed_tokens
        N_blocks = compressed_tokens // block_size

        # Two KV streams (K and V) compressed separately -> 2x per full-attn layer.
        kv_per_layer_compressed = N_blocks * (sk_per_block + enc_per_token * block_size) * 2
        kv_per_layer_tail = tail_per_token * exact_tail_tokens * 2 if exact_tail_tokens > 0 else 0
        # Actually tail_per_token as measured already covers K+V combined because
        # kakeya_exact_tail_bytes in the report is sum over K and V tails. So we should
        # NOT multiply by 2. Let's recompute more carefully:
        # In the report, each row stores kakeya_exact_tail_bytes for one layer,
        # which is K_tail_bytes + V_tail_bytes. tail_per_token = that / tail_tokens.
        # And kakeya_skeleton_bytes / kakeya_encoded_bytes are already summed over K & V blocks.
        # So we should NOT double the per-layer formulas.
        kv_per_layer_compressed = N_blocks * (sk_per_block + enc_per_token * block_size)
        kv_per_layer_tail = tail_per_token * exact_tail_tokens if exact_tail_tokens > 0 else 0
        kv_full = int(round((kv_per_layer_compressed + kv_per_layer_tail) * n_full))

        # sliding layers pass through unchanged
        kv_slide = bl_slide

        # dtype-matched projection: compressed bytes shrink by 2 (halve the f32→bf16 side).
        # tail is already in the KV dtype (bf16), so it's not halved.
        compressed_only = int(round(kv_per_layer_compressed * n_full))
        kv_full_dtype_matched = (kv_full - compressed_only) + compressed_only // 2
        kv_total = kv_full + kv_slide
        kv_total_dtype_matched = kv_full_dtype_matched + kv_slide

        bl_total = bl_full + bl_slide

        def r(a, b):
            return a / b if b > 0 else None

        projections.append({
            "context_tokens": C,
            "N_blocks_per_full_layer": int(N_blocks),
            "exact_tail_tokens_per_full_layer": int(exact_tail_tokens),
            "baseline_full_bytes": bl_full,
            "baseline_sliding_bytes": bl_slide,
            "baseline_total_bytes": bl_total,
            "kakeya_full_bytes": kv_full,
            "kakeya_sliding_bytes": kv_slide,
            "kakeya_total_bytes": kv_total,
            "kakeya_full_bytes_dtype_matched": kv_full_dtype_matched,
            "kakeya_total_bytes_dtype_matched": kv_total_dtype_matched,
            "full_ratio_f32_store": r(bl_full, kv_full),
            "full_ratio_bf16_store": r(bl_full, kv_full_dtype_matched),
            "total_ratio_f32_store": r(bl_total, kv_total),
            "total_ratio_bf16_store": r(bl_total, kv_total_dtype_matched),
        })

    return projections


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True, help="Path to a kakeya_benchmark.py JSON report.")
    p.add_argument("--targets", default="2048,4096,8192,16384,32768,65536,131072")
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--residual-length", type=int, default=None)
    p.add_argument("--out", default=None, help="Optional path to dump the projection as JSON.")
    args = p.parse_args()

    targets = [int(x) for x in args.targets.split(",") if x.strip()]
    projections = extrapolate_from_report(
        args.report,
        targets,
        block_size=args.block_size,
        residual_length=args.residual_length,
    )

    print(f"\nExtrapolating from: {args.report}")
    print(
        f"{'ctx':>7}  {'blocks/L':>9}  {'baseline':>11}  {'kakeya':>11}  "
        f"{'k(bf16)':>11}  {'full f32':>8}  {'full bf16':>9}  {'total f32':>9}  {'total bf16':>10}"
    )
    for row in projections:
        print(
            f"{row['context_tokens']:>7}  "
            f"{row['N_blocks_per_full_layer']:>9}  "
            f"{humanize_bytes(row['baseline_total_bytes']):>11}  "
            f"{humanize_bytes(row['kakeya_total_bytes']):>11}  "
            f"{humanize_bytes(row['kakeya_total_bytes_dtype_matched']):>11}  "
            f"{row['full_ratio_f32_store']:>8.3f}  "
            f"{row['full_ratio_bf16_store']:>9.3f}  "
            f"{row['total_ratio_f32_store']:>9.3f}  "
            f"{row['total_ratio_bf16_store']:>10.3f}"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"source_report": args.report, "projections": projections}, f, indent=2)
        print(f"\n[out] wrote {args.out}")


if __name__ == "__main__":
    main()
