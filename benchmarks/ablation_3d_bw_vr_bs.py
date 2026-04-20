#!/usr/bin/env python3
"""3-D codec ablation on the pre-RoPE cache: bit_width × variance_ratio × block_size.

Fixed axes (from the 2×2×2 result, the dominant cell):
  - pca_method       = exact
  - skeleton_dtype   = fp16
  - share_basis      = per_block (i.e. one PCA fit per block)

Swept axes:
  - bit_width       ∈ {2, 3, 4}
  - variance_ratio  ∈ {0.995, 0.999, 1.000}
  - block_size      ∈ {128, 256, 512}

Everything else (ctx_len, n_eval, n_passages, rotation seed, K) is held.
One model load for all 27 cells.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transformers import AutoModelForCausalLM, AutoTokenizer

import benchmarks.pre_rope_cache as prc
from benchmarks.e2e_ppl_pre_rope import (
    load_wikitext_passages, prefill_cache, roundtrip_cache,
    logits_with_cache, compare_logits,
)


def run_one(model, tok, passages, *, ctx_len, n_eval, block_size, bit_width,
            variance_ratio, prefill_chunk, compress):
    per_passage = []
    for p in passages:
        ids = tok(p, return_tensors="pt")["input_ids"]
        if ids.shape[-1] < ctx_len + n_eval:
            continue
        prefix = ids[:, :ctx_len]
        cont = ids[:, ctx_len:ctx_len + n_eval]
        cache_ref = prefill_cache(model, prefix, prefill_chunk)
        cache_alt, stats = roundtrip_cache(
            model, cache_ref,
            block_size=block_size, bit_width=bit_width,
            pca_method="exact", variance_ratio=variance_ratio,
            compress=compress,
            skeleton_dtype="fp16",
            share_basis_k=False, share_basis_v=False,
        )
        c_ref = copy.deepcopy(cache_ref)
        c_alt = copy.deepcopy(cache_alt)
        lo_ref = logits_with_cache(model, c_ref, cont)
        lo_alt = logits_with_cache(model, c_alt, cont)
        per_passage.append({"metrics": compare_logits(lo_ref, lo_alt, cont),
                            "stats": stats})
    if not per_passage:
        return None
    md = float(np.mean([r["metrics"]["ppl_delta_rel"] for r in per_passage]))
    mk = float(np.mean([r["metrics"]["mean_kl"] for r in per_passage]))
    mt = float(np.mean([r["metrics"]["top1_agreement"] for r in per_passage]))
    return {"ppl_delta_mean": md, "kl_mean": mk, "top1_mean": mt,
            "per_passage": per_passage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--ctx-len", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--prefill-chunk", type=int, default=0)
    ap.add_argument("--n-passages", type=int, default=2)
    ap.add_argument("--compress", choices=["kv", "k_only", "v_only"], default="kv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bit-widths", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--variance-ratios", type=float, nargs="+",
                    default=[0.995, 0.999, 1.0])
    ap.add_argument("--block-sizes", type=int, nargs="+",
                    default=[128, 256, 512])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.model_name}] loading model…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()
    info = prc.install(model)
    print(f"  patched {info['patched_layers']} attention layers", flush=True)

    passages = load_wikitext_passages(tok, args.ctx_len + args.n_eval, args.n_passages)
    print(f"  got {len(passages)} WikiText passages", flush=True)

    grid = list(product(args.bit_widths, args.variance_ratios, args.block_sizes))
    print(f"  running {len(grid)} configurations "
          f"(exact PCA, per_block, fp16 skeleton, compress={args.compress})", flush=True)

    rows = []
    t_start = time.time()
    for i, (bw, vr, bs) in enumerate(grid):
        label = f"b={bw} vr={vr} bs={bs}"
        t0 = time.time()
        res = run_one(
            model, tok, passages,
            ctx_len=args.ctx_len, n_eval=args.n_eval,
            block_size=bs, bit_width=bw, variance_ratio=vr,
            prefill_chunk=args.prefill_chunk, compress=args.compress,
        )
        dur = time.time() - t0
        if res is None:
            print(f"  [{i+1}/{len(grid)}] {label:<28}  SKIP ({dur:.1f}s)", flush=True)
            continue
        row = {"bit_width": bw, "variance_ratio": vr, "block_size": bs,
               "ppl_delta": res["ppl_delta_mean"],
               "kl": res["kl_mean"], "top1": res["top1_mean"],
               "seconds": dur}
        rows.append(row)
        print(f"  [{i+1}/{len(grid)}] {label:<28}  "
              f"Δppl={res['ppl_delta_mean']*100:+7.2f}%  "
              f"KL={res['kl_mean']:.4f}  top1={res['top1_mean']*100:.2f}%  "
              f"({dur:.1f}s)", flush=True)

        out_name = f"{args.model_name}_{args.compress}_b{bw}_vr{vr}_bs{bs}.json"
        (args.out_dir / out_name).write_text(json.dumps({
            "model_name": args.model_name,
            "compress": args.compress,
            "config": {"pca_method": "exact", "skeleton_dtype": "fp16",
                       "share_basis": False, "bit_width": bw,
                       "variance_ratio": vr, "block_size": bs},
            **res,
        }, indent=2))

    total = time.time() - t_start
    print(f"\n  total wall time: {total/60:.1f} min over {len(rows)} cells", flush=True)

    # Consolidated summary
    (args.out_dir / f"{args.model_name}_{args.compress}_3d_summary.json").write_text(
        json.dumps({
            "model_name": args.model_name,
            "compress": args.compress,
            "ctx_len": args.ctx_len, "n_eval": args.n_eval,
            "fixed": {"pca_method": "exact", "skeleton_dtype": "fp16",
                      "share_basis": False},
            "rows": rows,
        }, indent=2)
    )

    # Pretty table: one table per block_size, bit_width × variance_ratio grid
    if not rows:
        return
    print(f"\n\n============ 3-D ABLATION SUMMARY ({args.model_name}, compress={args.compress}) ============\n")
    bss = sorted({r["block_size"] for r in rows})
    vrs = sorted({r["variance_ratio"] for r in rows})
    bws = sorted({r["bit_width"] for r in rows})
    for bs in bss:
        print(f"  block_size = {bs}")
        header = " " * 8 + "  ".join(f"vr={v:<7}" for v in vrs)
        print("  " + " b \\ vr".ljust(8) + header)
        print("  " + "-" * (8 + 12 * len(vrs)))
        for bw in bws:
            cells = []
            for vr in vrs:
                cand = [r for r in rows if r["bit_width"] == bw
                        and r["variance_ratio"] == vr
                        and r["block_size"] == bs]
                if not cand:
                    cells.append("     -     ")
                    continue
                r = cand[0]
                cells.append(f"{r['ppl_delta']*100:+7.2f}%")
            print(f"   b={bw}  " + "    ".join(cells))
        print()


if __name__ == "__main__":
    main()
