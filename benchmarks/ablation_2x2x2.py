#!/usr/bin/env python3
"""2×2×2 codec ablation on the pre-RoPE cache architecture.

Three axes:
  - pca_method   : {exact, randomized}
  - skeleton_dtype: {fp16, fp32}
  - share_basis  : {per_block, layer_shared}   (applied to both K and V)

Fixed config:
  - bit_width = 3  (we established b=3→b=4 gains are small)
  - variance_ratio = 0.995
  - block_size = 512
  - rsvd_target_rank = D/2, oversample=8, power_iters=2 when pca_method==randomized

Isolates the skeleton-precision floor by holding every other parameter
fixed.  Reuses one loaded model across all 8 points for speed.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from itertools import product

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import benchmarks.pre_rope_cache as prc
from benchmarks.e2e_ppl_pre_rope import (
    load_wikitext_passages, prefill_cache, roundtrip_cache,
    logits_with_cache, compare_logits,
)


def run_one(model, tok, passages, *, ctx_len, n_eval, block_size, bit_width,
            prefill_chunk, pca_method, vr, skeleton_dtype, share_basis, compress):
    per_passage = []
    for p in passages:
        ids = tok(p, return_tensors="pt")["input_ids"]
        if ids.shape[-1] < ctx_len + n_eval:
            continue
        prefix = ids[:, :ctx_len]
        cont = ids[:, ctx_len:ctx_len + n_eval]

        cache_ref = prefill_cache(model, prefix, prefill_chunk)
        cache_alt, stats = roundtrip_cache(
            model, cache_ref, block_size=block_size, bit_width=bit_width,
            pca_method=pca_method, variance_ratio=vr, compress=compress,
            skeleton_dtype=skeleton_dtype,
            share_basis_k=share_basis, share_basis_v=share_basis,
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
    return {
        "ppl_delta_mean": md,
        "kl_mean": mk,
        "top1_mean": mt,
        "per_passage": per_passage,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--ctx-len", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--bit-width", type=int, default=3)
    ap.add_argument("--variance-ratio", type=float, default=0.995)
    ap.add_argument("--prefill-chunk", type=int, default=0)
    ap.add_argument("--n-passages", type=int, default=2)
    ap.add_argument("--compress", choices=["kv", "k_only", "v_only"], default="kv")
    ap.add_argument("--out-dir", type=Path, required=True)
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

    axes = {
        "pca_method":       ["exact", "randomized"],
        "skeleton_dtype":   ["fp16", "fp32"],
        "share_basis":      [False, True],
    }
    grid = list(product(axes["pca_method"], axes["skeleton_dtype"], axes["share_basis"]))
    print(f"  running {len(grid)} configurations (compress={args.compress})", flush=True)

    table_rows = []
    for i, (pca, sk, shared) in enumerate(grid):
        label = f"{pca}/{sk}/{'shared' if shared else 'per_block'}"
        print(f"\n  [{i+1}/{len(grid)}] {label}", flush=True)
        res = run_one(
            model, tok, passages,
            ctx_len=args.ctx_len, n_eval=args.n_eval,
            block_size=args.block_size, bit_width=args.bit_width,
            prefill_chunk=args.prefill_chunk,
            pca_method=pca, vr=args.variance_ratio,
            skeleton_dtype=sk, share_basis=shared,
            compress=args.compress,
        )
        if res is None:
            print("    SKIP (no passage long enough)"); continue
        row = {
            "pca_method": pca, "skeleton_dtype": sk, "share_basis": shared,
            "ppl_delta": res["ppl_delta_mean"],
            "kl": res["kl_mean"],
            "top1": res["top1_mean"],
        }
        table_rows.append(row)
        print(f"    Δppl={res['ppl_delta_mean']*100:+7.2f}%   "
              f"KL={res['kl_mean']:.4f}   top1={res['top1_mean']*100:.2f}%", flush=True)

        out_name = (f"{args.model_name}_{args.compress}_{pca}_{sk}"
                    f"_{'shared' if shared else 'perblock'}.json")
        (args.out_dir / out_name).write_text(json.dumps({
            "model_name": args.model_name,
            "compress": args.compress,
            "bit_width": args.bit_width,
            "variance_ratio": args.variance_ratio,
            "block_size": args.block_size,
            "ctx_len": args.ctx_len,
            "n_eval": args.n_eval,
            "config": {"pca_method": pca, "skeleton_dtype": sk,
                       "share_basis": shared},
            **res,
        }, indent=2))

    summary_path = args.out_dir / f"{args.model_name}_{args.compress}_ablation_summary.json"
    summary_path.write_text(json.dumps({
        "model_name": args.model_name,
        "compress": args.compress,
        "bit_width": args.bit_width,
        "variance_ratio": args.variance_ratio,
        "rows": table_rows,
    }, indent=2))

    # Pretty-print a summary table.
    if table_rows:
        print("\n\n================ 2×2×2 ABLATION SUMMARY ================")
        print(f"  model={args.model_name}  compress={args.compress}  b={args.bit_width}")
        print(f"  vr={args.variance_ratio}  ctx={args.ctx_len}  n_passages={len(passages)}")
        print()
        print(f"  {'pca':<11} {'skel':<5} {'share':<9} {'Δppl':>10} {'KL':>8} {'top1':>7}")
        print(f"  {'-'*11} {'-'*5} {'-'*9} {'-'*10} {'-'*8} {'-'*7}")
        for r in table_rows:
            share_str = "shared" if r["share_basis"] else "per_block"
            print(f"  {r['pca_method']:<11} {r['skeleton_dtype']:<5} {share_str:<9}"
                  f"   {r['ppl_delta']*100:+7.2f}% {r['kl']:8.4f} {r['top1']*100:6.2f}%")


if __name__ == "__main__":
    main()
