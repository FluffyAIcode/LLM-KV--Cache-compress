#!/usr/bin/env python3
"""Long-context + shared-basis ablation on the pre-RoPE cache.

Rationale:
  At ctx=1024 our previous sweeps showed:
    - share_basis=false is always strictly better PPL-wise (by ~20 pp).
    - shrinking block_size improves PPL but explodes skeleton bytes.

  At ctx=8192 the block-level economics are unchanged (compression
  ratio is identical per cell because each block is self-contained),
  but the per-layer PPL error compounds across ~16× more blocks.

  The one configuration that should legitimately change at ctx=8192 is
  share_basis=true: one PCA basis amortised over many blocks within a
  layer.  The fixed +20 pp PPL penalty from basis sharing may now buy a
  much larger byte-ratio gain.

Driver design:
  - load model once
  - build per-passage prefills once (expensive at ctx=8192: ~40s each)
  - sweep (block_size, bit_width, share_basis) cells reusing the same
    prefilled caches
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


def build_prefills(model, tok, passages, ctx_len, n_eval, prefill_chunk):
    entries = []
    for i, p in enumerate(passages):
        ids = tok(p, return_tensors="pt")["input_ids"]
        if ids.shape[-1] < ctx_len + n_eval:
            continue
        prefix = ids[:, :ctx_len]
        cont = ids[:, ctx_len:ctx_len + n_eval]
        t0 = time.time()
        cache_ref = prefill_cache(model, prefix, prefill_chunk)
        c_ref_fwd = copy.deepcopy(cache_ref)
        lo_ref = logits_with_cache(model, c_ref_fwd, cont)
        entries.append({"idx": i, "cache_ref": cache_ref, "cont": cont,
                        "logits_ref": lo_ref.detach().clone()})
        print(f"  passage {i+1}: prefilled ctx={prefix.shape[-1]}, "
              f"cont={cont.shape[-1]}, {time.time()-t0:.1f}s", flush=True)
    return entries


def run_one(model, entries, *, block_size, bit_width, variance_ratio,
            share_basis, compress):
    per_passage = []
    for e in entries:
        cache_alt, stats = roundtrip_cache(
            model, e["cache_ref"],
            block_size=block_size, bit_width=bit_width,
            pca_method="exact", variance_ratio=variance_ratio,
            compress=compress, skeleton_dtype="fp16",
            share_basis_k=share_basis, share_basis_v=share_basis,
        )
        c_alt = copy.deepcopy(cache_alt)
        lo_alt = logits_with_cache(model, c_alt, e["cont"])
        per_passage.append({
            "metrics": compare_logits(e["logits_ref"], lo_alt, e["cont"]),
            "stats": stats,
        })
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
    ap.add_argument("--ctx-len", type=int, required=True)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--n-passages", type=int, default=2)
    ap.add_argument("--compress", choices=["kv", "k_only", "v_only"], default="kv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bit-widths", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--variance-ratios", type=float, nargs="+", default=[1.0])
    ap.add_argument("--block-sizes", type=int, nargs="+",
                    default=[128, 256, 512])
    ap.add_argument("--share-basis-values", type=int, nargs="+", default=[0, 1],
                    help="0 = per-block, 1 = layer-shared")
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

    t_pre = time.time()
    entries = build_prefills(
        model, tok, passages, args.ctx_len, args.n_eval, args.prefill_chunk
    )
    print(f"  prefilled {len(entries)} passages in {time.time() - t_pre:.1f}s",
          flush=True)

    grid = list(product(args.bit_widths, args.variance_ratios,
                        args.block_sizes, args.share_basis_values))
    print(f"  running {len(grid)} configurations (exact PCA, fp16 skel, "
          f"ctx={args.ctx_len}, compress={args.compress})", flush=True)

    rows = []
    t_start = time.time()
    for i, (bw, vr, bs, sb) in enumerate(grid):
        label = f"b={bw} vr={vr} bs={bs} share={'Y' if sb else 'N'}"
        t0 = time.time()
        res = run_one(
            model, entries, block_size=bs, bit_width=bw,
            variance_ratio=vr, share_basis=bool(sb), compress=args.compress,
        )
        dur = time.time() - t0
        if res is None:
            print(f"  [{i+1}/{len(grid)}] {label:<32}  SKIP ({dur:.1f}s)", flush=True)
            continue
        row = {"bit_width": bw, "variance_ratio": vr, "block_size": bs,
               "share_basis": bool(sb),
               "ppl_delta": res["ppl_delta_mean"],
               "kl": res["kl_mean"], "top1": res["top1_mean"],
               "seconds": dur}
        rows.append(row)
        print(f"  [{i+1}/{len(grid)}] {label:<32}  "
              f"Δppl={res['ppl_delta_mean']*100:+7.2f}%  "
              f"KL={res['kl_mean']:.4f}  top1={res['top1_mean']*100:.2f}%  "
              f"({dur:.1f}s)", flush=True)
        out_name = (f"{args.model_name}_{args.compress}_b{bw}_vr{vr}"
                    f"_bs{bs}_share{sb}.json")
        (args.out_dir / out_name).write_text(json.dumps({
            "model_name": args.model_name, "compress": args.compress,
            "ctx_len": args.ctx_len,
            "config": {"pca_method": "exact", "skeleton_dtype": "fp16",
                       "share_basis": bool(sb), "bit_width": bw,
                       "variance_ratio": vr, "block_size": bs},
            **res,
        }, indent=2))

    total = time.time() - t_start
    print(f"\n  total wall time: {total/60:.1f} min over {len(rows)} cells",
          flush=True)

    (args.out_dir / f"{args.model_name}_{args.compress}_long_ctx_summary.json").write_text(
        json.dumps({"model_name": args.model_name,
                    "compress": args.compress,
                    "ctx_len": args.ctx_len,
                    "rows": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
