#!/usr/bin/env python3
"""Q-preconditioning ablation on the pre-RoPE cache.

Sweeps a user-specified grid of (bit_width, variance_ratio, block_size)
and, for each cell, runs two configurations back-to-back on the same
cached prefills:

  - Q-precondition OFF (v1.3 baseline)
  - Q-precondition ON  (K is whitened by L = chol(Sigma_q) before the
                        codec and un-whitened by L^{-1} after decode)

Everything else (PCA method, skeleton dtype, share_basis, metric
routing) is held fixed.  The V stream is untouched in both cases.
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
from benchmarks.q_precondition import load as load_q_precond, sanity_check


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


def run_cell(model, entries, *, block_size, bit_width, variance_ratio,
             pca_method, skeleton_dtype, share_basis, compress, q_precond):
    per_passage = []
    for e in entries:
        cache_alt, stats = roundtrip_cache(
            model, e["cache_ref"],
            block_size=block_size, bit_width=bit_width,
            pca_method=pca_method, variance_ratio=variance_ratio,
            compress=compress, skeleton_dtype=skeleton_dtype,
            share_basis_k=share_basis, share_basis_v=share_basis,
            q_precond=q_precond,
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
    ap.add_argument("--q-precondition", required=True, type=Path,
                    help="Path to Sigma_q calibration .safetensors")
    ap.add_argument("--q-precond-skip-layers", type=int, nargs="+", default=[0],
                    help="Layer indices to exclude from Q-preconditioning "
                         "(default=[0], attention-sink layer)")
    ap.add_argument("--ctx-len", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--n-passages", type=int, default=2)
    ap.add_argument("--compress", choices=["kv", "k_only", "v_only"], default="kv")
    ap.add_argument("--pca-method", choices=["exact", "randomized"], default="exact")
    ap.add_argument("--skeleton-dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--share-basis", action="store_true")
    ap.add_argument("--bit-widths", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--variance-ratios", type=float, nargs="+", default=[1.0])
    ap.add_argument("--block-sizes", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.model_name}] loading model…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.eval()
    info = prc.install(model)
    print(f"  patched {info['patched_layers']} attention layers", flush=True)

    qp = load_q_precond(args.q_precondition, skip_layers=args.q_precond_skip_layers)
    san = sanity_check(qp)
    print(f"  Q-precondition: {qp.n_calibrated_layers} layers, n_kv={qp.n_kv}, D={qp.head_dim}")
    print(f"  skip_layers={sorted(qp.skip_layers)}")
    print(f"  sanity: whiten∘unwhiten round-trip max_abs={san['max_abs_err']:.3e}, "
          f"max_rel={san['max_rel_err']:.3e}")

    passages = load_wikitext_passages(tok, args.ctx_len + args.n_eval, args.n_passages)
    print(f"  got {len(passages)} WikiText passages", flush=True)

    t_pre = time.time()
    entries = build_prefills(model, tok, passages, args.ctx_len, args.n_eval,
                             args.prefill_chunk)
    print(f"  prefilled {len(entries)} passages in {time.time() - t_pre:.1f}s",
          flush=True)

    grid = list(product(args.bit_widths, args.variance_ratios, args.block_sizes))
    rows = []
    t_start = time.time()
    for i, (bw, vr, bs) in enumerate(grid):
        label = f"b={bw} vr={vr} bs={bs}"
        print(f"\n  [{i+1}/{len(grid)}] {label}", flush=True)
        for use_qp in [False, True]:
            t0 = time.time()
            res = run_cell(
                model, entries, block_size=bs, bit_width=bw, variance_ratio=vr,
                pca_method=args.pca_method, skeleton_dtype=args.skeleton_dtype,
                share_basis=args.share_basis, compress=args.compress,
                q_precond=qp if use_qp else None,
            )
            dur = time.time() - t0
            if res is None:
                print(f"      Q-precond {'ON ' if use_qp else 'OFF'}: SKIP ({dur:.1f}s)")
                continue
            tag = "ON " if use_qp else "OFF"
            print(f"      Q-precond {tag}: Δppl={res['ppl_delta_mean']*100:+8.3f}%  "
                  f"KL={res['kl_mean']:.4f}  top1={res['top1_mean']*100:6.2f}%  "
                  f"({dur:.1f}s)", flush=True)
            rows.append({
                "bit_width": bw, "variance_ratio": vr, "block_size": bs,
                "q_precond": use_qp,
                "ppl_delta": res["ppl_delta_mean"],
                "kl": res["kl_mean"], "top1": res["top1_mean"],
                "seconds": dur,
            })
            out_name = (f"{args.model_name}_{args.compress}_b{bw}_vr{vr}"
                        f"_bs{bs}_qp{int(use_qp)}.json")
            (args.out_dir / out_name).write_text(json.dumps({
                "model_name": args.model_name,
                "compress": args.compress, "ctx_len": args.ctx_len,
                "config": {
                    "pca_method": args.pca_method,
                    "skeleton_dtype": args.skeleton_dtype,
                    "share_basis": args.share_basis,
                    "bit_width": bw, "variance_ratio": vr,
                    "block_size": bs, "q_precond": use_qp,
                },
                **res,
            }, indent=2))

    total = time.time() - t_start
    print(f"\n  total wall time: {total/60:.1f} min over {len(rows)} cells",
          flush=True)

    summary_path = args.out_dir / f"{args.model_name}_{args.compress}_qp_summary.json"
    summary_path.write_text(json.dumps({
        "model_name": args.model_name,
        "compress": args.compress, "ctx_len": args.ctx_len,
        "rows": rows,
    }, indent=2))

    # Pretty table (Δppl vs block_size, grouped by bit_width × q_precond)
    if rows:
        print(f"\n============ Q-PRECONDITION ABLATION ({args.model_name}) ============\n")
        bss = sorted({r["block_size"] for r in rows})
        for bw in sorted({r["bit_width"] for r in rows}):
            for vr in sorted({r["variance_ratio"] for r in rows}):
                print(f"  bit_width={bw} variance_ratio={vr}:")
                print(f"    {'bs':>5} | {'OFF Δppl':>10} {'ON Δppl':>10} | {'ΔON-OFF':>10} | "
                      f"{'OFF top1':>9} {'ON top1':>9}")
                for bs in bss:
                    off = next((r for r in rows
                                if r["bit_width"] == bw and r["variance_ratio"] == vr
                                and r["block_size"] == bs and not r["q_precond"]), None)
                    on = next((r for r in rows
                               if r["bit_width"] == bw and r["variance_ratio"] == vr
                               and r["block_size"] == bs and r["q_precond"]), None)
                    if off is None or on is None:
                        continue
                    delta = (on["ppl_delta"] - off["ppl_delta"]) * 100
                    print(f"    {bs:>5} | {off['ppl_delta']*100:+9.2f}% {on['ppl_delta']*100:+9.2f}% | "
                          f"{delta:+9.2f}pp | {off['top1']*100:>7.2f}% {on['top1']*100:>7.2f}%")
                print()


if __name__ == "__main__":
    main()
