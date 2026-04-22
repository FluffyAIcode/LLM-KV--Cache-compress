#!/usr/bin/env python3
"""Phase 3 — per-layer codec attribution on vLLM.

For each full-attention layer L in [0, num_layers), run the production
v1.3 PPL codec cell with codec active ONLY on layer L (all other
layers stay bf16). Measure \u0394ppl / top-1 per (L, stream). Emits a
heat-map-ready JSON.

Reuses the production harness building blocks from
e2e_ppl_validation_vllm_full.py: same Q-precond + calibrated Lloyd-Max
+ outlier T=2.0 recipe. Only the `boundary_skip_layers` set is
inverted to singleton-exclude layer L.

Cost: 28 layers * 4 passages * ~8 s/passage * 2 runs per layer (ref
is cached across layers) ~= 15-20 min on H200.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))

# Reuse the production harness's codec and hook. We import its module,
# use its CodecState, install_qwen2_pre_rope_patch, ppl_and_top1,
# and reuse its logic by flipping only CodecState.boundary_skip_layers.
from e2e_ppl_validation_vllm_full import (  # type: ignore  # noqa: E402
    CodecState, install_qwen2_pre_rope_patch,
    prompt_logprobs_for_ids, ppl_and_top1,
    load_wikitext_passages, build_llm,
)
from q_precondition import load as qp_load  # type: ignore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--num-layers", type=int, default=28)
    ap.add_argument("--q-calib", type=str,
        default="reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors")
    ap.add_argument("--k-centroids", type=str,
        default="reports/v1_4_q_pca/calibrated_codebook/ds_K_b3_centroids.f32")
    ap.add_argument("--v-centroids", type=str,
        default="reports/v1_4_q_pca/calibrated_codebook/ds_V_b2_centroids.f32")
    ap.add_argument("--outlier-threshold", type=float, default=2.0)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--bit-width-k", type=int, default=3)
    ap.add_argument("--bit-width-v", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--only-layers", type=int, nargs="*", default=None,
                    help="If set, only analyse these layers. Default: "
                         "all layers in [0, num_layers).")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Populate the production CodecState. We only vary
    # boundary_skip_layers per cell.
    CodecState.block_size = args.block_size
    CodecState.bit_width_k = args.bit_width_k
    CodecState.bit_width_v = args.bit_width_v
    CodecState.pca_method = "randomized"
    CodecState.rsvd_target_rank_factor = 0.5
    CodecState.variance_ratio = 0.95
    CodecState.k_centroids_file = args.k_centroids
    CodecState.v_centroids_file = args.v_centroids
    CodecState.k_outlier_threshold = args.outlier_threshold
    CodecState.share_basis_v = True
    CodecState.compress_stream = "kv"  # both K and V compressed, on this layer
    print(f"[setup] loading Q-precond from {args.q_calib}", flush=True)
    CodecState.q_precond = qp_load(args.q_calib, skip_layers=[0])

    install_qwen2_pre_rope_patch()
    print(f"[{args.model_name}] loading vLLM engine…", flush=True)
    llm = build_llm(args.model_path, args.ctx_len + args.n_eval + 16,
                    args.gpu_mem_util)
    tok = llm.get_tokenizer()

    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval,
        n_passages=args.n_passages,
    )
    passages_ids = [tok.encode(p)[: args.ctx_len + args.n_eval]
                    for p in passages
                    if len(tok.encode(p)) >= args.ctx_len + args.n_eval]
    print(f"  usable passages: {len(passages_ids)}", flush=True)

    # Reference once (codec off on every layer).
    CodecState.active = False
    ref_pls = [prompt_logprobs_for_ids(llm, ids) for ids in passages_ids]

    def _mean_ppl_delta_and_top1(ref_pls_list, alt_pls_list,
                                  ids_list) -> tuple[float, float]:
        rels, agrees = [], []
        for ri, ids in enumerate(ids_list):
            end = min(args.ctx_len + args.n_eval, len(ids))
            ppl_r, lp_r, t_r = ppl_and_top1(ref_pls_list[ri], ids,
                                            args.ctx_len, end)
            ppl_a, lp_a, t_a = ppl_and_top1(alt_pls_list[ri], ids,
                                            args.ctx_len, end)
            if np.isfinite(ppl_r) and np.isfinite(ppl_a) and ppl_r > 0:
                rels.append((ppl_a - ppl_r) / ppl_r)
            n = min(len(t_r), len(t_a))
            if n:
                agrees.append(float(np.mean([1.0 if t_r[i] == t_a[i] else 0.0
                                             for i in range(n)])))
        return (
            float(np.mean(rels)) if rels else float("nan"),
            float(np.mean(agrees)) if agrees else float("nan"),
        )

    all_layers = list(range(args.num_layers)) if args.only_layers is None \
        else list(args.only_layers)
    per_layer: list[dict] = []
    for L in all_layers:
        # Codec active only on layer L: skip every OTHER layer.
        CodecState.boundary_skip_layers = set(
            l for l in range(args.num_layers) if l != L
        )
        CodecState.active = True
        t0 = time.perf_counter()
        alt_pls = [prompt_logprobs_for_ids(llm, ids) for ids in passages_ids]
        t = time.perf_counter() - t0
        CodecState.active = False
        d_ppl, top1 = _mean_ppl_delta_and_top1(ref_pls, alt_pls, passages_ids)
        print(
            f"  L{L:02d}: Δppl={d_ppl*100:+.3f}% "
            f"top1={top1*100:.2f}% ({t:.2f}s)",
            flush=True,
        )
        per_layer.append({
            "layer": L, "mean_ppl_delta_rel": d_ppl,
            "mean_top1_agreement": top1, "t_sec": t,
        })

    summary = {
        "engine": "vllm",
        "model_name": args.model_name,
        "ctx_len": args.ctx_len,
        "n_eval": args.n_eval,
        "n_passages": len(passages_ids),
        "num_layers": args.num_layers,
        "config": {
            "bit_width_k": args.bit_width_k, "bit_width_v": args.bit_width_v,
            "outlier_threshold": args.outlier_threshold,
            "block_size": args.block_size,
            "q_calib": args.q_calib,
            "k_centroids": args.k_centroids,
            "v_centroids": args.v_centroids,
        },
        "per_layer": per_layer,
    }
    out_path = args.out_dir / f"{args.model_name}_vllm_per_layer.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}", flush=True)

    # Quick tabular summary.
    print("\n===== SUMMARY =====", flush=True)
    print("  layer   Δppl(%)    top1(%)", flush=True)
    for r in per_layer:
        print(f"  L{r['layer']:02d}  {r['mean_ppl_delta_rel']*100:+9.3f}  "
              f"{r['mean_top1_agreement']*100:6.2f}", flush=True)
    # Rank by Δppl magnitude.
    ranked = sorted(per_layer, key=lambda r: -abs(r["mean_ppl_delta_rel"]))
    print("\n  top-5 |Δppl| layers:", flush=True)
    for r in ranked[:5]:
        print(f"    L{r['layer']:02d}  {r['mean_ppl_delta_rel']*100:+.3f}%",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
