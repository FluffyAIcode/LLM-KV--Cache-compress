#!/usr/bin/env python3
"""Q calibration for Q-preconditioned PCA.

For every full-attention layer l and every kv-head h_k we compute

    Sigma_q^{(l, h_k)} = sum_{h_q in group(h_k)}  E[ q^{(l, h_q)} (q^{(l, h_q)})^T ]

where q is the *pre-RoPE* query (right after q_proj, no rotary).  That is
what the codec's k_pre sees when attention is factored as

    q_post^T k_post = q_pre^T R^T R k_pre = q_pre^T R(delta) k_pre

and the coupling with k_pre under marginalised relative position is the
pre-RoPE Gram matrix (to leading order — exact at zero position lag,
orthogonally mixed at nonzero lags, which is why we collect q_pre over
many positions and many prompts, not just one).

We then Cholesky-factor each Sigma_q and store L (lower triangular).
The downstream codec pipeline whitens K via `K_tilde = K @ L` before
encode and unwhitens via `K_hat = K_hat_tilde @ L^{-T}` after decode.
This is mathematically identical to using a Sigma_q-weighted distortion
in PCA / K-means / Lloyd-Max, but requires zero Rust codec change.

Math.  We want to minimise

    sum_i  e_i^T  Sigma_q  e_i  = tr(E Sigma_q E^T)  = || E L ||_F^2
                                                     = || K L - K_hat L ||_F^2

Whiten before codec:       K_tilde     = K @ L
Un-whiten after decode:    K_hat       = K_hat_tilde @ L^{-1}

where L is the lower-triangular Cholesky factor of Sigma_q (L L^T = Sigma_q).

Output file format (safetensors):

    layer_<l>_chol     : [n_kv, D, D]  fp32  (lower triangular L)
    layer_<l>_inv_chol : [n_kv, D, D]  fp32  (lower triangular L^{-1})
    layer_<l>_sigma    : [n_kv, D, D]  fp32  (for diagnostics only)

plus a `config.json` sidecar with shapes and axis meanings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transformers import AutoModelForCausalLM, AutoTokenizer

import benchmarks.pre_rope_cache as prc
from benchmarks.e2e_ppl_pre_rope import load_wikitext_passages


def _kv_group_ranges(num_q_heads: int, num_kv_heads: int):
    """For each kv-head, return the list of query-head indices that share it."""
    assert num_q_heads % num_kv_heads == 0, "non-evenly-divisible GQA unsupported"
    group_size = num_q_heads // num_kv_heads
    return [list(range(h * group_size, (h + 1) * group_size))
            for h in range(num_kv_heads)]


@torch.inference_mode()
def calibrate(model_path: str, out_path: Path, *,
              n_passages: int, ctx_len: int, prefill_chunk: int,
              ridge: float):
    print(f"loading {model_path}…", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.eval()
    info = prc.install(model)
    print(f"  patched {info['patched_layers']} attention layers", flush=True)

    cfg = model.config.get_text_config(decoder=True)
    # Qwen3 and similar families set an explicit head_dim that differs
    # from hidden_size // num_attention_heads. Honor the explicit value
    # when present; fall back to the classical formula otherwise.
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    groups = _kv_group_ranges(n_q, n_kv)
    n_layers = cfg.num_hidden_layers
    layer_types = getattr(cfg, "layer_types", None) or (
        ["full_attention"] * n_layers
    )
    print(f"  D={head_dim}, n_q={n_q}, n_kv={n_kv}, groups per kv = {len(groups[0])}")

    passages = load_wikitext_passages(tok, ctx_len, n_passages)
    print(f"  using {len(passages)} WikiText passages of >= {ctx_len} tokens", flush=True)

    # Accumulate gram sums, not per-passage tensors — cheap and bounded memory.
    # gram[l][h_kv]  shape [D, D]   sum_{all tokens in group} q q^T
    gram = [[np.zeros((head_dim, head_dim), dtype=np.float64)
             for _ in range(n_kv)]
            for _ in range(n_layers)]
    count = [0 for _ in range(n_layers)]

    for i, passage in enumerate(passages):
        ids = tok(passage, return_tensors="pt")["input_ids"][:, :ctx_len]
        if ids.shape[-1] < ctx_len:
            print(f"  passage {i+1}: SKIP (too short: {ids.shape[-1]})")
            continue

        # Reset recorder for this passage so we can consume incrementally
        # without letting memory pile up at long context.
        cfg._q_recorder = {}

        # Chunked prefill; recorder accumulates q_pre per chunk.
        if prefill_chunk <= 0 or ids.shape[-1] <= prefill_chunk:
            _ = model(input_ids=ids, use_cache=False)
        else:
            # use_cache=False means no kv cache is kept — we only want Q stats.
            # BUT attention still needs the full context so calibrate on full
            # input in one shot (bf16 CPU is fine at ctx=1024-2048).
            _ = model(input_ids=ids, use_cache=False)

        # Consume recorder: accumulate into grams, then drop refs.
        for l_idx, q_list in cfg._q_recorder.items():
            if layer_types[l_idx] != "full_attention":
                continue
            # q_list entries: [bsz=1, n_q, seq, D]
            for q in q_list:
                q = q[0]  # [n_q, seq, D]
                for h_kv in range(n_kv):
                    # Sum over all q heads in this kv group
                    group = groups[h_kv]
                    q_group = q[group]  # [g, seq, D]
                    q_flat = q_group.reshape(-1, head_dim).numpy()  # [g*seq, D]
                    gram[l_idx][h_kv] += q_flat.T @ q_flat
                count[l_idx] += q.shape[1]  # seq, same for every layer
        cfg._q_recorder = None
        print(f"  passage {i+1}/{len(passages)}: processed", flush=True)

    # Finalize: Sigma = gram / (group_size * total_tokens_seen)
    # then L = chol(Sigma + ridge*I),  L^{-T}
    print(f"\nfactoring Sigma_q for {n_layers} layers ×  {n_kv} kv-heads…",
          flush=True)
    out_tensors = {}
    diagnostics = []
    for l in range(n_layers):
        if layer_types[l] != "full_attention":
            continue
        if count[l] == 0:
            continue
        chol_stack = np.zeros((n_kv, head_dim, head_dim), dtype=np.float32)
        inv_chol_stack = np.zeros_like(chol_stack)
        sigma_stack = np.zeros_like(chol_stack)
        group_size = len(groups[0])
        n_tokens_per_head = group_size * count[l]
        for h_kv in range(n_kv):
            sigma = gram[l][h_kv] / n_tokens_per_head
            # Symmetrize against fp drift
            sigma = 0.5 * (sigma + sigma.T)
            # Ridge for numerical stability — ridge * mean_diag
            mean_diag = float(np.mean(np.diag(sigma)))
            sigma_reg = sigma + ridge * mean_diag * np.eye(head_dim)
            L = np.linalg.cholesky(sigma_reg)       # lower triangular, L L^T = Sigma
            # Inverse of L (also lower triangular) for the unwhitening post-decode step.
            L_inv = np.linalg.solve(L, np.eye(head_dim))
            chol_stack[h_kv]     = L.astype(np.float32)
            inv_chol_stack[h_kv] = L_inv.astype(np.float32)
            sigma_stack[h_kv]    = sigma.astype(np.float32)

            evals = np.linalg.eigvalsh(sigma_reg)
            diagnostics.append({
                "layer": l, "kv_head": h_kv,
                "sigma_trace": float(np.trace(sigma)),
                "eig_min":     float(evals.min()),
                "eig_max":     float(evals.max()),
                "condition":   float(evals.max() / max(evals.min(), 1e-30)),
                "diag_mean":   mean_diag,
                "off_diag_max_abs": float(np.abs(sigma - np.diag(np.diag(sigma))).max()),
            })
        out_tensors[f"layer_{l}_chol"]     = torch.from_numpy(chol_stack)
        out_tensors[f"layer_{l}_inv_chol"] = torch.from_numpy(inv_chol_stack)
        out_tensors[f"layer_{l}_sigma"]    = torch.from_numpy(sigma_stack)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file(out_tensors, str(out_path))

    cfg_sidecar = out_path.with_suffix(".json")
    cfg_sidecar.write_text(json.dumps({
        "model_path": model_path,
        "head_dim": head_dim,
        "num_q_heads": n_q,
        "num_kv_heads": n_kv,
        "num_layers": n_layers,
        "layer_types": layer_types,
        "n_passages_used": sum(1 for c in count if c > 0),
        "ctx_len": ctx_len,
        "ridge": ridge,
        "diagnostics": diagnostics,
    }, indent=2))
    print(f"wrote {out_path} + {cfg_sidecar}", flush=True)

    # Summary stats: is Sigma_q anisotropic, or close to isotropic?
    conds = [d["condition"] for d in diagnostics]
    off_ratios = [
        d["off_diag_max_abs"] / max(d["diag_mean"], 1e-30)
        for d in diagnostics
    ]
    print(f"\nSigma_q anisotropy summary (across all (layer, kv_head) pairs):")
    print(f"  condition number:         min={min(conds):.2f}  "
          f"median={np.median(conds):.2f}  max={max(conds):.2f}")
    print(f"  max(|off-diag|)/mean_diag: min={min(off_ratios):.3f}  "
          f"median={np.median(off_ratios):.3f}  max={max(off_ratios):.3f}")
    print("  (condition ≫ 1 or off/diag ≫ 0 ⇒ Sigma_q is anisotropic, "
          "so Q-precondition has something to do)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-path", type=Path, required=True,
                    help=".safetensors file path for the calibration")
    ap.add_argument("--n-passages", type=int, default=8)
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--prefill-chunk", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=1e-3,
                    help="Cholesky ridge = ridge * mean_diag(Sigma) for numerical stability")
    args = ap.parse_args()
    calibrate(
        args.model_path, args.out_path,
        n_passages=args.n_passages, ctx_len=args.ctx_len,
        prefill_chunk=args.prefill_chunk, ridge=args.ridge,
    )


if __name__ == "__main__":
    main()
