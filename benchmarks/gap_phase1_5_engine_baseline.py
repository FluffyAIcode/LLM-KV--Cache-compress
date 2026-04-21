#!/usr/bin/env python3
"""Phase 1 + 5 — cross-engine baseline PPL + logit-KL, codec OFF.

Is vLLM's ppl_ref on a given prompt the same as HF eager's? If the two
engines already disagree on the CLEAN model at the same tokens, the
"HF v1.3 PPL +7.82 %" and "vLLM v1.3 PPL +35.33 %" numbers are sitting
on different baselines, and the 27 pp gap needs to be decomposed
accordingly.

Runs FOUR WikiText-103 test passages of >= ctx_len + n_eval tokens
through both engines (same tokens) and reports:

  - ppl_ref(engine)                         per passage (Phase 1)
  - top-1 agreement between engines         per passage (Phase 5)
  - mean KL(HF top-K ‖ vLLM top-K)          per passage (Phase 5, proxy)
  - mean |Δ log P(true_token)|              per passage (Phase 5)

The KL is over the top-K logprob bucket that both engines agreed to
emit (K ≈ 20 via `prompt_logprobs=20`); it is an undercount of the
true KL over the full vocab, but it is a lower bound large enough to
tell us if the engines are materially different.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


# =============================================================================
# WikiText-103 loader — the SAME loader as the production harness, so we
# pick the same passages.
# =============================================================================

def load_wikitext_passages(
    tokenizer: Any, min_tokens: int, n_passages: int, split: str = "test",
) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    passages, cur, approx = [], [], 0
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        cur.append(text)
        approx += int(len(text.split()) * 1.3)
        if approx >= min_tokens:
            passage = "".join(cur)
            if len(tokenizer.encode(passage)) >= min_tokens:
                passages.append(passage)
                if len(passages) >= n_passages:
                    return passages
            cur, approx = [], 0
    return passages


# =============================================================================
# vLLM path
# =============================================================================

def run_vllm(model_path: str, passages_ids: list[list[int]],
             ctx_len: int, n_eval: int,
             gpu_mem_util: float, top_k: int) -> list[dict]:
    from vllm import LLM, SamplingParams  # type: ignore
    llm = LLM(
        model=model_path, dtype="bfloat16",
        max_model_len=ctx_len + n_eval + 16,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True, trust_remote_code=True,
    )
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=top_k)
    out: list[dict] = []
    for i, ids in enumerate(passages_ids):
        t0 = time.perf_counter()
        r = llm.generate(
            prompts=None, prompt_token_ids=[ids],
            sampling_params=sp, use_tqdm=False,
        )
        t = time.perf_counter() - t0
        pls = r[0].prompt_logprobs  # list[dict[token_id -> Logprob]]
        # Collect eval window.
        lps_true, top_k_map = [], []
        for pos in range(ctx_len, min(ctx_len + n_eval, len(ids))):
            entry = pls[pos]
            if entry is None:
                lps_true.append(float("-inf"))
                top_k_map.append({})
                continue
            tok = ids[pos]
            def _lp(v: Any) -> float:
                return float(v.logprob if hasattr(v, "logprob") else v["logprob"])
            lps_true.append(_lp(entry[tok]) if tok in entry else float("-inf"))
            # Normalise top-K map to dict[int, float]
            top_k_map.append({int(k): _lp(v) for k, v in entry.items()})
        out.append({
            "t_sec": t, "lps_true": lps_true, "top_k": top_k_map,
        })
    return out


# =============================================================================
# HF eager path (teacher-forced, single forward over ctx+n_eval tokens)
# =============================================================================

def run_hf_eager(model_path: str, passages_ids: list[list[int]],
                 ctx_len: int, n_eval: int, top_k: int) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    print(f"[hf] loading {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True,
    ).to("cuda").eval()
    out: list[dict] = []
    with torch.inference_mode():
        for i, ids in enumerate(passages_ids):
            ids_t = torch.tensor([ids[: ctx_len + n_eval]], device="cuda")
            t0 = time.perf_counter()
            logits = model(ids_t).logits[0]  # [L, V]
            t = time.perf_counter() - t0
            # Position t predicts token at position t+1 (teacher-forced)
            lps_true, top_k_map = [], []
            for pos in range(ctx_len, min(ctx_len + n_eval, ids_t.shape[1])):
                # Predictor lives at pos-1; the true token is ids[pos]
                # (match vLLM's convention: prompt_logprobs[t] reports
                # P(ids[t] | ids[<t])).
                logp_all = torch.log_softmax(
                    logits[pos - 1].to(torch.float32), dim=-1
                ).cpu().numpy()
                tok_id = ids[pos]
                lps_true.append(float(logp_all[tok_id]))
                # Top-K over full vocab
                topk_idx = np.argpartition(-logp_all, top_k)[:top_k]
                top_k_map.append({int(j): float(logp_all[j]) for j in topk_idx})
            out.append({
                "t_sec": t, "lps_true": lps_true, "top_k": top_k_map,
            })
    del model
    torch.cuda.empty_cache()
    return out


# =============================================================================
# Metrics
# =============================================================================

def ppl_from_lps(lps: list[float]) -> float:
    valid = [x for x in lps if np.isfinite(x)]
    if not valid:
        return float("inf")
    mean_nll = -float(np.mean(valid))
    return float(np.exp(mean_nll))


def symmetric_kl_on_topk(a: dict[int, float], b: dict[int, float]) -> float:
    """Symmetric KL over the union of top-K indices. Missing entries are
    assigned the minimum of the present entries (a soft floor, since we
    can't know their true log-prob without the full vocab)."""
    if not a or not b:
        return float("nan")
    keys = set(a) | set(b)
    floor_a = min(a.values()) - 2.0  # 2 nats below observed floor
    floor_b = min(b.values()) - 2.0
    logp_a = np.array([a.get(k, floor_a) for k in keys], dtype=np.float64)
    logp_b = np.array([b.get(k, floor_b) for k in keys], dtype=np.float64)
    # Re-normalise locally over the union set so they are proper distributions.
    logp_a -= np.logaddexp.reduce(logp_a)
    logp_b -= np.logaddexp.reduce(logp_b)
    p_a = np.exp(logp_a)
    p_b = np.exp(logp_b)
    kl_ab = float(np.sum(p_a * (logp_a - logp_b)))
    kl_ba = float(np.sum(p_b * (logp_b - logp_a)))
    return 0.5 * (kl_ab + kl_ba)


def compare(hf_pass: dict, vllm_pass: dict) -> dict:
    lps_hf = hf_pass["lps_true"]
    lps_v = vllm_pass["lps_true"]
    n = min(len(lps_hf), len(lps_v))
    lps_hf = lps_hf[:n]
    lps_v = lps_v[:n]
    ppl_hf = ppl_from_lps(lps_hf)
    ppl_v = ppl_from_lps(lps_v)

    deltas = [
        abs(lps_hf[i] - lps_v[i])
        for i in range(n)
        if np.isfinite(lps_hf[i]) and np.isfinite(lps_v[i])
    ]
    # Top-1 agreement: at each position, whose top-K max id matches?
    agree = 0
    total = 0
    kl_list = []
    for i in range(n):
        a = hf_pass["top_k"][i]
        b = vllm_pass["top_k"][i]
        if not a or not b:
            continue
        total += 1
        if max(a, key=a.get) == max(b, key=b.get):
            agree += 1
        kl_list.append(symmetric_kl_on_topk(a, b))
    kl_valid = [x for x in kl_list if np.isfinite(x)]
    return {
        "ppl_hf": ppl_hf, "ppl_vllm": ppl_v,
        "ppl_rel_gap": (ppl_v - ppl_hf) / max(ppl_hf, 1e-8),
        "mean_abs_dlogp_true": float(np.mean(deltas)) if deltas else float("nan"),
        "top1_agreement": (agree / total) if total else float("nan"),
        "mean_sym_kl_topk": (float(np.mean(kl_valid)) if kl_valid else float("nan")),
        "n_tokens": n,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--gpu-mem-util", type=float, default=0.80)
    ap.add_argument("--order", choices=["hf_first", "vllm_first"],
                    default="vllm_first",
                    help="Run vLLM first (releases GPU), then HF. Avoids the "
                         "'vLLM already holds all GPU memory' deadlock.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer once (via HF; vLLM uses the same HF tokenizer under the
    # hood). This lets us build the fixed passage list BEFORE spinning up
    # either engine.
    from transformers import AutoTokenizer  # type: ignore
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"[setup] loading {args.n_passages} WikiText passages "
          f"(min_tokens={args.ctx_len + args.n_eval})", flush=True)
    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval,
        n_passages=args.n_passages,
    )
    passages_ids: list[list[int]] = []
    for p in passages:
        ids = tok.encode(p)[: args.ctx_len + args.n_eval]
        if len(ids) >= args.ctx_len + args.n_eval:
            passages_ids.append(ids)
    print(f"        usable: {len(passages_ids)}", flush=True)

    # Run both engines.
    if args.order == "vllm_first":
        print("\n[phase1+5] vLLM engine first", flush=True)
        vllm_out = run_vllm(args.model_path, passages_ids,
                            args.ctx_len, args.n_eval,
                            args.gpu_mem_util, args.top_k)
        # vLLM destroys the engine on process exit, but we are still inside
        # the same process so its worker holds GPU memory. We release
        # explicitly by deleting the LLM ref and emptying cache.
        print("\n[phase1+5] HF eager next", flush=True)
        torch.cuda.empty_cache()
        hf_out = run_hf_eager(args.model_path, passages_ids,
                              args.ctx_len, args.n_eval, args.top_k)
    else:
        hf_out = run_hf_eager(args.model_path, passages_ids,
                              args.ctx_len, args.n_eval, args.top_k)
        torch.cuda.empty_cache()
        vllm_out = run_vllm(args.model_path, passages_ids,
                            args.ctx_len, args.n_eval,
                            args.gpu_mem_util, args.top_k)

    # Compare.
    per_passage = []
    for i in range(len(passages_ids)):
        m = compare(hf_out[i], vllm_out[i])
        print(
            f"  passage {i+1}: ppl_hf={m['ppl_hf']:.3f} "
            f"ppl_vllm={m['ppl_vllm']:.3f} "
            f"gap={m['ppl_rel_gap']*100:+.2f}% "
            f"top1_agree={m['top1_agreement']*100:.1f}% "
            f"KL={m['mean_sym_kl_topk']:.4f} "
            f"Δlogp(true)={m['mean_abs_dlogp_true']:.4f}",
            flush=True,
        )
        per_passage.append({"passage": i, **m})

    summary = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "ctx_len": args.ctx_len,
        "n_eval": args.n_eval,
        "n_passages": len(passages_ids),
        "top_k": args.top_k,
        "per_passage": per_passage,
    }
    if per_passage:
        valid = [r for r in per_passage if np.isfinite(r["ppl_rel_gap"])]
        summary["mean_ppl_rel_gap"] = float(np.mean(
            [r["ppl_rel_gap"] for r in valid]
        ))
        summary["mean_top1_agreement"] = float(np.mean(
            [r["top1_agreement"] for r in valid
             if np.isfinite(r["top1_agreement"])]
        ))
        summary["mean_sym_kl_topk"] = float(np.mean(
            [r["mean_sym_kl_topk"] for r in valid
             if np.isfinite(r["mean_sym_kl_topk"])]
        ))
        summary["mean_abs_dlogp_true"] = float(np.mean(
            [r["mean_abs_dlogp_true"] for r in valid
             if np.isfinite(r["mean_abs_dlogp_true"])]
        ))
        print(f"\n  ===== SUMMARY =====", flush=True)
        print(f"  mean PPL rel gap (vLLM-HF)/HF     = "
              f"{summary['mean_ppl_rel_gap']*100:+.3f}%", flush=True)
        print(f"  mean top-1 agreement (engines)    = "
              f"{summary['mean_top1_agreement']*100:.2f}%", flush=True)
        print(f"  mean symmetric KL on top-K        = "
              f"{summary['mean_sym_kl_topk']:.4f}", flush=True)
        print(f"  mean |Δ log P(true_token)|        = "
              f"{summary['mean_abs_dlogp_true']:.4f}", flush=True)

    out_path = args.out_dir / f"{args.model_name}_engine_baseline.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
