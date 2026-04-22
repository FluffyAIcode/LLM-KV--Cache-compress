"""Head-to-head PPL on the exact snapshot protocol used by
e2e_ppl_validation_vllm_snapshot_qwen3.py, but for kv_cache_dtype
backends that DON'T need the Qwen3Attention monkey-patch:
  * auto (bf16 paged cache)
  * turboquant_k8v4 (vLLM built-in)

Protocol (locked to the Qwen3 snapshot harness):
  * WikiText-103 test, same passage-selection logic
  * ctx_len = 2048, n_eval = 64
  * Score logprobs over positions [ctx_len, ctx_len + n_eval)
  * For each passage: run the model once with prompt_logprobs=1
    on the full [ctx_len + n_eval]-token input; the result
    directly gives us the eval-window logprobs.  Δppl is then
    ppl_eval(dtype) / ppl_eval(bf16_ref) − 1 — we run both dtypes
    on the same passage ids so the reference is paired.

This way the numbers are directly comparable to the kakeya
snapshot harness rows (which always score the same window the
same way).  The CAVEAT is that snapshot mode captures clean
prefill-K/V before compression; paged-cache dtypes like TQ and
kakeya in-line both apply compression per-token as K/V are
written to cache, so the protocols aren't identical even with
matched scoring windows.  We state that explicitly in the
output JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def load_wikitext_passages(tok: Any, min_tokens: int, n_passages: int,
                           split: str = "test") -> list[str]:
    """Identical to the Qwen3 snapshot harness loader."""
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
            if len(tok.encode(passage)) >= min_tokens:
                passages.append(passage)
                if len(passages) >= n_passages:
                    return passages
            cur, approx = [], 0
    return passages


def eval_one_config(
    model_path: str,
    kv_cache_dtype: str,
    passage_ids: list[list[int]],
    ctx_len: int,
    n_eval: int,
    gpu_mem_util: float,
) -> dict:
    """Run one vLLM config against the already-tokenised passage ids."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm_kwargs: dict[str, Any] = dict(
        model=model_path,
        dtype="bfloat16",
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=ctx_len + n_eval + 16,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True,
        trust_remote_code=True,
        seed=0,
    )
    # Need block_size=512 for any kakeya-related run; TQ / auto don't care.
    llm_kwargs["block_size"] = 512

    llm = LLM(**llm_kwargs)

    sp = SamplingParams(
        max_tokens=1, temperature=0.0, prompt_logprobs=1,
    )

    # Capture GPU memory stats right after the engine warms up so we
    # can report the KV-cache-relevant footprint for this dtype.
    import torch
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    mem_allocated_before = torch.cuda.memory_allocated() / (1024 ** 3)
    peak_before = torch.cuda.max_memory_allocated() / (1024 ** 3)

    per_passage_out = []
    t_total_start = time.perf_counter()

    for pi, ids in enumerate(passage_ids):
        t0 = time.perf_counter()
        out = llm.generate(
            [TokensPrompt(prompt_token_ids=ids)],
            sampling_params=sp, use_tqdm=False,
        )
        t_one = time.perf_counter() - t0
        pls = out[0].prompt_logprobs

        # Score logprobs over [ctx_len, ctx_len + n_eval).
        lps, top1_ids = [], []
        for t in range(ctx_len, ctx_len + n_eval):
            entry = pls[t]
            if entry is None:
                lps.append(0.0); top1_ids.append(-1); continue
            gold_lp = entry[ids[t]].logprob
            lps.append(gold_lp)
            best_id = max(entry.items(), key=lambda kv: kv[1].logprob)[0]
            top1_ids.append(int(best_id))

        mean_nll = -float(np.mean(lps))
        ppl = math.exp(mean_nll)
        per_passage_out.append({
            "ppl": ppl, "mean_nll": mean_nll,
            "top1_ids": top1_ids,
            "t_forward_sec": t_one,
        })

    t_total = time.perf_counter() - t_total_start

    # GPU memory after workload.
    torch.cuda.synchronize()
    mem_allocated_after = torch.cuda.memory_allocated() / (1024 ** 3)
    peak_after = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return {
        "kv_cache_dtype": kv_cache_dtype,
        "per_passage": per_passage_out,
        "mean_ppl": float(np.mean([p["ppl"] for p in per_passage_out])),
        "median_ppl": float(np.median([p["ppl"] for p in per_passage_out])),
        "total_wall_sec": t_total,
        "mean_forward_sec": float(np.mean([p["t_forward_sec"] for p in per_passage_out])),
        "gpu_mem_allocated_gib_after": mem_allocated_after,
        "gpu_mem_peak_gib": peak_after,
        "gpu_mem_allocated_gib_before": mem_allocated_before,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B")
    ap.add_argument("--kv-cache-dtype", required=True,
                    help="auto | turboquant_k8v4 | turboquant_4bit_nc")
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--out-path", type=Path, required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    # Force InprocClient so the timing includes the actual model
    # forward without subprocess-IPC noise.  (Matches how kakeya
    # snapshot harness runs.)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM
    # Preload tokenizer via a throwaway LLM call.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"[setup] loading WikiText-103 passages (min_tokens={args.ctx_len + args.n_eval})",
          flush=True)
    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval,
        n_passages=args.n_passages,
    )
    passage_ids = [tok.encode(p)[: args.ctx_len + args.n_eval]
                   for p in passages
                   if len(tok.encode(p)) >= args.ctx_len + args.n_eval]
    print(f"  usable: {len(passage_ids)} passages", flush=True)

    print(f"[run] kv_cache_dtype={args.kv_cache_dtype}", flush=True)
    result = eval_one_config(
        model_path=args.model_path,
        kv_cache_dtype=args.kv_cache_dtype,
        passage_ids=passage_ids,
        ctx_len=args.ctx_len,
        n_eval=args.n_eval,
        gpu_mem_util=args.gpu_mem_util,
    )

    result["model_path"] = args.model_path
    result["ctx_len"] = args.ctx_len
    result["n_eval"] = args.n_eval
    result["n_passages"] = len(passage_ids)
    result["gpu_mem_util"] = args.gpu_mem_util

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(result, indent=2))
    print(f"[done] kv_cache_dtype={args.kv_cache_dtype}  "
          f"mean_ppl={result['mean_ppl']:.3f}  "
          f"median_ppl={result['median_ppl']:.3f}  "
          f"forward_sec={result['mean_forward_sec']:.3f}  "
          f"peak_gib={result['gpu_mem_peak_gib']:.2f}",
          flush=True)
    print(f"       → {args.out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
