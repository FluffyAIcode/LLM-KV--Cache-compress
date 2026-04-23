"""M1 TPOT (time-per-output-token) benchmark.

Measures sustained single-request decode throughput for a given
kv-cache-dtype on Qwen3-4B. The M1 exit criterion (per PLAN.md) is
"TPOT within 10 % of published numbers on our H200" for the
turboquant_k8v4 configuration.

Methodology (identical for every kv-cache-dtype — no config-specific
shortcuts):

  * Context length 4096 (prefill), then generate 1024 tokens with
    temperature=0. max_model_len = 5120.
  * 3 warmup iterations, 5 measurement iterations, report min/median/p95
    per-token decode latency.
  * Report prefill latency (TTFT) and aggregate throughput separately.

The prompt for prefill is a deterministic 4096-token excerpt drawn from
the beginning of WikiText-103 raw (public, already used for PPL probes).
Same prompt for every config.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def load_wikitext_chunk(tokenizer, n_tokens: int) -> str:
    """Return a decoded prompt whose token count == n_tokens exactly."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    # Collect enough text to cover n_tokens tokens
    buf_parts: list[str] = []
    tot_tok = 0
    for row in ds:
        t = row["text"]
        if not t.strip():
            continue
        buf_parts.append(t)
        tot_tok += len(tokenizer.encode(t))
        if tot_tok > n_tokens * 3:  # margin for re-encoding
            break
    raw = "\n".join(buf_parts)
    ids = tokenizer.encode(raw)[:n_tokens]
    prompt = tokenizer.decode(ids)
    # Re-encode to make sure we land on exactly n_tokens after tokenization
    # roundtrip — trim again if necessary.
    ids2 = tokenizer.encode(prompt)
    if len(ids2) > n_tokens:
        ids2 = ids2[:n_tokens]
        prompt = tokenizer.decode(ids2)
    return prompt, len(ids2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--gen-tokens", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=5120)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--block-size", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"[m1_tpot_bench] kv_cache_dtype={args.kv_cache_dtype}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompt, prompt_len = load_wikitext_chunk(tok, args.ctx_len)
    print(f"[m1_tpot_bench] prompt_len={prompt_len} tokens", flush=True)

    llm_kwargs: dict[str, Any] = dict(
        model=args.model,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        seed=0,
    )
    if args.block_size is not None:
        llm_kwargs["block_size"] = args.block_size
    llm = LLM(**llm_kwargs)

    sp_warm = SamplingParams(temperature=0.0, top_p=1.0,
                             max_tokens=args.gen_tokens)
    sp_meas = SamplingParams(temperature=0.0, top_p=1.0,
                             max_tokens=args.gen_tokens)

    for i in range(args.warmup):
        t0 = time.time()
        out = llm.generate([prompt], sp_warm, use_tqdm=False)
        t1 = time.time()
        n_out = len(out[0].outputs[0].token_ids)
        print(f"[m1_tpot_bench] warmup {i}: {t1 - t0:.2f} s, "
              f"{n_out} tokens", flush=True)

    per_token_s: list[float] = []
    ttft_s: list[float] = []
    agg_tok_per_s: list[float] = []
    out_tokens: list[int] = []

    for i in range(args.iters):
        t0 = time.time()
        out = llm.generate([prompt], sp_meas, use_tqdm=False)
        t1 = time.time()
        n_out = len(out[0].outputs[0].token_ids)
        out_tokens.append(n_out)
        ttft = getattr(out[0], "metrics", None)
        ttft_val: float
        if ttft is not None and getattr(ttft, "first_token_time", None):
            ttft_val = ttft.first_token_time - ttft.arrival_time
        else:
            # Approximation: assume first token ~ equal share; but vLLM
            # generally provides metrics; we still record the whole-request
            # wall-clock as a conservative upper bound.
            ttft_val = float("nan")
        wall = t1 - t0
        # TPOT = (wall - TTFT) / (n_out - 1) if TTFT known; else wall / n_out
        if n_out >= 2 and ttft_val == ttft_val:  # not NaN
            tpot = (wall - ttft_val) / (n_out - 1)
        else:
            tpot = wall / max(n_out, 1)
        per_token_s.append(tpot)
        ttft_s.append(ttft_val if ttft_val == ttft_val else 0.0)
        agg_tok_per_s.append(n_out / wall if wall > 0 else 0.0)
        print(f"[m1_tpot_bench] iter {i}: wall={wall:.2f}s "
              f"n_out={n_out} TPOT={tpot * 1000:.2f} ms "
              f"TTFT={ttft_val * 1000 if ttft_val == ttft_val else float('nan'):.2f} ms "
              f"agg={n_out / wall:.1f} tok/s",
              flush=True)

    def stats(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"min": 0.0, "median": 0.0, "p95": 0.0, "mean": 0.0}
        xs = sorted(xs)
        return {
            "min": xs[0],
            "median": statistics.median(xs),
            "p95": xs[min(len(xs) - 1, int(0.95 * len(xs)))],
            "mean": statistics.mean(xs),
        }

    summary = {
        "model": args.model,
        "kv_cache_dtype": args.kv_cache_dtype,
        "ctx_len": args.ctx_len,
        "prompt_len": prompt_len,
        "gen_tokens": args.gen_tokens,
        "warmup_iters": args.warmup,
        "measured_iters": args.iters,
        "max_model_len": args.max_model_len,
        "block_size": args.block_size,
        "enforce_eager": args.enforce_eager,
        "tpot_s": stats(per_token_s),
        "ttft_s": stats(ttft_s),
        "aggregate_toks_per_s": stats(agg_tok_per_s),
        "out_tokens": out_tokens,
    }
    print(json.dumps(summary, indent=2), flush=True)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[m1_tpot_bench] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
