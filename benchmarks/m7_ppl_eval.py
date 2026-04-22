"""M7 perplexity evaluator.

Measures LM perplexity on WikiText-2-raw (test split) for a single KV
cache configuration — same protocol for every config so the numbers
are directly comparable.

Protocol (locked to reduce noise):
  * Dataset: wikitext-2-raw-v1 / test  (standard choice for PPL).
  * Model tokenises the concat of all non-empty lines joined by '\\n\\n'.
  * Sliding-window evaluation: fixed ctx of `--window` tokens, stride
    `--stride` tokens, sum the per-token NLLs of the last
    (window - stride) tokens of each window (so each token contributes
    to exactly one window).  Matches the GPT-2 WikiText PPL convention
    in Hugging Face's `perplexity` guide.
  * Forward-only — no sampling — so batching is trivial and the number
    is deterministic.

Two backends share this runner:
  a) vLLM path  (`--engine vllm`): drives inference through vLLM's
     `LLM.generate` with prompt_logprobs=1 so we get log-probs of the
     tokens that *were* the prompt.  This is the only path that
     exercises the KV-cache-dtype under test, so it's the one used
     for baseline / turboquant_k8v4 / turboquant_4bit_nc /
     kakeya_v1_3_ppl.
  b) HF path    (`--engine hf`): falls back to a pure HuggingFace
     forward for sanity-checking / producing reference PPL on
     dtype=bfloat16 without any KV compression.  Useful to quantify
     how much PPL drift is codec-attributable vs vLLM-pipeline noise.

Output: JSON with fields {window, stride, n_tokens_scored,
mean_nll, perplexity, wall_s}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


def build_eval_text() -> str:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Follow the HF PPL guide: concatenate all non-empty lines with "\n\n".
    lines = [r["text"] for r in ds if r["text"] and r["text"].strip()]
    return "\n\n".join(lines)


def tokenise(text: str, tokenizer) -> list[int]:
    out = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return out["input_ids"][0].tolist()


def eval_with_vllm(args, tokens: list[int]) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = dict(
        model=args.model,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        # +1 for the "dummy" output token vLLM insists on generating;
        # prompt_logprobs=1 + max_tokens=1 still needs one output slot.
        max_model_len=args.window + 1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        seed=0,
    )
    if args.block_size is not None:
        llm_kwargs["block_size"] = args.block_size
    if args.kv_cache_dtype == "kakeya_v1_3_ppl":
        llm_kwargs["attention_backend"] = "CUSTOM"
        if "block_size" not in llm_kwargs:
            llm_kwargs["block_size"] = 512
    llm = LLM(**llm_kwargs)

    # SamplingParams with prompt_logprobs=1 returns the log-prob of each
    # prompt token under the model, which is exactly what we need for
    # NLL summation.  max_tokens=1 means we generate 1 output token
    # (required by vLLM) that we discard.
    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
    )

    total_nll = 0.0
    n_scored = 0
    window = args.window
    stride = args.stride
    # Number of tokens scored per window = stride (for the "new" tokens
    # in the rightmost chunk) except the very first window where we
    # score `window - 1` tokens (all except the BOS placeholder).
    first_window = True

    # Build the list of window token-id sequences.
    prompts: list[list[int]] = []
    score_slices: list[tuple[int, int]] = []  # [start_idx_in_window, end_idx_in_window) exclusive
    pos = 0
    while pos < len(tokens):
        end = min(pos + window, len(tokens))
        prompts.append(tokens[pos:end])
        if first_window:
            # Skip the first token (no preceding context, nothing to
            # predict from).
            score_slices.append((1, end - pos))
            first_window = False
        else:
            # Only score the last `stride` tokens of this window.
            score_slices.append((window - stride, end - pos))
        if end == len(tokens):
            break
        pos += stride

    from vllm.inputs import TokensPrompt
    t0 = time.time()
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts],
        sampling_params=sp,
    )
    t1 = time.time()

    for out, (s, e) in zip(outs, score_slices):
        pl = out.prompt_logprobs
        # vLLM returns None for the very first token (no context);
        # subsequent entries are dict[token_id → Logprob].
        for i in range(s, e):
            entry = pl[i]
            if entry is None:
                continue
            prompt_token_id = out.prompt_token_ids[i]
            lp = entry[prompt_token_id].logprob
            total_nll += -lp
            n_scored += 1

    mean_nll = total_nll / max(n_scored, 1)
    return {
        "window": window,
        "stride": stride,
        "n_tokens_scored": n_scored,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "vllm_gen_wall_s": t1 - t0,
    }


def eval_with_hf(args, tokens: list[int]) -> dict[str, Any]:
    """HF-only reference path (no vLLM, no KV compression)."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
    window = args.window
    stride = args.stride

    total_nll = 0.0
    n_scored = 0
    pos = 0
    first_window = True
    t0 = time.time()
    with torch.no_grad():
        while pos < ids.shape[1]:
            end = min(pos + window, ids.shape[1])
            chunk = ids[:, pos:end]
            out = model(chunk, labels=chunk)
            # Per-token NLL: model returns mean loss; recompute
            # per-token to get exact window-scoped sum.
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()
            losses = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)

            if first_window:
                score_lo = 0   # score every predicted token
                first_window = False
            else:
                score_lo = window - stride - 1   # -1: logits are shifted
            sel = losses[0, score_lo:]
            total_nll += sel.sum().item()
            n_scored += sel.numel()

            if end == ids.shape[1]:
                break
            pos += stride
    t1 = time.time()

    mean_nll = total_nll / max(n_scored, 1)
    return {
        "window": window,
        "stride": stride,
        "n_tokens_scored": n_scored,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "hf_wall_s": t1 - t0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--stride", type=int, default=1024)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--output-path", required=True)
    p.add_argument("--max-tokens-eval", type=int, default=0,
                   help="Cap on total evaluation tokens (0 = full test split)")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = build_eval_text()
    tokens = tokenise(text, tokenizer)
    if args.max_tokens_eval > 0:
        tokens = tokens[:args.max_tokens_eval]

    print(f"[m7_ppl_eval] model={args.model} dtype={args.kv_cache_dtype} "
          f"engine={args.engine} tokens={len(tokens)} "
          f"window={args.window} stride={args.stride}", flush=True)

    t0 = time.time()
    if args.engine == "vllm":
        res = eval_with_vllm(args, tokens)
    else:
        res = eval_with_hf(args, tokens)
    total_wall = time.time() - t0

    summary = {
        "model": args.model,
        "kv_cache_dtype": args.kv_cache_dtype,
        "engine": args.engine,
        "n_input_tokens": len(tokens),
        "block_size": args.block_size,
        "enforce_eager": args.enforce_eager,
        "total_wall_s": total_wall,
        **res,
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[m7_ppl_eval] {args.kv_cache_dtype}: ppl={summary['perplexity']:.3f} "
          f"(n_scored={summary['n_tokens_scored']}) → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
