"""M1 exit-criterion GSM8K evaluator.

Runs Qwen3-4B on the full GSM8K test split (1319 questions) via vLLM,
with a deterministic 8-shot chain-of-thought prompt, then extracts the
final numerical answer and computes exact-match accuracy.

This script is the *same* for every KV-cache configuration we benchmark
(baseline / turboquant_k8v4 / turboquant_4bit_nc / kakeya_v1_3_ppl). No
kv-cache-dtype-specific branching, per the M1 Non-negotiables:
  * No fallback paths
  * No overfit-to-calibration hacks

Usage:
    python m1_gsm8k_eval.py \
        --kv-cache-dtype auto \
        --output-path reports/v1_3_ppl/vllm_backend/gsm8k_baseline.json

    python m1_gsm8k_eval.py \
        --kv-cache-dtype turboquant_k8v4 \
        --output-path reports/v1_3_ppl/vllm_backend/gsm8k_tq_k8v4.json

Exit criterion for M1 (turboquant_k8v4): accuracy ≥ 0.80 on full GSM8K
test split.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


# 8-shot CoT prompt per the GSM8K paper, verbatim. Identical across all
# runs — no prompt overfitting to a specific KV-cache config.
GSM8K_8SHOT: list[tuple[str, str]] = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did Natalia sell "
        "altogether in April and May?",
        "Natalia sold 48/2 = 24 clips in May.\n"
        "Natalia sold 48+24 = 72 clips altogether in April and May.\n"
        "#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
        "minutes of babysitting. How much did she earn?",
        "Weng earns 12/60 = $0.2 per minute.\n"
        "Working 50 minutes, she earned 0.2 x 50 = $10.\n"
        "#### 10",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has "
        "only half of the money she needs. Her parents decided to give her "
        "$15 for that purpose, and her grandparents twice as much as her "
        "parents. How much more money does Betty need to buy the wallet?",
        "In the beginning, Betty has only 100 / 2 = $50.\n"
        "Betty's grandparents gave her 15 * 2 = $30.\n"
        "This means, Betty needs 100 - 50 - 30 - 15 = $5 more.\n"
        "#### 5",
    ),
    (
        "Julie is reading a 120-page book. Yesterday, she was able to read 12 "
        "pages and today, she read twice as many pages as yesterday. If she "
        "wants to read half of the remaining pages tomorrow, how many pages "
        "should she read?",
        "Julie read 12 x 2 = 24 pages today.\n"
        "So she was able to read a total of 12 + 24 = 36 pages since "
        "yesterday.\n"
        "There are 120 - 36 = 84 pages left to be read.\n"
        "Since she wants to read half of the remaining pages tomorrow, then "
        "she should read 84/2 = 42 pages.\n"
        "#### 42",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. "
        "How many pages does he write a year?",
        "He writes each friend 3*2=6 pages a week.\n"
        "So he writes 6*2=12 pages every week.\n"
        "That means he writes 12*52=624 pages a year.\n"
        "#### 624",
    ),
    (
        "Mark has a garden with flowers. He planted plants of three different "
        "colors in it. Ten of them are yellow, and there are 80% more of "
        "those in purple. There are only 25% as many green flowers as there "
        "are yellow and purple flowers. How many flowers does Mark have in "
        "his garden?",
        "There are 80/100 * 10 = 8 more purple flowers than yellow flowers.\n"
        "So in Mark's garden, there are 10 + 8 = 18 purple flowers.\n"
        "Purple and yellow flowers sum up to 10 + 18 = 28 flowers.\n"
        "That means in Mark's garden there are 25/100 * 28 = 7 green "
        "flowers.\n"
        "So in total Mark has 28 + 7 = 35 plants in his garden.\n"
        "#### 35",
    ),
    (
        "Albert is wondering how much pizza he can eat in one day. He buys 2 "
        "large pizzas and 2 small pizzas. A large pizza has 16 slices and a "
        "small pizza has 8 slices. If he eats it all, how many pieces does "
        "he eat that day?",
        "He eats 32 from the largest pizzas because 2 x 16 = 32.\n"
        "He eats 16 from the small pizza because 2 x 8 = 16.\n"
        "He eats 48 pieces because 32 + 16 = 48.\n"
        "#### 48",
    ),
    (
        "Ken created a care package to send to his brother, who was away at "
        "boarding school. Ken placed a box on a scale, and then he poured "
        "into the box enough jelly beans to bring the weight to 2 pounds. "
        "Then, he added enough brownies to cause the weight to triple. Next, "
        "he added another 2 pounds of jelly beans. And finally, he added "
        "enough gummy worms to double the weight once again. What was the "
        "final weight of the box of goodies, in pounds?",
        "To the initial 2 pounds of jelly beans, he added enough brownies to "
        "cause the weight to triple, bringing the weight to 2*3=6 pounds.\n"
        "Next, he added another 2 pounds of jelly beans, bringing the weight "
        "to 6+2=8 pounds.\n"
        "And finally, he added enough gummy worms to double the weight once "
        "again, to a final weight of 8*2=16 pounds.\n"
        "#### 16",
    ),
]


def build_prompt(question: str) -> str:
    parts: list[str] = []
    for q, a in GSM8K_8SHOT:
        parts.append(f"Question: {q}\nAnswer: {a}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
_FALLBACK_NUMBER_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)")


def extract_pred(text: str) -> str | None:
    """Pull out the predicted answer.

    Strategy:
      1. First `####` number in the response.
      2. If absent (e.g. model cut off), take the last number that appears
         on the line starting with the final "Answer:" section.

    No per-question tweaking, no prompt-specific cleanup.
    """
    m = _ANSWER_RE.search(text)
    if m is not None:
        return m.group(1).replace(",", "")
    # Fallback: last number in the generated text
    nums = _FALLBACK_NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def extract_gold(answer: str) -> str:
    m = _ANSWER_RE.search(answer)
    if m is None:
        raise ValueError(f"GSM8K gold answer missing ####: {answer!r}")
    return m.group(1).replace(",", "")


def normalize_number(s: str) -> str:
    try:
        f = float(s)
    except ValueError:
        return s
    # Normalize integer vs float representation so "72" == "72.0"
    if f.is_integer():
        return str(int(f))
    return repr(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--kv-cache-dtype", default="auto",
                        help="auto | turboquant_k8v4 | turboquant_4bit_nc | "
                             "kakeya_v1_3_ppl")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-questions", type=int, default=0,
                        help="0 = full test split (1319). >0 limits for smoke.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--block-size", type=int, default=None,
                        help="vLLM cache block size; required = 512 for "
                             "kakeya_v1_3_ppl codec block alignment")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    print(f"[m1_gsm8k_eval] kv_cache_dtype={args.kv_cache_dtype}", flush=True)
    t0 = time.time()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.num_questions > 0:
        ds = ds.select(range(args.num_questions))
    print(f"[m1_gsm8k_eval] GSM8K test questions: {len(ds)}", flush=True)

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
    # Our custom backend is registered under AttentionBackendEnum.CUSTOM
    # (the reserved third-party slot); vLLM's auto-selector doesn't
    # enumerate it, so we have to ask for it explicitly.
    if args.kv_cache_dtype == "kakeya_v1_3_ppl":
        llm_kwargs["attention_backend"] = "CUSTOM"
        if "block_size" not in llm_kwargs:
            llm_kwargs["block_size"] = 512
    llm = LLM(**llm_kwargs)

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["\n\nQuestion:"],
    )

    prompts = [build_prompt(x["question"]) for x in ds]
    golds = [extract_gold(x["answer"]) for x in ds]

    t_gen0 = time.time()
    out = llm.generate(prompts, sp)
    t_gen1 = time.time()

    records: list[dict[str, Any]] = []
    n_correct = 0
    total_out_tokens = 0
    for rec, o, gold in zip(ds, out, golds, strict=True):
        gen = o.outputs[0].text
        pred = extract_pred(gen)
        ok = pred is not None and normalize_number(pred) == normalize_number(gold)
        n_correct += int(ok)
        total_out_tokens += len(o.outputs[0].token_ids)
        records.append({
            "question": rec["question"],
            "gold": gold,
            "pred": pred,
            "correct": ok,
            "gen": gen,
            "n_out_tokens": len(o.outputs[0].token_ids),
        })

    acc = n_correct / len(ds)
    wall = t_gen1 - t0
    gen_wall = t_gen1 - t_gen0
    tok_per_s = total_out_tokens / gen_wall if gen_wall > 0 else 0.0

    summary = {
        "model": args.model,
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "block_size": args.block_size,
        "enforce_eager": args.enforce_eager,
        "n_questions": len(ds),
        "n_correct": n_correct,
        "accuracy": acc,
        "total_out_tokens": total_out_tokens,
        "wall_s": wall,
        "gen_wall_s": gen_wall,
        "aggregate_throughput_toks_per_s": tok_per_s,
    }
    print(json.dumps(summary, indent=2), flush=True)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f)
    print(f"[m1_gsm8k_eval] wrote {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
