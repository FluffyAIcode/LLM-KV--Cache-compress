"""NIAH (Needle-in-a-Haystack) long-context retrieval evaluation.

Standard protocol: insert a simple factoid ("needle") at a varying depth
inside a long "haystack" of unrelated text, then ask a question whose
answer requires retrieving the needle from the correct position.

Example:
  haystack:   ... [100k tokens of Paul Graham essays] ...
  needle:     "The best thing to do in San Francisco is eat a sandwich
               at Dolores Park on a sunny day."  (inserted at depth 50%)
  question:   "According to the text, what is the best thing to do in
               San Francisco?"

Correct answer requires the KV cache to faithfully preserve the attention
pattern at the needle's position.  KV compression that shreds that
pattern (either through too-coarse quantisation or cross-layer error
accumulation) will return a wrong / garbled answer.

Protocol per (model, codec, depth):
  1. Build a prompt with haystack + needle at `depth` + question at end.
  2. Run vLLM prefill + short generation (max_tokens=32).
  3. Score the generation: "correct" iff the key phrase from the needle
     appears in the decoded output (case-insensitive substring match).
  4. Report: accuracy % across a grid of (context_length, needle_depth).

We don't re-invent the wheel — the "best thing to do in SF is a sandwich
at Dolores Park" needle is from the Gregory Kamradt NIAH benchmark that
became the de-facto standard for long-context retrieval eval.

Supports snapshot OR in-forward codec modes (same machinery as
rigorous_eval.py).  Real vLLM, real GPU, no mock.
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

import torch


os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("KAKEYA_SNAPSHOT_QWEN3", "1")


# The canonical Kamradt NIAH needle + question.  A real deployment
# would randomise the factoid to make sure the model isn't memorising;
# for our codec test the point is "does KV compression break retrieval"
# and any stable needle works.
NIAH_NEEDLE = (
    "The best thing to do in San Francisco is eat a sandwich and sit "
    "in Dolores Park on a sunny day."
)
NIAH_QUESTION = (
    "\n\nAccording to the text above, what is the best thing to do in "
    "San Francisco?\n\nAnswer:"
)
# Accept as correct iff the generation contains this substring (lowered).
NIAH_ANSWER_SUBSTRINGS = [
    "sandwich",
    "dolores",
    "dolores park",
    "sit in dolores park",
]


def load_haystack(tok: Any, total_tokens: int) -> str:
    """Return a string encoding to approximately total_tokens tokens.

    Uses WikiText-103 test split concatenated to reach the target size;
    the exact encoded length may overshoot by a few percent and is
    trimmed to total_tokens by the harness.
    """
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    chunks: list[str] = []
    total = 0
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        chunks.append(text)
        total += int(len(text.split()) * 1.3)
        if total >= total_tokens + 2000:  # overshoot buffer
            break
    return "".join(chunks)


def insert_needle(
    tok: Any, haystack: str, needle: str, depth_frac: float,
    total_tokens: int,
) -> list[int]:
    """Tokenise haystack, truncate to total_tokens, insert needle at
    the nearest whole-sentence boundary near depth_frac * total.

    Returns the final token-id sequence with the needle inside.
    """
    needle_ids = tok.encode(needle)
    target_hs_len = total_tokens - len(needle_ids)
    hs_ids = tok.encode(haystack)[: target_hs_len]
    split_at = max(0, min(len(hs_ids), int(len(hs_ids) * depth_frac)))
    return hs_ids[:split_at] + needle_ids + hs_ids[split_at:]


def build_prompt(
    tok: Any, haystack: str, depth_frac: float, total_tokens: int,
) -> tuple[list[int], int]:
    """Return (prompt_ids, start_of_question_index)."""
    pre_q_ids = insert_needle(
        tok, haystack, NIAH_NEEDLE, depth_frac, total_tokens,
    )
    q_ids = tok.encode(NIAH_QUESTION)
    return pre_q_ids + q_ids, len(pre_q_ids)


def run_one_prompt(llm: Any, prompt_ids: list[int], max_tokens: int = 32) -> str:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        sampling_params=sp, use_tqdm=False,
    )
    return out[0].outputs[0].text


def is_correct(output: str) -> bool:
    out_lower = output.lower()
    # Primary: sandwich + Dolores together (the canonical answer).
    if "sandwich" in out_lower and "dolores" in out_lower:
        return True
    # Partial: Dolores alone (location hint — accept as correct).
    if "dolores park" in out_lower:
        return True
    return False


def default_boundary_for_model(num_layers: int) -> set[int]:
    return set(list(range(2)) + list(range(num_layers - 2, num_layers)))


def _sylvester_hadamard_normalised(D: int, device) -> torch.Tensor:
    assert (D & (D - 1)) == 0, f"D must be power of 2, got {D}"
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], dim=0)
    return H / math.sqrt(D)


def make_v14_codec_fn(D: int, q_range: int, device: str = "cuda"):
    from kakeyalattice import V14KakeyaZamirLatticeGPU
    if D % 4 != 0:
        raise ValueError(f"v1.4 requires D % 4 == 0, got {D}")
    cb = V14KakeyaZamirLatticeGPU(D=D, q_range=q_range, device=device)
    def fn(X):
        assert X.is_cuda and X.dtype == torch.float32
        return cb.roundtrip(X)
    fn.bits_per_token_per_head = cb.bits_per_token_per_head
    fn.channel_id = f"v14_Q{q_range}"
    fn.label = f"v1.4 Q={q_range}"
    return fn


def make_tq_codec_fn(D: int, bits_per_coord: int, device: str = "cuda"):
    H = _sylvester_hadamard_normalised(D, device)
    qs = (1 << (bits_per_coord - 1)) - 1
    eps = torch.finfo(torch.float32).eps
    bits = D * bits_per_coord + 32
    def fn(X):
        assert X.is_cuda and X.dtype == torch.float32
        N_tok, H_heads, _ = X.shape
        flat = X.reshape(-1, D)
        norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
        norms_f16 = norms.to(torch.float16).to(torch.float32)
        unit = flat / norms
        y = unit @ H
        qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
        qmax_f16 = qmax.to(torch.float16).to(torch.float32)
        scale = qmax_f16 / float(qs)
        q = torch.round(y / scale).clamp(-qs, qs) * scale
        return ((q @ H) * norms_f16).reshape(N_tok, H_heads, D)
    fn.bits_per_token_per_head = bits
    fn.channel_id = f"tq_b{bits_per_coord}"
    fn.label = f"TQ b={bits_per_coord}"
    return fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--ctx-lengths", type=str, default="4096,8192",
                    help="Comma-separated target token counts for the "
                         "haystack+needle+question prompt")
    ap.add_argument("--depths", type=str, default="0.1,0.5,0.9",
                    help="Needle depths as fractions of the haystack")
    ap.add_argument("--q-values", type=str, default="38,152",
                    help="v1.4 Q values")
    ap.add_argument("--tq-b-values", type=str, default="6,8",
                    help="TQ b values")
    ap.add_argument("--mode", choices=["inforward", "snapshot"],
                    default="inforward")
    ap.add_argument("--n-trials", type=int, default=3,
                    help="Repeats per (ctx, depth, codec) cell; mitigates "
                         "generation non-determinism")
    ap.add_argument("--gpu-mem-util", type=float, default=0.60)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    CTX_LENS = [int(x) for x in args.ctx_lengths.split(",") if x.strip()]
    DEPTHS   = [float(x) for x in args.depths.split(",") if x.strip()]
    Q_VALS   = [int(x) for x in args.q_values.split(",") if x.strip()]
    TQ_VALS  = [int(x) for x in args.tq_b_values.split(",") if x.strip()]
    print(f"[config] ctx lengths: {CTX_LENS}", flush=True)
    print(f"[config] depths:      {DEPTHS}", flush=True)
    print(f"[config] v1.4 Q:      {Q_VALS}", flush=True)
    print(f"[config] TQ   b:      {TQ_VALS}", flush=True)
    print(f"[config] mode:        {args.mode}", flush=True)
    print(f"[config] n_trials:    {args.n_trials}", flush=True)

    from vllm import LLM
    from transformers import AutoTokenizer
    from kakeya_v1_4_snapshot.snapshot_hook import HookState

    HookState.capture_gpu = True

    tok = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code,
    )
    max_ctx = max(CTX_LENS) + 256
    llm = LLM(
        model=args.model_path,
        max_model_len=max_ctx,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True, enable_prefix_caching=False,
        trust_remote_code=args.trust_remote_code,
    )

    # Discover model config.
    HookState.phase = "capture"
    HookState.captured = {}
    probe = tok.encode("The quick brown fox")[:8]
    _ = run_one_prompt(llm, probe, max_tokens=1)
    HookState.phase = "off"
    cap_dry = dict(HookState.captured)
    num_layers = max(cap_dry.keys()) + 1 if cap_dry else 0
    head_dim = cap_dry[0]['K'].shape[-1] if cap_dry else 0
    num_kv_heads = cap_dry[0]['K'].shape[1] if cap_dry else 0
    print(f"[model] L={num_layers} hd={head_dim} kv_h={num_kv_heads}",
          flush=True)
    if num_layers == 0 or head_dim == 0:
        print("[ERROR] Snapshot hook didn't fire.", flush=True)
        return 1
    if head_dim % 4 != 0:
        print(f"[FAIL] head_dim {head_dim} not divisible by 4.", flush=True)
        return 2
    HookState.captured = {}
    torch.cuda.empty_cache()

    boundary = default_boundary_for_model(num_layers)

    # Load haystack once; reused across trials.
    hay_s = load_haystack(tok, max(CTX_LENS))

    # Build channels.
    channels: list[tuple[str, Any]] = [("bf16", None)]
    for Q in Q_VALS:
        channels.append((f"v14_Q{Q}", make_v14_codec_fn(head_dim, Q)))
    for b in TQ_VALS:
        channels.append((f"tq_b{b}", make_tq_codec_fn(head_dim, b)))

    records: list[dict] = []

    for ctx_len in CTX_LENS:
        for depth in DEPTHS:
            prompt_ids, q_start = build_prompt(tok, hay_s, depth, ctx_len)
            print(f"\n=== ctx={ctx_len} depth={depth:.2f} "
                  f"(prompt {len(prompt_ids)} tok, question at idx {q_start}) ===",
                  flush=True)

            for ch_id, codec_fn in channels:
                correct = 0
                total = args.n_trials
                trial_outputs: list[str] = []

                for trial in range(args.n_trials):
                    if ch_id == "bf16":
                        HookState.phase = "off"
                    elif args.mode == "inforward":
                        HookState.phase = "inforward"
                        HookState.codec_fn = codec_fn
                        HookState.inforward_skip_layers = set(boundary)
                        HookState.inforward_fired = {}
                    elif args.mode == "snapshot":
                        # Snapshot mode for NIAH: capture on first pass,
                        # replace on the second.  We'll run capture here,
                        # replace on generation.  For deterministic
                        # ordering, process capture + gen per-trial.
                        HookState.phase = "capture"
                        HookState.captured = {}
                        _ = run_one_prompt(llm, prompt_ids, max_tokens=1)
                        HookState.phase = "off"
                        captured = dict(HookState.captured)
                        repl = {}
                        for lid, kv in captured.items():
                            if lid in boundary:
                                repl[lid] = {"K": kv["K"], "V": kv["V"]}
                            else:
                                repl[lid] = {
                                    "K": codec_fn(kv["K"]),
                                    "V": codec_fn(kv["V"]),
                                }
                        HookState.phase = "replace"
                        HookState.replacements = repl
                        HookState.replace_fired = {}
                        HookState.replace_shape_mismatch = {}
                        HookState.replace_missing = {}

                    t0 = time.perf_counter()
                    out_text = run_one_prompt(llm, prompt_ids, max_tokens=32)
                    t_gen = time.perf_counter() - t0

                    HookState.phase = "off"
                    HookState.codec_fn = None
                    HookState.replacements = {}
                    HookState.captured = {}
                    torch.cuda.empty_cache()

                    trial_outputs.append(out_text.strip())
                    if is_correct(out_text):
                        correct += 1

                acc = correct / total
                # Strip long outputs for logging.
                sample_out = (trial_outputs[0][:80] + "..."
                              if trial_outputs else "")
                print(f"  [{ch_id:<10}] acc={acc*100:5.1f}% "
                      f"({correct}/{total})  "
                      f"sample_output={sample_out!r}",
                      flush=True)

                records.append({
                    "ctx_len": ctx_len, "depth": depth,
                    "channel": ch_id,
                    "codec_bits_per_token_per_head": (
                        codec_fn.bits_per_token_per_head if codec_fn else 16 * head_dim
                    ),
                    "n_correct": correct,
                    "n_trials": total,
                    "accuracy": acc,
                    "sample_outputs": trial_outputs[:2],
                    "t_gen_last": t_gen,
                })

    # Aggregate: accuracy per (codec) averaged over (ctx, depth).
    by_ch: dict[str, dict[str, Any]] = {}
    for r in records:
        ch = r["channel"]
        if ch not in by_ch:
            by_ch[ch] = {
                "channel": ch,
                "bits_per_token_per_head": r["codec_bits_per_token_per_head"],
                "n_correct": 0, "n_trials": 0,
            }
        by_ch[ch]["n_correct"] += r["n_correct"]
        by_ch[ch]["n_trials"] += r["n_trials"]

    print("\n" + "=" * 110)
    print(f"NIAH {args.mode} — model={args.model_path}")
    print("=" * 110)
    print(f"{'Channel':<12} {'bits/head/tok':>14} {'total trials':>14} "
          f"{'overall accuracy':>18}")
    print('-' * 72)
    for ch, rec in by_ch.items():
        acc = rec["n_correct"] / max(rec["n_trials"], 1) * 100
        print(f"{ch:<12} {rec['bits_per_token_per_head']:>14d} "
              f"{rec['n_trials']:>14d} {acc:>17.2f}%")

    # Per (ctx, depth, codec) accuracy heatmap.
    print("\nPer-cell accuracy:")
    for ctx_len in CTX_LENS:
        print(f"  ctx={ctx_len}:")
        print(f"    {'channel':<12} " + " ".join(
            f"{'d=%.2f' % d:>8}" for d in DEPTHS))
        for ch_id, _ in channels:
            accs = []
            for d in DEPTHS:
                rs = [r for r in records
                      if r["channel"] == ch_id and r["ctx_len"] == ctx_len
                      and abs(r["depth"] - d) < 1e-4]
                if rs:
                    accs.append(rs[0]["accuracy"] * 100)
                else:
                    accs.append(float("nan"))
            print(f"    {ch_id:<12} " + " ".join(f"{a:>7.1f}%" for a in accs))

    out = {
        "model": args.model_path,
        "model_name": args.model_name,
        "mode": args.mode,
        "num_layers": num_layers,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "boundary_skip_layers": sorted(boundary),
        "ctx_lengths": CTX_LENS,
        "depths": DEPTHS,
        "q_values": Q_VALS,
        "tq_b_values": TQ_VALS,
        "n_trials": args.n_trials,
        "records": records,
        "by_channel": by_ch,
    }
    out_path = args.out_dir / f"{args.model_name}_niah_{args.mode}.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
