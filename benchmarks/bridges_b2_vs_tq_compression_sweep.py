"""Compression-rate Pareto sweep: Bridge B2 vs TurboQuant on Qwen3-4B K.

Tests both codecs across bit budgets matching TQ's per-coord bit widths
b ∈ {2, 3, 4, 5, 6, 7, 8}.  For each bit level, measures:

  * K rel-MSE on captured Qwen3-4B K
  * Δppl vs bf16 baseline (via full vLLM snapshot harness)
  * top-1 agreement with bf16 baseline
  * Absolute compression ratio vs raw bf16 K

TQ configurations:
    bits_total = 128·b + 32   (32 = fp16 ‖K‖ + fp16 qmax)

B2 configurations (chosen to closely match TQ's bits at each level):
    bits_per_block = ceil(4·log2(2Q+1) − 1)
    bits_total = 32·bits_per_block + 32

The Q values are derived from the relation b ≈ log2(2Q+1) − 0.25
to put B2 at the nearest block-granular bit budget ≥ TQ's count.

All codecs operate on the SAME captured K for each passage, with V
passed through unchanged.  One vLLM capture pass per passage, then
reused across all sweep configs.

Key output: Pareto table (bits vs rel-MSE, Δppl, top-1) per codec,
plus head-to-head ratio at each bit level.

No mocks, no simplifications, no fallbacks.
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
import torch


os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("KAKEYA_SNAPSHOT_QWEN3", "1")


# ---------------------------------------------------------------------------
# Codec implementations.
# ---------------------------------------------------------------------------
def _sylvester_hadamard_normalised(D: int, device: str) -> torch.Tensor:
    assert (D & (D - 1)) == 0
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat(
            [torch.cat([H, H], 1), torch.cat([H, -H], 1)],
            dim=0,
        )
    return H / math.sqrt(D)


def recode_bf16(K_np: np.ndarray) -> np.ndarray:
    """bf16 round-trip baseline."""
    K_t = torch.from_numpy(K_np).cuda()
    return K_t.to(torch.bfloat16).to(torch.float32).cpu().numpy().astype(np.float32)


def recode_tq(K_np: np.ndarray, bits_per_coord: int) -> tuple[np.ndarray, int]:
    """TurboQuant-style K recode with parameterised bits_per_coord.

    Pipeline:
        unit = K / ‖K‖              (store ‖K‖ fp16)
        y    = unit · H/√D          (Hadamard D=128)
        qmax = max|y|               (store qmax fp16)
        q    = round(y · qs / qmax).clamp(±qs)    qs = 2^(b-1) − 1
        decode: y_hat = q · qmax / qs
        unit_hat = y_hat · H/√D     (Hadamard self-inverse)
        K_hat = unit_hat · ‖K‖

    Returns (K_hat, bits_per_token_per_head).
    """
    K_t = torch.from_numpy(K_np).cuda().float()
    N_tok, H_heads, D = K_t.shape
    flat = K_t.reshape(-1, D)
    eps = torch.finfo(torch.float32).eps

    norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
    norms_f16 = norms.to(torch.float16).to(torch.float32)
    unit = flat / norms

    Hmat = _sylvester_hadamard_normalised(D, "cuda")
    y = unit @ Hmat

    qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
    qmax_f16 = qmax.to(torch.float16).to(torch.float32)

    qs = (1 << (bits_per_coord - 1)) - 1
    scale = qmax_f16 / float(qs)
    q = torch.round(y / scale).clamp(-qs, qs) * scale
    unit_hat = q @ Hmat
    K_hat = unit_hat * norms_f16
    K_hat_np = K_hat.reshape(N_tok, H_heads, D).cpu().numpy().astype(np.float32)

    bits = D * bits_per_coord + 32        # 32 = fp16 ‖K‖ + fp16 qmax
    return K_hat_np, bits


def recode_b2(K_np: np.ndarray, q_range: int) -> tuple[np.ndarray, int]:
    """Bridge B2 recode at specified q_range.

    Uses the D4 lattice + TurboQuant engineering stack.
    """
    from kakeyaturbo_py.bridge_b2_d4_tq_style import D4TQStyleCodebook
    D = K_np.shape[-1]
    cb = D4TQStyleCodebook(D=D, q_range=q_range, device="cuda")
    K_t = torch.from_numpy(K_np).cuda().float()
    K_hat = cb.roundtrip(K_t)
    return (
        K_hat.cpu().numpy().astype(np.float32),
        cb.bits_per_token_per_head,
    )


def compute_k_mse(K: np.ndarray, K_hat: np.ndarray) -> tuple[float, float]:
    diff = K_hat - K
    sq = (diff * diff).sum(axis=-1)
    norm_sq = (K * K).sum(axis=-1)
    rel_mse = float(sq.mean() / norm_sq.mean())
    dot = (K_hat * K).sum(axis=-1)
    n1 = np.sqrt((K * K).sum(axis=-1)).clip(min=1e-12)
    n2 = np.sqrt((K_hat * K_hat).sum(axis=-1)).clip(min=1e-12)
    cos = dot / (n1 * n2)
    return rel_mse, float(cos.mean())


# ---------------------------------------------------------------------------
# vLLM harness wrappers (copied from bridges_b2_vs_tq_vllm_ppl.py).
# ---------------------------------------------------------------------------
def load_wikitext_passages(tok: Any, min_tokens: int, n_passages: int,
                           split: str = "test") -> list[str]:
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


def prompt_logprobs_for_ids(llm: Any, ids: list[int]) -> list[dict]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        sampling_params=sp, use_tqdm=False,
    )
    return out[0].prompt_logprobs


def ppl_and_top1(pls: list[Any], ids: list[int],
                 start: int, end: int) -> tuple[float, list[float], list[int]]:
    lps, top1 = [], []
    for t in range(start, end):
        entry = pls[t]
        if entry is None:
            lps.append(0.0); top1.append(0); continue
        gold_id = ids[t]
        lps.append(entry[gold_id].logprob)
        best_id = max(entry.items(), key=lambda kv: kv[1].logprob)[0]
        top1.append(int(best_id == gold_id))
    mean_nll = -float(np.mean(lps))
    return float(np.exp(mean_nll)), lps, top1


def top1_ids_from_pls(pls: list[Any]) -> list[int]:
    out = []
    for entry in pls:
        if entry is None:
            out.append(-1); continue
        out.append(max(entry.items(), key=lambda kv: kv[1].logprob)[0])
    return out


# ---------------------------------------------------------------------------
# Sweep configuration.
# ---------------------------------------------------------------------------
# TQ bit sweep: bits_per_coord ∈ {2, 3, 4, 5, 6, 7, 8}.
# B2 matched Q values computed from: log2(2Q+1) ≈ b + 0.25, round up.
TQ_B_VALUES = [2, 3, 4, 5, 6, 7, 8]
B2_Q_VALUES = [2, 5, 10, 19, 38, 76, 152]

# Raw K bit count = D × 16 (bf16).
RAW_BITS = 128 * 16        # 2048 bits/token/head for raw bf16 K.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B")
    ap.add_argument("--model-name", default="qwen3_4b_compression_sweep")
    ap.add_argument("--ctx-len",    type=int, default=2048)
    ap.add_argument("--n-eval",     type=int, default=64)
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--boundary-skip-layers", type=int, nargs="*",
                    default=[0, 1, 2, 3, 4, 5, 6, 29, 30, 31, 32, 33, 34, 35])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(args.boundary_skip_layers)

    from vllm import LLM
    from transformers import AutoTokenizer
    from kakeya_v1_3_ppl.snapshot_hook import HookState

    tok = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        max_model_len=args.ctx_len + args.n_eval + 1,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True, enable_prefix_caching=False,
    )

    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval,
        n_passages=args.n_passages,
    )
    passages_ids = [tok.encode(p)[: args.ctx_len + args.n_eval]
                    for p in passages
                    if len(tok.encode(p)) >= args.ctx_len + args.n_eval]

    # Build sweep config list.
    # Each entry: (channel_id, codec_fn, config_label)
    sweep_configs: list[tuple[str, Any, str, int]] = [
        ("bf16_baseline", lambda K: (recode_bf16(K), RAW_BITS), "bf16", RAW_BITS),
    ]
    for b in TQ_B_VALUES:
        sweep_configs.append((
            f"tq_b{b}",
            (lambda b_=b: (lambda K: recode_tq(K, bits_per_coord=b_)))(),
            f"TQ b={b}",
            128 * b + 32,
        ))
    for Q in B2_Q_VALUES:
        sweep_configs.append((
            f"b2_Q{Q}",
            (lambda Q_=Q: (lambda K: recode_b2(K, q_range=Q_)))(),
            f"B2 Q={Q}",
            0,   # computed at runtime from codebook
        ))

    # Per-passage per-config results.
    per_passage: list[dict] = []

    for pi, ids in enumerate(passages_ids):
        print(f"\n=== passage {pi + 1}/{len(passages_ids)} ===", flush=True)

        # --- Pass 1: clean capture ---
        HookState.phase = "capture"
        HookState.captured = {}
        t0 = time.perf_counter()
        ref_pls = prompt_logprobs_for_ids(llm, ids)
        t_ref = time.perf_counter() - t0
        HookState.phase = "off"
        captured = dict(HookState.captured)
        n_tokens = captured[0]['K'].shape[0]
        print(f"  [capture] {len(captured)} layers, {n_tokens} tokens, {t_ref:.2f}s",
              flush=True)

        ppl_ref, _, _ = ppl_and_top1(
            ref_pls, ids, args.ctx_len, args.ctx_len + args.n_eval,
        )
        ref_top1 = top1_ids_from_pls(ref_pls)[args.ctx_len:args.ctx_len + args.n_eval]
        print(f"  [ref] ppl={ppl_ref:.3f}", flush=True)

        for ch_id, recode_fn, label, _hint_bits in sweep_configs:
            # Recode K (V unchanged).  Skip boundary layers → passthrough.
            replacements: dict[int, dict[str, torch.Tensor]] = {}
            k_mses, k_coss = [], []
            actual_bits = None
            t0 = time.perf_counter()
            for lid, kv in captured.items():
                if lid in skip:
                    replacements[lid] = {
                        "K": torch.from_numpy(kv["K"].astype(np.float32)).cuda(),
                        "V": torch.from_numpy(kv["V"].astype(np.float32)).cuda(),
                    }
                    continue
                K_orig = np.asarray(kv["K"], dtype=np.float32)
                K_hat, bits = recode_fn(K_orig)
                if actual_bits is None:
                    actual_bits = bits
                rel_mse, cos_mean = compute_k_mse(K_orig, K_hat)
                k_mses.append(rel_mse); k_coss.append(cos_mean)
                replacements[lid] = {
                    "K": torch.from_numpy(K_hat).cuda(),
                    "V": torch.from_numpy(kv["V"].astype(np.float32)).cuda(),
                }
            t_codec = time.perf_counter() - t0

            # --- Pass 2: replace + teacher-force ---
            HookState.phase = "replace"
            HookState.replacements = replacements
            HookState.replace_fired = {}
            HookState.replace_shape_mismatch = {}
            HookState.replace_missing = {}
            t0 = time.perf_counter()
            alt_pls = prompt_logprobs_for_ids(llm, ids)
            t_alt = time.perf_counter() - t0
            HookState.phase = "off"
            HookState.replacements = {}

            ppl_alt, _, _ = ppl_and_top1(
                alt_pls, ids, args.ctx_len, args.ctx_len + args.n_eval,
            )
            alt_top1 = top1_ids_from_pls(alt_pls)[args.ctx_len:args.ctx_len + args.n_eval]
            top1_pair = float(
                sum(1 for a, r in zip(alt_top1, ref_top1) if a == r and a != -1)
                / max(len(alt_top1), 1)
            )

            delta_ppl = (ppl_alt - ppl_ref) / max(ppl_ref, 1e-9)
            rel_mse_mean = float(np.mean(k_mses)) if k_mses else 0.0
            cos_mean_val = float(np.mean(k_coss)) if k_coss else 1.0
            # Compression ratio: raw bf16 / codec bits (for non-boundary layers).
            compression = RAW_BITS / max(actual_bits, 1) if actual_bits else 1.0

            print(f"  [{label:>8s}] bits={actual_bits:>4d} "
                  f"cr={compression:5.2f}×  "
                  f"Δppl={delta_ppl * 100:+7.3f}%  "
                  f"top1={top1_pair * 100:6.2f}%  "
                  f"K-MSE={rel_mse_mean:.2e}  "
                  f"cos={cos_mean_val:.4f}  "
                  f"codec={t_codec:.2f}s",
                  flush=True)

            per_passage.append({
                "passage":    pi,
                "channel":    ch_id,
                "label":      label,
                "bits":       actual_bits,
                "compression_ratio": compression,
                "ppl_ref":    ppl_ref,
                "ppl_alt":    ppl_alt,
                "delta_ppl":  delta_ppl,
                "top1_pair":  top1_pair,
                "k_mse_rel":  rel_mse_mean,
                "k_cos_mean": cos_mean_val,
                "t_codec":    t_codec,
                "t_alt":      t_alt,
            })

            del replacements
            torch.cuda.empty_cache()

    # Aggregate per channel.
    def agg_channel(ch_id: str) -> dict:
        rows = [p for p in per_passage if p["channel"] == ch_id]
        if not rows:
            return {}
        return {
            "channel":      ch_id,
            "label":        rows[0]["label"],
            "bits":         rows[0]["bits"],
            "compression_ratio": rows[0]["compression_ratio"],
            "mean_delta_ppl":    float(np.mean([p["delta_ppl"]  for p in rows])),
            "mean_abs_delta_ppl":float(np.mean([abs(p["delta_ppl"])  for p in rows])),
            "mean_top1_pair":    float(np.mean([p["top1_pair"]   for p in rows])),
            "mean_k_mse_rel":    float(np.mean([p["k_mse_rel"]   for p in rows])),
            "mean_k_cos":        float(np.mean([p["k_cos_mean"]  for p in rows])),
        }

    channel_ids = list(dict.fromkeys(p["channel"] for p in per_passage))
    aggregates = [agg_channel(cid) for cid in channel_ids]

    print("\n\n=== Aggregate results (mean over {} passages) ===".format(
        len(passages_ids)))
    print(f"{'Config':>10} {'bits':>5} {'CR':>6} "
          f"{'Δppl':>10} {'|Δppl|':>8} {'top1':>8} {'K-MSE':>10} {'cos':>8}")
    for a in aggregates:
        print(f"{a['label']:>10} {a['bits']:>5} "
              f"{a['compression_ratio']:5.2f}× "
              f"{a['mean_delta_ppl'] * 100:+9.3f}% "
              f"{a['mean_abs_delta_ppl'] * 100:7.3f}% "
              f"{a['mean_top1_pair'] * 100:6.2f}% "
              f"{a['mean_k_mse_rel']:10.3e} "
              f"{a['mean_k_cos']:8.4f}")

    # Head-to-head TQ vs B2 at matched bit levels.
    print("\n=== TQ vs B2 Pareto head-to-head ===")
    print(f"{'b':>3} {'TQ bits':>8} {'B2 bits':>8} "
          f"{'K-MSE ratio':>12} {'|Δppl| ratio':>14} "
          f"{'top1 Δpp':>9}")
    for b_idx, b in enumerate(TQ_B_VALUES):
        Q = B2_Q_VALUES[b_idx]
        tq_agg = next((a for a in aggregates if a["channel"] == f"tq_b{b}"), None)
        b2_agg = next((a for a in aggregates if a["channel"] == f"b2_Q{Q}"), None)
        if tq_agg is None or b2_agg is None:
            continue
        kmse_ratio = b2_agg["mean_k_mse_rel"] / max(tq_agg["mean_k_mse_rel"], 1e-12)
        dppl_ratio = b2_agg["mean_abs_delta_ppl"] / max(tq_agg["mean_abs_delta_ppl"], 1e-12)
        top1_diff_pp = (b2_agg["mean_top1_pair"] - tq_agg["mean_top1_pair"]) * 100
        print(f"{b:>3} {tq_agg['bits']:>8} {b2_agg['bits']:>8} "
              f"{kmse_ratio:>12.3f} {dppl_ratio:>14.3f} "
              f"{top1_diff_pp:+9.3f}")

    out = {
        "model": args.model_path,
        "ctx_len": args.ctx_len,
        "n_eval": args.n_eval,
        "n_passages": len(passages_ids),
        "raw_bits_per_token_per_head": RAW_BITS,
        "boundary_skip": sorted(skip),
        "tq_b_values": TQ_B_VALUES,
        "b2_q_values": B2_Q_VALUES,
        "per_passage": per_passage,
        "aggregates": aggregates,
    }
    out_path = args.out_dir / f"{args.model_name}.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] written → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
