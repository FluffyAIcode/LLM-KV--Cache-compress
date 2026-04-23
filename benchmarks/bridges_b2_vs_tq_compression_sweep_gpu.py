"""Compression-rate Pareto sweep: Bridge B2 vs TurboQuant on Qwen3-4B K.

STRICT ALL-GPU PATH: no CPU detour, no numpy round-trip, no mock,
no simplification, no fallback.

Pipeline (every step on CUDA):

  1. vLLM captures K/V via HookState.capture_gpu=True → torch.Tensor
     fp32 stays on GPU at `HookState.captured[lid]["K" | "V"]`.
  2. Per codec config: recode K entirely on GPU (Hadamard, per-vector
     qmax, D4 closest-point, all tensor ops).
  3. Replace dict maps layer_id → {"K": torch.Tensor on cuda, "V":
     torch.Tensor on cuda}.  Pass 2 reads GPU tensors directly.
  4. PPL / top-1 computed from vLLM's per-token logprobs (same code
     path as snapA/snapF).

Tests TQ at bits_per_coord ∈ {2, 3, 4, 5, 6, 7, 8} and matched-
bits B2 at q_range ∈ {2, 5, 10, 19, 38, 76, 152}.  Per-point
outputs: K rel-MSE, |Δppl|, top-1 pair, compression ratio vs raw
bf16 (2048 bits/token/head).
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


# ---------------------------------------------------------------------------
# GPU-only codec implementations.
# ---------------------------------------------------------------------------
def _sylvester_hadamard_normalised(D: int, device) -> torch.Tensor:
    """Sylvester Hadamard divided by √D.  Self-inverse (H·H = I)."""
    assert (D & (D - 1)) == 0
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat(
            [torch.cat([H, H], 1), torch.cat([H, -H], 1)],
            dim=0,
        )
    return H / math.sqrt(D)


def recode_bf16_gpu(K: torch.Tensor) -> tuple[torch.Tensor, int]:
    """bf16 round-trip baseline on GPU."""
    assert K.is_cuda and K.dtype == torch.float32
    return K.to(torch.bfloat16).to(torch.float32), 2048    # raw bf16 "budget"


def recode_tq_gpu(K: torch.Tensor, bits_per_coord: int) -> tuple[torch.Tensor, int]:
    """TurboQuant-style codec, pure GPU.

    Args:
        K: [N, H, D] fp32 on GPU (captured K from vLLM hook).
        bits_per_coord: int in [2, 8].

    Returns:
        (K_hat on GPU, total bits/token/head).
    """
    assert K.is_cuda and K.dtype == torch.float32
    N_tok, H_heads, D = K.shape
    flat = K.reshape(-1, D)
    eps = torch.finfo(torch.float32).eps

    norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
    norms_f16 = norms.to(torch.float16).to(torch.float32)
    unit = flat / norms

    Hmat = _sylvester_hadamard_normalised(D, K.device)
    y = unit @ Hmat

    qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
    qmax_f16 = qmax.to(torch.float16).to(torch.float32)
    qs = (1 << (bits_per_coord - 1)) - 1
    scale = qmax_f16 / float(qs)
    q = torch.round(y / scale).clamp(-qs, qs) * scale
    unit_hat = q @ Hmat
    K_hat = (unit_hat * norms_f16).reshape(N_tok, H_heads, D)

    bits = D * bits_per_coord + 32       # 32 = fp16 ‖K‖ + fp16 qmax
    return K_hat, bits


def recode_b2_gpu(K: torch.Tensor, q_range: int) -> tuple[torch.Tensor, int]:
    """Bridge B2 codec (D4 lattice + TurboQuant wrapper), pure GPU.

    Args:
        K: [N, H, D] fp32 on GPU.
        q_range: int, the D4 lattice quantisation range.

    Returns:
        (K_hat on GPU, total bits/token/head).
    """
    assert K.is_cuda and K.dtype == torch.float32
    from kakeyaturbo_py.bridge_b2_d4_tq_style import D4TQStyleCodebook
    D = K.shape[-1]
    cb = D4TQStyleCodebook(D=D, q_range=q_range, device=K.device)
    K_hat = cb.roundtrip(K)
    return K_hat, cb.bits_per_token_per_head


def compute_k_mse_gpu(K: torch.Tensor, K_hat: torch.Tensor) -> tuple[float, float]:
    """K rel-MSE and mean cos(K, K_hat) on GPU.  Returns Python floats."""
    assert K.is_cuda and K_hat.is_cuda
    diff = K_hat - K
    sq = (diff * diff).sum(dim=-1)
    norm_sq = (K * K).sum(dim=-1)
    rel_mse = float((sq.mean() / norm_sq.mean()).item())
    dot = (K_hat * K).sum(dim=-1)
    n1 = (K * K).sum(dim=-1).sqrt().clamp(min=1e-12)
    n2 = (K_hat * K_hat).sum(dim=-1).sqrt().clamp(min=1e-12)
    cos = (dot / (n1 * n2)).mean()
    return rel_mse, float(cos.item())


# ---------------------------------------------------------------------------
# vLLM harness.
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
                 start: int, end: int) -> tuple[float, list[int]]:
    """Returns (ppl, top1_pred_ids)."""
    lps: list[float] = []
    top1_ids: list[int] = []
    for t in range(start, end):
        entry = pls[t]
        if entry is None:
            lps.append(0.0)
            top1_ids.append(-1)
            continue
        gold_id = ids[t]
        lps.append(entry[gold_id].logprob)
        best_id = max(entry.items(), key=lambda kv: kv[1].logprob)[0]
        top1_ids.append(best_id)
    import statistics
    mean_nll = -statistics.fmean(lps)
    return float(math.exp(mean_nll)), top1_ids


# ---------------------------------------------------------------------------
# Sweep configuration.
# ---------------------------------------------------------------------------
TQ_B_VALUES = [2, 3, 4, 5, 6, 7, 8]
B2_Q_VALUES = [2, 5, 10, 19, 38, 76, 152]
RAW_BITS = 128 * 16                  # 2048 bits/token/head for raw bf16 K.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B")
    ap.add_argument("--model-name", default="qwen3_4b_compression_sweep_gpu")
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

    # Enable strict-GPU capture mode on the snapshot hook.
    HookState.capture_gpu = True

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

    # Build sweep config list.  Each entry: (ch_id, recode_fn, label)
    sweep: list[tuple[str, Any, str]] = [
        ("bf16_baseline", recode_bf16_gpu, "bf16"),
    ]
    for b in TQ_B_VALUES:
        sweep.append((
            f"tq_b{b}",
            (lambda b_=b: (lambda K: recode_tq_gpu(K, bits_per_coord=b_)))(),
            f"TQ b={b}",
        ))
    for Q in B2_Q_VALUES:
        sweep.append((
            f"b2_Q{Q}",
            (lambda Q_=Q: (lambda K: recode_b2_gpu(K, q_range=Q_)))(),
            f"B2 Q={Q}",
        ))

    per_passage: list[dict] = []

    for pi, ids in enumerate(passages_ids):
        print(f"\n=== passage {pi + 1}/{len(passages_ids)} ===", flush=True)

        # --- Pass 1: clean capture (GPU tensors stay on GPU) ---
        HookState.phase = "capture"
        HookState.captured = {}
        t0 = time.perf_counter()
        ref_pls = prompt_logprobs_for_ids(llm, ids)
        t_ref = time.perf_counter() - t0
        HookState.phase = "off"
        captured = dict(HookState.captured)
        # Sanity: every captured K must be a GPU torch.Tensor.
        for lid, kv in captured.items():
            assert isinstance(kv["K"], torch.Tensor) and kv["K"].is_cuda, \
                f"layer {lid} K is not a GPU tensor (capture_gpu={HookState.capture_gpu})"
            assert isinstance(kv["V"], torch.Tensor) and kv["V"].is_cuda
        n_tokens = captured[0]['K'].shape[0]
        print(f"  [capture] {len(captured)} layers, {n_tokens} tokens, {t_ref:.2f}s  "
              f"[strict-GPU]", flush=True)

        ppl_ref, ref_top1_full = ppl_and_top1(
            ref_pls, ids, args.ctx_len, args.ctx_len + args.n_eval,
        )
        print(f"  [ref] ppl={ppl_ref:.3f}", flush=True)

        for ch_id, recode_fn, label in sweep:
            # Recode K on GPU; V untouched.
            replacements: dict[int, dict[str, torch.Tensor]] = {}
            k_mse_sum, k_cos_sum, k_count = 0.0, 0.0, 0
            actual_bits: int | None = None
            t0 = time.perf_counter()
            for lid, kv in captured.items():
                K_gpu: torch.Tensor = kv["K"]
                V_gpu: torch.Tensor = kv["V"]
                if lid in skip:
                    replacements[lid] = {"K": K_gpu, "V": V_gpu}
                    continue
                K_hat, bits = recode_fn(K_gpu)
                if actual_bits is None:
                    actual_bits = bits
                rel_mse, cos_mean = compute_k_mse_gpu(K_gpu, K_hat)
                k_mse_sum += rel_mse
                k_cos_sum += cos_mean
                k_count += 1
                replacements[lid] = {"K": K_hat, "V": V_gpu}
            torch.cuda.synchronize()
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

            ppl_alt, alt_top1 = ppl_and_top1(
                alt_pls, ids, args.ctx_len, args.ctx_len + args.n_eval,
            )
            top1_pair = float(
                sum(1 for a, r in zip(alt_top1, ref_top1_full) if a == r and a != -1)
                / max(len(alt_top1), 1)
            )
            delta_ppl = (ppl_alt - ppl_ref) / max(ppl_ref, 1e-9)
            rel_mse_mean = k_mse_sum / max(k_count, 1)
            cos_mean_val = k_cos_sum / max(k_count, 1)
            compression = RAW_BITS / max(actual_bits or RAW_BITS, 1)

            print(f"  [{label:>8s}] bits={actual_bits:>4d} "
                  f"cr={compression:5.2f}×  "
                  f"Δppl={delta_ppl * 100:+7.3f}%  "
                  f"top1={top1_pair * 100:6.2f}%  "
                  f"K-MSE={rel_mse_mean:.2e}  "
                  f"cos={cos_mean_val:.4f}  "
                  f"codec={t_codec:.2f}s",
                  flush=True)

            per_passage.append({
                "passage": pi,
                "channel": ch_id, "label": label,
                "bits": actual_bits,
                "compression_ratio": compression,
                "ppl_ref": ppl_ref, "ppl_alt": ppl_alt,
                "delta_ppl": delta_ppl, "top1_pair": top1_pair,
                "k_mse_rel": rel_mse_mean, "k_cos_mean": cos_mean_val,
                "t_codec": t_codec, "t_alt": t_alt,
            })
            del replacements
            torch.cuda.empty_cache()

        # Free the captured tensors after we've consumed all configs for
        # this passage.
        del captured
        HookState.captured = {}
        torch.cuda.empty_cache()

    # Aggregate per channel.
    def agg(ch_id: str) -> dict | None:
        import statistics
        rows = [p for p in per_passage if p["channel"] == ch_id]
        if not rows:
            return None
        return {
            "channel": ch_id,
            "label": rows[0]["label"],
            "bits": rows[0]["bits"],
            "compression_ratio": rows[0]["compression_ratio"],
            "mean_delta_ppl":     statistics.fmean(p["delta_ppl"]  for p in rows),
            "mean_abs_delta_ppl": statistics.fmean(abs(p["delta_ppl"]) for p in rows),
            "mean_top1_pair":     statistics.fmean(p["top1_pair"]  for p in rows),
            "mean_k_mse_rel":     statistics.fmean(p["k_mse_rel"]  for p in rows),
            "mean_k_cos":         statistics.fmean(p["k_cos_mean"] for p in rows),
        }

    channel_ids = list(dict.fromkeys(p["channel"] for p in per_passage))
    aggregates = [agg(cid) for cid in channel_ids]
    aggregates = [a for a in aggregates if a is not None]

    print("\n=== Aggregate results (mean over {} passages, STRICT-GPU) ===".format(
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

    # Head-to-head at matched bit levels.
    print("\n=== TQ vs B2 Pareto head-to-head (all metrics strict GPU) ===")
    print(f"{'b':>3} {'TQ bits':>8} {'B2 bits':>8} "
          f"{'K-MSE ratio':>12} {'|Δppl| ratio':>14} "
          f"{'top1 Δpp':>10}")
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
              f"{top1_diff_pp:+10.3f}")

    out = {
        "model": args.model_path,
        "ctx_len": args.ctx_len,
        "n_eval": args.n_eval,
        "n_passages": len(passages_ids),
        "strict_gpu": True,
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
