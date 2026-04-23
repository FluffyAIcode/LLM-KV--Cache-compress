"""Bridge B2 vs TurboQuant apples-to-apples PPL evaluation on vLLM.

The ONLY codec difference across the three channels is the K-recode.
Everything else — vLLM forward, snapshot hook, per-passage logprob
extraction, PPL / top-1 computation — uses the exact same code as
the production snapA/snapF harness, for a fair comparison.

Channels tested:

  A. bf16 baseline       — identity, no K compression (upper bound,
                            Δppl = 0 by construction).
  B. TurboQuant k8v4-style — per-vector qmax + Hadamard + int8 per-coord.
                            Bit budget: D × 8 + 32 fp16 = 1056 bits/tok/head.
  C. Bridge B2 (D4+TQ)   — per-vector qmax + Hadamard + D4 closest-point
                            per 4-dim block.
                            Bit budget: 32 × 32 + 32 fp16 = 1056 bits/tok/head.

All three use the SAME captured K from vLLM's snapshot hook.  We
only change the K reconstruction algorithm between Pass 1 (clean)
and Pass 2 (replace).

V is passed through unchanged in all three channels, so K-stream
effects are cleanly isolated.

Output: JSON with per-passage ppl_ref / ppl_alt / top1_pair / K-MSE
for each channel + aggregate.  Directly comparable to snapA/snapF's
output format.
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
# Codec implementations — all produce the same I/O: K_np → K_hat_np (float32).
# No mock, no simplification, no fallback.
# ---------------------------------------------------------------------------
def _sylvester_hadamard_normalised(D: int, device: str) -> torch.Tensor:
    """Sylvester Hadamard /√D.  Self-inverse (H·H = I)."""
    assert (D & (D - 1)) == 0
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat(
            [torch.cat([H, H], 1), torch.cat([H, -H], 1)],
            dim=0,
        )
    return H / math.sqrt(D)


def recode_bf16_identity(K_np: np.ndarray) -> np.ndarray:
    """Channel A: identity passthrough, bf16 round-trip to emulate no
    codec (but still vLLM's bf16 activation precision).
    """
    K_t = torch.from_numpy(K_np).cuda()
    # Simulate the bf16 activation round-trip that vLLM does in the
    # attention forward; this is the honest baseline because the reference
    # pass also computes attention in bf16.
    K_bf16 = K_t.to(torch.bfloat16).to(torch.float32)
    return K_bf16.cpu().numpy().astype(np.float32)


def recode_turboquant_k8(K_np: np.ndarray) -> np.ndarray:
    """Channel B: TurboQuant k8v4-style K path.

    Full pipeline:
      unit = K / ‖K‖                         (store ‖K‖ fp16)
      y    = unit · H/√D                     (Hadamard)
      qmax = max_i |y_i|                     (store qmax fp16, per vector)
      q    = round(y · 127 / qmax).clamp(±127) × qmax / 127
      unit_hat = q · H/√D                    (Hadamard self-inverse)
      K_hat = unit_hat × ‖K‖                 (rescale)

    1024 lattice bits + 32 fp16 overhead = 1056 bits/tok/head.
    """
    K_t = torch.from_numpy(K_np).cuda().float()
    N_tok, H_heads, D = K_t.shape
    flat = K_t.reshape(-1, D)
    eps = torch.finfo(torch.float32).eps

    norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
    norms_f16 = norms.to(torch.float16).to(torch.float32)
    unit = flat / norms

    Hmat = _sylvester_hadamard_normalised(D, "cuda")
    y = unit @ Hmat                                               # [N, D]

    qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps) # [N, 1]
    qmax_f16 = qmax.to(torch.float16).to(torch.float32)
    scale = qmax_f16 / 127.0
    q = torch.round(y / scale).clamp(-127, 127) * scale
    unit_hat = q @ Hmat
    K_hat = unit_hat * norms_f16
    return K_hat.reshape(N_tok, H_heads, D).cpu().numpy().astype(np.float32)


def recode_bridge_b2(K_np: np.ndarray, q_range: int = 152) -> np.ndarray:
    """Channel C: Bridge B2 — D4 lattice + full TurboQuant engineering.

    Same pre/post processing as TQ (unit-norm + Hadamard + per-vector
    qmax), but replaces int8 per-coord with D4 closest-lattice-point
    on 4-dim blocks.  Bit budget matches TQ at q_range=152:

      per-block bits = 4·log₂(2·152 + 1) − 1 = 32 bits / 4 dims
      32 blocks × 32 bits = 1024 lattice bits
      + 32 fp16 overhead (‖K‖ + qmax)
      = 1056 bits/tok/head — EXACT match with TQ.
    """
    from kakeyaturbo_py.bridge_b2_d4_tq_style import D4TQStyleCodebook
    D = K_np.shape[-1]
    cb = D4TQStyleCodebook(D=D, q_range=q_range, device="cuda")
    K_t = torch.from_numpy(K_np).cuda().float()
    K_hat = cb.roundtrip(K_t)
    return K_hat.cpu().numpy().astype(np.float32)


def compute_k_mse(K: np.ndarray, K_hat: np.ndarray) -> tuple[float, float]:
    """Return (relative MSE, mean cos(K, K̂)) on per-token basis."""
    diff = K_hat - K
    sq = (diff * diff).sum(axis=-1)                  # [N, H]
    norm_sq = (K * K).sum(axis=-1)
    rel_mse = float(sq.mean() / norm_sq.mean())
    dot = (K_hat * K).sum(axis=-1)
    n1 = np.sqrt((K * K).sum(axis=-1)).clip(min=1e-12)
    n2 = np.sqrt((K_hat * K_hat).sum(axis=-1)).clip(min=1e-12)
    cos = dot / (n1 * n2)
    return rel_mse, float(cos.mean())


# ---------------------------------------------------------------------------
# vLLM harness — mirrors the snapA/snapF harness exactly.
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
            lps.append(0.0)
            top1.append(0)
            continue
        gold_id = ids[t]
        gold_lp = entry[gold_id].logprob
        lps.append(gold_lp)
        best_id = max(entry.items(), key=lambda kv: kv[1].logprob)[0]
        top1.append(int(best_id == gold_id))
    mean_nll = -float(np.mean(lps))
    return float(np.exp(mean_nll)), lps, top1


def top1_ids_from_pls(pls: list[Any]) -> list[int]:
    out = []
    for entry in pls:
        if entry is None:
            out.append(-1)
            continue
        best_id = max(entry.items(), key=lambda kv: kv[1].logprob)[0]
        out.append(best_id)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B")
    ap.add_argument("--model-name", default="qwen3_4b_b2_vs_tq")
    ap.add_argument("--ctx-len",    type=int, default=2048)
    ap.add_argument("--n-eval",     type=int, default=64)
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--boundary-skip-layers", type=int, nargs="*",
                    default=[0, 1, 2, 3, 4, 5, 6, 29, 30, 31, 32, 33, 34, 35])
    ap.add_argument("--b2-q-range", type=int, default=152)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(args.boundary_skip_layers)

    # Load vLLM + snapshot hook — same as snapA/snapF harness.
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

    # Load WikiText passages.
    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval,
        n_passages=args.n_passages,
    )
    passages_ids = [tok.encode(p)[: args.ctx_len + args.n_eval]
                    for p in passages
                    if len(tok.encode(p)) >= args.ctx_len + args.n_eval]

    channels = {
        "bf16":       recode_bf16_identity,
        "tq_k8":      recode_turboquant_k8,
        "bridge_b2":  lambda K: recode_bridge_b2(K, q_range=args.b2_q_range),
    }

    results: dict[str, list[dict]] = {ch: [] for ch in channels}

    for pi, ids in enumerate(passages_ids):
        print(f"\n=== passage {pi + 1}/{len(passages_ids)} ===", flush=True)

        # --- Pass 1: clean prefill, capture K/V ---
        HookState.phase = "capture"
        HookState.captured = {}
        t0 = time.perf_counter()
        ref_pls = prompt_logprobs_for_ids(llm, ids)
        t_ref = time.perf_counter() - t0
        HookState.phase = "off"
        captured = dict(HookState.captured)     # snapshot
        print(f"  [capture] {len(captured)} layers, "
              f"{captured[0]['K'].shape[0]} tokens, {t_ref:.2f}s",
              flush=True)

        ppl_ref, _, _ = ppl_and_top1(
            ref_pls, ids, args.ctx_len, args.ctx_len + args.n_eval,
        )
        ref_top1 = top1_ids_from_pls(ref_pls)[args.ctx_len:args.ctx_len + args.n_eval]
        print(f"  [ref] ppl={ppl_ref:.3f}", flush=True)

        # --- For each channel: recode K, run pass 2, compute PPL ---
        for ch_name, recode_fn in channels.items():
            # Recode every non-boundary layer's K; keep V unchanged.
            replacements: dict[int, dict[str, torch.Tensor]] = {}
            k_mses, k_coss = [], []
            t0 = time.perf_counter()
            for lid, kv in captured.items():
                if lid in skip:
                    # Boundary: identity pass (bf16 only, same for all channels)
                    replacements[lid] = {
                        "K": torch.from_numpy(
                            kv["K"].astype(np.float32)
                        ).cuda(),
                        "V": torch.from_numpy(
                            kv["V"].astype(np.float32)
                        ).cuda(),
                    }
                    continue
                K_orig = np.asarray(kv["K"], dtype=np.float32)
                K_hat = recode_fn(K_orig)
                rel_mse, cos_mean = compute_k_mse(K_orig, K_hat)
                k_mses.append(rel_mse)
                k_coss.append(cos_mean)
                replacements[lid] = {
                    "K": torch.from_numpy(K_hat).cuda(),
                    "V": torch.from_numpy(
                        np.asarray(kv["V"], dtype=np.float32)
                    ).cuda(),
                }
            t_codec = time.perf_counter() - t0

            # --- Pass 2 with this channel's K replacement ---
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
            cos_mean = float(np.mean(k_coss)) if k_coss else 1.0
            print(f"  [{ch_name:>10}] ppl_alt={ppl_alt:.3f} "
                  f"Δppl={delta_ppl * 100:+.3f}% "
                  f"top1_pair={top1_pair * 100:.2f}% "
                  f"K-MSE_rel={rel_mse_mean:.2e} "
                  f"cos={cos_mean:.4f} "
                  f"codec={t_codec:.2f}s alt={t_alt:.2f}s",
                  flush=True)

            results[ch_name].append({
                "passage": pi,
                "ppl_ref":    ppl_ref,
                "ppl_alt":    ppl_alt,
                "delta_ppl":  delta_ppl,
                "top1_pair":  top1_pair,
                "k_mse_rel":  rel_mse_mean,
                "k_cos_mean": cos_mean,
                "t_codec":    t_codec,
                "t_alt":      t_alt,
            })

            del replacements
            torch.cuda.empty_cache()

    # ---- Aggregate ----
    def agg(chans: list[dict]) -> dict:
        if not chans:
            return {}
        return {
            "mean_delta_ppl": float(np.mean([p["delta_ppl"] for p in chans])),
            "mean_top1_pair": float(np.mean([p["top1_pair"] for p in chans])),
            "mean_k_mse_rel": float(np.mean([p["k_mse_rel"] for p in chans])),
            "mean_k_cos":     float(np.mean([p["k_cos_mean"] for p in chans])),
        }

    print("\n\n=============================")
    print("=== aggregate (mean over passages) ===")
    print("=============================")
    agg_results = {}
    for ch_name, per_pass in results.items():
        a = agg(per_pass)
        agg_results[ch_name] = a
        print(f"  {ch_name:>10}: "
              f"Δppl={a['mean_delta_ppl'] * 100:+.3f}% "
              f"top1_pair={a['mean_top1_pair'] * 100:.2f}% "
              f"K-MSE_rel={a['mean_k_mse_rel']:.2e} "
              f"cos={a['mean_k_cos']:.4f}")

    # ---- Head-to-head: Bridge B2 vs TQ ----
    print("\n=== Bridge B2 vs TQ head-to-head ===")
    b2 = agg_results["bridge_b2"]
    tq = agg_results["tq_k8"]
    print(f"  K-MSE_rel ratio (B2/TQ):  "
          f"{b2['mean_k_mse_rel'] / tq['mean_k_mse_rel']:.3f}  "
          f"(< 1.0 means B2 better)")
    print(f"  Δ(Δppl)  (B2 − TQ):       "
          f"{(b2['mean_delta_ppl'] - tq['mean_delta_ppl']) * 100:+.4f} pp")
    print(f"  Δ(top1)  (B2 − TQ):       "
          f"{(b2['mean_top1_pair'] - tq['mean_top1_pair']) * 100:+.4f} pp")

    out = {
        "model":        args.model_path,
        "ctx_len":      args.ctx_len,
        "n_eval":       args.n_eval,
        "n_passages":   len(passages_ids),
        "boundary_skip": sorted(skip),
        "b2_q_range":   args.b2_q_range,
        "per_channel_per_passage": results,
        "aggregate":    agg_results,
    }
    out_path = args.out_dir / f"{args.model_name}_vllm_ppl.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] written → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
