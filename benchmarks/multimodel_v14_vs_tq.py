"""Multi-model head-to-head: v1.4 kakeya zamir lattice GPU vs TurboQuant.

Strict-GPU, real vLLM, no mock / simplification / fallback.

Measurement dimensions per model:
  1. K rel-MSE on non-boundary layers (full compression-rate sweep)
  2. |Δppl| vs bf16 baseline (4 passages)
  3. top-1 pair agreement vs bf16
  4. Decode speed (ms / M vec, codec side)
  5. KV-cache memory savings (raw bf16 bytes / codec bits per token)
  6. Absolute compression ratio

Accepted snapshot-hook patches (any one fires, depending on which
model is loaded by the harness):
  * Qwen3Attention  (Qwen3 family — e.g. Qwen/Qwen3-4B)
  * Qwen2Attention  (DeepSeek-R1-Distill-Qwen-1.5B and base Qwen2)
  * Gemma4Attention (google/gemma-4-E4B / E2B / 26B-A4B / 31B)
  * GLMAttention    (zai-org/GLM-4-9B-Chat)

All patches are installed by `install_all_snapshot_patches()` via
the plugin entry point when `KAKEYA_SNAPSHOT_QWEN3=1` is set.

Strict-GPU capture: `HookState.capture_gpu = True` keeps K/V tensors
on device; the codec pipeline never touches CPU / numpy in the hot
path.  `assert K.is_cuda` guards against regression.
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


# Bits per token per head for raw bf16 K.
def raw_bits_per_token_per_head(head_dim: int) -> int:
    return 16 * head_dim


# ---------------------------------------------------------------------------
# Codec implementations (pure GPU, identical to compression_sweep_gpu).
# ---------------------------------------------------------------------------
def _sylvester_hadamard_normalised(D: int, device) -> torch.Tensor:
    """Sylvester Hadamard divided by √D; self-inverse on ortho H·H=I."""
    assert (D & (D - 1)) == 0, f"Hadamard requires D = power of 2, got {D}"
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat(
            [torch.cat([H, H], 1), torch.cat([H, -H], 1)], dim=0,
        )
    return H / math.sqrt(D)


def recode_bf16_gpu(K: torch.Tensor) -> tuple[torch.Tensor, int]:
    assert K.is_cuda and K.dtype == torch.float32
    D = K.shape[-1]
    return K.to(torch.bfloat16).to(torch.float32), raw_bits_per_token_per_head(D)


def recode_tq_gpu(K: torch.Tensor, bits_per_coord: int) -> tuple[torch.Tensor, int]:
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
    bits = D * bits_per_coord + 32
    return K_hat, bits


def recode_v14_gpu(K: torch.Tensor, q_range: int) -> tuple[torch.Tensor, int]:
    """v1.4 kakeya zamir lattice GPU."""
    assert K.is_cuda and K.dtype == torch.float32
    from kakeyaturbo_py import V14KakeyaZamirLatticeGPU
    D = K.shape[-1]
    # Only D divisible by 4 is supported.  If the model's head_dim isn't
    # divisible by 4, we cannot run v1.4 on it — RAISE so the harness
    # sees a loud failure, NO fallback.
    if D % 4 != 0:
        raise ValueError(
            f"v1.4 requires head_dim divisible by 4 for D4 blocks, "
            f"got {D}.  No fallback by design."
        )
    cb = V14KakeyaZamirLatticeGPU(D=D, q_range=q_range, device=K.device)
    return cb.roundtrip(K), cb.bits_per_token_per_head


def compute_k_mse_gpu(K: torch.Tensor, K_hat: torch.Tensor) -> tuple[float, float]:
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
# WikiText + vLLM logprob utilities (same as compression_sweep_gpu).
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
    import statistics
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
    mean_nll = -statistics.fmean(lps)
    return float(math.exp(mean_nll)), top1_ids


# ---------------------------------------------------------------------------
# Per-model config: which layers are 'boundary' (kept bf16 for stability)
# and which are eligible for codec.  On 4-passage test we cover all layers
# for the four-model comparison, since we're not deploying snapA.
# ---------------------------------------------------------------------------
# A modest symmetric boundary: keep the first + last 2 layers bf16.  This
# stabilises the PPL measurement (snapA used a deeper 14-layer boundary,
# but that was tuned for a specific codec + model).  For a fair
# cross-model harness we use a conservative constant.
def default_boundary_for_model(num_layers: int) -> set[int]:
    return set(list(range(2)) + list(range(num_layers - 2, num_layers)))


# ---------------------------------------------------------------------------
# Sweep configurations.  TQ b ∈ {4, 6, 8}, v1.4 matched-ish Q ∈ {10, 38, 152}.
# Three operating points spanning the Pareto curve (aggressive, mid, quality).
# ---------------------------------------------------------------------------
TQ_B_VALUES = [4, 6, 8]
V14_Q_VALUES = [10, 38, 152]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True,
                    help="HF model id, e.g. Qwen/Qwen3-4B, "
                         "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, "
                         "google/gemma-4-E4B, zai-org/GLM-4-9B-Chat")
    ap.add_argument("--model-name", required=True,
                    help="Short name for output JSON filename")
    ap.add_argument("--ctx-len",    type=int, default=2048)
    ap.add_argument("--n-eval",     type=int, default=64)
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="Required for GLM-4-9B (uses custom code)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM
    from transformers import AutoTokenizer
    from kakeya_v1_4_snapshot.snapshot_hook import HookState

    HookState.capture_gpu = True

    tok = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code,
    )

    llm = LLM(
        model=args.model_path,
        max_model_len=args.ctx_len + args.n_eval + 1,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True, enable_prefix_caching=False,
        trust_remote_code=args.trust_remote_code,
    )

    # Discover model config: num layers, num_kv_heads, head_dim — via
    # the first captured layer.  We do a tiny dry-run to populate
    # HookState.head_size / num_kv_heads.
    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval,
        n_passages=args.n_passages,
    )
    passages_ids = [tok.encode(p)[: args.ctx_len + args.n_eval]
                    for p in passages
                    if len(tok.encode(p)) >= args.ctx_len + args.n_eval]

    if not passages_ids:
        print("[ERROR] No passages long enough — abort.", flush=True)
        return 1

    # Dry pass to get num_layers etc.
    HookState.phase = "capture"
    HookState.captured = {}
    _ = prompt_logprobs_for_ids(llm, passages_ids[0][:args.ctx_len])
    HookState.phase = "off"
    captured_dry = dict(HookState.captured)
    num_layers = max(captured_dry.keys()) + 1 if captured_dry else 0
    if num_layers == 0:
        print(f"[ERROR] Snapshot hook did not fire for model {args.model_path}. "
              f"Either the model architecture isn't supported by "
              f"install_all_snapshot_patches(), or the KAKEYA_SNAPSHOT_QWEN3 "
              f"env var wasn't set before vLLM import.", flush=True)
        return 1

    head_dim = captured_dry[0]['K'].shape[-1]
    num_kv_heads = captured_dry[0]['K'].shape[1]
    print(f"[model] {args.model_path}", flush=True)
    print(f"[model] num_layers={num_layers}  head_dim={head_dim}  "
          f"num_kv_heads={num_kv_heads}", flush=True)

    # Sanity: v1.4 requires head_dim % 4 == 0.
    if head_dim % 4 != 0:
        print(f"[FAIL] head_dim={head_dim} is not divisible by 4; "
              f"v1.4 cannot run on this model without redesigning the "
              f"D4-block split.  Aborting.", flush=True)
        return 2

    # Release dry-run memory.
    del captured_dry
    HookState.captured = {}
    torch.cuda.empty_cache()

    boundary = default_boundary_for_model(num_layers)

    # Build sweep.
    sweep: list[tuple[str, Any, str]] = [
        ("bf16_baseline", recode_bf16_gpu, "bf16"),
    ]
    for b in TQ_B_VALUES:
        sweep.append((
            f"tq_b{b}",
            (lambda b_=b: (lambda K: recode_tq_gpu(K, bits_per_coord=b_)))(),
            f"TQ b={b}",
        ))
    for Q in V14_Q_VALUES:
        sweep.append((
            f"v14_Q{Q}",
            (lambda Q_=Q: (lambda K: recode_v14_gpu(K, q_range=Q_)))(),
            f"v1.4 Q={Q}",
        ))

    raw_bits = raw_bits_per_token_per_head(head_dim)
    per_passage: list[dict] = []

    for pi, ids in enumerate(passages_ids):
        print(f"\n=== passage {pi + 1}/{len(passages_ids)} ===", flush=True)

        HookState.phase = "capture"
        HookState.captured = {}
        t0 = time.perf_counter()
        ref_pls = prompt_logprobs_for_ids(llm, ids)
        t_ref = time.perf_counter() - t0
        HookState.phase = "off"
        captured = dict(HookState.captured)
        for lid, kv in captured.items():
            assert isinstance(kv["K"], torch.Tensor) and kv["K"].is_cuda
            assert isinstance(kv["V"], torch.Tensor) and kv["V"].is_cuda
        print(f"  [capture] {len(captured)} layers, "
              f"{captured[next(iter(captured))]['K'].shape[0]} tokens, "
              f"{t_ref:.2f}s  [strict-GPU]", flush=True)

        ppl_ref, ref_top1 = ppl_and_top1(
            ref_pls, ids, args.ctx_len, args.ctx_len + args.n_eval,
        )
        print(f"  [ref] ppl={ppl_ref:.3f}", flush=True)

        for ch_id, recode_fn, label in sweep:
            replacements: dict[int, dict[str, torch.Tensor]] = {}
            k_mse_sum, k_cos_sum, k_count = 0.0, 0.0, 0
            actual_bits: int | None = None
            t0 = time.perf_counter()
            for lid, kv in captured.items():
                K_gpu: torch.Tensor = kv["K"]
                V_gpu: torch.Tensor = kv["V"]
                if lid in boundary:
                    replacements[lid] = {"K": K_gpu, "V": V_gpu}
                    continue
                K_hat, bits = recode_fn(K_gpu)
                if actual_bits is None:
                    actual_bits = bits
                rel_mse, cos_m = compute_k_mse_gpu(K_gpu, K_hat)
                k_mse_sum += rel_mse
                k_cos_sum += cos_m
                k_count += 1
                replacements[lid] = {"K": K_hat, "V": V_gpu}
            torch.cuda.synchronize()
            t_codec = time.perf_counter() - t0

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

            n_fired = sum(HookState.replace_fired.values())
            n_mismatch = sum(len(v) for v in HookState.replace_shape_mismatch.values())
            n_missing = sum(HookState.replace_missing.values())
            if ch_id != "bf16_baseline":
                # For any non-baseline, expect fire count > 0 for non-boundary
                # layers.  A zero fire count means the hook silently passed
                # through LIVE K/V and the measurement is meaningless.
                expected_fires = (num_layers - len(boundary))
                if n_fired < expected_fires:
                    print(f"  [{label:>10}] ERROR: only {n_fired} layer "
                          f"replacements fired out of {expected_fires} expected. "
                          f"mismatch={n_mismatch} missing={n_missing}. "
                          f"Measurement invalid — aborting channel.",
                          flush=True)
                    # Record the failure and continue with next channel.
                    per_passage.append({
                        "passage": pi, "channel": ch_id, "label": label,
                        "fire_count": n_fired,
                        "expected_fires": expected_fires,
                        "fatal": "silent passthrough — snapshot hook did not replace K/V",
                    })
                    del replacements
                    torch.cuda.empty_cache()
                    continue

            ppl_alt, alt_top1 = ppl_and_top1(
                alt_pls, ids, args.ctx_len, args.ctx_len + args.n_eval,
            )
            top1_pair = float(
                sum(1 for a, r in zip(alt_top1, ref_top1) if a == r and a != -1)
                / max(len(alt_top1), 1)
            )
            delta_ppl = (ppl_alt - ppl_ref) / max(ppl_ref, 1e-9)
            rel_mse_mean = k_mse_sum / max(k_count, 1)
            cos_mean_val = k_cos_sum / max(k_count, 1)
            compression = raw_bits / max(actual_bits or raw_bits, 1)
            kv_memory_saved_frac = 1.0 - (1.0 / compression)

            # Decode speed: per token per head, how many ms per million vectors
            # the codec took (= t_codec / (num_codec_layers * num_tokens * num_heads))
            n_tokens = captured[next(iter(captured))]['K'].shape[0]
            total_vecs = max(1, (num_layers - len(boundary)) * n_tokens * num_kv_heads)
            decode_speed_ms_per_M = t_codec * 1e3 * 1e6 / total_vecs

            print(f"  [{label:>10}] bits={actual_bits:>4d} "
                  f"CR={compression:5.2f}×  "
                  f"Δppl={delta_ppl*100:+7.3f}%  "
                  f"top1={top1_pair*100:6.2f}%  "
                  f"K-MSE={rel_mse_mean:.2e}  "
                  f"cos={cos_mean_val:.4f}  "
                  f"codec={t_codec:.2f}s "
                  f"({decode_speed_ms_per_M:.1f}ms/M)  "
                  f"fires={n_fired}",
                  flush=True)

            per_passage.append({
                "passage": pi, "channel": ch_id, "label": label,
                "bits": actual_bits,
                "compression_ratio": compression,
                "kv_memory_saved_frac": kv_memory_saved_frac,
                "ppl_ref": ppl_ref, "ppl_alt": ppl_alt,
                "delta_ppl": delta_ppl, "top1_pair": top1_pair,
                "k_mse_rel": rel_mse_mean, "k_cos_mean": cos_mean_val,
                "t_codec": t_codec, "t_alt": t_alt,
                "decode_speed_ms_per_M_vec": decode_speed_ms_per_M,
                "fire_count": n_fired,
                "mismatch_count": n_mismatch,
                "missing_count": n_missing,
            })
            del replacements
            torch.cuda.empty_cache()

        del captured
        HookState.captured = {}
        torch.cuda.empty_cache()

    # Aggregate per channel.
    def agg(ch_id: str) -> dict | None:
        import statistics as _st
        rows = [p for p in per_passage
                if p["channel"] == ch_id and "fatal" not in p]
        if not rows:
            return None
        return {
            "channel": ch_id,
            "label": rows[0]["label"],
            "bits": rows[0]["bits"],
            "compression_ratio": rows[0]["compression_ratio"],
            "kv_memory_saved_frac": rows[0]["kv_memory_saved_frac"],
            "mean_delta_ppl":     _st.fmean(p["delta_ppl"]  for p in rows),
            "mean_abs_delta_ppl": _st.fmean(abs(p["delta_ppl"]) for p in rows),
            "mean_top1_pair":     _st.fmean(p["top1_pair"]  for p in rows),
            "mean_k_mse_rel":     _st.fmean(p["k_mse_rel"]  for p in rows),
            "mean_k_cos":         _st.fmean(p["k_cos_mean"] for p in rows),
            "mean_decode_ms_per_M": _st.fmean(p["decode_speed_ms_per_M_vec"] for p in rows),
        }

    channel_ids = list(dict.fromkeys(p["channel"] for p in per_passage))
    aggregates = [agg(cid) for cid in channel_ids]
    aggregates = [a for a in aggregates if a is not None]

    print("\n=== Aggregate (mean over {} passages, model={}, strict-GPU) ===".format(
        len(passages_ids), args.model_path))
    print(f"{'Config':>10} {'bits':>5} {'CR':>6} {'KVsave':>7} "
          f"{'Δppl':>10} {'|Δppl|':>8} {'top1':>8} "
          f"{'K-MSE':>10} {'cos':>8} {'ms/M':>7}")
    for a in aggregates:
        print(f"{a['label']:>10} {a['bits']:>5} "
              f"{a['compression_ratio']:5.2f}× "
              f"{a['kv_memory_saved_frac']*100:6.2f}% "
              f"{a['mean_delta_ppl']*100:+9.3f}% "
              f"{a['mean_abs_delta_ppl']*100:7.3f}% "
              f"{a['mean_top1_pair']*100:6.2f}% "
              f"{a['mean_k_mse_rel']:10.3e} "
              f"{a['mean_k_cos']:8.4f} "
              f"{a['mean_decode_ms_per_M']:7.2f}")

    # v1.4 vs TQ head-to-head at matched (nearest) bit levels.
    print("\n=== v1.4 vs TQ head-to-head (matched bit levels) ===")
    print(f"{'TQ b':>4} {'TQ bits':>8} {'v1.4 Q':>6} {'v1.4 bits':>9} "
          f"{'K-MSE ratio':>12} {'|Δppl| ratio':>14} "
          f"{'top1 Δpp':>10} {'speed ratio':>12}")
    for b, Q in zip(TQ_B_VALUES, V14_Q_VALUES):
        tq_agg = next((a for a in aggregates if a["channel"] == f"tq_b{b}"), None)
        v14_agg = next((a for a in aggregates if a["channel"] == f"v14_Q{Q}"), None)
        if tq_agg is None or v14_agg is None:
            continue
        kmse_ratio = v14_agg["mean_k_mse_rel"] / max(tq_agg["mean_k_mse_rel"], 1e-12)
        dppl_ratio = v14_agg["mean_abs_delta_ppl"] / max(tq_agg["mean_abs_delta_ppl"], 1e-12)
        top1_diff_pp = (v14_agg["mean_top1_pair"] - tq_agg["mean_top1_pair"]) * 100
        speed_ratio = v14_agg["mean_decode_ms_per_M"] / max(tq_agg["mean_decode_ms_per_M"], 1e-6)
        print(f"{b:>4} {tq_agg['bits']:>8} {Q:>6} {v14_agg['bits']:>9} "
              f"{kmse_ratio:>12.3f} {dppl_ratio:>14.3f} "
              f"{top1_diff_pp:+10.3f} {speed_ratio:>12.3f}")

    out = {
        "model": args.model_path,
        "ctx_len": args.ctx_len,
        "n_eval": args.n_eval,
        "n_passages": len(passages_ids),
        "strict_gpu": True,
        "num_layers": num_layers,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "boundary_skip_layers": sorted(boundary),
        "raw_bits_per_token_per_head": raw_bits,
        "tq_b_values": TQ_B_VALUES,
        "v14_q_values": V14_Q_VALUES,
        "per_passage": per_passage,
        "aggregates": aggregates,
    }
    out_path = args.out_dir / f"{args.model_name}_multimodel.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] written → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
