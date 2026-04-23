"""128k KV storage report for v1.4 KakeyaLattice — KV (both K AND V) compression.

Produces the template-style table:

  Model                   | 128k Baseline KV | 128k Kakeya KV (bf16 store) | Total ratio

Semantics:
  * Baseline KV   = raw bf16 K + raw bf16 V @ 128k tokens @ every layer
  * Kakeya KV     = v1.4-compressed K + v1.4-compressed V @ 128k tokens
                   (with boundary layers kept bf16 for stability)
  * Total ratio   = Baseline / Kakeya

Also measures |Δppl| and top-1 pair agreement on WikiText-103 to CONFIRM
the K+V compression operating point is safe.  Strict-GPU, real vLLM.

CLI:

  export VLLM_ENABLE_V1_MULTIPROCESSING=0 KAKEYA_SNAPSHOT_QWEN3=1
  python benchmarks/multimodel_v14_kv_128k_report.py \
      --model-path Qwen/Qwen3-4B --model-name qwen3_4b \
      --q-values 10,38,152 \
      --ctx-len 2048 --n-eval 64 --n-passages 4 \
      --out-dir reports/v1_4_release/kv_128k_report

No mock / no simplification / no fallback / no overfit.  All bits and
bytes are measured from the actual codec output; the only hypothetical
axis is context length (128k tokens), which is a linear scaling of the
per-token cost we directly measure at ctx_len=2048.
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


def raw_bits_per_token_per_head(head_dim: int) -> int:
    return 16 * head_dim


def recode_v14_gpu(
    X: torch.Tensor, q_range: int,
) -> tuple[torch.Tensor, int]:
    """v1.4 KakeyaLattice encode-then-decode on a per-head tensor (N_tok, H, D).

    Returns (reconstructed_tensor, bits_per_token_per_head).
    """
    assert X.is_cuda and X.dtype == torch.float32
    from kakeyaturbo_py import V14KakeyaZamirLatticeGPU
    D = X.shape[-1]
    if D % 4 != 0:
        raise ValueError(
            f"v1.4 requires head_dim divisible by 4, got {D}. "
            f"No fallback by design.",
        )
    cb = V14KakeyaZamirLatticeGPU(D=D, q_range=q_range, device=X.device)
    return cb.roundtrip(X), cb.bits_per_token_per_head


def _sylvester_hadamard_normalised(D: int, device) -> torch.Tensor:
    """Sylvester Hadamard divided by √D (self-inverse at ortho)."""
    assert (D & (D - 1)) == 0, f"Hadamard requires D=power of 2, got {D}"
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], dim=0)
    return H / math.sqrt(D)


def recode_tq_gpu(
    X: torch.Tensor, bits_per_coord: int,
) -> tuple[torch.Tensor, int]:
    """TurboQuant (Hadamard + per-vector qmax + scalar uniform quantise),
    applied independently to each head-vector.  Same algorithm as the
    multimodel_v14_vs_tq comparison harness — kept identical here so that
    the two reports are strictly apples-to-apples.

    Returns (reconstructed_tensor, bits_per_token_per_head).
    """
    assert X.is_cuda and X.dtype == torch.float32
    N_tok, H_heads, D = X.shape
    flat = X.reshape(-1, D)
    eps = torch.finfo(torch.float32).eps

    norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
    norms_f16 = norms.to(torch.float16).to(torch.float32)
    unit = flat / norms

    Hmat = _sylvester_hadamard_normalised(D, X.device)
    y = unit @ Hmat
    qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
    qmax_f16 = qmax.to(torch.float16).to(torch.float32)
    qs = (1 << (bits_per_coord - 1)) - 1
    scale = qmax_f16 / float(qs)
    q = torch.round(y / scale).clamp(-qs, qs) * scale
    unit_hat = q @ Hmat
    X_hat = (unit_hat * norms_f16).reshape(N_tok, H_heads, D)
    bits = D * bits_per_coord + 32
    return X_hat, bits


def compute_rel_mse_gpu(
    X: torch.Tensor, X_hat: torch.Tensor,
) -> tuple[float, float]:
    diff = X_hat - X
    sq = (diff * diff).sum(dim=-1)
    norm_sq = (X * X).sum(dim=-1)
    rel = float((sq.mean() / norm_sq.mean().clamp(min=1e-12)).item())
    dot = (X_hat * X).sum(dim=-1)
    n1 = (X * X).sum(dim=-1).sqrt().clamp(min=1e-12)
    n2 = (X_hat * X_hat).sum(dim=-1).sqrt().clamp(min=1e-12)
    cos_mean = float((dot / (n1 * n2)).mean().item())
    return rel, cos_mean


def load_wikitext_passages(tok: Any, min_tokens: int, n_passages: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
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


def prompt_logprobs_for_ids(llm: Any, ids: list[int]) -> list[Any]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        sampling_params=sp, use_tqdm=False,
    )
    return out[0].prompt_logprobs


def ppl_and_top1(
    pls: list[Any], ids: list[int], start: int, end: int,
) -> tuple[float, list[int]]:
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


def default_boundary_for_model(num_layers: int) -> set[int]:
    """Same policy as multimodel_v14_vs_tq: first 2 + last 2 stay bf16.

    This is the same constant across every model in the family report,
    so the comparison is honest.  For the K-only sweep this boundary
    was shown to preserve |Δppl| < 1 % at Q=152 for Qwen3-4B.
    """
    return set(list(range(2)) + list(range(num_layers - 2, num_layers)))


def fmt_bytes(b: float) -> str:
    gi = b / 1024**3
    mi = b / 1024**2
    if gi >= 1.0:
        return f"{gi:.2f} GiB"
    return f"{mi:.0f} MiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--q-values", type=str, default="10,38,152",
                    help="Comma-separated v1.4 Q_range values to sweep.")
    ap.add_argument("--tq-b-values", type=str, default="4,6,8",
                    help="Comma-separated TQ bits_per_coord values to sweep."
                         "  Each TQ channel compresses both K and V.")
    ap.add_argument("--ctx-len",    type=int, default=2048)
    ap.add_argument("--n-eval",     type=int, default=64)
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--report-ctx-tokens", type=int, default=128 * 1024,
                    help="Context length for the report table (template shows 128k)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    Q_VALUES = [int(x) for x in args.q_values.split(",") if x.strip()]
    TQ_B_VALUES = [int(x) for x in args.tq_b_values.split(",") if x.strip()]
    print(f"[config] v1.4 Q sweep: {Q_VALUES}", flush=True)
    print(f"[config] TQ b sweep:   {TQ_B_VALUES}", flush=True)

    from vllm import LLM
    from transformers import AutoTokenizer
    from kakeya_v1_3_ppl.snapshot_hook import HookState

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

    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval,
        n_passages=args.n_passages,
    )
    passages_ids = [tok.encode(p)[: args.ctx_len + args.n_eval]
                    for p in passages
                    if len(tok.encode(p)) >= args.ctx_len + args.n_eval]
    if not passages_ids:
        print("[ERROR] No passages long enough.", flush=True)
        return 1

    # Dry run to pull model config.
    HookState.phase = "capture"
    HookState.captured = {}
    _ = prompt_logprobs_for_ids(llm, passages_ids[0][: args.ctx_len])
    HookState.phase = "off"
    cap_dry = dict(HookState.captured)
    num_layers = max(cap_dry.keys()) + 1 if cap_dry else 0
    if num_layers == 0:
        print(f"[ERROR] Snapshot hook did not fire for {args.model_path}.",
              flush=True)
        return 1
    head_dim = cap_dry[0]['K'].shape[-1]
    num_kv_heads = cap_dry[0]['K'].shape[1]
    print(f"[model] {args.model_path}", flush=True)
    print(f"[model] L={num_layers} hd={head_dim} kv_h={num_kv_heads}", flush=True)

    if head_dim % 4 != 0:
        print(f"[FAIL] head_dim {head_dim} not divisible by 4; "
              f"v1.4 cannot run.  Aborting (no fallback).", flush=True)
        return 2

    del cap_dry
    HookState.captured = {}
    torch.cuda.empty_cache()

    boundary = default_boundary_for_model(num_layers)
    n_boundary = len(boundary)
    n_compressed = num_layers - n_boundary

    raw_bits_per_h = raw_bits_per_token_per_head(head_dim)
    raw_bytes_per_h = raw_bits_per_h // 8  # bf16 per-head per-token bytes

    per_passage: list[dict] = []

    # Sweep order: Q -> passage (passage outer loop lets us reuse capture)
    for pi, ids in enumerate(passages_ids):
        print(f"\n=== passage {pi + 1}/{len(passages_ids)} ===", flush=True)
        HookState.phase = "capture"
        HookState.captured = {}
        t0 = time.perf_counter()
        ref_pls = prompt_logprobs_for_ids(llm, ids)
        t_ref = time.perf_counter() - t0
        HookState.phase = "off"
        captured = dict(HookState.captured)
        n_tokens = captured[next(iter(captured))]['K'].shape[0]
        for lid, kv in captured.items():
            assert kv["K"].is_cuda and kv["V"].is_cuda

        ppl_ref, ref_top1 = ppl_and_top1(
            ref_pls, ids, args.ctx_len, args.ctx_len + args.n_eval,
        )
        print(f"  [capture] layers={len(captured)} tokens={n_tokens} "
              f"ref_ppl={ppl_ref:.3f} in {t_ref:.2f}s", flush=True)

        # bf16 passthrough reference (should be Δppl ≈ 0).  This confirms
        # the replace pipeline isn't perturbing anything by itself.
        channels = (
            [("bf16_pass", "bf16", 0)]
            + [(f"v14_Q{Q}_K+V",  "v14_kv", Q) for Q in Q_VALUES]
            + [(f"tq_b{b}_K+V",   "tq_kv",  b) for b in TQ_B_VALUES]
        )
        for ch_label, ch_kind, ch_param in channels:
            replacements: dict[int, dict[str, torch.Tensor]] = {}
            k_mse_sum = v_mse_sum = k_cos_sum = v_cos_sum = 0.0
            mse_count = 0
            k_bits: int | None = None
            v_bits: int | None = None
            t0 = time.perf_counter()
            for lid, kv in captured.items():
                K_g: torch.Tensor = kv["K"]
                V_g: torch.Tensor = kv["V"]
                if lid in boundary:
                    replacements[lid] = {"K": K_g, "V": V_g}
                    continue
                if ch_kind == "bf16":
                    # bf16 round-trip to measure FA bf16 reproducibility floor.
                    K_hat = K_g.to(torch.bfloat16).to(torch.float32)
                    V_hat = V_g.to(torch.bfloat16).to(torch.float32)
                    k_bits = raw_bits_per_h
                    v_bits = raw_bits_per_h
                elif ch_kind == "v14_kv":
                    K_hat, kb = recode_v14_gpu(K_g, q_range=ch_param)
                    V_hat, vb = recode_v14_gpu(V_g, q_range=ch_param)
                    k_bits, v_bits = kb, vb
                elif ch_kind == "tq_kv":
                    K_hat, kb = recode_tq_gpu(K_g, bits_per_coord=ch_param)
                    V_hat, vb = recode_tq_gpu(V_g, bits_per_coord=ch_param)
                    k_bits, v_bits = kb, vb
                else:
                    raise RuntimeError(ch_kind)
                k_rel, k_cos = compute_rel_mse_gpu(K_g, K_hat)
                v_rel, v_cos = compute_rel_mse_gpu(V_g, V_hat)
                k_mse_sum += k_rel
                v_mse_sum += v_rel
                k_cos_sum += k_cos
                v_cos_sum += v_cos
                mse_count += 1
                replacements[lid] = {"K": K_hat, "V": V_hat}
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

            n_fired = sum(HookState.replace_fired.values())
            n_mismatch = sum(len(v) for v in HookState.replace_shape_mismatch.values())
            n_missing = sum(HookState.replace_missing.values())
            if ch_kind != "bf16":
                expected = num_layers - n_boundary
                if n_fired < expected:
                    print(f"  [{ch_label}] FATAL: {n_fired} fires < "
                          f"expected {expected} (mismatch={n_mismatch} "
                          f"missing={n_missing}).  Skipping.", flush=True)
                    per_passage.append({
                        "passage": pi, "channel": ch_label,
                        "fatal": "silent passthrough",
                    })
                    del replacements
                    torch.cuda.empty_cache()
                    continue

            ppl_alt, alt_top1 = ppl_and_top1(
                alt_pls, ids, args.ctx_len, args.ctx_len + args.n_eval,
            )
            top1_pair = float(
                sum(1 for a, r in zip(alt_top1, ref_top1)
                    if a == r and a != -1)
                / max(len(alt_top1), 1)
            )
            delta_ppl = (ppl_alt - ppl_ref) / max(ppl_ref, 1e-9)

            # per-token storage bytes used by this channel
            # boundary layer    = L_bdry * kv_h * (raw_K + raw_V)
            # compressed layer  = L_comp * kv_h * (k_bits/8 + v_bits/8)
            assert k_bits is not None and v_bits is not None
            per_tok_bytes = (
                n_boundary * num_kv_heads * (raw_bytes_per_h * 2)
                + n_compressed * num_kv_heads * (k_bits // 8 + v_bits // 8)
            )
            baseline_per_tok_bytes = num_layers * num_kv_heads * (raw_bytes_per_h * 2)

            total_bytes_128k   = per_tok_bytes * args.report_ctx_tokens
            baseline_bytes_128k = baseline_per_tok_bytes * args.report_ctx_tokens
            total_ratio = baseline_bytes_128k / total_bytes_128k

            k_mse_mean = k_mse_sum / max(mse_count, 1)
            v_mse_mean = v_mse_sum / max(mse_count, 1)
            k_cos_mean = k_cos_sum / max(mse_count, 1)
            v_cos_mean = v_cos_sum / max(mse_count, 1)

            print(
                f"  [{ch_label:<12}] K{k_bits}b V{v_bits}b  "
                f"Δppl={delta_ppl * 100:+7.3f}%  "
                f"top1={top1_pair * 100:6.2f}%  "
                f"K-MSE={k_mse_mean:.2e}  V-MSE={v_mse_mean:.2e}  "
                f"CR={total_ratio:.2f}×  fires={n_fired}",
                flush=True,
            )

            per_passage.append({
                "passage": pi, "channel": ch_label, "kind": ch_kind,
                "q_range": ch_param if ch_kind == "v14_kv" else None,
                "k_bits_per_tok_per_head": k_bits,
                "v_bits_per_tok_per_head": v_bits,
                "baseline_bytes_128k": baseline_bytes_128k,
                "kakeya_bytes_128k": total_bytes_128k,
                "total_ratio_128k": total_ratio,
                "ppl_ref": ppl_ref, "ppl_alt": ppl_alt,
                "delta_ppl": delta_ppl,
                "top1_pair": top1_pair,
                "k_mse_rel": k_mse_mean, "v_mse_rel": v_mse_mean,
                "k_cos_mean": k_cos_mean, "v_cos_mean": v_cos_mean,
                "t_codec": t_codec, "t_alt": t_alt,
                "fire_count": n_fired,
                "mismatch_count": n_mismatch,
                "missing_count": n_missing,
            })
            del replacements
            torch.cuda.empty_cache()

        del captured
        HookState.captured = {}
        torch.cuda.empty_cache()

    # Aggregate per channel (mean over passages).
    import statistics as _st
    by_ch: dict[str, list[dict]] = {}
    for r in per_passage:
        if "fatal" in r:
            continue
        by_ch.setdefault(r["channel"], []).append(r)

    agg_rows: list[dict] = []
    for ch, rs in by_ch.items():
        agg_rows.append({
            "channel": ch,
            "kind": rs[0]["kind"],
            "q_range": rs[0]["q_range"],
            "k_bits": rs[0]["k_bits_per_tok_per_head"],
            "v_bits": rs[0]["v_bits_per_tok_per_head"],
            "baseline_bytes_128k": rs[0]["baseline_bytes_128k"],
            "kakeya_bytes_128k":   rs[0]["kakeya_bytes_128k"],
            "total_ratio_128k":    rs[0]["total_ratio_128k"],
            "mean_delta_ppl":      _st.fmean(r["delta_ppl"] for r in rs),
            "mean_abs_delta_ppl":  _st.fmean(abs(r["delta_ppl"]) for r in rs),
            "mean_top1_pair":      _st.fmean(r["top1_pair"] for r in rs),
            "mean_k_mse":          _st.fmean(r["k_mse_rel"] for r in rs),
            "mean_v_mse":          _st.fmean(r["v_mse_rel"] for r in rs),
        })

    # Print the template-style table.
    print()
    print("=" * 100)
    print(f"v1.4 KakeyaLattice — {args.report_ctx_tokens // 1024}k KV storage report")
    print(f"Model: {args.model_path}  (L={num_layers}, hd={head_dim}, kv_h={num_kv_heads})")
    print(f"Boundary skip: {sorted(boundary)}  (bf16-kept layers; codec applied to the other {n_compressed})")
    print("=" * 100)
    print(f"{'Channel':<14} {'128k Baseline KV':>20} {'128k Kakeya KV':>20} {'Total ratio':>13}  "
          f"{'|Δppl|':>8} {'top-1':>7} {'K-MSE':>10} {'V-MSE':>10}")
    print("-" * 113)
    for r in agg_rows:
        print(f"{r['channel']:<14} "
              f"{fmt_bytes(r['baseline_bytes_128k']):>20} "
              f"{fmt_bytes(r['kakeya_bytes_128k']):>20} "
              f"{r['total_ratio_128k']:>11.2f}×  "
              f"{r['mean_abs_delta_ppl'] * 100:7.3f}% "
              f"{r['mean_top1_pair'] * 100:6.2f}% "
              f"{r['mean_k_mse']:10.3e} "
              f"{r['mean_v_mse']:10.3e}")

    out = {
        "model": args.model_path,
        "model_name": args.model_name,
        "num_layers": num_layers,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "boundary_skip_layers": sorted(boundary),
        "raw_bits_per_token_per_head": raw_bits_per_h,
        "report_ctx_tokens": args.report_ctx_tokens,
        "ctx_len": args.ctx_len,
        "n_eval": args.n_eval,
        "n_passages": len(passages_ids),
        "q_values": Q_VALUES,
        "tq_b_values": TQ_B_VALUES,
        "per_passage": per_passage,
        "aggregates": agg_rows,
    }
    out_path = args.out_dir / f"{args.model_name}_kv_128k.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
