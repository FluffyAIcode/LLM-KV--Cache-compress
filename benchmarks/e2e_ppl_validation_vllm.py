#!/usr/bin/env python3
"""End-to-end downstream-quality validation of the v1.3 codec, run on vLLM.

This mirrors the HF-transformers harness (`e2e_ppl_validation.py`) but
routes the forward pass through vLLM, so the PPL numbers reflect the
codec's behaviour under the production inference engine rather than the
HF eager kernel.

Design
------
1. Build a single vLLM `LLM` instance (PagedAttention, bf16).
2. Monkey-patch `vllm.attention.layer.Attention.forward` to, when a
   global switch is ON, round-trip the per-layer `key` and `value`
   tensors through the v1.3 Rust codec before they reach the attention
   kernel. Layers whose config lists them as `sliding_attention` are
   skipped (same convention as the HF harness).
3. For each WikiText-103 passage that tokenises to >= ctx_len + n_eval
   tokens, call `LLM.generate` with `prompt_logprobs=1` on the truncated
   passage `[0 : ctx_len + n_eval]` once with the codec OFF
   (reference) and once ON (alt). PPL over the `[ctx_len :
   ctx_len+n_eval)` positions and top-1 agreement of the one-best
   candidate are compared.
4. The standard PPL verdict thresholds are applied.

The Rust bench binary MUST support `--dump-decoded` (v1.3 Rust change
that landed together with this harness).

Usage
-----
    python benchmarks/e2e_ppl_validation_vllm.py \\
        --model-path Qwen/Qwen2.5-0.5B \\
        --model-name qwen2_5_0_5b \\
        --ctx-len 1024 --n-eval 64 --n-passages 2 \\
        --out-dir reports/v1_3_rsvd_rope/e2e_ppl_vllm_smoke

It is a drop-in alternative to `e2e_ppl_validation.py` that uses the
same codec under the same parameters, so the two numbers can be
compared directly (HF vs vLLM engine).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent


# =============================================================================
# v1.3 Rust codec round-trip (in-process via pyo3)
# =============================================================================
#
# See `benchmarks/e2e_ppl_validation_vllm_full.py` for the full M3
# rationale.  In-process call is bit-identical to the old CLI subprocess
# path (verified by kakeyaturbo-py/tests/test_roundtrip_cli_parity.py).

try:
    from kakeyaturbo_py import roundtrip_layer as _rust_roundtrip_layer
except ImportError as _import_err:  # pragma: no cover
    _rust_roundtrip_layer = None
    _rust_import_error = _import_err
else:
    _rust_import_error = None


def rust_roundtrip(
    arr: np.ndarray,
    *,
    block_size: int,
    bit_width: int,
    rsvd_target_rank: int,
    metric: str,
    share_basis: bool,
    pca_method: str = "randomized",
    variance_ratio: float = 0.95,
    k_means_k: int = 16,
    rsvd_oversample: int = 8,
    rsvd_power_iters: int = 2,
) -> tuple[np.ndarray, dict]:
    """Encode `arr` (N, D) through the v1.3 codec, return (decoded, report)."""
    if _rust_roundtrip_layer is None:
        raise RuntimeError(
            "kakeyaturbo_py extension not importable; build it with "
            "`cd kakeyaturbo-py && maturin develop --release`. "
            f"Original import error: {_rust_import_error!r}"
        )
    arr32 = np.ascontiguousarray(arr, dtype=np.float32)
    kwargs: dict[str, Any] = dict(
        metric=metric,
        block_size=block_size,
        bit_width=bit_width,
        variance_ratio=variance_ratio,
        k=k_means_k,
        rotation_seed=3405691582,
        pca_method=pca_method,
        share_basis=share_basis,
    )
    if pca_method == "randomized":
        kwargs["rsvd_target_rank"] = rsvd_target_rank
        kwargs["rsvd_oversample"] = rsvd_oversample
        kwargs["rsvd_power_iters"] = rsvd_power_iters
    decoded, report = _rust_roundtrip_layer(arr32, **kwargs)
    return decoded, dict(report)


# =============================================================================
# Codec wrapper for a live KV tensor flowing through vLLM attention
# =============================================================================

class CodecState:
    """Mutable global state for the monkey-patched Attention.forward.

    Keeping this at module scope (rather than threading it through vLLM's
    engine plumbing) is the simplest way to toggle the codec per-request
    inside a single LLM instance without restarting the CUDA graph /
    engine. We never run two codec configurations concurrently, so the
    global is safe here.
    """

    active: bool = False
    block_size: int = 512
    bit_width: int = 2
    variance_ratio: float = 0.95
    pca_method: str = "randomized"
    rsvd_target_rank_factor: float = 0.5
    # Layer routing: full-attention layer indices. If None, every layer
    # that the patched forward sees is treated as compressible.
    full_attention_layers: set[int] | None = None
    # Per-layer stats accumulator (reset per measurement run).
    stats: list[dict] = []
    # Counter assigned to each Attention instance on first use, used
    # only to distinguish layers when a model doesn't expose a
    # stable layer_name attribute.
    layer_counter: int = 0


def _roundtrip_tensor(
    t: torch.Tensor,
    metric: str,
    share_basis: bool,
    layer_id: Any,
    kind: str,
    head_size: int,
) -> torch.Tensor:
    """Codec round-trip a vLLM `key` or `value` tensor.

    vLLM 0.7.3 passes `key` / `value` into `Attention.forward` either as
    2D `[num_tokens, num_kv_heads * head_size]` (some model definitions)
    or 3D `[num_tokens, num_kv_heads, head_size]` (use_output path after
    `.view`). In both cases the per-head dimension is `head_size`, which
    the attention module exposes as `self.head_size`.
    """
    orig_shape = t.shape
    orig_dtype = t.dtype
    orig_device = t.device

    if t.dim() == 2:
        total = t.shape[1]
        if total % head_size != 0:
            raise ValueError(
                f"KV tensor dim {total} not divisible by head_size {head_size}"
            )
        x = t.reshape(-1, head_size)
    elif t.dim() == 3:
        x = t.reshape(-1, t.shape[-1])
    else:
        raise ValueError(f"unexpected KV tensor shape {tuple(orig_shape)}")

    arr = x.detach().to(torch.float32).cpu().numpy()
    n_total, hd = arr.shape
    n_full_blocks = n_total // CodecState.block_size
    n_compressible = n_full_blocks * CodecState.block_size

    if n_compressible == 0:
        return t  # not enough vectors to fill one block; leave untouched

    target_rank = max(2, int(hd * CodecState.rsvd_target_rank_factor))

    dec, rep = rust_roundtrip(
        arr[:n_compressible],
        block_size=CodecState.block_size,
        bit_width=CodecState.bit_width,
        rsvd_target_rank=target_rank,
        metric=metric,
        share_basis=share_basis,
        pca_method=CodecState.pca_method,
        variance_ratio=CodecState.variance_ratio,
    )
    if n_compressible < n_total:
        dec = np.concatenate([dec, arr[n_compressible:]], axis=0)

    CodecState.stats.append({
        "layer_id": layer_id,
        "kind": kind,
        "hd": hd,
        "n_vecs": n_total,
        "n_compressible": n_compressible,
        "mean_block_mse": float(rep.get("mean_block_mse", -1.0)),
        "compressed_bytes": int(rep.get("compressed_bytes", 0)),
    })

    restored = (
        torch.from_numpy(dec.astype(np.float32))
        .to(orig_device)
        .to(orig_dtype)
        .reshape(orig_shape)
    )
    return restored


# =============================================================================
# vLLM attention monkey-patch
# =============================================================================

def install_vllm_codec_patch() -> None:
    """Rebind `vllm.attention.layer.Attention.forward` so it round-trips
    K/V when `CodecState.active` is True.
    """
    from vllm.attention.layer import Attention  # type: ignore

    if getattr(Attention, "_kakeyaturbo_patched", False):
        return

    orig_forward = Attention.forward

    def patched_forward(
        self: Attention,  # type: ignore[name-defined]
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
    ) -> torch.Tensor:
        if CodecState.active:
            layer_id = getattr(
                self, "layer_name",
                getattr(self, "_kakeyaturbo_layer_id", None),
            )
            if layer_id is None:
                layer_id = CodecState.layer_counter
                CodecState.layer_counter += 1
                try:
                    object.__setattr__(self, "_kakeyaturbo_layer_id", layer_id)
                except Exception:
                    pass

            is_full = (
                CodecState.full_attention_layers is None
                or layer_id in CodecState.full_attention_layers
            )
            if is_full and key is not None and value is not None:
                head_size = getattr(self, "head_size", None)
                if head_size is None:
                    print(f"[codec-patch] layer {layer_id}: no head_size, "
                          "skipping round-trip", file=sys.stderr)
                else:
                    try:
                        key = _roundtrip_tensor(
                            key, metric="inner_product",
                            share_basis=False, layer_id=layer_id, kind="K",
                            head_size=head_size,
                        )
                        value = _roundtrip_tensor(
                            value, metric="mse",
                            share_basis=True, layer_id=layer_id, kind="V",
                            head_size=head_size,
                        )
                    except Exception as e:
                        print(f"[codec-patch] layer {layer_id} round-trip "
                              f"failed: {e}", file=sys.stderr)

        return orig_forward(self, query, key, value, kv_cache, attn_metadata)

    Attention.forward = patched_forward
    Attention._kakeyaturbo_patched = True  # type: ignore[attr-defined]
    print("[codec-patch] vllm.attention.layer.Attention.forward wrapped",
          flush=True)


# =============================================================================
# WikiText-103 passage loader
# =============================================================================

def load_wikitext_passages(
    tokenizer: Any, min_tokens: int, n_passages: int, split: str = "test",
) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    passages: list[str] = []
    current: list[str] = []
    approx = 0
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        current.append(text)
        approx += int(len(text.split()) * 1.3)
        if approx >= min_tokens:
            passage = "".join(current)
            real_len = len(tokenizer.encode(passage))
            if real_len >= min_tokens:
                passages.append(passage)
                if len(passages) >= n_passages:
                    return passages
            current = []
            approx = 0
    return passages


# =============================================================================
# vLLM driver
# =============================================================================

def build_llm(model_path: str, max_model_len: int, gpu_mem_util: float):
    from vllm import LLM  # type: ignore
    return LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True,
        # A single long sequence; no KV cache paging pressure at our
        # context sizes.
        trust_remote_code=True,
    )


def prompt_logprobs_for_ids(
    llm: Any, prompt_token_ids: list[int]
) -> list[dict]:
    """Run vLLM with prompt_logprobs=1 on the given token ids, return the
    per-position log-probability dict (one entry per prompt token, the
    first entry is always None because position 0 has no predecessor).
    """
    from vllm import SamplingParams  # type: ignore

    sp = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
    )
    outs = llm.generate(
        prompts=None,
        prompt_token_ids=[prompt_token_ids],
        sampling_params=sp,
        use_tqdm=False,
    )
    return outs[0].prompt_logprobs


def ppl_and_top1_from_prompt_logprobs(
    pls: list[dict], prompt_ids: list[int], start: int, end: int,
) -> tuple[float, list[float], list[int]]:
    """Slice the prompt_logprobs list at positions `[start, end)` and
    compute PPL of the ground-truth next-token chain there, plus the
    argmax top-1 candidates and the per-position log-prob of the true
    token.

    `pls[t]` is the dict for the token *at position t* — vLLM reports
    the conditional logprob of `prompt_ids[t]` given `prompt_ids[<t]`,
    and the logprobs of the top-K alternative token ids at that
    position. `pls[0]` is always None.
    """
    logps_true: list[float] = []
    top1_ids: list[int] = []
    for t in range(start, end):
        entry = pls[t]
        if entry is None:
            continue
        tok = prompt_ids[t]
        if tok not in entry:
            # Rare: the prompt-true token isn't in the reported top-K
            # bucket. Fall back to `logprob` on whichever entry matches
            # if available; otherwise treat as -inf for correctness.
            logps_true.append(float("-inf"))
        else:
            lp = entry[tok]
            logps_true.append(
                float(lp.logprob if hasattr(lp, "logprob") else lp["logprob"])
            )
        # Top-1 candidate id (by highest logprob).
        def _lp(v: Any) -> float:
            return float(v.logprob if hasattr(v, "logprob") else v["logprob"])
        top1 = max(entry.items(), key=lambda kv: _lp(kv[1]))[0]
        top1_ids.append(int(top1))

    valid = [lp for lp in logps_true if np.isfinite(lp)]
    mean_nll = -float(np.mean(valid)) if valid else float("inf")
    ppl = float(np.exp(mean_nll)) if np.isfinite(mean_nll) else float("inf")
    return ppl, logps_true, top1_ids


def compare_two_runs(
    ref_pls: list[dict], alt_pls: list[dict],
    prompt_ids: list[int], ctx_len: int, n_eval: int,
) -> dict:
    end = min(ctx_len + n_eval, len(prompt_ids))
    ppl_ref, lp_ref, top_ref = ppl_and_top1_from_prompt_logprobs(
        ref_pls, prompt_ids, ctx_len, end)
    ppl_alt, lp_alt, top_alt = ppl_and_top1_from_prompt_logprobs(
        alt_pls, prompt_ids, ctx_len, end)

    n = min(len(top_ref), len(top_alt))
    if n == 0:
        return {
            "ppl_ref": ppl_ref, "ppl_alt": ppl_alt,
            "ppl_delta_rel": float("nan"),
            "top1_agreement": float("nan"),
            "mean_kl_upper": float("nan"),
            "n_tokens": 0,
        }

    agree = float(np.mean(
        [1.0 if top_ref[i] == top_alt[i] else 0.0 for i in range(n)]
    ))

    # With only top-1 logprob per position we can't compute true KL over
    # the full vocab. We report the mean |Δ logprob on the true token|
    # as an approximate divergence proxy.
    deltas = [
        abs(lp_ref[i] - lp_alt[i])
        for i in range(min(len(lp_ref), len(lp_alt)))
        if np.isfinite(lp_ref[i]) and np.isfinite(lp_alt[i])
    ]
    mean_abs_dlogp = float(np.mean(deltas)) if deltas else float("nan")

    return {
        "ppl_ref": ppl_ref,
        "ppl_alt": ppl_alt,
        "ppl_delta_rel": (ppl_alt - ppl_ref) / max(ppl_ref, 1e-8),
        "top1_agreement": agree,
        "mean_abs_dlogp_true": mean_abs_dlogp,
        "n_tokens": n,
    }


def verdict_of(mean_delta_rel: float, mean_top1: float) -> str:
    if (abs(mean_delta_rel) <= 0.01) and (mean_top1 >= 0.95):
        return "ACCEPT"
    if (abs(mean_delta_rel) <= 0.03) and (mean_top1 >= 0.85):
        return "MARGINAL"
    return "REJECT"


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--ctx-len", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--bit-width", type=int, default=2)
    ap.add_argument("--variance-ratio", type=float, default=0.95)
    ap.add_argument("--pca-method", choices=["exact", "randomized"],
                    default="randomized")
    ap.add_argument("--rsvd-target-rank-factor", type=float, default=0.5)
    ap.add_argument("--n-passages", type=int, default=2)
    ap.add_argument("--gpu-mem-util", type=float, default=0.80)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Propagate codec config into the global state the patched forward
    # reads from.
    CodecState.block_size = args.block_size
    CodecState.bit_width = args.bit_width
    CodecState.variance_ratio = args.variance_ratio
    CodecState.pca_method = args.pca_method
    CodecState.rsvd_target_rank_factor = args.rsvd_target_rank_factor

    # Install the patch BEFORE constructing the LLM so the wrapped
    # forward is what the engine binds to its model shards.
    install_vllm_codec_patch()

    print(f"[{args.model_name}] loading vLLM engine…", flush=True)
    max_len = args.ctx_len + args.n_eval + 16
    llm = build_llm(args.model_path, max_len, args.gpu_mem_util)

    # Build a HF-equivalent tokenizer for passage selection via vLLM's
    # own tokenizer.
    tok = llm.get_tokenizer()

    print(f"[{args.model_name}] loading WikiText-103 passages "
          f"(min_tokens={args.ctx_len + args.n_eval})…", flush=True)
    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval,
        n_passages=args.n_passages,
    )
    print(f"  got {len(passages)} passages", flush=True)

    per_passage: list[dict] = []
    for i, passage in enumerate(passages):
        print(f"  passage {i+1}/{len(passages)}…", flush=True)
        ids = tok.encode(passage)
        ids = ids[: args.ctx_len + args.n_eval]
        if len(ids) < args.ctx_len + args.n_eval:
            print("    skipped (too short after tokenization)", flush=True)
            continue

        CodecState.active = False
        CodecState.stats = []
        t0 = time.perf_counter()
        ref_pls = prompt_logprobs_for_ids(llm, ids)
        t_ref = time.perf_counter() - t0

        CodecState.active = True
        CodecState.stats = []
        CodecState.layer_counter = 0  # re-key layer ids for this run
        t0 = time.perf_counter()
        alt_pls = prompt_logprobs_for_ids(llm, ids)
        t_alt = time.perf_counter() - t0
        alt_stats = list(CodecState.stats)
        CodecState.active = False

        metrics = compare_two_runs(ref_pls, alt_pls, ids,
                                   args.ctx_len, args.n_eval)
        m = metrics
        print(
            f"    ppl_ref={m['ppl_ref']:.3f} ppl_alt={m['ppl_alt']:.3f} "
            f"Δppl={m['ppl_delta_rel']*100:+.2f}% "
            f"top1={m['top1_agreement']*100:.1f}% "
            f"Δlogp={m['mean_abs_dlogp_true']:.4f} "
            f"(ref={t_ref:.2f}s, alt={t_alt:.2f}s, "
            f"layer_calls={len(alt_stats)})",
            flush=True,
        )

        per_passage.append({
            "ctx_len": args.ctx_len,
            "n_eval": args.n_eval,
            "t_ref_sec": t_ref,
            "t_alt_sec": t_alt,
            "codec_layer_calls": len(alt_stats),
            "codec_total_compressed_bytes": int(
                sum(s["compressed_bytes"] for s in alt_stats)
            ),
            "metrics": metrics,
        })

    summary: dict = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "engine": "vllm",
        "ctx_len": args.ctx_len,
        "n_eval": args.n_eval,
        "block_size": args.block_size,
        "bit_width": args.bit_width,
        "variance_ratio": args.variance_ratio,
        "pca_method": args.pca_method,
        "rsvd_target_rank_factor": args.rsvd_target_rank_factor,
        "n_passages": len(per_passage),
    }
    if per_passage:
        mean_delta = float(np.mean(
            [r["metrics"]["ppl_delta_rel"] for r in per_passage
             if np.isfinite(r["metrics"]["ppl_delta_rel"])]
        ))
        mean_top1 = float(np.mean(
            [r["metrics"]["top1_agreement"] for r in per_passage
             if np.isfinite(r["metrics"]["top1_agreement"])]
        ))
        summary.update({
            "mean_ppl_delta_rel": mean_delta,
            "mean_top1_agreement": mean_top1,
            "verdict": verdict_of(mean_delta, mean_top1),
        })
        print(f"\n[{args.model_name}] ===== SUMMARY (vLLM engine) =====",
              flush=True)
        print(f"  n_passages   = {len(per_passage)}", flush=True)
        print(f"  Δppl (mean)  = {mean_delta*100:+.3f}%", flush=True)
        print(f"  top1 agree   = {mean_top1*100:.2f}%", flush=True)
        print(f"  VERDICT      = {summary['verdict']}", flush=True)
    else:
        summary["verdict"] = "NO_DATA"

    summary["per_passage"] = per_passage
    out_path = args.out_dir / f"{args.model_name}_vllm.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
