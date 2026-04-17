#!/usr/bin/env python3
"""
Real, end-to-end compression benchmark for the Kakeya KV cache on
Gemma 4 (google/gemma-4-E2B-it).

What this script does
---------------------
1. Loads the real Gemma 4 E2B model + tokenizer (BF16 on CPU by default).
2. Runs a single prefill pass on a long prompt twice:
     - once with the standard `DynamicCache` (ground-truth baseline)
     - once with `KakeyaKVCache`
   Both caches are populated with the same prefill.
3. Measures the exact byte footprint of each cache, layer by layer,
   split into sliding-window layers (untouched) and full-attention
   layers (the only layers the Kakeya codec actually compresses).
4. Generates a short continuation with each cache so we can check the
   Kakeya path is still producing coherent text on real weights.
5. Prints a JSON report with the concrete compression ratios.

Why the sliding-window layers are excluded from "gain"
------------------------------------------------------
Gemma 4's sliding-window layers have a hard cap at `sliding_window - 1`
tokens, so they are essentially O(1) memory regardless of context
length. Any KV-compression scheme, including this one, targets the
full-attention layers where memory grows linearly with context. The
report therefore breaks out full vs. sliding, and the headline gain is
computed on the full-attention layers plus on the total cache.

Byte-accounting conventions
---------------------------
- Baseline KV bytes = sum over layers of `keys.numel()*element_size()
  + values.numel()*element_size()`.
- Kakeya cache bytes = exact tail (same formula) plus, for every
  compressed block, the serialized skeleton tensors (basis, mean,
  t_dir, centers) plus the encoded tensors (seg_id, alpha, t,
  residual_vals, residual_idx). This is what you would have to keep in
  RAM (or write to disk) to later reconstruct the full KV tensor.
- All compressed-side tensors are float32 on CPU (that is how the
  codec currently stores them). The report shows both the raw ratio
  and a "dtype-matched" projection that accounts for the fact that an
  optimized implementation would keep them in the KV dtype (bf16).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, set_seed

from kakeya_kv_codec import (
    KakeyaCompressedBlock,
    KakeyaCompressedLayer,
    KakeyaKVCache,
    build_gemma4_kakeya_cache,
)


LONG_PROMPT_SEED = (
    "You are a careful technical writer. Please produce a long, self-contained "
    "explanation of how transformer key/value caches work during autoregressive "
    "decoding, why they grow linearly with the number of decoded tokens, why "
    "the memory pressure can dominate system throughput at large batch sizes, "
    "and what different compression strategies look like in practice, "
    "including quantization, low-rank projection, token eviction (H2O / Scissorhands), "
    "and learned codecs. Include concrete numerical examples throughout.\n\n"
)


# ---------------------------------------------------------------------------
# Byte accounting helpers
# ---------------------------------------------------------------------------


def tensor_bytes(t: torch.Tensor) -> int:
    if t is None:
        return 0
    return int(t.numel()) * int(t.element_size())


def kakeya_block_bytes(block: KakeyaCompressedBlock) -> Dict[str, int]:
    sk = block.skeleton
    en = block.encoded
    skeleton_bytes = (
        tensor_bytes(sk.basis)
        + tensor_bytes(sk.mean)
        + tensor_bytes(sk.t_dir)
        + tensor_bytes(sk.centers)
    )
    encoded_bytes = (
        tensor_bytes(en.seg_id)
        + tensor_bytes(en.alpha)
        + tensor_bytes(en.t)
        + tensor_bytes(en.residual_vals)
        + tensor_bytes(en.residual_idx)
    )
    return {
        "skeleton_bytes": skeleton_bytes,
        "encoded_bytes": encoded_bytes,
        "total_bytes": skeleton_bytes + encoded_bytes,
    }


def dtype_matched_projection(kakeya_total_bytes: int, exact_tail_bytes: int) -> int:
    """Project the compressed-side bytes to the KV dtype (bf16, i.e. 2B/el).

    The codec stores its tensors as float32 on CPU. If the implementation
    kept them in the KV dtype (bf16), the non-integer tensors would shrink
    by 2x. The integer index tensors are unchanged. We approximate this
    by halving the *compressed* bytes (excluding the exact tail, which is
    already in the KV dtype)."""
    compressed_only = max(kakeya_total_bytes - exact_tail_bytes, 0)
    return exact_tail_bytes + compressed_only // 2


@dataclass
class LayerReport:
    layer_idx: int
    layer_type: str
    is_sliding: bool
    seq_len: int
    head_dim: int
    baseline_bytes: int
    kakeya_bytes: int
    kakeya_compressed_blocks: int
    kakeya_skeleton_bytes: int
    kakeya_encoded_bytes: int
    kakeya_exact_tail_bytes: int


def baseline_layer_bytes(layer) -> Tuple[int, int, int]:
    k = getattr(layer, "keys", None)
    v = getattr(layer, "values", None)
    kb = tensor_bytes(k) if k is not None else 0
    vb = tensor_bytes(v) if v is not None else 0
    seq = layer.get_seq_length()
    head_dim = k.shape[-1] if (k is not None and k.ndim == 4) else 0
    return kb + vb, seq, head_dim


def kakeya_layer_bytes(layer: KakeyaCompressedLayer) -> Dict[str, int]:
    exact_tail = 0
    if layer.keys is not None:
        exact_tail += tensor_bytes(layer.keys)
    if layer.values is not None:
        exact_tail += tensor_bytes(layer.values)

    skeleton_b = 0
    encoded_b = 0
    for block in layer._compressed_key_blocks + layer._compressed_value_blocks:
        bb = kakeya_block_bytes(block)
        skeleton_b += bb["skeleton_bytes"]
        encoded_b += bb["encoded_bytes"]

    return {
        "exact_tail_bytes": exact_tail,
        "skeleton_bytes": skeleton_b,
        "encoded_bytes": encoded_b,
        "total_bytes": exact_tail + skeleton_b + encoded_b,
        "compressed_blocks": len(layer._compressed_key_blocks),
    }


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


def build_long_prompt(tokenizer, target_tokens: int) -> torch.Tensor:
    """Build a prompt that tokenizes to approximately `target_tokens` tokens."""
    text = LONG_PROMPT_SEED
    while True:
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        if ids.shape[-1] >= target_tokens:
            return ids[:, :target_tokens]
        text = text + LONG_PROMPT_SEED


@torch.inference_mode()
def run_prefill(model, input_ids: torch.Tensor, cache, chunk: int = 0) -> float:
    """Prefill with optional chunking to keep the attention activation memory bounded.

    With chunk=0 the whole prompt is fed in one forward pass (the default when the
    machine has enough RAM). With chunk>0 the prompt is fed chunk tokens at a time,
    updating the same cache incrementally. The resulting cache is byte-identical
    whether chunked or not, because the full-attention / sliding attention layers
    re-use the same `past_key_values` object across sub-calls."""
    start = time.perf_counter()
    if chunk <= 0 or input_ids.shape[-1] <= chunk:
        _ = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    else:
        n = input_ids.shape[-1]
        for start_i in range(0, n, chunk):
            end_i = min(start_i + chunk, n)
            _ = model(
                input_ids=input_ids[:, start_i:end_i],
                past_key_values=cache,
                use_cache=True,
            )
    return time.perf_counter() - start


@torch.inference_mode()
def run_generate(model, tokenizer, input_ids: torch.Tensor, cache, max_new_tokens: int) -> Tuple[str, float, int]:
    start = time.perf_counter()
    out = model.generate(
        input_ids=input_ids,
        past_key_values=cache,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, int) else tokenizer.eos_token_id[0],
    )
    elapsed = time.perf_counter() - start
    new_tokens = int(out.shape[-1] - input_ids.shape[-1])
    text = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=False)
    return text, elapsed, new_tokens


def layer_is_sliding(layer) -> bool:
    return bool(getattr(layer, "is_sliding", False))


def analytic_baseline_bytes_for_layer(
    layer_type: str, seq_len: int, config, dtype_bytes: int, bsz: int = 1
) -> Tuple[int, int]:
    """Return (bytes, head_dim) for a deterministic DynamicCache of `seq_len`
    tokens at this layer, matching how transformers would populate it.

    The baseline stores K and V as [bsz, num_kv_heads, seq_len, head_dim] in the
    model dtype, so bytes = 2 * bsz * n_kv * seq_len * head_dim * dtype_bytes.
    """
    text_cfg = config.get_text_config(decoder=True)
    if layer_type == "sliding_attention":
        head_dim = text_cfg.head_dim
        sw = getattr(text_cfg, "sliding_window", seq_len)
        if seq_len >= sw:
            effective_seq = sw - 1
        else:
            effective_seq = seq_len
    else:
        head_dim = getattr(text_cfg, "global_head_dim", None) or text_cfg.head_dim
        effective_seq = seq_len
    n_kv = text_cfg.num_key_value_heads
    return 2 * bsz * n_kv * effective_seq * head_dim * dtype_bytes, head_dim


def summarize_cache_analytic_baseline(
    kakeya_cache: KakeyaKVCache,
    layer_types: List[str],
    config,
    seq_len: int,
    dtype_bytes: int,
    bsz: int = 1,
) -> Dict[str, Any]:
    """Same shape of output as `summarize_cache`, but the baseline side is
    computed analytically instead of materialized as a real DynamicCache."""
    per_layer: List[Dict[str, Any]] = []
    bl_full = bl_slide = 0
    kv_full = kv_slide = 0
    kv_full_compressed = 0

    for idx, (kv_layer, lt) in enumerate(zip(kakeya_cache.layers, layer_types)):
        bl_bytes, head_dim = analytic_baseline_bytes_for_layer(
            lt, seq_len, config, dtype_bytes, bsz=bsz
        )
        sliding = lt == "sliding_attention"

        if isinstance(kv_layer, KakeyaCompressedLayer):
            kv_stats = kakeya_layer_bytes(kv_layer)
            kv_bytes_total = kv_stats["total_bytes"]
            kv_compressed_only = kv_stats["skeleton_bytes"] + kv_stats["encoded_bytes"]
            kv_blocks = kv_stats["compressed_blocks"]
            kv_tail = kv_stats["exact_tail_bytes"]
            kv_sk = kv_stats["skeleton_bytes"]
            kv_enc = kv_stats["encoded_bytes"]
            kv_seq = kv_layer.get_seq_length()
        else:
            kv_bytes_total, _, _ = baseline_layer_bytes(kv_layer)
            kv_compressed_only = 0
            kv_blocks = 0
            kv_tail = kv_bytes_total
            kv_sk = 0
            kv_enc = 0
            kv_seq = kv_layer.get_seq_length()

        per_layer.append({
            "layer_idx": idx,
            "layer_type": lt,
            "is_sliding": sliding,
            "baseline_seq_len": (
                seq_len if not sliding else (
                    (getattr(config.get_text_config(decoder=True), "sliding_window", seq_len) - 1)
                    if seq_len >= getattr(config.get_text_config(decoder=True), "sliding_window", seq_len)
                    else seq_len
                )
            ),
            "kakeya_seq_len": kv_seq,
            "head_dim": head_dim,
            "baseline_bytes": bl_bytes,
            "kakeya_bytes_total": kv_bytes_total,
            "kakeya_compressed_blocks": kv_blocks,
            "kakeya_skeleton_bytes": kv_sk,
            "kakeya_encoded_bytes": kv_enc,
            "kakeya_exact_tail_bytes": kv_tail,
            "layer_ratio_baseline_over_kakeya": (bl_bytes / kv_bytes_total) if kv_bytes_total > 0 else None,
        })

        if sliding:
            bl_slide += bl_bytes
            kv_slide += kv_bytes_total
        else:
            bl_full += bl_bytes
            kv_full += kv_bytes_total
            kv_full_compressed += kv_compressed_only

    bl_total = bl_full + bl_slide
    kv_total = kv_full + kv_slide
    kv_full_dtype_matched = kv_full - kv_full_compressed + kv_full_compressed // 2
    kv_total_dtype_matched = kv_full_dtype_matched + kv_slide

    def ratio(a, b):
        return (a / b) if b > 0 else None

    return {
        "per_layer": per_layer,
        "totals": {
            "baseline_full_bytes": bl_full,
            "baseline_sliding_bytes": bl_slide,
            "baseline_total_bytes": bl_total,
            "kakeya_full_bytes": kv_full,
            "kakeya_sliding_bytes": kv_slide,
            "kakeya_total_bytes": kv_total,
            "kakeya_full_bytes_dtype_matched": kv_full_dtype_matched,
            "kakeya_total_bytes_dtype_matched": kv_total_dtype_matched,
        },
        "ratios": {
            "full_attention_ratio_f32_compressed_store": ratio(bl_full, kv_full),
            "full_attention_ratio_bf16_compressed_store": ratio(bl_full, kv_full_dtype_matched),
            "total_ratio_f32_compressed_store": ratio(bl_total, kv_total),
            "total_ratio_bf16_compressed_store": ratio(bl_total, kv_total_dtype_matched),
        },
        "baseline_mode": "analytic",
    }


def summarize_cache(
    baseline_cache: DynamicCache,
    kakeya_cache: KakeyaKVCache,
    layer_types: List[str],
) -> Dict[str, Any]:
    assert len(baseline_cache.layers) == len(kakeya_cache.layers) == len(layer_types), (
        len(baseline_cache.layers),
        len(kakeya_cache.layers),
        len(layer_types),
    )

    per_layer: List[Dict[str, Any]] = []
    bl_full = bl_slide = 0
    kv_full = kv_slide = 0
    kv_full_compressed = 0  # sum of skeleton+encoded only on full-attn layers

    for idx, (bl_layer, kv_layer, lt) in enumerate(
        zip(baseline_cache.layers, kakeya_cache.layers, layer_types)
    ):
        bl_bytes, bl_seq, head_dim = baseline_layer_bytes(bl_layer)
        sliding = layer_is_sliding(bl_layer)
        kv_seq = kv_layer.get_seq_length()

        if isinstance(kv_layer, KakeyaCompressedLayer):
            kv_stats = kakeya_layer_bytes(kv_layer)
            kv_bytes_total = kv_stats["total_bytes"]
            kv_compressed_only = kv_stats["skeleton_bytes"] + kv_stats["encoded_bytes"]
            kv_blocks = kv_stats["compressed_blocks"]
            kv_tail = kv_stats["exact_tail_bytes"]
            kv_sk = kv_stats["skeleton_bytes"]
            kv_enc = kv_stats["encoded_bytes"]
        else:
            kv_bytes_total, kv_seq_s, kv_head_dim = baseline_layer_bytes(kv_layer)
            kv_compressed_only = 0
            kv_blocks = 0
            kv_tail = kv_bytes_total
            kv_sk = 0
            kv_enc = 0

        per_layer.append(
            {
                "layer_idx": idx,
                "layer_type": lt,
                "is_sliding": sliding,
                "baseline_seq_len": bl_seq,
                "kakeya_seq_len": kv_seq,
                "head_dim": head_dim,
                "baseline_bytes": bl_bytes,
                "kakeya_bytes_total": kv_bytes_total,
                "kakeya_compressed_blocks": kv_blocks,
                "kakeya_skeleton_bytes": kv_sk,
                "kakeya_encoded_bytes": kv_enc,
                "kakeya_exact_tail_bytes": kv_tail,
                "layer_ratio_baseline_over_kakeya": (bl_bytes / kv_bytes_total) if kv_bytes_total > 0 else None,
            }
        )

        if sliding:
            bl_slide += bl_bytes
            kv_slide += kv_bytes_total
        else:
            bl_full += bl_bytes
            kv_full += kv_bytes_total
            kv_full_compressed += kv_compressed_only

    bl_total = bl_full + bl_slide
    kv_total = kv_full + kv_slide

    # dtype-matched projection: halve the compressed-only bytes
    kv_full_dtype_matched = (
        kv_full - kv_full_compressed + kv_full_compressed // 2
    )
    kv_total_dtype_matched = kv_full_dtype_matched + kv_slide

    def ratio(a, b):
        return (a / b) if b > 0 else None

    return {
        "per_layer": per_layer,
        "totals": {
            "baseline_full_bytes": bl_full,
            "baseline_sliding_bytes": bl_slide,
            "baseline_total_bytes": bl_total,
            "kakeya_full_bytes": kv_full,
            "kakeya_sliding_bytes": kv_slide,
            "kakeya_total_bytes": kv_total,
            "kakeya_full_bytes_dtype_matched": kv_full_dtype_matched,
            "kakeya_total_bytes_dtype_matched": kv_total_dtype_matched,
        },
        "ratios": {
            "full_attention_ratio_f32_compressed_store": ratio(bl_full, kv_full),
            "full_attention_ratio_bf16_compressed_store": ratio(bl_full, kv_full_dtype_matched),
            "total_ratio_f32_compressed_store": ratio(bl_total, kv_total),
            "total_ratio_bf16_compressed_store": ratio(bl_total, kv_total_dtype_matched),
        },
    }


def humanize_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f"{v:,.2f} {u}"
        v /= 1024
    return f"{n} B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/workspace/models/gemma-4-E2B-it")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--context-tokens", type=int, default=2048)
    p.add_argument("--new-tokens", type=int, default=16)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--residual-length", type=int, default=256)
    p.add_argument("--variance-ratio", type=float, default=0.95)
    p.add_argument("--k-segments", type=int, default=16)
    p.add_argument("--d-res", type=int, default=8)
    p.add_argument("--attn", default="eager")
    p.add_argument("--report", default="/workspace/kakeya_bench_report.json")
    p.add_argument("--skip-baseline-generation", action="store_true")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--prefill-chunk", type=int, default=0,
                   help="Chunk size for incremental prefill (0 = single forward).")
    p.add_argument("--skip-baseline-prefill", action="store_true",
                   help="Derive baseline KV bytes analytically instead of running the DynamicCache "
                        "prefill. Much cheaper for long contexts (baseline is deterministic).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[load] dtype={args.dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        attn_implementation=args.attn,
    )
    model.eval()

    text_cfg = model.config.get_text_config(decoder=True)
    num_hidden = text_cfg.num_hidden_layers
    num_shared = getattr(text_cfg, "num_kv_shared_layers", 0) or 0
    non_shared_types = text_cfg.layer_types[: num_hidden - num_shared] if num_shared else text_cfg.layer_types
    print(f"[model] num_hidden_layers={num_hidden}, num_kv_shared_layers={num_shared}, cached_layers={len(non_shared_types)}", flush=True)
    print(f"[model] layer_types (cached): {non_shared_types}", flush=True)

    input_ids = build_long_prompt(tokenizer, args.context_tokens)
    print(f"[prompt] tokens={input_ids.shape[-1]}", flush=True)

    kakeya_cache = build_gemma4_kakeya_cache(
        model,
        variance_ratio=args.variance_ratio,
        K=args.k_segments,
        d_res=args.d_res,
        residual_length=args.residual_length,
        block_size=args.block_size,
    )

    if args.skip_baseline_prefill:
        print("[prefill] baseline skipped (will be derived analytically)", flush=True)
        baseline_cache = None
        t_bl = 0.0
    else:
        baseline_cache = DynamicCache(config=model.config)
        print("[prefill] baseline DynamicCache ...", flush=True)
        t_bl = run_prefill(model, input_ids, baseline_cache, chunk=args.prefill_chunk)
        print(f"[prefill] baseline done: {t_bl:.2f}s", flush=True)
        gc.collect()

    print("[prefill] KakeyaKVCache ...", flush=True)
    t_kv = run_prefill(model, input_ids, kakeya_cache, chunk=args.prefill_chunk)
    print(f"[prefill] kakeya   done: {t_kv:.2f}s", flush=True)

    print("[measure] summarizing caches ...", flush=True)
    if baseline_cache is None:
        summary = summarize_cache_analytic_baseline(
            kakeya_cache, non_shared_types, model.config, input_ids.shape[-1],
            dtype_bytes=torch.tensor([], dtype=dtype).element_size(),
            bsz=int(input_ids.shape[0]),
        )
    else:
        summary = summarize_cache(baseline_cache, kakeya_cache, non_shared_types)

    # -----------------------------------------------------------------
    # Generation sanity check
    # -----------------------------------------------------------------
    gen_report: Dict[str, Any] = {}
    if args.new_tokens > 0 and not args.skip_generation:
        print("[generate] building fresh caches for a short continuation ...", flush=True)
        fresh_baseline = DynamicCache(config=model.config)
        fresh_kakeya = build_gemma4_kakeya_cache(
            model,
            variance_ratio=args.variance_ratio,
            K=args.k_segments,
            d_res=args.d_res,
            residual_length=args.residual_length,
            block_size=args.block_size,
        )

        text_kv, t_gen_kv, ntok_kv = run_generate(
            model, tokenizer, input_ids, fresh_kakeya, args.new_tokens
        )
        print(f"[generate] kakeya: {ntok_kv} new tokens in {t_gen_kv:.2f}s", flush=True)
        gen_report["kakeya"] = {
            "new_tokens": ntok_kv,
            "elapsed_seconds": round(t_gen_kv, 3),
            "tokens_per_second": round(ntok_kv / t_gen_kv, 3) if t_gen_kv > 0 else None,
            "text": text_kv,
        }

        if not args.skip_baseline_generation:
            text_bl, t_gen_bl, ntok_bl = run_generate(
                model, tokenizer, input_ids, fresh_baseline, args.new_tokens
            )
            print(f"[generate] baseline: {ntok_bl} new tokens in {t_gen_bl:.2f}s", flush=True)
            gen_report["baseline"] = {
                "new_tokens": ntok_bl,
                "elapsed_seconds": round(t_gen_bl, 3),
                "tokens_per_second": round(ntok_bl / t_gen_bl, 3) if t_gen_bl > 0 else None,
                "text": text_bl,
            }

    # -----------------------------------------------------------------
    # Nice console summary
    # -----------------------------------------------------------------
    tot = summary["totals"]
    rat = summary["ratios"]

    print("\n============ KV cache byte report ============")
    print(f"context tokens            : {input_ids.shape[-1]}")
    print(f"baseline full-attn bytes  : {humanize_bytes(tot['baseline_full_bytes'])}")
    print(f"baseline sliding   bytes  : {humanize_bytes(tot['baseline_sliding_bytes'])}")
    print(f"baseline TOTAL     bytes  : {humanize_bytes(tot['baseline_total_bytes'])}")
    print("-----------------------------------------------")
    print(f"kakeya full-attn bytes    : {humanize_bytes(tot['kakeya_full_bytes'])}")
    print(f"kakeya sliding   bytes    : {humanize_bytes(tot['kakeya_sliding_bytes'])}")
    print(f"kakeya TOTAL     bytes    : {humanize_bytes(tot['kakeya_total_bytes'])}")
    print(f"kakeya full (dtype-matched) : {humanize_bytes(tot['kakeya_full_bytes_dtype_matched'])}")
    print(f"kakeya TOTAL (dtype-matched): {humanize_bytes(tot['kakeya_total_bytes_dtype_matched'])}")
    print("-----------------------------------------------")
    def fmt(r):
        return f"{r:.3f}x" if r is not None else "n/a"
    print(f"full-attn compression ratio (as-is f32 store)    : {fmt(rat['full_attention_ratio_f32_compressed_store'])}")
    print(f"full-attn compression ratio (bf16 store, projected): {fmt(rat['full_attention_ratio_bf16_compressed_store'])}")
    print(f"total compression ratio (as-is f32 store)        : {fmt(rat['total_ratio_f32_compressed_store'])}")
    print(f"total compression ratio (bf16 store, projected)  : {fmt(rat['total_ratio_bf16_compressed_store'])}")
    print("================================================\n")

    out_report = {
        "args": vars(args),
        "model": {
            "path": args.model_path,
            "num_hidden_layers": num_hidden,
            "num_kv_shared_layers": num_shared,
            "cached_layer_types": non_shared_types,
        },
        "context_tokens": int(input_ids.shape[-1]),
        "prefill_seconds": {"baseline": round(t_bl, 3), "kakeya": round(t_kv, 3)},
        "bytes": tot,
        "ratios": rat,
        "per_layer": summary["per_layer"],
        "generation": gen_report,
    }
    with open(args.report, "w") as f:
        json.dump(out_report, f, indent=2)
    print(f"[report] written to {args.report}")


if __name__ == "__main__":
    main()
