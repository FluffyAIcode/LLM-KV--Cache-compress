#!/usr/bin/env python3
"""Byte-for-byte comparison of Kakeya KV compression vs TurboQuant+
(PolarQuant + WHT + scalar quantization) on the same model, same context
length, same captured KV tensors.

Method
------
1. Load a HF model in bf16.
2. Run a single forward pass on a prompt of `context_tokens` tokens and
   capture the `past_key_values` -- these are the real KV tensors the
   Kakeya cache would build its blocks from.
3. Convert each layer's [bsz, n_kv, seq, head_dim] tensor to float32
   and run:
     - Kakeya: per-layer `build_kakeya_cache(model)` + block-wise
       PCA/K-means compression for tokens beyond `residual_length`,
       keeping the last `residual_length` exact.
     - TurboQuant+: per-vector PolarQuant (2/3/4 bits for K+V) on the
       *same* tensor, which is context-length-independent per-token
       quantization.
4. Count exact bytes for both paths:
     - Kakeya: serialized skeleton + encoded + exact-tail (as the
       benchmark reports do), with the bf16-store projection for
       dtype-matched cost.
     - TurboQuant+: `head_dim*b/8` bytes per vector + 32-bit norm
       per vector (bf16-equivalent since TurboQuant's compressed side
       is integer packing, not float tensors).
5. Produce a side-by-side JSON report per model, plus a unified
   Markdown table.

Output
------
- reports/compare/<model>/compare_<ctx>.json (per-context)
- reports/compare/SUMMARY.md (cross-model table)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, set_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kakeya_kv_codec import KakeyaCodec  # noqa: E402
from kakeya_benchmark import (  # noqa: E402
    build_long_prompt,
    kakeya_block_bytes,
    tensor_bytes,
)

# TurboQuant+ imports
sys.path.insert(0, "/workspace/turboquant_plus")
from turboquant import TurboQuant, TurboQuantMSE, KVCacheCompressor  # noqa: E402


# ---------------------------------------------------------------------------
# Kakeya on a single layer tensor -- same behaviour as KakeyaCompressedLayer
# but computed out-of-place so we can compare on the exact same tensor.
# ---------------------------------------------------------------------------


@dataclass
class KakeyaLayerResult:
    baseline_bytes: int
    compressed_bytes_f32: int
    compressed_bytes_bf16: int
    skeleton_bytes: int
    encoded_bytes: int
    exact_tail_bytes: int
    n_blocks: int


def kakeya_compress_tensor(
    x: torch.Tensor,
    block_size: int,
    residual_length: int,
    variance_ratio: float,
    K: int,
    d_res: int,
    dtype_bytes: int,
) -> KakeyaLayerResult:
    """Apply the Kakeya codec to a single 4D KV tensor [bsz, n_kv, seq, head_dim].

    Behaves identically to KakeyaCompressedLayer: everything past the
    `residual_length`-token tail is packed into `block_size`-sized blocks
    and compressed, the tail is kept at full precision."""
    bsz, n_kv, seq, head_dim = x.shape
    baseline_bytes = x.numel() * dtype_bytes

    compressed_tokens = max(((seq - residual_length) // block_size) * block_size, 0)
    tail_tokens = seq - compressed_tokens
    n_blocks = compressed_tokens // block_size

    codec = KakeyaCodec(
        d_model=head_dim,
        variance_ratio=variance_ratio,
        K=K,
        d_res=d_res,
        min_rows_to_build=8,
    )

    skeleton_bytes = 0
    encoded_bytes = 0
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        block = x[:, :, start:end, :].contiguous().reshape(-1, head_dim).float()
        if block.shape[0] < codec.min_rows_to_build:
            continue
        sk = codec.fit(block)
        enc = codec.encode(block, sk)
        bb = kakeya_block_bytes(_BlockAdapter(sk, enc))
        skeleton_bytes += bb["skeleton_bytes"]
        encoded_bytes += bb["encoded_bytes"]

    tail = x[:, :, compressed_tokens:, :].contiguous()
    exact_tail_bytes = tensor_bytes(tail.to(
        dtype=torch.bfloat16 if dtype_bytes == 2 else (torch.float16 if dtype_bytes == 2 else torch.float32)
    ))

    total_f32 = skeleton_bytes + encoded_bytes + exact_tail_bytes
    # bf16-store projection: halve the compressed tensors (non-integer side);
    # tail is already in the KV dtype. Roughly approximate by halving
    # skeleton+encoded. This matches the convention used in kakeya_benchmark.
    compressed_only = skeleton_bytes + encoded_bytes
    total_bf16 = (total_f32 - compressed_only) + compressed_only // 2

    return KakeyaLayerResult(
        baseline_bytes=baseline_bytes,
        compressed_bytes_f32=total_f32,
        compressed_bytes_bf16=total_bf16,
        skeleton_bytes=skeleton_bytes,
        encoded_bytes=encoded_bytes,
        exact_tail_bytes=exact_tail_bytes,
        n_blocks=n_blocks,
    )


class _BlockAdapter:
    """Shim so we can reuse kakeya_benchmark.kakeya_block_bytes."""
    def __init__(self, sk, enc):
        self.skeleton = sk
        self.encoded = enc
        self.shape = None
        self.dtype = None


# ---------------------------------------------------------------------------
# TurboQuant+ on a single layer tensor
# ---------------------------------------------------------------------------


@dataclass
class TurboQuantLayerResult:
    baseline_bytes: int
    compressed_bytes: int
    bits_per_value: float
    k_mse: float
    v_mse: float


def turboquant_compress_tensor(
    k: torch.Tensor,
    v: torch.Tensor,
    k_bits: int,
    v_bits: int,
    dtype_bytes: int,
    measure_mse: bool = True,
) -> TurboQuantLayerResult:
    """Apply TurboQuant+ to K and V tensors of shape [bsz, n_kv, seq, head_dim].

    Uses the TurboQuant+ KVCacheCompressor directly, which is the
    PolarQuant+QJL (K) and PolarQuant-MSE (V) pipeline from the
    official Python prototype.

    The compressed size formula (from turboquant.kv_cache.KVCacheCompressor.memory_stats):
      K: n_vectors * (head_dim * k_bits + 32 bits norm) / 8 bytes
      V: n_vectors * head_dim * v_bits / 8 bytes
    where n_vectors = num_layers * num_heads * seq_len.
    Note that turboquant+ does **not** compress across tokens; every
    token's KV is quantized independently.
    """
    bsz, n_kv, seq, head_dim = k.shape
    assert v.shape == k.shape

    baseline_bytes = (k.numel() + v.numel()) * dtype_bytes

    # Compressed bytes are a deterministic function of the shape + bit width.
    # n_vectors counts per stream (K and V each), matching the paper and code.
    n_vectors = bsz * n_kv * seq
    k_bits_total = n_vectors * (head_dim * k_bits + 32)  # +32 bits for norm
    v_bits_total = n_vectors * head_dim * v_bits          # no norm (MSE-only)
    compressed_bytes = (k_bits_total + v_bits_total) // 8
    bpv = (k_bits + v_bits) / 2 + 32 / (2 * head_dim)

    k_mse = v_mse = float("nan")
    if measure_mse and n_vectors > 0:
        # Sample MSE on a subset to save CPU time on large caches
        max_samples = 4096
        # Flatten [bsz, n_kv, seq, head_dim] -> [n_vectors, head_dim]
        k_np = k.reshape(-1, head_dim).float().numpy()
        v_np = v.reshape(-1, head_dim).float().numpy()
        if n_vectors > max_samples:
            idx = np.random.default_rng(0).choice(n_vectors, max_samples, replace=False)
            k_np = k_np[idx]
            v_np = v_np[idx]
        compressor = KVCacheCompressor(head_dim=head_dim, k_bits=k_bits, v_bits=v_bits)
        k_4d = k_np[np.newaxis, np.newaxis]
        v_4d = v_np[np.newaxis, np.newaxis]
        c = compressor.compress(k_4d, v_4d)
        k_hat, v_hat = compressor.decompress(c)
        k_mse = float(np.mean((k_np - k_hat[0, 0]) ** 2))
        v_mse = float(np.mean((v_np - v_hat[0, 0]) ** 2))

    return TurboQuantLayerResult(
        baseline_bytes=baseline_bytes,
        compressed_bytes=compressed_bytes,
        bits_per_value=bpv,
        k_mse=k_mse,
        v_mse=v_mse,
    )


# ---------------------------------------------------------------------------
# KV capture + comparison orchestration
# ---------------------------------------------------------------------------


def _is_sliding(layer) -> bool:
    return bool(getattr(layer, "is_sliding", False))


@torch.inference_mode()
def capture_kv_cache(
    model,
    input_ids: torch.Tensor,
    chunk: int = 0,
) -> Tuple[DynamicCache, List[str]]:
    """Run a prefill pass and return the populated DynamicCache plus the
    layer type labels ("full_attention" / "sliding_attention") for each
    cached layer (as used by HF transformers for this model family).
    """
    cache = DynamicCache(config=model.config)
    if chunk <= 0 or input_ids.shape[-1] <= chunk:
        _ = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    else:
        n = input_ids.shape[-1]
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            _ = model(input_ids=input_ids[:, s:e], past_key_values=cache, use_cache=True)

    text_cfg = model.config.get_text_config(decoder=True)
    lt = getattr(text_cfg, "layer_types", None)
    if lt is None:
        sw = getattr(text_cfg, "sliding_window", None) or getattr(text_cfg, "attention_chunk_size", None)
        lt = ["sliding_attention" if sw else "full_attention"] * text_cfg.num_hidden_layers
    n_shared = getattr(text_cfg, "num_kv_shared_layers", 0) or 0
    non_shared = list(lt)[: text_cfg.num_hidden_layers - n_shared] if n_shared else list(lt)
    return cache, non_shared


def compare_model_at_context(
    model_path: str,
    model_name: str,
    context_tokens: int,
    block_size: int = 512,
    residual_length: int = 256,
    d_res: int = 8,
    K: int = 16,
    variance_ratio: float = 0.95,
    tq_configs: List[Tuple[str, int, int]] = None,
    dtype: torch.dtype = torch.bfloat16,
    attn: str = "eager",
    chunk: int = 0,
) -> Dict[str, Any]:
    if tq_configs is None:
        tq_configs = [("turbo2", 2, 2), ("turbo3", 3, 3), ("turbo4", 4, 4)]

    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    print(f"[{model_name}] loading model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, attn_implementation=attn
    )
    model.eval()

    input_ids = build_long_prompt(tokenizer, context_tokens)
    print(f"[{model_name}] prefill ctx={input_ids.shape[-1]} ...", flush=True)
    t0 = time.perf_counter()
    cache, layer_types = capture_kv_cache(model, input_ids, chunk=chunk)
    t_prefill = time.perf_counter() - t0
    print(f"[{model_name}] prefill done in {t_prefill:.1f}s", flush=True)

    per_layer = []
    total_baseline_bytes = 0
    kakeya_total_f32 = 0
    kakeya_total_bf16 = 0
    kakeya_full_f32 = 0
    kakeya_full_bf16 = 0
    kakeya_full_baseline = 0
    tq_totals: Dict[str, int] = {name: 0 for (name, _, _) in tq_configs}
    tq_full_totals: Dict[str, int] = {name: 0 for (name, _, _) in tq_configs}

    for idx, (layer, lt) in enumerate(zip(cache.layers, layer_types)):
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        if k is None or v is None or k.numel() == 0:
            continue

        is_sliding = (lt == "sliding_attention") or _is_sliding(layer)
        # Baseline bytes in the model dtype (bf16):
        bl = (k.numel() + v.numel()) * dtype_bytes
        total_baseline_bytes += bl

        # -----------------------------------------------------------------
        # Kakeya compression (same tensor, same codec as the benchmark)
        # -----------------------------------------------------------------
        if is_sliding:
            # Sliding layers are untouched by Kakeya (the codec passes them
            # through as DynamicSlidingWindowLayer). For a fair size count,
            # treat them as baseline bytes on the Kakeya side too.
            kakeya_total_f32 += bl
            kakeya_total_bf16 += bl
            k_res = v_res = None
            k_sk = k_enc = k_tail = v_sk = v_enc = v_tail = 0
            k_blocks = 0
            k_cf32 = k_cbf16 = bl // 2
            v_cf32 = v_cbf16 = bl // 2
        else:
            k_res = kakeya_compress_tensor(
                k.cpu(), block_size, residual_length, variance_ratio, K, d_res, dtype_bytes
            )
            v_res = kakeya_compress_tensor(
                v.cpu(), block_size, residual_length, variance_ratio, K, d_res, dtype_bytes
            )
            k_cf32, k_cbf16 = k_res.compressed_bytes_f32, k_res.compressed_bytes_bf16
            v_cf32, v_cbf16 = v_res.compressed_bytes_f32, v_res.compressed_bytes_bf16
            k_sk, k_enc, k_tail, k_blocks = k_res.skeleton_bytes, k_res.encoded_bytes, k_res.exact_tail_bytes, k_res.n_blocks
            v_sk, v_enc, v_tail = v_res.skeleton_bytes, v_res.encoded_bytes, v_res.exact_tail_bytes
            kakeya_total_f32 += k_cf32 + v_cf32
            kakeya_total_bf16 += k_cbf16 + v_cbf16
            kakeya_full_f32 += k_cf32 + v_cf32
            kakeya_full_bf16 += k_cbf16 + v_cbf16
            kakeya_full_baseline += bl

        # -----------------------------------------------------------------
        # TurboQuant+ compression (same tensor)
        # -----------------------------------------------------------------
        tq_layer: Dict[str, Dict[str, Any]] = {}
        for (tq_name, kb, vb) in tq_configs:
            # Sliding layers: TurboQuant+ *would* compress them
            # (it's token-wise quantization), but for a fair comparison
            # against Kakeya at the full-cache level we apply it to every
            # layer. We also report the ratio computed on full-attention
            # layers only for a direct head-to-head on the layers Kakeya
            # actually targets.
            r = turboquant_compress_tensor(
                k.cpu(), v.cpu(), kb, vb, dtype_bytes,
                measure_mse=(idx == 0),  # MSE only on first layer to save time
            )
            tq_layer[tq_name] = {
                "compressed_bytes": r.compressed_bytes,
                "bits_per_value": r.bits_per_value,
                "ratio_vs_baseline": bl / r.compressed_bytes if r.compressed_bytes > 0 else None,
                "k_mse_sample": r.k_mse,
                "v_mse_sample": r.v_mse,
            }
            tq_totals[tq_name] += r.compressed_bytes
            if not is_sliding:
                tq_full_totals[tq_name] += r.compressed_bytes

        per_layer.append({
            "layer_idx": idx,
            "layer_type": lt,
            "is_sliding": is_sliding,
            "seq_len": int(k.shape[-2]),
            "head_dim": int(k.shape[-1]),
            "num_kv_heads": int(k.shape[1]),
            "baseline_bytes": bl,
            "kakeya": {
                "compressed_bytes_f32": k_cf32 + v_cf32,
                "compressed_bytes_bf16": k_cbf16 + v_cbf16,
                "k_skeleton_bytes": k_sk,
                "k_encoded_bytes": k_enc,
                "k_tail_bytes": k_tail,
                "v_skeleton_bytes": v_sk,
                "v_encoded_bytes": v_enc,
                "v_tail_bytes": v_tail,
                "k_blocks": k_blocks,
            },
            "turboquant": tq_layer,
        })

    # Clean up GPU/CPU memory for sequential model runs
    del cache, model
    gc.collect()

    totals = {
        "baseline_bytes": total_baseline_bytes,
        "kakeya_total_bytes_f32": kakeya_total_f32,
        "kakeya_total_bytes_bf16": kakeya_total_bf16,
        "kakeya_full_only_baseline_bytes": kakeya_full_baseline,
        "kakeya_full_only_bytes_f32": kakeya_full_f32,
        "kakeya_full_only_bytes_bf16": kakeya_full_bf16,
        "turboquant_total_bytes": tq_totals,
        "turboquant_full_only_bytes": tq_full_totals,
    }
    ratios = {
        "kakeya_total_ratio_f32": total_baseline_bytes / kakeya_total_f32 if kakeya_total_f32 > 0 else None,
        "kakeya_total_ratio_bf16": total_baseline_bytes / kakeya_total_bf16 if kakeya_total_bf16 > 0 else None,
        "kakeya_full_only_ratio_f32": (kakeya_full_baseline / kakeya_full_f32) if kakeya_full_f32 > 0 else None,
        "kakeya_full_only_ratio_bf16": (kakeya_full_baseline / kakeya_full_bf16) if kakeya_full_bf16 > 0 else None,
        "turboquant_total_ratios": {
            name: (total_baseline_bytes / tq_totals[name]) if tq_totals[name] > 0 else None
            for name in tq_totals
        },
        "turboquant_full_only_ratios": {
            name: (kakeya_full_baseline / tq_full_totals[name]) if tq_full_totals.get(name, 0) > 0 else None
            for name in tq_full_totals
        },
    }
    return {
        "model_name": model_name,
        "model_path": model_path,
        "context_tokens": int(input_ids.shape[-1]),
        "dtype": str(dtype),
        "codec_preset": {
            "block_size": block_size,
            "residual_length": residual_length,
            "d_res": d_res,
            "K": K,
            "variance_ratio": variance_ratio,
        },
        "per_layer": per_layer,
        "totals": totals,
        "ratios": ratios,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--context-tokens", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--residual-length", type=int, default=256)
    p.add_argument("--d-res", type=int, default=8)
    p.add_argument("--k-segments", type=int, default=16)
    p.add_argument("--variance-ratio", type=float, default=0.95)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--attn", default="eager")
    p.add_argument("--prefill-chunk", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_name is None:
        args.model_name = args.model_path.rstrip("/").split("/")[-1]
    set_seed(0)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    res = compare_model_at_context(
        model_path=args.model_path,
        model_name=args.model_name,
        context_tokens=args.context_tokens,
        block_size=args.block_size,
        residual_length=args.residual_length,
        d_res=args.d_res,
        K=args.k_segments,
        variance_ratio=args.variance_ratio,
        dtype=dtype,
        attn=args.attn,
        chunk=args.prefill_chunk,
    )

    t = res["totals"]; r = res["ratios"]
    print()
    print(f"======= {args.model_name} @ {res['context_tokens']} tokens =======")
    def mb(x): return f"{x/1024/1024:,.2f} MiB"
    print(f"baseline (bf16)           : {mb(t['baseline_bytes'])}")
    print(f"kakeya    total (f32)     : {mb(t['kakeya_total_bytes_f32'])}  ratio {r['kakeya_total_ratio_f32']:.3f}x")
    print(f"kakeya    total (bf16)    : {mb(t['kakeya_total_bytes_bf16'])}  ratio {r['kakeya_total_ratio_bf16']:.3f}x")
    for name, b in t['turboquant_total_bytes'].items():
        print(f"turbo   {name:6s} total      : {mb(b)}  ratio {r['turboquant_total_ratios'][name]:.3f}x")
    print()
    print(f"kakeya    full-only (bf16): {mb(t['kakeya_full_only_bytes_bf16'])}  ratio {r['kakeya_full_only_ratio_bf16'] or 0:.3f}x")
    for name, b in t['turboquant_full_only_bytes'].items():
        r_val = r['turboquant_full_only_ratios'].get(name)
        print(f"turbo   {name:6s} full-only  : {mb(b)}  ratio {r_val:.3f}x" if r_val else f"turbo {name} full-only: n/a")
    print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
