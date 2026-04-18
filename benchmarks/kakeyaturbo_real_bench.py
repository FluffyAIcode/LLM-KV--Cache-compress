#!/usr/bin/env python3
"""Real end-to-end benchmark: run a real HF model, capture a real KV
cache tensor (K or V from a chosen cached layer), dump it to disk in
the KKTV binary format the `kakeyaturbo-bench` Rust binary expects,
invoke the binary, and aggregate the JSON reports.

No mock, no fallback, no simplification: the compression is executed
by the unmodified `kakeyaturbo` crate compiled in release mode.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

KAKEYATURBO_BIN = Path(__file__).resolve().parent.parent / "kakeyaturbo" / "target" / "release" / "kakeyaturbo-bench"
KKTV_MAGIC = 0x4B4B5456
KKTV_VERSION = 1

LONG_PROMPT_SEED = (
    "You are a careful technical writer. Please produce a long, self-contained "
    "explanation of how transformer key/value caches work during autoregressive "
    "decoding, why they grow linearly with the number of decoded tokens, why the "
    "memory pressure can dominate system throughput at large batch sizes, and "
    "what different compression strategies look like in practice, including "
    "quantization, low-rank projection, token eviction (H2O / Scissorhands), and "
    "learned codecs. Include concrete numerical examples throughout.\n\n"
)


def build_long_prompt(tokenizer, target_tokens: int) -> torch.Tensor:
    text = LONG_PROMPT_SEED
    while True:
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        if ids.shape[-1] >= target_tokens:
            return ids[:, :target_tokens]
        text = text + LONG_PROMPT_SEED


def write_kktv(path: Path, tensor: np.ndarray) -> None:
    """Write a 2D [N, D] float32 tensor to disk in the KKTV format."""
    assert tensor.dtype == np.float32 and tensor.ndim == 2, (tensor.dtype, tensor.shape)
    n, d = tensor.shape
    with path.open("wb") as f:
        f.write(struct.pack("<I", KKTV_MAGIC))
        f.write(struct.pack("<I", KKTV_VERSION))
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<I", d))
        f.write(struct.pack("<I", 0))  # pad
        f.write(tensor.tobytes(order="C"))


@torch.inference_mode()
def capture_kv(
    model_path: str,
    context_tokens: int,
    prefill_chunk: int,
    attn: str,
    dtype: torch.dtype,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], dict[str, Any]]:
    """Run a real prefill on the model and return per-layer K, V as
    float32 numpy arrays of shape [n_vectors, head_dim] where
    n_vectors = bsz * num_kv_heads * seq_len (the pooling convention
    used by kakeyaturbo's PCA input).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation=attn)
    model.eval()

    text_cfg = model.config.get_text_config(decoder=True)
    layer_types = getattr(text_cfg, "layer_types", None)
    n_shared = getattr(text_cfg, "num_kv_shared_layers", 0) or 0
    if layer_types is None:
        sw = getattr(text_cfg, "sliding_window", None) or getattr(text_cfg, "attention_chunk_size", None)
        layer_types = ["sliding_attention" if sw else "full_attention"] * text_cfg.num_hidden_layers
    non_shared = list(layer_types)[: text_cfg.num_hidden_layers - n_shared] if n_shared else list(layer_types)

    input_ids = build_long_prompt(tokenizer, context_tokens)
    cache = DynamicCache(config=model.config)
    t0 = time.perf_counter()
    if prefill_chunk <= 0 or input_ids.shape[-1] <= prefill_chunk:
        _ = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    else:
        for s in range(0, input_ids.shape[-1], prefill_chunk):
            e = min(s + prefill_chunk, input_ids.shape[-1])
            _ = model(input_ids=input_ids[:, s:e], past_key_values=cache, use_cache=True)
    prefill_elapsed = time.perf_counter() - t0

    ks, vs = [], []
    for layer in cache.layers:
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        if k is None or v is None or k.numel() == 0:
            ks.append(None); vs.append(None); continue
        # [bsz, n_kv, seq, head_dim]  →  [bsz * n_kv * seq, head_dim]
        k_flat = k.to(torch.float32).cpu().reshape(-1, k.shape[-1]).contiguous().numpy()
        v_flat = v.to(torch.float32).cpu().reshape(-1, v.shape[-1]).contiguous().numpy()
        ks.append(k_flat)
        vs.append(v_flat)

    meta = {
        "model_path": model_path,
        "context_tokens": int(input_ids.shape[-1]),
        "prefill_seconds": round(prefill_elapsed, 3),
        "dtype": str(dtype),
        "attn_implementation": attn,
        "num_hidden_layers": text_cfg.num_hidden_layers,
        "num_kv_shared_layers": n_shared,
        "cached_layer_count": len(non_shared),
        "cached_layer_types": non_shared,
        "num_attention_heads": text_cfg.num_attention_heads,
        "num_key_value_heads": text_cfg.num_key_value_heads,
        "head_dim": getattr(text_cfg, "head_dim", None) or (text_cfg.hidden_size // text_cfg.num_attention_heads),
        "global_head_dim": getattr(text_cfg, "global_head_dim", None),
        "sliding_window": getattr(text_cfg, "sliding_window", None),
    }
    del model, cache
    return ks, vs, non_shared, meta


def bench_one_tensor(
    tensor_path: Path,
    report_path: Path,
    metric: str,
    block_size: int,
    variance_ratio: float,
    k: int,
    bit_width: int,
    rotation_seed: int,
    verify: bool,
) -> dict:
    cmd = [
        str(KAKEYATURBO_BIN),
        "--input", str(tensor_path),
        "--output", str(report_path),
        "--metric", metric,
        "--block-size", str(block_size),
        "--variance-ratio", str(variance_ratio),
        "--k", str(k),
        "--bit-width", str(bit_width),
        "--rotation-seed", str(rotation_seed),
    ]
    if verify:
        cmd.append("--verify")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"kakeyaturbo-bench failed (exit {r.returncode}):\nstderr:\n{r.stderr}\nstdout:\n{r.stdout}"
        )
    return json.loads(report_path.read_text())


def run_model(
    model_path: str,
    model_name: str,
    context_tokens: int,
    block_size: int,
    variance_ratio: float,
    k: int,
    bit_width: int,
    rotation_seed: int,
    prefill_chunk: int,
    out_dir: Path,
    verify: bool,
    dtype: torch.dtype = torch.bfloat16,
    attn: str = "eager",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{model_name}] loading and prefilling @ {context_tokens} tokens …", flush=True)
    ks, vs, layer_types, meta = capture_kv(model_path, context_tokens, prefill_chunk, attn, dtype)

    per_layer_reports: list[dict] = []
    total_baseline_bf16 = 0
    total_compressed = 0
    # Track only full-attention (compressible) layer stats separately so the
    # report matches the "full-attn bytes" column from the previous work.
    total_full_baseline_bf16 = 0
    total_full_compressed = 0
    total_sliding_bf16 = 0

    for i, lt in enumerate(layer_types):
        k_arr, v_arr = ks[i], vs[i]
        if k_arr is None or v_arr is None:
            continue
        is_sliding = lt == "sliding_attention"

        head_dim = int(k_arr.shape[-1])
        seq_layer = int(k_arr.shape[0])
        layer_baseline_bf16 = (k_arr.size + v_arr.size) * 2  # bf16 = 2 bytes

        if is_sliding:
            # Pass-through: kakeyaturbo does not compress sliding-window
            # layers (they are already O(sliding_window) in length).
            per_layer_reports.append({
                "layer_idx": i,
                "layer_type": lt,
                "is_sliding": True,
                "head_dim": head_dim,
                "seq": seq_layer,
                "baseline_bf16": layer_baseline_bf16,
                "compressed_bytes": layer_baseline_bf16,  # accounted as-is
                "compression_source": "pass_through",
            })
            total_baseline_bf16 += layer_baseline_bf16
            total_compressed += layer_baseline_bf16
            total_sliding_bf16 += layer_baseline_bf16
            continue

        # Compress K and V as independent streams with an asymmetric metric:
        # K → InnerProduct (attention Q·Kᵀ semantics),
        # V → MSE (weighted sum semantics).
        layer_dir = out_dir / f"layer_{i:02d}"
        layer_dir.mkdir(exist_ok=True)

        k_path = layer_dir / "k.kktv"
        v_path = layer_dir / "v.kktv"
        write_kktv(k_path, k_arr.astype(np.float32, copy=False))
        write_kktv(v_path, v_arr.astype(np.float32, copy=False))

        k_report = bench_one_tensor(
            k_path, layer_dir / "k.json",
            metric="inner_product",
            block_size=block_size,
            variance_ratio=variance_ratio,
            k=k, bit_width=bit_width,
            rotation_seed=rotation_seed,
            verify=verify,
        )
        v_report = bench_one_tensor(
            v_path, layer_dir / "v.json",
            metric="mse",
            block_size=block_size,
            variance_ratio=variance_ratio,
            k=k, bit_width=bit_width,
            rotation_seed=rotation_seed + 1,
            verify=verify,
        )

        # Tail vectors below block_size × round are NOT compressed by the
        # Rust binary (we drop them). For fairness, count them in baseline
        # bytes (matching uncompressed fp16) on both sides.
        full_blocks = seq_layer // block_size
        tail = seq_layer - full_blocks * block_size
        tail_bytes = 2 * tail * head_dim  # bf16, K + V duplicated
        k_compressed = k_report["compressed_bytes"] + tail * head_dim * 2 // 2  # K side tail
        v_compressed = v_report["compressed_bytes"] + tail * head_dim * 2 // 2  # V side tail
        layer_compressed = k_report["compressed_bytes"] + v_report["compressed_bytes"] + tail_bytes

        per_layer_reports.append({
            "layer_idx": i,
            "layer_type": lt,
            "is_sliding": False,
            "head_dim": head_dim,
            "seq": seq_layer,
            "baseline_bf16": layer_baseline_bf16,
            "tail_tokens": tail,
            "tail_bytes_bf16": tail_bytes,
            "k_report": k_report,
            "v_report": v_report,
            "compressed_bytes": layer_compressed,
            "compression_source": "kakeyaturbo",
        })
        total_baseline_bf16 += layer_baseline_bf16
        total_compressed += layer_compressed
        total_full_baseline_bf16 += layer_baseline_bf16
        total_full_compressed += layer_compressed

        # Be disk-polite: remove the raw tensors once the bin has read them.
        k_path.unlink(missing_ok=True)
        v_path.unlink(missing_ok=True)

    summary = {
        "model_name": model_name,
        "model_meta": meta,
        "codec_params": {
            "block_size": block_size,
            "variance_ratio": variance_ratio,
            "k": k,
            "bit_width": bit_width,
            "rotation_seed": rotation_seed,
        },
        "per_layer": per_layer_reports,
        "totals": {
            "baseline_bf16_bytes": total_baseline_bf16,
            "compressed_bytes": total_compressed,
            "ratio_bf16": total_baseline_bf16 / total_compressed if total_compressed else None,
            "full_attn_baseline_bf16_bytes": total_full_baseline_bf16,
            "full_attn_compressed_bytes": total_full_compressed,
            "full_attn_ratio_bf16": (
                total_full_baseline_bf16 / total_full_compressed if total_full_compressed else None
            ),
            "sliding_bytes_passthrough": total_sliding_bf16,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--context-tokens", type=int, default=2048)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--variance-ratio", type=float, default=0.95)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--bit-width", type=int, default=3)
    p.add_argument("--rotation-seed", type=int, default=0xCAFEBABE)
    p.add_argument("--prefill-chunk", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn", default="eager")
    return p.parse_args()


def main() -> None:
    if not KAKEYATURBO_BIN.exists():
        print(
            f"error: {KAKEYATURBO_BIN} does not exist. Run:\n"
            "  cd kakeyaturbo && cargo build --release --bin kakeyaturbo-bench",
            file=sys.stderr,
        )
        sys.exit(1)

    args = parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    summary = run_model(
        model_path=args.model_path,
        model_name=args.model_name,
        context_tokens=args.context_tokens,
        block_size=args.block_size,
        variance_ratio=args.variance_ratio,
        k=args.k,
        bit_width=args.bit_width,
        rotation_seed=args.rotation_seed,
        prefill_chunk=args.prefill_chunk,
        out_dir=args.out_dir,
        verify=args.verify,
        dtype=dtype,
        attn=args.attn,
    )
    tot = summary["totals"]
    print()
    print(f"======= {args.model_name} @ {args.context_tokens} tokens =======")
    print(f"baseline (bf16) total     : {tot['baseline_bf16_bytes'] / 1024 / 1024:,.2f} MiB")
    print(f"kakeyaturbo compressed    : {tot['compressed_bytes'] / 1024 / 1024:,.2f} MiB")
    ratio = tot.get("ratio_bf16")
    if ratio:
        print(f"total bf16 compression    : {ratio:.3f}x")
    full_ratio = tot.get("full_attn_ratio_bf16")
    if full_ratio:
        print(f"full-attn bf16 compression: {full_ratio:.3f}x")


if __name__ == "__main__":
    main()
