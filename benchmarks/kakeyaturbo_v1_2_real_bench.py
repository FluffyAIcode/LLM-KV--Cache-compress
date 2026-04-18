#!/usr/bin/env python3
"""Real kakeyaturbo v1.2 (A+B') end-to-end benchmark.

Per-layer pipeline:
  - K stream: per-block PCA (same as v1), InnerProduct metric, BF16 skeleton (A)
  - V stream: layer-pooled PCA via encode_layer --share-basis (B'), MSE metric,
              BF16 skeleton (A)

Everything else is identical to the v1 real bench so the comparison is
apples-to-apples. Drives the updated kakeyaturbo-bench Rust binary.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

REPO = Path(__file__).resolve().parent.parent
BENCH_BIN = REPO / "kakeyaturbo" / "target" / "release" / "kakeyaturbo-bench"
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
    assert tensor.dtype == np.float32 and tensor.ndim == 2
    n, d = tensor.shape
    with path.open("wb") as f:
        f.write(struct.pack("<I", KKTV_MAGIC))
        f.write(struct.pack("<I", KKTV_VERSION))
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<I", d))
        f.write(struct.pack("<I", 0))
        f.write(tensor.tobytes(order="C"))


@torch.inference_mode()
def capture(model_path: str, ctx: int, chunk: int, dtype=torch.bfloat16, attn="eager"):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation=attn)
    model.eval()
    text_cfg = model.config.get_text_config(decoder=True)
    layer_types = getattr(text_cfg, "layer_types", None)
    n_shared = getattr(text_cfg, "num_kv_shared_layers", 0) or 0
    if layer_types is None:
        sw = getattr(text_cfg, "sliding_window", None) or getattr(text_cfg, "attention_chunk_size", None)
        layer_types = ["sliding_attention" if sw else "full_attention"] * text_cfg.num_hidden_layers
    non_shared = list(layer_types)[: text_cfg.num_hidden_layers - n_shared] if n_shared else list(layer_types)

    ids = build_long_prompt(tok, ctx)
    cache = DynamicCache(config=model.config)
    t0 = time.perf_counter()
    if chunk <= 0 or ids.shape[-1] <= chunk:
        _ = model(input_ids=ids, past_key_values=cache, use_cache=True)
    else:
        for s in range(0, ids.shape[-1], chunk):
            e = min(s + chunk, ids.shape[-1])
            _ = model(input_ids=ids[:, s:e], past_key_values=cache, use_cache=True)
    prefill = time.perf_counter() - t0

    ks, vs = [], []
    for layer in cache.layers:
        k = getattr(layer, "keys", None); v = getattr(layer, "values", None)
        if k is None or v is None or k.numel() == 0:
            ks.append(None); vs.append(None); continue
        ks.append(k.to(torch.float32).cpu().reshape(-1, k.shape[-1]).contiguous().numpy())
        vs.append(v.to(torch.float32).cpu().reshape(-1, v.shape[-1]).contiguous().numpy())

    meta = {
        "model_path": model_path,
        "ctx": int(ids.shape[-1]),
        "prefill_seconds": round(prefill, 2),
        "num_hidden_layers": text_cfg.num_hidden_layers,
        "num_kv_shared_layers": n_shared,
        "cached_layer_count": len(non_shared),
        "num_attention_heads": text_cfg.num_attention_heads,
        "num_key_value_heads": text_cfg.num_key_value_heads,
        "head_dim": getattr(text_cfg, "head_dim", None) or (text_cfg.hidden_size // text_cfg.num_attention_heads),
        "global_head_dim": getattr(text_cfg, "global_head_dim", None),
        "sliding_window": getattr(text_cfg, "sliding_window", None),
    }
    del model, cache
    return ks, vs, non_shared, meta


def bench_one(tensor_path: Path, out_json: Path, metric: str, share_basis: bool,
              block_size: int, variance_ratio: float, k: int, bit_width: int, seed: int,
              pca_method: str = "exact",
              rsvd_target_rank: int | None = None,
              rsvd_oversample: int = 8,
              rsvd_power_iters: int = 2):
    cmd = [
        str(BENCH_BIN),
        "--input", str(tensor_path), "--output", str(out_json),
        "--metric", metric,
        "--block-size", str(block_size),
        "--variance-ratio", str(variance_ratio),
        "--k", str(k), "--bit-width", str(bit_width),
        "--rotation-seed", str(seed),
        "--pca-method", pca_method,
        "--verify",
    ]
    if share_basis:
        cmd.append("--share-basis")
    if pca_method == "randomized":
        if rsvd_target_rank is not None:
            cmd += ["--rsvd-target-rank", str(rsvd_target_rank)]
        cmd += ["--rsvd-oversample", str(rsvd_oversample),
                "--rsvd-power-iters", str(rsvd_power_iters)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bench failed ({r.returncode}): {r.stderr}")
    return json.loads(out_json.read_text())


def run_model(model_path, model_name, ctx, prefill_chunk, block_size, variance_ratio,
              k, bit_width, rotation_seed, out_dir: Path,
              pca_method: str = "exact",
              rsvd_target_rank: int | None = None,
              rsvd_oversample: int = 8,
              rsvd_power_iters: int = 2,
              rsvd_target_rank_k: int | None = None,
              rsvd_target_rank_v: int | None = None,
              bit_width_k: int | None = None,
              bit_width_v: int | None = None):
    """Per-stream overrides:
      - rsvd_target_rank_k / _v: per-stream PCA rank cap (falls back to
        rsvd_target_rank when None)
      - bit_width_k / _v: per-stream Lloyd-Max bit width (falls back to
        bit_width when None). Useful for MHA models with near-flat V
        spectra that need b=3 on V while K stays at b=2.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{model_name}] ctx={ctx} prefill + capture …", flush=True)
    ks, vs, layer_types, meta = capture(model_path, ctx, prefill_chunk)

    per_layer = []
    tot_base = 0; tot_comp = 0
    tot_full_base = 0; tot_full_comp = 0
    tot_slide_base = 0

    for i, lt in enumerate(layer_types):
        k_arr, v_arr = ks[i], vs[i]
        if k_arr is None or v_arr is None:
            continue
        head_dim = int(k_arr.shape[-1])
        layer_base = (k_arr.size + v_arr.size) * 2  # bf16

        if lt == "sliding_attention":
            per_layer.append({
                "layer_idx": i, "layer_type": lt, "is_sliding": True,
                "head_dim": head_dim, "seq": int(k_arr.shape[0]),
                "baseline_bf16": layer_base, "compressed_bytes": layer_base,
                "compression_source": "pass_through",
            })
            tot_base += layer_base; tot_comp += layer_base; tot_slide_base += layer_base
            continue

        # K: per-block PCA, InnerProduct
        # V: shared (layer-pooled) PCA, MSE
        layer_dir = out_dir / f"layer_{i:02d}"
        layer_dir.mkdir(exist_ok=True)

        kp = layer_dir / "k.kktv"; vp = layer_dir / "v.kktv"
        write_kktv(kp, k_arr.astype(np.float32, copy=False))
        write_kktv(vp, v_arr.astype(np.float32, copy=False))

        k_tgt = rsvd_target_rank_k if rsvd_target_rank_k is not None else rsvd_target_rank
        v_tgt = rsvd_target_rank_v if rsvd_target_rank_v is not None else rsvd_target_rank
        k_bw = bit_width_k if bit_width_k is not None else bit_width
        v_bw = bit_width_v if bit_width_v is not None else bit_width
        k_rep = bench_one(kp, layer_dir / "k.json",
                          metric="inner_product", share_basis=False,
                          block_size=block_size, variance_ratio=variance_ratio,
                          k=k, bit_width=k_bw, seed=rotation_seed,
                          pca_method=pca_method,
                          rsvd_target_rank=k_tgt,
                          rsvd_oversample=rsvd_oversample,
                          rsvd_power_iters=rsvd_power_iters)
        v_rep = bench_one(vp, layer_dir / "v.json",
                          metric="mse", share_basis=True,
                          block_size=block_size, variance_ratio=variance_ratio,
                          k=k, bit_width=v_bw, seed=rotation_seed + 1,
                          pca_method=pca_method,
                          rsvd_target_rank=v_tgt,
                          rsvd_oversample=rsvd_oversample,
                          rsvd_power_iters=rsvd_power_iters)

        seq = int(k_arr.shape[0])
        n_full_blocks = seq // block_size
        tail = seq - n_full_blocks * block_size
        tail_bytes = 2 * tail * head_dim  # K + V tail, bf16

        layer_comp = k_rep["compressed_bytes"] + v_rep["compressed_bytes"] + tail_bytes

        per_layer.append({
            "layer_idx": i, "layer_type": lt, "is_sliding": False,
            "head_dim": head_dim, "seq": seq, "baseline_bf16": layer_base,
            "tail_tokens": tail, "tail_bytes_bf16": tail_bytes,
            "k_report": k_rep, "v_report": v_rep,
            "compressed_bytes": layer_comp, "compression_source": "kakeyaturbo_v1_2",
        })
        tot_base += layer_base; tot_comp += layer_comp
        tot_full_base += layer_base; tot_full_comp += layer_comp

        kp.unlink(missing_ok=True); vp.unlink(missing_ok=True)

    summary = {
        "model_name": model_name, "model_meta": meta,
        "codec_params": {"block_size": block_size, "variance_ratio": variance_ratio,
                         "k": k, "bit_width": bit_width, "rotation_seed": rotation_seed,
                         "version": "v1.2_A_plus_B_prime"},
        "per_layer": per_layer,
        "totals": {
            "baseline_bf16_bytes": tot_base,
            "compressed_bytes": tot_comp,
            "ratio_bf16": tot_base / tot_comp if tot_comp else None,
            "full_attn_baseline_bf16_bytes": tot_full_base,
            "full_attn_compressed_bytes": tot_full_comp,
            "full_attn_ratio_bf16": tot_full_base / tot_full_comp if tot_full_comp else None,
            "sliding_bytes_passthrough": tot_slide_base,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--context-tokens", type=int, required=True)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--variance-ratio", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--bit-width", type=int, default=3)
    ap.add_argument("--rotation-seed", type=int, default=0xCAFEBABE)
    ap.add_argument("--prefill-chunk", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pca-method", choices=["exact", "randomized"], default="exact")
    ap.add_argument("--rsvd-target-rank", type=int, default=None,
                    help="Symmetric target_rank for both K and V streams")
    ap.add_argument("--rsvd-target-rank-k", type=int, default=None,
                    help="K-stream target_rank; falls back to --rsvd-target-rank if unset")
    ap.add_argument("--rsvd-target-rank-v", type=int, default=None,
                    help="V-stream target_rank; falls back to --rsvd-target-rank if unset")
    ap.add_argument("--rsvd-oversample", type=int, default=8)
    ap.add_argument("--rsvd-power-iters", type=int, default=2)
    ap.add_argument("--bit-width-k", type=int, default=None,
                    help="Per-stream override for K; falls back to --bit-width if unset")
    ap.add_argument("--bit-width-v", type=int, default=None,
                    help="Per-stream override for V; falls back to --bit-width if unset")
    args = ap.parse_args()

    if not BENCH_BIN.exists():
        print(f"error: build first (cargo build --release --bin kakeyaturbo-bench)", file=sys.stderr)
        sys.exit(1)

    s = run_model(args.model_path, args.model_name, args.context_tokens,
                  args.prefill_chunk, args.block_size, args.variance_ratio,
                  args.k, args.bit_width, args.rotation_seed, args.out_dir,
                  pca_method=args.pca_method,
                  rsvd_target_rank=args.rsvd_target_rank,
                  rsvd_oversample=args.rsvd_oversample,
                  rsvd_power_iters=args.rsvd_power_iters,
                  rsvd_target_rank_k=args.rsvd_target_rank_k,
                  rsvd_target_rank_v=args.rsvd_target_rank_v,
                  bit_width_k=args.bit_width_k,
                  bit_width_v=args.bit_width_v)
    t = s["totals"]
    print(f"\n===== {args.model_name} @ {args.context_tokens} tokens =====")
    print(f"baseline (bf16) total     : {t['baseline_bf16_bytes']/1024/1024:.2f} MiB")
    print(f"kakeyaturbo v1.2 compressed: {t['compressed_bytes']/1024/1024:.2f} MiB")
    if t["ratio_bf16"]:
        print(f"total bf16 compression    : {t['ratio_bf16']:.3f}x")
    if t["full_attn_ratio_bf16"]:
        print(f"full-attn bf16 compression: {t['full_attn_ratio_bf16']:.3f}x")


if __name__ == "__main__":
    main()
