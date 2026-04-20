#!/usr/bin/env python3
"""Drive the stage-by-stage codec ablation and report per-stage SNR /
MSE on a real KV tensor.

Stages:
  1. PCA only                          — isolates PCA/d_eff truncation error
  2. PCA + K-means (exact residual)    — should equal PCA (K-means is exact)
  3. PCA + K-means + WHT round-trip    — isolates WHT numerical error (should be ~exact)
  4. Full encode+decode (with Lloyd-Max quantization) — the real decode path

If stage 1 has low error and stage 4 has high error, the gap is caused by
whichever stage introduces the jump. By design stage 2 should == stage 1
and stage 3 should == stage 2 (if codec is correct); the actual error
budget lives in stage 3 -> stage 4, which is the quantization step.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / "kakeyaturbo" / "target" / "release" / "stage-by-stage-decode"
MAGIC = 0x4B4B5456


def write_kktv(path: Path, arr: np.ndarray):
    assert arr.dtype == np.float32 and arr.ndim == 2
    n, d = arr.shape
    with path.open("wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<I", d))
        f.write(struct.pack("<I", 0))
        f.write(arr.tobytes("C"))


def read_kktv(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        assert struct.unpack("<I", f.read(4))[0] == MAGIC
        _ = f.read(4)
        n = struct.unpack("<Q", f.read(8))[0]
        d = struct.unpack("<I", f.read(4))[0]
        _ = f.read(4)
        return np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d).copy()


def capture_layer_kv(model_path: str, layer_idx: int, n_tokens: int = 2048):
    """Run prefill and extract K/V for one layer. Returns (k_flat, v_flat)."""
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.eval()
    # Long prompt
    text = ("The quick brown fox jumps over the lazy dog. " * 100)
    ids = tok(text, return_tensors="pt")["input_ids"][:, :n_tokens]
    cache = DynamicCache(config=model.config)
    with torch.inference_mode():
        _ = model(input_ids=ids, past_key_values=cache, use_cache=True)
    layer = cache.layers[layer_idx]
    k = layer.keys.to(torch.float32).cpu().numpy()
    v = layer.values.to(torch.float32).cpu().numpy()
    # flatten [bsz, n_kv, seq, hd] -> [bsz*n_kv*seq, hd]
    k_flat = k.reshape(-1, k.shape[-1])
    v_flat = v.reshape(-1, v.shape[-1])
    return k_flat, v_flat


def run_stages(arr: np.ndarray, block_size: int, bit_width: int,
               pca_method: str, variance_ratio: float,
               rsvd_target_rank: int | None = None,
               tmp_dir: Path | None = None) -> list[np.ndarray]:
    import tempfile
    with tempfile.TemporaryDirectory(dir=str(tmp_dir) if tmp_dir else None) as td:
        tdp = Path(td)
        in_p = tdp / "x.kktv"
        write_kktv(in_p, arr)
        paths = [tdp / f"s{i}.kktv" for i in range(1, 5)]
        cmd = [
            str(BIN),
            "--input", str(in_p),
            "--stage1-out", str(paths[0]),
            "--stage2-out", str(paths[1]),
            "--stage3-out", str(paths[2]),
            "--stage4-out", str(paths[3]),
            "--block-size", str(block_size),
            "--variance-ratio", str(variance_ratio),
            "--k", "16",
            "--bit-width", str(bit_width),
            "--pca-method", pca_method,
        ]
        if pca_method == "randomized":
            rank = rsvd_target_rank or max(2, arr.shape[1] // 2)
            cmd += ["--rsvd-target-rank", str(rank)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        return [read_kktv(p) for p in paths]


def report(arr: np.ndarray, stages: list[np.ndarray], label: str):
    sig_var = arr.var()
    print(f"\n=== {label} ===")
    print(f"input:       shape={arr.shape} var={sig_var:.4e} rmse={np.sqrt(sig_var):.3f}")
    stage_names = [
        "s1  PCA only               ",
        "s2  + kmeans center/resid   ",
        "s3  + WHT round-trip (no Q) ",
        "s4  + Lloyd-Max quantization",
    ]
    prev_mse = 0.0
    for name, rec in zip(stage_names, stages):
        diff = arr - rec
        mse = (diff ** 2).mean()
        rmse = float(np.sqrt(mse))
        snr = sig_var / max(mse, 1e-30)
        correl = float(np.corrcoef(arr.flatten(), rec.flatten())[0, 1])
        delta = mse - prev_mse
        print(f"  {name}: mse={mse:.4e} rmse={rmse:.4f} snr={snr:6.2f}x corr={correl:.4f} Δ={delta:+.4e}")
        prev_mse = mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--n-tokens", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=512)
    args = ap.parse_args()

    print(f"Capturing layer {args.layer} K/V from {args.model_path}…")
    k_flat, v_flat = capture_layer_kv(args.model_path, args.layer, args.n_tokens)
    # trim to block-aligned
    n_blocks = k_flat.shape[0] // args.block_size
    n_align = n_blocks * args.block_size
    k_flat = k_flat[:n_align].astype(np.float32)
    v_flat = v_flat[:n_align].astype(np.float32)
    print(f"K: {k_flat.shape}  V: {v_flat.shape}  n_blocks={n_blocks}")

    configs = [
        ("b=4 vr=1.0 exact  (max fidelity)",  4, "exact",      1.0,  None),
        ("b=4 vr=0.95 exact (quality-biased)", 4, "exact",      0.95, None),
        ("b=3 vr=0.95 exact (v1.2 default)",  3, "exact",      0.95, None),
        ("b=2 vr=0.95 exact",                  2, "exact",      0.95, None),
        ("b=2 vr=0.95 rsvd r=D/2 (v1.3)",     2, "randomized", 0.95, k_flat.shape[1] // 2),
    ]

    for cfg_name, bw, method, vr, rank in configs:
        stages_k = run_stages(k_flat, args.block_size, bw, method, vr, rank)
        report(k_flat, stages_k, f"K stream, {cfg_name}")
        stages_v = run_stages(v_flat, args.block_size, bw, method, vr, rank)
        report(v_flat, stages_v, f"V stream, {cfg_name}")


if __name__ == "__main__":
    main()
