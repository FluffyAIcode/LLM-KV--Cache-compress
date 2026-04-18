#!/usr/bin/env python3
"""Run the PCA basis-sharing ablation on real KV tensors from all 7 models.

For each model:
  1. Real HF forward pass at a single context length
  2. For each full-attention layer's K and V separately:
     - Dump to KKTV binary
     - Invoke kakeyaturbo-pca-ablation on it
     - Collect the three-way MSE numbers
  3. Aggregate per-model and cross-model summaries.
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
ABLATION_BIN = REPO / "kakeyaturbo" / "target" / "release" / "kakeyaturbo-pca-ablation"
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

MODELS = [
    # (model_dir, short, ctx, prefill_chunk)
    ("Qwen2.5-0.5B-Instruct", "qwen2_5_0_5b", 4096, 0),
    ("Qwen3-0.6B", "qwen3_0_6b", 4096, 0),
    ("gemma-4-E2B-it", "gemma4_e2b", 4096, 0),
    ("DeepSeek-R1-Distill-Qwen-1.5B", "deepseek_r1_distill_qwen_1_5b", 4096, 0),
    ("glm-edge-1.5b-chat", "glm_edge_1_5b", 4096, 0),
    ("SmolLM2-1.7B-Instruct", "smollm2_1_7b", 4096, 1024),
    ("glm-edge-4b-chat", "glm_edge_4b", 4096, 1024),
]


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
def capture(model_dir: str, ctx: int, chunk: int) -> tuple[list, list, list, dict]:
    path = f"{REPO}/models/{model_dir}"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="eager"
    )
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
        k, v = getattr(layer, "keys", None), getattr(layer, "values", None)
        if k is None or v is None or k.numel() == 0:
            ks.append(None); vs.append(None); continue
        ks.append(k.to(torch.float32).cpu().reshape(-1, k.shape[-1]).contiguous().numpy())
        vs.append(v.to(torch.float32).cpu().reshape(-1, v.shape[-1]).contiguous().numpy())

    meta = {
        "ctx": int(ids.shape[-1]),
        "prefill_seconds": round(prefill, 2),
        "num_hidden_layers": text_cfg.num_hidden_layers,
        "num_kv_shared_layers": n_shared,
        "head_dim": getattr(text_cfg, "head_dim", None) or (text_cfg.hidden_size // text_cfg.num_attention_heads),
        "global_head_dim": getattr(text_cfg, "global_head_dim", None),
        "num_kv_heads": text_cfg.num_key_value_heads,
    }
    del model, cache
    return ks, vs, non_shared, meta


def run_ablation(tensor_path: Path, out_json: Path, block_size: int, variance_ratio: float) -> dict:
    cmd = [
        str(ABLATION_BIN),
        "--input", str(tensor_path),
        "--output", str(out_json),
        "--block-size", str(block_size),
        "--variance-ratio", str(variance_ratio),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ablation binary failed ({r.returncode}): {r.stderr}")
    return json.loads(out_json.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--variance-ratio", type=float, default=0.95)
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports" / "pca_ablation")
    args = ap.parse_args()

    if not ABLATION_BIN.exists():
        print(f"error: binary missing at {ABLATION_BIN}\n"
              "  cd kakeyaturbo && cargo build --release --bin kakeyaturbo-pca-ablation",
              file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    global_rows = []

    for model_dir, short, ctx, chunk in MODELS:
        print(f"\n==== {short} (ctx={ctx}) ====", flush=True)
        model_out = args.out_dir / short
        model_out.mkdir(exist_ok=True)

        ks, vs, layer_types, meta = capture(model_dir, ctx, chunk)
        print(f"  prefill {meta['prefill_seconds']}s, {len([t for t in layer_types if t=='full_attention'])} full-attn layers", flush=True)

        model_rows = []
        for li, lt in enumerate(layer_types):
            if lt != "full_attention":
                continue
            k_arr, v_arr = ks[li], vs[li]
            if k_arr is None or v_arr is None:
                continue
            for stream_name, arr in [("K", k_arr), ("V", v_arr)]:
                if arr.shape[0] < 2 * args.block_size:
                    continue
                dump = model_out / f"layer_{li:02d}_{stream_name}.kktv"
                write_kktv(dump, arr.astype(np.float32, copy=False))
                rep_json = model_out / f"layer_{li:02d}_{stream_name}.json"
                rep = run_ablation(dump, rep_json, args.block_size, args.variance_ratio)
                rep["layer_idx"] = li
                rep["stream"] = stream_name
                model_rows.append(rep)
                dump.unlink(missing_ok=True)
                print(
                    f"  L{li:02d} {stream_name} hd={rep['dim']:3d} blocks={rep['n_blocks']:3d}  "
                    f"per_block={rep['per_block_mean_mse']:.3e}  "
                    f"pooled={rep['pooled_over_per_block_mean']:.3f}x  "
                    f"first={rep['first_over_per_block_mean']:.3f}x",
                    flush=True,
                )

        # Aggregate model-level stats: average inflation across (layer, stream).
        if model_rows:
            pool_ratios = [r["pooled_over_per_block_mean"] for r in model_rows]
            first_ratios = [r["first_over_per_block_mean"] for r in model_rows]
            model_summary = {
                "model": short,
                "model_dir": model_dir,
                "ctx": ctx,
                "num_measurements": len(model_rows),
                "pooled_ratio_mean": float(np.mean(pool_ratios)),
                "pooled_ratio_median": float(np.median(pool_ratios)),
                "pooled_ratio_max": float(np.max(pool_ratios)),
                "first_ratio_mean": float(np.mean(first_ratios)),
                "first_ratio_median": float(np.median(first_ratios)),
                "first_ratio_max": float(np.max(first_ratios)),
                "per_measurement": model_rows,
                "meta": meta,
            }
            (model_out / "summary.json").write_text(json.dumps(model_summary, indent=2))
            global_rows.append(model_summary)
            print(f"  [{short}] pooled inflation: mean {model_summary['pooled_ratio_mean']:.3f}x  median {model_summary['pooled_ratio_median']:.3f}x  max {model_summary['pooled_ratio_max']:.3f}x")
            print(f"  [{short}] first  inflation: mean {model_summary['first_ratio_mean']:.3f}x  median {model_summary['first_ratio_median']:.3f}x  max {model_summary['first_ratio_max']:.3f}x")

    (args.out_dir / "global_summary.json").write_text(json.dumps(global_rows, indent=2))
    print(f"\n[done] wrote {args.out_dir / 'global_summary.json'}")


if __name__ == "__main__":
    main()
