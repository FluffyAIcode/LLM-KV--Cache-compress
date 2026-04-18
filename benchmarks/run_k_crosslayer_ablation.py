#!/usr/bin/env python3
"""K-stream cross-layer shared-basis ablation driver.

For every model we capture all full-attention K streams, write them
out as KKTV tensor files, build a manifest `<layer_idx>:<path>` and
invoke `kakeyaturbo-k-crosslayer-ablation`. That binary runs the full
v1.2 MSE codec under three basis-sharing strategies:

  1. per_block      — one PCA per block (v1.2 K default)
  2. per_layer_pool — one PCA per layer (share_basis=true within layer)
  3. per_type_pool  — one PCA for ALL full-attention layers of the model

Outputs per-model summary.json and a global_summary.json containing
aggregate MSE inflation and byte savings relative to per_block.

No mock, no fallback, no simplification — the same monomorphic MSE
kernel that the bench binary uses.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

REPO = Path(__file__).resolve().parent.parent
ABLATION_BIN = REPO / "kakeyaturbo" / "target" / "release" / "kakeyaturbo-k-crosslayer-ablation"
KKTV_MAGIC = 0x4B4B5456

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
    ("Qwen2.5-0.5B-Instruct", "qwen2_5_0_5b", 4096, 0),
    ("Qwen3-0.6B", "qwen3_0_6b", 4096, 0),
    ("gemma-4-E2B-it", "gemma4_e2b", 4096, 0),
    ("DeepSeek-R1-Distill-Qwen-1.5B", "deepseek_r1_distill_qwen_1_5b", 4096, 0),
    ("glm-edge-1.5b-chat", "glm_edge_1_5b", 4096, 0),
    ("SmolLM2-1.7B-Instruct", "smollm2_1_7b", 4096, 1024),
    ("glm-edge-4b-chat", "glm_edge_4b", 4096, 1024),
]


def build_long_prompt(tokenizer, target: int):
    text = LONG_PROMPT_SEED
    while True:
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        if ids.shape[-1] >= target:
            return ids[:, :target]
        text += LONG_PROMPT_SEED


def write_kktv(path: Path, tensor: np.ndarray):
    assert tensor.dtype == np.float32 and tensor.ndim == 2
    n, d = tensor.shape
    with path.open("wb") as f:
        f.write(struct.pack("<I", KKTV_MAGIC))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<I", d))
        f.write(struct.pack("<I", 0))
        f.write(tensor.tobytes(order="C"))


@torch.inference_mode()
def capture_k_streams(model_dir: str, ctx: int, chunk: int):
    path = f"{REPO}/models/{model_dir}"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager")
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

    ks = []
    for i, layer in enumerate(cache.layers):
        if non_shared[i] != "full_attention":
            ks.append(None); continue
        k = getattr(layer, "keys", None)
        if k is None or k.numel() == 0:
            ks.append(None); continue
        ks.append(k.to(torch.float32).cpu().reshape(-1, k.shape[-1]).contiguous().numpy())

    meta = {
        "ctx": int(ids.shape[-1]),
        "prefill_seconds": round(prefill, 2),
        "head_dim": getattr(text_cfg, "head_dim", None) or (text_cfg.hidden_size // text_cfg.num_attention_heads),
        "global_head_dim": getattr(text_cfg, "global_head_dim", None),
        "num_kv_heads": text_cfg.num_key_value_heads,
        "num_full_layers": sum(1 for t in non_shared if t == "full_attention"),
    }
    del model, cache
    return ks, non_shared, meta


def run_crosslayer(manifest_path: Path, report_path: Path, block_size, variance_ratio, k, bit_width):
    r = subprocess.run(
        [
            str(ABLATION_BIN),
            "--manifest", str(manifest_path),
            "--output", str(report_path),
            "--block-size", str(block_size),
            "--variance-ratio", f"{variance_ratio}",
            "--k", str(k),
            "--bit-width", str(bit_width),
        ],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"xlayer failed ({r.returncode}): {r.stderr}")
    return json.loads(report_path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--variance-ratio", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--bit-width", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports" / "k_crosslayer_ablation")
    ap.add_argument("--only-model", type=str, default=None)
    ap.add_argument("--keep-tensors", action="store_true",
                    help="retain the per-layer .kktv dumps after the run (default: delete)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    global_rows = []
    for dir_name, short, ctx, chunk in MODELS:
        if args.only_model and short != args.only_model:
            continue
        print(f"\n==== {short} (ctx={ctx}) ====", flush=True)
        ks, layer_types, meta = capture_k_streams(dir_name, ctx, chunk)
        print(
            f"  prefill {meta['prefill_seconds']}s, {meta['num_full_layers']} full-attn layers",
            flush=True,
        )
        model_out = args.out_dir / short
        model_out.mkdir(exist_ok=True)

        # Dump each full-attn K stream and build a manifest.
        dumps = []
        manifest_lines = []
        for li, lt in enumerate(layer_types):
            if lt != "full_attention" or ks[li] is None:
                continue
            arr = ks[li]
            if arr.shape[0] < args.block_size:
                continue
            dump = model_out / f"layer_{li:02d}_K.kktv"
            write_kktv(dump, arr.astype(np.float32, copy=False))
            dumps.append(dump)
            manifest_lines.append(f"{li}:{dump}")

        if not manifest_lines:
            print(f"  [{short}] no usable full-attn layers at block_size={args.block_size}, skipping")
            continue

        manifest_path = model_out / "manifest.txt"
        manifest_path.write_text("\n".join(manifest_lines) + "\n")

        rep = run_crosslayer(
            manifest_path, model_out / "report.json",
            args.block_size, args.variance_ratio, args.k, args.bit_width,
        )

        if not args.keep_tensors:
            for d in dumps:
                d.unlink(missing_ok=True)

        # Print strategy summary.
        s1 = rep["per_block"]
        s2 = rep["per_layer_pooled"]
        s3 = rep["per_type_pooled"]
        print(f"  [{short}] strategies (block_size={args.block_size}):")
        print(
            f"    per_block         : mse={s1['mean_mse']:.3e}  bytes={s1['total_payload_bytes']}"
        )
        print(
            f"    per_layer_pooled  : mse={s2['mean_mse']:.3e}  "
            f"inflate={s2['mse_inflation_vs_per_block']:.3f}x  "
            f"bytes={s2['total_payload_bytes']} ({s2['bytes_vs_per_block']:.3f}x)"
        )
        print(
            f"    per_type_pooled   : mse={s3['mean_mse']:.3e}  "
            f"inflate={s3['mse_inflation_vs_per_block']:.3f}x  "
            f"bytes={s3['total_payload_bytes']} ({s3['bytes_vs_per_block']:.3f}x)"
            f"  (pooled d_eff={rep['per_type_pooled_d_eff']})"
        )

        summary = {
            "model": short,
            "ctx": ctx,
            "meta": meta,
            "block_size": args.block_size,
            "variance_ratio": args.variance_ratio,
            "k": args.k,
            "bit_width": args.bit_width,
            "num_layers": rep["num_layers"],
            "per_type_pooled_d_eff": rep["per_type_pooled_d_eff"],
            "per_type_pooled_pca_bytes": rep["per_type_pooled_pca_bytes"],
            "strategies": {
                "per_block": s1,
                "per_layer_pooled": s2,
                "per_type_pooled": s3,
            },
        }
        (model_out / "summary.json").write_text(json.dumps(summary, indent=2))
        global_rows.append(summary)

    (args.out_dir / "global_summary.json").write_text(json.dumps(global_rows, indent=2))
    print(f"\n[done] wrote {args.out_dir / 'global_summary.json'}")


if __name__ == "__main__":
    main()
