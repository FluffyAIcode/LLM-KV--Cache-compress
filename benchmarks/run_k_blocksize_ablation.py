#!/usr/bin/env python3
"""K-stream block_size ablation driver.

For each of the 7 open-source models we capture every full-attention
K-stream (real HF forward pass, bf16, eager attention, long prompt),
then run the `kakeyaturbo-k-blocksize-ablation` binary at several
block sizes. The Rust binary runs the full v1.2 MSE codec pipeline
(PCA + spherical K-means + WHT + Lloyd-Max) — **no mock, no fallback,
no simplification**.

The per-layer JSON reports are reduced into a per-model summary and
then a cross-model summary that answers the question:

> "If we double `block_size` on K only, do we preserve K MSE
> while amortising the skeleton over 2x more rows?"

Written to `reports/k_blocksize_ablation/<model>/…` and
`reports/k_blocksize_ablation/global_summary.json`.
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
ABLATION_BIN = REPO / "kakeyaturbo" / "target" / "release" / "kakeyaturbo-k-blocksize-ablation"
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
    # (dir, short, ctx, prefill_chunk)
    ("Qwen2.5-0.5B-Instruct", "qwen2_5_0_5b", 4096, 0),
    ("Qwen3-0.6B", "qwen3_0_6b", 4096, 0),
    ("gemma-4-E2B-it", "gemma4_e2b", 4096, 0),
    ("DeepSeek-R1-Distill-Qwen-1.5B", "deepseek_r1_distill_qwen_1_5b", 4096, 0),
    ("glm-edge-1.5b-chat", "glm_edge_1_5b", 4096, 0),
    ("SmolLM2-1.7B-Instruct", "smollm2_1_7b", 4096, 1024),
    ("glm-edge-4b-chat", "glm_edge_4b", 4096, 1024),
]

BLOCK_SIZES = [512, 1024, 2048]


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


def run_blocksize_ablation(tensor_path: Path, report_path: Path, block_sizes, variance_ratio, k, bit_width):
    bs_str = ",".join(str(x) for x in block_sizes)
    r = subprocess.run(
        [
            str(ABLATION_BIN),
            "--input", str(tensor_path),
            "--output", str(report_path),
            "--block-sizes", bs_str,
            "--variance-ratio", f"{variance_ratio}",
            "--k", str(k),
            "--bit-width", str(bit_width),
        ],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ablation failed ({r.returncode}): {r.stderr}")
    return json.loads(report_path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variance-ratio", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--bit-width", type=int, default=3)
    ap.add_argument("--block-sizes", type=str, default=",".join(str(x) for x in BLOCK_SIZES))
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports" / "k_blocksize_ablation")
    ap.add_argument("--only-model", type=str, default=None, help="restrict to one model short name")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    block_sizes = [int(x) for x in args.block_sizes.split(",") if x]

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

        per_layer_reports = []
        for li, lt in enumerate(layer_types):
            if lt != "full_attention" or ks[li] is None:
                continue
            arr = ks[li]
            if arr.shape[0] < max(block_sizes):
                # Skip layers that can't support all block sizes — we want
                # apples-to-apples per-layer comparisons.
                continue
            dump = model_out / f"layer_{li:02d}_K.kktv"
            write_kktv(dump, arr.astype(np.float32, copy=False))
            rep = run_blocksize_ablation(
                dump, model_out / f"layer_{li:02d}_K.json",
                block_sizes, args.variance_ratio, args.k, args.bit_width,
            )
            rep["layer_idx"] = li
            rep["head_dim"] = arr.shape[-1]
            per_layer_reports.append(rep)
            dump.unlink(missing_ok=True)
            # Compact per-layer line.
            cells_by_bs = {c["block_size"]: c for c in rep["cells"]}
            msg = f"  L{li:02d} hd={arr.shape[-1]:3d}  "
            msg += "  ".join(
                f"bs={bs}:mse={cells_by_bs[bs]['mean_mse']:.3e}"
                f"({cells_by_bs[bs]['mse_inflation_vs_baseline']:.2f}x)"
                f"/bytes={cells_by_bs[bs]['total_payload_bytes']}"
                f"({cells_by_bs[bs]['bytes_vs_baseline']:.2f}x)"
                for bs in block_sizes
            )
            print(msg, flush=True)

        # Cross-layer aggregation: mean per bs across all layers.
        agg = {bs: {"mse": [], "bytes": [], "d_eff": [], "inflate": [], "byte_ratio": []} for bs in block_sizes}
        for rep in per_layer_reports:
            for cell in rep["cells"]:
                bs = cell["block_size"]
                agg[bs]["mse"].append(cell["mean_mse"])
                agg[bs]["bytes"].append(cell["total_payload_bytes"])
                agg[bs]["d_eff"].append(cell["mean_d_eff"])
                agg[bs]["inflate"].append(cell["mse_inflation_vs_baseline"])
                agg[bs]["byte_ratio"].append(cell["bytes_vs_baseline"])

        summary = {
            "model": short,
            "ctx": ctx,
            "meta": meta,
            "variance_ratio": args.variance_ratio,
            "k": args.k,
            "bit_width": args.bit_width,
            "num_layers_measured": len(per_layer_reports),
            "aggregate": [
                {
                    "block_size": bs,
                    "mean_mse_across_layers": float(np.mean(agg[bs]["mse"])) if agg[bs]["mse"] else None,
                    "total_bytes_across_layers": int(np.sum(agg[bs]["bytes"])) if agg[bs]["bytes"] else 0,
                    "mean_d_eff_across_layers": float(np.mean(agg[bs]["d_eff"])) if agg[bs]["d_eff"] else None,
                    "mean_inflation_vs_baseline": float(np.mean(agg[bs]["inflate"])) if agg[bs]["inflate"] else None,
                    "median_inflation_vs_baseline": float(np.median(agg[bs]["inflate"])) if agg[bs]["inflate"] else None,
                    "max_inflation_vs_baseline": float(np.max(agg[bs]["inflate"])) if agg[bs]["inflate"] else None,
                    "mean_bytes_ratio_vs_baseline": float(np.mean(agg[bs]["byte_ratio"])) if agg[bs]["byte_ratio"] else None,
                }
                for bs in block_sizes
            ],
            "per_layer": per_layer_reports,
        }
        (model_out / "summary.json").write_text(json.dumps(summary, indent=2))
        global_rows.append(summary)

        print(f"  [{short}] cross-layer mean across {summary['num_layers_measured']} layers:")
        for cell in summary["aggregate"]:
            bs = cell["block_size"]
            mse = cell["mean_mse_across_layers"]
            inflate = cell["mean_inflation_vs_baseline"]
            byte_ratio = cell["mean_bytes_ratio_vs_baseline"]
            total = cell["total_bytes_across_layers"]
            print(
                f"    bs={bs:4}  mean_mse={mse:.3e}  inflation={inflate:.3f}x  "
                f"bytes={total} ({byte_ratio:.3f}x)",
                flush=True,
            )

    (args.out_dir / "global_summary.json").write_text(json.dumps(global_rows, indent=2))
    print(f"\n[done] wrote {args.out_dir / 'global_summary.json'}")


if __name__ == "__main__":
    main()
