#!/usr/bin/env python3
"""Run the K-side d_eff × d_res ablation on all 7 models' captured K
tensors. Uses real HF forward passes, bf16, eager attention, same
prompt as every other real-benchmark PR.
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
ABLATION_BIN = REPO / "kakeyaturbo" / "target" / "release" / "kakeyaturbo-deff-outlier-ablation"
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

VARIANCE_RATIOS = [0.95, 0.90, 0.85, 0.80, 0.70]
D_RES_VALUES = [0, 2, 4, 8]


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


def run_ablation(tensor_path: Path, report_path: Path, block_size: int):
    vr_str = ",".join(f"{x:.2f}" for x in VARIANCE_RATIOS)
    dr_str = ",".join(str(x) for x in D_RES_VALUES)
    r = subprocess.run([
        str(ABLATION_BIN),
        "--input", str(tensor_path),
        "--output", str(report_path),
        "--block-size", str(block_size),
        "--variance-ratios", vr_str,
        "--d-res-values", dr_str,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ablation failed ({r.returncode}): {r.stderr}")
    return json.loads(report_path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports" / "k_deff_outlier_ablation")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    global_rows = []

    for dir_name, short, ctx, chunk in MODELS:
        print(f"\n==== {short} (ctx={ctx}) ====", flush=True)
        ks, layer_types, meta = capture_k_streams(dir_name, ctx, chunk)
        print(f"  prefill {meta['prefill_seconds']}s, {meta['num_full_layers']} full-attn layers", flush=True)
        model_out = args.out_dir / short
        model_out.mkdir(exist_ok=True)

        # Run per-layer ablations (K only).
        rows = []
        for li, lt in enumerate(layer_types):
            if lt != "full_attention" or ks[li] is None:
                continue
            arr = ks[li]
            if arr.shape[0] < 2 * args.block_size:
                continue
            dump = model_out / f"layer_{li:02d}_K.kktv"
            write_kktv(dump, arr.astype(np.float32, copy=False))
            rep = run_ablation(dump, model_out / f"layer_{li:02d}_K.json", args.block_size)
            rep["layer_idx"] = li
            rep["head_dim"] = arr.shape[-1]
            rows.append(rep)
            dump.unlink(missing_ok=True)
            # Compact line per layer showing inflation grid.
            grid = {}
            for cell in rep["cells"]:
                grid[(cell["variance_ratio"], cell["d_res"])] = cell["inflation_over_baseline"]
            print(
                f"  L{li:02d} hd={arr.shape[-1]:3d} "
                + " ".join(
                    f"{vr:.2f}/{dr}={grid[(vr, dr)]:.2f}x"
                    for vr in VARIANCE_RATIOS for dr in [0, 4]
                ),
                flush=True,
            )

        # Aggregate per-cell across layers: mean / median / max inflation.
        agg = {}
        for rep in rows:
            for cell in rep["cells"]:
                key = (cell["variance_ratio"], cell["d_res"])
                agg.setdefault(key, []).append(cell["inflation_over_baseline"])

        summary = {
            "model": short, "ctx": ctx, "meta": meta,
            "num_layers_measured": len(rows),
            "aggregate": [
                {
                    "variance_ratio": vr, "d_res": dr,
                    "mean_inflation": float(np.mean(agg[(vr, dr)])),
                    "median_inflation": float(np.median(agg[(vr, dr)])),
                    "max_inflation": float(np.max(agg[(vr, dr)])),
                    "p90_inflation": float(np.percentile(agg[(vr, dr)], 90)),
                }
                for vr in VARIANCE_RATIOS for dr in D_RES_VALUES
            ],
            "per_layer": rows,
        }
        (model_out / "summary.json").write_text(json.dumps(summary, indent=2))
        global_rows.append(summary)

        # Print aggregate table for this model.
        print(f"  [{short}] K inflation (mean over {len(rows)} layers):")
        for vr in VARIANCE_RATIOS:
            cells = [f"d_res={dr}:{next(a['mean_inflation'] for a in summary['aggregate'] if a['variance_ratio']==vr and a['d_res']==dr):.3f}x"
                     for dr in D_RES_VALUES]
            print(f"    vr={vr:.2f}  " + "  ".join(cells), flush=True)

    (args.out_dir / "global_summary.json").write_text(json.dumps(global_rows, indent=2))
    print(f"\n[done] wrote {args.out_dir / 'global_summary.json'}")


if __name__ == "__main__":
    main()
