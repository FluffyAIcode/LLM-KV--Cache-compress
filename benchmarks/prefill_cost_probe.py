#!/usr/bin/env python3
"""Measure the prefill cost of each model at a short context (1k) so we
can extrapolate the wall-clock for the full 2k/4k/8k matrix before
committing to the long run.
"""

import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODELS = [
    ("Qwen2.5-0.5B-Instruct", "qwen2_5_0_5b"),
    ("Qwen3-0.6B", "qwen3_0_6b"),
    ("gemma-4-E2B-it", "gemma4_e2b"),
    ("SmolLM2-1.7B-Instruct", "smollm2_1_7b"),
    ("DeepSeek-R1-Distill-Qwen-1.5B", "deepseek_r1_distill_qwen_1_5b"),
    ("glm-edge-1.5b-chat", "glm_edge_1_5b"),
    ("glm-edge-4b-chat", "glm_edge_4b"),
]

PROBE_CTX = 1024
PROMPT = "You are a careful technical writer. " * 512


@torch.inference_mode()
def probe(path: str, name: str) -> dict:
    tok = AutoTokenizer.from_pretrained(path)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()
    load_elapsed = time.perf_counter() - t0

    ids = tok(PROMPT, return_tensors="pt")["input_ids"][:, :PROBE_CTX]
    cache = DynamicCache(config=model.config)

    t0 = time.perf_counter()
    _ = model(input_ids=ids, past_key_values=cache, use_cache=True)
    prefill_1k = time.perf_counter() - t0

    text_cfg = model.config.get_text_config(decoder=True)
    n_heads = text_cfg.num_attention_heads
    n_kv = text_cfg.num_key_value_heads
    hd = getattr(text_cfg, "head_dim", None) or (text_cfg.hidden_size // n_heads)
    ghd = getattr(text_cfg, "global_head_dim", None)
    sw = getattr(text_cfg, "sliding_window", None)
    lt = getattr(text_cfg, "layer_types", None)
    n_shared = getattr(text_cfg, "num_kv_shared_layers", 0) or 0
    n_cached = text_cfg.num_hidden_layers - n_shared
    n_full = (
        sum(1 for t in lt if t == "full_attention") if lt else (0 if sw else n_cached)
    )
    n_slide = (
        sum(1 for t in lt if t == "sliding_attention") if lt else (n_cached if sw else 0)
    )

    # KV bytes per token @ ctx L (bf16, K+V, per cached layer):
    #   full    = 2 * n_kv * (ghd or hd) * 2 bytes
    #   sliding = 2 * n_kv * hd          * 2 bytes  (capped by sliding_window at long ctx)
    bytes_per_tok_full = 2 * n_kv * (ghd or hd) * 2
    bytes_per_tok_slide = 2 * n_kv * hd * 2
    # At PROBE_CTX=1024 sliding layers are not yet capped for sw=512 except Gemma 4; the
    # estimate below uses min(ctx, sw-1) at long ctx.

    del model, cache

    return {
        "name": name,
        "path": path,
        "load_seconds": round(load_elapsed, 2),
        "prefill_1k_seconds": round(prefill_1k, 2),
        "num_hidden_layers": text_cfg.num_hidden_layers,
        "num_cached_layers": n_cached,
        "num_full_layers": n_full,
        "num_sliding_layers": n_slide,
        "num_attention_heads": n_heads,
        "num_kv_heads": n_kv,
        "head_dim": hd,
        "global_head_dim": ghd,
        "sliding_window": sw,
        "bytes_per_tok_full_layer_bf16": bytes_per_tok_full,
        "bytes_per_tok_slide_layer_bf16": bytes_per_tok_slide,
    }


def main() -> None:
    out = []
    for dir_name, short in MODELS:
        path = f"/workspace/models/{dir_name}"
        print(f"[probe] {short} …", flush=True)
        r = probe(path, short)
        out.append(r)
        print(f"   load={r['load_seconds']}s prefill_1k={r['prefill_1k_seconds']}s "
              f"layers={r['num_hidden_layers']} cached={r['num_cached_layers']} "
              f"full={r['num_full_layers']} slide={r['num_sliding_layers']} "
              f"hd={r['head_dim']} n_kv={r['num_kv_heads']}", flush=True)
    Path("/workspace/reports/real_kakeyaturbo/prefill_cost_probe.json").parent.mkdir(parents=True, exist_ok=True)
    Path("/workspace/reports/real_kakeyaturbo/prefill_cost_probe.json").write_text(
        json.dumps(out, indent=2)
    )


if __name__ == "__main__":
    main()
