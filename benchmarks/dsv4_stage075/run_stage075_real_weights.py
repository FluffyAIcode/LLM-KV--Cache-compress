r"""Stage 0.75 — non-Gaussian audit of V4-Flash KV using TRAINED weights.

Upgrade path from Stage 0.5:
  * Stage 0.5 used random-Gaussian init for the V4 attention + compressor
    weights and fed real Gemma-4-E4B hidden states through them.
  * Stage 0.75 loads the ACTUAL trained V4-Flash weights for layers
    0 (SWA), 2 (c4a), and 3 (c128a), and feeds real Qwen2-0.5B (or any
    other available) hidden states through them.

What this does NOT do:
  * Load MoE experts, shared experts, Hyper-Connection params, embed
    tables — we bypass them by using the host model's hidden states
    directly as input to V4's per-layer attention.
  * Propagate hidden states through V4's own 43 layers — that would
    need MoE weights. We're measuring the KV distribution produced
    by a single isolated V4 attention layer, given external hidden
    states.
  * End-to-end Δppl — needs the full stack.

What this DOES do (the real question the user asked):
  * Produce a rigorous non-Gaussian audit (kurtosis, isotropy ratio,
    Hadamard-whitened variance ratio, RMS W2/σ) of V4-Flash's
    trained KV tensors, for all three attention variants.
  * Compare Stage 0.5 (random weights) vs Stage 0.75 (trained weights)
    directly, so we can say whether the 22% / 15% gains predicted by
    Stage 0.5 transfer to actual V4 deployment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

# Make our Stage 0.5 generator importable.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "dsv4_stage0_5"))
sys.path.insert(0, str(REPO / "benchmarks" / "dsv4_stage075"))

from dsv4_kv_generator import (  # type: ignore[import-not-found]
    DSV4Compressor,
    DSV4FlashArchConfig,
    DSV4MainKVProjection,
    _simulate_fp8_block_quant_dequant,
)
from dsv4_weight_loader import (  # type: ignore[import-not-found]
    inject_weights_into_compressor,
    inject_weights_into_main_kv,
    load_single_layer_weights,
    load_v4_shard_paths,
)

# Borrow the audit + metrics from Stage 0.5's rigorous harness
sys.path.insert(0, str(REPO / "benchmarks" / "dsv4_stage0_5"))
from run_dsv4_stage0_5 import (  # type: ignore[import-not-found]
    compute_cosine,
    compute_rel_mse,
    fp8_baseline_roundtrip,
    non_gaussian_audit,
)

# KakeyaLattice codecs
from kakeyalattice import V14KakeyaZamirLatticeGPU, V15KakeyaZamirE8GPU  # type: ignore


def load_host_hidden(
    model_id: str,
    seqlen: int,
    batch_size: int,
    target_hidden_size: int,
    device: str,
) -> torch.Tensor:
    """Return [B, S, target_hidden_size] bf16 hidden states from the
    host model's embedding layer (projected to 4096 if needed)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[host] loading {model_id} tokenizer + embedding only", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()

    passage = (
        "The history of topology is deeply intertwined with the emergence of "
        "modern mathematics itself. In the late nineteenth century, Henri "
        "Poincaré's study of the three-body problem led him to formulate the "
        "first rigorous ideas about the topology of manifolds. Betti numbers, "
        "originally defined by Enrico Betti in the 1870s as counts of "
        "independent cycles, were gradually reformulated by Poincaré and later "
        "by Emmy Noether into the algebraic language of homology groups. "
    ) * 8

    ids = tok(
        [passage] * batch_size,
        return_tensors="pt", padding="max_length",
        truncation=True, max_length=seqlen,
    )["input_ids"].to(device)

    with torch.inference_mode():
        hidden = model.get_input_embeddings()(ids).to(torch.bfloat16)
    native = hidden.shape[-1]

    # Project to V4's hidden_size=4096 with a fixed-seed linear if needed.
    if native != target_hidden_size:
        print(f"[host] projecting native hidden={native} → {target_hidden_size}", flush=True)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.startswith("cuda") else []):
            torch.manual_seed(20260425)
            if device.startswith("cuda"):
                torch.cuda.manual_seed(20260425)
            W = (torch.randn(target_hidden_size, native, device=device, dtype=torch.bfloat16)
                 * native ** -0.5)
            hidden = torch.nn.functional.linear(hidden, W)

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(f"[host] hidden states ready: {tuple(hidden.shape)} bf16", flush=True)
    return hidden


def build_and_load_dsv4_blocks(
    shard_paths: Dict[int, str],
    device: str,
    config: DSV4FlashArchConfig,
) -> Dict[str, object]:
    """Load trained weights for layer 0 (SWA), layer 2 (c4a), layer 3 (c128a)
    and inject into freshly-built DSV4MainKVProjection + DSV4Compressor
    modules. Returns a dict with keys:
      'main_kv_swa'  : DSV4MainKVProjection — layer 0 trained wkv
      'main_kv_c4a'  : DSV4MainKVProjection — layer 2 trained wkv
      'main_kv_c128a': DSV4MainKVProjection — layer 3 trained wkv
      'compressor_c4a'  : DSV4Compressor ratio=4  — layer 2 compressor
      'compressor_c128a': DSV4Compressor ratio=128 — layer 3 compressor
    """
    print(f"[load] reading trained weights from {len(shard_paths)} shards", flush=True)
    t0 = time.perf_counter()

    blocks: Dict[str, object] = {}

    # Layer 0: SWA-only (no compressor)
    params_layer0 = load_single_layer_weights(shard_paths[2], layer_id=0)
    swa_cfg = DSV4FlashArchConfig(**{**config.__dict__, "compress_ratio": 0})
    blocks["main_kv_swa"] = DSV4MainKVProjection(swa_cfg, device=device)
    inject_weights_into_main_kv(blocks["main_kv_swa"], params_layer0, layer_id=0, device=device)

    # Layer 2: c4a
    params_layer2 = load_single_layer_weights(shard_paths[4], layer_id=2)
    c4a_cfg = DSV4FlashArchConfig(**{**config.__dict__, "compress_ratio": 4})
    blocks["main_kv_c4a"] = DSV4MainKVProjection(c4a_cfg, device=device)
    inject_weights_into_main_kv(blocks["main_kv_c4a"], params_layer2, layer_id=2, device=device)
    blocks["compressor_c4a"] = DSV4Compressor(c4a_cfg, compress_ratio=4, rotate=False, device=device)
    inject_weights_into_compressor(blocks["compressor_c4a"], params_layer2, layer_id=2, device=device)

    # Layer 3: c128a
    params_layer3 = load_single_layer_weights(shard_paths[5], layer_id=3)
    c128a_cfg = DSV4FlashArchConfig(**{**config.__dict__, "compress_ratio": 128})
    blocks["main_kv_c128a"] = DSV4MainKVProjection(c128a_cfg, device=device)
    inject_weights_into_main_kv(blocks["main_kv_c128a"], params_layer3, layer_id=3, device=device)
    blocks["compressor_c128a"] = DSV4Compressor(c128a_cfg, compress_ratio=128, rotate=False, device=device)
    inject_weights_into_compressor(blocks["compressor_c128a"], params_layer3, layer_id=3, device=device)

    t1 = time.perf_counter()
    print(f"[load] weight loading: {t1-t0:.2f}s; "
          f"num params: L0={len(params_layer0)} L2={len(params_layer2)} L3={len(params_layer3)}",
          flush=True)
    return blocks


def run_trio(blocks: Dict[str, object], hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Produce the three KV streams from trained weights."""
    with torch.inference_mode():
        sliding_window_kv = blocks["main_kv_swa"](hidden)     # [B, S, 512]
        csa_pool_kv = blocks["compressor_c4a"](hidden)        # [B, S/4, 512]
        hca_pool_kv = blocks["compressor_c128a"](hidden)      # [B, S/128, 512]

    print(f"[kv] sliding_window_kv {tuple(sliding_window_kv.shape)}", flush=True)
    print(f"[kv] csa_pool_kv_ratio4 {tuple(csa_pool_kv.shape)}", flush=True)
    print(f"[kv] hca_pool_kv_ratio128 {tuple(hca_pool_kv.shape)}", flush=True)
    return {
        "sliding_window_kv": sliding_window_kv,
        "csa_pool_kv_ratio4": csa_pool_kv,
        "hca_pool_kv_ratio128": hca_pool_kv,
    }


def evaluate_stream(name: str, kv: torch.Tensor, codecs: List) -> Dict:
    """Audit + codec roundtrip eval for one stream."""
    result = {
        "stream": name,
        "shape": list(kv.shape),
        "dtype": str(kv.dtype),
        "audit": non_gaussian_audit(kv),
        "codecs": {},
    }
    for codec_name, c in codecs:
        t0 = time.perf_counter()
        kv_hat = c.roundtrip(kv.float())
        if kv.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        result["codecs"][codec_name] = {
            "bits_per_vector": int(c.bits_per_token_per_head),
            "rel_mse": compute_rel_mse(kv, kv_hat),
            "cos_sim": compute_cosine(kv, kv_hat),
            "wall_time_sec": t1 - t0,
        }
    # FP8 baseline
    fp8_hat = fp8_baseline_roundtrip(kv)
    bits_per_vec = kv.shape[-1] * 8 + (kv.shape[-1] // 64) * 16
    result["codecs"]["fp8_per64_baseline"] = {
        "bits_per_vector": bits_per_vec,
        "rel_mse": compute_rel_mse(kv, fp8_hat),
        "cos_sim": compute_cosine(kv, fp8_hat),
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host-model", default="Qwen/Qwen2-0.5B")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--q-values", default="10,38")
    p.add_argument("--enable-e8", action="store_true", default=True)
    p.add_argument("--out", default="reports/v1_5_release/dsv4_stage075/stage075_trained.json")
    p.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/workspace/.hf_home"))
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 0.75 requires CUDA for efficient bf16 matmul on attention forward.")
    device = "cuda"
    if args.seqlen % 128 != 0:
        raise ValueError(f"seqlen must be multiple of 128 (HCA ratio); got {args.seqlen}")

    q_values = [int(q) for q in args.q_values.split(",") if q.strip()]
    print(f"[config] host={args.host_model} seqlen={args.seqlen} batch={args.batch_size} "
          f"q_values={q_values}", flush=True)

    # 1. Locate the downloaded V4-Flash shards
    shard_paths = load_v4_shard_paths(args.hf_home, "deepseek-ai/DeepSeek-V4-Flash")
    for needed in (2, 4, 5):
        if needed not in shard_paths:
            raise FileNotFoundError(
                f"Shard {needed} not found in HF cache at {args.hf_home}. "
                f"Re-run the download script before running Stage 0.75."
            )
    print(f"[shards] found {len(shard_paths)} V4 shards; needed: 2, 4, 5", flush=True)

    # 2. Build DSV4 blocks with trained weights
    cfg = DSV4FlashArchConfig(simulate_fp8=True)  # FP8 on nope dims matches V4 production
    blocks = build_and_load_dsv4_blocks(shard_paths, device=device, config=cfg)

    # 3. Host hidden states
    hidden = load_host_hidden(
        args.host_model, args.seqlen, args.batch_size,
        target_hidden_size=cfg.hidden_size, device=device,
    )

    # 4. Run forward + measure
    streams = run_trio(blocks, hidden)

    # 5. Build codec list
    D = cfg.head_dim  # 512
    codecs = []
    for q in q_values:
        codecs.append((f"v14_d4_Q{q}", V14KakeyaZamirLatticeGPU(D=D, q_range=q, device=device)))
    if args.enable_e8:
        for q in q_values:
            codecs.append((f"v15_e8_Q{q}", V15KakeyaZamirE8GPU(D=D, q_range=q, device=device)))
    for name, c in codecs:
        print(f"[codec] {name}: bits={c.bits_per_token_per_head}", flush=True)

    results = []
    for name, kv in streams.items():
        print(f"\n[stream {name}] shape={tuple(kv.shape)}", flush=True)
        results.append(evaluate_stream(name, kv, codecs))

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "host_model": args.host_model,
            "seqlen": args.seqlen,
            "batch_size": args.batch_size,
            "q_values": q_values,
            "enable_e8": args.enable_e8,
            "simulate_fp8": cfg.simulate_fp8,
            "dsv4_config": {
                "hidden_size": cfg.hidden_size,
                "head_dim": cfg.head_dim,
                "qk_rope_head_dim": cfg.qk_rope_head_dim,
                "v4_layers_used": {0: "SWA", 2: "c4a", 3: "c128a"},
                "weight_source": "deepseek-ai/DeepSeek-V4-Flash safetensors shards 2/4/5",
                "trained_weights": True,
            },
        },
        "results_by_stream": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[out] {out}", flush=True)

    # Human-readable table
    print()
    print(f"{'stream':<25s}  {'codec':<20s}  {'bits':>5s}  {'rel-MSE':>11s}  {'cos':>7s}")
    print("-" * 75)
    for r in results:
        for cn, c in r["codecs"].items():
            print(f"{r['stream']:<25s}  {cn:<20s}  {c['bits_per_vector']:>5d}  "
                  f"{c['rel_mse']:11.4e}  {c['cos_sim']:>7.4f}")

    print()
    print(f"{'stream':<25s}  {'|kurt-3|':>9s}  {'iso-var':>10s}  {'had-var':>10s}  {'W2/σ':>7s}  {'N':>5s}")
    print("-" * 75)
    for r in results:
        a = r["audit"]
        print(f"{r['stream']:<25s}  {a['excess_kurtosis_abs']:>9.3f}  "
              f"{a['isotropy_variance_ratio']:>10.2f}  {a['hadamard_post_variance_ratio']:>10.2f}  "
              f"{a['rms_wasserstein2_over_sigma_per_dim']:>7.3f}  {a['num_vectors']:>5d}")


if __name__ == "__main__":
    main()
