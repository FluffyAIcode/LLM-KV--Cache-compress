#!/usr/bin/env python3
"""RoPE-aware K POC.

Hypothesis: Qwen/DeepSeek family's K MSE quality tax comes from RoPE
injecting per-position phase rotation into every K vector. If we
invert RoPE before compression (bring every K back into the pre-RoPE
coordinate system), the resulting "pre-RoPE K" should live on a much
simpler, low-rank subspace and compress dramatically better — both
in bytes and MSE — than raw post-RoPE K.

Pipeline (all real, no mock):
  1. Capture post-RoPE K from a real HF forward pass (same prompt
     as the v1.2 sweep), dump as KKTV.
  2. Inverse-RoPE every K vector in place using the exact rotation
     angles used by the model's attention (theta, positional freq).
     Dump as a second KKTV.
  3. Run the kakeyaturbo-bench binary at bit_width=2, randomized
     PCA (r=D/2), on BOTH files.
  4. Report K MSE delta, K bytes delta, reconstruction bytes delta.
  5. Print Apples-to-apples comparison against the existing ctx=4096
     v1.2 b=2 exact baseline already measured on Qwen3-0.6B / DeepSeek.

This does NOT yet change the codec — it only validates whether
inverse-RoPE would unlock structural byte savings on K. If K MSE
drops by an order of magnitude and K bytes shrink by 20%+, v1.3 ships
with a RoPE-aware K path; otherwise we acknowledge the RoPE-based
20% byte gap as structural.
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

REPO = Path(__file__).resolve().parent.parent
BENCH_BIN = REPO / "kakeyaturbo" / "target" / "release" / "kakeyaturbo-bench"
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


def build_long_prompt(tok, target_ctx: int):
    text = LONG_PROMPT_SEED
    while True:
        ids = tok(text, return_tensors="pt")["input_ids"]
        if ids.shape[-1] >= target_ctx:
            return ids[:, :target_ctx]
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


def rope_freqs(head_dim: int, base: float = 10000.0):
    """Standard RoPE inv-freq: theta_i = base^(-2i/head_dim), i = 0..head_dim/2-1."""
    idx = np.arange(0, head_dim, 2, dtype=np.float64)
    return 1.0 / (base ** (idx / head_dim))


def inverse_rope(k: np.ndarray, head_dim: int, base: float) -> np.ndarray:
    """Undo RoPE on a flattened `[n_vecs, head_dim]` tensor where
    `n_vecs = seq_len * n_kv_heads`. RoPE was applied per-token with
    angle = position * inv_freq, so inverse-RoPE is angle = -position * inv_freq.

    IMPORTANT: the rotation is on pairs (k[2i], k[2i+1]); HF's Llama
    style pairs the first half with the second half instead. This POC
    uses the **adjacent-pairs** convention which is what Qwen/GPT-NeoX
    models use. If the model is Llama-style, set `half_split=True`.
    """
    raise NotImplementedError("use inverse_rope_paired below")


def inverse_rope_adjacent(k_per_token: np.ndarray, positions: np.ndarray,
                           inv_freq: np.ndarray) -> np.ndarray:
    """Input shape: [seq_len, n_kv_heads, head_dim]. Rotate back each
    (even, odd) pair by angle = -position * inv_freq[pair_idx].
    """
    seq_len, n_heads, d = k_per_token.shape
    half = d // 2
    out = k_per_token.copy()
    # angle[s, j] = positions[s] * inv_freq[j], j = 0..half-1.
    angles = np.outer(positions.astype(np.float64), inv_freq)  # [seq, half]
    cos = np.cos(angles).astype(np.float32)  # [seq, half]
    sin = np.sin(angles).astype(np.float32)
    # Adjacent-pairs: out[s,h,2j] = k[2j] * cos + k[2j+1] * sin  (inverse of forward)
    #                  out[s,h,2j+1] = -k[2j] * sin + k[2j+1] * cos
    even = k_per_token[..., 0::2]  # [seq, heads, half]
    odd = k_per_token[..., 1::2]
    cos_b = cos[:, None, :]  # broadcast over heads
    sin_b = sin[:, None, :]
    out_even = even * cos_b + odd * sin_b
    out_odd = -even * sin_b + odd * cos_b
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


def inverse_rope_halfsplit(k_per_token: np.ndarray, positions: np.ndarray,
                             inv_freq: np.ndarray) -> np.ndarray:
    """Llama/Qwen2/Qwen3-style: rotate with pairs (j, j + half), not
    (2j, 2j+1). Inverse angle = -position * inv_freq[j].
    """
    seq_len, n_heads, d = k_per_token.shape
    half = d // 2
    out = k_per_token.copy()
    angles = np.outer(positions.astype(np.float64), inv_freq)  # [seq, half]
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    # forward: y[j] = x[j]*cos - x[j+half]*sin
    #          y[j+half] = x[j]*sin + x[j+half]*cos
    # inverse: x[j] = y[j]*cos + y[j+half]*sin
    #          x[j+half] = -y[j]*sin + y[j+half]*cos
    low = k_per_token[..., :half]
    high = k_per_token[..., half:]
    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]
    out_low = low * cos_b + high * sin_b
    out_high = -low * sin_b + high * cos_b
    out[..., :half] = out_low
    out[..., half:] = out_high
    return out


@torch.inference_mode()
def capture_k(model_dir: str, ctx: int, prefill_chunk: int):
    path = f"{REPO}/models/{model_dir}"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()
    cfg = model.config.get_text_config(decoder=True)
    layer_types = getattr(cfg, "layer_types", None)
    n_shared = getattr(cfg, "num_kv_shared_layers", 0) or 0
    if layer_types is None:
        sw = getattr(cfg, "sliding_window", None) or getattr(cfg, "attention_chunk_size", None)
        layer_types = ["sliding_attention" if sw else "full_attention"] * cfg.num_hidden_layers
    non_shared = list(layer_types)[: cfg.num_hidden_layers - n_shared] if n_shared else list(layer_types)

    ids = build_long_prompt(tok, ctx)
    cache = DynamicCache(config=model.config)
    t0 = time.perf_counter()
    if prefill_chunk <= 0 or ids.shape[-1] <= prefill_chunk:
        _ = model(input_ids=ids, past_key_values=cache, use_cache=True)
    else:
        for s in range(0, ids.shape[-1], prefill_chunk):
            e = min(s + prefill_chunk, ids.shape[-1])
            _ = model(input_ids=ids[:, s:e], past_key_values=cache, use_cache=True)
    prefill = time.perf_counter() - t0

    ks = []
    for i, layer in enumerate(cache.layers):
        if non_shared[i] != "full_attention":
            ks.append(None); continue
        k = getattr(layer, "keys", None)
        if k is None or k.numel() == 0:
            ks.append(None); continue
        # k shape: [bsz=1, n_kv_heads, seq, head_dim]
        k_np = k.to(torch.float32).cpu().numpy()
        # reshape to [seq, n_kv_heads, head_dim] so positions are the leading axis
        k_reshaped = np.transpose(k_np[0], (1, 0, 2))  # [seq, n_kv_heads, head_dim]
        ks.append(k_reshaped)

    meta = {
        "ctx": int(ids.shape[-1]),
        "prefill_s": round(prefill, 2),
        "num_full_layers": sum(1 for t in non_shared if t == "full_attention"),
        "head_dim": getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads),
        "num_key_value_heads": cfg.num_key_value_heads,
        "rope_theta": float(getattr(cfg, "rope_theta", None)
                             or (getattr(cfg, "rope_scaling", None) or {}).get("rope_theta", 10000.0)),
    }
    del model, cache
    return ks, non_shared, meta


def run_bench(kktv_path: Path, out_json: Path, bit_width: int, rsvd_target: int, rsvd_oversample: int,
              rsvd_power_iters: int, share_basis: bool = False, metric: str = "inner_product"):
    cmd = [
        str(BENCH_BIN),
        "--input", str(kktv_path), "--output", str(out_json),
        "--metric", metric, "--block-size", "512",
        "--variance-ratio", "0.95", "--k", "16", "--bit-width", str(bit_width),
        "--rotation-seed", "3405691582",
        "--pca-method", "randomized",
        "--rsvd-target-rank", str(rsvd_target),
        "--rsvd-oversample", str(rsvd_oversample),
        "--rsvd-power-iters", str(rsvd_power_iters),
        "--verify",
    ]
    if share_basis:
        cmd.append("--share-basis")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bench failed: {r.stderr}")
    return json.loads(out_json.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--prefill-chunk", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rope-pairing", choices=["adjacent", "halfsplit"], default="halfsplit",
                    help="RoPE coordinate pairing style. Qwen/Llama = halfsplit, GPT-NeoX = adjacent")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ks, layer_types, meta = capture_k(args.model_dir, args.ctx, args.prefill_chunk)
    meta["rope_pairing"] = args.rope_pairing
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{args.model_name}] captured {meta['num_full_layers']} full-attn K layers "
          f"@ ctx={meta['ctx']} in {meta['prefill_s']}s")
    print(f"  head_dim={meta['head_dim']}  rope_theta={meta['rope_theta']}")

    rope_fn = inverse_rope_halfsplit if args.rope_pairing == "halfsplit" else inverse_rope_adjacent

    rows = []
    for li, lt in enumerate(layer_types):
        if lt != "full_attention" or ks[li] is None:
            continue
        k_reshaped = ks[li]  # [seq, n_kv_heads, head_dim]
        seq = k_reshaped.shape[0]
        positions = np.arange(seq, dtype=np.float64)
        # Per-layer head_dim: e.g. Gemma-4 uses head_dim=256 for sliding
        # layers and 512 for global full-attn layers.
        hd = k_reshaped.shape[-1]
        inv_freq = rope_freqs(hd, base=meta["rope_theta"])

        # Pre-RoPE K: invert positional rotation
        k_pre = rope_fn(k_reshaped, positions, inv_freq)

        # Flatten both to [seq * n_kv_heads, head_dim] for the codec
        k_post_flat = k_reshaped.reshape(-1, hd).astype(np.float32, copy=False)
        k_pre_flat = k_pre.reshape(-1, hd).astype(np.float32, copy=False)

        layer_dir = args.out_dir / f"layer_{li:02d}"
        layer_dir.mkdir(exist_ok=True)
        kp_post = layer_dir / "k_post.kktv"
        kp_pre = layer_dir / "k_pre.kktv"
        write_kktv(kp_post, k_post_flat)
        write_kktv(kp_pre, k_pre_flat)

        tgt = hd // 2
        rep_post = run_bench(kp_post, layer_dir / "k_post.json", 2, tgt, 8, 2)
        rep_pre = run_bench(kp_pre, layer_dir / "k_pre.json", 2, tgt, 8, 2)
        kp_post.unlink(missing_ok=True); kp_pre.unlink(missing_ok=True)

        row = {
            "layer_idx": li,
            "post_rope_bytes": rep_post["compressed_bytes"],
            "pre_rope_bytes": rep_pre["compressed_bytes"],
            "post_rope_mse": rep_post["mean_block_mse"],
            "pre_rope_mse": rep_pre["mean_block_mse"],
            "post_rope_ratio_bf16": rep_post["ratio_vs_bf16"],
            "pre_rope_ratio_bf16": rep_pre["ratio_vs_bf16"],
            "post_rope_skel_bytes": rep_post["skeleton_bytes"],
            "pre_rope_skel_bytes": rep_pre["skeleton_bytes"],
        }
        rows.append(row)
        print(
            f"  L{li:02d}  post: mse={row['post_rope_mse']:.3e} bytes={row['post_rope_bytes']:>8}  "
            f"pre: mse={row['pre_rope_mse']:.3e} bytes={row['pre_rope_bytes']:>8}  "
            f"ΔMSE={row['pre_rope_mse']/row['post_rope_mse']:.2f}x  "
            f"Δbytes={row['pre_rope_bytes']/row['post_rope_bytes']:.2f}x",
            flush=True,
        )

    # Aggregate excluding layer 0 (RoPE degenerate @ position 0).
    filt = [r for r in rows if r["layer_idx"] != 0]
    n = len(filt)
    post_b = sum(r["post_rope_bytes"] for r in filt)
    pre_b = sum(r["pre_rope_bytes"] for r in filt)
    post_m = sum(r["post_rope_mse"] for r in filt) / max(n, 1)
    pre_m = sum(r["pre_rope_mse"] for r in filt) / max(n, 1)
    summary = {
        "model": args.model_name,
        "ctx": meta["ctx"],
        "rope_pairing": args.rope_pairing,
        "n_full_attn_layers_measured": n,
        "mean_post_rope_mse": post_m,
        "mean_pre_rope_mse": pre_m,
        "mse_ratio_pre_over_post": pre_m / max(post_m, 1e-30),
        "total_post_rope_bytes": post_b,
        "total_pre_rope_bytes": pre_b,
        "bytes_ratio_pre_over_post": pre_b / max(post_b, 1),
        "per_layer": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== {args.model_name} @ ctx={meta['ctx']} ===")
    print(f"mean K MSE:  post-RoPE={post_m:.3e}  pre-RoPE={pre_m:.3e}  "
          f"ratio={pre_m/max(post_m,1e-30):.3f}x")
    print(f"total K bytes: post-RoPE={post_b}  pre-RoPE={pre_b}  "
          f"ratio={pre_b/max(post_b,1):.3f}x")
    if pre_m < 0.9 * post_m and pre_b < 0.9 * post_b:
        print("VERDICT: ACCEPT (both MSE and bytes improved >= 10%)")
    elif pre_m < 1.1 * post_m and pre_b < 0.95 * post_b:
        print("VERDICT: MARGINAL (bytes improved, MSE similar)")
    else:
        print("VERDICT: REJECT (no structural improvement)")


if __name__ == "__main__":
    main()
