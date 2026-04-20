#!/usr/bin/env python3
"""End-to-end PPL validation of v1.3 codec WITH RoPE-aware K path.

Architecture principle (what was missing from the original harness):
the codec must never see RoPE phase on K. The cache stores post-RoPE K
(attention semantics require this), but the codec receives pre-RoPE K:

    Attention forward → K_post (this is what DynamicCache holds)
    ──────────────────────────────────────────────────────────
    Codec inbound:  K_post  →  RoPE⁻¹(pos)  =  K_pre
                    K_pre   →  encode → decode  =  K̂_pre
                    K̂_pre   →  RoPE(pos)  =  K̂_post  (written back to cache)
    ──────────────────────────────────────────────────────────
    Attention consumes K̂_post during decode forward

Two modes:
  --rope-mode=none    pass K_post directly through codec (wrong, matches
                      original v1.3 harness, kept as baseline)
  --rope-mode=halfsplit  Qwen / Llama / DeepSeek / MiniMax (halfsplit RoPE)
"""
from __future__ import annotations

import argparse
import copy
import json
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

REPO = Path(__file__).resolve().parent.parent
BENCH_BIN = REPO / "kakeyaturbo" / "target" / "release" / "kakeyaturbo-bench"
KKTV_MAGIC = 0x4B4B5456


# -----------------------------------------------------------------------------
# Half-split RoPE forward and inverse (Qwen2 / Qwen3 / Llama / DeepSeek convention)
# -----------------------------------------------------------------------------

def rope_inv_freq(head_dim: int, base: float) -> np.ndarray:
    """Standard RoPE inv-freq over half of head_dim."""
    idx = np.arange(0, head_dim, 2, dtype=np.float64)
    return 1.0 / (base ** (idx / head_dim))


def _cos_sin(positions: np.ndarray, inv_freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cos[s, half], sin[s, half] for positions[s] and inv_freq[half]."""
    angles = np.outer(positions.astype(np.float64), inv_freq)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def rope_halfsplit_forward(k: np.ndarray, positions: np.ndarray,
                            inv_freq: np.ndarray) -> np.ndarray:
    """k shape: [seq, n_kv, head_dim]. Apply RoPE."""
    half = k.shape[-1] // 2
    cos, sin = _cos_sin(positions, inv_freq)  # [seq, half]
    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]
    low = k[..., :half]
    high = k[..., half:]
    out = k.copy()
    out[..., :half] = low * cos_b - high * sin_b
    out[..., half:] = low * sin_b + high * cos_b
    return out


def rope_halfsplit_inverse(k: np.ndarray, positions: np.ndarray,
                            inv_freq: np.ndarray) -> np.ndarray:
    """Undo RoPE. The inverse of forward(angle) is forward(-angle), which
    is the transpose of the rotation block."""
    half = k.shape[-1] // 2
    cos, sin = _cos_sin(positions, inv_freq)
    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]
    low = k[..., :half]
    high = k[..., half:]
    out = k.copy()
    out[..., :half] = low * cos_b + high * sin_b
    out[..., half:] = -low * sin_b + high * cos_b
    return out


def test_rope_roundtrip():
    """Sanity check: rope_forward(rope_inverse(x)) == x."""
    rng = np.random.default_rng(0)
    k = rng.normal(size=(5, 2, 64)).astype(np.float32)
    inv = rope_inv_freq(64, base=1_000_000.0)
    positions = np.arange(5, dtype=np.float64)
    k_pre = rope_halfsplit_inverse(k, positions, inv)
    k_back = rope_halfsplit_forward(k_pre, positions, inv)
    max_err = float(np.abs(k - k_back).max())
    assert max_err < 1e-4, f"RoPE round-trip broken: max err = {max_err}"


# -----------------------------------------------------------------------------
# KKTV I/O
# -----------------------------------------------------------------------------

def write_kktv(path: Path, arr: np.ndarray) -> None:
    assert arr.dtype == np.float32 and arr.ndim == 2, arr.dtype
    n, d = arr.shape
    with path.open("wb") as f:
        f.write(struct.pack("<I", KKTV_MAGIC))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<I", d))
        f.write(struct.pack("<I", 0))
        f.write(arr.tobytes(order="C"))


def read_kktv(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        assert struct.unpack("<I", f.read(4))[0] == KKTV_MAGIC
        _ = f.read(4)
        n = struct.unpack("<Q", f.read(8))[0]
        d = struct.unpack("<I", f.read(4))[0]
        _ = f.read(4)
        return np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d).copy()


def rust_roundtrip(arr: np.ndarray, block_size: int, bit_width: int,
                   rsvd_target_rank: int, metric: str, share_basis: bool,
                   pca_method: str = "randomized",
                   variance_ratio: float = 0.95):
    import tempfile
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        tdp = Path(td)
        in_p = tdp / "x.kktv"
        rep_p = tdp / "r.json"
        dec_p = tdp / "dec.kktv"
        write_kktv(in_p, arr.astype(np.float32, copy=False))
        cmd = [
            str(BENCH_BIN),
            "--input", str(in_p),
            "--output", str(rep_p),
            "--metric", metric,
            "--block-size", str(block_size),
            "--variance-ratio", str(variance_ratio),
            "--k", "16",
            "--bit-width", str(bit_width),
            "--rotation-seed", "3405691582",
            "--pca-method", pca_method,
            "--verify",
            "--dump-decoded", str(dec_p),
        ]
        if pca_method == "randomized":
            cmd += [
                "--rsvd-target-rank", str(rsvd_target_rank),
                "--rsvd-oversample", "8",
                "--rsvd-power-iters", "2",
            ]
        if share_basis:
            cmd.append("--share-basis")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        return read_kktv(dec_p), json.loads(rep_p.read_text())


# -----------------------------------------------------------------------------
# Cache round-trip
# -----------------------------------------------------------------------------

@torch.inference_mode()
def prefill_cache(model, input_ids: torch.Tensor, prefill_chunk: int = 0) -> DynamicCache:
    cache = DynamicCache(config=model.config)
    if prefill_chunk <= 0 or input_ids.shape[-1] <= prefill_chunk:
        _ = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    else:
        for s in range(0, input_ids.shape[-1], prefill_chunk):
            e = min(s + prefill_chunk, input_ids.shape[-1])
            _ = model(input_ids=input_ids[:, s:e], past_key_values=cache, use_cache=True)
    return cache


def roundtrip_cache(model, cache_ref: DynamicCache, *,
                    block_size: int, bit_width: int,
                    rope_mode: str, rope_theta: float,
                    pca_method: str = "randomized",
                    variance_ratio: float = 0.95,
                    rsvd_target_rank_factor: float = 0.5,
                    compress: str = "kv") -> tuple[DynamicCache, dict]:
    """Build an alt cache whose full-attention-layer K,V have been
    round-tripped through the codec. On K, optionally apply RoPE⁻¹
    before encoding and RoPE after decoding."""
    cfg = model.config.get_text_config(decoder=True)
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types is None:
        sw = getattr(cfg, "sliding_window", None) or getattr(cfg, "attention_chunk_size", None)
        layer_types = ["sliding_attention" if sw else "full_attention"] * cfg.num_hidden_layers

    cache_alt = DynamicCache(config=model.config)
    stats = {"per_layer": [], "n_full": 0, "rope_mode": rope_mode}

    for i, layer_kv in enumerate(cache_ref.layers):
        if not hasattr(layer_kv, "keys") or layer_kv.keys is None or layer_kv.keys.numel() == 0:
            continue
        k_ref = layer_kv.keys      # [bsz, n_kv, seq, hd]
        v_ref = layer_kv.values

        if layer_types[i] != "full_attention":
            cache_alt.layers[i].update(k_ref.clone(), v_ref.clone(), 0)
            continue

        bsz, n_kv, seq, hd = k_ref.shape
        # Reshape K: [bsz, n_kv, seq, hd] -> [seq, n_kv, hd] (bsz=1 assumed)
        assert bsz == 1, "multi-batch not yet supported"
        k_post_np = k_ref[0].to(torch.float32).permute(1, 0, 2).cpu().numpy()   # [seq, n_kv, hd]
        v_np = v_ref[0].to(torch.float32).permute(1, 0, 2).cpu().numpy()         # [seq, n_kv, hd]
        positions = np.arange(seq, dtype=np.float64)

        # Step 1: apply RoPE⁻¹ on K if requested
        if rope_mode == "halfsplit":
            inv_freq = rope_inv_freq(hd, base=rope_theta)
            k_pre_np = rope_halfsplit_inverse(k_post_np, positions, inv_freq)
        elif rope_mode == "none":
            k_pre_np = k_post_np
        else:
            raise ValueError(f"unknown rope_mode {rope_mode}")

        # Step 2: flatten to [seq * n_kv, hd] and send through codec
        k_flat = k_pre_np.reshape(-1, hd).astype(np.float32, copy=False)
        v_flat = v_np.reshape(-1, hd).astype(np.float32, copy=False)
        n_total = k_flat.shape[0]
        n_compressible = (n_total // block_size) * block_size
        target_rank = max(2, int(hd * rsvd_target_rank_factor))

        if n_compressible > 0:
            if compress in ("kv", "k_only"):
                k_dec, k_rep = rust_roundtrip(
                    k_flat[:n_compressible], block_size=block_size,
                    bit_width=bit_width, rsvd_target_rank=target_rank,
                    metric="inner_product", share_basis=False,
                    pca_method=pca_method, variance_ratio=variance_ratio,
                )
            else:
                k_dec = k_flat[:n_compressible].copy()
                k_rep = {"mean_block_mse": 0.0, "compressed_bytes": 0, "skipped": True}
            if compress in ("kv", "v_only"):
                v_dec, v_rep = rust_roundtrip(
                    v_flat[:n_compressible], block_size=block_size,
                    bit_width=bit_width, rsvd_target_rank=target_rank,
                    metric="mse", share_basis=True,
                    pca_method=pca_method, variance_ratio=variance_ratio,
                )
            else:
                v_dec = v_flat[:n_compressible].copy()
                v_rep = {"mean_block_mse": 0.0, "compressed_bytes": 0, "skipped": True}
        else:
            k_dec = k_flat[:0]
            v_dec = v_flat[:0]
            k_rep = {"mean_block_mse": 0.0, "compressed_bytes": 0}
            v_rep = {"mean_block_mse": 0.0, "compressed_bytes": 0}

        # Reassemble + tail (tail stays exact, codec only touches aligned prefix)
        k_pre_full = np.concatenate([k_dec, k_flat[n_compressible:]]) if n_compressible < n_total else k_dec
        v_full = np.concatenate([v_dec, v_flat[n_compressible:]]) if n_compressible < n_total else v_dec
        # Reshape to [seq, n_kv, hd]
        k_pre_full = k_pre_full.reshape(seq, n_kv, hd)
        v_full = v_full.reshape(seq, n_kv, hd)

        # Step 3: re-apply RoPE on K if we stripped it earlier
        if rope_mode == "halfsplit":
            inv_freq = rope_inv_freq(hd, base=rope_theta)
            k_post_full = rope_halfsplit_forward(k_pre_full, positions, inv_freq)
        else:
            k_post_full = k_pre_full

        # Step 4: reshape back to [1, n_kv, seq, hd] and store
        k_tensor = torch.from_numpy(np.ascontiguousarray(
            k_post_full.transpose(1, 0, 2))).unsqueeze(0).to(k_ref.dtype).to(k_ref.device)
        v_tensor = torch.from_numpy(np.ascontiguousarray(
            v_full.transpose(1, 0, 2))).unsqueeze(0).to(v_ref.dtype).to(v_ref.device)
        cache_alt.layers[i].update(k_tensor, v_tensor, 0)

        stats["per_layer"].append({
            "layer": i, "hd": hd, "seq": seq,
            "k_mse_pre_rope": float(k_rep["mean_block_mse"]),
            "v_mse": float(v_rep["mean_block_mse"]),
        })
        stats["n_full"] += 1

    return cache_alt, stats


# -----------------------------------------------------------------------------
# WikiText + PPL metrics (shared with e2e_ppl_validation)
# -----------------------------------------------------------------------------

def load_wikitext_passages(tok, min_tokens: int, n_passages: int, split: str = "test"):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    passages = []
    current = []
    current_tok = 0
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        current.append(text)
        current_tok += int(len(text.split()) * 1.3)
        if current_tok >= min_tokens:
            passage = "".join(current)
            if tok(passage, return_tensors="pt")["input_ids"].shape[-1] >= min_tokens:
                passages.append(passage)
                if len(passages) >= n_passages:
                    return passages
            current = []
            current_tok = 0
    return passages


def compare_logits(logits_ref, logits_alt, cont_ids):
    sl_ref = logits_ref[..., :-1, :].float()
    sl_alt = logits_alt[..., :-1, :].float()
    labels = cont_ids[..., 1:]
    log_p_ref = F.log_softmax(sl_ref, dim=-1)
    p_ref = log_p_ref.exp()
    log_p_alt = F.log_softmax(sl_alt, dim=-1)
    kl = (p_ref * (log_p_ref - log_p_alt)).sum(dim=-1)
    top1_ref = sl_ref.argmax(dim=-1)
    top1_alt = sl_alt.argmax(dim=-1)
    nll_ref = F.cross_entropy(sl_ref.reshape(-1, sl_ref.size(-1)), labels.reshape(-1))
    nll_alt = F.cross_entropy(sl_alt.reshape(-1, sl_alt.size(-1)), labels.reshape(-1))
    return {
        "mean_kl": float(kl.mean().item()),
        "top1_agreement": float((top1_ref == top1_alt).float().mean().item()),
        "ppl_ref": float(torch.exp(nll_ref).item()),
        "ppl_alt": float(torch.exp(nll_alt).item()),
        "ppl_delta_rel": float((torch.exp(nll_alt) - torch.exp(nll_ref)) / torch.exp(nll_ref)),
    }


@torch.inference_mode()
def logits_with_cache(model, cache, cont_ids):
    out = model(input_ids=cont_ids, past_key_values=cache, use_cache=True)
    return out.logits


def evaluate(model, tok, passage, ctx_len, n_eval, block_size, bit_width,
             prefill_chunk, rope_mode, rope_theta, pca_method, vr, compress):
    ids = tok(passage, return_tensors="pt")["input_ids"]
    if ids.shape[-1] < ctx_len + n_eval:
        return None
    prefix = ids[:, :ctx_len]
    cont = ids[:, ctx_len:ctx_len + n_eval]

    cache_ref = prefill_cache(model, prefix, prefill_chunk)
    cache_alt, stats = roundtrip_cache(
        model, cache_ref,
        block_size=block_size, bit_width=bit_width,
        rope_mode=rope_mode, rope_theta=rope_theta,
        pca_method=pca_method, variance_ratio=vr,
        compress=compress,
    )

    cache_ref_fwd = copy.deepcopy(cache_ref)
    cache_alt_fwd = copy.deepcopy(cache_alt)
    logits_ref = logits_with_cache(model, cache_ref_fwd, cont)
    logits_alt = logits_with_cache(model, cache_alt_fwd, cont)
    return {
        "metrics": compare_logits(logits_ref, logits_alt, cont),
        "stats": stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--ctx-len", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--bit-width", type=int, default=2)
    ap.add_argument("--prefill-chunk", type=int, default=0)
    ap.add_argument("--n-passages", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pca-method", choices=["exact", "randomized"], default="randomized")
    ap.add_argument("--variance-ratio", type=float, default=0.95)
    ap.add_argument("--rope-mode", choices=["none", "halfsplit"], default="halfsplit",
                    help="Apply inverse RoPE before codec (halfsplit for Qwen/DeepSeek/Llama)")
    ap.add_argument("--compress", choices=["kv", "k_only", "v_only"], default="kv")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    test_rope_roundtrip()
    print("  [sanity] RoPE forward/inverse round-trip OK")

    print(f"[{args.model_name}] loading model…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.eval()
    cfg = model.config.get_text_config(decoder=True)
    rope_theta = float(getattr(cfg, "rope_theta", None)
                       or (getattr(cfg, "rope_scaling", None) or {}).get("rope_theta", 10000.0))
    print(f"  rope_theta = {rope_theta}, rope_mode = {args.rope_mode}")

    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval, n_passages=args.n_passages
    )
    print(f"  got {len(passages)} WikiText passages")

    per_passage = []
    for i, p in enumerate(passages):
        print(f"  passage {i+1}/{len(passages)}…", flush=True)
        res = evaluate(
            model, tok, p, args.ctx_len, args.n_eval, args.block_size,
            args.bit_width, args.prefill_chunk,
            args.rope_mode, rope_theta, args.pca_method, args.variance_ratio,
            args.compress,
        )
        if res is None:
            print("    skipped (too short)")
            continue
        per_passage.append(res)
        m = res["metrics"]
        print(f"    ppl_ref={m['ppl_ref']:.3f} ppl_alt={m['ppl_alt']:.3f} "
              f"Δppl={m['ppl_delta_rel']*100:+.2f}% "
              f"KL={m['mean_kl']:.4f} top1={m['top1_agreement']*100:.1f}%", flush=True)

    if per_passage:
        mean_delta = float(np.mean([r["metrics"]["ppl_delta_rel"] for r in per_passage]))
        mean_kl = float(np.mean([r["metrics"]["mean_kl"] for r in per_passage]))
        mean_top1 = float(np.mean([r["metrics"]["top1_agreement"] for r in per_passage]))
        verdict = "ACCEPT" if abs(mean_delta) <= 0.01 and mean_top1 >= 0.95 else \
                  "MARGINAL" if abs(mean_delta) <= 0.03 and mean_top1 >= 0.85 else \
                  "REJECT"
        print(f"\n[{args.model_name}] rope_mode={args.rope_mode}  VERDICT={verdict}")
        print(f"  Δppl mean = {mean_delta*100:+.3f}%")
        print(f"  top1 mean = {mean_top1*100:.2f}%")
        print(f"  KL mean   = {mean_kl:.5f}")

    (args.out_dir / f"{args.model_name}_rope_{args.rope_mode}.json").write_text(json.dumps({
        "model_name": args.model_name,
        "ctx_len": args.ctx_len,
        "n_eval": args.n_eval,
        "bit_width": args.bit_width,
        "pca_method": args.pca_method,
        "variance_ratio": args.variance_ratio,
        "rope_mode": args.rope_mode,
        "rope_theta": rope_theta,
        "per_passage": per_passage,
    }, indent=2))


if __name__ == "__main__":
    main()
