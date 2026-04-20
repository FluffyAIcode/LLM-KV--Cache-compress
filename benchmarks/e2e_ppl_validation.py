#!/usr/bin/env python3
"""End-to-end downstream-quality validation of the v1.3 codec on WikiText-103.

Experimental design
-------------------
For each WikiText-103 passage of length >= ctx_len + n_eval:
  1. Prefill ctx_len tokens into a reference DynamicCache (bf16).
  2. Clone the reference cache and round-trip every full-attention layer
     through the v1.3 Rust codec (encode -> decode, real reconstructed KV).
  3. On both caches, compute next-token logits for the n_eval evaluation
     tokens (teacher-forced).
  4. Compare logit distributions: mean/max KL, top-1 agreement, PPL ratio.

Both caches start identical; only step 2 perturbs the alt cache. The
only source of divergence in step 3 is therefore the v1.3 KV
reconstruction error. This is the clean end-to-end signal we want.

The Rust bench binary MUST support --dump-decoded (added in commit
introducing this harness).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
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
# WikiText-103 loader + concatenator
# -----------------------------------------------------------------------------

def load_wikitext_passages(tokenizer, min_tokens: int, n_passages: int,
                           split: str = "test") -> list[str]:
    """Concatenate consecutive non-empty WikiText rows into passages of
    length >= min_tokens, return the first n_passages."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    passages = []
    current = []
    current_tok_count = 0
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        current.append(text)
        # Approximate token count via word count * 1.3 to avoid re-tokenising
        current_tok_count += int(len(text.split()) * 1.3)
        if current_tok_count >= min_tokens:
            passage = "".join(current)
            real_len = tokenizer(passage, return_tensors="pt")["input_ids"].shape[-1]
            if real_len >= min_tokens:
                passages.append(passage)
                if len(passages) >= n_passages:
                    return passages
            current = []
            current_tok_count = 0
    return passages


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


def read_kktv_f32(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == KKTV_MAGIC, f"{path}: bad magic"
        _version = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        d = struct.unpack("<I", f.read(4))[0]
        _pad = struct.unpack("<I", f.read(4))[0]
        raw = f.read(n * d * 4)
    return np.frombuffer(raw, dtype=np.float32).reshape(n, d).copy()


# -----------------------------------------------------------------------------
# v1.3 Rust codec round-trip
# -----------------------------------------------------------------------------

def rust_roundtrip(
    arr: np.ndarray, block_size: int, bit_width: int,
    rsvd_target_rank: int, metric: str, share_basis: bool,
    pca_method: str = "randomized",
    variance_ratio: float = 0.95, k_means_k: int = 16,
    rsvd_oversample: int = 8, rsvd_power_iters: int = 2,
) -> tuple[np.ndarray, dict]:
    """Full v1.3 round-trip. Always applies spherical k-means + WHT + Lloyd-Max."""
    """Encode `arr` (N,D) through v1.3 codec and return decoded + report."""
    import tempfile
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        tdp = Path(td)
        in_path = tdp / "x.kktv"
        rep_path = tdp / "report.json"
        dec_path = tdp / "decoded.kktv"
        write_kktv(in_path, arr.astype(np.float32, copy=False))
        cmd = [
            str(BENCH_BIN),
            "--input", str(in_path),
            "--output", str(rep_path),
            "--metric", metric,
            "--block-size", str(block_size),
            "--variance-ratio", str(variance_ratio),
            "--k", str(k_means_k),
            "--bit-width", str(bit_width),
            "--rotation-seed", "3405691582",
            "--pca-method", pca_method,
            "--verify",
            "--dump-decoded", str(dec_path),
        ]
        if pca_method == "randomized":
            cmd += [
                "--rsvd-target-rank", str(rsvd_target_rank),
                "--rsvd-oversample", str(rsvd_oversample),
                "--rsvd-power-iters", str(rsvd_power_iters),
            ]
        if share_basis:
            cmd.append("--share-basis")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"rust bench failed: {res.stderr}")
        report = json.loads(rep_path.read_text())
        decoded = read_kktv_f32(dec_path)
        return decoded, report


# -----------------------------------------------------------------------------
# Cache manipulation
# -----------------------------------------------------------------------------

@torch.inference_mode()
def prefill_cache(model, input_ids: torch.Tensor,
                  prefill_chunk: int = 0) -> DynamicCache:
    cache = DynamicCache(config=model.config)
    if prefill_chunk <= 0 or input_ids.shape[-1] <= prefill_chunk:
        _ = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    else:
        for s in range(0, input_ids.shape[-1], prefill_chunk):
            e = min(s + prefill_chunk, input_ids.shape[-1])
            _ = model(input_ids=input_ids[:, s:e], past_key_values=cache, use_cache=True)
    return cache


def roundtrip_cache(
    model, cache_ref: DynamicCache, block_size: int, bit_width: int,
    rsvd_target_rank_factor: float = 0.5, pca_method: str = "randomized",
    variance_ratio: float = 0.95,
) -> tuple[DynamicCache, dict]:
    """Build an alt cache whose full-attention-layer K,V have been
    round-tripped through the v1.3 codec. Sliding-window layers are
    copied unchanged."""
    cfg = model.config.get_text_config(decoder=True)
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types is None:
        sw = getattr(cfg, "sliding_window", None) or getattr(cfg, "attention_chunk_size", None)
        layer_types = ["sliding_attention" if sw else "full_attention"] * cfg.num_hidden_layers

    cache_alt = DynamicCache(config=model.config)
    stats = {"per_layer": [], "n_full": 0}

    for i, layer_kv in enumerate(cache_ref.layers):
        if not hasattr(layer_kv, "keys") or layer_kv.keys is None \
                or layer_kv.keys.numel() == 0:
            continue
        k_ref = layer_kv.keys    # [bsz, n_kv, seq, hd]
        v_ref = layer_kv.values

        if layer_types[i] != "full_attention":
            cache_alt.layers[i].update(k_ref.clone(), v_ref.clone(), 0)
            continue

        bsz, n_kv, seq, hd = k_ref.shape
        k_flat = k_ref.to(torch.float32).cpu().numpy().reshape(-1, hd)
        v_flat = v_ref.to(torch.float32).cpu().numpy().reshape(-1, hd)

        # Only compressible-aligned prefix goes through the codec; tail
        # stays exact.
        n_total = k_flat.shape[0]
        n_full_blocks = n_total // block_size
        n_compressible = n_full_blocks * block_size

        target_rank = max(2, int(hd * rsvd_target_rank_factor))

        if n_compressible > 0:
            k_dec, k_rep = rust_roundtrip(
                k_flat[:n_compressible], block_size=block_size,
                bit_width=bit_width, rsvd_target_rank=target_rank,
                metric="inner_product", share_basis=False,
                pca_method=pca_method, variance_ratio=variance_ratio,
            )
            v_dec, v_rep = rust_roundtrip(
                v_flat[:n_compressible], block_size=block_size,
                bit_width=bit_width, rsvd_target_rank=target_rank,
                metric="mse", share_basis=True,
                pca_method=pca_method, variance_ratio=variance_ratio,
            )
        else:
            k_dec, v_dec = k_flat[:0], v_flat[:0]
            k_rep = v_rep = {"mean_block_mse": 0.0, "compressed_bytes": 0}

        k_full_decoded = np.concatenate(
            [k_dec, k_flat[n_compressible:]], axis=0) if n_compressible < n_total else k_dec
        v_full_decoded = np.concatenate(
            [v_dec, v_flat[n_compressible:]], axis=0) if n_compressible < n_total else v_dec

        k_restore = torch.from_numpy(k_full_decoded.copy()) \
            .reshape(bsz, n_kv, seq, hd).to(k_ref.dtype).to(k_ref.device)
        v_restore = torch.from_numpy(v_full_decoded.copy()) \
            .reshape(bsz, n_kv, seq, hd).to(v_ref.dtype).to(v_ref.device)

        cache_alt.layers[i].update(k_restore, v_restore, 0)

        stats["per_layer"].append({
            "layer": i, "hd": hd, "seq": seq,
            "k_mse": float(k_rep["mean_block_mse"]),
            "v_mse": float(v_rep["mean_block_mse"]),
            "k_bytes": int(k_rep["compressed_bytes"]),
            "v_bytes": int(v_rep["compressed_bytes"]),
            "n_compressible_vecs": int(n_compressible),
            "n_tail_vecs": int(n_total - n_compressible),
        })
        stats["n_full"] += 1

    return cache_alt, stats


@torch.inference_mode()
def logits_with_prefilled_cache(model, cache: DynamicCache,
                                cont_ids: torch.Tensor) -> torch.Tensor:
    """Teacher-force cont_ids through model using pre-filled cache,
    return logits over cont_ids positions."""
    out = model(input_ids=cont_ids, past_key_values=cache, use_cache=True)
    return out.logits  # [1, len(cont_ids), V]


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compare_logits(
    logits_ref: torch.Tensor, logits_alt: torch.Tensor,
    cont_ids: torch.Tensor
) -> dict:
    """Compare next-token distributions position-by-position."""
    assert logits_ref.shape == logits_alt.shape
    # Position t predicts cont_ids[t+1] (teacher-forced), so shift by 1
    sl_ref = logits_ref[..., :-1, :].float()
    sl_alt = logits_alt[..., :-1, :].float()
    labels = cont_ids[..., 1:]

    log_p_ref = F.log_softmax(sl_ref, dim=-1)
    p_ref = log_p_ref.exp()
    log_p_alt = F.log_softmax(sl_alt, dim=-1)
    kl_per_tok = (p_ref * (log_p_ref - log_p_alt)).sum(dim=-1)  # [1, T-1]
    mean_kl = float(kl_per_tok.mean().item())
    max_kl = float(kl_per_tok.max().item())

    top1_ref = sl_ref.argmax(dim=-1)
    top1_alt = sl_alt.argmax(dim=-1)
    agree = float((top1_ref == top1_alt).float().mean().item())

    nll_ref = F.cross_entropy(
        sl_ref.reshape(-1, sl_ref.size(-1)), labels.reshape(-1), reduction="mean")
    nll_alt = F.cross_entropy(
        sl_alt.reshape(-1, sl_alt.size(-1)), labels.reshape(-1), reduction="mean")
    ppl_ref = float(torch.exp(nll_ref).item())
    ppl_alt = float(torch.exp(nll_alt).item())

    return {
        "mean_kl": mean_kl,
        "max_kl": max_kl,
        "top1_agreement": agree,
        "ppl_ref": ppl_ref,
        "ppl_alt": ppl_alt,
        "ppl_delta_rel": (ppl_alt - ppl_ref) / max(ppl_ref, 1e-8),
        "nll_ref": float(nll_ref.item()),
        "nll_alt": float(nll_alt.item()),
        "n_tokens": int(labels.numel()),
    }


# -----------------------------------------------------------------------------
# Per-passage evaluation
# -----------------------------------------------------------------------------

def evaluate_passage(
    model, tokenizer, passage: str, ctx_len: int, n_eval: int,
    block_size: int, bit_width: int, prefill_chunk: int,
    pca_method: str = "randomized", variance_ratio: float = 0.95,
) -> dict | None:
    ids = tokenizer(passage, return_tensors="pt")["input_ids"]
    if ids.shape[-1] < ctx_len + n_eval:
        return None
    prefix_ids = ids[:, :ctx_len]
    cont_ids = ids[:, ctx_len : ctx_len + n_eval]

    t0 = time.perf_counter()
    cache_ref = prefill_cache(model, prefix_ids, prefill_chunk)
    t_prefill = time.perf_counter() - t0

    t0 = time.perf_counter()
    cache_alt, stats = roundtrip_cache(model, cache_ref, block_size, bit_width,
                                       pca_method=pca_method,
                                       variance_ratio=variance_ratio)
    t_roundtrip = time.perf_counter() - t0

    # We need two SEPARATE cache instances so the teacher-forced forward
    # pass mutates each independently. cache_ref has already been used --
    # but DynamicCache is append-only, so re-forwarding cont_ids just
    # appends; that's fine for our comparison. To be safe, deep-copy both
    # before the forward.
    cache_ref_fwd = copy.deepcopy(cache_ref)
    cache_alt_fwd = copy.deepcopy(cache_alt)

    logits_ref = logits_with_prefilled_cache(model, cache_ref_fwd, cont_ids)
    logits_alt = logits_with_prefilled_cache(model, cache_alt_fwd, cont_ids)

    metrics = compare_logits(logits_ref, logits_alt, cont_ids)

    return {
        "ctx_len": ctx_len,
        "n_eval": n_eval,
        "prefill_sec": t_prefill,
        "roundtrip_sec": t_roundtrip,
        "compression_stats": stats,
        "metrics": metrics,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def verdict_of(mean_delta_rel: float, mean_top1: float) -> str:
    """ACCEPT: |delta ppl| <= 1% AND top1 agreement >= 95%
       MARGINAL: |delta ppl| <= 3% AND top1 agreement >= 85%
       REJECT: otherwise
    Standard LLM-compression PPL thresholds.
    """
    if abs(mean_delta_rel) <= 0.01 and mean_top1 >= 0.95:
        return "ACCEPT"
    if abs(mean_delta_rel) <= 0.03 and mean_top1 >= 0.85:
        return "MARGINAL"
    return "REJECT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=64,
                    help="Number of teacher-forced evaluation tokens")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--bit-width", type=int, default=2)
    ap.add_argument("--prefill-chunk", type=int, default=0)
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pca-method", choices=["exact", "randomized"], default="randomized")
    ap.add_argument("--variance-ratio", type=float, default=0.95)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.model_name}] loading model…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.eval()

    print(f"[{args.model_name}] loading WikiText-103 passages "
          f"(min_tokens={args.ctx_len + args.n_eval})…", flush=True)
    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval,
        n_passages=args.n_passages,
    )
    print(f"  got {len(passages)} passages")

    per_passage = []
    for i, passage in enumerate(passages):
        print(f"  passage {i + 1}/{len(passages)} "
              f"(ctx={args.ctx_len}, n_eval={args.n_eval})…", flush=True)
        res = evaluate_passage(
            model, tok, passage, args.ctx_len, args.n_eval,
            args.block_size, args.bit_width, args.prefill_chunk,
            pca_method=args.pca_method, variance_ratio=args.variance_ratio,
        )
        if res is None:
            print("    skipped (too short after tokenisation)")
            continue
        per_passage.append(res)
        m = res["metrics"]
        print(
            f"    ppl_ref={m['ppl_ref']:.3f} ppl_alt={m['ppl_alt']:.3f} "
            f"Δppl={m['ppl_delta_rel']*100:+.2f}% "
            f"KL={m['mean_kl']:.4f} top1={m['top1_agreement']*100:.1f}%",
            flush=True,
        )

    # Aggregate
    summary = {
        "model_name": args.model_name,
        "ctx_len": args.ctx_len,
        "n_eval": args.n_eval,
        "block_size": args.block_size,
        "bit_width": args.bit_width,
        "pca_method": args.pca_method,
        "variance_ratio": args.variance_ratio,
        "n_passages": len(per_passage),
    }
    if per_passage:
        mean_delta = float(np.mean([r["metrics"]["ppl_delta_rel"] for r in per_passage]))
        mean_kl = float(np.mean([r["metrics"]["mean_kl"] for r in per_passage]))
        mean_top1 = float(np.mean([r["metrics"]["top1_agreement"] for r in per_passage]))
        summary.update({
            "mean_ppl_delta_rel": mean_delta,
            "mean_kl": mean_kl,
            "mean_top1_agreement": mean_top1,
            "verdict": verdict_of(mean_delta, mean_top1),
        })
        print(f"\n[{args.model_name}] ===== SUMMARY =====")
        print(f"  n_passages   = {len(per_passage)}")
        print(f"  Δppl (mean)  = {mean_delta*100:+.3f}%")
        print(f"  KL (mean)    = {mean_kl:.5f}")
        print(f"  top1 agree   = {mean_top1*100:.2f}%")
        print(f"  VERDICT      = {summary['verdict']}")
    else:
        summary["verdict"] = "NO_DATA"
        print(f"\n[{args.model_name}] no usable passages")

    summary["per_passage"] = per_passage
    (args.out_dir / f"{args.model_name}.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
