#!/usr/bin/env python3
"""End-to-end PPL validation of v1.3 codec on a TRUE pre-RoPE cache.

Difference from e2e_ppl_rope_aware.py:
  - no inverse-RoPE / re-apply-RoPE wrapper around the codec
  - the attention forward itself is patched so the cache holds pre-RoPE K
  - codec operates on cache.layers[i].keys directly; it never sees RoPE phase
  - RoPE is still applied, but inline inside attention at read time

This matches vLLM / SGLang / TRT-LLM's paged-attention architecture.
"""
from __future__ import annotations

import argparse
import copy
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import benchmarks.pre_rope_cache as prc

BENCH_BIN = REPO / "kakeyaturbo" / "target" / "release" / "kakeyaturbo-bench"
KKTV_MAGIC = 0x4B4B5456


def write_kktv(path: Path, arr: np.ndarray) -> None:
    assert arr.dtype == np.float32 and arr.ndim == 2
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
            str(BENCH_BIN), "--input", str(in_p), "--output", str(rep_p),
            "--metric", metric, "--block-size", str(block_size),
            "--variance-ratio", str(variance_ratio),
            "--k", "16", "--bit-width", str(bit_width),
            "--rotation-seed", "3405691582",
            "--pca-method", pca_method, "--verify",
            "--dump-decoded", str(dec_p),
        ]
        if pca_method == "randomized":
            cmd += ["--rsvd-target-rank", str(rsvd_target_rank),
                    "--rsvd-oversample", "8", "--rsvd-power-iters", "2"]
        if share_basis:
            cmd.append("--share-basis")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        return read_kktv(dec_p), json.loads(rep_p.read_text())


@torch.inference_mode()
def prefill_cache(model, input_ids, prefill_chunk=0) -> DynamicCache:
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
                    pca_method: str = "randomized",
                    variance_ratio: float = 0.95,
                    rsvd_target_rank_factor: float = 0.5,
                    compress: str = "kv") -> tuple[DynamicCache, dict]:
    """Build an alt cache whose full-attention-layer K,V are round-tripped
    through the codec. K is PRE-RoPE (cache already holds it that way)."""
    cfg = model.config.get_text_config(decoder=True)
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types is None:
        sw = getattr(cfg, "sliding_window", None) or getattr(cfg, "attention_chunk_size", None)
        layer_types = ["sliding_attention" if sw else "full_attention"] * cfg.num_hidden_layers

    cache_alt = DynamicCache(config=model.config)
    stats = {"per_layer": [], "n_full": 0}

    for i, layer_kv in enumerate(cache_ref.layers):
        if not hasattr(layer_kv, "keys") or layer_kv.keys is None or layer_kv.keys.numel() == 0:
            continue
        k_ref = layer_kv.keys
        v_ref = layer_kv.values

        if layer_types[i] != "full_attention":
            cache_alt.layers[i].update(k_ref.clone(), v_ref.clone(), 0)
            continue

        bsz, n_kv, seq, hd = k_ref.shape
        assert bsz == 1
        k_np = k_ref[0].to(torch.float32).permute(1, 0, 2).cpu().numpy()
        v_np = v_ref[0].to(torch.float32).permute(1, 0, 2).cpu().numpy()
        k_flat = k_np.reshape(-1, hd).astype(np.float32, copy=False)
        v_flat = v_np.reshape(-1, hd).astype(np.float32, copy=False)
        n_total = k_flat.shape[0]
        n_comp = (n_total // block_size) * block_size
        target_rank = max(2, int(hd * rsvd_target_rank_factor))

        if n_comp > 0:
            if compress in ("kv", "k_only"):
                k_dec, k_rep = rust_roundtrip(
                    k_flat[:n_comp], block_size=block_size, bit_width=bit_width,
                    rsvd_target_rank=target_rank, metric="inner_product",
                    share_basis=False, pca_method=pca_method,
                    variance_ratio=variance_ratio,
                )
            else:
                k_dec = k_flat[:n_comp].copy()
                k_rep = {"mean_block_mse": 0.0, "skipped": True}
            if compress in ("kv", "v_only"):
                v_dec, v_rep = rust_roundtrip(
                    v_flat[:n_comp], block_size=block_size, bit_width=bit_width,
                    rsvd_target_rank=target_rank, metric="mse",
                    share_basis=True, pca_method=pca_method,
                    variance_ratio=variance_ratio,
                )
            else:
                v_dec = v_flat[:n_comp].copy()
                v_rep = {"mean_block_mse": 0.0, "skipped": True}
        else:
            k_dec = k_flat[:0]; v_dec = v_flat[:0]
            k_rep = v_rep = {"mean_block_mse": 0.0}

        k_full = np.concatenate([k_dec, k_flat[n_comp:]]) if n_comp < n_total else k_dec
        v_full = np.concatenate([v_dec, v_flat[n_comp:]]) if n_comp < n_total else v_dec
        k_full = k_full.reshape(seq, n_kv, hd)
        v_full = v_full.reshape(seq, n_kv, hd)

        k_tensor = torch.from_numpy(np.ascontiguousarray(k_full.transpose(1, 0, 2))).unsqueeze(0).to(k_ref.dtype).to(k_ref.device)
        v_tensor = torch.from_numpy(np.ascontiguousarray(v_full.transpose(1, 0, 2))).unsqueeze(0).to(v_ref.dtype).to(v_ref.device)
        cache_alt.layers[i].update(k_tensor, v_tensor, 0)

        stats["per_layer"].append({
            "layer": i, "hd": hd, "seq": seq,
            "k_mse": float(k_rep.get("mean_block_mse", 0.0)),
            "v_mse": float(v_rep.get("mean_block_mse", 0.0)),
        })
        stats["n_full"] += 1

    return cache_alt, stats


def load_wikitext_passages(tok, min_tokens, n_passages, split="test"):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    passages, current, current_tok = [], [], 0
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
            current, current_tok = [], 0
    return passages


def compare_logits(lo_ref, lo_alt, cont_ids):
    s_ref = lo_ref[..., :-1, :].float()
    s_alt = lo_alt[..., :-1, :].float()
    labels = cont_ids[..., 1:]
    log_p_ref = F.log_softmax(s_ref, dim=-1)
    p_ref = log_p_ref.exp()
    log_p_alt = F.log_softmax(s_alt, dim=-1)
    kl = (p_ref * (log_p_ref - log_p_alt)).sum(dim=-1)
    top1_ref = s_ref.argmax(-1)
    top1_alt = s_alt.argmax(-1)
    nll_ref = F.cross_entropy(s_ref.reshape(-1, s_ref.size(-1)), labels.reshape(-1))
    nll_alt = F.cross_entropy(s_alt.reshape(-1, s_alt.size(-1)), labels.reshape(-1))
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
             prefill_chunk, pca_method, vr, compress):
    ids = tok(passage, return_tensors="pt")["input_ids"]
    if ids.shape[-1] < ctx_len + n_eval:
        return None
    prefix = ids[:, :ctx_len]
    cont = ids[:, ctx_len:ctx_len + n_eval]

    cache_ref = prefill_cache(model, prefix, prefill_chunk)
    cache_alt, stats = roundtrip_cache(
        model, cache_ref, block_size=block_size, bit_width=bit_width,
        pca_method=pca_method, variance_ratio=vr, compress=compress,
    )
    cache_ref_fwd = copy.deepcopy(cache_ref)
    cache_alt_fwd = copy.deepcopy(cache_alt)
    logits_ref = logits_with_cache(model, cache_ref_fwd, cont)
    logits_alt = logits_with_cache(model, cache_alt_fwd, cont)
    return {"metrics": compare_logits(logits_ref, logits_alt, cont), "stats": stats}


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
    ap.add_argument("--compress", choices=["kv", "k_only", "v_only"], default="kv")
    ap.add_argument("--skip-sanity", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.model_name}] loading model…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()

    info = prc.install(model)
    print(f"  patched {info['patched_layers']} attention layers; cache now stores PRE-RoPE K")

    if not args.skip_sanity:
        ids_s = tok("Hello world. " * 20, return_tensors="pt")["input_ids"][:, :64]
        cache_s = DynamicCache(config=model.config)
        with torch.inference_mode():
            lo = model(input_ids=ids_s, past_key_values=cache_s, use_cache=True).logits
        k0 = cache_s.layers[0].keys
        print(f"  sanity: logits finite = {torch.isfinite(lo).all().item()},  K_pre[0] norm = {k0.norm().item():.3f}")

    passages = load_wikitext_passages(tok, args.ctx_len + args.n_eval, args.n_passages)
    print(f"  got {len(passages)} WikiText passages")

    per_passage = []
    for i, p in enumerate(passages):
        print(f"  passage {i+1}/{len(passages)}…", flush=True)
        res = evaluate(
            model, tok, p, args.ctx_len, args.n_eval, args.block_size,
            args.bit_width, args.prefill_chunk, args.pca_method,
            args.variance_ratio, args.compress,
        )
        if res is None:
            print("    skipped (too short)"); continue
        per_passage.append(res)
        m = res["metrics"]
        print(f"    ppl_ref={m['ppl_ref']:.3f} ppl_alt={m['ppl_alt']:.3f} "
              f"Δppl={m['ppl_delta_rel']*100:+.2f}% "
              f"KL={m['mean_kl']:.4f} top1={m['top1_agreement']*100:.1f}%", flush=True)

    if per_passage:
        md = float(np.mean([r["metrics"]["ppl_delta_rel"] for r in per_passage]))
        mk = float(np.mean([r["metrics"]["mean_kl"] for r in per_passage]))
        mt = float(np.mean([r["metrics"]["top1_agreement"] for r in per_passage]))
        verdict = "ACCEPT" if abs(md) <= 0.01 and mt >= 0.95 else \
                  "MARGINAL" if abs(md) <= 0.03 and mt >= 0.85 else "REJECT"
        print(f"\n[{args.model_name}] PRE-ROPE CACHE  VERDICT = {verdict}")
        print(f"  compress={args.compress} bit_width={args.bit_width} vr={args.variance_ratio} pca={args.pca_method}")
        print(f"  Δppl mean = {md*100:+.3f}%")
        print(f"  top1 mean = {mt*100:.2f}%")
        print(f"  KL mean   = {mk:.5f}")

    (args.out_dir / f"{args.model_name}_prerope_{args.compress}_b{args.bit_width}.json").write_text(json.dumps({
        "model_name": args.model_name, "ctx_len": args.ctx_len,
        "n_eval": args.n_eval, "bit_width": args.bit_width,
        "pca_method": args.pca_method, "variance_ratio": args.variance_ratio,
        "compress": args.compress, "architecture": "pre_rope_cache",
        "per_passage": per_passage,
    }, indent=2))


if __name__ == "__main__":
    main()
