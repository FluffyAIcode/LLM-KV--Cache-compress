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
from benchmarks.q_precondition import QPrecond, load as load_q_precond

def turboquant_k_roundtrip(*a, **kw):
    from benchmarks.turboquant_roundtrip import (
        turboquant_k_roundtrip as _impl,
    )
    return _impl(*a, **kw)

def turboquant_v_roundtrip(*a, **kw):
    from benchmarks.turboquant_roundtrip import (
        turboquant_v_roundtrip as _impl,
    )
    return _impl(*a, **kw)

BENCH_BIN = REPO / "kakeyaturbo" / "target" / "release" / "kakeyaturbo-bench"
BESI_BIN = REPO / "kakeyaturbo" / "target" / "release" / "besicovitch-bench"
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
                   variance_ratio: float = 0.95,
                   skeleton_dtype: str = "fp16",
                   exact_rank_cap: int | None = None,
                   centroids_file: str | None = None,
                   outlier_threshold: float | None = None,
                   residual_besi: dict | None = None):
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
            "--pca-method", pca_method,
            "--skeleton-dtype", skeleton_dtype,
            "--verify",
            "--dump-decoded", str(dec_p),
        ]
        if pca_method == "randomized":
            cmd += ["--rsvd-target-rank", str(rsvd_target_rank),
                    "--rsvd-oversample", "8", "--rsvd-power-iters", "2"]
        if exact_rank_cap is not None and pca_method == "exact":
            cmd += ["--exact-rank-cap", str(exact_rank_cap)]
        if centroids_file is not None:
            cmd += ["--centroids-file", str(centroids_file)]
        if outlier_threshold is not None:
            cmd += ["--outlier-threshold", str(outlier_threshold)]
        if residual_besi is not None:
            cmd += ["--residual-besi-direction-bits", str(residual_besi["direction_bits"]),
                    "--residual-besi-group-size", str(residual_besi.get("group_size", 2)),
                    "--residual-besi-magnitude-bits", str(residual_besi["magnitude_bits"]),
                    "--residual-besi-magnitude-mode", residual_besi.get("magnitude_mode", "quantized")]
        if share_basis:
            cmd.append("--share-basis")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        return read_kktv(dec_p), json.loads(rep_p.read_text())


def besicovitch_roundtrip(arr: np.ndarray, block_size: int,
                          group_size: int, direction_bits: int,
                          magnitude_bits: int, magnitude_mode: str,
                          subtract_mean: bool):
    """Round-trip a tensor through the Besicovitch-product codec.
    Returns (decoded_array, report_dict)."""
    import tempfile
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        tdp = Path(td)
        in_p = tdp / "x.kktv"
        rep_p = tdp / "r.json"
        dec_p = tdp / "dec.kktv"
        write_kktv(in_p, arr.astype(np.float32, copy=False))
        cmd = [
            str(BESI_BIN), "--input", str(in_p), "--output", str(rep_p),
            "--block-size", str(block_size),
            "--group-size", str(group_size),
            "--direction-bits", str(direction_bits),
            "--magnitude-bits", str(magnitude_bits),
            "--magnitude-mode", magnitude_mode,
            "--verify",
            "--dump-decoded", str(dec_p),
        ]
        if subtract_mean:
            cmd.append("--subtract-mean")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"besicovitch-bench failed: {r.stderr}")
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
                    compress: str = "kv",
                    skeleton_dtype: str = "fp16",
                    share_basis_k: bool = False,
                    share_basis_v: bool = True,
                    q_precond: QPrecond | None = None,
                    exact_rank_cap: int | None = None,
                    codec: str = "kakeyaturbo",
                    bit_width_v: int | None = None,
                    exact_rank_cap_v: int | None = None,
                    pca_method_v: str | None = None,
                    k_centroids_file: str | None = None,
                    v_centroids_file: str | None = None,
                    boundary_skip_layers: list[int] | None = None,
                    boundary_mode: str = "bf16",
                    boundary_bit_width: int = 4,
                    k_outlier_threshold: float | None = None,
                    v_outlier_threshold: float | None = None,
                    besi_group_size: int = 2,
                    besi_direction_bits: int = 5,
                    besi_magnitude_bits: int = 4,
                    besi_magnitude_mode: str = "quantized",
                    besi_subtract_mean: bool = True,
                    codec_v: str | None = None,
                    k_residual_besi: dict | None = None,
                    riemann_scale_method: str = "sqrt_trace",
                    riemann_centroids_file: str | None = None) -> tuple[DynamicCache, dict]:
    """Per-stream bit_width: `bit_width` applies to K (and V if bit_width_v
    is None); `bit_width_v` (if given) overrides V-stream bit width for
    asymmetric K/V codec operation.  Likewise `exact_rank_cap_v` is the
    V-side override for the PCA rank cap (K keeps `exact_rank_cap`), and
    `pca_method_v` overrides the V-side PCA method ('exact' or 'randomized')."""
    """Build an alt cache whose full-attention-layer K,V are round-tripped
    through the codec. K is PRE-RoPE (cache already holds it that way)."""
    cfg = model.config.get_text_config(decoder=True)
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types is None:
        sw = getattr(cfg, "sliding_window", None) or getattr(cfg, "attention_chunk_size", None)
        layer_types = ["sliding_attention" if sw else "full_attention"] * cfg.num_hidden_layers

    cache_alt = DynamicCache(config=model.config)
    stats = {"per_layer": [], "n_full": 0}

    # Resolve boundary_skip_layers: these layers are kept at bf16 (fully
    # uncompressed) regardless of compression config.  Mirrors
    # TurboQuant+'s documented production trick: first 2 + last 2 layers
    # at q8_0 to recover 37-91% of the K-compression quality gap.
    boundary_skip_set = set(boundary_skip_layers or [])

    for i, layer_kv in enumerate(cache_ref.layers):
        if not hasattr(layer_kv, "keys") or layer_kv.keys is None or layer_kv.keys.numel() == 0:
            continue
        k_ref = layer_kv.keys
        v_ref = layer_kv.values

        if layer_types[i] != "full_attention":
            cache_alt.layers[i].update(k_ref.clone(), v_ref.clone(), 0)
            continue

        if i in boundary_skip_set:
            # Boundary layer protection. The user's choice:
            #   boundary_mode="bf16": keep full precision, no codec applied
            #   boundary_mode="conservative": apply codec at conservative bit_width
            #     (e.g. b_K=4, b_V=4) instead of the user's aggressive b=2 recipe.
            if boundary_mode == "bf16":
                cache_alt.layers[i].update(k_ref.clone(), v_ref.clone(), 0)
                stats["per_layer"].append({
                    "layer": i, "boundary_uncompressed": True,
                })
                continue
            # "conservative" mode falls through to normal compression but
            # with bit_width override below.

        bsz, n_kv, seq, hd = k_ref.shape
        assert bsz == 1
        k_np = k_ref[0].to(torch.float32).permute(1, 0, 2).cpu().numpy()  # [seq, n_kv, hd]
        v_np = v_ref[0].to(torch.float32).permute(1, 0, 2).cpu().numpy()
        # Q-precondition: apply whitening BEFORE flattening so every row
        # goes through the codec already in Sigma_q-whitened coordinates.
        # The codec sees "an attention-importance-normalised K stream" and
        # its MSE loss is now a faithful proxy for the true InnerProduct
        # distortion.  The untouched tail is also whitened (no info loss —
        # unwhiten∘whiten = I to fp32 precision, ~6e-6 relative).
        use_qp = q_precond is not None and compress in ("kv", "k_only")
        if use_qp:
            k_np_enc = q_precond.whiten(k_np, layer=i)
        else:
            k_np_enc = k_np
        k_flat = k_np_enc.reshape(-1, hd).astype(np.float32, copy=False)
        v_flat = v_np.reshape(-1, hd).astype(np.float32, copy=False)
        n_total = k_flat.shape[0]
        n_comp = (n_total // block_size) * block_size
        target_rank = max(2, int(hd * rsvd_target_rank_factor))

        bit_width_k_eff = bit_width
        bit_width_v_eff = bit_width if bit_width_v is None else bit_width_v
        # Centroid files are calibrated for a specific bit width; override
        # to None on boundary layers where we change bit_width.
        k_centroids_eff = k_centroids_file
        v_centroids_eff = v_centroids_file
        # Conservative boundary protection: override both K and V bit widths
        # to `boundary_bit_width` on boundary layers only.
        is_boundary = i in boundary_skip_set and boundary_mode == "conservative"
        if is_boundary:
            bit_width_k_eff = boundary_bit_width
            bit_width_v_eff = boundary_bit_width
            # Don't use mid-layer calibrated centroids at a different bit width.
            k_centroids_eff = None
            v_centroids_eff = None
        # Asymmetric K/V codec selection: V may use a different codec
        # than K (useful because K cares about inner-product / Σ_q-weighted
        # distortion while V cares about plain MSE — so Besicovitch's
        # Haar-uniform codebook is a natural fit for V).
        codec_v_eff = codec_v if codec_v is not None else codec

        def _encode_k(kv_flat_slice, codec_name):
            """Encode K stream with the specified codec. Returns (decoded, report)."""
            if codec_name == "kakeyaturbo":
                k_metric = "mse" if use_qp else "inner_product"
                return rust_roundtrip(
                    kv_flat_slice, block_size=block_size, bit_width=bit_width_k_eff,
                    rsvd_target_rank=target_rank, metric=k_metric,
                    share_basis=share_basis_k, pca_method=pca_method,
                    variance_ratio=variance_ratio,
                    skeleton_dtype=skeleton_dtype,
                    exact_rank_cap=exact_rank_cap,
                    centroids_file=k_centroids_eff,
                    outlier_threshold=(None if is_boundary else k_outlier_threshold),
                    residual_besi=(None if is_boundary else k_residual_besi),
                )
            elif codec_name == "turboquant":
                return turboquant_k_roundtrip(
                    kv_flat_slice, bit_width=bit_width_k_eff, seed=42 + i * 2,
                )
            elif codec_name == "besicovitch":
                if is_boundary:
                    # Besicovitch struggles on L=0-type extreme-magnitude
                    # boundary layers (even with mean subtraction).  Fall
                    # back to Kakeya-PCA b=boundary_bit_width for safety.
                    return rust_roundtrip(
                        kv_flat_slice, block_size=block_size,
                        bit_width=bit_width_k_eff,
                        rsvd_target_rank=target_rank, metric="mse",
                        share_basis=False, pca_method="exact",
                        variance_ratio=variance_ratio,
                        skeleton_dtype=skeleton_dtype,
                        exact_rank_cap=exact_rank_cap,
                        centroids_file=None,
                    )
                return besicovitch_roundtrip(
                    kv_flat_slice, block_size=block_size,
                    group_size=besi_group_size,
                    direction_bits=besi_direction_bits,
                    magnitude_bits=besi_magnitude_bits,
                    magnitude_mode=besi_magnitude_mode,
                    subtract_mean=besi_subtract_mean,
                )
            elif codec_name == "riemann_besi":
                # Riemannian K-Besi: per-(layer, group) offline-calibrated scale.
                # Breaks the Q-precond + quantized-magnitude trilemma by
                # replacing per-vector max|α_k| with a stable fixed scale.
                # Requires the input to already be Q-preconditioned (whitened)
                # — harness does this when --q-precondition is set.
                if is_boundary:
                    return rust_roundtrip(
                        kv_flat_slice, block_size=block_size,
                        bit_width=bit_width_k_eff,
                        rsvd_target_rank=target_rank, metric="mse",
                        share_basis=False, pca_method="exact",
                        variance_ratio=variance_ratio,
                        skeleton_dtype=skeleton_dtype,
                        exact_rank_cap=exact_rank_cap,
                        centroids_file=None,
                    )
                from benchmarks.k_riemann_besi_codec import (
                    calibrate_offline_scales, roundtrip_k_whitened,
                    load_centroids_file,
                )
                scales = calibrate_offline_scales(
                    kv_flat_slice, g=besi_group_size,
                    method=riemann_scale_method)
                cal_cents = None
                if riemann_centroids_file:
                    cal_cents = load_centroids_file(
                        riemann_centroids_file, besi_magnitude_bits)
                rec, rep = roundtrip_k_whitened(
                    kv_flat_slice, block_size=block_size,
                    g=besi_group_size,
                    direction_bits=besi_direction_bits,
                    magnitude_bits=besi_magnitude_bits,
                    scales=scales,
                    subtract_mean=besi_subtract_mean,
                    calibrated_centroids=cal_cents,
                )
                return rec, rep
            raise ValueError(f"unknown codec: {codec_name}")

        def _encode_v(kv_flat_slice, codec_name):
            """Encode V stream with the specified codec. Returns (decoded, report)."""
            if codec_name == "kakeyaturbo":
                rank_cap_v_eff = (exact_rank_cap_v if exact_rank_cap_v is not None
                                  else exact_rank_cap)
                pca_v_eff = (pca_method_v if pca_method_v is not None
                             else pca_method)
                return rust_roundtrip(
                    kv_flat_slice, block_size=block_size, bit_width=bit_width_v_eff,
                    rsvd_target_rank=target_rank, metric="mse",
                    share_basis=share_basis_v, pca_method=pca_v_eff,
                    variance_ratio=variance_ratio,
                    skeleton_dtype=skeleton_dtype,
                    exact_rank_cap=rank_cap_v_eff,
                    centroids_file=v_centroids_eff,
                    outlier_threshold=(None if is_boundary else v_outlier_threshold),
                )
            elif codec_name == "turboquant":
                return turboquant_v_roundtrip(
                    kv_flat_slice, bit_width=bit_width_v_eff, seed=42 + i * 2 + 1,
                )
            elif codec_name == "besicovitch":
                if is_boundary:
                    # Same boundary safety as K-side: fall back to Kakeya-PCA
                    # on L=0-type layers where V may also be badly-behaved.
                    return rust_roundtrip(
                        kv_flat_slice, block_size=block_size,
                        bit_width=bit_width_v_eff,
                        rsvd_target_rank=target_rank, metric="mse",
                        share_basis=share_basis_v, pca_method="exact",
                        variance_ratio=variance_ratio,
                        skeleton_dtype=skeleton_dtype,
                        exact_rank_cap=None,
                        centroids_file=None,
                    )
                return besicovitch_roundtrip(
                    kv_flat_slice, block_size=block_size,
                    group_size=besi_group_size,
                    direction_bits=besi_direction_bits,
                    magnitude_bits=besi_magnitude_bits,
                    magnitude_mode=besi_magnitude_mode,
                    subtract_mean=besi_subtract_mean,
                )
            elif codec_name == "riemann_besi":
                # Riemann-Besi is K-specific. If V is asked to use it,
                # fall through to Kakeya-PCA (V doesn't benefit from
                # Σ_q-weighted distortion; V's correct metric is MSE).
                rank_cap_v_eff = (exact_rank_cap_v if exact_rank_cap_v is not None
                                  else exact_rank_cap)
                pca_v_eff = (pca_method_v if pca_method_v is not None
                             else pca_method)
                return rust_roundtrip(
                    kv_flat_slice, block_size=block_size, bit_width=bit_width_v_eff,
                    rsvd_target_rank=target_rank, metric="mse",
                    share_basis=share_basis_v, pca_method=pca_v_eff,
                    variance_ratio=variance_ratio,
                    skeleton_dtype=skeleton_dtype,
                    exact_rank_cap=rank_cap_v_eff,
                    centroids_file=v_centroids_eff,
                    outlier_threshold=(None if is_boundary else v_outlier_threshold),
                )
            raise ValueError(f"unknown codec: {codec_name}")

        if n_comp > 0:
            if compress in ("kv", "k_only"):
                k_dec, k_rep = _encode_k(k_flat[:n_comp], codec)
            else:
                k_dec = k_flat[:n_comp].copy()
                k_rep = {"mean_block_mse": 0.0, "skipped": True}
            if compress in ("kv", "v_only"):
                v_dec, v_rep = _encode_v(v_flat[:n_comp], codec_v_eff)
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
        # Un-whiten the K stream (entire sequence, including the untouched
        # tail — unwhiten∘whiten = I for uncompressed rows).
        if use_qp:
            k_full = q_precond.unwhiten(k_full, layer=i)

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
             prefill_chunk, pca_method, vr, compress,
             skeleton_dtype, share_basis_k, share_basis_v,
             q_precond=None, rsvd_rank_factor=0.5,
             exact_rank_cap=None, codec="kakeyaturbo",
             bit_width_v=None, exact_rank_cap_v=None,
             pca_method_v=None,
             k_centroids_file=None, v_centroids_file=None,
             boundary_skip_layers=None, boundary_mode="bf16", boundary_bit_width=4,
             k_outlier_threshold=None, v_outlier_threshold=None,
             besi_group_size=2, besi_direction_bits=5, besi_magnitude_bits=4,
             besi_magnitude_mode="quantized", besi_subtract_mean=True,
             codec_v=None,
             k_residual_besi=None,
             riemann_scale_method="sqrt_trace",
             riemann_centroids_file=None):
    ids = tok(passage, return_tensors="pt")["input_ids"]
    if ids.shape[-1] < ctx_len + n_eval:
        return None
    prefix = ids[:, :ctx_len]
    cont = ids[:, ctx_len:ctx_len + n_eval]

    cache_ref = prefill_cache(model, prefix, prefill_chunk)
    cache_alt, stats = roundtrip_cache(
        model, cache_ref, block_size=block_size, bit_width=bit_width,
        pca_method=pca_method, variance_ratio=vr, compress=compress,
        skeleton_dtype=skeleton_dtype,
        share_basis_k=share_basis_k, share_basis_v=share_basis_v,
        q_precond=q_precond, rsvd_target_rank_factor=rsvd_rank_factor,
        exact_rank_cap=exact_rank_cap,
        codec=codec,
        bit_width_v=bit_width_v,
        exact_rank_cap_v=exact_rank_cap_v,
        pca_method_v=pca_method_v,
        k_centroids_file=k_centroids_file,
        v_centroids_file=v_centroids_file,
        boundary_skip_layers=boundary_skip_layers,
        boundary_mode=boundary_mode,
        boundary_bit_width=boundary_bit_width,
        k_outlier_threshold=k_outlier_threshold,
        v_outlier_threshold=v_outlier_threshold,
        besi_group_size=besi_group_size,
        besi_direction_bits=besi_direction_bits,
        besi_magnitude_bits=besi_magnitude_bits,
        besi_magnitude_mode=besi_magnitude_mode,
        besi_subtract_mean=besi_subtract_mean,
        codec_v=codec_v,
        k_residual_besi=k_residual_besi,
        riemann_scale_method=riemann_scale_method,
        riemann_centroids_file=riemann_centroids_file,
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
    ap.add_argument("--rsvd-rank-factor", type=float, default=0.5,
                    help="RSVD target_rank / D. Default 0.5 = D/2.")
    ap.add_argument("--exact-rank-cap", type=int, default=None,
                    help="Hard cap on d_eff in exact PCA (like RSVD's target_rank "
                         "but without the RSVD approximation error).")
    ap.add_argument("--riemann-centroids-file", type=str, default=None,
                    help="Path to .f32 binary file with calibrated Lloyd-Max "
                         "centroids for the Riemannian K-Besi magnitude quantizer. "
                         "Produced by benchmarks/riemann_calibrate_codebook.py. "
                         "When omitted, uses unit-Gaussian default centroids.")
    ap.add_argument("--riemann-scale-method",
                    choices=["sqrt_trace", "rms_alpha", "pct95_alpha",
                             "pct99_alpha", "pct999_alpha"],
                    default="sqrt_trace",
                    help="Per-group offline scale calibration method for "
                         "Riemannian K-Besi codec. pct99_alpha is better for "
                         "heavy-tailed whitened K distributions.")
    ap.add_argument("--codec", choices=["kakeyaturbo", "turboquant", "besicovitch", "riemann_besi"],
                    default="kakeyaturbo",
                    help="Codec to apply to the K stream (also V if --codec-v "
                         "is not set).\n"
                         "'kakeyaturbo': PCA/RSVD skeleton + Lloyd-Max residual.\n"
                         "'turboquant': reference PolarQuant+QJL (K) / PolarQuant (V).\n"
                         "'besicovitch': fixed direction codebook + per-group "
                         "magnitude, optional per-block mean (no per-block PCA).")
    ap.add_argument("--codec-v",
                    choices=["kakeyaturbo", "turboquant", "besicovitch", "riemann_besi"],
                    default=None,
                    help="Asymmetric V-stream codec. If unset, V uses --codec. "
                         "V cares about MSE (not inner-product), so Besicovitch's "
                         "Haar-uniform codebook may be well-matched for V "
                         "while K stays on Kakeya-PCA.")
    ap.add_argument("--bit-width-v", type=int, default=None,
                    help="If set, overrides --bit-width for the V stream "
                         "(asymmetric K/V codec).  Default: V uses --bit-width "
                         "(symmetric).")
    ap.add_argument("--exact-rank-cap-v", type=int, default=None,
                    help="If set, overrides --exact-rank-cap for the V stream. "
                         "V tolerates aggressive rank caps better than K because "
                         "attention consumes V linearly (no softmax non-linearity).")
    ap.add_argument("--pca-method-v", choices=["exact", "randomized"], default=None,
                    help="If set, overrides --pca-method for the V stream "
                         "(per-stream PCA method). RSVD gives a natural rank cap "
                         "at target_rank = D/2, which combined with share_basis "
                         "can amortise V skeleton bytes beyond what exact PCA "
                         "can achieve at vr=1.0.")
    ap.add_argument("--k-centroids-file", type=Path, default=None,
                    help="Calibrated Lloyd-Max centroids for K-stream residual "
                         "quantiser (output of benchmarks/lloyd_max_calibration.py "
                         "--stream K).")
    ap.add_argument("--v-centroids-file", type=Path, default=None,
                    help="Calibrated Lloyd-Max centroids for V-stream.")
    ap.add_argument("--boundary-skip-layers", type=int, nargs="+", default=None,
                    help="Layers that get boundary protection. Typical: "
                         "[0, 1, L-2, L-1] where L = num_hidden_layers.")
    ap.add_argument("--boundary-mode", choices=["bf16", "conservative"], default="bf16",
                    help="Protection mode for boundary layers. "
                         "'bf16': no codec (full precision). "
                         "'conservative': apply codec at --boundary-bit-width "
                         "(default 4) instead of --bit-width on these layers.")
    ap.add_argument("--boundary-bit-width", type=int, default=4,
                    help="Bit width for boundary layers when --boundary-mode=conservative.")
    ap.add_argument("--besi-group-size", type=int, default=2,
                    help="Besicovitch: group size g (D % g == 0).")
    ap.add_argument("--besi-direction-bits", type=int, default=5,
                    help="Besicovitch: direction codebook size = 2^direction_bits.")
    ap.add_argument("--besi-magnitude-bits", type=int, default=4,
                    help="Besicovitch: bits for magnitude quantization.")
    ap.add_argument("--besi-magnitude-mode", type=str, default="quantized",
                    choices=["f16", "quantized"],
                    help="Besicovitch: how to encode the per-group magnitude.")
    ap.add_argument("--besi-no-subtract-mean", action="store_true",
                    help="Besicovitch: disable per-block mean subtraction.")
    ap.add_argument("--k-residual-besi-direction-bits", type=int, default=0,
                    help="When >0, replace Lloyd-Max K residual quantizer with "
                         "Besicovitch-product codec (2^direction_bits directions).")
    ap.add_argument("--k-residual-besi-group-size", type=int, default=2,
                    help="K-residual Besi group size g (default 2 = unit circle).")
    ap.add_argument("--k-residual-besi-magnitude-bits", type=int, default=4,
                    help="K-residual Besi magnitude quantization bits.")
    ap.add_argument("--k-residual-besi-magnitude-mode", type=str, default="quantized",
                    choices=["f16", "quantized"],
                    help="K-residual Besi magnitude encoding mode.")
    ap.add_argument("--k-outlier-threshold", type=float, default=None,
                    help="If set, outlier compensation threshold on the K "
                         "residual quantizer (scaled-residual space). "
                         "Coordinates with |scaled_residual| > T are stored "
                         "exact (u16+f16 = 4 bytes each). Typical T=2.0.")
    ap.add_argument("--v-outlier-threshold", type=float, default=None,
                    help="Same as --k-outlier-threshold but for V stream.")
    ap.add_argument("--compress", choices=["kv", "k_only", "v_only"], default="kv")
    ap.add_argument("--skeleton-dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--share-basis-k", action="store_true",
                    help="Layer-shared PCA basis on K stream (default off: per-block)")
    ap.add_argument("--share-basis-v", action="store_true",
                    help="Layer-shared PCA basis on V stream (default off; use explicit flag)")
    ap.add_argument("--q-precondition", type=Path, default=None,
                    help="Path to Q-precondition calibration .safetensors (from "
                         "benchmarks/q_calibration.py).  When set, K stream is "
                         "whitened by L = chol(Sigma_q) before the codec and "
                         "un-whitened by L^{-1} after decode, converting the "
                         "codec's MSE into a faithful proxy for Sigma_q-weighted "
                         "(attention-importance) distortion on K.")
    ap.add_argument("--q-precond-skip-layers", type=int, nargs="+", default=[0],
                    help="Layer indices to EXCLUDE from Q-preconditioning. "
                         "Layer 0 is skipped by default because on many models "
                         "it carries attention-sink K values whose Sigma_q "
                         "has huge eigenvalues, causing L^{-1} unwhitening "
                         "to amplify codec error beyond f16 range.")
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

    q_precond = load_q_precond(args.q_precondition,
                                skip_layers=args.q_precond_skip_layers)
    if q_precond is not None:
        print(f"  Q-preconditioning: loaded {q_precond.n_calibrated_layers} layers, "
              f"n_kv={q_precond.n_kv}, D={q_precond.head_dim}, "
              f"skip_layers={sorted(q_precond.skip_layers)}")
        from benchmarks.q_precondition import sanity_check
        san = sanity_check(q_precond)
        print(f"  Q-precondition sanity: max_abs={san['max_abs_err']:.3e}, "
              f"max_rel={san['max_rel_err']:.3e}")
    else:
        print("  Q-preconditioning: OFF")

    passages = load_wikitext_passages(tok, args.ctx_len + args.n_eval, args.n_passages)
    print(f"  got {len(passages)} WikiText passages")

    per_passage = []
    for i, p in enumerate(passages):
        print(f"  passage {i+1}/{len(passages)}…", flush=True)
        res = evaluate(
            model, tok, p, args.ctx_len, args.n_eval, args.block_size,
            args.bit_width, args.prefill_chunk, args.pca_method,
            args.variance_ratio, args.compress,
            args.skeleton_dtype, args.share_basis_k, args.share_basis_v,
            q_precond, args.rsvd_rank_factor,
            args.exact_rank_cap, args.codec,
            args.bit_width_v, args.exact_rank_cap_v,
            args.pca_method_v,
            str(args.k_centroids_file) if args.k_centroids_file else None,
            str(args.v_centroids_file) if args.v_centroids_file else None,
            args.boundary_skip_layers,
            args.boundary_mode,
            args.boundary_bit_width,
            args.k_outlier_threshold,
            args.v_outlier_threshold,
            args.besi_group_size,
            args.besi_direction_bits,
            args.besi_magnitude_bits,
            args.besi_magnitude_mode,
            not args.besi_no_subtract_mean,
            args.codec_v,
            (None if args.k_residual_besi_direction_bits == 0 else {
                "direction_bits": args.k_residual_besi_direction_bits,
                "magnitude_bits": args.k_residual_besi_magnitude_bits,
                "magnitude_mode": args.k_residual_besi_magnitude_mode,
                "group_size": args.k_residual_besi_group_size,
            }),
            args.riemann_scale_method,
            args.riemann_centroids_file,
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
        print(f"  compress={args.compress} bit_width={args.bit_width} vr={args.variance_ratio} "
              f"pca={args.pca_method} skeleton={args.skeleton_dtype} "
              f"share_k={args.share_basis_k} share_v={args.share_basis_v}")
        print(f"  Δppl mean = {md*100:+.3f}%")
        print(f"  top1 mean = {mt*100:.2f}%")
        print(f"  KL mean   = {mk:.5f}")

    out_name = (
        f"{args.model_name}_prerope_{args.compress}_b{args.bit_width}"
        f"_{args.pca_method}_{args.skeleton_dtype}"
        f"_sk{int(args.share_basis_k)}_sv{int(args.share_basis_v)}.json"
    )
    (args.out_dir / out_name).write_text(json.dumps({
        "model_name": args.model_name, "ctx_len": args.ctx_len,
        "n_eval": args.n_eval, "bit_width": args.bit_width,
        "pca_method": args.pca_method, "variance_ratio": args.variance_ratio,
        "skeleton_dtype": args.skeleton_dtype,
        "share_basis_k": args.share_basis_k, "share_basis_v": args.share_basis_v,
        "compress": args.compress, "architecture": "pre_rope_cache",
        "per_passage": per_passage,
    }, indent=2))


if __name__ == "__main__":
    main()
