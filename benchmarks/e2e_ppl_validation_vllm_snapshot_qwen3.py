#!/usr/bin/env python3
"""Snapshot-mode vLLM harness for Qwen3 (Scenario A).

Same two-pass semantics as `e2e_ppl_validation_vllm_snapshot.py`
(which targeted Qwen2) — see the docstring there for the full
rationale.  This version adapts the capture + replace hook to vLLM's
`Qwen3Attention` module, which differs from Qwen2 in three ways:

  1. Qwen3Attention has per-head QK-norm (`q_norm`, `k_norm`) applied
     *after* the qkv projection and *before* the RoPE rotation.  The
     natural codec-capture point is POST-qk-norm, PRE-RoPE — i.e. the
     tensor that best corresponds to Qwen2's pre-RoPE K.  This matches
     the Σ_q calibration from M2 (`qwen3_4b_sigma_q.safetensors`)
     because that calibration also captured queries after q_norm.

  2. Newer vLLM's Qwen3Attention.forward signature is
     `(positions, hidden_states)` only — no `kv_cache` / `attn_metadata`
     argument (the backend pulls them itself via `self.attn(...)`).

  3. The `llm.generate(prompts=None, prompt_token_ids=[ids], ...)`
     API is deprecated; we use `TokensPrompt` instead.

All other machinery (WikiText loader, Rust codec CLI, Q-precond,
Lloyd-Max centroids, boundary-skip) is reused verbatim.

Canonical named configurations
------------------------------

See `reports/v1_3_ppl/snapshot_mode_qwen3/NAMING.md` for the
full naming schema.  The current best-measured recipes on
Qwen3-4B + WikiText-103 test (ctx=2048, n_eval=64, 4 passages):

  v1.3-GPU-Qwen-snap-bK64-bdry14   spoken: v1.3-GPU-snapA
      Δppl-optimal. Δppl = +61.84%, top-1 = 79.30%.
      Flags:
        --bit-width-k 4 --k-kmeans-k 64 --rsvd-target-rank-factor 0.75
        --bit-width-v 2 --v-kmeans-k 16
        --boundary-skip-layers 0 1 2 3 4 5 6 29 30 31 32 33 34 35
        --gpu-codec --no-share-basis-v
        --disable-q-precond --disable-centroids --disable-outlier

  v1.3-GPU-Qwen-snap-bK128-bdry14  spoken: v1.3-GPU-snapB
      top-1-optimal. Δppl = +65.98%, top-1 = 81.64%.
      Same as snapA but with `--k-kmeans-k 128`.

Usage (Qwen3-4B, plain defaults):
    python benchmarks/e2e_ppl_validation_vllm_snapshot_qwen3.py \\
        --model-path Qwen/Qwen3-4B --model-name qwen3_4b_snapshot \\
        --ctx-len 2048 --n-eval 64 --n-passages 4 \\
        --q-calib reports/v1_3_ppl/vllm_backend/calibration/\\
qwen3_4b_sigma_q.safetensors \\
        --k-centroids reports/v1_3_ppl/vllm_backend/calibration/\\
qwen3_4b_lloyd_max_K_b3.f32 \\
        --v-centroids reports/v1_3_ppl/vllm_backend/calibration/\\
qwen3_4b_lloyd_max_V_b2.f32 \\
        --out-dir reports/v1_3_ppl/snapshot_mode_qwen3
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
BENCH_BIN = REPO / "kakeyaturbo" / "target" / "release" / "kakeyaturbo-bench"
KKTV_MAGIC = 0x4B4B_5456

# Opt into the snapshot-hook plugin BEFORE vLLM is imported.
#
# Critical: VLLM_ENABLE_V1_MULTIPROCESSING=0 forces vLLM v1's
# LLMEngine into `InprocClient` mode (EngineCoreClient.make_client,
# vllm/v1/engine/core_client.py:~80).  That keeps model forward
# inside the same Python process as the harness, which is required
# for the monkey-patch-based hook: `HookState` is module-level
# Python state, and a fork/spawn would give the engine subprocess
# its own copy of the class attribute that the harness cannot reach.
import os  # noqa: E402
os.environ["KAKEYA_SNAPSHOT_QWEN3"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

sys.path.insert(0, str(REPO / "benchmarks"))
from q_precondition import QPrecond, load as qp_load  # noqa: E402

# Import HookState directly from the plugin so the harness and the
# patched forward share the same module-level state (this is the
# normal way to do it — Python caches imports per process).  The
# package is installed via `vllm_backend/pyproject.toml` as
# `kakeya_v1_3_ppl` (top-level), not under the source-tree
# `vllm_backend.` namespace.
from kakeya_v1_3_ppl.snapshot_hook import HookState  # noqa: E402


# =============================================================================
# KKTV I/O + rust codec (identical to Qwen2 harness)
# =============================================================================

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


def read_kktv_f32(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == KKTV_MAGIC
        struct.unpack("<I", f.read(4))
        n = struct.unpack("<Q", f.read(8))[0]
        d = struct.unpack("<I", f.read(4))[0]
        struct.unpack("<I", f.read(4))
        raw = f.read(n * d * 4)
    return np.frombuffer(raw, dtype=np.float32).reshape(n, d).copy()


def rust_roundtrip(
    arr: np.ndarray, *, block_size: int, bit_width: int,
    rsvd_target_rank: int, metric: str, share_basis: bool,
    pca_method: str = "randomized", variance_ratio: float = 0.95,
    centroids_file: str | None = None,
    outlier_threshold: float | None = None,
) -> tuple[np.ndarray, dict]:
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        tdp = Path(td)
        in_p, rep, dec = tdp/"x.kktv", tdp/"r.json", tdp/"d.kktv"
        write_kktv(in_p, arr.astype(np.float32, copy=False))
        cmd = [
            str(BENCH_BIN), "--input", str(in_p), "--output", str(rep),
            "--metric", metric, "--block-size", str(block_size),
            "--variance-ratio", str(variance_ratio),
            "--k", "16", "--bit-width", str(bit_width),
            "--rotation-seed", "3405691582",
            "--pca-method", pca_method, "--verify",
            "--dump-decoded", str(dec),
        ]
        if pca_method == "randomized":
            cmd += ["--rsvd-target-rank", str(rsvd_target_rank),
                    "--rsvd-oversample", "8", "--rsvd-power-iters", "2"]
        if share_basis:
            cmd.append("--share-basis")
        if centroids_file is not None:
            cmd += ["--centroids-file", str(centroids_file)]
        if outlier_threshold is not None:
            cmd += ["--outlier-threshold", str(outlier_threshold)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"codec rc={res.returncode}: "
                               f"{res.stderr[:2000]}")
        return read_kktv_f32(dec), json.loads(rep.read_text())


# =============================================================================
# GPU per-head codec roundtrip
# =============================================================================
#
# Strict CPU-vs-GPU comparison on real Qwen3-4B layer-15 post-qk-norm K
# (see /tmp/strict_cpu_vs_gpu.py run on H200) showed the CPU CLI path
# decoding at 37.66% L2 rel-err while the GPU per-head path achieved
# 16.67% rel-err on the SAME input.  The root cause is the data-grouping
# difference:
#
#   CPU path  flattens K [n, H, D] → [n*H, D]       and slices 512 rows
#             into each codec block, so one block sees 64 tokens × 8 heads
#             interleaved.  Qwen3-style qk-norm has per-head independent
#             scales; the pooled-heads PCA basis can only capture
#             shared directions and wastes d_eff on the head-scale
#             variance rather than on intra-head signal.
#
#   GPU path  processes each kv-head independently: K [n, H, D] →
#             [H, n, D] with batch=H.  Each block's PCA basis is fit to
#             a single head's distribution, so all d_eff basis rows
#             point at head-intrinsic variance.
#
# This per-head helper replaces `rust_roundtrip()` when --gpu-codec is
# passed on the CLI.  It reuses the exact same GPU kernels the vLLM
# attention backend uses in production (fit_skeleton_batched +
# encode_and_pack_batched + _unpack_slot_into_parts +
# decode_block_torch_from_parts), so the harness measures the
# same codec as the live deployment path.

def _gpu_codec_per_head(
    K_bnd: np.ndarray,
    *,
    block_size: int,
    bit_width: int,
    d_eff: int,
    k: int,
    metric: str,
    variance_ratio: float,
    centroids_file: str | None,
    outlier_threshold: float | None,
) -> np.ndarray:
    """Run the GPU codec per kv-head on a [n, H, D] fp32 tensor and
    return the decoded [n, H, D] fp32 array.

    Only full codec blocks are processed (trailing partial rows are
    returned unchanged).  `centroids_file`, if given, is a raw fp32
    file with `2^bit_width` entries matching
    `kakeyaturbo_py._core.centroids_gaussian(bit_width)`'s layout.
    """
    import torch
    # Package name is `kakeya_v1_3_ppl` as installed via
    # `vllm_backend/pyproject.toml`; NOT the source-tree path.
    from kakeya_v1_3_ppl.config import KakeyaV13PPLConfig
    from kakeya_v1_3_ppl.impl import KakeyaV13PPLAttentionImpl
    from kakeyaturbo_py.gpu_skeleton import fit_skeleton_batched
    from kakeyaturbo_py.gpu_encode import encode_and_pack_batched
    from kakeyaturbo_py.reference_torch import decode_block_torch_from_parts
    from kakeyaturbo_py import _core

    n_total, H, D = K_bnd.shape
    n_comp = (n_total // block_size) * block_size
    if n_comp == 0:
        return K_bnd.astype(np.float32, copy=False)

    cfg = KakeyaV13PPLConfig(
        head_dim=D, d_eff=d_eff, block_size_codec=block_size,
        variance_ratio=variance_ratio, k_centers=k, bit_width=bit_width,
        # Budget large enough to accept outliers on any realistic
        # scaled-residual distribution; harmless when
        # outlier_threshold is None (the side-buffer is left at 0).
        outlier_budget_frac=0.15 if outlier_threshold is not None else 0.0,
    )
    impl = KakeyaV13PPLAttentionImpl(
        num_heads=H, head_size=D, scale=1.0,
        num_kv_heads=H, kv_cache_dtype="kakeya_v1_3_ppl",
    )
    cfg_offsets = impl._config_offsets(cfg)

    # Custom Lloyd-Max centroids (optional).  Raw fp32 file, `2^bit_width`
    # little-endian floats — same format kakeyaturbo-bench expects.
    if centroids_file is not None:
        raw = np.fromfile(centroids_file, dtype=np.float32)
        expected = 1 << int(bit_width)
        if raw.size != expected:
            raise ValueError(
                f"centroids file {centroids_file} has {raw.size} f32 "
                f"values, expected 2^bit_width = {expected}"
            )
        centroids_gpu = torch.from_numpy(raw).cuda()
    else:
        centroids_gpu = None

    K_gpu = torch.from_numpy(
        K_bnd[:n_comp].astype(np.float32, copy=False)
    ).cuda()                                           # [n_comp, H, D]

    out = np.empty((n_total, H, D), dtype=np.float32)
    if n_comp < n_total:
        out[n_comp:] = K_bnd[n_comp:]

    n_blocks = n_comp // block_size
    for bi in range(n_blocks):
        lo, hi = bi * block_size, (bi + 1) * block_size
        # [H, block_size, D] — batch = num_kv_heads.
        K_block = K_gpu[lo:hi].permute(1, 0, 2).contiguous()
        skel = fit_skeleton_batched(
            K_block, d_eff=d_eff, k=k, seed=3405691582,
            rsvd_oversample=8, rsvd_power_iters=2,
            kmeans_max_iter=8, variance_ratio=variance_ratio,
        )
        sign_np = np.asarray(
            _core.wht_sign_pattern(
                int(skel["rotation_seed"]), int(skel["wht_len"]),
            )
        ).reshape(-1).astype(np.float32)
        sign = torch.from_numpy(sign_np).cuda()

        slots = encode_and_pack_batched(
            K_block, skel,
            bit_width=bit_width, metric=metric,
            slot_size_bytes=cfg.slot_size_bytes,
            config_offsets=cfg_offsets,
            custom_centroids=centroids_gpu,
            outlier_threshold=outlier_threshold,
            wht_sign=sign,
        )                                              # [H, slot_bytes]

        # Decode each head slot (decode_block_torch_from_parts is
        # light-weight and byte-matched to the production decoder).
        for h in range(H):
            slot_cpu = slots[h].cpu().numpy()
            parts = impl._unpack_slot_into_parts(
                slot_cpu, cfg, head_size=D,
            )
            dec_h = decode_block_torch_from_parts(parts, device="cpu")
            dec_h = dec_h.numpy() if hasattr(dec_h, "numpy") else np.asarray(dec_h)
            out[lo:hi, h, :] = dec_h

    return out


def gpu_roundtrip(
    K_bnd: np.ndarray,
    *,
    block_size: int,
    bit_width: int,
    rsvd_target_rank: int,
    metric: str,
    variance_ratio: float = 0.95,
    centroids_file: str | None = None,
    outlier_threshold: float | None = None,
    k_centers: int = 16,
) -> tuple[np.ndarray, dict]:
    """Signature-compatible stand-in for `rust_roundtrip` operating
    on the [n, H, D] pre-flatten layout.  Returns (decoded, report)."""
    n, H, D = K_bnd.shape
    dec = _gpu_codec_per_head(
        K_bnd,
        block_size=block_size,
        bit_width=bit_width,
        d_eff=rsvd_target_rank,
        k=k_centers,
        metric=metric,
        variance_ratio=variance_ratio,
        centroids_file=centroids_file,
        outlier_threshold=outlier_threshold,
    )
    # Best-effort report: mean per-block MSE + compressed-bytes
    # estimate (matching kakeyaturbo-bench's report keys).
    mse = float(np.mean((dec - K_bnd.astype(np.float32)) ** 2))
    # Compressed bytes per block are deterministic from config;
    # compute via KakeyaV13PPLConfig.slot_size_bytes for one head × one block.
    from kakeya_v1_3_ppl.config import KakeyaV13PPLConfig
    cfg = KakeyaV13PPLConfig(
        head_dim=D, d_eff=rsvd_target_rank,
        block_size_codec=block_size,
        variance_ratio=variance_ratio,
        k_centers=k_centers, bit_width=bit_width,
        outlier_budget_frac=0.15 if outlier_threshold is not None else 0.0,
    )
    n_blocks = (n // block_size)
    compressed_bytes = n_blocks * H * cfg.slot_size_bytes
    return dec, {
        "mean_block_mse": mse,
        "compressed_bytes": compressed_bytes,
        "codec_backend": "gpu-per-head",
    }


# =============================================================================
# Offline codec of a per-layer K (or V) snapshot — same recipe as PR #17
# =============================================================================

def codec_layer(
    K_or_V: np.ndarray, *, is_v: bool, layer_id: int,
    q_precond: QPrecond | None,
    block_size: int, bit_width_k: int, bit_width_v: int,
    k_centroids: str | None, v_centroids: str | None,
    k_outlier_threshold: float | None, v_outlier_threshold: float | None,
    boundary_skip: set[int], rsvd_target_rank_factor: float = 0.5,
    share_basis_v: bool = True, share_basis_k: bool = False,
    use_gpu_codec: bool = False, disable_share_basis_v: bool = False,
    k_kmeans_k: int = 16, v_kmeans_k: int = 16,
) -> tuple[np.ndarray, dict]:
    """Per-layer codec of a captured K/V tensor.

    When `use_gpu_codec=True`, the CPU Rust CLI path is bypassed and
    the codec runs per-kv-head on the [n, H, D] layout — see the
    `_gpu_codec_per_head` docstring for the rationale.  The Rust path
    is retained for exact PR #17 reproducibility.
    """
    n, n_kv, hd = K_or_V.shape
    if layer_id in boundary_skip:
        return K_or_V.astype(np.float32, copy=False), {
            "layer": layer_id, "stream": "V" if is_v else "K",
            "boundary_skip": True,
        }
    rank = max(2, int(hd * rsvd_target_rank_factor))

    use_whiten = (
        (not is_v) and q_precond is not None
        and q_precond.n_kv == n_kv and q_precond.head_dim == hd
        and q_precond.is_active(layer_id)
    )
    arr_enc = q_precond.whiten(K_or_V, layer=layer_id) if use_whiten else K_or_V

    if is_v:
        bit_width = bit_width_v
        centroids = v_centroids
        outlier_thr = v_outlier_threshold
        share = share_basis_v and not disable_share_basis_v
        metric = "mse"
        kmeans_k = v_kmeans_k
    else:
        bit_width = bit_width_k
        centroids = k_centroids
        outlier_thr = k_outlier_threshold
        share = share_basis_k
        metric = "mse" if use_whiten else "inner_product"
        kmeans_k = k_kmeans_k

    if use_gpu_codec:
        # GPU path operates on [n, H, D] directly — no flatten.
        # share_basis is NOT yet implemented on the GPU path; when
        # the caller asked for it we fall back to Rust CLI.  (Qwen3
        # V-stream with share_basis=True needs the pooled-across-
        # blocks basis that only Rust currently provides.)
        if share:
            # Rust CLI bench binary hardcodes --k to the value passed
            # on the command line; rust_roundtrip currently uses
            # k=16 unconditionally (kakeyaturbo-bench's --k flag isn't
            # wired through here).  A non-default kmeans_k on the V
            # stream therefore only takes effect via the GPU path —
            # callers who want to sweep V K-means k must disable
            # share_basis_v.
            dec, rep = rust_roundtrip(
                arr_enc.reshape(-1, hd).astype(np.float32, copy=False),
                block_size=block_size, bit_width=bit_width,
                rsvd_target_rank=rank, metric=metric, share_basis=share,
                centroids_file=centroids, outlier_threshold=outlier_thr,
            )
            # Reshape back to [n, n_kv, hd]; the Rust path uses the
            # pooled-heads layout so we have to un-pool the result.
            dec = dec.reshape(n, n_kv, hd)
        else:
            dec, rep = gpu_roundtrip(
                arr_enc.astype(np.float32, copy=False),
                block_size=block_size, bit_width=bit_width,
                rsvd_target_rank=rank, metric=metric,
                centroids_file=centroids, outlier_threshold=outlier_thr,
                k_centers=kmeans_k,
            )
    else:
        flat = arr_enc.reshape(-1, hd).astype(np.float32, copy=False)
        n_total = flat.shape[0]
        n_comp = (n_total // block_size) * block_size
        if n_comp == 0:
            return K_or_V.astype(np.float32, copy=False), {
                "layer": layer_id, "stream": "V" if is_v else "K",
                "skipped_short": True,
            }
        dec, rep = rust_roundtrip(
            flat[:n_comp], block_size=block_size, bit_width=bit_width,
            rsvd_target_rank=rank, metric=metric, share_basis=share,
            centroids_file=centroids, outlier_threshold=outlier_thr,
        )
        if n_comp < n_total:
            dec = np.concatenate([dec, flat[n_comp:]], axis=0)
        dec = dec.reshape(n, n_kv, hd)

    if use_whiten:
        dec = q_precond.unwhiten(dec, layer=layer_id)
    # Both paths process floor(n / block_size) full blocks and pass
    # any trailing partial rows through unchanged.
    n_compressible = (n // block_size) * block_size
    return dec.astype(np.float32, copy=False), {
        "layer": layer_id, "stream": "V" if is_v else "K",
        "n_tokens": int(n), "n_compressible": int(n_compressible),
        "mean_block_mse": float(rep.get("mean_block_mse", -1.0)),
        "compressed_bytes": int(rep.get("compressed_bytes", 0)),
        "whitened": bool(use_whiten),
        "metric": metric, "bit_width": bit_width,
        "codec_backend": rep.get("codec_backend", "rust-pooled"),
    }


# =============================================================================
# WikiText loader + vLLM driver
# =============================================================================

def load_wikitext_passages(tok: Any, min_tokens: int, n_passages: int,
                           split: str = "test") -> list[str]:
    """Identical to the Qwen2 harness loader."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    passages, cur, approx = [], [], 0
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        cur.append(text)
        approx += int(len(text.split()) * 1.3)
        if approx >= min_tokens:
            passage = "".join(cur)
            if len(tok.encode(passage)) >= min_tokens:
                passages.append(passage)
                if len(passages) >= n_passages:
                    return passages
            cur, approx = [], 0
    return passages


def prompt_logprobs_for_ids(llm: Any, ids: list[int]) -> list[dict]:
    """Drive vLLM with the new TokensPrompt API (the old
    `prompts=None, prompt_token_ids=[...]` kwargs are deprecated
    in the version installed on the H200 box).
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        sampling_params=sp, use_tqdm=False,
    )
    return out[0].prompt_logprobs


def ppl_and_top1(pls: list[Any], ids: list[int],
                 start: int, end: int) -> tuple[float, list[float], list[int]]:
    """NLL over [start, end), PPL = exp(mean NLL). `top1` is the top-1
    agreement indicator against the gold id."""
    lps, top1 = [], []
    for t in range(start, end):
        entry = pls[t]
        if entry is None:
            lps.append(0.0)
            top1.append(0)
            continue
        gold_id = ids[t]
        gold_lp = entry[gold_id].logprob
        lps.append(gold_lp)
        # Top-1 agreement: which candidate has the max logprob?
        best_id = max(entry.items(), key=lambda kv: kv[1].logprob)[0]
        top1.append(int(best_id == gold_id))
    mean_nll = -float(np.mean(lps))
    return float(np.exp(mean_nll)), lps, top1


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B")
    ap.add_argument("--model-name", default="qwen3_4b_snapshot")
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--bit-width-k", type=int, default=3)
    ap.add_argument("--bit-width-v", type=int, default=2)
    ap.add_argument("--rsvd-target-rank-factor", type=float, default=0.5,
                    help="d_eff = max(2, int(head_dim * factor)).  Default "
                         "0.5 (= d_eff=64 on head_dim=128).  Raise toward "
                         "1.0 to trade slot bytes for reconstruction quality.")
    ap.add_argument("--k-kmeans-k", type=int, default=16,
                    help="Spherical K-means cluster count for the K stream's "
                         "stage-3 assignment.  Default 16 matches PR #17.  "
                         "Larger k gives a finer angular codebook at the "
                         "cost of `ceil(log2(k))` bits per token per block.")
    ap.add_argument("--v-kmeans-k", type=int, default=16,
                    help="Same for V stream; must be compatible with "
                         "bit_width_v (V residual carries log2(k) bits "
                         "of seg_id information separately).  Ignored when "
                         "V falls back to the CPU Rust share_basis path.")
    ap.add_argument("--q-calib", type=str,
        default="reports/v1_3_ppl/vllm_backend/calibration/"
                "qwen3_4b_sigma_q.safetensors")
    ap.add_argument("--k-centroids", type=str,
        default="reports/v1_3_ppl/vllm_backend/calibration/"
                "qwen3_4b_lloyd_max_K_b3.f32")
    ap.add_argument("--v-centroids", type=str,
        default="reports/v1_3_ppl/vllm_backend/calibration/"
                "qwen3_4b_lloyd_max_V_b2.f32")
    ap.add_argument("--outlier-threshold", type=float, default=2.0)
    ap.add_argument("--boundary-skip-layers", type=int, nargs="*",
                    default=[0, 1, 34, 35])
    ap.add_argument("--disable-q-precond", action="store_true",
                    help="Skip Σ_q whitening; codec K with raw metric only.")
    ap.add_argument("--disable-centroids", action="store_true",
                    help="Use Gaussian default Lloyd-Max centroids.")
    ap.add_argument("--disable-outlier", action="store_true",
                    help="Skip outlier side-buffer extraction.")
    ap.add_argument("--gpu-codec", action="store_true",
                    help="Route codec through per-kv-head GPU path "
                         "(vllm_backend.kakeya_v1_3_ppl + kakeyaturbo_py "
                         "gpu_skeleton/gpu_encode) instead of the CPU Rust "
                         "CLI pooled-heads path.  See codec_layer docstring.")
    ap.add_argument("--no-share-basis-v", action="store_true",
                    help="Disable share_basis=True on the V stream.  Required "
                         "to exercise the GPU codec on V (which doesn't yet "
                         "support share_basis); without this flag the V "
                         "stream falls back to the CPU Rust path even when "
                         "--gpu-codec is on.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.disable_q_precond:
        print("[setup] Σ_q whitening DISABLED via --disable-q-precond", flush=True)
        qp = None
    else:
        print(f"[setup] loading Q-precond from {args.q_calib}", flush=True)
        qp = qp_load(args.q_calib, skip_layers=[0])
    k_centroids_arg = None if args.disable_centroids else args.k_centroids
    v_centroids_arg = None if args.disable_centroids else args.v_centroids
    outlier_thr_arg = None if args.disable_outlier else args.outlier_threshold
    boundary_skip = set(args.boundary_skip_layers or [])

    # The Qwen3Attention.forward monkey-patch is installed by the
    # kakeya_v1_3_ppl plugin in every vLLM process (including the
    # engine-core subprocess), gated on KAKEYA_SNAPSHOT_QWEN3=1 which
    # we set at the top of this module BEFORE importing vLLM.

    from vllm import LLM
    print(f"[{args.model_name}] loading vLLM engine…", flush=True)
    llm = LLM(
        model=args.model_path, dtype="bfloat16",
        max_model_len=args.ctx_len + args.n_eval + 16,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True, trust_remote_code=True,
    )
    tok = llm.get_tokenizer()

    print(f"[{args.model_name}] loading WikiText passages…", flush=True)
    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval,
        n_passages=args.n_passages,
    )
    passages_ids = [tok.encode(p)[: args.ctx_len + args.n_eval]
                    for p in passages
                    if len(tok.encode(p)) >= args.ctx_len + args.n_eval]
    print(f"  usable: {len(passages_ids)}", flush=True)

    per_passage: list[dict] = []
    codec_stats_total: list[dict] = []

    for pi, ids in enumerate(passages_ids):
        print(f"\n  passage {pi + 1}/{len(passages_ids)}", flush=True)

        # ---- Pass 1: clean prefill (codec OFF), capture per-layer K/V ----
        HookState.phase = "capture"
        HookState.captured = {}
        t0 = time.perf_counter()
        ref_pls = prompt_logprobs_for_ids(llm, ids)
        t_ref = time.perf_counter() - t0
        HookState.phase = "off"
        n_layers_captured = len(HookState.captured)
        n_tokens_captured = HookState.captured[0]["K"].shape[0]
        print(f"    [capture] {n_layers_captured} layers, "
              f"{n_tokens_captured} tokens, {t_ref:.2f}s", flush=True)

        # ---- Offline: per-layer codec (Q-precond + Lloyd-Max + outliers) ----
        t0 = time.perf_counter()
        replacements: dict[int, dict[str, torch.Tensor]] = {}
        stats_this: list[dict] = []
        for lid, kv in HookState.captured.items():
            k_hat, k_rep = codec_layer(
                kv["K"], is_v=False, layer_id=lid, q_precond=qp,
                block_size=args.block_size,
                bit_width_k=args.bit_width_k, bit_width_v=args.bit_width_v,
                k_centroids=k_centroids_arg, v_centroids=v_centroids_arg,
                k_outlier_threshold=outlier_thr_arg,
                v_outlier_threshold=None, boundary_skip=boundary_skip,
                rsvd_target_rank_factor=args.rsvd_target_rank_factor,
                use_gpu_codec=args.gpu_codec,
                disable_share_basis_v=args.no_share_basis_v,
                k_kmeans_k=args.k_kmeans_k, v_kmeans_k=args.v_kmeans_k,
            )
            v_hat, v_rep = codec_layer(
                kv["V"], is_v=True, layer_id=lid, q_precond=qp,
                block_size=args.block_size,
                bit_width_k=args.bit_width_k, bit_width_v=args.bit_width_v,
                k_centroids=k_centroids_arg, v_centroids=v_centroids_arg,
                k_outlier_threshold=outlier_thr_arg,
                v_outlier_threshold=None, boundary_skip=boundary_skip,
                rsvd_target_rank_factor=args.rsvd_target_rank_factor,
                use_gpu_codec=args.gpu_codec,
                disable_share_basis_v=args.no_share_basis_v,
                k_kmeans_k=args.k_kmeans_k, v_kmeans_k=args.v_kmeans_k,
            )
            replacements[lid] = {
                "K": torch.from_numpy(k_hat).cuda(),
                "V": torch.from_numpy(v_hat).cuda(),
            }
            stats_this.extend([k_rep, v_rep])
        t_codec = time.perf_counter() - t0
        n_boundary = sum(1 for s in stats_this if s.get("boundary_skip"))
        print(f"    [codec] {len(HookState.captured)} layers "
              f"({n_boundary} boundary-skipped), {t_codec:.2f}s", flush=True)

        # ---- Pass 2: replace, teacher-force eval tokens ----
        HookState.phase = "replace"
        HookState.replacements = replacements
        HookState.replace_fired = {}
        HookState.replace_shape_mismatch = {}
        HookState.replace_missing = {}
        t0 = time.perf_counter()
        alt_pls = prompt_logprobs_for_ids(llm, ids)
        t_alt = time.perf_counter() - t0
        HookState.phase = "off"
        HookState.replacements = {}
        n_fired = sum(HookState.replace_fired.values())
        n_mismatch = sum(len(v) for v in HookState.replace_shape_mismatch.values())
        n_missing = sum(HookState.replace_missing.values())
        print(f"    [replace] fired={n_fired} mismatch={n_mismatch} "
              f"missing={n_missing}", flush=True)
        if n_mismatch:
            # Show the first 3 mismatches so we can see what shapes vLLM
            # is actually feeding through the hook on pass 2.
            sample = [(l, pairs) for l, pairs in
                      list(HookState.replace_shape_mismatch.items())[:3]]
            print(f"      first mismatches: {sample}", flush=True)

        # Free per-passage GPU replacements so the next passage starts clean.
        del replacements
        torch.cuda.empty_cache()

        ppl_ref, lps_ref, top1_ref = ppl_and_top1(
            ref_pls, ids, args.ctx_len, args.ctx_len + args.n_eval,
        )
        ppl_alt, lps_alt, top1_alt = ppl_and_top1(
            alt_pls, ids, args.ctx_len, args.ctx_len + args.n_eval,
        )
        # Top-1 agreement computed as a pair: fraction of eval positions
        # where the codec-pass top-1 equals the clean-pass top-1.  We
        # need the actual predicted ids, not just "matches gold".
        def _top1_ids(pls: list[Any]) -> list[int]:
            out = []
            for t in range(args.ctx_len, args.ctx_len + args.n_eval):
                entry = pls[t]
                if entry is None:
                    out.append(-1)
                    continue
                best_id = max(entry.items(), key=lambda kv: kv[1].logprob)[0]
                out.append(int(best_id))
            return out
        ref_ids = _top1_ids(ref_pls)
        alt_ids = _top1_ids(alt_pls)
        top1_pair = float(np.mean([
            1 if r == a else 0 for r, a in zip(ref_ids, alt_ids)
        ]))
        delta_ppl = (ppl_alt - ppl_ref) / max(ppl_ref, 1e-9)
        print(f"    ppl_ref={ppl_ref:.3f} ppl_alt={ppl_alt:.3f} "
              f"Δppl={delta_ppl*100:+.2f}% top1={top1_pair*100:.2f}% "
              f"t_alt={t_alt:.2f}s", flush=True)

        per_passage.append({
            "ctx_len": args.ctx_len, "n_eval": args.n_eval,
            "t_ref_sec": t_ref, "t_codec_sec": t_codec, "t_alt_sec": t_alt,
            "ppl_ref": ppl_ref, "ppl_alt": ppl_alt,
            "delta_ppl": delta_ppl, "top1_pair": top1_pair,
            "n_boundary_skipped": n_boundary,
        })
        codec_stats_total.extend(stats_this)

    mean_d = float(np.mean([p["delta_ppl"] for p in per_passage])) if per_passage else 0.0
    mean_top1 = float(np.mean([p["top1_pair"] for p in per_passage])) if per_passage else 0.0
    # Same verdict thresholds used by PR #17.
    if mean_d <= 0.10 and mean_top1 >= 0.80:
        verdict = "PASS"
    elif mean_d <= 0.20 and mean_top1 >= 0.70:
        verdict = "MARGINAL"
    else:
        verdict = "REJECT"

    summary = {
        "model_name": args.model_name, "model_path": args.model_path,
        "engine": "vllm", "recipe": "v1.3 PPL snapshot-mode (Qwen3)",
        "ctx_len": args.ctx_len, "n_eval": args.n_eval,
        "n_passages_total": len(passages_ids),
        "block_size": args.block_size,
        "bit_width_k": args.bit_width_k, "bit_width_v": args.bit_width_v,
        "rsvd_target_rank_factor": args.rsvd_target_rank_factor,
        "k_kmeans_k": args.k_kmeans_k, "v_kmeans_k": args.v_kmeans_k,
        "outlier_threshold": args.outlier_threshold,
        "boundary_skip_layers": sorted(boundary_skip),
        "q_calib": args.q_calib,
        "k_centroids": args.k_centroids, "v_centroids": args.v_centroids,
        "mean_delta_ppl": mean_d, "mean_top1_pair": mean_top1,
        "verdict": verdict,
        "per_passage": per_passage,
        "codec_stats": codec_stats_total,
    }
    out_path = args.out_dir / f"{args.model_name}_vllm_snapshot.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] mean Δppl={mean_d*100:+.2f}%  "
          f"mean top1_pair={mean_top1*100:.2f}%  verdict={verdict}",
          flush=True)
    print(f"       written → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
