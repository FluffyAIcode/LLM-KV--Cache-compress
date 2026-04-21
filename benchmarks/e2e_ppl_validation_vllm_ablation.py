#!/usr/bin/env python3
"""vLLM v1.3 PPL ablation harness.

Goal: localise the gap between the HF v1.3-PPL MARGINAL cell
(+7.82 % Δppl) and the vLLM v1.3-PPL REJECT cell (+35.33 % Δppl)
reported in PR #15, FINDINGS.md.

Two hypotheses, tested with the four cells below:

  H1 — "Q-preconditioning was calibrated on pre-RoPE K, but
        Flash-Attention computes its scores on post-RoPE K. The
        Σ_q -> K̃ metric equivalence therefore does not hold under
        FA's numerics."
  H2 — "Per-forward CPU↔GPU and fp32↔bf16 round-trips accumulate
        enough numerical noise to degrade PPL on their own, even
        with a no-op codec."

Cells
-----
  (O) ref               — codec OFF (reference)
  (I) identity-pre      — whiten pre-RoPE → NO codec → unwhiten
                           (tests H2; identity-codec + the fp32/bf16
                           CPU↔GPU round-trip at the same hook point
                           as production)
  (C) codec-no-qp       — codec ON, no whitening
                           (tests "how much is just the codec?")
  (P) codec-pre-qp      — production recipe (matches PR #15)
  (S) codec-post-qp     — codec ON, whitening applied POST-RoPE with
                          Σ_q^post self-calibrated from THIS run's
                          own post-RoPE Q tensors (tests H1)

All cells share the same vLLM LLM instance + the same WikiText-103
passages, so the comparison is strictly paired. Runtime dominated by
the CPU-subprocess codec call; the identity cell is still expensive
because the KKTV write/read path runs.
"""
from __future__ import annotations

import argparse
import json
import os
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

sys.path.insert(0, str(REPO / "benchmarks"))
from q_precondition import QPrecond, load as qp_load  # noqa: E402


# =============================================================================
# KKTV I/O
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
        in_path, rep_path, dec_path = tdp/"x.kktv", tdp/"r.json", tdp/"d.kktv"
        write_kktv(in_path, arr.astype(np.float32, copy=False))
        cmd = [
            str(BENCH_BIN), "--input", str(in_path), "--output", str(rep_path),
            "--metric", metric, "--block-size", str(block_size),
            "--variance-ratio", str(variance_ratio),
            "--k", "16", "--bit-width", str(bit_width),
            "--rotation-seed", "3405691582",
            "--pca-method", pca_method, "--verify",
            "--dump-decoded", str(dec_path),
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
            raise RuntimeError(
                f"kakeyaturbo-bench rc={res.returncode}: "
                f"{res.stderr[:2000]}"
            )
        return read_kktv_f32(dec_path), json.loads(rep_path.read_text())


# =============================================================================
# Online self-calibrated Σ_q (post-RoPE)
# =============================================================================

class PostRopeQCalib:
    """Accumulates Σ_q_post per (layer, kv-head) from post-RoPE Q during a
    calibration forward pass, then exposes whiten / unwhiten that are
    consistent with Flash-Attention's actual K metric.

    The number of Q heads (num_heads) can be > num_kv_heads under GQA;
    we pool all Q heads in the same KV group into a single Σ_q for that
    KV head (that is how the queries this KV will actually be dotted
    against are distributed).
    """

    def __init__(self, head_dim: int, num_kv_heads: int, num_heads: int):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.group_size = num_heads // num_kv_heads
        self.accum: dict[int, np.ndarray] = {}
        self.count: dict[int, int] = {}
        self.chol: dict[int, np.ndarray] = {}
        self.inv_chol: dict[int, np.ndarray] = {}

    def accumulate(self, layer_id: int, q_post: torch.Tensor) -> None:
        """q_post: [num_tokens, num_heads * head_dim] post-RoPE.

        Reshape to [num_tokens, num_kv_heads, group_size, head_dim], pool
        the group_size axis into tokens, then accumulate Σ per kv-head.
        """
        q = q_post.detach().to(torch.float32).cpu().numpy()
        n_tok, total = q.shape
        assert total == self.num_heads * self.head_dim
        q = q.reshape(n_tok, self.num_kv_heads, self.group_size, self.head_dim)
        q = q.reshape(n_tok * self.group_size, self.num_kv_heads, self.head_dim)
        # For each kv-head, accumulate outer products.
        if layer_id not in self.accum:
            self.accum[layer_id] = np.zeros(
                (self.num_kv_heads, self.head_dim, self.head_dim),
                dtype=np.float64,
            )
            self.count[layer_id] = 0
        for h in range(self.num_kv_heads):
            qh = q[:, h, :]  # [n_tok*group_size, D]
            self.accum[layer_id][h] += qh.T @ qh
        self.count[layer_id] += q.shape[0]

    def finalize(self, ridge: float = 1e-4) -> None:
        """Factor each layer's pooled Σ_q = (1/N) Σ qq^T into L Lᵀ."""
        for l, S in self.accum.items():
            N = max(self.count.get(l, 1), 1)
            S_mean = (S / N).astype(np.float64)
            chol = np.zeros_like(S_mean, dtype=np.float32)
            inv = np.zeros_like(S_mean, dtype=np.float32)
            for h in range(self.num_kv_heads):
                M = S_mean[h]
                # Ridge-regularise for numerical stability.
                M = M + ridge * np.trace(M) / self.head_dim * np.eye(self.head_dim)
                try:
                    L = np.linalg.cholesky(M).astype(np.float32)
                except np.linalg.LinAlgError:
                    L = np.linalg.cholesky(
                        M + 1e-2 * np.trace(M) / self.head_dim
                        * np.eye(self.head_dim)
                    ).astype(np.float32)
                chol[h] = L
                inv[h] = np.linalg.inv(L).astype(np.float32)
            self.chol[l] = chol
            self.inv_chol[l] = inv

    def is_active(self, layer_id: int) -> bool:
        return layer_id in self.chol

    def whiten(self, k: np.ndarray, layer_id: int) -> np.ndarray:
        """k: [n_tok, num_kv_heads, head_dim] post-RoPE."""
        if layer_id not in self.chol:
            return k.astype(np.float32, copy=False)
        L = self.chol[layer_id]
        return np.einsum("thj,hjk->thk", k, L, optimize=True).astype(np.float32)

    def unwhiten(self, kt: np.ndarray, layer_id: int) -> np.ndarray:
        if layer_id not in self.inv_chol:
            return kt.astype(np.float32, copy=False)
        Linv = self.inv_chol[layer_id]
        return np.einsum("thj,hjk->thk", kt, Linv, optimize=True).astype(np.float32)


# =============================================================================
# Global config (read by monkey-patched forwards)
# =============================================================================

class CodecState:
    mode: str = "off"            # "off" | "identity" | "real"
    qp_mode: str = "off"         # "off" | "pre_rope" | "post_rope_self"
    qp_calibrating: bool = False  # True during the post-rope-self calibration pass
    block_size: int = 512
    bit_width_k: int = 3
    bit_width_v: int = 2
    variance_ratio: float = 0.95
    pca_method: str = "randomized"
    rsvd_target_rank_factor: float = 0.5
    k_centroids_file: str | None = None
    v_centroids_file: str | None = None
    k_outlier_threshold: float | None = None
    boundary_skip_layers: set[int] = set()
    share_basis_v: bool = True
    share_basis_k: bool = False

    qp_pre_rope: QPrecond | None = None
    qp_post_rope: PostRopeQCalib | None = None

    stats: list[dict] = []


# =============================================================================
# Codec mechanics shared by pre- and post-RoPE hooks
# =============================================================================

def _codec_or_identity(
    arr_enc: np.ndarray, *, bit_width: int, rsvd_target_rank: int,
    metric: str, share_basis: bool, centroids: str | None,
    outlier_thr: float | None,
) -> tuple[np.ndarray, dict]:
    """Run the codec, or return bit-exact data if mode=='identity'.

    Even in identity mode we still do the fp32→numpy→CPU→numpy→fp32
    path that costs most of the per-forward noise we are trying to
    measure. We just skip the `kakeyaturbo-bench` subprocess.
    """
    if CodecState.mode == "identity":
        return arr_enc.astype(np.float32, copy=False), {
            "mean_block_mse": 0.0,
            "compressed_bytes": arr_enc.nbytes,
            "identity": True,
        }

    dec, rep = rust_roundtrip(
        arr_enc, block_size=CodecState.block_size, bit_width=bit_width,
        rsvd_target_rank=rsvd_target_rank, metric=metric,
        share_basis=share_basis, pca_method=CodecState.pca_method,
        variance_ratio=CodecState.variance_ratio,
        centroids_file=centroids, outlier_threshold=outlier_thr,
    )
    return dec, rep


# =============================================================================
# Monkey-patch #1: Qwen2Attention.forward — pre-RoPE hook + Q-precond
# =============================================================================

def install_qwen2_pre_rope_patch(num_kv_heads: int, num_q_heads: int,
                                 head_dim: int) -> None:
    """Wraps Qwen2Attention.forward. Depending on CodecState.qp_mode,
    does its work either here (pre-RoPE) or defers to the
    Attention.forward patch below (post-RoPE).
    """
    from vllm.model_executor.models.qwen2 import Qwen2Attention  # type: ignore

    if getattr(Qwen2Attention, "_kk_ab_patched", False):
        return

    orig_forward = Qwen2Attention.forward

    def patched_forward(
        self: Qwen2Attention,  # type: ignore[name-defined]
        positions: torch.Tensor, hidden_states: torch.Tensor,
        kv_cache: torch.Tensor, attn_metadata: Any,
    ) -> torch.Tensor:
        if CodecState.mode == "off" and not CodecState.qp_calibrating:
            return orig_forward(self, positions, hidden_states, kv_cache,
                                attn_metadata)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        layer_id = _layer_id_from_attn_module(self.attn)

        # Pre-RoPE codec+guardrails (original production path).
        if CodecState.mode != "off" and CodecState.qp_mode == "pre_rope":
            k = _apply_codec_pre_rope(
                k, is_v=False, layer_id=layer_id, attn=self.attn,
            )
            v = _apply_codec_pre_rope(
                v, is_v=True, layer_id=layer_id, attn=self.attn,
            )
        elif CodecState.mode != "off" and CodecState.qp_mode == "off":
            # "Codec, no Q-precond" — same pre-RoPE hook, but whiten is
            # disabled via qp_mode=off, metric falls back to inner_product.
            k = _apply_codec_pre_rope(
                k, is_v=False, layer_id=layer_id, attn=self.attn,
            )
            v = _apply_codec_pre_rope(
                v, is_v=True, layer_id=layer_id, attn=self.attn,
            )

        q, k = self.rotary_emb(positions, q, k)

        # Post-RoPE path: either accumulate Σ_q_post during calibration,
        # or apply whiten→codec→unwhiten using the captured Σ_q_post.
        if CodecState.qp_calibrating:
            CodecState.qp_post_rope.accumulate(layer_id, q)
        elif CodecState.mode != "off" and CodecState.qp_mode == "post_rope_self":
            k = _apply_codec_post_rope(
                k, layer_id=layer_id, attn=self.attn, is_v=False,
            )
            v = _apply_codec_post_rope(
                v, layer_id=layer_id, attn=self.attn, is_v=True,
            )

        attn_out = self.attn(q, k, v, kv_cache, attn_metadata)
        out, _ = self.o_proj(attn_out)
        return out

    Qwen2Attention.forward = patched_forward
    Qwen2Attention._kk_ab_patched = True  # type: ignore[attr-defined]
    print("[codec-patch] Qwen2Attention.forward wrapped "
          "(pre+post-RoPE ablation hook)", flush=True)


def _layer_id_from_attn_module(attn_mod: Any) -> int:
    name = getattr(attn_mod, "layer_name", None)
    if name:
        for i, p in enumerate(name.split(".")):
            if p == "layers" and i + 1 < len(name.split(".")):
                try:
                    return int(name.split(".")[i + 1])
                except ValueError:
                    pass
    return 0


def _apply_codec_pre_rope(
    t: torch.Tensor, *, is_v: bool, layer_id: int, attn: Any,
) -> torch.Tensor:
    """Round-trip K (or V) through codec+optionally Q-precond at the
    pre-RoPE hook. Shape convention: t is `[num_tokens, kv_size]`."""
    if layer_id in CodecState.boundary_skip_layers:
        CodecState.stats.append({
            "layer": layer_id, "stream": "V" if is_v else "K",
            "skipped_boundary": True, "hook": "pre_rope",
        })
        return t

    orig_shape = t.shape
    orig_dtype = t.dtype
    orig_device = t.device
    head_size = getattr(attn, "head_size", None)
    num_kv_heads = getattr(attn, "num_kv_heads", None)
    if head_size is None or num_kv_heads is None:
        return t

    x = t.reshape(-1, num_kv_heads, head_size)
    arr = x.detach().to(torch.float32).cpu().numpy()

    qp = CodecState.qp_pre_rope if CodecState.qp_mode == "pre_rope" else None
    use_whiten = (
        (not is_v) and qp is not None
        and qp.n_kv == num_kv_heads and qp.head_dim == head_size
        and qp.is_active(layer_id)
    )
    arr_enc = qp.whiten(arr, layer=layer_id) if use_whiten else arr
    flat = arr_enc.reshape(-1, head_size).astype(np.float32, copy=False)
    n_total = flat.shape[0]
    bs = CodecState.block_size
    n_comp = (n_total // bs) * bs
    if n_comp == 0:
        return t

    bit_width = CodecState.bit_width_v if is_v else CodecState.bit_width_k
    rank = max(2, int(head_size * CodecState.rsvd_target_rank_factor))
    if is_v:
        centroids = CodecState.v_centroids_file
        outlier_thr = None
        share = CodecState.share_basis_v
        metric = "mse"
    else:
        centroids = CodecState.k_centroids_file
        outlier_thr = CodecState.k_outlier_threshold
        share = CodecState.share_basis_k
        metric = "mse" if use_whiten else "inner_product"

    dec, rep = _codec_or_identity(
        flat[:n_comp], bit_width=bit_width, rsvd_target_rank=rank,
        metric=metric, share_basis=share, centroids=centroids,
        outlier_thr=outlier_thr,
    )
    if n_comp < n_total:
        dec = np.concatenate([dec, flat[n_comp:]], axis=0)
    dec = dec.reshape(-1, num_kv_heads, head_size)
    if use_whiten:
        dec = qp.unwhiten(dec, layer=layer_id)

    CodecState.stats.append({
        "layer": layer_id, "stream": "V" if is_v else "K",
        "hook": "pre_rope", "metric": metric, "bit_width": bit_width,
        "whitened": bool(use_whiten),
        "mean_block_mse": float(rep.get("mean_block_mse", -1.0)),
        "compressed_bytes": int(rep.get("compressed_bytes", 0)),
        "identity": bool(rep.get("identity", False)),
    })

    restored = (
        torch.from_numpy(dec.astype(np.float32))
        .to(orig_device).to(orig_dtype)
        .reshape(orig_shape)
    )
    return restored


def _apply_codec_post_rope(
    t: torch.Tensor, *, is_v: bool, layer_id: int, attn: Any,
) -> torch.Tensor:
    """Round-trip K (or V) through codec+Σ_q_post whitening at the
    post-RoPE hook. Shape convention: same as pre-RoPE hook."""
    if layer_id in CodecState.boundary_skip_layers:
        CodecState.stats.append({
            "layer": layer_id, "stream": "V" if is_v else "K",
            "skipped_boundary": True, "hook": "post_rope",
        })
        return t

    orig_shape = t.shape
    orig_dtype = t.dtype
    orig_device = t.device
    head_size = getattr(attn, "head_size", None)
    num_kv_heads = getattr(attn, "num_kv_heads", None)
    if head_size is None or num_kv_heads is None:
        return t

    x = t.reshape(-1, num_kv_heads, head_size)
    arr = x.detach().to(torch.float32).cpu().numpy()

    qp_post = CodecState.qp_post_rope
    use_whiten = (
        (not is_v) and qp_post is not None and qp_post.is_active(layer_id)
    )
    arr_enc = qp_post.whiten(arr, layer_id=layer_id) if use_whiten else arr
    flat = arr_enc.reshape(-1, head_size).astype(np.float32, copy=False)
    n_total = flat.shape[0]
    bs = CodecState.block_size
    n_comp = (n_total // bs) * bs
    if n_comp == 0:
        return t

    bit_width = CodecState.bit_width_v if is_v else CodecState.bit_width_k
    rank = max(2, int(head_size * CodecState.rsvd_target_rank_factor))
    if is_v:
        centroids = CodecState.v_centroids_file
        outlier_thr = None
        share = CodecState.share_basis_v
        metric = "mse"
    else:
        centroids = CodecState.k_centroids_file
        outlier_thr = CodecState.k_outlier_threshold
        share = CodecState.share_basis_k
        metric = "mse" if use_whiten else "inner_product"

    dec, rep = _codec_or_identity(
        flat[:n_comp], bit_width=bit_width, rsvd_target_rank=rank,
        metric=metric, share_basis=share, centroids=centroids,
        outlier_thr=outlier_thr,
    )
    if n_comp < n_total:
        dec = np.concatenate([dec, flat[n_comp:]], axis=0)
    dec = dec.reshape(-1, num_kv_heads, head_size)
    if use_whiten:
        dec = qp_post.unwhiten(dec, layer_id=layer_id)

    CodecState.stats.append({
        "layer": layer_id, "stream": "V" if is_v else "K",
        "hook": "post_rope", "metric": metric, "bit_width": bit_width,
        "whitened": bool(use_whiten),
        "mean_block_mse": float(rep.get("mean_block_mse", -1.0)),
        "compressed_bytes": int(rep.get("compressed_bytes", 0)),
        "identity": bool(rep.get("identity", False)),
    })

    restored = (
        torch.from_numpy(dec.astype(np.float32))
        .to(orig_device).to(orig_dtype)
        .reshape(orig_shape)
    )
    return restored


# =============================================================================
# WikiText loader + vLLM driver
# =============================================================================

def load_wikitext_passages(tok: Any, min_tokens: int, n_passages: int,
                           split: str = "test") -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    passages, current, approx = [], [], 0
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        current.append(text)
        approx += int(len(text.split()) * 1.3)
        if approx >= min_tokens:
            passage = "".join(current)
            if len(tok.encode(passage)) >= min_tokens:
                passages.append(passage)
                if len(passages) >= n_passages:
                    return passages
            current, approx = [], 0
    return passages


def build_llm(model_path: str, max_model_len: int, gpu_mem_util: float):
    from vllm import LLM  # type: ignore
    return LLM(model=model_path, dtype="bfloat16",
               max_model_len=max_model_len,
               gpu_memory_utilization=gpu_mem_util,
               enforce_eager=True, trust_remote_code=True)


def prompt_logprobs_for_ids(llm: Any, ids: list[int]) -> list[dict]:
    from vllm import SamplingParams  # type: ignore
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    out = llm.generate(prompts=None, prompt_token_ids=[ids],
                       sampling_params=sp, use_tqdm=False)
    return out[0].prompt_logprobs


def ppl_and_top1(pls: list[dict], ids: list[int],
                 start: int, end: int) -> tuple[float, list[float], list[int]]:
    lps, top1 = [], []
    for t in range(start, end):
        entry = pls[t]
        if entry is None:
            continue
        tok = ids[t]
        if tok in entry:
            lp = entry[tok]
            lps.append(float(lp.logprob if hasattr(lp, "logprob")
                             else lp["logprob"]))
        else:
            lps.append(float("-inf"))

        def _lp(v: Any) -> float:
            return float(v.logprob if hasattr(v, "logprob") else v["logprob"])

        top1.append(int(max(entry.items(), key=lambda kv: _lp(kv[1]))[0]))
    valid = [lp for lp in lps if np.isfinite(lp)]
    mean_nll = -float(np.mean(valid)) if valid else float("inf")
    ppl = float(np.exp(mean_nll)) if np.isfinite(mean_nll) else float("inf")
    return ppl, lps, top1


def compare(ref_pls: list[dict], alt_pls: list[dict], ids: list[int],
            ctx_len: int, n_eval: int) -> dict:
    end = min(ctx_len + n_eval, len(ids))
    ppl_r, lp_r, t_r = ppl_and_top1(ref_pls, ids, ctx_len, end)
    ppl_a, lp_a, t_a = ppl_and_top1(alt_pls, ids, ctx_len, end)
    n = min(len(t_r), len(t_a))
    agree = (float(np.mean([1.0 if t_r[i] == t_a[i] else 0.0
                            for i in range(n)])) if n else float("nan"))
    deltas = [abs(lp_r[i] - lp_a[i]) for i in range(min(len(lp_r), len(lp_a)))
              if np.isfinite(lp_r[i]) and np.isfinite(lp_a[i])]
    return {
        "ppl_ref": ppl_r, "ppl_alt": ppl_a,
        "ppl_delta_rel": (ppl_a - ppl_r) / max(ppl_r, 1e-8),
        "top1_agreement": agree,
        "mean_abs_dlogp_true": float(np.mean(deltas)) if deltas
        else float("nan"),
        "n_tokens": n,
    }


def verdict_of(d: float, t: float) -> str:
    if abs(d) <= 0.01 and t >= 0.95:
        return "ACCEPT"
    if abs(d) <= 0.03 and t >= 0.85:
        return "MARGINAL"
    return "REJECT"


# =============================================================================
# Cell orchestration
# =============================================================================

CELLS = [
    # (name, mode,        qp_mode)
    ("identity-pre_qp",  "identity", "pre_rope"),
    ("codec-no_qp",      "real",     "off"),
    ("codec-pre_qp",     "real",     "pre_rope"),
    ("codec-post_qp",    "real",     "post_rope_self"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--bit-width-k", type=int, default=3)
    ap.add_argument("--bit-width-v", type=int, default=2)
    ap.add_argument("--variance-ratio", type=float, default=0.95)
    ap.add_argument("--pca-method", choices=["exact", "randomized"],
                    default="randomized")
    ap.add_argument("--rsvd-target-rank-factor", type=float, default=0.5)
    ap.add_argument("--q-calib-pre-rope", type=str, default=None)
    ap.add_argument("--k-centroids", type=str, default=None)
    ap.add_argument("--v-centroids", type=str, default=None)
    ap.add_argument("--outlier-threshold", type=float, default=None)
    ap.add_argument("--boundary-skip-layers", type=int, nargs="*",
                    default=[0, 1, 7, 14, 26, 27])
    ap.add_argument("--share-basis-v", action="store_true", default=True)
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.80)
    ap.add_argument("--post-rope-qp-calib-passages", type=int, default=2)
    ap.add_argument("--cells", nargs="+",
                    default=[c[0] for c in CELLS],
                    help="Subset of cell names to run (space-separated)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Populate global codec state.
    CodecState.block_size = args.block_size
    CodecState.bit_width_k = args.bit_width_k
    CodecState.bit_width_v = args.bit_width_v
    CodecState.variance_ratio = args.variance_ratio
    CodecState.pca_method = args.pca_method
    CodecState.rsvd_target_rank_factor = args.rsvd_target_rank_factor
    CodecState.k_centroids_file = args.k_centroids
    CodecState.v_centroids_file = args.v_centroids
    CodecState.k_outlier_threshold = args.outlier_threshold
    CodecState.boundary_skip_layers = set(args.boundary_skip_layers or [])
    CodecState.share_basis_v = args.share_basis_v

    if args.q_calib_pre_rope:
        print(f"[setup] pre-RoPE Σ_q  : {args.q_calib_pre_rope}", flush=True)
        CodecState.qp_pre_rope = qp_load(args.q_calib_pre_rope,
                                         skip_layers=[0])
        print(f"        calibrated {CodecState.qp_pre_rope.n_calibrated_layers}"
              f"/{CodecState.qp_pre_rope.n_layers} layers", flush=True)

    # We need a throw-away forward to get num_heads / num_kv_heads /
    # head_dim before we can size PostRopeQCalib. The cleanest way is to
    # install the patch first (patch merely reads CodecState.mode), then
    # introspect the model after LLM is built.

    print(f"[{args.model_name}] loading vLLM engine…", flush=True)
    # We need num_q / num_kv / head_dim, install the patch first at
    # "off" mode, then peek after the LLM is constructed.
    # We install with dummy sizes that get overwritten at calibration time.
    install_qwen2_pre_rope_patch(
        num_kv_heads=1, num_q_heads=1, head_dim=1,
    )
    llm = build_llm(args.model_path,
                    args.ctx_len + args.n_eval + 16, args.gpu_mem_util)
    tok = llm.get_tokenizer()

    # Introspect one Qwen2Attention module to size PostRopeQCalib.
    from vllm.model_executor.models.qwen2 import Qwen2Attention  # type: ignore
    qatt = None
    for m in llm.llm_engine.model_executor.driver_worker.model_runner.model.modules():
        if isinstance(m, Qwen2Attention):
            qatt = m
            break
    assert qatt is not None, "no Qwen2Attention instance found"
    head_size = qatt.attn.head_size
    num_kv = qatt.attn.num_kv_heads
    num_q = qatt.attn.num_heads
    print(f"[setup] head_size={head_size} num_kv={num_kv} num_q={num_q}",
          flush=True)

    # Load passages.
    print(f"[{args.model_name}] loading WikiText-103 passages…", flush=True)
    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval,
        n_passages=args.n_passages,
    )
    print(f"  got {len(passages)} passages", flush=True)
    passage_ids = []
    for p in passages:
        ids = tok.encode(p)[: args.ctx_len + args.n_eval]
        if len(ids) >= args.ctx_len + args.n_eval:
            passage_ids.append(ids)
    print(f"  usable: {len(passage_ids)}", flush=True)

    # Cell O — reference logprobs for every passage (codec off).
    CodecState.mode = "off"
    CodecState.qp_mode = "off"
    CodecState.qp_calibrating = False
    ref_pls: list[list[dict]] = []
    t_ref_total = 0.0
    for i, ids in enumerate(passage_ids):
        print(f"  [ref] passage {i + 1}/{len(passage_ids)}…", flush=True)
        t0 = time.perf_counter()
        ref_pls.append(prompt_logprobs_for_ids(llm, ids))
        t_ref_total += time.perf_counter() - t0

    results: dict[str, Any] = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "engine": "vllm",
        "ctx_len": args.ctx_len,
        "n_eval": args.n_eval,
        "block_size": args.block_size,
        "bit_width_k": args.bit_width_k,
        "bit_width_v": args.bit_width_v,
        "pca_method": args.pca_method,
        "rsvd_target_rank_factor": args.rsvd_target_rank_factor,
        "k_centroids": args.k_centroids,
        "v_centroids": args.v_centroids,
        "outlier_threshold": args.outlier_threshold,
        "boundary_skip_layers": sorted(CodecState.boundary_skip_layers),
        "n_passages": len(passage_ids),
        "t_ref_total_sec": t_ref_total,
        "cells": {},
    }

    selected = [c for c in CELLS if c[0] in set(args.cells)]
    print(f"\nRunning cells: {[c[0] for c in selected]}", flush=True)

    for cell_name, mode, qp_mode in selected:
        print(f"\n================ CELL: {cell_name} "
              f"(mode={mode}, qp_mode={qp_mode}) ================",
              flush=True)

        # If post-RoPE self-calibration is requested, do a calibration
        # pass first to populate Σ_q_post (codec stays off during it).
        if qp_mode == "post_rope_self":
            print("  [calib] accumulating Σ_q_post over "
                  f"{args.post_rope_qp_calib_passages} passage(s)…",
                  flush=True)
            CodecState.qp_post_rope = PostRopeQCalib(
                head_dim=head_size, num_kv_heads=num_kv,
                num_heads=num_q,
            )
            CodecState.mode = "off"
            CodecState.qp_mode = "off"
            CodecState.qp_calibrating = True
            n_calib = min(args.post_rope_qp_calib_passages, len(passage_ids))
            for i in range(n_calib):
                _ = prompt_logprobs_for_ids(llm, passage_ids[i])
            CodecState.qp_calibrating = False
            t0 = time.perf_counter()
            CodecState.qp_post_rope.finalize(ridge=1e-4)
            n_accum = len(CodecState.qp_post_rope.accum)
            print(f"  [calib] factored {n_accum} layers "
                  f"(finalize {time.perf_counter() - t0:.2f}s)",
                  flush=True)

        CodecState.mode = mode
        CodecState.qp_mode = qp_mode

        per_passage = []
        for i, ids in enumerate(passage_ids):
            CodecState.stats = []
            t0 = time.perf_counter()
            alt_pls = prompt_logprobs_for_ids(llm, ids)
            t_alt = time.perf_counter() - t0
            stats_this = list(CodecState.stats)

            m = compare(ref_pls[i], alt_pls, ids, args.ctx_len, args.n_eval)
            print(
                f"  [{cell_name}] p{i+1}/{len(passage_ids)}: "
                f"ppl_ref={m['ppl_ref']:.3f} ppl_alt={m['ppl_alt']:.3f} "
                f"Δppl={m['ppl_delta_rel']*100:+.3f}% "
                f"top1={m['top1_agreement']*100:.2f}% "
                f"Δlogp={m['mean_abs_dlogp_true']:.4f} "
                f"(alt={t_alt:.2f}s calls={len(stats_this)})",
                flush=True,
            )
            per_passage.append({
                "ctx_len": args.ctx_len, "n_eval": args.n_eval,
                "t_alt_sec": t_alt,
                "codec_layer_calls": len(stats_this),
                "boundary_skips": int(sum(
                    1 for s in stats_this if s.get("skipped_boundary")
                )),
                "metrics": m,
            })

        valid = [r for r in per_passage
                 if np.isfinite(r["metrics"]["ppl_delta_rel"])]
        mean_delta = (float(np.mean([r["metrics"]["ppl_delta_rel"] for r in valid]))
                      if valid else float("nan"))
        mean_top1 = (float(np.mean([r["metrics"]["top1_agreement"] for r in valid]))
                     if valid else float("nan"))
        v = verdict_of(mean_delta, mean_top1)
        print(f"\n  [{cell_name}] SUMMARY: "
              f"Δppl={mean_delta*100:+.3f}% top1={mean_top1*100:.2f}% "
              f"→ {v}", flush=True)

        results["cells"][cell_name] = {
            "mode": mode, "qp_mode": qp_mode,
            "mean_ppl_delta_rel": mean_delta,
            "mean_top1_agreement": mean_top1,
            "verdict": v,
            "per_passage": per_passage,
        }

    # Print a small table.
    print("\n================ ALL CELLS ================", flush=True)
    print(f"  {'cell':<20} {'Δppl':>10}  {'top-1':>7}  verdict", flush=True)
    for k, v in results["cells"].items():
        print(f"  {k:<20} {v['mean_ppl_delta_rel']*100:+10.3f}% "
              f"{v['mean_top1_agreement']*100:6.2f}%  {v['verdict']}",
              flush=True)

    out_path = args.out_dir / f"{args.model_name}_vllm_ablation.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
