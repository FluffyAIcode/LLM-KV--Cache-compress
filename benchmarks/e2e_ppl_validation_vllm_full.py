#!/usr/bin/env python3
"""End-to-end PPL validation of the FULL 'v1.3 PPL' production recipe
(= v1.3 RSVD + four PPL-stabilization guardrails) running on vLLM.

Guardrails are applied, per `reports/SPRINT_CLOSEOUT.md`:

  1. Q-preconditioning        K_tilde = K @ L   (pre-RoPE, per (layer, kv-head))
  2. Calibrated Lloyd-Max     K residual codebook (and V at b=2)
  3. 6-layer boundary skip    layers [0, 1, 7, 14, 26, 27] kept bf16
  4. Outlier compensation     T = 2.0 on K residual coords (~4.5% → f16)

Unlike the scaffolding harness (`e2e_ppl_validation_vllm.py`) that
patches `vllm.attention.layer.Attention.forward` — i.e. sees K/V AFTER
RoPE — this harness patches the model's own `Qwen2Attention.forward`
so we can touch K BEFORE RoPE, apply whitening, round-trip through
the codec, un-whiten, and let RoPE + attention run normally on the
repaired K. That is the mathematically correct place to do
Q-preconditioning (L is calibrated on pre-RoPE distributions).

V is round-tripped (without whitening) at the same hook point.

Usage
-----

  python benchmarks/e2e_ppl_validation_vllm_full.py \\
      --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
      --model-name ds_distill_qwen_1_5b \\
      --q-calib reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors \\
      --k-centroids reports/v1_4_q_pca/calibrated_codebook/ds_K_b3_centroids.f32 \\
      --v-centroids reports/v1_4_q_pca/calibrated_codebook/ds_V_b2_centroids.f32 \\
      --bit-width-k 3 --bit-width-v 2 \\
      --outlier-threshold 2.0 \\
      --boundary-skip-layers 0 1 7 14 26 27 \\
      --ctx-len 2048 --n-eval 64 --n-passages 4 \\
      --out-dir reports/v1_3_ppl/vllm/

The default flags land the SPRINT_CLOSEOUT production cell:
    K b=3, V b=2, share-basis-v, T=2.0, 6 bdry on DS-Distill.
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
    assert arr.dtype == np.float32 and arr.ndim == 2, (arr.dtype, arr.shape)
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
        _ = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        d = struct.unpack("<I", f.read(4))[0]
        _ = struct.unpack("<I", f.read(4))[0]
        raw = f.read(n * d * 4)
    return np.frombuffer(raw, dtype=np.float32).reshape(n, d).copy()


# =============================================================================
# Rust codec round-trip
# =============================================================================

def rust_roundtrip(
    arr: np.ndarray,
    *,
    block_size: int,
    bit_width: int,
    rsvd_target_rank: int,
    metric: str,
    share_basis: bool,
    pca_method: str = "randomized",
    variance_ratio: float = 0.95,
    centroids_file: str | None = None,
    outlier_threshold: float | None = None,
) -> tuple[np.ndarray, dict]:
    if not BENCH_BIN.exists():
        raise FileNotFoundError(
            f"{BENCH_BIN} missing; build with "
            "`cargo build --release --bin kakeyaturbo-bench` in kakeyaturbo/"
        )

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
            "--k", "16", "--bit-width", str(bit_width),
            "--rotation-seed", "3405691582",
            "--pca-method", pca_method,
            "--verify",
            "--dump-decoded", str(dec_path),
        ]
        if pca_method == "randomized":
            cmd += [
                "--rsvd-target-rank", str(rsvd_target_rank),
                "--rsvd-oversample", "8",
                "--rsvd-power-iters", "2",
            ]
        if share_basis:
            cmd.append("--share-basis")
        if centroids_file is not None:
            cmd += ["--centroids-file", str(centroids_file)]
        if outlier_threshold is not None:
            cmd += ["--outlier-threshold", str(outlier_threshold)]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"kakeyaturbo-bench failed (rc={res.returncode}): "
                f"{res.stderr[:2000]}"
            )
        report = json.loads(rep_path.read_text())
        decoded = read_kktv_f32(dec_path)
        return decoded, report


# =============================================================================
# Global codec config (read by the monkey-patched Qwen2Attention.forward)
# =============================================================================

class CodecState:
    active: bool = False
    block_size: int = 512
    bit_width_k: int = 3
    bit_width_v: int = 2
    variance_ratio: float = 0.95
    pca_method: str = "randomized"
    rsvd_target_rank_factor: float = 0.5
    k_centroids_file: str | None = None
    v_centroids_file: str | None = None
    k_outlier_threshold: float | None = None
    v_outlier_threshold: float | None = None
    boundary_skip_layers: set[int] = set()
    q_precond: QPrecond | None = None
    share_basis_k: bool = False
    share_basis_v: bool = True
    # Stream selector. "kv" = compress both (production).
    # "k"  = compress only K, V stays bf16 (diagnose K-only PPL cost).
    # "v"  = compress only V, K stays bf16 (diagnose V-only PPL cost).
    compress_stream: str = "kv"
    stats: list[dict] = []


# =============================================================================
# Monkey-patch vLLM's Qwen2Attention.forward (pre-RoPE hook point)
# =============================================================================

def install_qwen2_pre_rope_patch() -> None:
    """Patch vllm.model_executor.models.qwen2.Qwen2Attention.forward.

    Replaces the stock forward with one that, when CodecState.active is
    True, inserts:
        K_tilde = whiten(K)        (per-layer, per-kv-head)
        K_hat_tilde = codec_roundtrip(K_tilde)
        K_hat = unwhiten(K_hat_tilde)
        V_hat = codec_roundtrip(V)
    immediately after the QKV projection (so BEFORE RoPE), then lets the
    rest of the stock forward run. This is the same hook point as the
    HF pre-RoPE harness in PR #13.
    """
    from vllm.model_executor.models.qwen2 import Qwen2Attention  # type: ignore

    if getattr(Qwen2Attention, "_kk_full_patched", False):
        return

    orig_forward = Qwen2Attention.forward

    def patched_forward(
        self: Qwen2Attention,  # type: ignore[name-defined]
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
    ) -> torch.Tensor:
        if not CodecState.active:
            return orig_forward(
                self, positions, hidden_states, kv_cache, attn_metadata
            )

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        layer_id = _layer_id_from_attn_module(self.attn)
        k = _apply_k_guardrails(k, v_ref=False, layer_id=layer_id, attn=self.attn)
        v = _apply_k_guardrails(v, v_ref=True, layer_id=layer_id, attn=self.attn)
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v, kv_cache, attn_metadata)
        out, _ = self.o_proj(attn_out)
        return out

    Qwen2Attention.forward = patched_forward
    Qwen2Attention._kk_full_patched = True  # type: ignore[attr-defined]
    print("[codec-patch] vllm Qwen2Attention.forward wrapped "
          "(pre-RoPE hook)", flush=True)


def _layer_id_from_attn_module(attn_mod: Any) -> int:
    """Map a vLLM Attention module back to an integer layer index.

    vLLM assigns `self.layer_name = "model.layers.{L}.self_attn.attn"`.
    We parse the "{L}" field.
    """
    name = getattr(attn_mod, "layer_name", None)
    if name is None:
        # Fall back to a monotonic counter the first time we see each
        # instance.
        if not hasattr(attn_mod, "_kk_layer_counter"):
            CodecState.stats.append({"warn": "no-layer-name"})
        cnt = getattr(attn_mod, "_kk_layer_counter", None)
        if cnt is None:
            cnt = len([s for s in CodecState.stats
                       if s.get("_layer_counter_assignment")])
            object.__setattr__(attn_mod, "_kk_layer_counter", cnt)
            CodecState.stats.append({"_layer_counter_assignment": True})
        return cnt
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return 0


def _apply_k_guardrails(
    t: torch.Tensor, *, v_ref: bool, layer_id: int, attn: Any,
) -> torch.Tensor:
    """Round-trip one K or V tensor through the v1.3 PPL pipeline.

    `t` is post-QKV-projection, pre-RoPE: shape `[num_tokens,
    kv_size]` where `kv_size = num_kv_heads * head_size`. We reshape to
    `[num_tokens, num_kv_heads, head_size]`, round-trip, reshape back.
    """
    # Boundary-skip layers stay fully bf16.
    if layer_id in CodecState.boundary_skip_layers:
        CodecState.stats.append({
            "layer": layer_id, "kind": "V" if v_ref else "K",
            "skipped_boundary": True,
        })
        return t

    orig_shape = t.shape
    orig_dtype = t.dtype
    orig_device = t.device

    head_size = getattr(attn, "head_size", None)
    num_kv_heads = getattr(attn, "num_kv_heads", None)
    if head_size is None or num_kv_heads is None:
        return t

    # Stream gating: if this stream is not selected, pass through.
    stream_on = (
        (not v_ref and "k" in CodecState.compress_stream)
        or (v_ref and "v" in CodecState.compress_stream)
    )
    if not stream_on:
        CodecState.stats.append({
            "layer": layer_id, "kind": "V" if v_ref else "K",
            "stream_off": True,
        })
        return t

    x = t.reshape(-1, num_kv_heads, head_size)  # [tokens, n_kv, head_size]
    arr = x.detach().to(torch.float32).cpu().numpy()
    n_tokens, n_kv, hd = arr.shape

    # Q-preconditioning: whiten only K (v_ref is False) and only when
    # a calibrated Cholesky is present for this layer.
    qp = CodecState.q_precond
    use_whiten = (
        (not v_ref) and qp is not None
        and qp.n_kv == n_kv and qp.head_dim == hd
        and qp.is_active(layer_id)
    )
    if use_whiten:
        arr_enc = qp.whiten(arr, layer=layer_id)
    else:
        arr_enc = arr

    flat = arr_enc.reshape(-1, hd).astype(np.float32, copy=False)
    n_total = flat.shape[0]
    bs = CodecState.block_size
    n_comp = (n_total // bs) * bs
    if n_comp == 0:
        return t  # not enough vectors to fill one block

    bit_width = CodecState.bit_width_v if v_ref else CodecState.bit_width_k
    target_rank = max(2, int(hd * CodecState.rsvd_target_rank_factor))
    if v_ref:
        centroids = CodecState.v_centroids_file
        outlier_thr = CodecState.v_outlier_threshold
        share = CodecState.share_basis_v
        metric = "mse"
    else:
        centroids = CodecState.k_centroids_file
        outlier_thr = CodecState.k_outlier_threshold
        share = CodecState.share_basis_k
        # With Q-precond, the codec's MSE on K_tilde is the proper
        # proxy for the true \Sigma_q-weighted K distortion. If we
        # didn't whiten, we'd fall back to inner_product metric like
        # the scaffolding harness does.
        metric = "mse" if use_whiten else "inner_product"

    dec, rep = rust_roundtrip(
        flat[:n_comp],
        block_size=bs, bit_width=bit_width,
        rsvd_target_rank=target_rank,
        metric=metric, share_basis=share,
        pca_method=CodecState.pca_method,
        variance_ratio=CodecState.variance_ratio,
        centroids_file=centroids,
        outlier_threshold=outlier_thr,
    )

    # Stitch tail (vectors past the last full block) back in.
    if n_comp < n_total:
        dec = np.concatenate([dec, flat[n_comp:]], axis=0)

    dec = dec.reshape(n_tokens, n_kv, hd)
    if use_whiten:
        dec = qp.unwhiten(dec, layer=layer_id)

    restored = (
        torch.from_numpy(dec.astype(np.float32))
        .to(orig_device).to(orig_dtype)
        .reshape(orig_shape)
    )
    CodecState.stats.append({
        "layer": layer_id, "kind": "V" if v_ref else "K",
        "metric": metric, "whitened": bool(use_whiten),
        "bit_width": bit_width,
        "n_compressible": int(n_comp),
        "n_tail": int(n_total - n_comp),
        "mean_block_mse": float(rep.get("mean_block_mse", -1.0)),
        "compressed_bytes": int(rep.get("compressed_bytes", 0)),
        "outlier_threshold": outlier_thr,
        "centroids_file": centroids,
    })
    return restored


# =============================================================================
# WikiText-103 loader
# =============================================================================

def load_wikitext_passages(
    tokenizer: Any, min_tokens: int, n_passages: int, split: str = "test",
) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    passages: list[str] = []
    current: list[str] = []
    approx = 0
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        current.append(text)
        approx += int(len(text.split()) * 1.3)
        if approx >= min_tokens:
            passage = "".join(current)
            if len(tokenizer.encode(passage)) >= min_tokens:
                passages.append(passage)
                if len(passages) >= n_passages:
                    return passages
            current = []
            approx = 0
    return passages


# =============================================================================
# vLLM engine build + PPL measurement
# =============================================================================

def build_llm(model_path: str, max_model_len: int, gpu_mem_util: float):
    from vllm import LLM  # type: ignore
    return LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True,
        trust_remote_code=True,
    )


def prompt_logprobs_for_ids(llm: Any, ids: list[int]) -> list[dict]:
    from vllm import SamplingParams  # type: ignore
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    out = llm.generate(
        prompts=None,
        prompt_token_ids=[ids],
        sampling_params=sp,
        use_tqdm=False,
    )
    return out[0].prompt_logprobs


def ppl_and_top1(
    pls: list[dict], ids: list[int], start: int, end: int,
) -> tuple[float, list[float], list[int]]:
    lps: list[float] = []
    top1: list[int] = []
    for t in range(start, end):
        entry = pls[t]
        if entry is None:
            continue
        tok = ids[t]
        if tok in entry:
            lp = entry[tok]
            lps.append(float(lp.logprob if hasattr(lp, "logprob") else lp["logprob"]))
        else:
            lps.append(float("-inf"))

        def _lp(v: Any) -> float:
            return float(v.logprob if hasattr(v, "logprob") else v["logprob"])

        top1.append(int(max(entry.items(), key=lambda kv: _lp(kv[1]))[0]))
    valid = [lp for lp in lps if np.isfinite(lp)]
    mean_nll = -float(np.mean(valid)) if valid else float("inf")
    ppl = float(np.exp(mean_nll)) if np.isfinite(mean_nll) else float("inf")
    return ppl, lps, top1


def compare(
    ref_pls: list[dict], alt_pls: list[dict],
    ids: list[int], ctx_len: int, n_eval: int,
) -> dict:
    end = min(ctx_len + n_eval, len(ids))
    ppl_r, lp_r, t_r = ppl_and_top1(ref_pls, ids, ctx_len, end)
    ppl_a, lp_a, t_a = ppl_and_top1(alt_pls, ids, ctx_len, end)
    n = min(len(t_r), len(t_a))
    agree = (
        float(np.mean([1.0 if t_r[i] == t_a[i] else 0.0 for i in range(n)]))
        if n else float("nan")
    )
    deltas = [
        abs(lp_r[i] - lp_a[i])
        for i in range(min(len(lp_r), len(lp_a)))
        if np.isfinite(lp_r[i]) and np.isfinite(lp_a[i])
    ]
    return {
        "ppl_ref": ppl_r,
        "ppl_alt": ppl_a,
        "ppl_delta_rel": (ppl_a - ppl_r) / max(ppl_r, 1e-8),
        "top1_agreement": agree,
        "mean_abs_dlogp_true": float(np.mean(deltas)) if deltas else float("nan"),
        "n_tokens": n,
    }


def verdict_of(delta: float, top1: float) -> str:
    if abs(delta) <= 0.01 and top1 >= 0.95:
        return "ACCEPT"
    if abs(delta) <= 0.03 and top1 >= 0.85:
        return "MARGINAL"
    return "REJECT"


# =============================================================================
# Main
# =============================================================================

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
    ap.add_argument("--q-calib", type=str, default=None,
                    help="Path to Σ_q Cholesky safetensors "
                         "(set None to disable Q-preconditioning)")
    ap.add_argument("--k-centroids", type=str, default=None)
    ap.add_argument("--v-centroids", type=str, default=None)
    ap.add_argument("--outlier-threshold", type=float, default=None,
                    help="K residual outlier T (e.g. 2.0 for v1.3 PPL)")
    ap.add_argument("--v-outlier-threshold", type=float, default=None,
                    help="V residual outlier T. Unset by default (V has no "
                         "outlier compensation in SPRINT_CLOSEOUT v1.3 PPL); "
                         "enables symmetric outlier compensation on V when "
                         "set (e.g. 2.0).")
    ap.add_argument("--boundary-skip-layers", type=int, nargs="*",
                    default=[0, 1, 7, 14, 26, 27],
                    help="Layer indices kept at full precision (bf16)")
    ap.add_argument("--compress-stream", choices=["kv", "k", "v"],
                    default="kv",
                    help="Which streams go through the codec. 'kv' is the "
                         "production config; 'k' / 'v' run one stream through "
                         "the codec and leave the other pass-through (bf16) "
                         "for per-channel PPL attribution.")
    ap.add_argument("--share-basis-v", action="store_true", default=True)
    ap.add_argument("--no-share-basis-v", dest="share_basis_v",
                    action="store_false")
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.80)
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
    CodecState.v_outlier_threshold = args.v_outlier_threshold
    CodecState.boundary_skip_layers = set(args.boundary_skip_layers or [])
    CodecState.share_basis_v = args.share_basis_v
    CodecState.compress_stream = args.compress_stream
    if args.compress_stream != "kv":
        print(f"[setup] compress_stream={args.compress_stream}: "
              f"{'K' if args.compress_stream == 'k' else 'V'} through codec, "
              f"{'V' if args.compress_stream == 'k' else 'K'} stays bf16",
              flush=True)
    if args.q_calib:
        print(f"[setup] loading Q-preconditioner from {args.q_calib}",
              flush=True)
        CodecState.q_precond = qp_load(args.q_calib, skip_layers=[0])
        print(f"        calibrated layers: "
              f"{CodecState.q_precond.n_calibrated_layers}/"
              f"{CodecState.q_precond.n_layers} "
              f"(n_kv={CodecState.q_precond.n_kv}, D={CodecState.q_precond.head_dim})",
              flush=True)

    # Install patch BEFORE LLM is constructed.
    install_qwen2_pre_rope_patch()

    print(f"[{args.model_name}] loading vLLM engine…", flush=True)
    max_len = args.ctx_len + args.n_eval + 16
    llm = build_llm(args.model_path, max_len, args.gpu_mem_util)

    tok = llm.get_tokenizer()
    print(f"[{args.model_name}] loading WikiText-103 passages…", flush=True)
    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval, n_passages=args.n_passages,
    )
    print(f"  got {len(passages)} passages", flush=True)

    per_passage: list[dict] = []
    for i, p in enumerate(passages):
        print(f"  passage {i + 1}/{len(passages)}…", flush=True)
        ids = tok.encode(p)[: args.ctx_len + args.n_eval]
        if len(ids) < args.ctx_len + args.n_eval:
            print("    skipped (short)", flush=True)
            continue

        CodecState.active = False
        t0 = time.perf_counter()
        ref_pls = prompt_logprobs_for_ids(llm, ids)
        t_ref = time.perf_counter() - t0

        CodecState.active = True
        CodecState.stats = []
        t0 = time.perf_counter()
        alt_pls = prompt_logprobs_for_ids(llm, ids)
        t_alt = time.perf_counter() - t0
        stats_this = list(CodecState.stats)
        CodecState.active = False

        m = compare(ref_pls, alt_pls, ids, args.ctx_len, args.n_eval)
        print(
            f"    ppl_ref={m['ppl_ref']:.3f} ppl_alt={m['ppl_alt']:.3f} "
            f"Δppl={m['ppl_delta_rel']*100:+.3f}% "
            f"top1={m['top1_agreement']*100:.2f}% "
            f"Δlogp={m['mean_abs_dlogp_true']:.4f} "
            f"(ref={t_ref:.2f}s alt={t_alt:.2f}s calls={len(stats_this)})",
            flush=True,
        )
        per_passage.append({
            "ctx_len": args.ctx_len,
            "n_eval": args.n_eval,
            "t_ref_sec": t_ref,
            "t_alt_sec": t_alt,
            "codec_layer_calls": len(stats_this),
            "codec_total_compressed_bytes": int(
                sum(s.get("compressed_bytes", 0) for s in stats_this)
            ),
            "boundary_skips": int(
                sum(1 for s in stats_this if s.get("skipped_boundary"))
            ),
            "metrics": m,
        })

    summary: dict = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "engine": "vllm",
        "recipe": "v1.3 PPL full guardrails",
        "ctx_len": args.ctx_len,
        "n_eval": args.n_eval,
        "block_size": args.block_size,
        "bit_width_k": args.bit_width_k,
        "bit_width_v": args.bit_width_v,
        "variance_ratio": args.variance_ratio,
        "pca_method": args.pca_method,
        "rsvd_target_rank_factor": args.rsvd_target_rank_factor,
        "q_calib": args.q_calib,
        "k_centroids": args.k_centroids,
        "v_centroids": args.v_centroids,
        "outlier_threshold": args.outlier_threshold,
        "v_outlier_threshold": args.v_outlier_threshold,
        "boundary_skip_layers": sorted(CodecState.boundary_skip_layers),
        "share_basis_v": args.share_basis_v,
        "n_passages": len(per_passage),
    }
    if per_passage:
        valid = [r for r in per_passage
                 if np.isfinite(r["metrics"]["ppl_delta_rel"])]
        mean_delta = (float(np.mean([r["metrics"]["ppl_delta_rel"] for r in valid]))
                      if valid else float("nan"))
        mean_top1 = (float(np.mean([r["metrics"]["top1_agreement"] for r in valid]))
                     if valid else float("nan"))
        summary.update({
            "mean_ppl_delta_rel": mean_delta,
            "mean_top1_agreement": mean_top1,
            "verdict": verdict_of(mean_delta, mean_top1),
        })
        print(f"\n[{args.model_name}] ===== SUMMARY (vLLM v1.3 PPL) =====",
              flush=True)
        print(f"  n_passages   = {len(per_passage)}", flush=True)
        print(f"  Δppl (mean)  = {mean_delta*100:+.3f}%", flush=True)
        print(f"  top1 agree   = {mean_top1*100:.2f}%", flush=True)
        print(f"  VERDICT      = {summary['verdict']}", flush=True)
    else:
        summary["verdict"] = "NO_DATA"

    summary["per_passage"] = per_passage
    out_path = args.out_dir / f"{args.model_name}_vllm_full.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
