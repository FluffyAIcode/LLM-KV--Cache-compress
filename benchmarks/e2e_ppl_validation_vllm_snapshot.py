#!/usr/bin/env python3
"""Snapshot-mode vLLM harness: HF two-pass semantics inside vLLM.

Goal
----
Phase 6 (PR #16) localised the HF (+7.82 %) \u2194 vLLM (+35.33 %) \u0394ppl
gap to a +39 pp "cross-layer non-linear compounding" term that came
from the vLLM harness applying the codec INSIDE the forward graph at
every layer \u2014 so layer l+1's K/V projection is computed from a
residual already shifted by layer l's codec. HF's harness avoids
that by running the codec on a CLEAN prefill snapshot in the
DynamicCache and then teacher-forcing the eval tokens against that
cache. This script reproduces HF's semantics in vLLM:

  Step 1: clean forward through vLLM (codec OFF). Capture per-layer
          pre-RoPE K / V snapshots for all positions.
  Step 2: offline \u2014 run the production v1.3 codec on each snapshot
          (Q-precond on K + Lloyd-Max + outlier + boundary skip;
          Lloyd-Max + share_basis on V).
  Step 3: second forward through vLLM. Hook the Qwen2Attention.forward
          so that instead of projecting K/V from the current (maybe
          codec-shifted) residual, we FORCE the layer to use the
          PRE-COMPUTED codec'd snapshot from Step 2. This kills the
          in-forward cross-layer pollution path. Q still comes from
          the running residual, matching HF's teacher-force flow.

If this run measures \u0394ppl \u2248 +8 %, the entire +39 pp compounding is
harness-integration (snapshot-vs-inline), and "deploy codec as a
post-prefill cache compressor" is the honest production number on
vLLM too. If it's still materially > +8 %, there IS an intrinsic
engine component left.
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

sys.path.insert(0, str(REPO / "benchmarks"))
from q_precondition import QPrecond, load as qp_load  # noqa: E402


# =============================================================================
# KKTV I/O + rust codec (reused)
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
# Offline codec of a per-layer K (or V) snapshot, returning the decoded
# tensor of identical shape. Applies the same recipe as the production
# cell does inside its forward hook.
# =============================================================================

def codec_layer(
    K_or_V: np.ndarray, *, is_v: bool, layer_id: int,
    q_precond: QPrecond | None,
    block_size: int, bit_width_k: int, bit_width_v: int,
    k_centroids: str | None, v_centroids: str | None,
    k_outlier_threshold: float | None, v_outlier_threshold: float | None,
    boundary_skip: set[int], rsvd_target_rank_factor: float = 0.5,
    share_basis_v: bool = True, share_basis_k: bool = False,
) -> tuple[np.ndarray, dict]:
    """Same logic as e2e_ppl_validation_vllm_full._apply_k_guardrails but
    pure offline on a fp32 numpy array of shape [n_tokens, num_kv_heads,
    head_size]."""
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
    flat = arr_enc.reshape(-1, hd).astype(np.float32, copy=False)
    n_total = flat.shape[0]
    n_comp = (n_total // block_size) * block_size
    if n_comp == 0:
        return K_or_V.astype(np.float32, copy=False), {
            "layer": layer_id, "stream": "V" if is_v else "K",
            "skipped_short": True,
        }

    if is_v:
        bit_width = bit_width_v
        centroids = v_centroids
        outlier_thr = v_outlier_threshold
        share = share_basis_v
        metric = "mse"
    else:
        bit_width = bit_width_k
        centroids = k_centroids
        outlier_thr = k_outlier_threshold
        share = share_basis_k
        metric = "mse" if use_whiten else "inner_product"

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

    return dec.astype(np.float32, copy=False), {
        "layer": layer_id, "stream": "V" if is_v else "K",
        "n_tokens": int(n), "n_compressible": int(n_comp),
        "mean_block_mse": float(rep.get("mean_block_mse", -1.0)),
        "compressed_bytes": int(rep.get("compressed_bytes", 0)),
        "whitened": bool(use_whiten),
        "metric": metric, "bit_width": bit_width,
    }


# =============================================================================
# vLLM hook — snapshot-mode replacement
# =============================================================================

class HookState:
    """Module-level state for the Qwen2Attention hook.

    Three phases that the hook distinguishes:

      phase == "capture"   record per-layer pre-RoPE K, V for all
                           num_tokens of the current prompt; call the
                           original forward unchanged so the forward is
                           a true clean pass.
      phase == "replace"   ignore the live pre-RoPE K, V projections
                           and use the pre-codec'd tensor from
                           `replacements[layer_id]` instead (same n
                           tokens, same shape).
      phase == "off"       no-op: the hook is equivalent to the stock
                           Qwen2Attention.forward.
    """
    phase: str = "off"
    captured: dict[int, dict[str, np.ndarray]] = {}
    replacements: dict[int, dict[str, torch.Tensor]] = {}  # fp32 GPU tensors
    head_size: int = 0
    num_kv_heads: int = 0
    num_heads: int = 0


def install_qwen2_snapshot_patch() -> None:
    from vllm.model_executor.models.qwen2 import Qwen2Attention  # type: ignore
    if getattr(Qwen2Attention, "_kk_snapshot_patched", False):
        return
    orig = Qwen2Attention.forward

    def patched(self, positions, hidden_states, kv_cache, attn_metadata):  # type: ignore[no-untyped-def]
        if HookState.phase == "off":
            return orig(self, positions, hidden_states, kv_cache, attn_metadata)
        # Reimplement the parts of Qwen2Attention.forward we need.
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        # Parse layer id
        layer_id = 0
        name = getattr(self.attn, "layer_name", None)
        if name:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_id = int(parts[i + 1])
                    except ValueError:
                        pass
        HookState.head_size = self.attn.head_size
        HookState.num_kv_heads = self.attn.num_kv_heads
        HookState.num_heads = self.attn.num_heads

        nkv = self.attn.num_kv_heads
        hd = self.attn.head_size

        if HookState.phase == "capture":
            # Record pre-RoPE K, V (and shape) to numpy fp32.
            k_np = (k.detach().to(torch.float32).cpu().numpy()
                     .reshape(-1, nkv, hd))
            v_np = (v.detach().to(torch.float32).cpu().numpy()
                     .reshape(-1, nkv, hd))
            HookState.captured[layer_id] = {"K": k_np, "V": v_np}
            # Fall through to the normal forward with untouched k, v.
        elif HookState.phase == "replace":
            if layer_id in HookState.replacements:
                repl = HookState.replacements[layer_id]
                k_new = repl["K"]  # fp32 GPU tensor [n_tokens, nkv, hd]
                v_new = repl["V"]
                # Make sure shapes match THIS forward's n_tokens
                n_tokens = k.shape[0]
                if k_new.shape[0] == n_tokens:
                    # Reshape back to [n_tokens, nkv*hd] and cast.
                    k = k_new.reshape(n_tokens, -1).to(k.dtype)
                    v = v_new.reshape(n_tokens, -1).to(v.dtype)
                else:
                    # Token count mismatch \u2014 typically means this is a
                    # second forward over a different prompt length.
                    # Skip replacement (shouldn't happen if we stick to
                    # the same prompt ids across capture/replace).
                    pass

        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v, kv_cache, attn_metadata)
        out, _ = self.o_proj(attn_out)
        return out

    Qwen2Attention.forward = patched
    Qwen2Attention._kk_snapshot_patched = True  # type: ignore[attr-defined]
    print("[snap-patch] Qwen2Attention.forward wrapped "
          "(capture / replace / off)", flush=True)


# =============================================================================
# WikiText loader + vLLM driver
# =============================================================================

def load_wikitext_passages(tok: Any, min_tokens: int, n_passages: int,
                           split: str = "test") -> list[str]:
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
    agree = (float(np.mean([1.0 if t_r[i] == t_a[i] else 0.0 for i in range(n)]))
             if n else float("nan"))
    deltas = [abs(lp_r[i] - lp_a[i])
              for i in range(min(len(lp_r), len(lp_a)))
              if np.isfinite(lp_r[i]) and np.isfinite(lp_a[i])]
    return {
        "ppl_ref": ppl_r, "ppl_alt": ppl_a,
        "ppl_delta_rel": (ppl_a - ppl_r) / max(ppl_r, 1e-8),
        "top1_agreement": agree,
        "mean_abs_dlogp_true": (float(np.mean(deltas)) if deltas
                                else float("nan")),
        "n_tokens": n,
    }


def verdict_of(d: float, t: float) -> str:
    if abs(d) <= 0.01 and t >= 0.95:
        return "ACCEPT"
    if abs(d) <= 0.03 and t >= 0.85:
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
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--bit-width-k", type=int, default=3)
    ap.add_argument("--bit-width-v", type=int, default=2)
    ap.add_argument("--q-calib", type=str,
        default="reports/v1_4_q_pca/flagship/"
                "deepseek_distill_q_calib.safetensors")
    ap.add_argument("--k-centroids", type=str,
        default="reports/v1_4_q_pca/calibrated_codebook/ds_K_b3_centroids.f32")
    ap.add_argument("--v-centroids", type=str,
        default="reports/v1_4_q_pca/calibrated_codebook/ds_V_b2_centroids.f32")
    ap.add_argument("--outlier-threshold", type=float, default=2.0)
    ap.add_argument("--boundary-skip-layers", type=int, nargs="*",
                    default=[0, 1, 7, 14, 26, 27])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] loading Q-precond from {args.q_calib}", flush=True)
    qp = qp_load(args.q_calib, skip_layers=[0])
    boundary_skip = set(args.boundary_skip_layers or [])

    install_qwen2_snapshot_patch()
    from vllm import LLM  # type: ignore
    print(f"[{args.model_name}] loading vLLM engine\u2026", flush=True)
    llm = LLM(model=args.model_path, dtype="bfloat16",
              max_model_len=args.ctx_len + args.n_eval + 16,
              gpu_memory_utilization=args.gpu_mem_util,
              enforce_eager=True, trust_remote_code=True)
    tok = llm.get_tokenizer()

    print(f"[{args.model_name}] loading WikiText passages\u2026", flush=True)
    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval,
        n_passages=args.n_passages,
    )
    passages_ids = [tok.encode(p)[: args.ctx_len + args.n_eval]
                    for p in passages
                    if len(tok.encode(p)) >= args.ctx_len + args.n_eval]
    print(f"  usable: {len(passages_ids)}", flush=True)

    per_passage = []
    codec_stats_total: list[dict] = []

    for pi, ids in enumerate(passages_ids):
        print(f"\n  passage {pi + 1}/{len(passages_ids)}", flush=True)

        # ---- Pass 1: clean, codec OFF, captures pre-RoPE K/V ----
        HookState.phase = "capture"
        HookState.captured = {}
        t0 = time.perf_counter()
        ref_pls = prompt_logprobs_for_ids(llm, ids)
        t_ref = time.perf_counter() - t0
        HookState.phase = "off"
        # Sanity: captured layer count.
        n_layers_captured = len(HookState.captured)
        print(f"    [capture] {n_layers_captured} layers, "
              f"{HookState.captured[0]['K'].shape[0]} tokens, "
              f"{t_ref:.2f}s", flush=True)

        # ---- Offline: codec every layer ----
        t0 = time.perf_counter()
        replacements: dict[int, dict[str, torch.Tensor]] = {}
        stats_this = []
        for lid, kv in HookState.captured.items():
            k_hat, k_rep = codec_layer(
                kv["K"], is_v=False, layer_id=lid, q_precond=qp,
                block_size=args.block_size,
                bit_width_k=args.bit_width_k, bit_width_v=args.bit_width_v,
                k_centroids=args.k_centroids, v_centroids=args.v_centroids,
                k_outlier_threshold=args.outlier_threshold,
                v_outlier_threshold=None, boundary_skip=boundary_skip,
            )
            v_hat, v_rep = codec_layer(
                kv["V"], is_v=True, layer_id=lid, q_precond=qp,
                block_size=args.block_size,
                bit_width_k=args.bit_width_k, bit_width_v=args.bit_width_v,
                k_centroids=args.k_centroids, v_centroids=args.v_centroids,
                k_outlier_threshold=args.outlier_threshold,
                v_outlier_threshold=None, boundary_skip=boundary_skip,
            )
            replacements[lid] = {
                "K": torch.from_numpy(k_hat).to("cuda").to(torch.float32),
                "V": torch.from_numpy(v_hat).to("cuda").to(torch.float32),
            }
            stats_this.append(k_rep); stats_this.append(v_rep)
        t_codec = time.perf_counter() - t0
        n_boundary = sum(1 for s in stats_this if s.get("boundary_skip"))
        print(f"    [codec] {len(stats_this)} layer-streams "
              f"({n_boundary} boundary-skipped), {t_codec:.2f}s", flush=True)

        # ---- Pass 2: codec'd K/V injected via replace hook ----
        HookState.replacements = replacements
        HookState.phase = "replace"
        t0 = time.perf_counter()
        alt_pls = prompt_logprobs_for_ids(llm, ids)
        t_alt = time.perf_counter() - t0
        HookState.phase = "off"
        HookState.replacements = {}
        # Free GPU memory held by replacements.
        for r in replacements.values():
            del r["K"]; del r["V"]
        torch.cuda.empty_cache()

        # ---- Compare ----
        metrics = compare(ref_pls, alt_pls, ids, args.ctx_len, args.n_eval)
        m = metrics
        print(
            f"    [result] ppl_ref={m['ppl_ref']:.3f} "
            f"ppl_alt={m['ppl_alt']:.3f} "
            f"Δppl={m['ppl_delta_rel']*100:+.3f}% "
            f"top1={m['top1_agreement']*100:.2f}% "
            f"Δlogp={m['mean_abs_dlogp_true']:.4f} "
            f"(ref={t_ref:.2f}s codec={t_codec:.2f}s alt={t_alt:.2f}s)",
            flush=True,
        )
        per_passage.append({
            "passage": pi, "ctx_len": args.ctx_len, "n_eval": args.n_eval,
            "t_ref_sec": t_ref, "t_codec_sec": t_codec, "t_alt_sec": t_alt,
            "metrics": metrics,
            "n_layers_captured": n_layers_captured,
            "n_boundary_skipped": n_boundary,
        })

    summary = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "engine": "vllm",
        "recipe": "v1.3 PPL snapshot-mode",
        "ctx_len": args.ctx_len, "n_eval": args.n_eval,
        "bit_width_k": args.bit_width_k, "bit_width_v": args.bit_width_v,
        "outlier_threshold": args.outlier_threshold,
        "boundary_skip_layers": sorted(boundary_skip),
        "q_calib": args.q_calib,
        "k_centroids": args.k_centroids,
        "v_centroids": args.v_centroids,
        "n_passages": len(per_passage),
    }
    if per_passage:
        valid = [r for r in per_passage
                 if np.isfinite(r["metrics"]["ppl_delta_rel"])]
        mean_delta = float(np.mean([r["metrics"]["ppl_delta_rel"] for r in valid]))
        mean_top1 = float(np.mean([r["metrics"]["top1_agreement"] for r in valid]))
        summary.update({
            "mean_ppl_delta_rel": mean_delta,
            "mean_top1_agreement": mean_top1,
            "verdict": verdict_of(mean_delta, mean_top1),
        })
        print(f"\n[{args.model_name}] ===== SUMMARY (snapshot-mode) =====",
              flush=True)
        print(f"  n_passages  = {len(per_passage)}", flush=True)
        print(f"  Δppl (mean) = {mean_delta*100:+.3f}%", flush=True)
        print(f"  top1 agree  = {mean_top1*100:.2f}%", flush=True)
        print(f"  VERDICT     = {summary['verdict']}", flush=True)
    else:
        summary["verdict"] = "NO_DATA"

    summary["per_passage"] = per_passage
    out_path = args.out_dir / f"{args.model_name}_vllm_snapshot.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
