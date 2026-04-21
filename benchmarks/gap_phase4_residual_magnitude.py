#!/usr/bin/env python3
"""Phase 4 — compare codec residual magnitude on HF vs vLLM prefill snapshots.

Captures pre-RoPE K and V per (layer, passage) from BOTH engines on the
same token ids, runs the production v1.3 codec (Q-precond + calibrated
Lloyd-Max + outlier T=2.0 for K; Lloyd-Max share_basis for V) on each
snapshot, and compares per-layer:

  - mean_block_mse of the codec (as reported by kakeyaturbo-bench)
  - relative residual norm ||X - X̂||_F / ||X||_F on the block-aligned
    prefix (what the codec actually saw)
  - (optional) per-layer logit impact: the "effective sigma" implied
    by the codec residual, to be picked up by Phase 2 noise curves

If the HF and vLLM distributions of (K, V) entering the codec are
statistically similar, the codec residuals will be too, and Phase 4
will show matched per-layer errors. If they differ, we will see it
here.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
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


def rust_roundtrip(arr: np.ndarray, *, block_size: int, bit_width: int,
                   rsvd_target_rank: int, metric: str, share_basis: bool,
                   centroids_file: str | None,
                   outlier_threshold: float | None) -> tuple[np.ndarray, dict]:
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        tdp = Path(td)
        in_path, rep, dec = tdp / "x.kktv", tdp / "r.json", tdp / "d.kktv"
        write_kktv(in_path, arr.astype(np.float32, copy=False))
        cmd = [
            str(BENCH_BIN), "--input", str(in_path), "--output", str(rep),
            "--metric", metric, "--block-size", str(block_size),
            "--variance-ratio", "0.95",
            "--k", "16", "--bit-width", str(bit_width),
            "--rotation-seed", "3405691582",
            "--pca-method", "randomized",
            "--rsvd-target-rank", str(rsvd_target_rank),
            "--rsvd-oversample", "8", "--rsvd-power-iters", "2",
            "--verify", "--dump-decoded", str(dec),
        ]
        if share_basis:
            cmd.append("--share-basis")
        if centroids_file is not None:
            cmd += ["--centroids-file", str(centroids_file)]
        if outlier_threshold is not None:
            cmd += ["--outlier-threshold", str(outlier_threshold)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"codec rc={res.returncode}: {res.stderr[:2000]}")
        return read_kktv_f32(dec), json.loads(rep.read_text())


# =============================================================================
# Capture pre-RoPE K / V from vLLM (non-invasive, single forward per passage)
# =============================================================================

class _Cap:
    active: bool = False
    K: dict[int, list[np.ndarray]] = {}
    V: dict[int, list[np.ndarray]] = {}
    head_size: int = 0
    num_kv_heads: int = 0
    num_heads: int = 0


def install_vllm_capture_patch() -> None:
    from vllm.model_executor.models.qwen2 import Qwen2Attention  # type: ignore
    if getattr(Qwen2Attention, "_kk_p4_patched", False):
        return
    orig = Qwen2Attention.forward

    def patched(self, positions, hidden_states, kv_cache, attn_metadata):  # type: ignore[no-untyped-def]
        if not _Cap.active:
            return orig(self, positions, hidden_states, kv_cache, attn_metadata)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Parse layer id from self.attn.layer_name.
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
        _Cap.head_size = self.attn.head_size
        _Cap.num_kv_heads = self.attn.num_kv_heads
        _Cap.num_heads = self.attn.num_heads
        hd = self.attn.head_size
        nkv = self.attn.num_kv_heads
        k_np = k.detach().to(torch.float32).cpu().numpy().reshape(-1, nkv, hd)
        v_np = v.detach().to(torch.float32).cpu().numpy().reshape(-1, nkv, hd)
        _Cap.K.setdefault(layer_id, []).append(k_np)
        _Cap.V.setdefault(layer_id, []).append(v_np)
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v, kv_cache, attn_metadata)
        out, _ = self.o_proj(attn_out)
        return out

    Qwen2Attention.forward = patched
    Qwen2Attention._kk_p4_patched = True  # type: ignore[attr-defined]


def capture_vllm(model_path: str, passages_ids: list[list[int]],
                 max_model_len: int, gpu_mem_util: float) -> dict[int, dict]:
    install_vllm_capture_patch()
    from vllm import LLM, SamplingParams  # type: ignore
    llm = LLM(model=model_path, dtype="bfloat16",
              max_model_len=max_model_len,
              gpu_memory_utilization=gpu_mem_util,
              enforce_eager=True, trust_remote_code=True)
    _Cap.active = True
    _Cap.K.clear()
    _Cap.V.clear()
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    for ids in passages_ids:
        _ = llm.generate(prompts=None, prompt_token_ids=[ids],
                         sampling_params=sp, use_tqdm=False)
    _Cap.active = False
    # Stack per layer.
    result: dict[int, dict] = {}
    for l in _Cap.K:
        result[l] = {
            "K": np.concatenate(_Cap.K[l], axis=0),
            "V": np.concatenate(_Cap.V[l], axis=0),
            "head_size": _Cap.head_size,
            "num_kv_heads": _Cap.num_kv_heads,
        }
    return result


# =============================================================================
# Capture pre-RoPE K / V from HF transformers (eager, pre_rope_cache hook)
# =============================================================================

def capture_hf(model_path: str, passages_ids: list[list[int]]
               ) -> dict[int, dict]:
    from transformers import AutoModelForCausalLM  # type: ignore
    # Use the existing pre_rope_cache utility that patches HF Qwen2's
    # attention to record pre-RoPE K/V into cfg._q_recorder.
    # We want K and V separately, so we repurpose the pattern used
    # in benchmarks/q_calibration.py but extend to V.
    # Minimal inline patch: monkey-patch eager attention of Qwen2 to
    # record k, v BEFORE rotary_emb.
    import importlib
    mod_q = importlib.import_module(
        "transformers.models.qwen2.modeling_qwen2"
    )
    orig = mod_q.Qwen2Attention.forward

    state: dict[int, dict] = {}
    n_layer = {"i": -1}

    def patched(self, hidden_states, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Emulate the first few lines of the eager forward to extract
        # pre-RoPE q, k, v, then fall back to the original for the
        # rest of the computation.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        layer_id = self.layer_idx
        k_np = k[0].transpose(0, 1).to(torch.float32).cpu().numpy()
        v_np = v[0].transpose(0, 1).to(torch.float32).cpu().numpy()
        # shape [seq, n_kv, head_dim]
        if layer_id not in state:
            state[layer_id] = {
                "K": [k_np], "V": [v_np],
                "head_size": self.head_dim,
                "num_kv_heads": k_np.shape[1],
            }
        else:
            state[layer_id]["K"].append(k_np)
            state[layer_id]["V"].append(v_np)
        return orig(self, hidden_states, *args, **kwargs)

    mod_q.Qwen2Attention.forward = patched
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", trust_remote_code=True,
        ).to("cuda").eval()
        with torch.inference_mode():
            for ids in passages_ids:
                ids_t = torch.tensor([ids], device="cuda")
                _ = model(ids_t)
        del model
        torch.cuda.empty_cache()
    finally:
        mod_q.Qwen2Attention.forward = orig
    return {l: {
        "K": np.concatenate(s["K"], axis=0),
        "V": np.concatenate(s["V"], axis=0),
        "head_size": s["head_size"],
        "num_kv_heads": s["num_kv_heads"],
    } for l, s in state.items()}


# =============================================================================
# Codec residual analysis
# =============================================================================

def analyse_layer(
    K: np.ndarray, V: np.ndarray, layer_id: int, head_size: int,
    q_precond: QPrecond | None,
    k_centroids: str | None, v_centroids: str | None,
    k_outlier: float | None, boundary_skip: set[int],
    block_size: int = 512, bit_width_k: int = 3, bit_width_v: int = 2,
) -> dict:
    """Run the production codec on captured pre-RoPE K/V for one layer."""
    if layer_id in boundary_skip:
        return {"layer": layer_id, "boundary_skip": True}
    n, nkv, hd = K.shape
    n_comp = (n // block_size) * block_size
    if n_comp == 0:
        return {"layer": layer_id, "skipped_short": True}
    rank = max(2, hd // 2)

    # K path: whiten with Σ_q L (if Qprecond has this layer) → codec → unwhiten
    k_whiten_used = False
    if q_precond is not None and q_precond.is_active(layer_id):
        k_enc = q_precond.whiten(K, layer=layer_id)
        k_whiten_used = True
    else:
        k_enc = K
    k_flat = k_enc.reshape(-1, hd).astype(np.float32, copy=False)[:n_comp]
    k_dec, k_rep = rust_roundtrip(
        k_flat, block_size=block_size, bit_width=bit_width_k,
        rsvd_target_rank=rank,
        metric="mse" if k_whiten_used else "inner_product",
        share_basis=False, centroids_file=k_centroids,
        outlier_threshold=k_outlier,
    )
    k_dec_ = k_dec.reshape(-1, nkv, hd)
    if k_whiten_used:
        k_dec_full = q_precond.unwhiten(k_dec_, layer=layer_id)
    else:
        k_dec_full = k_dec_
    # Per-vector relative residual norm.
    k_gt = K[:n_comp]
    k_err = k_gt - k_dec_full
    k_mse_vs = float(np.mean(k_err ** 2))
    k_relnorm = float(np.linalg.norm(k_err) / max(np.linalg.norm(k_gt), 1e-30))

    # V path.
    v_flat = V.reshape(-1, hd).astype(np.float32, copy=False)[:n_comp]
    v_dec, v_rep = rust_roundtrip(
        v_flat, block_size=block_size, bit_width=bit_width_v,
        rsvd_target_rank=rank, metric="mse", share_basis=True,
        centroids_file=v_centroids, outlier_threshold=None,
    )
    v_dec_ = v_dec.reshape(-1, nkv, hd)
    v_err = V[:n_comp] - v_dec_
    v_mse_vs = float(np.mean(v_err ** 2))
    v_relnorm = float(np.linalg.norm(v_err) /
                      max(np.linalg.norm(V[:n_comp]), 1e-30))

    return {
        "layer": layer_id,
        "n_vecs": int(n),
        "n_compressible": int(n_comp),
        "K": {
            "codec_mean_block_mse": float(k_rep.get("mean_block_mse", -1)),
            "mse_vs_ground_truth": k_mse_vs,
            "relnorm": k_relnorm,
            "whiten": k_whiten_used,
            "mean_K_abs": float(np.mean(np.abs(K))),
        },
        "V": {
            "codec_mean_block_mse": float(v_rep.get("mean_block_mse", -1)),
            "mse_vs_ground_truth": v_mse_vs,
            "relnorm": v_relnorm,
            "mean_V_abs": float(np.mean(np.abs(V))),
        },
    }


# =============================================================================
# WikiText loader
# =============================================================================

def load_passages(tok: Any, min_tokens: int, n_passages: int,
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


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--q-calib", type=str,
        default="reports/v1_4_q_pca/flagship/"
                "deepseek_distill_q_calib.safetensors")
    ap.add_argument("--k-centroids", type=str,
        default="reports/v1_4_q_pca/calibrated_codebook/ds_K_b3_centroids.f32")
    ap.add_argument("--v-centroids", type=str,
        default="reports/v1_4_q_pca/calibrated_codebook/ds_V_b2_centroids.f32")
    ap.add_argument("--k-outlier-threshold", type=float, default=2.0)
    ap.add_argument("--boundary-skip-layers", type=int, nargs="*",
                    default=[0, 1, 7, 14, 26, 27])
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load passages with HF tokenizer (vLLM uses the same).
    from transformers import AutoTokenizer  # type: ignore
    tok = AutoTokenizer.from_pretrained(args.model_path,
                                        trust_remote_code=True)
    passages = load_passages(tok, args.ctx_len, args.n_passages)
    passages_ids = [tok.encode(p)[:args.ctx_len] for p in passages
                    if len(tok.encode(p)) >= args.ctx_len]
    print(f"[setup] {len(passages_ids)} passages of {args.ctx_len} tokens",
          flush=True)

    qp = qp_load(args.q_calib, skip_layers=[0])
    boundary_skip = set(args.boundary_skip_layers or [])

    print("\n[phase4] capturing pre-RoPE K/V from vLLM", flush=True)
    vllm_cap = capture_vllm(args.model_path, passages_ids,
                            max_model_len=args.ctx_len + 16,
                            gpu_mem_util=args.gpu_mem_util)
    print(f"        captured {len(vllm_cap)} layers "
          f"(layer 0 K shape={vllm_cap[0]['K'].shape})", flush=True)
    torch.cuda.empty_cache()

    print("\n[phase4] capturing pre-RoPE K/V from HF eager", flush=True)
    hf_cap = capture_hf(args.model_path, passages_ids)
    print(f"        captured {len(hf_cap)} layers "
          f"(layer 0 K shape={hf_cap[0]['K'].shape})", flush=True)

    # Alignment check: should agree on shape.
    for l in sorted(vllm_cap):
        if l not in hf_cap:
            print(f"  warn: layer {l} missing from HF capture", flush=True)
            continue
        vh, hh = vllm_cap[l]["head_size"], hf_cap[l]["head_size"]
        if vh != hh:
            print(f"  warn: head_size mismatch layer {l}: v={vh} h={hh}",
                  flush=True)

    # Residual analysis per layer per engine.
    print("\n[phase4] running codec residual analysis per layer", flush=True)
    per_layer_vllm, per_layer_hf, pair = [], [], []
    for l in sorted(vllm_cap):
        if l not in hf_cap:
            continue
        r_v = analyse_layer(
            vllm_cap[l]["K"], vllm_cap[l]["V"], l,
            head_size=vllm_cap[l]["head_size"], q_precond=qp,
            k_centroids=args.k_centroids, v_centroids=args.v_centroids,
            k_outlier=args.k_outlier_threshold, boundary_skip=boundary_skip,
            block_size=args.block_size,
        )
        r_h = analyse_layer(
            hf_cap[l]["K"], hf_cap[l]["V"], l,
            head_size=hf_cap[l]["head_size"], q_precond=qp,
            k_centroids=args.k_centroids, v_centroids=args.v_centroids,
            k_outlier=args.k_outlier_threshold, boundary_skip=boundary_skip,
            block_size=args.block_size,
        )
        per_layer_vllm.append(r_v)
        per_layer_hf.append(r_h)
        if r_v.get("boundary_skip") or r_v.get("skipped_short"):
            continue
        dk = r_v["K"]["mse_vs_ground_truth"] - r_h["K"]["mse_vs_ground_truth"]
        dv = r_v["V"]["mse_vs_ground_truth"] - r_h["V"]["mse_vs_ground_truth"]
        pair.append({
            "layer": l,
            "K_mse_vllm": r_v["K"]["mse_vs_ground_truth"],
            "K_mse_hf":   r_h["K"]["mse_vs_ground_truth"],
            "K_mse_ratio": r_v["K"]["mse_vs_ground_truth"]
                           / max(r_h["K"]["mse_vs_ground_truth"], 1e-30),
            "K_relnorm_vllm": r_v["K"]["relnorm"],
            "K_relnorm_hf":   r_h["K"]["relnorm"],
            "V_mse_vllm": r_v["V"]["mse_vs_ground_truth"],
            "V_mse_hf":   r_h["V"]["mse_vs_ground_truth"],
            "V_mse_ratio": r_v["V"]["mse_vs_ground_truth"]
                           / max(r_h["V"]["mse_vs_ground_truth"], 1e-30),
            "V_relnorm_vllm": r_v["V"]["relnorm"],
            "V_relnorm_hf":   r_h["V"]["relnorm"],
            "mean_K_abs_vllm": r_v["K"]["mean_K_abs"],
            "mean_K_abs_hf":   r_h["K"]["mean_K_abs"],
            "mean_V_abs_vllm": r_v["V"]["mean_V_abs"],
            "mean_V_abs_hf":   r_h["V"]["mean_V_abs"],
        })
        print(
            f"  L{l:02d}  K: mse v={r_v['K']['mse_vs_ground_truth']:.6f} "
            f"h={r_h['K']['mse_vs_ground_truth']:.6f} "
            f"(v/h={r_v['K']['mse_vs_ground_truth']/max(r_h['K']['mse_vs_ground_truth'],1e-30):.2f}x)  "
            f"V: mse v={r_v['V']['mse_vs_ground_truth']:.6f} "
            f"h={r_h['V']['mse_vs_ground_truth']:.6f} "
            f"(v/h={r_v['V']['mse_vs_ground_truth']/max(r_h['V']['mse_vs_ground_truth'],1e-30):.2f}x)",
            flush=True,
        )

    # Aggregate.
    def _stat(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"min": float("nan"), "max": float("nan"),
                    "mean": float("nan"), "median": float("nan")}
        arr = np.array(xs)
        return {"min": float(arr.min()), "max": float(arr.max()),
                "mean": float(arr.mean()), "median": float(np.median(arr))}

    summary = {
        "model_name": args.model_name,
        "ctx_len": args.ctx_len,
        "n_passages": len(passages_ids),
        "n_layers_compared": len(pair),
        "K_mse_ratio_stats":  _stat([p["K_mse_ratio"] for p in pair]),
        "V_mse_ratio_stats":  _stat([p["V_mse_ratio"] for p in pair]),
        "K_relnorm_vllm_stats": _stat([p["K_relnorm_vllm"] for p in pair]),
        "K_relnorm_hf_stats":   _stat([p["K_relnorm_hf"]   for p in pair]),
        "V_relnorm_vllm_stats": _stat([p["V_relnorm_vllm"] for p in pair]),
        "V_relnorm_hf_stats":   _stat([p["V_relnorm_hf"]   for p in pair]),
        "mean_K_abs_vllm": float(np.mean([p["mean_K_abs_vllm"] for p in pair])),
        "mean_K_abs_hf":   float(np.mean([p["mean_K_abs_hf"]   for p in pair])),
        "mean_V_abs_vllm": float(np.mean([p["mean_V_abs_vllm"] for p in pair])),
        "mean_V_abs_hf":   float(np.mean([p["mean_V_abs_hf"]   for p in pair])),
        "per_layer": pair,
    }
    out_path = args.out_dir / f"{args.model_name}_residual_magnitude.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    print(f"\n===== SUMMARY =====", flush=True)
    print(f"  layers compared: {len(pair)}", flush=True)
    print(f"  K mse ratio (vllm/hf): "
          f"median={summary['K_mse_ratio_stats']['median']:.3f}, "
          f"max={summary['K_mse_ratio_stats']['max']:.3f}", flush=True)
    print(f"  V mse ratio (vllm/hf): "
          f"median={summary['V_mse_ratio_stats']['median']:.3f}, "
          f"max={summary['V_mse_ratio_stats']['max']:.3f}", flush=True)
    print(f"  K mean |K|:  vllm={summary['mean_K_abs_vllm']:.4f}  "
          f"hf={summary['mean_K_abs_hf']:.4f}", flush=True)
    print(f"  V mean |V|:  vllm={summary['mean_V_abs_vllm']:.4f}  "
          f"hf={summary['mean_V_abs_hf']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
