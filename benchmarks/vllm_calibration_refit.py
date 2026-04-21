#!/usr/bin/env python3
"""Re-fit Σ_q Cholesky + Lloyd-Max centroids on vLLM prefill snapshots.

Motivation
----------
The ablation in `reports/v1_3_ppl/vllm_ablation/FINDINGS.md` localised
the remaining HF-vs-vLLM Δppl gap (+7.82 % vs +35.33 % at the exact
same codec config) to **calibration-distribution drift**: Σ_q and
Lloyd-Max centroids in `reports/v1_4_q_pca/` were all fit on HF
DynamicCache snapshots, but vLLM produces slightly different prefill
distributions (bf16 accumulation order, RoPE impl, attention bias).

This script:

  1. Spins up a vLLM LLM.
  2. Installs a capture-only monkey patch on
     `Qwen2Attention.forward` that records the **pre-RoPE** q, k, v
     tensors (after qkv_proj.split, BEFORE rotary_emb) into per-layer
     buffers WITHOUT touching the forward's computation.
  3. Runs N calibration passages (from WikiText-103 train split by
     default, so disjoint from the test passages used for PPL
     measurement).
  4. Computes Σ_q per (layer, kv-head) by pooling over Q heads in the
     same GQA KV group, then Cholesky + inv.
  5. Writes a safetensors file in the exact schema expected by
     `q_precondition.QPrecond`:
         layer_<l>_chol     [n_kv, D, D] fp32
         layer_<l>_inv_chol [n_kv, D, D] fp32
         layer_<l>_sigma    [n_kv, D, D] fp32
     and a sidecar .json with model/config metadata.
  6. Runs Lloyd-Max on the collected K (whitened with the freshly
     fit Σ_q) and V residuals to produce Gaussian-replacement
     centroid tables compatible with `kakeyaturbo-bench
     --centroids-file`. Outputs one .f32 per (stream, bit-width) pair.

The output file names match the HF-calibrated ones so the ablation
harness can drop them in:

  <out-dir>/q_calib.safetensors
  <out-dir>/q_calib.json
  <out-dir>/K_b{2,3}_centroids.f32
  <out-dir>/V_b2_centroids.f32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))

# Import ONLY the pure-numpy math utilities from the Lloyd-Max helper
# without triggering its HF-side imports (transformers + pre_rope_cache).
# We do this by pulling the needed functions out of the module source.
def _load_lm_helpers():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lmc_math", str(REPO / "benchmarks" / "lloyd_max_calibration.py"),
    )
    # Rather than execute the module (which imports transformers +
    # pre_rope_cache), parse it and exec only the function defs we need.
    src = (REPO / "benchmarks" / "lloyd_max_calibration.py").read_text()
    # Strip the top-level HF imports so exec() doesn't fail.
    safe_lines = []
    skip_imports = (
        "from transformers",
        "import benchmarks.pre_rope_cache",
        "from benchmarks.q_precondition",
        "from benchmarks.e2e_ppl_pre_rope",
    )
    for line in src.splitlines():
        if any(line.lstrip().startswith(p) for p in skip_imports):
            continue
        safe_lines.append(line)
    ns: dict = {"__name__": "lmc_math"}
    exec("\n".join(safe_lines), ns)
    return ns

_lmc = _load_lm_helpers()
fit_pca_simple = _lmc["fit_pca_simple"]
next_pow2 = _lmc["next_pow2"]
wht_rotate = _lmc["wht_rotate"]
lloyd_max_iterate = _lmc["lloyd_max_iterate"]


# =============================================================================
# Capture-only monkey patch on Qwen2Attention.forward
# =============================================================================

class CaptureState:
    """Per-run buffers; one dict keyed by layer index.

    - `pre_rope_q[l]`: list of [num_tokens, num_heads, head_dim] fp32 cpu
    - `pre_rope_k[l]`: list of [num_tokens, num_kv_heads, head_dim] fp32 cpu
    - `pre_rope_v[l]`: list of [num_tokens, num_kv_heads, head_dim] fp32 cpu
    """
    active: bool = False
    pre_rope_q: dict[int, list[np.ndarray]] = {}
    pre_rope_k: dict[int, list[np.ndarray]] = {}
    pre_rope_v: dict[int, list[np.ndarray]] = {}
    head_size: int = 0
    num_kv_heads: int = 0
    num_heads: int = 0


def install_qwen2_capture_patch() -> None:
    from vllm.model_executor.models.qwen2 import Qwen2Attention  # type: ignore

    if getattr(Qwen2Attention, "_kk_capture_patched", False):
        return

    orig_forward = Qwen2Attention.forward

    def patched(
        self: Qwen2Attention,  # type: ignore[name-defined]
        positions: torch.Tensor, hidden_states: torch.Tensor,
        kv_cache: torch.Tensor, attn_metadata: Any,
    ) -> torch.Tensor:
        if not CaptureState.active:
            return orig_forward(self, positions, hidden_states, kv_cache,
                                attn_metadata)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )

        # Layer id from self.attn.layer_name = "model.layers.{L}.self_attn.attn"
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

        # Record sizes once
        CaptureState.head_size = self.attn.head_size
        CaptureState.num_kv_heads = self.attn.num_kv_heads
        CaptureState.num_heads = self.attn.num_heads

        # Shape: [num_tokens, {n_heads|n_kv_heads} * head_size] → reshape
        nh = self.attn.num_heads
        nkv = self.attn.num_kv_heads
        hd = self.attn.head_size
        q_np = q.detach().to(torch.float32).cpu().numpy().reshape(-1, nh, hd)
        k_np = k.detach().to(torch.float32).cpu().numpy().reshape(-1, nkv, hd)
        v_np = v.detach().to(torch.float32).cpu().numpy().reshape(-1, nkv, hd)

        CaptureState.pre_rope_q.setdefault(layer_id, []).append(q_np)
        CaptureState.pre_rope_k.setdefault(layer_id, []).append(k_np)
        CaptureState.pre_rope_v.setdefault(layer_id, []).append(v_np)

        # Run the rest of the normal forward. The parent class does:
        #   q, k = self.rotary_emb(positions, q, k)
        #   attn_out = self.attn(q, k, v, kv_cache, attn_metadata)
        #   out, _ = self.o_proj(attn_out)
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v, kv_cache, attn_metadata)
        out, _ = self.o_proj(attn_out)
        return out

    Qwen2Attention.forward = patched
    Qwen2Attention._kk_capture_patched = True  # type: ignore[attr-defined]
    print("[calib-patch] Qwen2Attention.forward wrapped (capture mode)",
          flush=True)


# =============================================================================
# vLLM driver
# =============================================================================

def build_llm(model_path: str, max_model_len: int, gpu_mem_util: float):
    from vllm import LLM  # type: ignore
    return LLM(
        model=model_path, dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True, trust_remote_code=True,
    )


def load_wikitext_passages(
    tokenizer: Any, min_tokens: int, n_passages: int, split: str = "test",
) -> list[str]:
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
            if len(tokenizer.encode(passage)) >= min_tokens:
                passages.append(passage)
                if len(passages) >= n_passages:
                    return passages
            cur, approx = [], 0
    return passages


def _run_prefill(llm: Any, ids: list[int]) -> None:
    """Trigger a single forward so the capture patch fills its buffers.
    Uses max_tokens=1, temperature=0 — we discard the output."""
    from vllm import SamplingParams  # type: ignore
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    _ = llm.generate(
        prompts=None, prompt_token_ids=[ids],
        sampling_params=sp, use_tqdm=False,
    )


# =============================================================================
# Σ_q calibration
# =============================================================================

def fit_sigma_q(
    num_layers: int, num_kv_heads: int, num_heads: int, head_dim: int,
    layer_types: list[str], ridge: float,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    """Factor Σ_q per (layer, kv-head) from CaptureState.pre_rope_q.

    For GQA, for each kv-head h we pool the Q heads in the same group
    `range(h*gs, (h+1)*gs)` — those are the Q heads that will dot with
    this kv-head's K.
    """
    gs = num_heads // num_kv_heads
    out_tensors: dict[str, torch.Tensor] = {}
    diagnostics: list[dict] = []

    for l in range(num_layers):
        if l not in CaptureState.pre_rope_q:
            continue
        if layer_types[l] != "full_attention":
            continue
        chol_stack = np.zeros((num_kv_heads, head_dim, head_dim), dtype=np.float32)
        inv_chol_stack = np.zeros_like(chol_stack)
        sigma_stack = np.zeros_like(chol_stack)

        # Stack all captured tokens for this layer along the token axis.
        all_q = np.concatenate(CaptureState.pre_rope_q[l], axis=0)  # [N, nh, D]
        n_tokens = all_q.shape[0]
        for h_kv in range(num_kv_heads):
            group = list(range(h_kv * gs, (h_kv + 1) * gs))
            q_group = all_q[:, group, :]                   # [N, gs, D]
            q_flat = q_group.reshape(-1, head_dim)          # [N*gs, D]
            gram = (q_flat.astype(np.float64).T @
                    q_flat.astype(np.float64))              # [D, D]
            sigma = gram / q_flat.shape[0]
            sigma = 0.5 * (sigma + sigma.T)
            mean_diag = float(np.mean(np.diag(sigma)))
            sigma_reg = sigma + ridge * mean_diag * np.eye(head_dim)
            try:
                L = np.linalg.cholesky(sigma_reg)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(
                    sigma_reg + 1e-2 * mean_diag * np.eye(head_dim)
                )
            L_inv = np.linalg.solve(L, np.eye(head_dim))
            chol_stack[h_kv] = L.astype(np.float32)
            inv_chol_stack[h_kv] = L_inv.astype(np.float32)
            sigma_stack[h_kv] = sigma.astype(np.float32)

            evals = np.linalg.eigvalsh(sigma_reg)
            diagnostics.append({
                "layer": l, "kv_head": h_kv,
                "n_tokens_pooled": int(q_flat.shape[0]),
                "sigma_trace": float(np.trace(sigma)),
                "eig_min": float(evals.min()),
                "eig_max": float(evals.max()),
                "condition": float(evals.max() / max(evals.min(), 1e-30)),
                "diag_mean": mean_diag,
                "off_diag_max_abs": float(np.abs(
                    sigma - np.diag(np.diag(sigma))
                ).max()),
            })

        out_tensors[f"layer_{l}_chol"] = torch.from_numpy(chol_stack)
        out_tensors[f"layer_{l}_inv_chol"] = torch.from_numpy(inv_chol_stack)
        out_tensors[f"layer_{l}_sigma"] = torch.from_numpy(sigma_stack)

    return out_tensors, diagnostics


# =============================================================================
# Lloyd-Max centroid re-fit (K / V streams)
# =============================================================================

def collect_kv_residuals(
    stream: str,           # "K" or "V"
    num_layers: int, num_kv_heads: int, head_dim: int,
    layer_types: list[str], chols: dict[int, np.ndarray] | None,
    block_size: int, rotation_seed: int, vr: float = 1.0,
) -> np.ndarray:
    """Apply the same residual pipeline as lloyd_max_calibration.collect_residuals
    but on the in-memory vLLM capture buffers.
    """
    buf = (CaptureState.pre_rope_k if stream == "K"
           else CaptureState.pre_rope_v)
    residual_pool: list[np.ndarray] = []
    for l in range(num_layers):
        if l not in buf:
            continue
        if layer_types[l] != "full_attention":
            continue
        all_t = np.concatenate(buf[l], axis=0)  # [N, n_kv, D]
        if stream == "K" and chols is not None and l in chols:
            # Apply L whitening per kv-head: K̃[n, h, :] = K[n, h, :] @ L[h]
            L = chols[l]  # [n_kv, D, D]
            all_t = np.einsum("thj,hjk->thk", all_t, L, optimize=True).astype(
                np.float32, copy=False,
            )

        flat = all_t.reshape(-1, head_dim).astype(np.float32, copy=False)
        n_total = flat.shape[0]
        n_comp = (n_total // block_size) * block_size
        if n_comp == 0:
            continue

        # Replicate the codec's post-PCA post-WHT unit-scaled residual pipeline.
        for bs0 in range(0, n_comp, block_size):
            block = flat[bs0:bs0 + block_size]
            mean, basis, d_eff = fit_pca_simple(block, vr=vr)
            coeff = (block - mean) @ basis.T  # [bs, d_eff]
            wht_len = next_pow2(d_eff)
            padded = np.zeros((coeff.shape[0], wht_len), dtype=np.float32)
            padded[:, :d_eff] = coeff
            rotated = wht_rotate(padded, rotation_seed)
            norms = np.linalg.norm(coeff, axis=1, keepdims=True).clip(min=1e-12)
            scaled = rotated / norms
            residual_pool.append(scaled.reshape(-1).astype(np.float32))
    return np.concatenate(residual_pool) if residual_pool else np.empty((0,), np.float32)


def write_centroids(centroids: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(centroids.astype("<f4").tobytes())


GAUSSIAN_INIT = {
    1: [-0.798156, 0.798156],
    2: [-1.5100, -0.4528, 0.4528, 1.5100],
    3: [-2.151945, -1.343757, -0.756268, -0.244943,
        0.244943, 0.756268, 1.343757, 2.151945],
    4: [-2.7322, -2.0690, -1.6177, -1.2563, -0.9422, -0.6566, -0.3885,
        -0.1281, 0.1281, 0.3885, 0.6566, 0.9422, 1.2563, 1.6177,
        2.0690, 2.7322],
}


def fit_centroids(
    samples: np.ndarray, bit_width: int, max_iter: int = 200,
) -> tuple[np.ndarray, dict]:
    if samples.size > 5_000_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(samples.size, size=5_000_000, replace=False)
        samples = samples[idx]
    init = np.array(GAUSSIAN_INIT[bit_width], dtype=np.float64)
    centroids = lloyd_max_iterate(samples, bit_width,
                                  init_centroids=init, max_iter=max_iter)
    # MSE vs Gaussian.
    gauss = np.array(GAUSSIAN_INIT[bit_width], dtype=np.float64)
    rec_g = gauss[np.argmin(np.abs(samples[:, None] - gauss[None, :]), axis=1)]
    rec_c = centroids[np.argmin(
        np.abs(samples[:, None] - centroids[None, :]), axis=1)]
    mse_g = float(np.mean((samples - rec_g) ** 2))
    mse_c = float(np.mean((samples - rec_c) ** 2))
    return centroids, {
        "n_samples": int(samples.size),
        "std": float(samples.std()),
        "mse_gaussian": mse_g,
        "mse_calibrated": mse_c,
        "mse_improvement_x": (mse_g / max(mse_c, 1e-30)),
        "gaussian_init": GAUSSIAN_INIT[bit_width],
        "calibrated": centroids.tolist(),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-passages", type=int, default=8)
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--wikitext-split", default="train",
                    help="wikitext split used for calibration; use a split "
                         "DISJOINT from the PPL test split (default: train)")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--k-bits", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--v-bits", type=int, nargs="+", default=[2])
    ap.add_argument("--block-size-calib", type=int, default=1024)
    ap.add_argument("--rotation-seed", type=int, default=3405691582)
    ap.add_argument("--variance-ratio", type=float, default=1.0)
    ap.add_argument("--gpu-mem-util", type=float, default=0.80)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    install_qwen2_capture_patch()
    print(f"[calib] loading vLLM engine on {args.model_path}…", flush=True)
    max_len = args.ctx_len + 16
    llm = build_llm(args.model_path, max_len, args.gpu_mem_util)
    tok = llm.get_tokenizer()

    # Load disjoint calibration passages.
    print(f"[calib] loading WikiText-103/{args.wikitext_split} passages "
          f"(min_tokens={args.ctx_len}, n={args.n_passages})…", flush=True)
    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len, n_passages=args.n_passages,
        split=args.wikitext_split,
    )
    print(f"  got {len(passages)} passages", flush=True)

    # Capture pass: prefill each passage to fill CaptureState buffers.
    CaptureState.active = True
    CaptureState.pre_rope_q.clear()
    CaptureState.pre_rope_k.clear()
    CaptureState.pre_rope_v.clear()
    for i, p in enumerate(passages):
        ids = tok.encode(p)[: args.ctx_len]
        if len(ids) < args.ctx_len:
            print(f"  passage {i + 1}: SKIP (short: {len(ids)})", flush=True)
            continue
        _run_prefill(llm, ids)
        n_tokens = sum(arr.shape[0]
                       for arr in CaptureState.pre_rope_q.get(0, []))
        print(f"  passage {i + 1}/{len(passages)}: captured "
              f"(layer-0 tokens so far: {n_tokens})", flush=True)
    CaptureState.active = False

    # Introspect model dims from what we captured.
    num_layers = max(CaptureState.pre_rope_q.keys()) + 1
    head_size = CaptureState.head_size
    num_kv = CaptureState.num_kv_heads
    num_q = CaptureState.num_heads
    layer_types = ["full_attention"] * num_layers
    print(f"[calib] captured {num_layers} layers, "
          f"head_size={head_size}, num_kv={num_kv}, num_q={num_q}",
          flush=True)

    # ---- 1. Σ_q ----
    print("\n[sigma] fitting Σ_q + Cholesky per (layer, kv-head)…",
          flush=True)
    sigma_tensors, diagnostics = fit_sigma_q(
        num_layers, num_kv, num_q, head_size,
        layer_types=layer_types, ridge=args.ridge,
    )
    from safetensors.torch import save_file
    sigma_path = args.out_dir / "q_calib.safetensors"
    save_file(sigma_tensors, str(sigma_path))
    (args.out_dir / "q_calib.json").write_text(json.dumps({
        "source": "vllm prefill re-calibration",
        "model_path": args.model_path,
        "head_dim": head_size,
        "num_q_heads": num_q,
        "num_kv_heads": num_kv,
        "num_layers": num_layers,
        "layer_types": layer_types,
        "n_passages_used": len(passages),
        "ctx_len": args.ctx_len,
        "wikitext_split": args.wikitext_split,
        "ridge": args.ridge,
        "diagnostics": diagnostics,
    }, indent=2))
    # Summary stats.
    conds = [d["condition"] for d in diagnostics]
    offr = [d["off_diag_max_abs"] / max(d["diag_mean"], 1e-30)
            for d in diagnostics]
    print(f"  condition: min={min(conds):.2f} "
          f"median={np.median(conds):.2f} max={max(conds):.2f}", flush=True)
    print(f"  off/diag : min={min(offr):.3f} "
          f"median={np.median(offr):.3f} max={max(offr):.3f}", flush=True)
    print(f"  wrote {sigma_path}", flush=True)

    # Prepare chol lookup for whitening K prior to Lloyd-Max.
    chols: dict[int, np.ndarray] = {}
    for l in range(num_layers):
        key = f"layer_{l}_chol"
        if key in sigma_tensors:
            chols[l] = sigma_tensors[key].numpy()

    # ---- 2. Lloyd-Max centroids, per (stream, bit) ----
    calib_summary: dict[str, Any] = {
        "sigma_path": str(sigma_path), "centroids": {},
    }

    for b in args.k_bits:
        print(f"\n[centroids] collecting K-stream residuals at b={b}…",
              flush=True)
        samples = collect_kv_residuals(
            "K", num_layers, num_kv, head_size, layer_types, chols,
            block_size=args.block_size_calib,
            rotation_seed=args.rotation_seed, vr=args.variance_ratio,
        )
        print(f"  n={samples.size:,} samples "
              f"mean={samples.mean():.4f} std={samples.std():.4f}",
              flush=True)
        print(f"  running Lloyd-Max (b={b})…", flush=True)
        cent, info = fit_centroids(samples, b)
        path = args.out_dir / f"K_b{b}_centroids.f32"
        write_centroids(cent, path)
        print(f"  MSE Gaussian={info['mse_gaussian']:.4e} "
              f"Calibrated={info['mse_calibrated']:.4e} "
              f"({info['mse_improvement_x']:.2f}× better)", flush=True)
        print(f"  wrote {path}", flush=True)
        calib_summary["centroids"][f"K_b{b}"] = {
            "path": str(path), **info,
        }

    for b in args.v_bits:
        print(f"\n[centroids] collecting V-stream residuals at b={b}…",
              flush=True)
        samples = collect_kv_residuals(
            "V", num_layers, num_kv, head_size, layer_types, None,
            block_size=args.block_size_calib,
            rotation_seed=args.rotation_seed, vr=args.variance_ratio,
        )
        print(f"  n={samples.size:,} samples "
              f"mean={samples.mean():.4f} std={samples.std():.4f}",
              flush=True)
        print(f"  running Lloyd-Max (b={b})…", flush=True)
        cent, info = fit_centroids(samples, b)
        path = args.out_dir / f"V_b{b}_centroids.f32"
        write_centroids(cent, path)
        print(f"  MSE Gaussian={info['mse_gaussian']:.4e} "
              f"Calibrated={info['mse_calibrated']:.4e} "
              f"({info['mse_improvement_x']:.2f}× better)", flush=True)
        print(f"  wrote {path}", flush=True)
        calib_summary["centroids"][f"V_b{b}"] = {
            "path": str(path), **info,
        }

    summary_path = args.out_dir / "SUMMARY.json"
    summary_path.write_text(json.dumps(calib_summary, indent=2))
    print(f"\nwrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
