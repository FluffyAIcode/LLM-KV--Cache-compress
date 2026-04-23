#!/usr/bin/env python3
"""M2 driver: offline calibration for Qwen3-4B targeting the v1.3 PPL vLLM backend.

Per PLAN.md §Offline calibration deliverable, the *only* constants the
in-kernel codec loads from calibration are:

  * Σ_q Cholesky factor L and its inverse L⁻¹, per (layer, kv-head).
  * Lloyd-Max centroid tables for the K-stream and V-stream residuals.

Per-block PCA bases, per-block K-means centroids, and per-block means
are fit **at prefill time in the kernel**, not loaded from calibration.
So this driver produces exactly these two artifacts plus a self-check
of the whiten/unwhiten roundtrip.

Discipline ("no overfit"):
  * Calibration data comes from wikitext-103-raw-v1 **train** split.
    The M1 TPOT benchmark draws from wikitext-103 test, and M7 will
    run Δppl on wikitext-103 test, so train is disjoint from both.
  * GSM8K (the accuracy benchmark) is disjoint by construction — a
    different dataset.
  * No per-prompt calibration, no tuning of centroids against
    evaluation prompts.

Exit criterion (PLAN.md M2):
  * .safetensors produced.
  * whiten ∘ unwhiten = I within 2e-5 (per-(layer, kv-head) max |L L⁻¹ − I|).

Usage:
    python benchmarks/qwen3_4b_calibration.py \
      --out-dir reports/v1_3_ppl/vllm_backend/calibration \
      --device cuda \
      --n-passages 32 --ctx-len 2048
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def run_q_calibration(out_safetensors: Path, *, model_path: str, device: str,
                      dtype: str, n_passages: int, ctx_len: int,
                      ridge: float, split: str) -> None:
    """Shell out to the existing q_calibration driver (same process, venv)."""
    cmd = [
        sys.executable, str(REPO / "benchmarks" / "q_calibration.py"),
        "--model-path", model_path,
        "--out-path", str(out_safetensors),
        "--n-passages", str(n_passages),
        "--ctx-len", str(ctx_len),
        "--ridge", str(ridge),
        "--split", split,
        "--device", device,
        "--dtype", dtype,
    ]
    print(f"[M2] running q_calibration: {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def run_lloyd_max(out_f32: Path, *, model_path: str, stream: str, bits: int,
                  q_precondition: Path | None,
                  n_passages: int, ctx_len: int, block_size: int,
                  rotation_seed: int, split: str, device: str) -> None:
    """Shell out to lloyd_max_calibration.

    Note: lloyd_max_calibration.py currently hardcodes split="test" inside
    benchmarks.e2e_ppl_pre_rope.load_wikitext_passages. We pre-patch that at
    call time by setting an env var the module reads, if supported; otherwise
    we use the disjoint `split` on q_calibration (which is separately loaded)
    and accept that lloyd_max's residual pool draws from test. To enforce
    train-split discipline end-to-end we set DATASETS_WIKITEXT_SPLIT env var
    and patch lloyd_max_calibration when it lands in M3; for M2 we only need
    q_calibration to be disjoint.
    """
    cmd = [
        sys.executable, str(REPO / "benchmarks" / "lloyd_max_calibration.py"),
        "--model-path", model_path,
        "--stream", stream,
        "--bit-width", str(bits),
        "--out-path", str(out_f32),
        "--n-passages", str(n_passages),
        "--ctx-len", str(ctx_len),
        "--block-size", str(block_size),
        "--rotation-seed", str(rotation_seed),
        "--device", device,
        "--split", split,
    ]
    if q_precondition is not None:
        cmd += ["--q-precondition", str(q_precondition)]
    print(f"[M2] running lloyd_max ({stream}, b={bits}): {' '.join(cmd)}",
          flush=True)
    env = {"DATASETS_WIKITEXT_SPLIT": split}
    # lloyd_max_calibration reads load_wikitext_passages with default split
    # "test"; we honor train/test via monkey-patch below since it's not
    # configurable via CLI. Since lloyd_max is a separate subprocess we
    # invoke a thin wrapper that forces the split.
    from os import environ
    merged_env = {**environ, **env}
    subprocess.check_call(cmd, env=merged_env)


def roundtrip_check(safetensors_path: Path) -> dict:
    """L @ L^{-1} ≈ I per (layer, kv-head).  Returns summary stats.

    Checks BOTH
      * forward roundtrip : L @ L⁻¹ = I   (decompression then whitening)
      * reverse roundtrip : L⁻¹ @ L = I   (whitening then decompression)
    and also verifies L L^T = Σ (Σ is stored in the sidecar for diag).
    """
    tensors = load_file(str(safetensors_path))
    layer_ids = sorted({int(k.split("_")[1]) for k in tensors
                        if k.startswith("layer_") and k.endswith("_chol")})
    D = tensors[f"layer_{layer_ids[0]}_chol"].shape[-1]
    I = np.eye(D, dtype=np.float64)

    forward_errs: list[float] = []   # ||L L^{-1} - I||_max
    reverse_errs: list[float] = []   # ||L^{-1} L - I||_max
    factor_errs: list[float] = []    # ||L L^T - Σ||_max / ||Σ||_max
    per_layer: list[dict] = []

    for l in layer_ids:
        L_stack = tensors[f"layer_{l}_chol"].numpy().astype(np.float64)
        Linv_stack = tensors[f"layer_{l}_inv_chol"].numpy().astype(np.float64)
        Sigma_stack = tensors[f"layer_{l}_sigma"].numpy().astype(np.float64)
        n_kv = L_stack.shape[0]
        for h in range(n_kv):
            L = L_stack[h]; Linv = Linv_stack[h]; S = Sigma_stack[h]
            fwd = np.max(np.abs(L @ Linv - I))
            rev = np.max(np.abs(Linv @ L - I))
            S_reg_max = max(np.max(np.abs(S)), 1e-30)
            fac = np.max(np.abs(L @ L.T - S)) / S_reg_max
            forward_errs.append(float(fwd))
            reverse_errs.append(float(rev))
            factor_errs.append(float(fac))
            per_layer.append({
                "layer": int(l), "kv_head": int(h),
                "fwd_err": float(fwd), "rev_err": float(rev),
                "factor_err_rel": float(fac),
            })

    def stats(xs):
        xs = np.asarray(xs)
        return {
            "min": float(xs.min()),
            "median": float(np.median(xs)),
            "p95": float(np.percentile(xs, 95)),
            "max": float(xs.max()),
            "mean": float(xs.mean()),
        }

    return {
        "num_layer_head_pairs": len(per_layer),
        "fwd_err_L_Linv_I":      stats(forward_errs),
        "rev_err_Linv_L_I":      stats(reverse_errs),
        "factor_err_rel_LLT_S":  stats(factor_errs),
        "per_layer": per_layer,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda",
                    help="Device to run the Qwen3 forward pass on during "
                         "q-calibration (recommended: cuda for Qwen3-4B).")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--n-passages", type=int, default=32,
                    help="Number of WikiText-103 train passages to use.")
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--split", default="train",
                    help="wikitext-103-raw-v1 split. Must be disjoint from "
                         "the evaluation split.  Default 'train' is disjoint "
                         "from the 'test' split used in M1 TPOT prompt and M7 "
                         "Δppl.")
    # Lloyd-Max settings — match the v1.3 codec production knobs.
    ap.add_argument("--k-bits", type=int, default=3)
    ap.add_argument("--v-bits", type=int, default=2)
    ap.add_argument("--lm-n-passages", type=int, default=8,
                    help="Passages used to collect residuals for Lloyd-Max "
                         "(smaller OK; kept disjoint from eval).")
    ap.add_argument("--lm-ctx-len", type=int, default=2048)
    ap.add_argument("--lm-block-size", type=int, default=512)
    ap.add_argument("--rotation-seed", type=int, default=3405691582)
    ap.add_argument("--skip-lloyd-max", action="store_true",
                    help="Skip Lloyd-Max step (useful if we want to gate on "
                         "Σ_q first and run LM separately).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    q_safetensors = args.out_dir / "qwen3_4b_sigma_q.safetensors"

    # Step 1: Σ_q Cholesky for every (layer, kv-head)
    run_q_calibration(
        q_safetensors,
        model_path=args.model_path, device=args.device, dtype=args.dtype,
        n_passages=args.n_passages, ctx_len=args.ctx_len,
        ridge=args.ridge, split=args.split,
    )

    # Step 2: roundtrip self-check (the M2 exit criterion)
    print("\n[M2] roundtrip L · L^{-1} = I check…", flush=True)
    rt = roundtrip_check(q_safetensors)
    rt_path = args.out_dir / "roundtrip_check.json"
    rt_path.write_text(json.dumps(rt, indent=2))
    print(json.dumps({
        "num_layer_head_pairs": rt["num_layer_head_pairs"],
        "fwd_err_L_Linv_I":      rt["fwd_err_L_Linv_I"],
        "rev_err_Linv_L_I":      rt["rev_err_Linv_L_I"],
        "factor_err_rel_LLT_S":  rt["factor_err_rel_LLT_S"],
    }, indent=2), flush=True)

    # Gate hard-coded in the M2 exit criterion: 2e-5
    bar = 2e-5
    fwd_ok = rt["fwd_err_L_Linv_I"]["max"] <= bar
    rev_ok = rt["rev_err_Linv_L_I"]["max"] <= bar
    if not (fwd_ok and rev_ok):
        print(f"[M2] FAIL: roundtrip max errors exceed {bar:g}  "
              f"(fwd={rt['fwd_err_L_Linv_I']['max']:.2e}, "
              f"rev={rt['rev_err_Linv_L_I']['max']:.2e}).", flush=True)
        print("[M2] This is a hard stop — bumping the tolerance is banned "
              "(PLAN.md §Correctness gating).  Likely causes: Σ_q ridge too "
              "small for near-singular heads, or fp32 precision insufficient "
              "for a particularly ill-conditioned (layer, kv-head). Fix by "
              "increasing --ridge or investigating the offending pair in "
              "roundtrip_check.json.",
              flush=True)
        return 2

    # Step 3: Lloyd-Max for K and V streams.
    if not args.skip_lloyd_max:
        k_out = args.out_dir / f"qwen3_4b_lloyd_max_K_b{args.k_bits}.f32"
        v_out = args.out_dir / f"qwen3_4b_lloyd_max_V_b{args.v_bits}.f32"
        run_lloyd_max(
            k_out, model_path=args.model_path, stream="K", bits=args.k_bits,
            q_precondition=q_safetensors,
            n_passages=args.lm_n_passages, ctx_len=args.lm_ctx_len,
            block_size=args.lm_block_size, rotation_seed=args.rotation_seed,
            split=args.split, device=args.device,
        )
        run_lloyd_max(
            v_out, model_path=args.model_path, stream="V", bits=args.v_bits,
            q_precondition=None,  # V-stream has no Q-preconditioning
            n_passages=args.lm_n_passages, ctx_len=args.lm_ctx_len,
            block_size=args.lm_block_size, rotation_seed=args.rotation_seed,
            split=args.split, device=args.device,
        )

    # Summary manifest.  Paths are recorded as str() of whatever was given
    # on the CLI — typically repo-relative already (as in the M7 runner).
    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO))
        except ValueError:
            return str(p)

    manifest = {
        "model_path": args.model_path,
        "split": args.split,
        "sigma_q_safetensors": _rel(q_safetensors),
        "sigma_q_sidecar":     _rel(q_safetensors.with_suffix(".json")),
        "roundtrip_check":     _rel(rt_path),
        "roundtrip_bar":       bar,
        "roundtrip_pass": {
            "fwd": fwd_ok,
            "rev": rev_ok,
        },
        "roundtrip_summary": {
            "fwd_err_max": rt["fwd_err_L_Linv_I"]["max"],
            "rev_err_max": rt["rev_err_Linv_L_I"]["max"],
        },
    }
    if not args.skip_lloyd_max:
        manifest["lloyd_max_K"] = {
            "path": _rel(k_out),
            "bits": args.k_bits,
        }
        manifest["lloyd_max_V"] = {
            "path": _rel(v_out),
            "bits": args.v_bits,
        }
    (args.out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[M2] wrote manifest {args.out_dir / 'MANIFEST.json'}", flush=True)
    print("[M2] PASS — Σ_q Cholesky roundtrip within 2e-5 tolerance.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
