#!/usr/bin/env python3
"""Tikhonov-regularise a Σ_q calibration bundle: Σ_q_reg = Σ_q + λ·I.

Why
---
The snapshot-mode harness on Qwen3-4B showed +8 859% Δppl when Σ_q
whitening was active, vs +619% when it was off (see
`reports/v1_3_ppl/snapshot_mode_qwen3/QWEN3_SCENARIO_A_REPORT.md`).
Root cause: the saved Cholesky factors `L` (with `Σ_q = L Lᵀ`) on
Qwen3-4B have median cond(L) = 252 across (layer, head).  Unwhitening
at decode time (K_hat = K̂_tilde @ L⁻¹) amplifies codec error by the
spectral norm of L⁻¹, which is the condition number of L.  On a
model whose Σ_q eigenspectrum is more spread than DeepSeek-1.5B
(cond ≈ 65), this kills PPL.

Fix
---
Shrink Σ_q toward isotropic by adding λ·I, with λ chosen per
(layer, head) so that:

    cond(L_reg) ≤ target_cond

Since `cond(L_reg) = sqrt((λ_max + λ) / (λ_min + λ))`, the smallest
λ that satisfies the cap is:

    λ ≥ (λ_max - C² · λ_min) / (C² - 1),   C = target_cond

This is exactly a Tikhonov shrinkage of the query-weighted MSE
objective the codec minimises: the codec loss in whitened space
becomes `(K̂−K)ᵀ (Σ_q + λ·I) (K̂−K)` — i.e. the original Σ_q-weighted
MSE plus a small Euclidean-MSE regulariser.  No model re-run needed:
the reg only touches saved tensors.

Output
------
`<input stem>_reg<C>.safetensors` + matching sidecar `.json`,
consumable by `benchmarks/q_precondition.QPrecond` unchanged
(same schema, same dtype, same tensor names).

Usage
-----
    python benchmarks/q_regularize_sigma.py \\
        --input reports/v1_3_ppl/vllm_backend/calibration/qwen3_4b_sigma_q.safetensors \\
        --target-cond 50

    # output: ...qwen3_4b_sigma_q_reg50.safetensors  (+ .json)
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def _cond_of_L(L: np.ndarray) -> float:
    """cond(L) where L is a single [D, D] lower-triangular matrix."""
    # cond(L) via SVD (exact, no accumulated fp error from chained matmuls).
    return float(np.linalg.cond(L))


def _choose_lambda(sigma: np.ndarray, target_cond: float) -> float:
    """Smallest λ ≥ 0 such that sqrt((λ_max + λ) / (λ_min + λ)) ≤ target_cond.

    `sigma` is a [D, D] symmetric PSD matrix (reconstructed from L @ Lᵀ).
    Returns λ in the same units as sigma's eigenvalues.  Always ≥ 0.
    """
    # Use eigh on the symmetrised Σ so negatives from numerical drift
    # at the tail don't poison the ratio.
    evals = np.linalg.eigvalsh(0.5 * (sigma + sigma.T))
    lam_min = float(max(evals[0], 0.0))
    lam_max = float(evals[-1])
    C2 = float(target_cond) ** 2
    # Need: (lam_max + λ) / (lam_min + λ) ≤ C²
    # ⇔ λ_max + λ ≤ C² · λ_min + C² · λ
    # ⇔ λ_max - C² · λ_min ≤ (C² - 1) · λ
    # ⇔ λ ≥ (λ_max - C² · λ_min) / (C² - 1)
    num = lam_max - C2 * lam_min
    if num <= 0.0:
        return 0.0       # already well-conditioned; no shrinkage needed
    return num / (C2 - 1.0)


def _regularise_factor(L: np.ndarray, target_cond: float
                       ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Regularise a single [D, D] Cholesky factor.

    Returns (L_reg, L_reg_inv, lambda, cond_before, cond_after).
    """
    D = L.shape[0]
    # Σ = L @ Lᵀ (exact reconstruction of the calibration Gram matrix).
    sigma = L @ L.T
    cond_before = _cond_of_L(L)
    lam = _choose_lambda(sigma, target_cond)
    sigma_reg = sigma + lam * np.eye(D, dtype=sigma.dtype)
    # Refactor.  `numpy.linalg.cholesky` returns lower-triangular by
    # default (matches the calibration layout).
    L_reg = np.linalg.cholesky(sigma_reg)
    L_reg_inv = np.linalg.solve(L_reg, np.eye(D, dtype=sigma.dtype))
    # Double-check: L_reg is lower tri, so solve gives a lower tri inv —
    # but up to fp32 rounding.  QPrecond doesn't require strict
    # triangularity for the einsum whiten/unwhiten ops.
    cond_after = _cond_of_L(L_reg)
    return (
        L_reg.astype(np.float32),
        L_reg_inv.astype(np.float32),
        float(lam),
        cond_before,
        cond_after,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
        help="Existing Σ_q bundle (safetensors).  Must have a sibling "
             ".json sidecar in the QPrecond schema.")
    ap.add_argument("--target-cond", type=float, default=50.0,
        help="Upper bound on cond(L_reg).  Default 50 = matches the "
             "DeepSeek-1.5B MARGINAL regime (median cond 65, max 235).")
    ap.add_argument("--output", type=Path, default=None,
        help="Explicit output path.  Default: "
             "<input stem>_reg<cond>.safetensors next to the input.")
    args = ap.parse_args()

    in_st = args.input
    in_js = in_st.with_suffix(".json")
    assert in_st.is_file(), f"input {in_st} not found"
    assert in_js.is_file(), f"sidecar {in_js} not found"

    cfg = json.loads(in_js.read_text())
    n_layers = int(cfg["num_layers"])
    head_dim = int(cfg["head_dim"])
    num_kv = int(cfg["num_kv_heads"])
    tensors = load_file(str(in_st))
    print(f"[input] n_layers={n_layers}, head_dim={head_dim}, "
          f"num_kv_heads={num_kv}, target_cond={args.target_cond}", flush=True)

    out: dict[str, torch.Tensor] = {}
    # Diagnostics
    conds_before, conds_after, lambdas = [], [], []
    layers_regularised = 0

    for l in range(n_layers):
        k_chol = f"layer_{l}_chol"
        k_inv = f"layer_{l}_inv_chol"
        if k_chol not in tensors:
            continue
        L_all = tensors[k_chol].numpy().astype(np.float64)   # [n_kv, D, D]
        L_reg_all = np.empty_like(L_all, dtype=np.float32)
        Linv_reg_all = np.empty_like(L_all, dtype=np.float32)
        any_touched = False
        for h in range(num_kv):
            L_reg, L_reg_inv, lam, cb, ca = _regularise_factor(
                L_all[h], args.target_cond,
            )
            L_reg_all[h] = L_reg
            Linv_reg_all[h] = L_reg_inv
            conds_before.append(cb)
            conds_after.append(ca)
            lambdas.append(lam)
            if lam > 0.0:
                any_touched = True
        if any_touched:
            layers_regularised += 1
        out[k_chol] = torch.from_numpy(L_reg_all)
        out[k_inv] = torch.from_numpy(Linv_reg_all)

    # Write output.
    if args.output is None:
        stem = in_st.stem
        cond_tag = f"{int(args.target_cond)}"
        out_st = in_st.with_name(f"{stem}_reg{cond_tag}.safetensors")
    else:
        out_st = args.output
    out_js = out_st.with_suffix(".json")
    save_file(out, str(out_st))

    out_cfg = dict(cfg)
    out_cfg["regularisation"] = {
        "source": str(in_st.name),
        "scheme": "tikhonov",
        "target_cond": float(args.target_cond),
        "lambda_mean": float(statistics.mean(lambdas)) if lambdas else 0.0,
        "lambda_median": float(statistics.median(lambdas)) if lambdas else 0.0,
        "lambda_max": float(max(lambdas)) if lambdas else 0.0,
        "n_layers_regularised": int(layers_regularised),
        "cond_before_median": float(statistics.median(conds_before)),
        "cond_before_max": float(max(conds_before)),
        "cond_after_median": float(statistics.median(conds_after)),
        "cond_after_max": float(max(conds_after)),
    }
    out_js.write_text(json.dumps(out_cfg, indent=2))

    # Report.
    print(f"[output] wrote {out_st} ({out_st.stat().st_size / 1024 / 1024:.1f} MiB)", flush=True)
    print(f"         sidecar {out_js}", flush=True)
    print(f"  cond(L) before: median={statistics.median(conds_before):.2f}  "
          f"max={max(conds_before):.2f}", flush=True)
    print(f"  cond(L) after:  median={statistics.median(conds_after):.2f}  "
          f"max={max(conds_after):.2f}", flush=True)
    print(f"  λ:              median={statistics.median(lambdas):.3e}  "
          f"max={max(lambdas):.3e}", flush=True)
    print(f"  {layers_regularised}/{sum(1 for l in range(n_layers) if f'layer_{l}_chol' in tensors)} "
          f"layers required non-zero λ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
