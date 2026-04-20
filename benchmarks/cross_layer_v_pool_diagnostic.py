#!/usr/bin/env python3
"""Cross-layer V-skeleton pooling diagnostic — is path A viable?

Measures MSE inflation when the V-stream PCA basis is pooled across
ALL full-attention layers vs Sprint 3.5's current per-layer pooling.

Per-layer pooling (Sprint 3.5 baseline):
    for each layer l:
        V_l = cache.layers[l].values     # [seq, n_kv, D]
        basis_l = PCA(V_l flattened)
        MSE_perlayer[l] = reconstruct_and_measure(V_l, basis_l)

Cross-layer pooling (path A proposal):
    V_all = concat over layers (V_l for l in full_attn_layers)
    basis_global = PCA(V_all flattened)
    for each layer l:
        MSE_crosslayer[l] = reconstruct_and_measure(V_l, basis_global)

Inflation ratio = MSE_crosslayer / MSE_perlayer per layer.

If median inflation < 1.3, path A is likely OK.
If median inflation > 2.0, path A likely breaks PPL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import benchmarks.pre_rope_cache as prc


def fit_pca(X: np.ndarray, variance_ratio: float = 1.0,
            rank_cap: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Simple PCA fit. Returns (mean, basis [d_eff, D], d_eff).

    Matches KakeyaTurbo's fit_weighted_pca semantics at vr=1.0 (keep all
    meaningful components up to the numerical rank).  If `rank_cap` is set,
    d_eff is hard-capped at that value (after vr selection).
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    # Cov = X^T X / n
    cov = Xc.T @ Xc / X.shape[0]
    # Symmetric eig
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    total = max(float(evals.sum()), 1e-20)
    if variance_ratio >= 1.0:
        d_eff = X.shape[1]
    else:
        cum = np.cumsum(np.maximum(evals, 0.0)) / total
        d_eff = int(np.searchsorted(cum, variance_ratio) + 1)
        d_eff = max(1, min(d_eff, X.shape[1]))
    if rank_cap is not None:
        d_eff = min(d_eff, rank_cap)
    basis = evecs[:, :d_eff].T.astype(np.float32)  # [d_eff, D]
    return mean.astype(np.float32), basis, d_eff


def reconstruct_mse(X: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> float:
    """Project X - mean into basis then back; measure mean squared error."""
    Xc = X - mean
    # coeff = Xc @ basis^T   [n, d_eff]
    coeff = Xc @ basis.T
    # recon = coeff @ basis + mean
    recon = coeff @ basis
    err = Xc - recon
    return float(np.mean(err ** 2))


@torch.inference_mode()
def collect_v_tensors(model_path: str, n_passages: int, ctx_len: int):
    """Run model on n_passages WikiText-103 passages and collect per-layer
    V tensors (full-attention layers only).  Returns list of [n_vec, D] arrays
    indexed by layer_idx."""
    from benchmarks.e2e_ppl_pre_rope import load_wikitext_passages, prefill_cache
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.eval()
    prc.install(model)

    cfg = model.config.get_text_config(decoder=True)
    layer_types = getattr(cfg, "layer_types", None) or (
        ["full_attention"] * cfg.num_hidden_layers
    )
    n_layers = cfg.num_hidden_layers

    # Accumulate per-layer V across passages
    v_per_layer: dict[int, list[np.ndarray]] = {}
    passages = load_wikitext_passages(tok, ctx_len, n_passages)
    print(f"  got {len(passages)} passages", flush=True)

    for i, p in enumerate(passages):
        ids = tok(p, return_tensors="pt")["input_ids"][:, :ctx_len]
        cache = prefill_cache(model, ids, prefill_chunk=1024)
        for l in range(n_layers):
            if layer_types[l] != "full_attention":
                continue
            v = cache.layers[l].values  # [1, n_kv, seq, D]
            v_np = v[0].to(torch.float32).permute(1, 0, 2).cpu().numpy()  # [seq, n_kv, D]
            flat = v_np.reshape(-1, v.shape[-1]).astype(np.float32)
            v_per_layer.setdefault(l, []).append(flat)
        print(f"  passage {i+1}: collected V for {sum(1 for l in range(n_layers) if layer_types[l]=='full_attention')} full-attn layers",
              flush=True)

    # Concat within each layer
    out = {l: np.concatenate(arrs, axis=0) for l, arrs in v_per_layer.items()}
    return out, n_layers, layer_types


def run(model_path: str, n_passages: int = 2, ctx_len: int = 2048,
        skip_layers: list[int] | None = None):
    print(f"[{model_path}] loading + collecting V tensors…", flush=True)
    v_per_layer, n_layers, layer_types = collect_v_tensors(model_path, n_passages, ctx_len)

    D = next(iter(v_per_layer.values())).shape[1]
    full_attn_layers = sorted(v_per_layer.keys())
    if skip_layers:
        skip_set = set(skip_layers)
        full_attn_layers = [l for l in full_attn_layers if l not in skip_set]
        print(f"  skipping boundary layers {sorted(skip_set)}; "
              f"pooling over {len(full_attn_layers)} interior layers", flush=True)
    print(f"  D={D}, {len(full_attn_layers)} full-attention layers", flush=True)
    print(f"  per-layer V shape examples: "
          f"layer {full_attn_layers[0]} → {v_per_layer[full_attn_layers[0]].shape}",
          flush=True)

    # CRITICAL: we must truncate d_eff below D for the diagnostic to mean anything.
    # At vr=1.0 the basis spans the full space and reconstruction is perfect
    # (MSE = fp noise) regardless of which pool you fit on.  Use vr=0.95 to
    # match v1.2 ABLATION_REPORT and typical production config.
    import os
    VR_DIAG = float(os.environ.get("DIAG_VR", "0.95"))

    print(f"\n[step 1] fit per-layer pooled PCA at vr={VR_DIAG} (matches v1.2 ABLATION)…", flush=True)
    per_layer_bases = {}
    for l in full_attn_layers:
        mean_l, basis_l, d_eff_l = fit_pca(v_per_layer[l], variance_ratio=VR_DIAG)
        per_layer_bases[l] = (mean_l, basis_l, d_eff_l)
    d_effs = [per_layer_bases[l][2] for l in full_attn_layers]
    print(f"  per-layer d_eff: min={min(d_effs)}, median={int(np.median(d_effs))}, max={max(d_effs)}",
          flush=True)

    print(f"[step 2] fit cross-layer pooled PCA at vr={VR_DIAG} (path A proposal)…", flush=True)
    v_all = np.concatenate([v_per_layer[l] for l in full_attn_layers], axis=0)
    print(f"  pooled V shape: {v_all.shape}  (should be ~{len(full_attn_layers)}× per-layer)", flush=True)
    mean_g, basis_g, d_eff_g = fit_pca(v_all, variance_ratio=VR_DIAG)
    print(f"  cross-layer basis d_eff = {d_eff_g}", flush=True)

    print("\n[step 3] measure per-layer reconstruction MSE under both bases\n", flush=True)
    rows = []
    print(f"  {'L':>4} {'d_eff_l':>7} {'MSE_perlayer':>14} {'MSE_crosslayer':>16} {'inflation':>10}")
    for l in full_attn_layers:
        V_l = v_per_layer[l]
        mean_l, basis_l, d_eff_l = per_layer_bases[l]
        mse_pl = reconstruct_mse(V_l, mean_l, basis_l)
        mse_cl = reconstruct_mse(V_l, mean_g, basis_g)
        infl = mse_cl / max(mse_pl, 1e-20)
        rows.append({"layer": l, "d_eff_perlayer": d_eff_l,
                     "mse_per_layer": mse_pl,
                     "mse_cross_layer": mse_cl,
                     "inflation": infl})
        print(f"  {l:>4} {d_eff_l:>7}  {mse_pl:>12.4e}   {mse_cl:>14.4e}   {infl:>8.2f}×")

    infl_arr = np.array([r["inflation"] for r in rows])
    print(f"\n[summary] MSE inflation (cross-layer / per-layer) across {len(rows)} layers:")
    print(f"  min:    {infl_arr.min():6.2f}×")
    print(f"  median: {np.median(infl_arr):6.2f}×")
    print(f"  mean:   {infl_arr.mean():6.2f}×")
    print(f"  p90:    {np.percentile(infl_arr, 90):6.2f}×")
    print(f"  max:    {infl_arr.max():6.2f}×")

    print(f"\n[verdict]")
    med = float(np.median(infl_arr))
    if med < 1.3:
        print(f"  median inflation {med:.2f}× < 1.3  → Path A LIKELY OK, proceed to full Rust implementation")
    elif med < 2.0:
        print(f"  median inflation {med:.2f}× in 1.3-2.0 → MARGINAL, consider Tier 2 affine corrector to compensate")
    else:
        print(f"  median inflation {med:.2f}× > 2.0  → Path A is NOT VIABLE, abandon in favour of Path B")

    # PPL estimate via corr(log Δppl, log V-MSE) = 0.55
    # Sprint 3.5 baseline Δppl = -1.68%  →  log(1 + (-0.0168)) ≈ -0.0169
    # Δ log V-MSE = log(inflation)
    # Predicted Δlog(1+Δppl) ≈ 0.55 × log(inflation)
    # Predicted Δppl_new ≈ exp(-0.0169 + 0.55 × log(inflation)) - 1
    base_log = np.log(1 - 0.0168)
    for pct, label in [(50, "median"), (75, "p75"), (90, "p90")]:
        inf = float(np.percentile(infl_arr, pct))
        predicted_log = base_log + 0.55 * np.log(inf)
        predicted_delta = float(np.exp(predicted_log) - 1)
        print(f"  predicted Δppl at {label} V-MSE inflation ({inf:.2f}×): {predicted_delta*100:+6.2f}%")

    # Save
    out = {
        "model_path": model_path,
        "n_passages": n_passages,
        "ctx_len": ctx_len,
        "D": D,
        "n_full_attention_layers": len(full_attn_layers),
        "rows": rows,
        "summary": {
            "min": float(infl_arr.min()),
            "median": float(np.median(infl_arr)),
            "mean": float(infl_arr.mean()),
            "p90": float(np.percentile(infl_arr, 90)),
            "max": float(infl_arr.max()),
        },
    }
    return out


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--n-passages", type=int, default=2)
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-layers", type=int, nargs="+", default=None,
                    help="Full-attention layers to exclude from both per-layer "
                         "and cross-layer pools (e.g. boundary layers kept at "
                         "higher precision).")
    args = ap.parse_args()

    out = run(args.model_path, args.n_passages, args.ctx_len,
              skip_layers=args.skip_layers)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
