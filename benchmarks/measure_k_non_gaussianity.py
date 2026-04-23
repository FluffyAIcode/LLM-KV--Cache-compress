"""Phase 1 measurement: how non-Gaussian is Qwen3-4B's post-qk-norm K
after Hadamard rotation?

The full TurboQuant code path assumes that `y = Hx̂` (x̂ = K/‖K‖,
H = Sylvester Hadamard / √D) gives per-coordinate distributions that
are approximately 𝒩(0, 1/D).  This assumption is the foundation for
using the Gaussian-optimal Lloyd-Max codebook.  It is tight for
i.i.d. Gaussian inputs and relaxes on the sphere by O(1/√D) per
Stein's method.  For strictly i.i.d. Gaussian sources, Lloyd-Max is
optimal within 1.53 dB of Shannon's rate-distortion bound — and no
Kakeya-style construction (Wang-Zahl 2025, etc.) can beat it.

But LLM K is not strictly i.i.d. Gaussian.  Any systematic deviation
opens a window for non-Gaussian shaping / Kakeya-inspired codebooks
to beat the Shannon i.i.d. ceiling.  This script measures FOUR such
deviations on captured Qwen3-4B K, producing a report that serves as
the decision gate for research path (vi) in
`reports/v1_3_ppl/vllm_backend/HANDOFF.md` §5.8:

  Metric 1 — per-dim kurtosis        (Gaussian = 3)
  Metric 2 — per-dim Wasserstein-2   (to 𝒩(0, 1/D))
  Metric 3 — score function deviation  (Gaussian score(y) = −y · D)
  Metric 4 — isotropy of Hadamard output (Var_i should ≈ 1/D flat)

Decision gate:
  • All four show deviation within noise → K is i.i.d. Gaussian under
    Hadamard, no Kakeya-style savings > 1.53 dB available; stop.
  • Any metric shows systematic deviation → quantify; decide between
    research path (i) — (vi) based on WHICH metric deviates and by
    how much.

Reproduces the capture protocol from the snapshot harness verbatim
so that measurements are comparable to snapA/snapF Δppl numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Environment setup — match the snapshot harness exactly.
# ---------------------------------------------------------------------------
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("KAKEYA_SNAPSHOT_QWEN3", "1")
os.environ.setdefault("KAKEYA_DISABLE_SIGMA_Q", "1")
os.environ.setdefault("KAKEYA_USE_M2_CENTROIDS", "0")
os.environ.setdefault("KAKEYA_OUTLIER_THRESHOLD", "0")


# ---------------------------------------------------------------------------
# Hadamard rotation (Sylvester-normalised) — matches TurboQuant semantics.
# ---------------------------------------------------------------------------
def hadamard_normalised(D: int, device: str = "cuda") -> torch.Tensor:
    """Sylvester Hadamard H_D divided by √D, so H @ H.T = I.

    This is TQ's actual rotation:
    `turboquant_attn.py:74-79` — `(H / math.sqrt(d))`.
    """
    assert (D & (D - 1)) == 0, f"D must be power of 2, got {D}"
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat([
            torch.cat([H,  H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / math.sqrt(D)


# ---------------------------------------------------------------------------
# Non-Gaussianity metrics.
# ---------------------------------------------------------------------------
def per_dim_kurtosis(y: torch.Tensor) -> torch.Tensor:
    """Excess kurtosis per dimension.

    For y ~ 𝒩(0, σ²), kurtosis `E[y⁴] / E[y²]² = 3`.  We report the
    raw kurtosis (not excess); deviations from 3 indicate non-Gaussian
    marginals.

    Args:
        y: [N, D] samples, rows are independent draws.
    Returns:
        [D] kurtosis per coordinate.
    """
    mean = y.mean(dim=0, keepdim=True)
    centred = y - mean
    var = (centred * centred).mean(dim=0)                 # [D]
    fourth = (centred ** 4).mean(dim=0)                    # [D]
    eps = torch.finfo(y.dtype).eps
    return fourth / torch.clamp(var * var, min=eps)        # kurtosis, Gaussian = 3


def per_dim_wasserstein2_to_normal(
    y: torch.Tensor,
    target_var: float,
) -> torch.Tensor:
    """Squared Wasserstein-2 distance from empirical marginal to 𝒩(0, target_var),
    per dimension.

    W_2²(p, q) for 1-D distributions has a closed form via inverse CDFs.
    We use the standard-normal quantile sample approach:
        sort empirical samples and compare pointwise to quantiles of
        𝒩(0, target_var) evaluated at the same ranks.

    Args:
        y: [N, D] samples.
        target_var: 1/D for TurboQuant's unit-sphere rotated coords.
    Returns:
        [D] W_2² per coordinate.
    """
    N, D = y.shape
    target_sigma = math.sqrt(target_var)
    # Quantile positions at rank (k + 0.5) / N for k = 0..N-1
    # — matches torch.distributions.Normal.icdf convention.
    ranks = (torch.arange(N, device=y.device, dtype=torch.float32) + 0.5) / N
    # icdf of standard normal via inverse error function.
    gauss_quantiles = target_sigma * (
        torch.erfinv(2.0 * ranks - 1.0) * math.sqrt(2.0)
    )                                                      # [N]
    sorted_y, _ = y.sort(dim=0)                            # [N, D]
    diff = sorted_y - gauss_quantiles.unsqueeze(1)         # [N, D]
    return (diff * diff).mean(dim=0)                        # [D]


def score_function_deviation(
    y: torch.Tensor,
    target_var: float,
    bandwidth: float = 0.1,
) -> torch.Tensor:
    """Average per-dim |score(y) − score_gauss(y)|² measured on the
    training samples themselves.

    For 𝒩(0, target_var), score(y) = −y / target_var.  For arbitrary p,
    score = ∇ log p.  We estimate ∇ log p̂ via Gaussian KDE score
    matching: pick a bandwidth h, compute

        ŝ(y_i) = (1/N) Σ_j (y_j − y_i) / h² · K_h(y_i − y_j) / p̂(y_i)

    on a sub-sampled batch to keep it tractable.  Output is the RMSE
    per dim between ŝ(y_i) and s_gauss(y_i) = −y_i / target_var,
    averaged across the sub-sampled points.

    Large deviation → K's rotated coordinates have non-Gaussian density
    shape (not just tail kurtosis but body shape).

    Args:
        y: [N, D] samples.
        target_var: expected Gaussian variance per coordinate.
        bandwidth: KDE bandwidth; default 0.1 is tight enough for smooth
                   densities without over-smoothing narrow modes.
    Returns:
        [D] RMS score deviation per coordinate.
    """
    N, D = y.shape
    # Sub-sample for tractable KDE.  Score estimation is O(n²) in n_sub.
    n_sub = min(N, 2048)
    idx = torch.randperm(N, device=y.device)[:n_sub]
    y_sub = y[idx]                                          # [n_sub, D]

    h = float(bandwidth)
    h2 = h * h

    # pairwise squared distances per dim
    #   diff[i, j, d] = y_sub[j, d] - y_sub[i, d]
    # To avoid memory blow-up at D=128 and n_sub=2048 we iterate dims.
    dev_sq = torch.zeros(D, device=y.device, dtype=torch.float32)
    for d in range(D):
        col = y_sub[:, d:d + 1]                              # [n_sub, 1]
        diff = col - col.transpose(0, 1)                    # [n_sub, n_sub]
        w = torch.exp(-0.5 * diff * diff / h2)              # unnormalised
        # Row-normalise to get KDE weights; the leading constants cancel.
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-12)
        w_norm = w / w_sum                                   # [n_sub, n_sub]
        # Score estimate at y_sub[i, d] along coordinate d:
        #   ŝ_d(y_i) = Σ_j w_ij · (y_j,d − y_i,d) / h²
        score_est = (w_norm * diff).sum(dim=1) / h2          # [n_sub]
        score_gauss = -y_sub[:, d] / target_var              # [n_sub]
        dev_sq[d] = ((score_est - score_gauss) ** 2).mean()
    return dev_sq.sqrt()                                    # [D] RMS dev


def isotropy_flatness(y: torch.Tensor) -> dict:
    """How flat is the per-dim variance profile of the Hadamard output?

    For a truly i.i.d. 𝒩(0, 1/D) source, per-dim variance is constant at
    1/D.  Any systematic anisotropy (e.g. if Hadamard-rotated K still
    has structure along the first few coordinates) shows here.
    """
    var_per_dim = y.var(dim=0)                              # [D]
    return {
        "var_mean": float(var_per_dim.mean().item()),
        "var_std":  float(var_per_dim.std().item()),
        "var_min":  float(var_per_dim.min().item()),
        "var_max":  float(var_per_dim.max().item()),
        "var_ratio_max_to_min": float(
            (var_per_dim.max() / var_per_dim.min().clamp(min=1e-12)).item()
        ),
    }


# ---------------------------------------------------------------------------
# Capture protocol — reuse the snapshot harness's hook.
# ---------------------------------------------------------------------------
def capture_qwen3_k(
    model_path: str,
    n_passages: int,
    ctx_len: int,
    gpu_mem_util: float,
) -> dict[int, np.ndarray]:
    """Run Qwen3-4B on n_passages from WikiText-103 test, capture K per
    non-boundary layer via the existing snapshot hook.

    Returns {layer_id: K_np of shape [n_tokens_total, n_kv_heads, head_dim]}.
    """
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(model_path)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    # Match the snapshot harness's passage selection: take consecutive
    # 2048-token chunks from the concatenated corpus.
    joined = "\n\n".join(ds["text"])
    full_ids = tok(joined, return_tensors="pt").input_ids[0].tolist()
    passages = []
    stride = ctx_len
    for k in range(n_passages):
        start = k * stride
        end = start + ctx_len
        if end > len(full_ids):
            break
        passages.append(full_ids[start:end])
    assert len(passages) == n_passages, (
        f"only got {len(passages)} passages, need {n_passages}"
    )

    llm = LLM(
        model=model_path, max_model_len=ctx_len + 1,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True, enable_prefix_caching=False,
    )
    from kakeya_v1_3_ppl.snapshot_hook import HookState
    HookState.phase = "capture"

    # accumulate across passages per layer
    accum: dict[int, list[np.ndarray]] = {}
    for p_idx, ids in enumerate(passages):
        HookState.captured.clear()
        _ = llm.generate(
            [TokensPrompt(prompt_token_ids=ids)],
            SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1),
        )
        for lid, kv in HookState.captured.items():
            accum.setdefault(lid, []).append(np.asarray(kv["K"], dtype=np.float32))
        print(
            f"  [capture] passage {p_idx + 1}/{n_passages}: "
            f"{len(HookState.captured)} layers captured"
        )
    return {lid: np.concatenate(arrs, axis=0) for lid, arrs in accum.items()}


# ---------------------------------------------------------------------------
# Per-layer measurement driver.
# ---------------------------------------------------------------------------
def measure_layer(
    K_np: np.ndarray,
    layer_id: int,
    device: str = "cuda",
) -> dict:
    """K_np: [n_tokens, n_kv_heads, head_dim] — captured post-qk-norm K.

    Normalise per-vector, rotate by Hadamard, compute the four metrics
    on the flattened (token × kv_head) sample population.
    """
    n_tokens, H, D = K_np.shape
    target_var = 1.0 / D

    K_t = torch.from_numpy(K_np).to(device).float()                # [n, H, D]
    # Flatten (token, head) into independent samples — this is what
    # the TQ per-(token, head) quantiser operates on.
    flat = K_t.reshape(n_tokens * H, D)                             # [N, D]

    # Per-vector unit-normalise (as TQ does).
    norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    x_hat = flat / norms                                            # [N, D]

    # Hadamard-rotate.
    Hmat = hadamard_normalised(D, device=device)                    # [D, D]
    y = x_hat @ Hmat                                                # [N, D]

    # --- Metric 1: kurtosis per dim ---
    kurt = per_dim_kurtosis(y)                                      # [D]

    # --- Metric 2: W_2² per dim ---
    w2_sq = per_dim_wasserstein2_to_normal(y, target_var=target_var)  # [D]

    # --- Metric 3: score-function RMS dev per dim ---
    # Sub-sample to keep compute reasonable — score estimator is O(n_sub²).
    score_dev = score_function_deviation(y, target_var=target_var)   # [D]

    # --- Metric 4: isotropy of variance profile ---
    iso = isotropy_flatness(y)

    out = {
        "layer":             int(layer_id),
        "n_samples":         int(y.shape[0]),
        "head_dim":          int(D),
        "target_var":        float(target_var),
        "kurtosis": {
            "mean":  float(kurt.mean().item()),
            "std":   float(kurt.std().item()),
            "min":   float(kurt.min().item()),
            "max":   float(kurt.max().item()),
            # Gaussian = 3.  Deviations report as (kurt - 3).
            "excess_mean": float((kurt - 3.0).mean().item()),
            "excess_abs_max": float((kurt - 3.0).abs().max().item()),
        },
        "w2_sq": {
            # Report RMS Wasserstein distance (not squared) in units of
            # target std, so values are dimensionless and comparable
            # across dimensions.
            "rms_w2_in_sigma": float(
                (w2_sq.sqrt() / math.sqrt(target_var)).mean().item()
            ),
            "max_w2_in_sigma": float(
                (w2_sq.sqrt() / math.sqrt(target_var)).max().item()
            ),
        },
        "score_dev": {
            "rms_mean": float(score_dev.mean().item()),
            "rms_max":  float(score_dev.max().item()),
            # Normalise by Gaussian score magnitude 1/target_var = D so
            # the number is dimensionless ∈ (0, ∞); Gaussian = 0.
            "rel_mean": float(
                (score_dev * target_var).mean().item()
            ),
            "rel_max": float(
                (score_dev * target_var).max().item()
            ),
        },
        "isotropy": iso,
    }
    return out


# ---------------------------------------------------------------------------
# Gate decision.
# ---------------------------------------------------------------------------
# Thresholds chosen conservatively:
#  * |excess kurtosis| > 0.5 : tails clearly heavier/lighter than Gaussian,
#    a regime where Lloyd-Max's fixed Gaussian-optimal codebook mis-aligns
#    with the true density by enough to matter.
#  * mean W_2 / σ > 0.05     : distributional shape differs from Gaussian
#    by > 5 % in earth-mover's distance.
#  * relative score dev > 0.1 : density gradient differs by > 10 % from
#    the Gaussian score, which directly controls quantiser efficiency.
#  * var_ratio > 1.5         : Hadamard fails to isotropise per-dim variance.
KURT_GATE           = 0.5
W2_GATE             = 0.05
SCORE_GATE          = 0.10
ISO_RATIO_GATE      = 1.50


def gate_decision(layer_metrics: list[dict]) -> dict:
    """Aggregate per-layer metrics into a binary go/no-go decision."""
    kurt_excess = np.array([m["kurtosis"]["excess_mean"] for m in layer_metrics])
    w2_mean     = np.array([m["w2_sq"]["rms_w2_in_sigma"] for m in layer_metrics])
    score_rel   = np.array([m["score_dev"]["rel_mean"] for m in layer_metrics])
    iso_ratio   = np.array([m["isotropy"]["var_ratio_max_to_min"] for m in layer_metrics])

    triggered = []
    if np.abs(kurt_excess).max() > KURT_GATE:
        triggered.append(
            f"kurtosis: excess |kurt - 3|_max = "
            f"{np.abs(kurt_excess).max():.3f} > {KURT_GATE}"
        )
    if w2_mean.max() > W2_GATE:
        triggered.append(
            f"W_2: max per-layer mean-over-dim W_2/σ = "
            f"{w2_mean.max():.3f} > {W2_GATE}"
        )
    if score_rel.max() > SCORE_GATE:
        triggered.append(
            f"score: max per-layer mean-over-dim rel score dev = "
            f"{score_rel.max():.3f} > {SCORE_GATE}"
        )
    if iso_ratio.max() > ISO_RATIO_GATE:
        triggered.append(
            f"isotropy: max per-layer variance ratio (max/min) = "
            f"{iso_ratio.max():.3f} > {ISO_RATIO_GATE}"
        )

    non_gaussian = len(triggered) > 0
    verdict = (
        "NON-GAUSSIAN — Kakeya-style / non-Gaussian shaping has "
        "measurable space above Shannon i.i.d. Gaussian bound. "
        "Research paths (iii), (iv), (vi) from HANDOFF §5.8 are "
        "worth pursuing."
        if non_gaussian else
        "GAUSSIAN — K is i.i.d. Gaussian under Hadamard to within "
        "measurement noise.  No Kakeya-style construction can beat "
        "Shannon + 1.53 dB; TurboQuant is already at the ceiling for "
        "this workload.  Focus engineering effort on porting snapF to "
        "the slot path, not on Kakeya-style research."
    )

    return {
        "verdict":          verdict,
        "is_non_gaussian":  bool(non_gaussian),
        "triggered_gates":  triggered,
        "gate_thresholds": {
            "kurtosis_abs_excess_max":    KURT_GATE,
            "w2_rms_over_sigma_max":      W2_GATE,
            "score_rel_dev_mean_max":     SCORE_GATE,
            "isotropy_variance_ratio_max":ISO_RATIO_GATE,
        },
        "summary_stats": {
            "kurtosis_excess_abs_max": float(np.abs(kurt_excess).max()),
            "w2_rms_over_sigma_max":   float(w2_mean.max()),
            "score_rel_dev_max":       float(score_rel.max()),
            "isotropy_var_ratio_max":  float(iso_ratio.max()),
        },
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-passages", type=int, default=4,
                    help="WikiText-103 test passages to capture.")
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--boundary-skip-layers", type=int, nargs="*",
                    default=[0, 1, 2, 3, 4, 5, 6, 29, 30, 31, 32, 33, 34, 35],
                    help="Layers to exclude from measurement — must match "
                         "the snapshot-harness skip set so conclusions "
                         "apply to the SAME layers that codec runs on.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(args.boundary_skip_layers)

    print(f"[setup] capturing {args.n_passages} × {args.ctx_len}-token "
          f"passages of Qwen3-4B post-qk-norm K …")
    t0 = time.perf_counter()
    captured = capture_qwen3_k(
        args.model_path, args.n_passages, args.ctx_len, args.gpu_mem_util,
    )
    print(f"[capture] done in {time.perf_counter() - t0:.1f}s, "
          f"{len(captured)} layers captured")

    layer_ids = sorted(l for l in captured if l not in skip)
    print(f"[measure] running on {len(layer_ids)} non-boundary layers: "
          f"{layer_ids}")

    layer_metrics = []
    for lid in layer_ids:
        t0 = time.perf_counter()
        m = measure_layer(captured[lid], lid)
        dt = time.perf_counter() - t0
        print(
            f"  layer {lid:>2}: "
            f"kurt={m['kurtosis']['mean']:5.3f} "
            f"(Δ={m['kurtosis']['excess_mean']:+.3f})  "
            f"W_2/σ={m['w2_sq']['rms_w2_in_sigma']:5.3f}  "
            f"score_rel={m['score_dev']['rel_mean']:5.3f}  "
            f"var_ratio={m['isotropy']['var_ratio_max_to_min']:5.2f}  "
            f"[{dt:.1f}s]"
        )
        layer_metrics.append(m)

    decision = gate_decision(layer_metrics)

    print("\n" + "=" * 72)
    print("GATE DECISION")
    print("=" * 72)
    print(decision["verdict"])
    if decision["triggered_gates"]:
        print("\nTriggered gates:")
        for g in decision["triggered_gates"]:
            print(f"  • {g}")
    else:
        print("\nNo gates triggered — all four metrics within Gaussian tolerance.")
    print()

    out = {
        "model": args.model_path,
        "n_passages": args.n_passages,
        "ctx_len":   args.ctx_len,
        "boundary_skip_layers": sorted(skip),
        "per_layer_metrics": layer_metrics,
        "decision": decision,
    }
    out_path = args.out_dir / "qwen3_4b_k_non_gaussianity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[done] written → {out_path}")


if __name__ == "__main__":
    main()
