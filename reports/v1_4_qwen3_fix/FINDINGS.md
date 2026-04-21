# Qwen3-0.6B retuning — final answers

**Date.** 2026-04-17
**Question from user.** Fix the Qwen3 failure (+39-80% Δppl) identified
in the v1.4 multi-model validation sprint; produce concrete numbers.

**Answer.** Qwen3-0.6B's K-stream is **structurally incompatible** with
any v1.4 K-compression configuration we tested (29 cells total). The
one deployable recipe is **K=bf16 (no compression) + V=Besi d=3 m=4
+mean**: **1.73× ratio, Δppl=−0.25%, top-1=95.24% ACCEPT ★**.

## Isolation diagnostics (2-passage each)

| Cell | Config                                  | Δppl      | top-1   | Interpretation |
|:----:|:----------------------------------------|----------:|--------:|:---------------|
| A    | K-only b=4, **no Q-precond**            | **+12 762 %** | 18.25 % | K codec alone is catastrophic |
| B    | V-only b=2 share, no Q-precond          | **+0.19 %**   | 92.06 % | **V is perfectly fine** |
| C    | K-only b=4 + Q-precond                  | +91.34 %      | 59.52 % | Q-precond → 140× improvement but still REJECT |
| D    | K-only b=6 (intended) → failed (bit_width max=4) | N/A | N/A | Codec doesn't support b>4 |

**Diagnosis**: K compression is the sole failure mode. V compression is
not the issue.

## Per-layer K-MSE analysis

Compared K-only b=4 MSE under Q-precond (whitened space) vs raw
(no-Q-precond) across all 28 Qwen3 layers. Q-precond **hurt 20/28
layers** on Qwen3 (ratio >1), vs DS-Distill where Q-precond
systematically helps. Full table in commit notes.

Root cause: Qwen3 applies `RMSNorm(q)` and `RMSNorm(k)` **before RoPE**.
This gives K very specific norm structure per head. Q-precond's
whitening assumes raw K distribution; on already-normed K it **disturbs**
the head-level isotropy. Σ_q condition number on Qwen3 is 66 035 (vs
DS-Distill ~2 900) — Cholesky is near-singular, and whitening amplifies
the very directions RMSNorm was trying to suppress.

## Retune attempts (all fail to ACCEPT on full K+V compression)

| Attempt | Config | Δppl | top-1 | Verdict |
|---------|--------|-----:|------:|:-------:|
| Baseline | K b=4 + V Kakeya b=2 share + Q-precond (4 passages) | +39.50 % | 70.63 % | REJECT |
| Pareto | K b=4 + V Besi d=3 m=4 +mean | +80.22 % | 67.86 % | REJECT |
| R1 | **no Q-precond**, K b=4 + V b=2 share (2 pass) | +23 818 % | 15.08 % | DISASTER |
| R2 | no Q-precond + wider bf16 boundary (8 layers) | +1 250 % | 43.65 % | DISASTER |
| R3 | no Q-precond + V=Besi | +8 157 % | 19.05 % | DISASTER |
| F1 | Q-precond + 8 bf16 boundary layers (2 pass) | +75.86 % | 66.67 % | REJECT |
| F2 | Q-precond skip 17 noisy layers | +39.75 % | 74.60 % | REJECT |
| F3 | Q-precond + 8 bf16 boundary layers (different) | +41.26 % | 68.25 % | REJECT |
| — | 12 bf16 boundary + V=Besi | +24.09 % | 78.17 % | REJECT |
| — | 16 bf16 boundary + V=Besi | +30.04 % | 76.19 % | REJECT |
| — | V=Besi m=3, m=4, f16 (all variants) | +70-84 % | 63-72 % | REJECT |
| — | K b=4 + V b=3 share (less aggressive V) | +49.62 % | 74.60 % | REJECT |
| — | K b=4 + V b=4 share (near-lossless V) | +84.58 % | 65.10 % | REJECT |

**Notable**: removing Q-precond makes things catastrophically worse
(+23 818 %). Q-precond is doing real work, but cannot rescue K
compression alone. More bf16 boundary layers (up to 16) helps but
never reaches ACCEPT.

## The working config: V-only Besicovitch

```
--compress v_only --codec-v besicovitch
--besi-group-size 2 --besi-direction-bits 3 --besi-magnitude-bits 4
--besi-magnitude-mode quantized --besi-subtract-mean
```

| Metric | Value |
|---|---|
| Compression ratio (K stays bf16) | **1.73×** |
| Δppl (4 passages, WikiText-103) | **−0.25 %** (compression *improves* PPL) |
| top-1 | **95.24 %** (best top-1 across any Qwen3 config tested) |
| Verdict | **ACCEPT ★ 🏆** |

## Why Qwen3 is different

Architectural comparison for the three models we benchmarked:

| Model            | pre-RoPE norm? | K `|max|` / std | Σ_q cond | GQA ratio | Result |
|------------------|:---------------:|----------------:|---------:|----------:|:-------|
| DS-Distill 1.5B  | no              | 25              | 2 937    | 6:1       | Pareto WIN |
| GLM-edge 1.5b    | no              | ~25             | 1 871    | 4:1       | Pareto WIN |
| Qwen3-0.6B       | **yes (q/k)**   | **>30**         | **66 035** | 2:1     | K-compress fails |

Qwen3's combination of (i) pre-RoPE q/k-norm, (ii) heavier K tails,
(iii) near-singular Σ_q, (iv) tighter GQA ratio, and (v) smaller model
(0.6B — every bit of KV fidelity matters more) makes K compression
**~20× more sensitive** than on Qwen2/GLM-family models.

## Production matrix update

| Model family | Recommended v1.4 config | Ratio | Quality |
|--------------|-------------------------|------:|:--------|
| Qwen2 (DS-Distill, Qwen2.5) | K Kakeya b=4 + V Besi d=3 m=4 +mean | 2.97× | Δppl −2.04 %, top-1 91.27 % |
| GLM (edge, base) | K Kakeya b=4 + V Besi d=3 m=4 +mean | 2.98× | Δppl +1.47 %, top-1 90.48 % |
| **Qwen3** | **K bf16 + V Besi d=3 m=4 +mean** | **1.73×** | Δppl −0.25 %, top-1 95.24 % |

## Future work (not in scope here)

To enable K compression on Qwen3-style models, future codec work
would need one of:
1. **Per-head K-norm-aware PCA**: factor out the RMSNorm scaling
   before PCA fit, restore after decode.
2. **Σ_q ridge regularization**: add larger ridge (e.g., 1e-1 instead of
   1e-3) to bring Cholesky away from singular regime on condition
   numbers > 10k.
3. **Support bit_width up to 8**: the codec currently caps at b=4;
   Qwen3 may simply need b=5-6 per coord to be viable.
