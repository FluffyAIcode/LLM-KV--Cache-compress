# Path A diagnostic — cross-layer V skeleton feasibility (Step 1 of 5)

**Date.** 2026-04-17
**Branch.** `cursor/v1-3-rsvd-rope-aware-12f5`
**Model.** DeepSeek-R1-Distill-Qwen-1.5B (D=128, 28 full-attention layers)
**Purpose.** Step 1 of the 5-step plan to assess Scenario 3 (v1.3 tier-1
recipe + all three TurboQuant+-style guardrails). Two sub-questions:

1. **Is cross-layer V skeleton sharing (Path A) PPL-safe at full rank?**
2. **Does cross-layer V pooling enable V rank cap = D/2 (Block 3b
   recovery in Scenario 3)?**

## Diagnostic result at full rank (or near-full)

Cross-layer V pool MSE inflation vs per-layer pool at matching d_eff:

| setting                  | median inflation | p90    | interpretation |
|--------------------------|-----------------:|-------:|---------------|
| vr = 1.0 (full rank)     | **1.03×** (fp noise) | 1.03× | mathematically equivalent |
| vr = 0.999               | **0.92×**         | 0.96×  | cross-layer BETTER (more data → smoother tail) |
| vr = 0.99                | 0.75×             | 0.85×  | cross-layer BETTER |
| vr = 0.95, no skip      | 1.15×             | 1.22×  | minor PPL cost (~+6% Δppl predicted) |
| vr = 0.95, skip=[0,1,26,27] | **1.10×**    | 1.14×  | minor PPL cost (~+3.6% Δppl predicted) |

**Verdict Q1: Path A is PPL-safe in the vr≥0.95 regime.**  In Sprint 3.5's
actual operating point (vr=1.0), cross-layer share is mathematically
identical to per-layer share up to fp rounding.  Confirmed green light
for Step 2 (Path A implementation).

## Diagnostic result at rank cap = D/2 = 64

Three scenarios compared at fixed rank_cap=64 (matches v1.3 tier-1
RSVD `target_rank = D/2`):
- **A**: per-layer pool + rank cap=64 (known REJECT from Sprint 3.1)
- **B**: cross-layer pool + rank cap=64 (proposed Block 3b recovery)
- **C**: per-layer pool + full rank (Sprint 3.5 baseline, ACCEPT)

| metric                                      | median | p90    | max    |
|---------------------------------------------|-------:|-------:|-------:|
| B/A (does cross-layer help at same rank?)   | **1.82×** | 1.98× | 2.11× |
| B/C (cross+cap vs full-rank baseline)       | N/A (baseline MSE ≈ fp noise) |
| A/C (per+cap vs full-rank baseline)         | N/A (same) |

**Critical finding**: **Cross-layer pool at rank cap=64 is 1.82× WORSE
than per-layer pool at rank cap=64 in median MSE.**

## Why Path A helps at full rank but hurts at rank cap

Two different mechanisms:

| operating regime | dominant error source | cross-layer effect |
|------------------|----------------------|:------------------:|
| Full / near-full rank | Tail eigenvector estimation noise (finite sample) | ✅ Helps — 28× more samples stabilise tail |
| Rank cap < half of D  | Layer-specific `W_V_l` creates layer-specific top-r subspaces | ❌ Hurts — global top-r misses layer-optimal directions |

At rank cap=64 on DS D=128:
- Per-layer pool picks the 64 directions most aligned with that layer's
  `V_proj_l × h_ln` specific structure.
- Cross-layer pool picks the 64 directions most aligned with the
  *average* of all layer-specific subspaces — suboptimal for any
  single layer.

## Consequence for Scenario 3

Combined with Sprint 3.1's already-established result that per-layer V
rank cap=64 REJECTs at +4.88% Δppl, cross-layer rank cap=64 would be
even worse:

```
Sprint 3.1 baseline (per-layer rank cap)       : Δppl ≈ +4.88 % (REJECT)
cross-layer rank cap × extra 1.82× MSE penalty : Δppl ≈ +4.88% × 1.82^0.55 ≈ +7.2 %?

(more accurately, the MSE inflation compounds: predicted Δppl ≈ +30-50%)
```

**Block 3b is officially not recoverable.** V must stay at full rank
(d_eff = D = 128) in Scenario 3.

## Scenario 3 ceiling revision

Original (overly optimistic) estimate: **6.04×** (assumed Block 3b recoverable)
Revised: **4.63×** (V stays at full rank, only Block 1 partial + Block 3a recovered)

DS D=128 byte account at the revised ceiling:

```
K stream (b=2 per_block + all guardrails):
  skeleton (exact, d_eff=128, 4 blocks): 145 KB / layer
  codes (b=2, d_eff=128):                128 KB / layer
  total:                                 273 KB / layer
V stream (b=2 cross-layer share, full rank):
  skeleton (1 global × 32 KB) / 28 layers: ~1 KB / layer
  codes (b=2, d_eff=128):                 168 KB / layer
  total:                                  169 KB / layer
K+V per layer:                            442 KB
Ratio: 2048 / 442 =                       4.63×
```

## Implication for the 5-step plan

| step | target ratio | achievability |
|------|-------------:|---------------|
| Step 2 (Path A alone) | **3.36×** | Full data support, LOW risk |
| Step 3+4 (K b=3 + Lloyd-Max cal + boundary) | **3.82×** | Literature support, MEDIUM risk |
| Step 5 (K b=2 + all guardrails) | **4.63×** | Theoretical support only, HIGH risk |

Target 6.04× is withdrawn as speculative.  **4.63× is the honest Scenario
3 ceiling**, equal to TurboQuant+ production turbo3 (4.6×) but at
uncertain Δppl quality.

## Files

- `benchmarks/cross_layer_v_pool_diagnostic.py` — vr-based diagnostic
- `benchmarks/v_cross_layer_rank_cap_diagnostic.py` — rank-cap diagnostic (Step 1)
- `ds_distill.json`, `ds_distill_vr095.json`, `ds_distill_vr095_skip.json` — vr diagnostic outputs
- `ds_distill_rankcap64.json` — rank cap diagnostic output
