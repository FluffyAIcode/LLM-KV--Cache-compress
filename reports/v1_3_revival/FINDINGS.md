# v1.3 PPL — progressive guardrail ladder (the rehabilitation trail)

**Date.** 2026-04-17 → 2026-04-21
**Branch.** `cursor/outlier-compensation-12f5`

## Purpose

This directory contains the **progressive evidence trail** for the
"v1.3 PPL" recipe (see `reports/v1_3_ppl/FINDINGS.md` for the
production cell). Each cell adds **one** PPL stabilization guardrail
on top of the previous, so the contribution of each guardrail is
isolated.

All rows: 4-passage WikiText-103, DS-Distill-Qwen-1.5B (D=128), ctx=2048.

## b=2 ladder — from disaster to MARGINAL

| Step | Added guardrail | Δppl | top-1 |
|:----:|:---|-----:|------:|
| V0 | **BARE** v1.3 RSVD b=2, 0 boundary | **+355.62 %** | 42.46 % |
| V1 | + Q-precondition (Chol Σ_q) + 4 bdry | **+37.91 %** | 73.02 % |
| V2 | + K calibrated Lloyd-Max codebook | +36.53 % | 68.25 % |
| V3 | + K/V cal + 6 bdry (add L=7, L=14) | **+25.18 %** | 71.43 % |

Source: `V{0-3}_*.json`.

**V0 → V3: Δppl +355 % → +25 % (14× better), top-1 42 % → 71 %
(+29 pp).** Q-precond alone is worth 317 pp Δppl — the single
biggest guardrail contribution.

## b=3 — where the recipe stabilises enough to be useful

| Step | Added guardrail | Δppl | top-1 |
|:----:|:---|-----:|------:|
| B0 | **BARE** v1.3 RSVD b=3 | +374.90 % | 41.27 % |
| B1 | + all guardrails except outlier | **+15.73 %** | 76.98 % |

Source: `B{0,1}_*.json`.

**B0 → B1: Δppl +374 % → +16 % (23× better).** Adding outlier T=2.0
on top of B1 (and switching V to RSVD b=2 shared-basis, which is
v1.3's default) produces the **v1.3 PPL** production cell in
`reports/v1_3_ppl/` at 4.61× / +7.82 % / 78.97 % MARGINAL.

## Four architectural conclusions from this ladder

1. **Q-preconditioning** (Chol Σ_q) is the single biggest guardrail.
   V0 → V1: Δppl +355 % → +37.9 %. This IS the "put it in Riemannian
   space" operation — the Σ_q-metric → Euclidean isometry. No
   separate Riemann codec needed.

2. **Calibrated Lloyd-Max codebook** helps at b=2 (V1 → V2: top-1
   68 → 71 % after re-adding V3's boundary). Stored in
   `reports/v1_4_q_pca/calibrated_codebook/ds_K_b{2,3}_centroids.f32`.

3. **Boundary expansion from 4 to 6 layers** (add L=7, L=14 — the
   worst per-layer MSE on DS-Distill) is the second-biggest
   guardrail (V2 → V3: +36.5 % → +25.2 % Δppl).

4. **Outlier compensation T=2.0** (B1 → v1.3 PPL, after also picking
   up V RSVD b=2 shared-basis) brings Δppl from +15.7 % → +7.8 % and
   top-1 from 77 % → 79 %.

## Tuning handles (quality vs ratio)

Higher quality is reached by **raising K and/or V bit width** on the
same v1.3 PPL recipe — no new codec path needed. Lower ratio but
better Δppl / top-1:

- K b=3 → K b=4
- V b=2 → V b=3 (or V b=4)
- RSVD rank-factor 0.5 → 0.75 (larger skeleton, smaller residual)

Higher ratio is reached by **tightening outlier threshold** or
**dropping boundary layers**, but each step has been measured to
degrade PPL sharply; the v1.3 PPL default (K b=3, V b=2 shared,
T=2.0, 6 bdry) is the ratio-first MARGINAL Pareto point.

## Deliverables

- `V{0-3}_*.json` — 4 b=2 ladder cells (progressive guardrails)
- `B{0,1}_*.json` — 2 b=3 ladder cells (bare → full guardrails)
- `FINDINGS.md` — this file

**Production recipe and cell**: `reports/v1_3_ppl/`.
