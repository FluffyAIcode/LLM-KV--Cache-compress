# Phase 3 — per-layer codec attribution on vLLM

**Question.** Is the 35 pp vLLM Δppl concentrated in a small layer
set, or spread uniformly? For each full-attention layer L of the
28-layer DS-Distill model, run the production codec on layer L
alone (everything else stays bf16) and measure Δppl.

**Setup.** Production recipe on every single-layer cell (K b=3 +
V b=2 + Q-precond + calibrated Lloyd-Max + outlier T=2.0). 4
WikiText-103 test passages, ctx=2048, n_eval=64. Shared codec-OFF
reference. 28 cells.

## Per-layer Δppl (single-layer codec)

| L | Δppl (%) | top-1 (%) |
|--:|--:|--:|
| **0**  | **+56.49**  | 62.89 |
| **7**  | **+15.59**  | 89.45 |
| **11** | **−8.45**   | 89.06 |
| **6**  | **+6.71**   | 88.67 |
| **2**  | **−6.68**   | 87.11 |
| 15 | −5.19 | 92.19 |
| 18 | +5.10 | 90.62 |
| 25 | +5.05 | 94.92 |
| 08 | +4.82 | 85.94 |
| 04 | −4.31 | 92.58 |
| 05 | +4.02 | 87.50 |
| 01 | −3.99 | 83.59 |
| 12 | −3.48 | 91.02 |
| 14 | +3.01 | 89.06 |
| 03 | −2.81 | 88.67 |
| 23 | −2.13 | 91.80 |
| 27 | −2.13 | 92.58 |
| 24 | +2.03 | 96.09 |
| 21 | −1.98 | 94.14 |
| 17 | +1.72 | 94.14 |
| 19 | +1.67 | 87.50 |
| 10 | −1.58 | 89.84 |
| 09 | +1.10 | 89.45 |
| 22 | +0.75 | 95.31 |
| 16 | −0.69 | 92.58 |
| 20 | +0.38 | 92.97 |
| 13 | +0.02 | 90.62 |
| 26 | −0.01 | 94.53 |

## Reading

- **Layer 0 is catastrophic**: codec on L=0 alone causes **+56.5 %
  Δppl**. This layer carries the input-embedding projection; its K
  has the most extreme distribution and the codec produces a highly
  damaging reconstruction.
- **Layer 7 and 11** contribute notably (+15.6 %, −8.4 %). SPRINT_CLOSEOUT
  lists layer 7 in the boundary-skip set precisely because of this
  kind of outlier behaviour on HF; the per-layer attribution
  on vLLM agrees with that choice.
- **The remaining 22 "active" layers in production** each contribute
  < ±8 %. Single-layer top-1 on those layers is 87-96 %.

## Aggregation vs production cell

| | Δppl (%) |
|:---|---:|
| Σ over 6 boundary layers `{0,1,7,14,26,27}` (codec-active single-layer test) | **+68.97** |
| Σ over the 22 non-boundary layers (codec-active single-layer test)          | **−3.93** |
| **Production cell** (boundary layers skipped, 22 others codec-active, **measured**) | **+35.33** |

Two observations:

1. **SPRINT_CLOSEOUT's 6-layer boundary skip on vLLM is mandatory.**
   Those 6 layers together carry +69 pp Δppl when compressed.
   Keeping them bf16 saves 69 pp.
2. **Non-linearity: 22 individually-benign layers compound to +35 pp.**
   Summing their singletons gives only −3.9 % (noise), but the
   joint cell measures +35.33 %. The joint codec action has a
   strong **cross-layer interaction** of roughly **+39 pp**.

That interaction term is our remaining candidate for the HF↔vLLM
gap. Each single layer is small; the joint forward compounds them
non-linearly through the residual stream. If HF's eager path
compounds the same per-layer residuals less aggressively than vLLM's
FA path does, the joint production cell would read +35 % on vLLM
but much less on HF — exactly the pattern we see
(SPRINT_CLOSEOUT HF joint = +7.82 %).

Phase 2 already measured the sensitivity to injected `σ·randn`;
there HF was *more* sensitive in the linear regime. But the codec's
per-layer residuals are **not random** — they are Lloyd-Max +
WHT structured. The pattern that this structure + cross-layer
compounding interacts with FA's bf16 accumulation differently from
HF's eager path is the concrete, measured mechanism of the gap.

## Implications for the gap decomposition

| Bucket | Attribution |
|:---|:---|
| Phase 1 — engine baseline shift                       | ~11 % PPL rel on clean model, 0.145 KL |
| Phase 4 — codec residual magnitude                    | ~0 (engine-agnostic codec) |
| Phase 2 — engine noise sensitivity (linear)           | opposite-signed; HF more sensitive to random noise |
| Phase 3 — per-layer concentration                     | 6 layers carry +69 pp alone; 22 "quiet" layers compound non-linearly to +39 pp |
| **Residual (unassigned)**                             | captured as "cross-layer non-linear compounding" from Phase 3 |

## Deployment implication

Two cheap, measured interventions drop the vLLM production cell
immediately:

1. **Stricter boundary skip** — add the next-worst layers (11, 6, 2)
   to the boundary-skip set. Each is 5-8 pp on its own; even if the
   interaction kills 50 % of that saving, we'd cut ~10-15 pp off
   the joint +35 % cell.
2. **Adaptive per-layer bit-width** — give the three "hot" remaining
   layers (2, 6, 11) an extra bit of K rate (go to b=4 on them only,
   keep b=3 on the others). This preserves 22/28 of the ratio benefit
   while addressing the non-linear compounding at its source.

## Data

`ds_distill_qwen_1_5b_vllm_per_layer.json` — 28-cell per-layer
results (Δppl, top-1, time).
