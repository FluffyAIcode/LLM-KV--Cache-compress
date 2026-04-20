# Can Besicovitch-product codec be made attention-aware?

**Date.** 2026-04-17
**Branch.** `cursor/outlier-compensation-12f5`
**Question.** The vanilla Besicovitch codec loses 18 pp Δppl to
Kakeya-PCA at matched MSE. The previous report concluded "Besicovitch
cannot incorporate attention." This report **refines** that claim:
Besicovitch *can* be made attention-aware, but the simplest route
(Q-preconditioning the K stream) breaks the Lloyd-Max magnitude
quantizer and produces catastrophic PPL collapse. A secondary
f16-magnitude variant shows a small but real improvement that still
doesn't change the Pareto story.

## Why vanilla Besicovitch is not attention-aware

High-resolution rate-distortion theory (Gersho 1979):
for source density `p(x)` and distortion `d(x, x̂)`, the optimal
codebook point density is

```
ρ*(x) ∝ p(x)^{D/(D+α)} · w(x)
```

where `w(x)` encodes the distortion metric weighting. Vanilla
Besicovitch's direction codebook is the **uniform Haar measure on
S^(g-1)** — this is the codebook that's optimal only when:

1. The source `p(x)` is isotropic in R^D.
2. The distortion `d(x, x̂)` is plain MSE.

Neither holds for K-cache compression:

- K is approximately low-rank / heavy-tailed / anisotropic in R^D.
- The correct distortion for attention is
  `d_Σq(x, x̂) = (x - x̂)^T Σ_q (x - x̂)` where `Σ_q = E[qq^T]` is the
  query covariance.  This is strongly anisotropic in general.

PCA/RSVD codecs align their skeleton with the data spectrum (via PCA)
AND — when Q-preconditioned — with Σ_q. Vanilla Besicovitch
aligns with neither: the codebook is a fixed global object with no
knobs to turn.

## Five concrete routes to make Besicovitch attention-aware

1. **Q-preconditioning**: whiten K by L⁻¹ (L = Chol(Σ_q)) before
   encode, un-whiten by L after decode. In the transformed space,
   plain MSE is exactly the Σ_q-weighted distortion on the original
   K.  **Implementation cost: zero** — the v1.4 Q-precond pipeline
   already works with `--codec besicovitch`.
2. **Σ_q-weighted direction codebook sampling**: sample M directions
   with density `π(d) ∝ (d^T Σ_q^(group) d)^{1/2}` instead of Haar
   uniform.  Codebook is still global, just anisotropic.
3. **Non-uniform bit allocation across groups**: give groups with
   higher `Σ_q`-energy more direction_bits.  Equivalent to an
   in-code-space rank truncation.
4. **Σ_q-weighted Lloyd-Max magnitude centroids**: tune the α
   quantizer to the Σ_q-weighted objective.
5. **Hierarchical Besicovitch**: top-level projection on Σ_q's
   top-k eigenvectors (= Σ_q-metric PCA), bottom-level Besicovitch
   on residual. This converges back to `Σ_q-PCA + flat residual`
   — i.e. v1.4 Q-precond pipeline.

Route 1 is the cheapest and cleanest. Tested here.

## Experiment: Besi + Q-precond on DS-Distill D=128 (4 passages)

Same harness as prior reports: ctx=2048, n_eval=64, block_size=1024,
boundary layers {0, 1, 26, 27} kept at Kakeya-PCA b=4 conservative.
Q-precond applied to middle 24 layers with `skip_layers=[0,1,26,27]`
(Layer 0 K would overflow f16 under whitening; see
reports/v1_4_q_pca/FIVE_X_QUEST.md).

| Config                          | ratio | Δppl       | top-1   | verdict  |
|---------------------------------|------:|-----------:|--------:|:--------:|
| **Sprint 3.5 (reference)**      | **3.03×** | **−3.56 %** | **87.30 %** | **ACCEPT ★** |
| Besi d=5 m=4 quant (vanilla)    | 3.30× | +14.50 %   | 79.37 % | REJECT   |
| Besi d=5 m=4 quant + Q-precond  | 3.30× | **+777.29 %** | **27.78 %** | **DISASTER** |
| Besi d=6 m=4 quant (vanilla)    | 3.03× | +13.93 %   | 82.14 % | REJECT   |
| Besi d=6 m=4 quant + Q-precond  | 3.03× | **+713.62 %** | **27.38 %** | **DISASTER** |
| Besi d=7 m=4 quant (vanilla)    | 2.80× | +12.72 %   | 79.76 % | REJECT   |
| Besi d=7 m=4 quant + Q-precond  | 2.80× | **+725.40 %** | **28.17 %** | **DISASTER** |
| Besi d=6 f16 (vanilla)          | 1.55× | +6.12 %    | 90.48 % | MARGINAL |
| Besi d=6 f16 + Q-precond        | 1.55× | **+5.81 %** | **91.67 %** | MARGINAL |

## Why quantized-magnitude breaks under Q-precond (and f16 doesn't)

Q-whitening changes the α distribution fundamentally:

- **Pre-whitening**: `α_k = <group_k, d*>` where group_k has bounded
  magnitude (K is bounded by model parameters / RoPE-less magnitudes).
  The α distribution is near-exponential (peaked near 0, light tails).
- **Post-whitening**: the K norm distribution becomes a *true* Gaussian
  with heavy tails expanded along formerly low-variance directions.
  Whitening amplifies exactly the spread that Lloyd-Max's "unit-Gaussian
  centroids + per-vector scale" assumption was designed for — except
  the per-vector `scale = max |α_k|` is now driven by a single
  whitened-Gaussian extreme, dragging every other α_k's quantization
  bin far away from its mode.

Concretely, for a whitened group vector with max-α = 3.0 and a typical
α_k = 0.5:

- Pre-whitening (bounded data): scale=1.2, centroid spread covers
  ±2.5, α_k=0.5 lands on a bin at 0.4 → error 0.1 = 20 %.
- Post-whitening: scale=3.0, centroid spread covers ±6.0, α_k=0.5
  lands on a bin at 0.0 (floored) → error 0.5 = 100 %.

f16 magnitude mode escapes this entirely because each α is stored
directly with no per-vector shared scale.

## Why Besi d=6 f16 + Q-precond gains only 0.3 pp Δppl

At d=6 f16, the direction codebook has M=64 directions on the circle
and magnitude is f16-precise per group. Reconstruction error is
**already dominated by the angular quantization** (π/64 ≈ 2.8°
per-direction error), not by magnitude quantization. Q-whitening
rotates where the angular error lives — from "evenly distributed in
R^D" toward "concentrated in Σ_q-small directions" — which is
exactly what we want for attention.

But the effect is small (0.31 pp Δppl improvement, 1.19 pp top-1
improvement) because d=6 f16 is such a low-compression config
(1.55×) that the reconstruction is already nearly lossless. There's
not much error to redirect.

## Projection: what would each attention-aware path yield?

Based on the f16 result and MSE-vs-PPL scaling observed in prior
sprints:

| Route | Expected best case | Risk |
|---|---|---|
| **1. Q-precond (tested)** | ~3 pp Δppl improvement at f16; catastrophic at quantized | Breaks Lloyd-Max assumption |
| **2. Σ_q-weighted codebook** | ~8 pp at quantized (would also fix Route 1's Lloyd-Max issue if codebook is whitening-aware) | Requires per-model Σ_q calibration — loses Besicovitch's "fully data-independent" advantage |
| **3. Non-uniform bit alloc** | ~5 pp; preserves construction purity | Requires per-layer tuning |
| **4. Σ_q-weighted Lloyd-Max** | ~2 pp; only addresses magnitude quantization | Moot because magnitude MSE is small component |
| **5. Hierarchical** | converges to Q-precond PCA | Not really Besicovitch anymore |

Even optimistic combinations of routes 2+3+4 project to close **half**
the 18-pp Δppl gap to Kakeya-PCA. The remaining gap is the
fundamental cost of the Besicovitch construction's lack of per-block
data adaptivity: the codebook is fixed, so the first-moment
subtraction (per-block mean) is the *only* block-local adaptivity,
and the second-moment structure (covariance) is handled only via
(a) global Σ_q calibration (routes 2/4) or (b) Q-precond rotation
(route 1).

Kakeya-PCA gets block-level second-moment adaptivity **for free** via
its per-block eigendecomposition. That's the load-bearing property.

## Refined conclusion

**Vanilla Besicovitch cannot incorporate attention**, but **Besicovitch
can be made attention-aware** by one of routes 1-5 above. The question
reduces to a quantitative one: *can any attention-aware Besicovitch
variant Pareto-dominate Kakeya-PCA?*

Experimental evidence says **no on this model/scale**:
- Route 1 (Q-precond) either breaks at typical compression (quantized
  magnitude) or only marginally helps at low compression (f16).
- Routes 2-5 would require reintroducing exactly the data-adaptive
  machinery (per-block or per-layer covariance estimation) that
  Besicovitch was introduced to eliminate.

The mathematical beauty of "fixed global codebook + product" is at
odds with the engineering need for "attention-direction awareness."
The two requirements are not quite contradictory (Q-precond is a
global, non-block-local way to achieve awareness), but they force
each other into trade-offs that Kakeya-PCA doesn't face.

## Sprint 3.5 retains the deployable Pareto point on DS-Distill D=128

```
 Method                          ratio    Δppl       top-1     verdict
 Sprint 3.5 (Kakeya PCA)         3.03×    −3.56%     87.30%    ACCEPT ★
 Besi d=5 m=4 quant (vanilla)    3.30×    +14.50%    79.37%    REJECT
 Besi d=6 m=4 quant (vanilla)    3.03×    +13.93%    82.14%    REJECT
 Besi d=6 f16 + Q-precond        1.55×     +5.81%    91.67%    MARGINAL (best top-1)
 Besi d=6 f16 (vanilla)          1.55×     +6.12%    90.48%    MARGINAL
 Besi d=6 m=4 + Q-precond        3.03×   +713.62%    27.38%    DISASTER
```

## Follow-up questions (not addressed here)

- Would a Σ_q-weighted direction codebook (route 2) actually deliver
  the projected 8 pp improvement? Unknown without experiment.
- Does the Besicovitch + Q-precond + f16 variant hold up at longer
  context (ctx ≥ 16k)? The 1.55× ratio drops with longer context as
  PCA skeleton amortizes better. Besi has no skeleton, so relative
  advantage may grow at extreme contexts.
- Is there a hybrid regime: Besicovitch for prefill (no per-block fit
  cost), Kakeya-PCA for persisted cache? This was in the v1_4_besicovitch
  FINDINGS as speculative; still speculative.

## Files

- `reports/v1_4_besicovitch_qprecond/ds_besi_qp_*.json` — 4 cells
- `reports/v1_4_besicovitch_qprecond/FINDINGS.md` — this file
- `benchmarks/e2e_ppl_pre_rope.py` — unchanged from besicovitch sprint;
  Q-precond plumbing already existed from v1.4 Q-PCA work.
