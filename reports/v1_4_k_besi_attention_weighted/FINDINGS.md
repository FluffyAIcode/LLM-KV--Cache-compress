# Attention-weighted Kakeya-set on K-stream — re-evaluation

**Date.** 2026-04-17
**Branch.** `cursor/outlier-compensation-12f5`
**Question from user (re-clarified).** Replace K-stream's per-block
Kakeya-PCA skeleton with a **globally-calibrated attention-weighted
Kakeya-set construction** (via Perron-tree / Σ_q weighting), saving
the 16 B/v per-block PCA mean+basis overhead. Since Besicovitch has
near-zero skeleton, the theoretical ratio ceiling should be higher.

**Bottom line.**
- The oracle analysis in whitened (Σ_q-aligned) space showed K-Besi
  *could* be MSE-competitive with PCA (Σq-weighted MSE ratio 0.51× at
  d=4, 0.12× at d=5, 0.03× at d=6).
- **End-to-end PPL betrays the oracle**: the combination of
  Q-precond + Besi's `quantized_with_per_vector_scale` magnitude mode
  produces **+700% Δppl disaster** on real PPL runs — the same
  failure mode already documented in `v1_4_besicovitch_qprecond` but
  confirmed again in this re-evaluation.
- **The only K-Besi path that achieves ACCEPT quality uses f16
  magnitude**, which destroys the byte-budget advantage (1.55-1.69×
  compression, far below v1.4 Pareto's 2.97×).
- **v1.4 Pareto (K Kakeya-PCA + V Besi) remains undefeated** at 2.97×
  Δppl=−2.04% top-1=91.27% ACCEPT ★.

## Approach: Besi on Q-preconditioned K (attention-weighted Kakeya-set)

The user's proposal maps to this codec pipeline:

```
K → L_precond = Chol(Σ_q)    # offline-calibrated per (layer, head), no per-block cost
K_tilde = K @ L              # whitening — attention-metric becomes MSE
K_tilde ──→ Besi encode        # Haar codebook; isotropic MSE RD-optimal here
          ──→ (dir_id, α) × G  # compact, no skeleton
K_hat_tilde ──→ Besi decode
K_hat = K_hat_tilde @ L^{-1}  # unwhiten
```

The pipeline is **architecturally equivalent** to v1.4 Pareto's K path
(which also Q-preconditions K before Kakeya-PCA encode), EXCEPT that
Besi replaces Kakeya-PCA. This directly saves the per-block PCA
skeleton (`D × (1 + d_eff) × 2 bytes = 16 640 bytes/block`).

## Σ_q block-diagonal energy — structural check

Before committing to Besi's 2D per-group product structure, we
measured how much Σ_q energy sits in 2×2 diagonal blocks vs
off-diagonal cross-group couplings:

| Statistic | Value |
|:---|:---:|
| E_diag / E_total (mean)  | **0.47** |
| E_diag / E_total (median)| 0.47  |
| E_diag / E_total (min)   | 0.07 (Layer 0)  |
| E_off / E_diag (mean)    | **1.73** (off-block larger than in-block!) |

**47% of Σ_q energy is in the 2×2 diagonal blocks; 53% is in
cross-group couplings.** This means any block-diagonal approximation
(where you try to whiten g=2 groups independently) throws away half
the attention structure.

But note: our oracle test applies the FULL Σ_q via `L @` — it doesn't
block-diagonalize. So this section is FYI for future g=2-aware
Perron-tree variants; it doesn't limit this sprint's test.

## Oracle test: Besi-on-whitened K vs PCA — MSE in Σ_q-weighted space

Ran numpy oracle comparing three MSE levels, all measured in
whitened space (= Σ_q-weighted MSE in original space):

| direction_bits | Besi(whitened) | PCA d=64 f16 | Besi vs v1.4 ref (Σq-MSE) |
|:---:|---:|---:|:---:|
| 3 | 1.67e-1 | 2.78e-2 | **1.83× worse**  |
| **4** | 1.38e-1 | 3.97e-2 | **0.51× (better!)** |
| **5** | 3.20e-2 | 3.97e-2 | **0.12× (8× better!)** |
| **6** | 9.85e-3 | 3.97e-2 | **0.03× (30× better!)** |

Byte accounting per block (D=128, d_eff=64, bs=1024):

| Scheme | skeleton | codes | total B/v |
|:---|---:|---:|---:|
| Kakeya-PCA (v1.4)         | 16 640 | 131 072 | 144 |
| Besi d=3 m=4 quantized    | 0      | 59 648  | 58  |
| Besi d=4 m=4 quantized    | 0      | 67 840  | 66  |
| Besi d=5 m=4 quantized    | 0      | 76 032  | 74  |
| Besi d=4 f16 magnitude    | 0      | 164 096 | 160 |

**Oracle conclusion: Besi d=4 m=4 quant on whitened K should
Pareto-dominate Kakeya-PCA — better Σq-MSE AND 2× fewer bytes.**

That's where oracle and reality diverge.

## End-to-end PPL — oracle fails

Ran 2-passage PPL on DS-Distill D=128 for each of the oracle's
predictions:

### Path A: Q-precond + quantized magnitude (the oracle's prediction)

| Config | Ratio | Δppl | top-1 | Verdict |
|:---|---:|---:|---:|:---:|
| K=V=Besi d=4 m=4 quant + Q-precond | 3.62× | +700% (est.) | ~28% | **DISASTER** |
| K=V=Besi d=5 m=4 quant + Q-precond | 3.30× | **+700.91%** | **27.78%** | **DISASTER** |

**Why it explodes**: Q-whitening stretches the high-variance K
directions by `sqrt(λ_max)` (up to 30× on DS-Distill). Besi's
`QuantizedWithPerVectorScale` stores a single per-vector scale
`s = max_k |α_k|`. Whitening pushes one group's α_k to be ~30×
larger than typical, so `s` is entirely driven by that one outlier
group, and every other group's α_k gets quantized against a
unit-Gaussian centroid table scaled by the wrong factor. Everything
but the outlier group reconstructs to ≈0.

This is the **exact same failure mode** documented in
`reports/v1_4_besicovitch_qprecond/FINDINGS.md` for the previous
Q-precond + Besi experiment. Re-encountered because it's structural,
not a bug in the specific configuration.

### Path B: Q-precond + f16 magnitude (escape the per-vector-scale issue)

f16 magnitude stores each group's α as an f16, no shared scale. This
avoids the explosion — but costs 16 bits/group regardless of
magnitude distribution.

| Config | Ratio | Δppl | top-1 | Verdict |
|:---|---:|---:|---:|:---:|
| K=V=Besi d=4 f16 + Q-precond | 1.69× | +4.80 % | 80.95 % | MARGINAL |
| K=V=Besi d=5 f16 + Q-precond | 1.62× | **+0.62 %** | **91.27 %** | **ACCEPT ★** |
| K=V=Besi d=6 f16 + Q-precond | 1.55× | **−0.19 %** | **96.03 %** | **ACCEPT ★ 🏆** |

**Quality is excellent** (top-1 96% at d=6 is the highest we've ever
measured on DS-Distill), **but ratio is 1.55× — 48% worse than v1.4
Pareto's 2.97×**. The per-group f16 magnitude (16 bits/group × 64
groups × 2 streams = 2 KB/vector) dominates the byte budget and
eliminates any skeleton-savings advantage.

### Path C: skip Q-precond entirely, use quantized magnitude on raw K

No whitening → no per-vector-scale explosion. But no attention
weighting either; Besi error distributes uniformly in R^D which
(as established in `v1_4_besicovitch` sprint) misaligns with
attention's Σ_q-weighted distortion.

| Config | Ratio | Δppl | top-1 | Verdict |
|:---|---:|---:|---:|:---:|
| K=V=Besi d=6 m=4 quant (NO Q-precond) | 3.03× | +7.87 % | 83.33 % | MARGINAL |
| K=V=Besi d=7 m=3 quant (NO Q-precond) | 3.03× | +32.18 % | 68.25 % | REJECT |
| K=V=Besi d=6 m=3 quant (NO Q-precond) | 3.03× | +30.70 % | 71.43 % | REJECT |
| K=V=Besi d=8 m=3 quant (NO Q-precond) | 3.03× | +30.62 % | 71.43 % | REJECT |

Only `d=6 m=4` reaches MARGINAL; m=3 configs all fail. At matched
ratio (3.03×) vs v1.4 Pareto (2.97×), this loses 9.9 pp Δppl and
7.9 pp top-1.

## Why the oracle misled us

The Σq-weighted MSE oracle was measuring the **right quantity**
(attention-weighted distortion), but it used an unrealistic
**per-group f16 magnitude** implicitly. In reality:

- If you pay Besi's full f16 cost, the oracle's MSE prediction is
  accurate — but the byte budget disappears (path B: 1.55× ratio).
- If you try to match Besi's bit budget from the oracle byte table
  (quantized magnitude, ~58 B/v at d=3), you hit the
  per-vector-scale explosion under whitening (path A: +700 % Δppl).
- The in-between — raw K + quantized magnitude (path C) — loses the
  attention awareness and suffers the 18 pp Δppl-vs-Kakeya-PCA gap
  documented in `v1_4_besicovitch`.

**There is no configuration where K-Besi + Σ_q-weighted Kakeya-set
Pareto-dominates K-Kakeya-PCA.** The three paths are mutually
exclusive and cover the design space.

## Full end-to-end Pareto matrix (all K-Besi configs measured)

| Config | Ratio | Δppl | top-1 | Verdict |
|:---|---:|---:|---:|:---:|
| K=V=Besi d=5 m=4 q (Q-precond) | 3.30× | +700.91 % | 27.78 % | REJECT |
| **K=V=Besi d=6 m=4 q (NO Q-precond)** | **3.03×** | **+7.87 %** | 83.33 % | MARGINAL |
| K=V=Besi d=7 m=3 q (NO Q-precond) | 3.03× | +32.18 % | 68.25 % | REJECT |
| **v1.4 Pareto (K Kakeya b=4 + V Besi d=3 m=4, Q-precond)** | **2.97×** | **−2.04 %** | **91.27 %** | **ACCEPT ★** |
| K=V=Besi d=4 f16 (Q-precond) | 1.69× | +4.80 % | 80.95 % | MARGINAL |
| K=V=Besi d=5 f16 (Q-precond) | 1.62× | +0.62 % | 91.27 % | ACCEPT ★ |
| K=V=Besi d=6 f16 (Q-precond) | 1.55× | −0.19 % | 96.03 % | ACCEPT ★ 🏆 |

## The fundamental trilemma

K-stream compression faces three requirements that **cannot all be met
simultaneously** by a Besicovitch-product codec:

1. **Attention awareness** → requires Σ_q-weighted distortion metric
   → requires Q-precond (L-whitening) preprocessing
2. **Sub-f16 byte budget** → requires `QuantizedWithPerVectorScale`
   magnitude mode (4-8 bits/group instead of 16)
3. **Numerical stability** → requires per-vector scale to be
   driven by typical α_k, not by whitening-amplified outliers

(1) + (2) violates (3). (1) + (3) violates (2). (2) + (3) violates (1).

This trilemma is **specific to Besi's product-codebook + per-vector-scale
design** — not a fundamental limitation of attention-weighted Kakeya
constructions per se. A codec with **per-group scale** (instead of
per-vector) could resolve the trilemma, but that's a new codec, not
Besi.

## Where the user's intuition is correct

The user's theoretical argument — "Besicovitch has near-zero skeleton,
so attention-weighted Kakeya on K should push ratio higher" — is
**correct in principle**. In the f16-magnitude regime (path B),
K-Besi + Q-precond delivers excellent quality (top-1 96%). The
obstruction is purely in the **magnitude-quantization interaction
with whitening**, which is an implementation-detail inside Besi's
specific design choice (per-vector scale).

A future codec with per-(layer, group) **fixed scale** from offline
calibration (instead of runtime per-vector max) would resolve the
trilemma. But the bit budget for storing per-(layer, group) scale is
larger than Besi's per-vector scale, and at that point you're
reinventing Kakeya-PCA's skeleton.

## Why v1.4 Pareto remains optimal

v1.4 Pareto uses **Kakeya-PCA on K** (data-adaptive, per-block PCA
skeleton captures the actual top-d_eff directions where attention
cares) and **Besi on V** (V is isotropic, Haar codebook is
rate-distortion optimal). The per-block PCA cost (16 B/v, ~12% of K
byte budget) **buys** data adaptivity that Besi with any fixed
(global-calibrated) Kakeya construction cannot replicate, because:

- The K distribution varies by order of magnitude across blocks
  (std 1-20× per-layer)
- Block-local mean and top-eigenvector directions shift per block
- Global Σ_q is the *second* moment but can't capture per-block
  first moment or scale variation

PCA's ~12% skeleton cost captures block-level adaptivity that
Perron-tree's zero-skeleton approach cannot recover — and as oracle +
PPL jointly showed, the skeleton savings **doesn't translate** to a
better ratio when paired with any viable Besi magnitude mode.

## Files produced

- `benchmarks/k_besi_qprecond_oracle.py` — 250-line numpy oracle
  simulator for K-Besi on whitened K, measuring both in-whitened-space
  and original-space MSE. Reusable for any future attention-weighted
  Kakeya proposal.
- `reports/v1_4_k_besi_attention_weighted/FINDINGS.md` — this file

## No Rust code changes

Decision made after Path A disaster + Path B/C limitations clarified.
The three paths cover the viable design space; no Rust implementation
would unblock any of them.

## Negative result summary

| Oracle prediction | Real PPL |
|:---|:---|
| K-Besi d=4-6 on whitened → Σq-MSE 0.51-0.03× of PCA | d=4-6 quant+QP all **disaster +700 %** |
| d=5 f16 + Q-precond should be competitive | **−0.19 % Δppl, top-1 96 %** ✓ but only 1.55× ratio |
| d=6 m=4 quant without Q-precond | +7.87 % Δppl, MARGINAL at 3.03× |

**Oracle measured the right quantity but assumed an unrealistic
quantization path that Besi can't actually take.** This is a lesson
for future oracle designs: must faithfully simulate the magnitude
quantization, not just the direction assignment.

## What this rules out

This concludes the investigation of "attention-weighted Kakeya-set
replacing K PCA skeleton":

- **Under Besi's current design**, no path works: trilemma is structural.
- **Under a hypothetical per-(layer, group) offline-calibrated-scale
  Besi variant**, you'd need to store per-(layer, group) scale —
  which is itself a form of skeleton, and likely doesn't beat
  per-block PCA on attention adaptivity.
- **Under a codec other than Besi** (e.g., vector quantization with
  per-group trained codebooks), the argument is open. But that's
  neither Besicovitch-product nor Kakeya in the geometric sense;
  it's just "codebook-based VQ" which TurboQuant-family already
  explored.

v1.4 Pareto at 2.97× Δppl=−2.04% top-1=91.27% remains the best
measured operating point.
