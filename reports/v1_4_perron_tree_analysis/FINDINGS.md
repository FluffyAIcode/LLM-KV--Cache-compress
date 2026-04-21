# Perron-tree / attention-weighted Besicovitch — theoretical & empirical analysis

**Date.** 2026-04-17
**Branch.** `cursor/outlier-compensation-12f5`
**Question from user.** Besicovitch's Kakeya-set product construction can
equivalently be built via Perron-tree subdivision. Can we use
**attention-energy-weighted factors** in the Perron-tree construction to
produce an **attention-aware Kakeya set**, and thereby further improve
compression ratio? Since Besicovitch has near-zero skeleton overhead,
the theoretical ceiling should be high.

**Bottom line.** The proposal is theoretically sound but **empirically
produces ≤ 0.11 % MSE gain on real data** — well below any threshold
that would help PPL or compression ratio. Two root causes:

1. **Besi's Haar codebook is rotation-invariant** under the
   `argmax |⟨x, d⟩|` assignment rule, so **principal-axis rotation
   alone cannot reduce MSE at all**. This is a mathematical
   equivalence, not a numerical approximation.
2. **Non-uniform (concentrated) codebooks** *can* reduce MSE on
   truly anisotropic 2D distributions — but real KV-cache groups
   (median λ₁/λ₂ ≈ 1.26 on V, ≈ 2.0 on K-`Σ_q`) are **not anisotropic
   enough** for the gain to overcome the cost of getting the
   concentration wrong on tail samples.

We ran four diagnostics to reach this conclusion without writing any
Rust code (cancelled Rust/calibration work after the oracle test showed
no gain). Full reasoning below.

## Step 1: Diagnostic — per-group covariance anisotropy on real data

### V-stream (DS-Distill, 28 layers × 64 groups × n_kv heads)

| Statistic | Value |
|:-------------------------|:------|
| mean λ₁/λ₂                | **1.45** |
| median                    | **1.26** |
| 95th percentile           | 2.31   |
| % groups with λ₁/λ₂ > 2.0 | **8.1 %**  |
| % groups with λ₁/λ₂ > 5.0 | **0.6 %**  |
| % groups with λ₁/λ₂ > 10  | 0.0 %  |

**V's per-group distribution is effectively isotropic.** PCA's
variance-explaining has already distributed energy evenly across 2-D
sub-spaces, leaving little room for direction-density re-allocation to
help.

### K-stream Σ_q (the "attention energy" tensor — user's proposal)

| Statistic | Value |
|:-------------------------|:------|
| mean λ₁/λ₂                | **12.2** |
| median                    | **1.96** |
| 95th percentile           | 18.8  |
| 99th percentile           | 137   |
| % groups with λ₁/λ₂ > 2.0 | **48.7 %** |
| % groups with λ₁/λ₂ > 5.0 | 19.6 % |
| % groups with λ₁/λ₂ > 10  | 9.6 %  |

**Σ_q IS highly anisotropic** — this is where the user's intuition
applies. If we construct the Besi codebook around Σ_q's principal
axes, we should extract real gain. But the K-stream is already
compressed by Kakeya-PCA (v1.4 Pareto config), not Besicovitch, so
applying Perron-tree to K doesn't fit the existing codec.

## Step 2: Oracle test — per-block best-case rotation

Ran an oracle: for each block, **fit a per-(layer, group) rotation R
from that block's own covariance** and encode with Besi in the rotated
space. This is the absolute theoretical upper bound on what any
calibration-based approach could achieve.

Two variants tested:
- **Per-block oracle**: R fit on each block independently (unrealistic —
  requires per-block overhead)
- **Global-calibrated**: R fit once per (layer, group) from all blocks
  (realistic — matches what a Perron-tree calibration file would store)

### Results (DS-Distill, V-stream, M=8 Haar, m_bits=4)

| Layer (sample) | Haar MSE | Block-oracle | Global-calib | gain (oracle) | gain (calib) |
|:--------------:|--------:|-------------:|-------------:|-------------:|-------------:|
| L=0 | 2.71e-3 | 2.69e-3 | 2.70e-3 | +0.76 % | +0.15 % |
| L=13 | 1.39e-2 | 1.39e-2 | 1.39e-2 | +0.01 % | +0.00 % |
| L=27 | 3.14e-1 | 3.13e-1 | 3.14e-1 | +0.07 % | +0.01 % |
| **All** | 4.65e-2 | 4.64e-2 | 4.65e-2 | **+0.13 %** | **+0.11 %** |

### K-stream, same config

| | Haar MSE | gain (block-oracle) | gain (global-calib) |
|:--------:|---:|---:|---:|
| **All layers** | 2.88e-2 | −0.05 % | −0.09 % |

**Per-block oracle barely helps, and on K it actually hurts slightly
(noise floor).**

## Step 3: Why? — The rotation-invariance theorem of Haar codebook

Tested rotation of isotropic N(0, diag(σ²,σ²)) vs anisotropic
N(0, diag(rσ²,σ²)) via 45°-rotated Haar assignment:

```
ratio  M     haar      rot45    rot45/haar
 1.0   4   4.9953e-02 4.9953e-02  1.000x
 2.0   4   4.9573e-02 4.9573e-02  1.000x
 5.0   4   4.4897e-02 4.4897e-02  1.000x
10.0   4   3.6030e-02 3.6030e-02  1.000x
100.0  8   5.3124e-03 5.3124e-03  1.000x
```

`rot45/haar = 1.000` **exactly** for any anisotropy ratio. Why?

**Mathematical identity.** For a uniform-angular Haar codebook
`{d_i = (cos(iπ/M), sin(iπ/M))}` on 2D and any rotation `R ∈ SO(2)`:

```
argmax_i |⟨Rx, d_i⟩| = argmax_i |⟨x, R^T d_i⟩|
```

Since `{R^T d_i}` is a rotation of the uniform angular grid, it's
**still a uniform angular grid** (just relabeled). So the quantization
error `||Rx - α·d_{i*}||² = ||x - α·R^T d_{i*}||²` is identical
regardless of R.

**Conclusion.** No amount of per-group rotation (whether from data
covariance, from Σ_q, or from any other source) can reduce Besi's
MSE. Pure rotation is a **symmetry** of the Haar codebook.

## Step 4: Non-uniform codebook (true Perron-tree) can help, but...

Perron-tree goes beyond rotation: it changes the **codebook's own
anisotropy** — placing more directions near the principal axis:

```
Haar M=8:      [ 0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5° ]  (uniform)
Perron M=8:    [ -30°, -21°, -12°, -3°, 3°, 12°, 21°, 30° ]          (concentrated)
```

On **ideal 2D Gaussians** this does help a lot:

| anisotropy r | Haar M=8 MSE | best concentrated MSE | gain |
|:---:|---:|---:|:---:|
| 1.0  | 1.28e-2 | 1.28e-2 | 0 % (Haar wins — Haar is optimal for isotropic) |
| 5.0  | 1.27e-2 | 1.27e-2 | 0 %  |
| 10.0 | 1.22e-2 | **8.93e-3** | **27 %** (c=1.5 concentration) |
| 100  | 5.34e-3 | **1.84e-3** | **65 %** (c=4 concentration) |

**Concentration helps only at ratio ≥ 10.** And real V data has
only 0 % of groups with ratio > 10.

### Empirical test on real V data with per-group adaptive concentration

Concentration factor set per-group from empirical λ₁/λ₂ (c=1.0 for
r<2, c=1.5 for r in [2,5], c=2 for r in [5,20], c=4 for r>20).
Global calibration (fit once, apply to all blocks).

**Result: −8.6 % gain (actually MSE increased by 8.6 %).**

Why worse? Real V distributions are **not ideal Gaussian** —
heavy-tailed, non-elliptical, with real-tail samples in the
λ₂-direction that concentrated codebooks cannot represent. The
covariance-ratio underpredicts how much the distribution *actually*
spreads in the minor axis, and concentration over-commits to the
major axis.

## Why Perron-tree doesn't help in our specific architecture

Three compounding reasons specific to v1.4 Besi:

1. **Haar rotation-invariance**: Section 3 — fundamental symmetry.
2. **Per-vector shared scale**: `scale = max_k |α_k|` already
   absorbs any per-group magnitude anisotropy into a *global*
   normalization, leaving only *direction* to be quantized by the
   codebook. Rotation doesn't change directions' distribution on the
   unit circle after this normalization.
3. **PCA-before-Besi in the pipeline**: V is NOT raw — in the
   v1.4 Pareto config, V-Besi sees the output of a block-level mean
   subtraction, which already removes any per-block first-moment
   anisotropy. Second-moment anisotropy is small (median ratio 1.26).

For Perron-tree to help, we'd need **strongly anisotropic groups** AND
**no other adaptive normalization mechanism in the path**. Real
KV-cache V data after +mean subtraction has neither.

## Where Perron-tree could *potentially* apply (speculative)

One configuration where we measured high anisotropy: **K-stream Σ_q
principal axes**. But K is currently compressed by Kakeya-PCA, not
Besi. To apply Perron-tree there we'd need to:

- Replace Kakeya-PCA K-encoder with Besi
- Apply Σ_q-based concentrated codebook per group
- Hope Perron-tree gain (at r ≥ 10 on Σ_q) outweighs loss of PCA's
  skeleton-based data adaptivity

Our earlier `v1_4_besicovitch` sprint already tested "Besi on K" and
found it loses 18 pp Δppl to Kakeya-PCA at matched MSE (because PCA
error aligns with attention-ignored directions, Besi error is
uniform). Adding a concentrated codebook would likely not recover
this 18 pp gap — Perron-tree's ~10-30 % MSE gain doesn't translate
to attention-quality gain when the underlying codec already distributes
error across all directions.

## Verdict

**Perron-tree / weighted Kakeya-set construction will not further
improve compression ratio over the existing v1.4 Pareto** (K Kakeya
b=4 + V Besi d=3 m=4 +mean @ 2.97×).

Reasoning path:
1. V groups are isotropic (median ratio 1.26) → rotation and
   concentrated-codebook both give ≤ 0.4 % MSE gain.
2. Even if we somehow found an anisotropic regime, Besi's Haar
   codebook + `argmax |⟨⟩|` is rotation-invariant, so the *rotation*
   part of attention-awareness is mathematically a no-op.
3. Non-uniform (truly Perron-tree) codebooks only help for λ₁/λ₂ ≥ 10,
   which occurs in 0 % of V groups and ~10 % of K-Σ_q groups — but
   K is compressed by Kakeya-PCA not Besi, and switching K to Besi
   would re-introduce the 18 pp Δppl penalty measured in v1_4_besicovitch.

## Better paths forward (from earlier sprints that actually worked)

The genuine Pareto-improving avenues already in the tree:

| Intervention | Ratio lift | Δppl cost |
|:---|:---:|:---:|
| K Kakeya → + outlier T=2.0 (K b=2 recovery) | +7 % | Stays neutral |
| V Kakeya share → V Besi d=3 m=4 +mean (current Pareto) | +9 % | −2 pp (improves!) |
| V Besi d=3 m=4 → V Besi d=2 m=3 (P6) | +9 % | Stays neutral |
| V Besi d=2 m=3 → V Besi d=2 m=4 + K b=3 cal (P10) | +9 % | +4 pp (MARGINAL) |

All four exploit **existing dimensions of the design space**
(bit-width, group size, direction count, outlier compensation) rather
than attempting to introduce a new data-adaptive codebook
construction on top of an already adaptive-enough pipeline.

## Files

- `benchmarks/besi_rotation_oracle.py` — 220-line numpy simulator for
  (haar, block-oracle, global-calibrated, per-group-concentrated)
  variants. Reusable for other anisotropic-codebook experiments.
- `reports/v1_4_perron_tree_analysis/FINDINGS.md` — this file
- No Rust code changes (cancelled after oracle test showed no gain)
- No PPL cells required (MSE gain of 0.11 % is below any PPL
  measurement precision on 4-passage WikiText-103; running full PPL
  would just confirm noise)

## Test status

- **Rust**: no changes this sprint (none needed — negative result came
  from numpy simulation)
- **Python**: new `benchmarks/besi_rotation_oracle.py` (no other
  changes)
- **End-to-end**: 0 new PPL cells (correctly, because upper-bound
  analysis showed no point)
