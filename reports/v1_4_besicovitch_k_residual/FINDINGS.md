# K-residual Besicovitch: can replacing Lloyd-Max scalar quantizer with Besi-product suppress PPL?

**Date.** 2026-04-17
**Branch.** `cursor/outlier-compensation-12f5`

**Question.** The v1.4 K codec passes the PCA-residual through a
Walsh–Hadamard transform and a per-vector-norm scaling, then feeds the
resulting vector to a per-coord Lloyd-Max scalar quantizer. Could
replacing that final quantizer with a Besicovitch-product codec
(group-size 2 Haar direction codebook + per-vector-scale magnitude
Lloyd-Max) reduce PPL at the same or better bit budget?

**Answer. No.** At matched bit budget, Besi-residual has ~3× higher
block-MSE than Lloyd-Max-residual, and no end-to-end PPL configuration
Pareto-dominates the Lloyd-Max baseline.

## Why the conjecture was plausible

Theoretical argument for Besi-residual winning:
1. WHT Gaussianizes the residual → isotropic near-Gaussian source
2. Per-vector-norm scaling → near-unit-variance source
3. Besi Haar codebook is RD-optimal for isotropic sources under MSE
   (Gersho 1979).
4. Besi's 2D joint quantization could capture residual cross-coord
   correlations that Lloyd-Max's per-coord scalar quantizer ignores.

If any of these were the *dominant* failure mode of the current codec,
Besi should win. Experiment tests all four.

## Implementation

Rust additions:
- `besicovitch.rs`: `encode_vector` / `decode_vector` /
  `serialize_code` / `deserialize_code` / `serialized_nbytes` helpers
  for single-vector-at-a-time use (matches Lloyd-Max's per-code
  interface).
- `codec.rs`:
  - `CodecParams.residual_besi: Option<BesicovitchParams>`.
    When `Some`, encode bypasses Lloyd-Max and writes Besi-serialized
    bytes into `Code.residual_besi`; decode dispatches on
    `Skeleton.residual_besi`.
  - `Code.residual_besi: Vec<u8>` — Besi-encoded residual bytes
    (mutually exclusive with `residual_packed` + `outliers`).
  - `Skeleton.residual_besi: Option<BesicovitchParams>` — tells
    the decoder which path to take.
  - 3 new unit tests (178 tests total, all pass).
- `kakeyaturbo-bench`: CLI flags `--residual-besi-direction-bits`,
  `--residual-besi-magnitude-bits`, `--residual-besi-magnitude-mode`,
  `--residual-besi-group-size`.
- `e2e_ppl_pre_rope.py`: `--k-residual-besi-direction-bits` etc.
  plumbed from CLI through `evaluate → roundtrip_cache → _encode_k`.

## MSE smoke test (real DS-Distill pre-RoPE K L=13, D=128)

Block MSE after full v1.4 pipeline (Q-precond off for this smoke; would
shift absolute numbers but not relative ranking):

| Config                         | Ratio | Block MSE    | bits/vec |
|--------------------------------|------:|-------------:|---------:|
| Lloyd-Max b=2                  | 3.27× | 8.213 × 10⁻³ | 626      |
| Lloyd-Max b=3                  | 2.72× | **2.387 × 10⁻³** | 754  |
| Lloyd-Max b=4                  | 2.32× | **6.546 × 10⁻⁴** | 882  |
| Besi residual d=3 m=3 q (~3 bpc) | 2.66× | 7.169 × 10⁻³ | 770 |
| Besi residual d=3 m=4 q (~3.5 bpc) | 2.46× | 2.722 × 10⁻³ | 834 |
| Besi residual d=4 m=4 q (~4 bpc) | 2.28× | 2.049 × 10⁻³ | 898 |
| Besi residual d=6 m=4 q (~5 bpc) | 2.00× | 1.837 × 10⁻³ | 1026 |

**Besi-residual has ~3× higher MSE than Lloyd-Max at matched bit
budget.** And Besi at 4 bits/coord (898 b/v) still can't match
Lloyd-Max at 3 bits/coord (754 b/v).

## Is the residual really isotropic Gaussian? (Diagnostic)

Numpy simulation of full pipeline (PCA vr=1.0 d_eff=125 → k-means 16 →
subtract → pad to wht_len=128 → WHT → 1/‖res‖ scale):

| Property                                    | Value     | Interpretation |
|---------------------------------------------|----------:|:---------------|
| Scaled-residual kurtosis                    | **−0.18** | Near-Gaussian, slightly **light-tailed** (not heavy) |
| Per-coord std range                         | 0.084–0.092 | Flat across coords ✓ |
| 2D group angle KS vs uniform                | **0.019** | **Essentially Haar-uniform** |
| 2×2 pair covariance eigenvalue ratio        | median **1.13** | Near-isotropic |

**The residual is exactly the distribution Besi's Haar prior targets.**
So why does Besi lose?

## Why Lloyd-Max wins on this source

The residual being near-isotropic Gaussian means WHT **decorrelated**
the coordinates — which makes Lloyd-Max's core assumption (i.i.d.
unit-Gaussian per coord) **exactly correct**.

- Lloyd-Max at b=3 on i.i.d. unit-Gaussian achieves per-coord MSE
  = 0.03454 σ² (Max 1960). With 128 coordinates this is the RD lower
  bound for scalar quantization.
- Besi at d=3 (8 directions on the circle) has angular-quantization
  MSE ≈ sin²(11.25°) × σ² = **0.038 σ²** — already 10 % worse per
  dimension, and that's before magnitude-quantization error.
- Besi's shared per-group scale (`max|α_k|` over 2 coords) binds two
  coords together through a single extreme, hurting the typical case.
- Cross-coord correlations that Besi's 2D joint quantizer *would*
  capture don't exist in this data (WHT destroyed them).

In short: **WHT turns the residual into Lloyd-Max's best case, while
removing the structure Besi would exploit.**

## End-to-end 4-passage PPL, DS-Distill D=128

Same harness: ctx=2048, n_eval=64, block_size=1024, Q-precond
skip=[0,1,26,27], conservative boundary b=4, K PCA exact, vr=1.0,
V=Kakeya b=2 share.

| Config                                   | Ratio  | Δppl      | top-1    | Verdict    |
|------------------------------------------|-------:|----------:|---------:|:----------:|
| **Lloyd-Max b=4 baseline**               | **3.03×** | **−0.78 %** | **86.51 %** | **ACCEPT ★** |
| K-res Besi d=3 m=4 q                     | 3.13×  | +1.34 %   | 83.33 %  | ACCEPT (top-1 low) |
| K-res Besi d=6 m=4 q                     | 2.78×  | +3.92 %   | 88.10 %  | MARGINAL   |
| K-res Besi d=4 m=3 q                     | 3.13×  | +6.56 %   | 80.95 %  | MARGINAL   |
| K-res Besi d=4 m=4 q                     | 3.00×  | +9.92 %   | 86.11 %  | MARGINAL   |
| K-res Besi d=3 m=3 q                     | 3.26×  | +13.05 %  | 79.76 %  | REJECT     |
| K-res Besi d=5 m=3 q                     | 3.00×  | +19.52 %  | 81.35 %  | REJECT     |

**No K-residual-Besi configuration Pareto-dominates the baseline.**
The closest competitor (d=3 m=4 q @ 3.13×) trades 4 pp Δppl and
3.2 pp top-1 for a 3 % ratio gain — not a meaningful Pareto move.

## Why V-cache Besi wins and K-residual Besi loses

This result sits in interesting tension with the V-cache sprint
(`reports/v1_4_besicovitch_v_only/FINDINGS.md`), where Besi Pareto-
dominated Lloyd-Max on the V stream. The difference lies in **where
in the pipeline** Besi is inserted:

| Location | Source properties | Optimal quantizer |
|----------|-------------------|-------------------|
| **V cache** (raw, pre-any-transform) | anisotropic, correlated, non-zero mean | **Besicovitch** — captures 2D geometry, mean-subtraction fixes bias |
| **K residual** (post-PCA, post-kmeans, post-WHT, post-scale) | decorrelated, near-isotropic, zero-mean, near-Gaussian | **Lloyd-Max** — per-coord scalar is RD-optimal on independent sources |

V cache has the structure Besi exploits. K residual had that structure
**intentionally destroyed** by WHT (that's what Gaussianization is
for). Inserting Besi after WHT is asking it to find structure that
was just eliminated.

## Architectural lesson

**Codec choice depends on the stream's state in the pipeline**, not
just on whether the final distortion is MSE or inner-product:

- Use Besicovitch **upstream** of Gaussianizing transforms, where
  geometric structure still exists.
- Use Lloyd-Max **downstream** of Gaussianizing transforms, where
  the per-coord independence assumption holds.

Equivalently: WHT + Lloyd-Max is a joint design — the WHT creates the
conditions under which Lloyd-Max is optimal. Replacing Lloyd-Max
without also changing WHT breaks the joint design.

**To make Besi-residual viable, one would need to *remove* the WHT**
and feed Besi the post-k-means residual directly. The WHT is
Gaussianization-for-Lloyd-Max; Besi wants the opposite (correlated,
structured input). This would be a fundamentally different codec
architecture, not a drop-in quantizer swap.

## Production recommendation (unchanged from prior sprint)

Keep the Pareto point from the V-cache sprint:

| Method | Ratio | Δppl | top-1 | Verdict |
|---|---:|---:|---:|:---:|
| K=Kakeya b=4 + V=Besi d=3 m=4 +mean | 2.97× | **−2.04 %** | **91.27 %** | **ACCEPT ★ 🏆** |
| K=Kakeya b=4 + V=Kakeya b=2 share (no cal) | 3.03× | −0.78 % | 86.51 % | ACCEPT ★ (this sprint's baseline) |

**Use Lloyd-Max on K-residual, use Besicovitch on V-cache.** That is
the Pareto-optimal combination we have found.

## Test status

- Rust: 178 tests pass (3 new residual-besi tests + 11 besicovitch
  total + existing)
- Python: harness wired through; 7 end-to-end PPL cells run successfully
- No regressions in any prior sprint's tests or code paths.

## Files

- `kakeyaturbo/src/besicovitch.rs` — +5 new public fns, +2 new tests
- `kakeyaturbo/src/codec.rs` — `residual_besi` field on
  `Code`/`Skeleton`/`CodecParams`; encode/decode dispatch; 2 new tests
- `kakeyaturbo/src/bin/kakeyaturbo-bench.rs` — `--residual-besi-*` flags
- `benchmarks/e2e_ppl_pre_rope.py` — `--k-residual-besi-*` flags,
  plumbing through `rust_roundtrip` / `_encode_k`
- `reports/v1_4_besicovitch_k_residual/ds_kresbesi_*.json` — 7 cells
- `reports/v1_4_besicovitch_k_residual/FINDINGS.md` — this file
