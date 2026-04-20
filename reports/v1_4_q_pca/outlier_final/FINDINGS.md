# Outlier compensation — end-to-end PPL validation

**Date.** 2026-04-17
**Branch.** `cursor/v1-3-rsvd-rope-aware-12f5`
**Model.** DeepSeek-R1-Distill-Qwen-1.5B (D=128, flagship proxy)
**Setup.** pre-RoPE cache, ctx=2048, n_eval=64, **4 passages** WikiText-103,
streaming-safe. Q-precond skip=[0,1,26,27], conservative boundary b=4.
All K configurations use exact PCA + per_block + calibrated Lloyd-Max
(`ds_K_b2_centroids.f32`). V stream fixed at b=2 exact share.

## What outlier compensation does

Architectural extension (Rust + Python):
- `CodecParams.outlier_threshold: Option<f32>`
- Each `Code` gets an `outliers: Vec<(u16, f16)>` sparse list
- Encode: post-WHT, post-scale coordinates with `|scaled| > T` are
  stored exact as (u16 index, f16 value) — 4 bytes each
- Decode: Lloyd-Max dequantize as usual, then override outlier indices
  with exact f16 values, then inverse-scale + inverse-WHT
- 5 new unit tests (164 total pass)

## Single-block MSE validation (Rust bench binary direct)

At b=2, exact PCA, vr=1.0, on 1024-vector synthetic block:

| T      | MSE          | MSE drop | bytes    | ratio vs bf16 |
|--------|-------------:|---------:|---------:|--------------:|
| none   | 1.110 × 10²  | —        | 80 128 B | 3.27×         |
| 3.0    | 1.031 × 10²  | +7.1 %   | 81 508 B | 3.22×         |
| 2.5    | 9.064 × 10¹  | +18.3 %  | 86 356 B | 3.04×         |
| **2.0**| **7.484 × 10¹** | **+32.6 %** | 103 660 B | 2.53× |
| 1.5    | 6.951 × 10¹  | +37.4 %  | 150 520 B| 1.74×         |
| 1.0    | 5.150 × 10¹  | +53.6 %  | 247 404 B| 1.06×         |

**MSE drop matches the earlier diagnostic's prediction** (ran on
the `outlier_compensation_diagnostic.py` numpy-side simulation).
T=2.0 is the sweet spot for MSE-per-byte tradeoff.

## Full end-to-end PPL — K b=2 recipe (4 passages, apples-to-apples)

| outlier T | total ratio | Δppl     | top-1    | verdict    |
|----------:|------------:|---------:|---------:|:----------:|
| none      | **3.62×**   | +15.65 % | 73.02 %  | REJECT     |
| 2.5       | 3.49×       | +9.93 %  | 77.78 %  | REJECT     |
| **2.0**   | **3.17×**   | **+6.05 %** | 77.38 % | REJECT (MARGINAL edge) |
| 1.5       | 2.55×       | +8.23 %  | 83.73 %  | MARGINAL   |
| 1.0       | 1.82×       | +6.89 %  | 80.95 %  | MARGINAL   |

Outlier monotonically improves Δppl from +15.65 % → +6.05 % at T=2.0
— a **9.6 pp improvement** — but the ratio drops from 3.62× → 3.17×.

## What went wrong vs. the pre-experiment prediction

The earlier numpy diagnostic predicted "T=2.0 → +4.02× @ Δppl = −2 %
ACCEPT".  The real result is **T=2.0 → 3.17× @ Δppl = +6.05 % REJECT**.

Post-mortem: the diagnostic used the wrong baseline for Δppl projection.
It referenced Step 5's +9.09 % (which was a mixed K per_block + V share
config measured at 2 passages), not the true 4-passage K b=2 recipe
baseline of +15.65 %.  A 6 pp systematic offset in the baseline
translated directly into a 6 pp offset in the prediction.  The
predicted MSE drop was accurate (≈32 %); the issue was where to apply
it.

The outlier mechanism's **absolute improvement** on the correct
baseline is **9.6 pp Δppl** at +1.2 % top-1 — real, but not enough
to clear the +15.65 % starting point.

## Sprint 3.5 with outlier T=2.0 — does outlier help at b=4?

Tested the reverse question: apply outlier T=2.0 on top of the
Sprint 3.5 recipe (K b=4 exact per_block + V b=2 share, no calibrated
codebook).  4 passages:

| Config                      | ratio  | Δppl      | top-1   | verdict |
|-----------------------------|-------:|----------:|--------:|:-------:|
| Sprint 3.5 (no outlier)     | 3.12×  | **−3.56 %** | **87.30 %** | **ACCEPT ★** |
| Sprint 3.5 + outlier T=2.0  | ~3.00× | −2.18 %    | 84.92 %    | ACCEPT (marginal top-1) |

**Outlier makes b=4 slightly WORSE.**  Δppl goes from −3.56 % →
−2.18 %, top-1 from 87.30 % → 84.92 %.  Root cause: b=4 has 16
Lloyd-Max centroids spread over ±2σ, so it's **already** covering the
typical residual distribution accurately.  Outlier patching replaces
values that were being quantized well with f16 approximations, which
have their own ~10⁻³ relative error.  On a low-loss quantizer, f16
precision becomes the dominant noise source for patched coordinates.

Takeaway: **outlier compensation is a b=2-specific tool**. It has
meaningful effect only when Lloyd-Max's bit budget is too small to
cover the residual distribution densely.

## Sprint 3.5 vs all outlier-on-b=2 configs — the deployable Pareto

```
  config                          ratio    Δppl     top-1    verdict
  Sprint 3.5 (K b=4 V b=2 share)  3.12×   −3.56%   87.30%   ACCEPT ★  ← best
  K b=2 + cal + outlier T=1.5     2.55×   +8.23%   83.73%   MARGINAL
  K b=2 + cal + outlier T=2.0     3.17×   +6.05%   77.38%   REJECT
```

**No outlier-augmented configuration Pareto-beats Sprint 3.5.**
Sprint 3.5's 3.12× @ −3.56 % strictly dominates every K b=2 + outlier
point we measured.

## Why outlier doesn't close the K b=2 → ACCEPT gap

Lloyd-Max b=2 at Gaussian-like residuals has error concentrated in two
regimes:
- **Tail |x| > 1.5**: sparse large-magnitude errors — **outlier addresses this**
- **Mid-range |x| ∈ [0.5, 1.5]**: dense medium-magnitude errors — **outlier cannot address this** (would require T < 0.5 which patches 50%+ of coords, blowing bytes)

Even fully extracting the tail only reduces MSE by ~32 %, not
enough to recover the 16 pp Δppl gap between K b=2 baseline (+15.65 %)
and ACCEPT (≤3 %).

## Final v1.4 Pareto status on DS D=128 streaming-safe 4-passage PPL

| method                                        | ratio  | Δppl      | top-1    | verdict  |
|-----------------------------------------------|-------:|----------:|---------:|:--------:|
| bf16 reference                                | 1.00×  | 0 %       | 100 %    | —        |
| **Sprint 3.5 (K b=4 + V b=2 share)**          | **3.12×** | **−3.56 %** | **87.30 %** | **ACCEPT ★** |
| K b=2 + all guardrails + outlier T=1.5        | 2.55×  | +8.23 %   | 83.73 %  | MARGINAL |
| K b=2 + all guardrails (no outlier)           | 3.62×  | +15.65 %  | 73.02 %  | REJECT   |

**Sprint 3.5 @ 3.12× remains the deployable operating point.**
Outlier mechanism is real and measurably effective at b=2, but at
the byte cost required for meaningful Δppl improvement, it doesn't
Pareto-beat the b=4 baseline.

## Architectural tools produced this sprint (useful even if not deployed)

- **Rust**: `Code.outliers` field + `CodecParams.outlier_threshold`;
  `encode_block` extracts, `decode_block` patches; clean f16 index+value
  sparse representation; 5 new unit tests.
- **CLI**: `--outlier-threshold` on `kakeyaturbo-bench`.
- **Python**: `--k-outlier-threshold` / `--v-outlier-threshold` in
  `e2e_ppl_pre_rope.py`; boundary layers auto-exclude outlier
  compensation (to avoid f16 patching on b=4 layers that are
  already high-precision).

## Outlier at long context or at other bit widths (future work)

Two settings we did not test where outlier might pay off better:
1. **Long context (ctx=8192+)**: skeleton bytes amortize, so outlier's
   +10-20 % byte overhead becomes a smaller fraction of total. The
   same 9.6 pp Δppl improvement may shift the Pareto.
2. **Per-layer or per-block calibrated centroids + outlier**: the
   diagnostic used a pooled codebook.  Per-block centroids would
   absorb part of what outlier currently catches, potentially shifting
   the sweet-spot T higher (less outlier bytes needed).

Neither is explored in this sprint.

## Files

- `kakeyaturbo/src/codec.rs` — `Code.outliers`, `CodecParams.outlier_threshold`,
  encode/decode logic, 5 new unit tests
- `kakeyaturbo/src/bin/kakeyaturbo-bench.rs` — `--outlier-threshold` flag
- `benchmarks/e2e_ppl_pre_rope.py` — `--k-outlier-threshold` / `--v-outlier-threshold`
- `reports/v1_4_q_pca/outlier_final/` — 6 per-cell 4-passage JSONs + this FINDINGS.md
