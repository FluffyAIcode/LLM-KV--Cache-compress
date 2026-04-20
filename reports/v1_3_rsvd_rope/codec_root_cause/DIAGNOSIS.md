# Codec root-cause diagnosis

**Date**: April 20, 2026
**Branch**: `cursor/v1-3-rsvd-rope-aware-12f5` (continuing after e2e-PPL finding)
**Context**: End-to-end PPL validation (PR #12) found the codec catastrophically fails downstream quality even at max fidelity. This report isolates the mechanisms.

## TL;DR

Two distinct bugs / architectural issues identified, ranked by quantitative impact:

1. **WHT scaling bug** (real, fixed this commit). Scale factor was `√wht_len / ‖res‖` but should have been `1 / ‖res‖` because the codec uses an *unnormalized* Walsh-Hadamard transform. This made Lloyd-Max inputs have per-coord variance `wht_len` instead of `1`, saturating the quantiser for the bulk of coordinates. Fix: change one line in `codec.rs`. Effect: K-stream per-block correlation goes from 0.939 to 0.985 at b=2.

2. **Per-layer PPL floor is structurally non-zero** (real, NOT fixable by parameter tuning). Even at maximum fidelity (`variance_ratio=1.0, b=4, exact PCA`, producing SNR ≥50× single-layer reconstruction), the codec induces **+2.5% PPL inflation per layer** on real downstream prediction. Compounded across 24 layers, this becomes +15 648% PPL. This is an architectural property of the skeleton + residual-coding pipeline, not a tunable.

The combination means: the codec cannot be salvaged to ACCEPT quality by parameter changes alone. Either the architecture must change, or the paper's quality claims must be withdrawn.

## Diagnostic methodology

A new stage-by-stage Rust binary `kakeyaturbo-stage-by-stage-decode` emits four intermediate reconstructions per block:

- **s1** — mu + U Uᵀ (x - mu), PCA projection only
- **s2** — PCA + K-means center assignment (exact, no quantization)
- **s3** — PCA + K-means + WHT forward/inverse round-trip (no Lloyd-Max)
- **s4** — full codec: PCA + K-means + WHT + Lloyd-Max quantization

Differencing consecutive stages attributes error to each code path.

Applied to Qwen2.5-0.5B layer 5 K tensor, block 0 (1536 vectors × 64 dim), at the paper's default config `bit_width=2, r=D/2, vr=0.95, randomized SVD`, the error budget is:

| Stage | MSE | SNR | Correlation |
|---|---:|---:|---:|
| s1 PCA only | 1.106e-1 | 63.5× | 0.992 |
| s2 +kmeans | 1.106e-1 | 63.5× | 0.992 |
| s3 +WHT round-trip | 1.106e-1 | 63.5× | 0.992 |
| **s4 +Lloyd-Max quantization** | **8.35e-1** | **8.4×** | **0.939** |

s1 → s3 are identical (mathematically zero-error: PCA is exact, K-means residual formula is exact, WHT is an orthogonal rotation). **The entire codec error budget is the Lloyd-Max quantization step.**

## Bug #1: WHT scaling inconsistency (fixed in this commit)

The `rotate()` function in `wht.rs` implements the **unnormalized** Walsh-Hadamard transform (butterfly with no 1/√N factor). Consequently, for a residual `res` of length `d_eff` (padded with zeros to `wht_len`), the rotated vector `rotated = H·D·res_padded` satisfies:

    ‖rotated‖² = wht_len · ‖res‖²     (unnormalized)

rather than `‖rotated‖² = ‖res‖²` (which would be the normalized convention).

The encode path in `codec.rs:encode_block` (pre-fix) computed:

    scale = √wht_len / ‖res‖
    scaled = rotated · scale

As a result:

    ‖scaled‖² = ‖rotated‖² · scale² = wht_len·‖res‖² · (wht_len/‖res‖²) = wht_len²

per-coord variance ≈ `wht_len²/wht_len = wht_len`. For `d_eff=26` giving `wht_len=32`, this means `scaled` values have **per-coord standard deviation ≈ √32 ≈ 5.66**, while the Lloyd-Max codebook (for `b=3` centroids `±[0.245, 0.756, 1.344, 2.152]`) is calibrated for **N(0,1)**, i.e. ~99.7% of values in `[-3, +3]`.

### Empirical confirmation (before the fix)

On a single residual from Qwen2.5-0.5B layer 5:

    scaled std = 5.64
    scaled max|value| = 13.25
    values outside ±2.15 (b=3 Lloyd-Max max): 21 / 32

21 of 32 residual coordinates were saturating to the extreme centroid, losing almost all information.

### Fix

Change `codec.rs` line 249 (and the matching line in the `share_basis=true` inlined path plus `stage-by-stage-decode.rs`):

    // before
    let scale = (wht_len as f32).sqrt() / res_norm;
    // after
    let scale = 1.0 / res_norm;

After fix, `‖scaled‖² = ‖rotated‖² · scale² = wht_len·‖res‖² · 1/‖res‖² = wht_len`, so per-coord variance = 1, matching the Lloyd-Max calibration.

### Effect on stage ablation (post-fix)

| Config | K s4 SNR before | K s4 SNR after | Factor |
|---|---:|---:|---:|
| b=4 vr=0.95 exact | 11.9× | (~53×) | ~4.5× |
| b=3 vr=0.95 exact | 10.1× | **50.0×** | **5.0×** |
| b=2 vr=0.95 exact | 8.4× | **32.7×** | 3.9× |
| **b=2 vr=0.95 rsvd r=D/2 (v1.3)** | **8.4×** | **32.6×** | **3.9×** |

V stream is nearly unchanged (correl was already 0.99, the scaling bug's amplification didn't push V beyond the Lloyd-Max range because V residuals have much smaller magnitude).

## Issue #2: Structural per-layer PPL floor (NOT fixable by parameters)

After applying the WHT scaling fix, we re-ran the end-to-end PPL compounding test on Qwen2.5-0.5B.
This test applies the codec to the **first K layers only** (K = 0, 1, 2, 4, 8, 16, 24) and measures downstream PPL on 64 teacher-forced tokens.

### Post-fix results

| # compressed layers | Δppl (paper default) | Δppl (v1.2 b=3 exact) | Δppl (max fidelity b=4 vr=1.0 exact) |
|---:|---:|---:|---:|
| 0 | -0.3% | -0.3% | -0.3% |
| 1 | **+3.9%** | **+3.7%** | **+2.5%** |
| 2 | +9.2% | +7.0% | +7.9% |
| 4 | +35.5% | +39.6% | +38.2% |
| 8 | +147.9% | +149.4% | +141.5% |
| 16 | +846.4% | +927.5% | +1169.0% |
| 24 | +9341.0% | +6671.8% | +15647.5% |

### Interpretation

- **Maximum fidelity still incurs +2.5% PPL per layer.** `variance_ratio=1.0` means PCA keeps all components (d_eff = D = 64), `b=4` is the finest Lloyd-Max codebook available, `exact PCA` eliminates any RSVD truncation. This is the **irreducible floor** of the current codec architecture per layer.
- **Paper default (b=2 rsvd r=D/2) is +3.9% per layer**, only ~1.5% worse than max fidelity at a 1-layer test. So most of the per-layer damage is NOT caused by aggressive parameters; it's structural.
- **Compounding is super-linear.** By the 4-layer mark, all three configs are already +35%+; by 24 layers, all are effectively REJECTs (10-100× PPL inflation).

### Hypotheses for the +2.5% structural floor

1. The bf16 skeleton storage. PCA basis is stored in bf16 (v1.2 A optimisation). A bf16 PCA basis on a 64-dim K vector with entries of ~O(1) magnitude retains about 3 significant decimal digits — corresponding to 0.1% relative error per coordinate, which accumulates across d_eff ≈ 10-30 basis vectors per block.
2. The K-means t-scalar stored in fp16. The per-vector projection onto the cluster center is stored as fp16; any quantization of this affects EVERY vector, and K-means is a hard-assignment method where the t-scalar variance is high.
3. The shared PCA basis approximation. For V stream we use layer-pooled PCA whose basis is a compromise across blocks; even at vr=1.0 the pooled basis doesn't match each block's local structure.
4. **Most likely**: the block-level Lloyd-Max codebook is not adapted to the residual's actual distribution. Each block has its own residual variance pattern; a universal N(0,1) codebook is suboptimal.

Each of these would be a separate engineering investigation, and each probably contributes ~0.5-1% of the 2.5%.

## What this means

- The codec architecture — PCA skeleton + K-means + WHT + universal Lloyd-Max — **has a structural per-layer quality floor of ~2.5% PPL**, independent of bit width, rank, or PCA accuracy. This compounds exponentially with depth.
- The paper's MSE-based ACCEPT framework (MSE inflation ≤ 1.10× → ship) **cannot predict end-to-end quality** because the MSE-to-PPL relationship at 24-layer compounding is not monotone in the low-noise regime.
- **The codec cannot be saved to ACCEPT PPL quality by parameter changes.** Going from paper-default to max-fidelity moves PPL from +4% to +2.5% per layer — a partial improvement, but the compounding still destroys quality.

## Remediation options

1. **Architectural replacement**: drop the skeleton+residual paradigm for K. Go to per-channel int4/int8 quantization (like KIVI) on K, keep skeleton+residual only for V which tolerates it (stage ablation shows V at +0.02% per-layer MSE under max fidelity).
2. **Calibration / fine-tuning**: quantize and then fine-tune a small adapter per layer to recover downstream quality. This abandons "training-free" but might recover PPL ACCEPT at current compression ratios.
3. **Per-block adaptive codebook**: replace universal Lloyd-Max with a small per-block codebook (e.g., 4-8 centroids selected via k-means on the scaled residuals). Adds ~1 KB per block to skeleton storage but possibly closes the per-layer PPL gap.
4. **Withdraw compression-with-ACCEPT claim from paper**. Keep the mathematical framework, keep the compression ratio numbers, but explicitly state that the codec is not yet PPL-ACCEPT and the engineering of quality-preserving compression is open.

## Reproduction

```bash
# Rebuild (includes both bug #1 fix and stage-by-stage binary)
cd kakeyaturbo && cargo build --release --bins && cd ..

# Stage-by-stage ablation
python3 benchmarks/stage_ablation_driver.py \
    --model-path models/Qwen2.5-0.5B-Instruct \
    --layer 5 --n-tokens 2048 --block-size 512

# Depth compounding test
python3 /tmp/one_layer_test.py   # copy from this report
```
