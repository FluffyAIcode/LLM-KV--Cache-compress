# 2×2×2 codec ablation — findings

**Date.** 2026-04-17
**Branch.** `cursor/v1-3-rsvd-rope-aware-12f5`
**Model.** Qwen2.5-0.5B-Instruct, pre-RoPE cache (architecturally correct).
**Fixed config.** `block_size=512, bit_width=3, variance_ratio=0.995,
ctx_len=1024, n_eval=64, n_passages=2`, K inner-product metric, V MSE metric,
`rsvd_target_rank = D/2 = 32, oversample=8, power_iters=2`.
**Corpus.** 2 WikiText-103-raw-v1 passages of length ≥ 1088 tokens.

## Goal

Separate three originally-confounded variables and measure which one
dominates the remaining PPL floor on the pre-RoPE cache:

- **PCA construction**: `exact` DSYEVD vs `randomized` (HMT with 2
  power iterations).
- **Skeleton storage dtype**: `fp16` (v1.2/v1.3 default) vs `fp32`.
- **PCA basis reuse**: `per_block` (one fit per 512-token block) vs
  `shared` (one fit per layer).

All eight cells hold block_size, bit_width, variance_ratio, and the RoPE
architecture fixed, so any difference is attributable to these three axes.

## Bench hang found (and fixed) on the way

Running the RSVD + share_basis combo on real K cache data exposed a
long-standing instability in the randomized-SVD power iteration: on a
matrix with singular-value ratio ~55 (the layer-0 K stream here) the
un-orthogonalised power iteration `Z ← AᵀA Z` grows column norms
exponentially, and the subsequent `nalgebra` thin-SVD on the resulting
near-rank-deficient matrix enters an effectively non-terminating Jacobi
sweep. The fix is textbook HMT 2011 §4.5 Algorithm 4.4: re-orthogonalise
Z between iterations via intermediate QR on A·Z and AᵀA·Z. After the
fix every RSVD call completes in ~10 ms per block.

This bug was never hit by the earlier benchmark set because the synthetic
Gaussian data used in unit tests has a flat spectrum (singular-value
ratio near 1). All 144 existing unit tests still pass after the fix.

## Results

```
pca         skel   share         Δppl      KL      top1
exact       fp16   per_block    +94.28%   0.5868   62.70%
exact       fp16   shared      +103.93%   0.6225   61.90%
exact       fp32   per_block    +96.76%   0.5889   65.08%
exact       fp32   shared      +112.25%   0.6555   64.29%
randomized  fp16   per_block   +158.55%   0.8823   60.32%
randomized  fp16   shared      +179.02%   1.0192   55.56%
randomized  fp32   per_block   +154.85%   0.8895   59.52%
randomized  fp32   shared      +190.26%   1.0468   51.59%
```

Marginal effects (mean over the other two axes):

| axis              | level         | mean Δppl |
|-------------------|---------------|-----------|
| pca_method        | exact         | **+101.81 %** |
| pca_method        | randomized    | +170.67 %     |
| skeleton_dtype    | fp16          | +133.95 %     |
| skeleton_dtype    | fp32          | +138.53 %     |
| share_basis       | per_block     | **+126.11 %** |
| share_basis       | shared        | +146.36 %     |

## Interpretation

1. **PCA construction method dominates** (pca: +68.9 pp gap between
   `randomized` and `exact`). This is the single biggest variable in the
   ablation — larger than skeleton dtype and larger than basis reuse
   combined. With `exact` PCA the codec sits near +95-112 % Δppl; with
   RSVD it jumps to +155-190 %.

2. **Skeleton dtype is effectively noise** (+4.6 pp gap, within the
   variance of two passages). This falsifies the "fp16 skeleton storage
   is the structural PPL floor" hypothesis we raised in the earlier
   pre-RoPE report. The round-trip error introduced by storing the PCA
   mean/basis/centres in fp16 is small compared to whatever else is
   limiting quality.

3. **Layer-shared basis hurts modestly** (+20.3 pp gap). Pooling 2048
   vectors from four 512-token blocks into a single PCA fit gives a
   coarser basis than four block-specific fits, and the modest PPL
   penalty is consistent with that.

4. **Diminishing returns on RSVD + fp32**. In the randomized-PCA path,
   promoting the skeleton to fp32 actually *helps* slightly in the
   per-block case (+154.85 vs +158.55) but *hurts* in the shared case
   (+190.26 vs +179.02).  Both deltas are inside noise; the real
   takeaway is that once you have an RSVD approximation error in the
   basis, storing it more precisely does not help.

## What this tells us about v1.3 and the paper

- **RSVD is a real quality cost, not a free "cheap fit".** The earlier
  v1.3 marketing positioned RSVD as a drop-in replacement that only
  trades fit cost for a small byte increase. On an end-to-end PPL
  metric, RSVD adds ~69 pp of PPL inflation on Qwen2.5-0.5B's K/V cache
  compared to exact PCA at the same (d_eff, bit_width, variance_ratio).
  Any claim that RSVD "preserves quality" needs to be re-written in
  terms of MSE, not downstream quality.
- **Dropping RSVD recovers ~69 pp of PPL.** The best cell in the table
  is `exact / fp16 / per_block` at +94.28 % Δppl, which matches the
  earlier pre-RoPE finding. This is the baseline the paper's quality
  discussion should be calibrated to.
- **Skeleton precision was the wrong hypothesis.** The intuition that
  f16 skeleton storage is the hidden PPL tax was wrong. Whatever the
  remaining +94 % floor is, it is not in the skeleton bytes.
- **Layer-shared basis is a net loss**, modest but consistent. Not
  worth the amortised-byte savings until the RSVD and PCA-accuracy
  issues are resolved.

## Remaining open questions

At this point the per-layer K and V reconstruction errors are:
- independent of RoPE (pre-RoPE cache removes it)
- independent of skeleton storage precision (this ablation)
- dominated by PCA construction quality (this ablation)

The next wedge to rule in or out is **residual coding precision**.
Candidates:
- bit_width sensitivity at exact PCA + per_block (the earlier b=3→b=4
  jump was measured with RSVD on, which we now know confounds).
- `variance_ratio` sweep at exact PCA + per_block (higher retained rank
  may collapse what RSVD was losing).
- block_size sensitivity (512 is somewhat arbitrary; 256 or 128
  may give better PCA localisation).

None of these require architectural changes; they are pure parameter
sweeps on the exact-PCA code path.

## Artefacts

- `SUMMARY.json` — one row per ablation cell, the same data shown above.
- `qwen2_5_kv_{pca}_{skel}_{share}.json` — per-passage metrics for every
  cell (8 files for the 8 cells; the 4 RSVD files use the
  `prerope_kv_b3_…` naming suffix produced by the `e2e_ppl_pre_rope.py`
  driver).
- Codec fix: `kakeyaturbo/src/pca.rs::fit_weighted_pca_randomized_with_storage`
  — re-orthogonalise Z between power iterations.
- Harness extension: `benchmarks/e2e_ppl_pre_rope.py` now accepts
  `--skeleton-dtype`, `--share-basis-k`, `--share-basis-v`; driver
  `benchmarks/ablation_2x2x2.py` sweeps the 2×2×2 grid on one loaded
  model.
