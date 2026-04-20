# V-side rank cap ablation — v1.4 Sprint 3.1 (negative result)

**Date.** 2026-04-17
**Branch.** `cursor/v1-3-rsvd-rope-aware-12f5`
**Model.** DeepSeek-R1-Distill-Qwen-1.5B (D=128), pre-RoPE cache, ctx=2048,
2 WikiText passages, streaming-safe (per_block).

## Question

V's skeleton bytes were 46 % of V stream at b_V=2 (Sprint 3 finding).
RSVD and exact PCA produce the same-sized skeleton for a given `d_eff`;
the real lever is `d_eff` itself. The earlier v1.3 Sprint 2 on Qwen2.5
established that K tolerates a rank cap via Q-preconditioning.  Does V
also tolerate a rank cap, given V has no softmax non-linearity to
amplify reconstruction noise?

## Setup

Fix K at b_K=4, no rank cap, Q-precond ON with skip_layers=[0,1,26,27].
Vary V: b_V=2, exact PCA, `--exact-rank-cap-v ∈ {16, 24, 32, 48, 64, none}`.
All cells streaming-safe (per_block, no share_basis).

## Result

| V rank_cap | total ratio | Δppl     | top-1    | verdict |
|-----------:|------------:|---------:|---------:|:-------:|
| 16         | 3.97×       | +29.96 % | 57.94 %  | REJECT  |
| 24         | 3.79×       | +16.89 % | 62.70 %  | REJECT  |
| 32         | 3.72×       | +14.52 % | 63.49 %  | REJECT  |
| 48         | 3.41×       | +6.52 %  | 73.02 %  | REJECT (top-1 < 85 %) |
| 64         | 3.31×       | +4.88 %  | 78.57 %  | REJECT (top-1 < 85 %) |
| **none**   | **2.72×**   | **+2.44 %** | **89.68 %** | **ACCEPT** |

**Every V rank cap hurts quality.** The "free compression" we expected
(from the TurboQuant+ "V compression is free" documentation) did not
materialise via rank truncation. The Sprint 3 ACCEPT baseline at 2.72×
remains the best streaming-safe operating point.

## Why the intuition was wrong

My earlier claim was: "V has no softmax non-linearity, so rank
truncation's errors don't amplify." This is half-true — the linearity
of `p^T V` means errors don't get softmax-saturated, but:

1. The attention probability vector `p` is **itself anisotropic**
   (concentrated on a few positions — attention sink + top-k heavy
   hitters). This is the V analog of Q's anisotropy.
2. Plain L2 PCA picks rank directions that preserve V's **own
   variance**, not V's projection onto `p`. When rank is capped,
   directions that `p` actually queries (attention-salient channels)
   can fall outside the kept subspace.
3. The damage is modest but present — `p^T (V - UU^T V)` has
   non-zero expectation even when `p^T V - p^T UU^T V` has small L2
   norm, because `p`'s support doesn't align with `U`'s column space.

In principle a V-side "P-preconditioning" (analogous to Q-precond on K)
would fix this — whiten V by `E[p p^T]^(1/2)` before PCA, unwhiten
after. But `p` is input-dependent (no static second moment), so the
exact Q-precond calibration procedure doesn't carry over. **No easy
static fix**.

## What "V compression is free" really means

Re-reading the TurboQuant+ README:

> "Compressing the value cache (even down to 2 bits) has zero
> measurable effect on attention quality when key precision is
> maintained."

**"2 bits"** is a **bit-width** reduction, not a rank reduction.
TurboQuant has no PCA skeleton — it's a per-vector scalar quantiser
that maps each coordinate to 2-bit Lloyd-Max indices directly. So
their "V compression is free" applies to *residual-bit compression
only*, not subspace-rank compression.

On our Kakeya architecture:
- **bit-width compression**: same free-ness holds (Sprint 3 b_V=2 vs
  b_V=4 delta is just 4 pp Δppl).
- **rank compression**: NOT free. Every cell rejects.

These are genuinely different axes, and we conflated them.

## Does RSVD change this picture?

No. RSVD and exact PCA both produce a `d_eff × D × 2` byte skeleton.
RSVD's advantages are (a) cheaper fit, (b) natural target_rank cap.
Both are compute-side, not byte-side. RSVD skeletons don't use fewer
bytes than exact PCA skeletons at the same `d_eff`.

## Consequence for the v1.4 roadmap

The streaming-safe Pareto ceiling on DeepSeek D=128 is **2.72×**. To
cross it we need one of:

1. **Cross-layer V skeleton sharing** (prefill-frozen; breaks strict
   online streaming, but compatible with prefill-freeze + decode mode).
   Projected +30-50 % ratio for V stream.
2. **TurboQuant V stream (per-vector, no PCA)**. Streaming-safe by
   construction.  Projected +70-100 % ratio for V stream (b=2 gives
   ~7.1× vs our 3.27×).
3. **Calibration-aware V codec** (Tier 2 from the earlier design
   note).  A small offline-trained affine corrector on V, ~100 KB/model,
   projected to recover most of the rank-cap-induced damage if we
   insist on rank-cap on V.

**Option 2 is cheapest and cleanest.** `turboquant_v_roundtrip` is
already wired; only need a per-stream codec flag (analogous to
`--bit-width-v`) to route K to kakeyaturbo and V to turboquant.

## Files

- `benchmarks/e2e_ppl_pre_rope.py` — `--exact-rank-cap-v` flag
  (V-only rank cap override)
- `reports/v1_4_q_pca/flagship/ds_v_rankcap/` — 5 per-cell JSONs
- This FINDINGS.md
