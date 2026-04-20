# Streaming-safe Pareto — correcting an overstatement

**Date.** 2026-04-17
**Branch.** `cursor/v1-3-rsvd-rope-aware-12f5`

## What was wrong

Earlier PR/commit messages reported a "5.44× compression at Δppl=+17%"
cell on DeepSeek-R1-Distill as the flagship result under Q-precondition.
That cell used **`share_basis=True`**, which is **not** strictly
streaming-safe for token-by-token decode in a paged-attention runtime:
`share_basis=True` requires pooling every block of a layer to fit one
PCA basis, which on the decode path means waiting for the full prefill
**and** freezing the basis against future decode drift.

This contradicts the paper's own streaming contract
(§1, `kakeyaturbo.tex` line 61): a 512-token uncompressed hot tail plus
asynchronous block-ready encode, with **per-block** PCA fit at the
moment each block becomes full.  Only `share_basis=False` cleanly meets
that contract for both prefill and decode.

## Corrected streaming-safe numbers

All cells use: `pca_method=exact`, `share_basis=False` (per-block),
`skeleton_dtype=fp16`, `variance_ratio=1.0`, `Q-preconditioning ON`
with `skip_layers=[0, 13, 15]` (outlier attention-sink mitigation),
pre-RoPE cache, 2 passages × 64 eval tokens.

### DeepSeek-R1-Distill-Qwen-1.5B (D=128, flagship proxy)

| bs   | bw | ratio  | Δppl      | top-1    | KL     | verdict  |
|------|---:|-------:|----------:|---------:|-------:|:--------:|
| 1024 | 3  | 2.72×  | +4.40 %   | 84.92 %  | 0.068  | MARGINAL |
| 1024 | 4  | **2.32×** | **−1.54 %** | **88.89 %** | 0.026 | **ACCEPT** |
|  512 | 3  | 1.96×  | +1.12 %   | 87.30 %  | 0.057  | ACCEPT   |
|  512 | 4  | 1.75×  | −2.07 %   | 89.68 %  | 0.019  | ACCEPT   |
|  256 | 3  | 1.26×  | −1.41 %   | 86.51 %  | 0.044  | ACCEPT   |
|  256 | 4  | 1.17×  | −1.69 %   | 92.06 %  | 0.015  | ACCEPT   |
|  128 | 3  | 0.74×  | +3.72 %   | 91.27 %  | 0.038  | MARGINAL |
|  128 | 4  | 0.71×  | −1.67 %   | 90.48 %  | 0.012  | ACCEPT   |

**Pareto frontier (streaming-safe):**

| bs   | bw | ratio | Δppl     | verdict  |
|------|---:|------:|---------:|:--------:|
| 1024 | 3  | 2.72× | +4.40 %  | MARGINAL |
| **1024** | **4** | **2.32×** | **−1.54 %** | **ACCEPT** |
| 512  | 4  | 1.75× | −2.07 %  | ACCEPT   |

### Qwen2.5-0.5B (D=64, reference)

Already streaming-safe in the v1.4 Sprint 1 report — best ACCEPT cell:
**2.06× @ Δppl = −0.56 %, top-1 = 92.86 %** (bs=512 b=4 per_block exact).

## Head-to-head in streaming-safe mode

| system | operating point | ratio | Δppl | top-1 | mode |
|---|---|---:|---:|---:|---|
| **TurboQuant+ production (Qwen2.5-1.5B Metal, from README)** | turbo3 symmetric | **4.6×** | +1.06 % | (n/a) | streaming; uses boundary-layer protection + norm correction |
| TurboQuant+ production (same) | turbo4 symmetric | 3.8× | +0.23 % | (n/a) | streaming |
| TurboQuant+ production (same) | turbo2 symmetric | 6.4× | +6.48 % | (n/a) | streaming |
| **TurboQuant raw reference Python (our harness)** | b=3 symmetric | 4.57× | +7.7×10⁵ % | 9 % | streaming but no production trick |
| **KakeyaTurbo + Q-precond** (DS D=128) | bs=1024 b=4 | **2.32×** | **−1.54 %** | **88.9 %** | **streaming, per_block** |
| KakeyaTurbo + Q-precond (DS D=128) | bs=1024 b=3 | 2.72× | +4.40 % | 84.9 % | streaming, per_block |
| KakeyaTurbo + Q-precond (Qwen2.5 D=64) | bs=512 b=4 | 2.06× | −0.56 % | 92.9 % | streaming, per_block |

The honest gap in streaming-safe mode is **TurboQuant+ production 4.6×
vs KakeyaTurbo 2.32×**, a factor of ~2 at the same Δppl-ACCEPT quality
tier.  **TurboQuant+ leads by a factor of ~2 on streaming-safe
compression**, not trails by orders of magnitude.

The earlier "3-6 orders of magnitude advantage" we reported was against
the **raw reference** Python implementation, without TurboQuant+'s three
production mitigations (boundary-layer protection, norm correction,
asymmetric K/V).  When TurboQuant runs with those mitigations it is a
strong codec; without them, as a pure algorithm-level comparison, it
compounds catastrophically.

## Three production tricks we have not yet applied to KakeyaTurbo

From `turboquant_plus/README.md` (independently validated across 30+
testers, multiple hardware):

1. **Boundary layer protection.** "Protecting the first 2 + last 2
   layers at higher precision recovers 37-91 % of the quality gap."
   Our `skip_layers=[0, 13, 15]` is a partial instance (layer 0 is
   universal; 13/15 were outlier-selected on this specific model).
   We should test the systematic `[0, 1, L-2, L-1]` policy.
2. **Norm correction.** We effectively already have this in the Lloyd-Max
   decoder side, but the Q-precond + norm-correction interaction has
   not been measured.
3. **Asymmetric K/V.** "V compression is free. All quality degradation
   comes from K compression." TurboQuant+ production uses
   `q8_0 K + turbo3 V` or similar.  Our ablation has so far used
   symmetric `b=3`/`b=4` on both streams.  **This is the biggest
   unexplored lever** and all three mitigations are streaming-safe by
   construction (per-vector / per-block, no cross-block dependency).

## Estimated head-to-head if we match TurboQuant+'s engineering

Rough back-of-envelope (to be validated):

- Current: bs=1024 b=4 symmetric per_block → 2.32× @ ACCEPT
- Add asymmetric K (b=4) + V (b=2) per_block → K stream ~2.3×, V stream ~5-7×, **combined ~3.5-4.5×** at same PPL
- Add systematic boundary-layer skip `[0, 1, L-2, L-1]` → small further PPL margin, may enable dropping b=3 symmetric back in at higher ratio

Target: **3.5-4.5× @ ACCEPT on DS D=128, streaming-safe**, which would
be within ~0.5× of TurboQuant+ production's 4.6× on a different base
model.

## Action items

1. **Retract the 5.44× claim** in the PR body and the FLAGSHIP_FINDINGS.md
   (keep the data but label it "prefill-frozen-basis mode", not
   "streaming").
2. **Update TURBOQUANT_PPL_COMPARISON.md** with a prominent caveat that
   the comparison is raw-vs-raw algorithms, not production-vs-production.
3. **Run asymmetric K/V ablation**: K with Q-precond ON, V without
   (since V has no inner-product distortion). Per-stream bit widths
   b_K ∈ {3, 4}, b_V ∈ {2, 3}. All per_block. Target the 3.5-4.5× ACCEPT
   regime.
4. **Run systematic boundary-layer skip**: compare `[0]`, `[0, 1, L-2, L-1]`,
   `[0, L-2, L-1]` as skip lists (don't pick outlier layers
   post-hoc; use the universal policy).

## Artefacts

- `reports/v1_4_q_pca/flagship/deepseek_streaming/` — 8-cell streaming-safe
  grid on DS, 2 passages, Q-precond OFF/ON pairs.
- `reports/v1_4_q_pca/flagship/deepseek_streaming/ds_stream_kv_qp_summary.json`
