# V-side RSVD + share_basis + guardrails 1 & 2 — findings (v1.4 Sprint 3.2)

**Date.** 2026-04-17
**Model.** DeepSeek-R1-Distill-Qwen-1.5B (D=128, 28 layers).
**Setup.** pre-RoPE cache, ctx=2048, n_eval=64, 2 WikiText passages,
**streaming-safe**: K uses exact PCA per_block; V uses share_basis
(prefill-frozen share — one basis fit at prefill end, reused during
decode). Q-preconditioning on K only, skip_layers = [0, 1, 26, 27].

## Question

Can the three TurboQuant+-inspired guardrails
(asymmetric K/V, boundary-layer skip, calibrated codebook) enable v1.3's
RSVD tier-1 recipe to achieve both the original 5.8× compression target
**AND** Δppl ≤ 3 % ACCEPT quality?

**First two guardrails tested this sprint.** (Guardrail 3 — calibrated
codebook — is not yet implemented; see outlook below.)

## Byte accounting (theoretical ceiling)

| V config                            | V skel | V total | K+V total | **total ratio** |
|-------------------------------------|-------:|--------:|----------:|----------------:|
| V exact per_block (Sprint 3)        | 145 KB | 313 KB  | 754 KB    | 2.72×           |
| V exact SHARE (Sprint 3.5)          |  48 KB | 216 KB  | 657 KB    | **3.12×**       |
| V **RSVD r=64** per_block           |  73 KB | 177 KB  | 618 KB    | 3.31×           |
| V **RSVD r=64 SHARE**               |  24 KB | 128 KB  | 569 KB    | **3.60×**       |
| bs=2048, V RSVD r=64 SHARE          |  12 KB | 124 KB  | 493 KB    | **4.16×**       |

RSVD + SHARE is the most aggressive V-stream configuration; `target_rank
= D/2 = 64` halves `d_eff` vs exact's uncapped 128, and SHARE
amortises skeleton across all blocks of a layer. Combined ratio gain:
3× skeleton bytes savings (from 145 → 24 KB) with same codes bytes.

## PPL reality check — all four V-RSVD+share cells REJECT

| bs   | b_V | Δppl    | top-1   | KL     | verdict |
|-----:|----:|--------:|--------:|-------:|:-------:|
| 1024 | 2   | +5.73 % | 73.81 % | 0.24   | REJECT  |
| 1024 | 3   | +6.80 % | 76.19 % | 0.16   | REJECT  |
| 2048 | 2   | +10.29 %| 75.40 % | 0.21   | REJECT  |
| 2048 | 3   | +5.22 % | 75.40 % | 0.22   | REJECT  |

Direct comparison at the same (bs, b_V), **changing only V-PCA method**:

| V config                                 | Δppl      | top-1   | verdict |
|------------------------------------------|----------:|--------:|:-------:|
| bs=1024 b_V=2 exact SHARE (Sprint 3.5)   | **−1.68 %** | 88.10 % | ACCEPT  |
| bs=1024 b_V=2 **RSVD r=64** SHARE        | +5.73 %   | 73.81 % | REJECT  |
| bs=1024 b_V=3 exact SHARE                | +1.50 %   | 91.27 % | ACCEPT  |
| bs=1024 b_V=3 **RSVD r=64** SHARE        | +6.80 %   | 76.19 % | REJECT  |

**RSVD's rank cap (d_eff: 128 → 64) single-handedly costs +7.4 pp
Δppl and 14 pp top-1** even under share_basis and the two guardrails.

## Why the guardrails did not save RSVD

The three guardrails each attack a different PPL failure mechanism:

| guardrail             | failure it fixes                        | fixed under V-RSVD+share? |
|-----------------------|-----------------------------------------|:-------------------------:|
| 1. Asymmetric K/V     | softmax saturation on K errors          | ✅ K uses exact per_block + Q-precond |
| 2. Boundary-layer skip| attention-sink outliers on first/last  | ✅ [0, 1, 26, 27] skipped |
| 3. Calibrated codebook| Lloyd-Max Gaussian-prior mismatch      | ❌ not implemented |
| **∅** (missing)       | **rank truncation losing `p`-salient V directions** | **❌ no guardrail attacks this** |

The **fourth failure mechanism** — rank truncation removing the V
channels that the attention distribution `p` queries — is **not
addressed by any of the three guardrails**. This is consistent with
Sprint 3.1, where V rank cap alone (without share, without RSVD)
rejected at vcap = 64 with Δppl = +4.88 %.

## So: can v1.3's 5.8 × with RSVD tier-1 be rescued by the three
guardrails? **No.** At best 3.9-4.2 × by these means, still REJECT.

## Two forward paths remain (both require new work)

### Path A — Add the fourth guardrail: offline affine V-corrector (Tier 2)

Per-layer `A_l` (D × D fp16) + `b_l` (D fp16) applied after decode:
$$
\hat V_{corrected, l} = A_l\,\hat V_{decoded, l} + b_l
$$
Fit offline by `\min \|K_l - A_l \hat V_{decoded, l} - b_l\|_F^2` —
closed-form normal equation, no gradient descent. Per-layer cost:
33 KB for DS.

**Projected**: recover 5-10 pp Δppl on rank-capped V, pushing
V-RSVD+share into ACCEPT. Total ratio target: **4.16 × @ ACCEPT**.

**Streaming-safe** — affine applied per-vector at decode time.

### Path B — Implement guardrail 3 (calibrated V Lloyd-Max codebook)

Replace the `optimal_centroids(bit_width)` function in Rust with a
per-model, per-layer calibrated centroid table. Rust codec change.

**Projected**: 2-5 pp Δppl improvement. **Alone not enough** to rescue
RSVD tier-1; must combine with Path A.

## Currently-accepted operating point (Sprint 3.5 result, repeated for context)

Without touching RSVD, exact PCA + V-share already reaches:

| config                                        | ratio  | Δppl     | top-1   | verdict |
|-----------------------------------------------|-------:|---------:|--------:|:-------:|
| K b=4 exact per_block, V b=2 **exact** SHARE  | **3.12×** | **−1.68 %** | **88.10 %** | **ACCEPT** |
| K b=4 exact per_block, V b=3 **exact** SHARE  | 2.84 × | +1.50 %  | 91.27 % | ACCEPT  |

This is the current Pareto point available with zero additional
implementation work — already a Sprint 3 result (2.72×) improved by
+15 % via V-share and keeping full rank.

## Recommendation

The **RSVD tier-1 + three guardrails path cannot close the 5.8 × ACCEPT
gap alone**. It requires at least one of Path A or Path B (preferably
A). If either is done, **projected 4.16 × @ ACCEPT** is reachable on
DS D=128 — still not 5.8 × (that would require further work on K side
or a different V codec entirely, e.g. TurboQuant V) but closer.

**Honest take**: v1.3 paper's 5.8 × number was a byte-only claim
(MSE-ACCEPT only), not a downstream-PPL claim. The parameter-tuning
path to 5.8 × @ Δppl-ACCEPT on streaming-safe architecture does not
exist. It requires model-aware calibration (Path A or B), which goes
beyond "pure codec tuning" into "per-model offline fit".

## Artefacts

- `benchmarks/e2e_ppl_pre_rope.py` — new `--pca-method-v` CLI flag
  (per-stream PCA method selection)
- `reports/v1_4_q_pca/flagship/ds_v_rsvd_share/` — 4 per-cell JSONs
  this sprint
- This FINDINGS.md

## Reproduce

```bash
python3 benchmarks/e2e_ppl_pre_rope.py \
    --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
    --model-name ds_vrsvd_bs1024_bV2 \
    --ctx-len 2048 --n-eval 64 --n-passages 2 \
    --block-size 1024 --bit-width 4 --bit-width-v 2 \
    --pca-method exact --pca-method-v randomized \
    --rsvd-rank-factor 0.5 \
    --variance-ratio 1.0 \
    --skeleton-dtype fp16 --compress kv \
    --share-basis-v \
    --q-precondition reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors \
    --q-precond-skip-layers 0 1 26 27 \
    --prefill-chunk 1024 --skip-sanity \
    --out-dir reports/v1_4_q_pca/flagship/ds_v_rsvd_share
```
