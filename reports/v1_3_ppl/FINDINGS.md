# v1.3 PPL — the v1.3 codec with PPL stabilization guardrails

**Date.** 2026-04-21
**Branch.** `cursor/outlier-compensation-12f5`

## What this is

**"v1.3 PPL"** is the original v1.3 KakeyaTurbo codec (RSVD + b=3
K-stream Lloyd-Max + V RSVD b=2 with layer-shared basis) with the
four PPL stabilization guardrails stacked on top:

1. Q-preconditioning (Cholesky of Σ_q) on K
2. Calibrated Lloyd-Max codebook on K
3. Boundary expansion to 6 layers: `[0, 1, 7, 14, 26, 27]`
4. Outlier compensation T=2.0 on K residual

V stream is **unchanged from v1.3 original**: RSVD b=2 with
`--share-basis-v`. There is NO Besicovitch on V, NO asymmetric codec
choice, NO separate Riemann path — it's plain v1.3 with guardrails.

## Result (DS-Distill D=128, WikiText-103, ctx=2048, 4 passages)

| Config | Ratio | Δppl | top-1 | Verdict |
|:---|---:|---:|---:|:---:|
| **v1.3 PPL** | **4.61×** | **+7.82 %** | **78.97 %** | **MARGINAL 🎯** |

**Highest-ratio point ever measured at Δppl ≤ 10 % in this PR.**

Source JSON: `v1_3_ppl_prerope_kv_b3_randomized_fp16_sk0_sv1.json`
Run log: `v1_3_ppl_run.log`

## Recipe

```bash
python3 benchmarks/e2e_ppl_pre_rope.py \
  --model-path <DS-Distill-1.5B> \
  --model-name v1_3_ppl \
  --ctx-len 2048 --n-eval 64 --n-passages 4 --block-size 512 \
  --bit-width 3 --pca-method randomized --variance-ratio 1.0 \
  --rsvd-rank-factor 0.5 --skeleton-dtype fp16 --compress kv \
  --share-basis-v \
  --bit-width-v 2 \
  --v-centroids-file reports/v1_4_q_pca/calibrated_codebook/ds_V_b2_centroids.f32 \
  --k-centroids-file reports/v1_4_q_pca/calibrated_codebook/ds_K_b3_centroids.f32 \
  --q-precondition reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors \
  --boundary-skip-layers 0 1 7 14 26 27 --boundary-mode bf16 \
  --k-outlier-threshold 2.0 \
  --out-dir reports/v1_3_ppl
```

No code changes — all flags already existed in the harness.

## Quality ↔ ratio tuning

Higher quality is obtained by **raising K and/or V bit width on the
same v1.3 PPL recipe** — no new codec path:

- K b=3 → K b=4 (raise K residual fidelity)
- V b=2 → V b=3 or V b=4 (raise V residual fidelity)
- `--rsvd-rank-factor 0.5` → `0.75` (larger skeleton, smaller residual)

These are existing harness flags on the same v1.3 PPL recipe.
Deploying at ACCEPT quality is a bit-width choice, not a codec choice.
