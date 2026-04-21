# v1.3 native V-stream recipe — new high-ratio MARGINAL champion (NB3sv2)

**Date.** 2026-04-21
**Branch.** `cursor/outlier-compensation-12f5`

## The result: NB3sv2

v1.3's **native** V-stream treatment (V RSVD b=2 with layer-shared
basis) applied alongside K b=3 + outlier T=2.0 + 6-bdry gives:

| Config | Ratio | Δppl | top-1 | Verdict |
|:---|---:|---:|---:|:---:|
| **NB3sv2: K b=3 + K cal + outlier T=2.0 + V RSVD b=2 (shared-basis) + 6 bdry** | **4.61×** | **+7.82 %** | **78.97 %** | **MARGINAL 🎯** |

Measured on DS-Distill D=128, WikiText-103, ctx=2048, 4 passages.

Source JSON: `NB3sv2_noVBesi_T20_Vb2_prerope_kv_b3_randomized_fp16_sk0_sv1.json`

## What changed vs B3-orig

| Axis | B3-orig (V Besi) | **NB3sv2 (V RSVD b=2 shared)** | Delta |
|:---|:---:|:---:|:---:|
| V codec | Besicovitch d=3 m=4 (fixed Haar + per-group mag) | **v1.3 native: RSVD b=2, layer-shared basis** | — |
| Ratio | 4.30× | **4.61×** | **+7 %** |
| Δppl | +5.36 % | +7.82 % | +2.46 pp |
| top-1 | 85.32 % | 78.97 % | −6.35 pp |

The v1.3-native V path restores the byte efficiency that V Besi was
giving up:
- V Besi = 58 B/v regardless of context length (per-vector cost)
- V RSVD b=2 with shared basis ≈ 32 B/v for codes + ~1 B/v amortised
  skeleton = **~33 B/v**, i.e. ~43 % byte savings on V-stream

## Recipe

```bash
python3 benchmarks/e2e_ppl_pre_rope.py \
  --model-path <DS-Distill-1.5B> \
  --model-name NB3sv2 \
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
  --out-dir reports/v1_3_v_rsvd_noBesi
```

All flags already existed in the harness; no code changes needed.

## Position in production matrix

| Use case | Config | Ratio | Δppl | top-1 |
|:---|:---|---:|---:|---:|
| Quality-first (ACCEPT ★) | R3: v1.3 RSVD b=3 + K cal + outlier T=1.5 + V Besi + 6 bdry | 3.73× | +1.91 % | 87.30 % |
| Balanced MARGINAL | B3: v1.3 RSVD b=3 + K cal + outlier T=2.0 + V Besi + 6 bdry | 4.30× | +5.36 % | 85.32 % |
| **Ratio-first MARGINAL (NEW)** | **NB3sv2: v1.3 RSVD b=3 + K cal + outlier T=2.0 + V RSVD b=2 shared + 6 bdry** | **4.61×** | **+7.82 %** | 78.97 % |

NB3sv2 is the **highest-ratio point this PR has measured at Δppl
≤ 10 %**. Use for latency-dominated deployments where top-1 ≥ 75 %
is sufficient.

## Deliverables

- `NB3sv2_noVBesi_T20_Vb2_*.json` — per-cell PPL data (4 passages)
- `NB3sv2_run.log` — raw harness stdout
- `FINDINGS.md` — this file
