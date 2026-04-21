# v1.3 PPL on vLLM — production cell + per-channel attribution

**Setup.** vLLM 0.7.3, V0 engine, `enforce_eager=True`, bf16,
Flash-Attention backend. Model:
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (28 layers, 2 KV heads,
head_dim=128). GPU: NVIDIA H200 80 GB (Vast.ai). 4 WikiText-103 test
passages, ctx=2048, evaluate positions `[2048, 2112)` (64 teacher-
forced next tokens per passage). Shared reference logprobs per
passage — all three rows below are strictly paired.

**Codec config** (SPRINT_CLOSEOUT production cell):
K b=3 + V b=2 + randomized PCA rank=D/2 + calibrated Lloyd-Max +
outlier T=2.0 + 6-layer boundary skip `[0,1,7,14,26,27]` + pre-RoPE
Q-preconditioning. Integration hooks `Qwen2Attention.forward` before
RoPE so whitening applies to pre-RoPE K.

HF reference for this cell (SPRINT_CLOSEOUT, HF eager + 2-pass
DynamicCache): **+7.82 % Δppl, 78.97 % top-1, MARGINAL**.

## Results

| Row | Compressed stream(s) | K | V | **Δppl (mean)** | **top-1 (mean)** | Verdict |
|:----|:--------------------|:-:|:-:|----------------:|-----------------:|:-------:|
| **production** | K + V       | codec | codec  | **+35.33 %** | 59.38 %  | REJECT |
| K-only         | K (V bf16)  | codec | bf16   | **+22.55 %** | 69.14 %  | REJECT |
| V-only         | V (K bf16)  | bf16  | codec  | **+11.10 %** | 74.22 %  | REJECT |

Per-passage detail is in the three JSON artifacts below.

## Per-channel attribution

Under the assumption that K and V codec errors contribute roughly
additively at the Δppl-resolution we measure:

| Source | Δppl contribution | top-1 degradation |
|:-------|------------------:|------------------:|
| V stream (b=2)        | +11.10 pp  | 25.78 pp loss from K+V |
| K stream (b=3 + Q-precond + calib + outlier) | +22.55 pp | 15.08 pp loss from K+V |
| ∑ (K-only + V-only)   | **+33.65 pp** | — |
| measured K+V (production) | **+35.33 pp** | 59.38 % |
| residual interaction  | +1.68 pp | — |

The two channels' Δppl contributions sum to within ~1.7 pp of the
joint K+V cell, so the channels interact only mildly at this codec
config.

**K carries two-thirds of the damage** (+22.55 of +35.33 ≈ 64 %)
even though the whole v1.3 PPL guardrail stack (Q-preconditioning,
calibrated K Lloyd-Max centroids, outlier compensation T=2.0, 6-layer
boundary skip) was designed *for* the K stream. On HF that full
stack compresses K down to MARGINAL at +7.82 % joint K+V. On vLLM
the same stack leaves +22.55 % on K alone.

**V carries one-third of the damage** (+11.10 of +35.33 ≈ 31 %) at
b=2 with Lloyd-Max centroids and `--share-basis-v`, no outlier, no
whitening. Top-1 degradation on V-only is notably smaller (74.22 %
vs K-only's 69.14 %): V residual errors distort logits but don't
reorder the one-best as aggressively as K errors do.

## What this tells us

1. **Both channels independently exceed MARGINAL** on vLLM at the
   SPRINT_CLOSEOUT config (MARGINAL bar: \|Δppl\| ≤ 3 %). Neither
   stream alone survives the quality gate. HF gets to MARGINAL
   because its joint Δppl is +7.82 %; vLLM is materially further
   than that on either channel alone.
2. **K is the bigger lever on vLLM.** Improving K compression is
   worth more Δppl reduction than improving V, the opposite order
   from HF experience where the SPRINT_CLOSEOUT ladder shows V is
   "natively Gaussian" and K carries all the guardrail complexity.
   The cheapest path to narrowing the HF↔vLLM gap is therefore a
   v1.3-PPL-on-vLLM specific **K-side redesign**, not the V-side
   redesign that the earlier bit-width sweep (PR history) suggested
   via the KV b=4 datapoint.
3. **V at b=2 has non-trivial residual** (+11.10 pp) despite
   matching the HF recipe exactly. The `share-basis-v + Lloyd-Max at
   b=2` configuration that is near-lossless on HF leaves measurable
   PPL damage under vLLM's Flash-Attention accumulation.

## Deployment note

The production cell retains REJECT on vLLM. Ratio claim is unchanged
from PR #15: the **K-side** is the first target for a
vLLM-specific fix.

## Artifacts

- `vllm/ds_distill_qwen_1_5b_vllm_full.json` — production (K+V) row.
- `vllm_k_only/ds_distill_qwen_1_5b_k_only_vllm_full.json` — K-only row.
- `vllm_v_only/ds_distill_qwen_1_5b_v_only_vllm_full.json` — V-only row.

## Reproduce

All three rows run through the single driver
`benchmarks/run_v1_3_ppl_full_vllm.sh` with the `COMPRESS_STREAM`
env var:

```bash
# production
bash benchmarks/run_v1_3_ppl_full_vllm.sh

# K-only (V pass-through)
COMPRESS_STREAM=k MODEL_NAME=ds_distill_qwen_1_5b_k_only \
OUT_DIR=reports/v1_3_ppl/vllm_k_only \
bash benchmarks/run_v1_3_ppl_full_vllm.sh

# V-only (K pass-through)
COMPRESS_STREAM=v MODEL_NAME=ds_distill_qwen_1_5b_v_only \
OUT_DIR=reports/v1_3_ppl/vllm_v_only \
bash benchmarks/run_v1_3_ppl_full_vllm.sh
```
