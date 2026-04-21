# v1.3 PPL on vLLM — production cell, HF-calibrated vs vLLM-calibrated

**Setup.** vLLM 0.7.3, Flash-Attention backend, V0 engine,
`enforce_eager=True`, bf16. Model:
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (28 layers, 2 KV heads,
head_dim=128). GPU: NVIDIA H200 80 GB via Vast.ai. 4 WikiText-103
test passages, ctx=2048, evaluate positions `[2048, 2112)` (64
teacher-forced next tokens per passage).

**Codec config** (SPRINT_CLOSEOUT production cell):
K b=3 + V b=2 + randomized PCA rank=D/2 + calibrated Lloyd-Max +
outlier T=2.0 + 6-layer boundary skip `[0,1,7,14,26,27]` + pre-RoPE
Q-preconditioning. Integration hooks `Qwen2Attention.forward`
before RoPE so whitening applies to pre-RoPE K.

## Only retained row

| Calibration source | Δppl (mean) | top-1 (mean) | Verdict |
|:-------------------|------------:|-------------:|:-------:|
| HF DynamicCache prefill snapshots (from PR #13, used as-is) | **+35.33 %** | 59.38 % | REJECT |
| vLLM pre-RoPE prefill snapshots (self-calibrated, disjoint train passages) | **+38.69 %** | 61.33 % | REJECT |

Passage-by-passage for the vLLM-calibration run:

| Passage | ppl_ref | ppl_alt | Δppl | top-1 |
|:-:|--:|--:|--:|--:|
| 1 | 124.876 | 114.261 | **−8.50 %** | 60.94 % |
| 2 |  33.004 |  41.578 | **+25.98 %** | 48.44 % |
| 3 |   8.536 |  15.175 | **+77.79 %** | 68.75 % |
| 4 |  25.355 |  40.436 | **+59.48 %** | 67.19 % |

Reference (HF-engine on the HF harness at the same codec config):
**+7.82 % Δppl, 78.97 % top-1 (MARGINAL)**.

## What this row rules out

**H3** (calibration distribution drift — "Σ_q and Lloyd-Max centroids
were fit on HF DynamicCache snapshots, vLLM prefill distributions are
different, calibration tables are therefore off-distribution for vLLM"):
**ruled out**. Swapping in vLLM-origin calibration does not move
Δppl in the direction that would matter (+38.7 % vs +35.3 %, within
passage-level noise). Lloyd-Max improvement ratios on the two
engines are also nearly identical:

| stream / bit | HF-calibrated (PR #13) | vLLM-calibrated (this PR) |
|:-|-:|-:|
| K b=2 | 1.47× | 1.59× |
| K b=3 | 1.40× | 1.48× |
| V b=2 | ~1.00× | 1.00× |

So the pre-RoPE Q/K/V distributions vLLM produces are close enough
to HF's that the HF tables work as well as self-calibrated ones.

Earlier ablation rounds already ruled out:

- **H1** — Σ_q in the wrong frame for Flash-Attention.
  Post-RoPE self-calibrated Σ_q is strictly WORSE (+54 % vs +35 %)
  because pooled `R_t Σ_q R_tᵀ` over positions flattens the
  per-token FA metric. Pre-RoPE whitening already commutes
  correctly with the position-dependent rotation.
- **H2** — CPU↔GPU + fp32↔bf16 round-trip noise. Identity codec
  cell (everything except compression) reports Δppl −0.29 %,
  top-1 98.83 % (ACCEPT). Hook point is numerically clean.

## What's left

The remaining HF-vs-vLLM Δppl gap is **not** fixable by re-calibration
and **not** an artifact of the hook's numeric pipeline. Two viable
hypotheses to test next (follow-ups):

- **H4** — Flash-Attention bf16 softmax/score reduction amplifies
  codec residuals more than HF eager's f32-accumulate. Engine-level.
- **H5** — `prompt_logprobs=1` (vLLM single-forward) integrates codec
  residuals differently than HF's `prefill → teacher-force`
  two-pass. Measurement-path level.

If both H4 and H5 fail to close the gap, deployment falls back to
either raising K bit-width (sweep b=4) or modifying the codec to be
less sensitive to bf16-FA score accumulation.

## Reproduce

```bash
# HF-calibrated (as shipped) production cell
bash benchmarks/run_v1_3_ppl_full_vllm.sh
# → reports/v1_3_ppl/vllm/ds_distill_qwen_1_5b_vllm_full.json

# vLLM-calibrated production cell (using this PR's re-calibrated tables)
Q_CALIB=reports/v1_3_ppl/vllm_recalibrated/q_calib.safetensors \
K_CENTROIDS=reports/v1_3_ppl/vllm_recalibrated/K_b3_centroids.f32 \
V_CENTROIDS=reports/v1_3_ppl/vllm_recalibrated/V_b2_centroids.f32 \
MODEL_NAME=ds_distill_qwen_1_5b_vllm_calib \
OUT_DIR=reports/v1_3_ppl/vllm_calibrated \
bash benchmarks/run_v1_3_ppl_full_vllm.sh
# → reports/v1_3_ppl/vllm_calibrated/ds_distill_qwen_1_5b_vllm_calib_vllm_full.json
```

## Artifacts

- `vllm/ds_distill_qwen_1_5b_vllm_full.json` — HF-calibrated row.
- `vllm_calibrated/ds_distill_qwen_1_5b_vllm_calib_vllm_full.json`
  — vLLM-calibrated row.
- `vllm_recalibrated/` — the Σ_q + Lloyd-Max tables fit from vLLM
  prefill snapshots.
