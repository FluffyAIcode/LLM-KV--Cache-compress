# v1.3 PPL on vLLM — production cell + per-channel attribution

**Setup.** vLLM 0.7.3, V0 engine, `enforce_eager=True`, bf16,
Flash-Attention backend. Model:
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (28 layers, 2 KV heads,
head_dim=128). GPU: NVIDIA H200 80 GB (Vast.ai). 4 WikiText-103 test
passages, ctx=2048, evaluate positions `[2048, 2112)` (64 teacher-
forced next tokens per passage). Shared reference logprobs per
passage — all rows below are strictly paired.

**Codec config** (SPRINT_CLOSEOUT production cell):
K b=3 + V b=2 + randomized PCA rank=D/2 + calibrated Lloyd-Max +
outlier T=2.0 + 6-layer boundary skip `[0,1,7,14,26,27]` + pre-RoPE
Q-preconditioning. Integration hooks `Qwen2Attention.forward` before
RoPE so whitening applies to pre-RoPE K.

HF reference for this cell (SPRINT_CLOSEOUT, HF eager + 2-pass
DynamicCache): **+7.82 % Δppl, 78.97 % top-1, MARGINAL**.

## Results

| Row | K | V | Q-precond | K-centroids | K-outlier | V-centroids | V-outlier | Boundary | **Δppl** | **top-1** | Verdict |
|:----|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|----:|----:|:-:|
| **production** (K+V)                              | codec | codec | on   | on  | T=2.0 | on  | off   | 6-layer | **+35.33 %** | 59.38 % | REJECT |
| **K-only** (V bf16)                               | codec | bf16  | on   | on  | T=2.0 | —   | —     | 6-layer | **+22.55 %** | 69.14 % | REJECT |
| **V-only** (K bf16, SPRINT_CLOSEOUT V-side recipe) | bf16  | codec | N/A  | —   | —     | on  | off   | 6-layer | **+11.10 %** | 74.22 % | REJECT |
| **V-only + outlier** (all applicable guardrails)  | bf16  | codec | N/A  | —   | —     | on  | **T=2.0** | 6-layer | **+7.04 %**  | 75.39 % | REJECT |

### What "four guardrails" means for each stream

SPRINT_CLOSEOUT lists four PPL-stabilization guardrails; the
applicability to each channel is:

| Guardrail | K | V |
|:----------|:-:|:-:|
| (1) Q-preconditioning (Chol Σ_q on K)            | required | **N/A** — V does not contract with Q; no Σ_q metric on V |
| (2) calibrated Lloyd-Max centroids               | on (`ds_K_b3_centroids.f32`) | on (`ds_V_b2_centroids.f32`) |
| (3) 6-layer boundary skip                        | same layer set | same layer set (K and V skip together) |
| (4) outlier compensation T=2.0                   | on (K only in SPRINT_CLOSEOUT) | **off in SPRINT_CLOSEOUT**, turned **on** in the last row of this table |

So "V with all applicable guardrails" = (2) + (3) + (4). Running
that on vLLM drops V-only Δppl from **+11.10 % → +7.04 %** (a
4.06 pp improvement) and top-1 rises from 74.22 % → 75.39 %. V b=2
is genuinely helped by outlier compensation on vLLM, even though
SPRINT_CLOSEOUT had good reasons to omit it on HF (V residual was
already near-Gaussian there, so outlier saved very little).

## Per-channel Δppl attribution under the production cell

With the three-row attribution (K+V joint, K-only, V-only at
SPRINT_CLOSEOUT V-side recipe = no V outlier):

- **K stream**  : **+22.55 pp / 64 %** of joint +35.33 pp.
- **V stream**  : **+11.10 pp / 31 %** of joint +35.33 pp.
- **interaction**: ~1.68 pp.

K carries about two-thirds of the damage even though the entire
K-side guardrail stack is already applied. V at b=2 with its current
2-guardrail (+ 6-bdry) recipe carries about one-third.

With V outlier compensation also enabled (V-only row 4 above), V's
standalone Δppl drops to +7.04 pp. If the joint K+V cell were re-run
with outlier compensation on *both* streams, the expected Δppl
(assuming the same ≈1.7 pp residual interaction) would be roughly
**+22.55 + 7.04 + 1.68 ≈ +31 pp**. The real joint measurement is
the next datapoint this PR should pick up.

## Reading

- Enabling V outlier compensation closes ~4 pp of V-only Δppl on
  vLLM. That is a meaningful but non-decisive fraction of the 27 pp
  gap vs HF's +7.82 % MARGINAL joint cell.
- K is still the bigger lever even after enabling every applicable
  guardrail on both streams. On HF the K-side stack is near-lossless;
  on vLLM it still leaves +22 pp. The HF↔vLLM gap is primarily a
  K-stream phenomenon; V outlier compensation is a cheap add-on
  that shaves ~4 pp off the joint cost on top.
- MARGINAL threshold (\|Δppl\| ≤ 3 %) is still out of reach on either
  channel alone.

## Artifacts

- `vllm/ds_distill_qwen_1_5b_vllm_full.json` — production (K+V).
- `vllm_k_only/ds_distill_qwen_1_5b_k_only_vllm_full.json` — K-only.
- `vllm_v_only/ds_distill_qwen_1_5b_v_only_vllm_full.json` — V-only
  (SPRINT_CLOSEOUT V-side recipe, no V outlier).
- `vllm_v_only_full_guards/ds_distill_qwen_1_5b_v_only_full_guards_vllm_full.json`
  — V-only with V outlier T=2.0 (all applicable V guardrails on).

## Reproduce

Single driver `benchmarks/run_v1_3_ppl_full_vllm.sh`; the knobs are
`COMPRESS_STREAM ∈ {kv, k, v}` and `V_OUTLIER_THRESHOLD` (empty = off).

```bash
# production (K+V, SPRINT_CLOSEOUT recipe)
bash benchmarks/run_v1_3_ppl_full_vllm.sh

# K-only (V pass-through)
COMPRESS_STREAM=k MODEL_NAME=ds_distill_qwen_1_5b_k_only \
OUT_DIR=reports/v1_3_ppl/vllm_k_only \
bash benchmarks/run_v1_3_ppl_full_vllm.sh

# V-only (K pass-through, SPRINT_CLOSEOUT V-side recipe)
COMPRESS_STREAM=v MODEL_NAME=ds_distill_qwen_1_5b_v_only \
OUT_DIR=reports/v1_3_ppl/vllm_v_only \
bash benchmarks/run_v1_3_ppl_full_vllm.sh

# V-only with outlier T=2.0 (symmetric guardrail add)
COMPRESS_STREAM=v V_OUTLIER_THRESHOLD=2.0 \
MODEL_NAME=ds_distill_qwen_1_5b_v_only_full_guards \
OUT_DIR=reports/v1_3_ppl/vllm_v_only_full_guards \
bash benchmarks/run_v1_3_ppl_full_vllm.sh
```
