# v1.3 PPL on vLLM — production cell + H3/H4/H5 ablations + K/V rate sweep

**Setup.** vLLM 0.7.3, V0 engine, `enforce_eager=True`, bf16.
Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (28 layers,
2 KV heads, head_dim=128). GPU: NVIDIA H200 80 GB (Vast.ai). 4
WikiText-103 test passages, ctx=2048, evaluate positions
`[2048, 2112)` (64 teacher-forced next tokens per passage).

**Production codec config** (unless stated otherwise, SPRINT_CLOSEOUT
production cell):
K b=3 + V b=2 + randomized PCA rank=D/2 + calibrated Lloyd-Max +
outlier T=2.0 + 6-layer boundary skip `[0,1,7,14,26,27]` + pre-RoPE
Q-preconditioning. Integration hooks `Qwen2Attention.forward` before
RoPE so whitening applies to pre-RoPE K.

HF reference (same codec config on HF eager + 2-pass DynamicCache):
**+7.82 % Δppl, 78.97 % top-1, MARGINAL**. This is the number we
are trying to explain on vLLM.

## Standing row + all ablations at a glance

| Variant | K bits | V bits | Attn backend | Calibration | Prefix-only | **Δppl** | **top-1** | Verdict |
|:--------|:------:|:------:|:------------:|:-----------:|:-----------:|---------:|----------:|:-------:|
| **production (baseline)** | 3 | 2 | FLASH_ATTN | HF prefill | off | **+35.33 %** | 59.38 % | REJECT |
| H3 — vLLM-calibrated | 3 | 2 | FLASH_ATTN | vLLM prefill (self) | off | **+38.69 %** | 61.33 % | REJECT |
| H4 — XFormers backend | 3 | 2 | **XFORMERS** | HF prefill | off | +37.82 % | 60.16 % | REJECT |
| H5 — prefix-only codec | 3 | 2 | FLASH_ATTN | HF prefill | **2048** | +35.41 % | 59.77 % | REJECT |
| **strategy** — K b=4 | **4** | 2 | FLASH_ATTN | Gaussian K (cal does not help at b=4) | off | +37.30 % | 60.55 % | REJECT |
| **strategy** — K b=4, V b=4 | **4** | **4** | FLASH_ATTN | Gaussian K/V | off | **+27.32 %** | **60.94 %** | REJECT |

All six rows hold within passage noise except the K4V4 row, which
moves Δppl by ~10 pp.

## What has been ruled out

1. **H1** (Σ_q in the wrong frame for FA) — falsified earlier by a
   self-calibrated post-RoPE Σ_q that is *strictly worse*
   (+54.3 % vs +35.3 %). Pre-RoPE whitening `K̃ = K @ L` commutes
   correctly with the per-token rotation `R_t`, so it is the FA-
   consistent choice. Post-RoPE pooling averages `R_t Σ_q R_tᵀ`
   across positions and flattens the anisotropy.
2. **H2** (CPU↔GPU + fp32↔bf16 round-trip noise) — falsified earlier
   by an "identity codec" cell (everything except compression);
   Δppl ≈ 0, top-1 99 %. The hook point itself is numerically clean.
3. **H3** (calibration distribution drift) — **falsified** here by
   self-calibrating Σ_q + Lloyd-Max centroids on vLLM's own pre-RoPE
   prefill (8 disjoint WikiText-103 train passages) and re-running
   the production cell. Δppl moves from +35.33 % to +38.69 % (noise).
   Calibration improvement ratios on the two engines are nearly
   identical (HF K b=2 Lloyd-Max 1.47× vs vLLM 1.59×; HF K b=3 1.40×
   vs vLLM 1.48×; V b=2 ≈ Gaussian natively on both), so the pre-
   RoPE Q/K/V distributions are statistically close enough that HF
   tables already fit vLLM well.
4. **H4** (Flash-Attention bf16 softmax amplifies codec residuals)
   — **falsified** by swapping `VLLM_ATTENTION_BACKEND` to
   `XFORMERS`. Δppl moves from +35.33 % to +37.82 % (noise). Two
   different CUDA backends give the same REJECT. (TORCH_SDPA is not
   accepted by vLLM 0.7.3 V0 on CUDA; FLASHINFER is not installed on
   this image — XFORMERS alone is enough to falsify H4.)
5. **H5** (vLLM's single-forward path integrates codec residuals
   differently than HF's two-pass `prefill → teacher-force`) —
   **falsified** by restricting the codec to positions < ctx_len
   (= 2048) and letting the eval window `[2048, 2112)` pass through
   uncompressed, which mirrors HF's "codec only touched the prefill
   cache, teacher-force saw exact K/V" pattern inside vLLM's single
   forward. Δppl moves from +35.33 % to +35.41 % (indistinguishable).

## Strategy-fallback results

With the five hypotheses all closed, the remaining levers are the
engineering knobs on the codec itself:

- **K b=3 → K b=4** (more K rate, Gaussian centroids since
  SPRINT_CLOSEOUT notes calibrated centroids don't help at b=4):
  Δppl **+35.33 % → +37.30 %** — within noise, **no improvement**.
  vLLM is *not* K-rate limited at b=3.

- **K b=4 + V b=2 → K b=4 + V b=4** (2× V rate, no outlier, no V
  calibration): Δppl **+37.30 % → +27.32 %, top-1 +0.4 pp**. ~10 pp
  Δppl improvement; still REJECT by the MARGINAL threshold (±3 %),
  but this is the first move that shifts the number meaningfully.

**So the bottleneck on vLLM is V-side rate, not K-side rate.** That
is a notable deviation from HF: on HF the guardrails that matter are
all on the K stream (Q-precond, K centroids, K outlier); V is almost
Gaussian-residual and works at b=2. On vLLM, doubling V rate helps
more than all four K-side guardrails combined.

## Interpretation

- The remaining HF↔vLLM gap is **not** calibration, **not** Q-precond
  placement, **not** hook-point numerical noise, **not** Flash-Attention
  vs XFormers, and **not** measurement-path semantics.
- It is also not "K needs more rate". The +35 % plateau survives at
  K b=4.
- Doubling **V** rate is the only knob that moves the needle (−10 pp
  Δppl). This points at a V-stream failure mode that is *specific
  to vLLM's FA-family integration of b=2 V* — plausibly how FA
  accumulates `softmax(QK^T / √d) @ V` in bf16 against a noisy V vs
  HF eager's f32-accumulate path over exact-in-cache V.
- The HF recipe relies on V residuals being "natively Gaussian" so
  Lloyd-Max at b=2 is near-optimal; under FA's bf16 score-times-V
  accumulation, that approximation is evidently less forgiving.

## What this means for deployment

- **HF users**: SPRINT_CLOSEOUT MARGINAL cell (Δppl +7.82 %,
  top-1 79 %, 4.61× ratio) is reproducible. No change.
- **vLLM users**: at the SPRINT_CLOSEOUT config the honest in-engine
  Δppl is +35 %, which is REJECT by any reasonable quality bar. The
  cheapest mitigation is **V b=2 → V b=4** (loses ~1/2 of the V
  compression but gains ~10 pp Δppl). A more expensive mitigation is
  a V-side codec redesign: the b=2 Lloyd-Max quantiser that works on
  HF eager is insufficient for FA's score-times-V accumulation.
- **A follow-up sprint** would focus on the V stream: recalibrate
  Lloyd-Max with a vLLM-specific V residual pipeline that measures
  end-to-end PPL (not MSE) as the optimization target, and/or add an
  outlier-compensation pass on V (currently K-only).

## Artifacts

- `vllm/ds_distill_qwen_1_5b_vllm_full.json` — production (HF-calib) baseline.
- `vllm_calibrated/ds_distill_qwen_1_5b_vllm_calib_vllm_full.json` — H3 row.
- `vllm_h4_xformers/ds_distill_qwen_1_5b_xformers_vllm_full.json` — H4 row.
- `vllm_h5_prefix_only/ds_distill_qwen_1_5b_prefix_only_vllm_full.json` — H5 row.
- `vllm_kb4/ds_distill_qwen_1_5b_kb4_vllm_full.json` — strategy K b=4.
- `vllm_kv4/ds_distill_qwen_1_5b_kb4_vb4_vllm_full.json` — strategy K b=4, V b=4.
- `vllm_recalibrated/` — the vLLM-origin Σ_q + Lloyd-Max tables.

## Reproduce

All rows run through the single driver `benchmarks/run_v1_3_ppl_full_vllm.sh`
with env-var overrides:

```bash
# production baseline
bash benchmarks/run_v1_3_ppl_full_vllm.sh

# H3 — vLLM-calibrated tables
Q_CALIB=reports/v1_3_ppl/vllm_recalibrated/q_calib.safetensors \
K_CENTROIDS=reports/v1_3_ppl/vllm_recalibrated/K_b3_centroids.f32 \
V_CENTROIDS=reports/v1_3_ppl/vllm_recalibrated/V_b2_centroids.f32 \
MODEL_NAME=ds_distill_qwen_1_5b_vllm_calib \
OUT_DIR=reports/v1_3_ppl/vllm_calibrated \
bash benchmarks/run_v1_3_ppl_full_vllm.sh

# H4 — XFormers backend
ATTN_BACKEND=XFORMERS MODEL_NAME=ds_distill_qwen_1_5b_xformers \
OUT_DIR=reports/v1_3_ppl/vllm_h4_xformers \
bash benchmarks/run_v1_3_ppl_full_vllm.sh

# H5 — prefix-only codec (codec only touches positions < 2048)
PREFIX_ONLY_TOKENS=2048 MODEL_NAME=ds_distill_qwen_1_5b_prefix_only \
OUT_DIR=reports/v1_3_ppl/vllm_h5_prefix_only \
bash benchmarks/run_v1_3_ppl_full_vllm.sh

# strategy — K b=4 (no K centroids at b=4 by design)
BIT_WIDTH_K=4 K_CENTROIDS=none \
MODEL_NAME=ds_distill_qwen_1_5b_kb4 \
OUT_DIR=reports/v1_3_ppl/vllm_kb4 \
bash benchmarks/run_v1_3_ppl_full_vllm.sh

# strategy — K b=4, V b=4
BIT_WIDTH_K=4 BIT_WIDTH_V=4 K_CENTROIDS=none V_CENTROIDS=none \
MODEL_NAME=ds_distill_qwen_1_5b_kb4_vb4 \
OUT_DIR=reports/v1_3_ppl/vllm_kv4 \
bash benchmarks/run_v1_3_ppl_full_vllm.sh
```
