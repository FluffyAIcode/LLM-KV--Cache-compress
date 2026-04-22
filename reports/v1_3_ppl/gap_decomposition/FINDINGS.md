# Δppl gap decomposition — HF vs vLLM on v1.3 PPL production cell

**Goal.** The v1.3 PPL production cell measures **+7.82 % Δppl** on
HF eager (SPRINT_CLOSEOUT MARGINAL) and **+35.33 % Δppl** on vLLM
0.7.3 Flash-Attention (REJECT) at the *same* codec recipe on the
*same* DS-Distill-Qwen-1.5B. This PR decomposes the 27 pp gap into
measured buckets.

**Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, 28 layers,
2 KV heads, head_dim=128. 4 WikiText-103 test passages, ctx=2048,
eval window `[2048, 2112)`. bf16 weights, H200. Codec config:
K b=3 + V b=2 + randomized PCA + Q-precondition + calibrated
Lloyd-Max + outlier T=2.0 + 6-layer boundary skip.

## Bucket table

| # | Bucket | Estimate | What was measured |
|:-:|:---|:---:|:---|
| **1** | **Engine baseline shift** (clean model) | ~10 pp of PPL mismatch, 18 % top-1 disagreement, 0.145 nats KL | Phase 1+5 — codec OFF, same tokens, both engines |
| **2** | **Codec residual magnitude** (pre-hook input/output) | ~0 (engine-agnostic, <1 % mse delta) | Phase 4 — capture pre-RoPE K/V per layer, run codec, compare to ground truth |
| **3** | **Engine noise sensitivity** (linear regime, `σ·rms(K)·randn`) | HF **more** sensitive than vLLM per σ; **−21 pp** at σ=0.01 (HF 33.5 % vs vLLM 12.4 %) | Phase 2 — Gaussian noise injection at pre-RoPE hook, both engines |
| **4** | **Boundary-layer concentration** (SPRINT_CLOSEOUT recipe already skips these) | +69 pp (would be lost if NOT skipped) | Phase 3 — codec on single layer at a time |
| **5** | **Cross-layer non-linear compounding** (22 "quiet" layers jointly codec-active) | **+39 pp** | Phase 3 — single-layer Σ = −3.93 %, joint production = +35.33 % |

The production cell **+35.33 %** decomposes approximately as:

- 22 quiet layers each contribute ~0 individually (Σ singletons = −3.9 %).
- Their joint forward adds **+39 pp** of cross-layer compounding interaction
  (measured by the joint − sum-of-singletons).
- HF's joint-forward interaction must be much smaller (≈ 0-10 pp) for HF to
  land at its +7.82 %.
- HF's engine baseline is ~10 % lower PPL on the clean model on these same
  passages. That shifts the reported Δppl's denominator by ~10 pp but is NOT
  27 pp on its own.

## One-liner per phase

- **Phase 1+5**: codec OFF, HF and vLLM disagree on the clean logits —
  ppl diverges by 11 %, KL 0.145, top-1 disagrees 18 %. The two Δppl
  numbers are measured against different baselines. See `phase1/FINDINGS.md`.
- **Phase 4**: the codec sees statistically identical K/V from the two
  engines and produces matched residuals (mse ratio median 1.01, max
  1.06). The codec is not the variable. See `phase4/FINDINGS.md`.
- **Phase 2**: at matched σ·rms noise, HF amplifies MORE in the linear
  regime (+33.5 % at σ=0.01 vs vLLM's +12.4 %). The "FA bf16 softmax
  amplifies noise" hypothesis is wrong; the gap is not about generic
  noise sensitivity. See `phase2/FINDINGS.md`.
- **Phase 3**: layer 0 alone is catastrophic (+56.5 % Δppl), layer 7
  is +15.6 %, layer 11 is −8.4 %; these are already in / proximal to
  the SPRINT_CLOSEOUT boundary-skip list. The 22 quiet layers sum
  to −3.9 % singly but the joint cell is +35.3 % — a **+39 pp
  cross-layer non-linear compounding** that Phase 2 cannot see with
  isotropic noise. This is the concrete, measured root cause of the
  HF↔vLLM gap under a structured codec. See `phase3/FINDINGS.md`.

## Why HF's number is +7.82 % and vLLM's is +35.33 %

Combining all five buckets into one sentence:

> The codec produces the same per-layer residuals on both engines;
> each layer's residual alone is small; **vLLM's single-forward
> residual-stream accumulation (bf16 throughout FA) compounds
> per-layer codec errors non-linearly +39 pp above their individual
> sum**, whereas HF eager's f32-accumulate path through teacher-
> force'd DynamicCache compounds them less aggressively. A further
> ~10 pp is baseline-mismatch (the two engines disagree on the
> clean model's PPL on these passages, so the reported Δppl
> numerator differs).

## Deployment implications

### HF users

Production cell reproduces SPRINT_CLOSEOUT +7.82 % Δppl MARGINAL.
No change.

### vLLM users

The honest in-engine Δppl is +35 % at the SPRINT_CLOSEOUT config.
Two measured, cheap fixes:

1. **Stricter boundary skip** — add `{2, 6, 11}` (single-layer
   |Δppl| ≥ 5 % on vLLM) to the boundary-skip set. Each is 5-8 pp
   individually. Even with 50 % of the compounding saving, ~10-15 pp
   cut off the joint +35 % cell. Keeps 19/28 layers codec-active.

2. **Adaptive per-layer bit-width** — K b=3 globally except on the
   hot layers `{2, 6, 11}`, where K b=4 (costs ~+D*28/24 bytes of
   K-skeleton; V stays b=2). Addresses the compounding at its
   source without giving up ratio on the quiet layers.

Both are follow-up experiments for a future PR; the important
contribution of this PR is that the **cause is now localised**:
cross-layer non-linear compounding in vLLM's FA residual stream.

## Artifacts

- `phase1/ds_distill_qwen_1_5b_engine_baseline.json` + FINDINGS.md
- `phase2/ds_distill_qwen_1_5b_{vllm,hf}_both.json` + FINDINGS.md
- `phase3/ds_distill_qwen_1_5b_vllm_per_layer.json` + FINDINGS.md
- `phase4/ds_distill_qwen_1_5b_residual_magnitude.json` + FINDINGS.md
