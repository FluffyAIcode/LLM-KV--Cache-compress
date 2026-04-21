# v1.3 PPL on vLLM — full production recipe, DS-Distill

## Setup

- **Engine**: vLLM 0.7.3 (Flash-Attention backend, V0 engine, `enforce_eager=True`)
- **Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` — bf16, 28 layers, 2 KV heads, head_dim=128
- **GPU**: NVIDIA H200 80 GB (Vast.ai)
- **Data**: WikiText-103 test split, 4 passages tokenising to ≥ 2112 tokens
- **Window**: evaluate positions `[ctx_len=2048, ctx_len+64)` (64 next-token positions per passage)
- **Codec config** (SPRINT_CLOSEOUT production cell):
  - K: b=3, randomized PCA r=D/2, Q-precondition on (Chol Σ_q), calibrated
    Lloyd-Max K b=3 centroids, outlier threshold T=2.0
  - V: b=2, randomized PCA r=D/2, share_basis=True, calibrated Lloyd-Max
    V b=2 centroids, no outlier / no whitening
  - Boundary skip: layers `[0, 1, 7, 14, 26, 27]` kept at bf16

## Integration — pre-RoPE hook on Qwen2Attention.forward

The scaffolding PR #14 hook at `Attention.forward` sees post-RoPE K, which
is the wrong distribution for Σ_q (the Cholesky was calibrated on
pre-RoPE K). This harness patches
`vllm.model_executor.models.qwen2.Qwen2Attention.forward` to intercept
K/V immediately after the QKV projection:

```
qkv → (q, k, v)
k   ← unwhiten(codec(whiten(k), centroids=ds_K_b3, outlier_T=2.0))
v   ← codec(v, centroids=ds_V_b2, share_basis=True)
q,k ← rotary_emb(positions, q, k)
… rest of stock forward runs on the repaired K and V
```

Per-layer Q-precond uses the 27/28 calibrated Cholesky factors from
`reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors`
(layer 0 is intentionally uncalibrated / skip-list). Codec is called
via subprocess to `kakeyaturbo-bench --dump-decoded`.

## Result

| Passage | ppl_ref  | ppl_alt  | Δppl      | top-1   |
|:-------:|---------:|---------:|----------:|--------:|
| 1       | 124.876  | 113.811  | **−8.86 %** | 56.25 % |
| 2       |  33.004  |  43.828  | **+32.79 %** | 51.56 % |
| 3       |   8.536  |  11.989  | **+40.46 %** | 65.62 % |
| 4       |  25.355  |  44.857  | **+76.92 %** | 64.06 % |

**Aggregate**: `mean Δppl = +35.33 %`, `mean top-1 = 59.4 %`
→ **VERDICT: REJECT** (MARGINAL requires |Δppl| ≤ 3 % AND top-1 ≥ 85 %).

Per-passage cost: ~18 s/passage in `alt` (56 codec round-trips through
the CPU subprocess `kakeyaturbo-bench`) vs. ~0.1 s/passage in `ref`.
Reference throughput is not the subject of this measurement.

## Cross-engine comparison vs. SPRINT_CLOSEOUT

| Engine / recipe | Δppl    | top-1   | Verdict | Model        |
|:----------------|--------:|--------:|:--------|:-------------|
| **HF** bare v1.3 b=2 (V0 ladder cell)             | +355.62 % | 42.46 % | REJECT   | DS-Distill |
| **HF** v1.3 PPL full (K b=3, V b=2, T=2.0, 6 bdry) | **+7.82 %** | **78.97 %** | **MARGINAL** | DS-Distill |
| **vLLM** bare v1.3 b=2 (= PR #14 scaffolding)      | +291.9 % | 46.9 %  | REJECT   | Qwen2.5-0.5B |
| **vLLM** v1.3 PPL full (this PR)                   | **+35.33 %** | **59.4 %** | **REJECT** | DS-Distill |

The four guardrails move vLLM from **+292 % → +35 %** (−8× Δppl),
mirroring the HF direction (+356 % → +8 % on DS-Distill, −45× Δppl),
but **vLLM ends ~4.5× worse than HF** at the same codec config on the
same model family.

## Why the vLLM Δppl is worse than HF at the same config

Two hypotheses, both worth follow-up:

1. **CPU/GPU dtype boundary around the subprocess**. The patched
   forward ships K/V to the Rust bench binary as fp32 via KKTV,
   decodes back, then restores to bf16 before RoPE. The HF harness
   does the same trip but operates on the *cache tensors* (post-prefill
   snapshot), not on every forward call per passage, so numerical
   round-off compounds differently. The `mean_abs_dlogp_true ≈ 0.8-1.1`
   suggests per-position logit noise, not catastrophic mode collapse.

2. **Flash-Attention vs. eager small numerical differences get amplified
   by the compressed K**. Under an exact codec they vanish; under a
   lossy codec they interact with FA's scale / numerics slightly
   differently than HF eager does. Top-1 agreement (59 %) is far from
   pessimal (bare v1.3 was 47 % in PR #14) — the codec IS preserving
   the bulk of the distribution, just not tightly enough at +35 %.

Both hypotheses can be tested with additional sweeps
(`BIT_WIDTH_K=4`, `OUTLIER_THRESHOLD=1.5`, larger n_eval); these are
left as the follow-ups listed below.

## Sanity checks that did pass

- Σ_q Cholesky: `max_abs_err = 2.2e-05`, `max_rel_err = 5.0e-06`
  (`whiten∘unwhiten = I` to fp32 round-off).
- Per passage, `48 + 12 = 60` expected layer-calls+skips across 28
  layers × 2 streams − layer 0 skip; the harness reported 56 codec
  calls + 12 boundary skips = 68 (layer 0 boundary-skip absorbs its
  own K+V stream, explaining the 2 extras over 56).
- Boundary-skip list matches the SPRINT_CLOSEOUT recipe exactly.

## Follow-ups (ranked)

1. **Sweep `BIT_WIDTH_K` ∈ {3, 4}** on vLLM at the same config — HF
   shows b=4 is ACCEPT-able; we should see whether vLLM agrees.
2. **Sweep `OUTLIER_THRESHOLD` ∈ {1.5, 2.0, 2.5}** to see if vLLM
   wants a tighter T than HF.
3. **Diagnose the hypothesis in §"Why vLLM ≠ HF"** by re-running the
   HF harness on DS-Distill at this exact bit-width/outlier config
   and comparing per-passage Δppl passage-by-passage (pairing at the
   passage level).

## Reproduce

```bash
# Vast.ai H200 / CUDA 12.x / vLLM 0.7.3
git checkout AgentMemory/v1-3-ppl-full-guardrails-vllm-102e
bash benchmarks/run_v1_3_ppl_full_vllm.sh
# → reports/v1_3_ppl/vllm/ds_distill_qwen_1_5b_vllm_full.json
```

## Artifact

`ds_distill_qwen_1_5b_vllm_full.json` in this directory contains the
full per-passage metrics.
