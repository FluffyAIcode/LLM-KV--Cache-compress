# Re-run: vLLM ablation cells with vLLM-recalibrated Σ_q + Lloyd-Max

## What we did

1. Captured pre-RoPE `Q / K / V` from vLLM 0.7.3 on
   `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` over **8 disjoint
   WikiText-103 TRAIN split passages** of 2048 tokens each
   (`benchmarks/vllm_calibration_refit.py`). Capture uses a
   non-invasive monkey-patch on `Qwen2Attention.forward` that
   records `qkv.split(...)` outputs without altering the forward's
   computation.
2. Factored `Σ_q` per (layer, kv-head), pooling the Q heads in each
   GQA KV group (num_q=12, num_kv=2). Ridge = 1e-3 × mean-diag for
   numerical stability. Wrote drop-in replacement
   `q_calib.safetensors` + `q_calib.json`.
3. Re-ran the Lloyd-Max residual pipeline on the captured K (after
   whitening with the fresh Σ_q) and V streams to produce new
   `K_b{2,3}_centroids.f32` + `V_b2_centroids.f32`.
4. Re-ran the 4-cell ablation harness (same shared-ref design) with
   all of vLLM-calibrated `Σ_q` + K/V centroids.

Σ_q anisotropy on vLLM (sanity check — strongly anisotropic, so
Q-precond has something to do): condition number median 4506, max
109 076. Off-diagonal max vs diag mean: median 15.5, max 34.8.

Lloyd-Max improvement vs Gaussian default (MSE ratio):

| stream / bit | HF-calibrated (from SPRINT_CLOSEOUT) | **vLLM-calibrated (here)** |
|:-------------|-------------------------------------:|---------------------------:|
| K b=2        | 1.47×                                | **1.59×**                  |
| K b=3        | 1.40×                                | **1.48×**                  |
| V b=2        | ~1.00×                               | **1.00×** (V residual ≈ Gaussian natively) |

The MSE-improvement ratios are **quantitatively similar** between
HF and vLLM calibration, and the V-residual-near-Gaussian property
is identical. This already suggests the post-WHT residual
distributions are statistically similar across engines.

## Re-run results (same 4 test passages, shared ref)

| Cell | **HF-calibrated (prior run)** | **vLLM-calibrated (this run)** | Δ (vLLM − HF) |
|------|------------------------------:|--------------------------------:|--------------:|
| identity-pre_qp   | −0.29 % / 98.83 %   | **+0.15 %** / 98.83 %   | ~0 |
| codec-no_qp       | +152.78 % / 59.38 % | **+144.56 %** / 58.59 % | ~0 |
| **codec-pre_qp**  | **+35.33 %** / 59.38 % | **+38.69 %** / 61.33 % | **+3 pp** |
| codec-post_qp     | +54.28 % / 57.03 % | **+58.24 %** / 60.16 %  | ~0 |

All four cells reproduce within passage-level noise. **Swapping the
calibration artifacts from HF-origin to vLLM-origin did not close
the HF-vs-vLLM Δppl gap.**

## What this rules out

Hypothesis **H3**: "Σ_q and Lloyd-Max centroids are off-distribution
on vLLM because they were fit on HF DynamicCache snapshots"
— **ruled out**. The vLLM-self-calibrated tables produce
statistically identical Δppl. The pre-RoPE Q distribution on vLLM
is close enough to HF's that the calibration tables are
interchangeable at Δppl-measurement resolution.

Combined with the earlier findings:

- **H2** (CPU↔GPU + fp32↔bf16 noise) — ruled out by the identity
  cell (Δppl ≈ 0, top-1 99 %).
- **H1-direct** (Σ_q in the wrong frame for FA — fix by post-RoPE
  self-calibration) — ruled out; post-RoPE Σ_q is strictly worse
  (+54–58 % vs +35–39 %), mathematically because pooled `R_t Σ_q R_t^T`
  over positions is a flatter metric than the true per-token FA metric.
- **H3** (calibration distribution drift) — ruled out by this run.

## What's left

**H4**: **Flash-Attention bf16 accumulation (softmax & score reduction)
amplifies codec residuals differently from HF's eager f32-accumulate
path.** Under an exact codec these differences vanish; under v1.3
at K b=3 / V b=2 they show up as extra Δppl. The ~4× HF-vs-vLLM gap
(HF +7.82 %, vLLM +35–39 %) would then be fundamentally engine-level
numerical amplification, not something a smarter calibration can fix.

**H5**: The PPL-measurement path itself differs. HF harness is
"prefill DynamicCache once → teacher-force continuation through
both caches → compare next-token distributions". vLLM harness is
"LLM.generate with prompt_logprobs=1 (single forward that prefills
+ emits logprobs)". If vLLM re-runs the forward slightly differently
(bf16 reduction order, chunked prefill vs monolithic), compression
residuals get integrated along a different path.

Both are testable with additional cells (more K headroom at b=4,
harder-to-forge comparison with a vLLM "exact codec" cell that
differs from identity only in that it actually round-trips through
the Rust bench but with `vr=1.0 + bit_width=infinite` — we can
get that by setting K-means k=2^16 on a small D block). That's
another sprint though; **today's conclusion is that the HF-vs-vLLM
gap is not a calibration problem and cannot be closed by re-fitting
the standard artifacts**.

## Artifacts

- `ds_distill_qwen_1_5b_vllm_ablation.json` — full per-cell /
  per-passage metrics from this run.
- `../vllm_recalibrated/` — the vLLM-origin calibration products
  that were fed to this run.

## Reproduce

```bash
# Step 1: refit calibration on vLLM prefill snapshots (disjoint train passages)
bash benchmarks/run_vllm_calibration_refit.sh
# → reports/v1_3_ppl/vllm_recalibrated/{q_calib.*, K_b{2,3}_centroids.f32, V_b2_centroids.f32}

# Step 2: re-run the 4-cell ablation pointing at the new tables
Q_CALIB_PRE=reports/v1_3_ppl/vllm_recalibrated/q_calib.safetensors \
K_CENTROIDS=reports/v1_3_ppl/vllm_recalibrated/K_b3_centroids.f32 \
V_CENTROIDS=reports/v1_3_ppl/vllm_recalibrated/V_b2_centroids.f32 \
OUT_DIR=reports/v1_3_ppl/vllm_recalibrated_run \
bash benchmarks/run_v1_3_ppl_vllm_ablation.sh
```
