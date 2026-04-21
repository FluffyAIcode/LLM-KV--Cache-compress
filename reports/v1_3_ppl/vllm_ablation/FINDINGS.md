# Ablation: what actually causes the vLLM-vs-HF Δppl gap on v1.3 PPL?

**Setup**: DeepSeek-R1-Distill-Qwen-1.5B, vLLM 0.7.3 (Flash-Attention
backend, V0 engine, enforce_eager, bf16), WikiText-103 test split, 4
passages, ctx=2048, n_eval=64, codec config = K b=3 + V b=2 +
outlier T=2.0 + 6-layer boundary skip `[0,1,7,14,26,27]`, calibrated
Lloyd-Max centroids from PR #13.

All cells share the **same** `ref_pls` (codec off, one forward per
passage), so the comparison is strictly paired.

## The four cells

| Cell | What it does | Tests |
|------|--------------|------|
| **identity-pre_qp** | Pre-RoPE whiten → **identity** codec → unwhiten → RoPE | H2: is per-forward CPU↔GPU + fp32↔bf16 noise alone a PPL killer? |
| **codec-no_qp** | Codec ON, **no** whitening | "raw codec" baseline |
| **codec-pre_qp** | Production recipe: pre-RoPE whiten → codec → unwhiten → RoPE | Reproduces PR #15 |
| **codec-post_qp** | RoPE first, then self-calibrated `Σ_q^post` whiten → codec → unwhiten | H1: is Σ_q in the wrong frame because FA works on post-RoPE Q? |

`Σ_q^post` for the post-RoPE cell is accumulated online from the 2
calibration passages' own post-RoPE Q tensors, pooled over the Q
heads of each GQA KV group (num_q=12, num_kv=2), ridge=1e-4,
Cholesky-factored per (layer, kv-head).

## Result

| Cell | Δppl (mean) | top-1 (mean) | Verdict | t_alt / passage |
|------|-------------:|-------------:|:-------:|----------------:|
| **identity-pre_qp**    |   **−0.29 %** | **98.83 %** | **ACCEPT**  | 3.1 s |
| **codec-no_qp**        | **+152.78 %** |   59.38 %   | REJECT      | 15.2 s |
| **codec-pre_qp**       |  **+35.33 %** |   59.38 %   | REJECT      | 18.6 s |
| **codec-post_qp**      |  **+54.28 %** |   57.03 %   | REJECT      | 19.2 s |

## What this tells us

### H2 — "per-forward CPU↔GPU / fp32↔bf16 noise" — **ruled out**

The `identity-pre_qp` cell walks the complete production data path —
fp32 on GPU → fp32 numpy on CPU → whiten with Σ_q Cholesky → write
KKTV → read KKTV back (bit-exact, subprocess skipped) → unwhiten →
fp32 → bf16 → restore on GPU — and records **Δppl = −0.29 %,
top-1 = 98.83 % (ACCEPT)** against a fresh ref per passage. The
numerical envelope of the hook point is therefore negligible: the
harness is **not** adding PPL noise on top of the codec.

### H1 — "Σ_q was calibrated in the wrong frame for FA" — **also ruled out** (not the direction we expected)

If Σ_q needed to be re-calibrated in the post-RoPE frame, the
`codec-post_qp` cell should have moved Δppl **down** from +35.3 %.
It moved **up**, to +54.3 % — i.e. post-RoPE self-calibrated
whitening is **strictly worse** than pre-RoPE calibrated whitening.

Mechanism (math): RoPE is position-dependent block-diagonal
rotation. `Σ_q^post = E_t[R_t Σ_q R_t^T] ≈ avg_t(R_t) Σ_q
avg_t(R_t)^T` after pooling across tokens. The pooling averages
over the rotations, collapsing the pre-RoPE anisotropy and giving
a **flatter** (closer-to-isotropic) pooled covariance than the
true per-token FA metric. Whitening by that pooled Σ_q^post is
closer to "no whitening" than to "the right whitening" — hence
closer to the `codec-no_qp` number (+153 %) than the
`codec-pre_qp` number (+35 %).

The pre-RoPE Σ_q from PR #13 is **already doing the right thing**
at the vLLM hook point: pre-RoPE whitening is FA-consistent
*because* `R_t R_t^T = I` and `(R_t L)(R_t L)^T = R_t Σ_q R_t^T`
for every t — the whitened K contracts with any post-RoPE query
to the same scalar it would have produced under the Σ_q-metric
on pre-RoPE K. Swapping to a pooled post-RoPE Σ_q **loses** that
guarantee.

### So where does the remaining +35 % Δppl actually come from?

Isolating the codec alone on vLLM gives **+153 %** (codec-no_qp).
The four guardrails cut that to **+35 %** — a 4.4× reduction,
compared with HF's 45× reduction from bare v1.3 (+356 %) to v1.3
PPL (+7.8 %).

Guardrails are working on vLLM; they are just much less effective
than on HF. That can't be the Q-precond frame (just disproved); it
must be one of:

1. **Codec parameter drift**. Calibrated `Σ_q`, K centroids, V
   centroids were all fit offline on **HF DynamicCache snapshots**
   of prefill K/V. vLLM's Qwen2 layer produces slightly different
   prefill K/V distributions (different bf16 accumulation, different
   RoPE implementation, different attention bias). The calibrated
   Cholesky and centroids are therefore **off-distribution** on
   vLLM, and the codec has to eat that mismatch. The pre-RoPE hook
   is semantically correct but the *numerical table* it applies is
   calibrated for the wrong engine.

2. **Flash-Attention's post-softmax bf16 accumulation amplifies
   codec residuals differently than HF eager does**. Even if the
   codec's K̂ has the same inner-product MSE as on HF, the attention
   scores produced by FA on bf16 have less headroom than the f32
   reductions used by HF eager, so a fixed per-coord residual
   translates to a bigger Δppl.

These two are separable: re-fit Σ_q + centroids **on vLLM prefill
snapshots** (hypothesis 1). That is the next experiment.

## Implications for the production claim

- The SPRINT_CLOSEOUT "v1.3 PPL MARGINAL @ +7.82 %" number is
  specific to HF eager at this codec config. Until codec calibration
  is re-done on the vLLM prefill distribution, vLLM's honest
  in-engine number is **+35 % Δppl (REJECT)** with the existing HF
  calibration tables.
- The Q-precond / pre-RoPE architectural choice from PR #13 is
  **verified correct** for vLLM too. No need to redesign that.
- The identity-cell ACCEPT tells us there is no engineering
  obstacle to running the codec inside `Qwen2Attention.forward` in
  vLLM — the hook point is numerically safe.

## Follow-up (ordered)

1. **Re-fit `Σ_q` and the K/V Lloyd-Max centroids on vLLM prefill
   snapshots of DS-Distill** (drop-in replacement of the offline HF
   calibration). Re-run `codec-pre_qp`. If +35 % → ≤ +10 %, this
   confirms hypothesis 1 and closes the gap by a re-calibration,
   not a code change.
2. If the gap survives re-calibration, sweep `BIT_WIDTH_K ∈ {3, 4}`
   and `OUTLIER_THRESHOLD ∈ {1.5, 2.0, 2.5}` on vLLM. HF's b=4 is
   ACCEPT-able; we want to know if vLLM needs ~1-bit more headroom.
3. Port the harness to the vLLM V1 engine (chunked-prefill) once
   the calibration question is settled. V0 enforce_eager avoids
   cuda-graph capture issues with the pre-RoPE monkey patch but
   costs throughput.

## Reproduce

```bash
# Vast.ai H200 / vLLM 0.7.3 / CUDA 12.x
git checkout AgentMemory/v1-3-ppl-full-guardrails-vllm-102e
bash benchmarks/run_v1_3_ppl_vllm_ablation.sh
# → reports/v1_3_ppl/vllm_ablation/ds_distill_qwen_1_5b_vllm_ablation.json
```

## Artifact

`ds_distill_qwen_1_5b_vllm_ablation.json` in this directory contains
the full per-cell per-passage metrics.
