# Phase 4 — codec residual magnitude on HF vs vLLM prefill snapshots

**Question.** Does the codec produce **bigger errors** on vLLM-prefill
K/V than on HF-prefill K/V at the same layer? If yes, part of the
27 pp Δppl gap traces to the codec itself; if no, the gap is
entirely on the engine-side response to the codec.

**Setup.** DS-Distill-Qwen-1.5B, 4 WikiText-103 test passages of
2048 tokens each (8192 pre-RoPE K/V vectors per layer per engine).
Captured with non-invasive monkey patches at the pre-RoPE point in
each engine's attention module (the exact same semantic point both
harnesses use at run time). Codec: production v1.3 PPL
(Q-precond + calibrated Lloyd-Max + outlier T=2.0 for K;
Lloyd-Max + share_basis for V). 22 full-attention layers analysed
(6 boundary layers skipped).

## Result — per-layer v/h mse ratio

| Stream | median (v/h) | max (v/h) | min (v/h) |
|:------:|:------------:|:---------:|:---------:|
| K mse  |   **1.012**  |  1.056    |  0.98     |
| V mse  |   **1.018**  |  1.056    |  0.96     |

Mean input magnitudes:

| | vLLM |  HF  | delta |
|:-:|-----:|-----:|------:|
| mean \|K\| | 0.9806 | 0.9743 | +0.64 % |
| mean \|V\| | 0.6765 | 0.6690 | +1.12 % |

Per-layer K-mse ranges (codec, production config) ≈ 0.4 to 0.9 in
both engines; V-mse ranges ≈ 0.07 to 1.8 (layer 25 is the biggest V
outlier in both engines — same layer peaks on both).

## Finding — codec residuals are engine-independent

The codec sees statistically identical K/V distributions from HF and
vLLM (input mean magnitudes differ by <1.2 %), and produces
statistically identical reconstruction errors (mse ratio median 1.01,
max 1.06). This includes Q-preconditioning, where Σ_q was
calibrated on HF DynamicCache data — the whitening works equally
well on the vLLM-sourced K (confirming Phase 1 of PR #15 which
reached the same conclusion via an end-to-end run).

Therefore: the 27 pp HF↔vLLM Δppl gap **is not** explained by the
codec seeing different inputs, producing different residuals, or
benefiting from different calibration. The codec is the same
operation on both sides and it makes the same error.

## What's left

Combined with Phase 1 (clean-model KL = 0.145, PPL rel gap −11 %),
the gap decomposition so far:

- **Engine baseline mismatch (Phase 1)**: −11 % PPL rel on clean model,
  82 % top-1 agreement, 0.145 top-20 KL. This is a mix of eager-vs-FA
  bf16 accumulation and the HF eager Qwen2 path's spurious SWA flag
  slowing some numeric routes. Not fixable from our side.
- **Codec residual mismatch (Phase 4)**: **~0** (median ratio 1.01).
  The codec is engine-agnostic.

**27 pp ≠ −11 % baseline shift + 0 codec mismatch.** The remaining
damage — and it is the majority — must be in the engine's *response
to* a known, fixed codec residual: exactly the "engine sensitivity
curve" Phase 2 is designed to measure. A ~0.3 nat/token fixed codec
residual + a different softmax/score accumulation can easily turn
into a different Δppl on each engine.

## Data

`ds_distill_qwen_1_5b_residual_magnitude.json` — per-layer K/V
`mse_vs_ground_truth`, `relnorm`, `codec_mean_block_mse`, and input
magnitudes for both engines (22 layers × 2 streams × 2 engines).
