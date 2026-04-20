# End-to-End PPL Validation — Smoke Results and Consequences

**Date**: April 19, 2026
**Branch**: `cursor/v1-3-e2e-ppl-validation-12f5`
**Harness**: `benchmarks/e2e_ppl_validation.py`
**Rust binary**: `kakeyaturbo/target/release/kakeyaturbo-bench` (with new `--dump-decoded` flag)

## Context

Previous paper drafts reported reconstruction MSE as the primary quality metric, with the explicit claim that an ACCEPT MSE verdict (MSE inflation ≤ 1.10× vs exact-PCA b=3 baseline) upper-bounds attention-logit perturbation for bounded-norm queries. A reviewer raised that this has never been directly tested end-to-end on downstream token prediction (`quality-measurement` limitation of the v6/v7 paper).

This report documents the outcome of the first end-to-end test.

## Experimental design

For each WikiText-103 passage of length ≥ ctx_len + n_eval:

1. Prefill `ctx_len=1024` tokens into a reference DynamicCache (bf16, eager attention, CPU).
2. Round-trip every full-attention layer's K and V through the v1.3 Rust codec. The **actual decoded KV** is read back via the new `--dump-decoded` flag — not a Gaussian-noise proxy.
3. Deep-copy both caches; feed the same 64-token continuation through each; compare logits.

Metrics: next-token KL divergence, top-1 agreement rate, PPL ratio.
Verdict thresholds (standard LLM-compression): ACCEPT = |Δppl| ≤ 1% and top-1 ≥ 95%; MARGINAL = |Δppl| ≤ 3% and top-1 ≥ 85%; REJECT otherwise.

## Results on Qwen2.5-0.5B-Instruct, 2 WikiText-103 passages

| Configuration | Codec params | Mean Δppl | Mean top-1 | Verdict |
|---|---|---:|---:|---|
| v1.3 default (paper tier-1) | b=2, rsvd r=D/2=32, vr=0.95 | **+29 086%** | 23.0% | **REJECT** |
| v1.2 default (ACCEPT baseline) | b=3, exact PCA, vr=0.95 | **+11 030%** | 23.8% | **REJECT** |
| v1.2 but exact PCA | b=3, exact PCA, vr=0.95 | **+46 622%** | 17.5% | **REJECT** |
| Max fidelity | b=4, exact PCA, vr=1.0 | **+24 310%** | 19.8% | **REJECT** |

**Every configuration REJECTs on end-to-end PPL**, including the maximum-fidelity setting (keep all PCA components, quantize at 4 bits, exact PCA).

## Direct codec audit on real Qwen2.5 K tensors

To rule out harness bugs, we separately fed a single real K tensor (Qwen2.5-0.5B layer 5, shape [1536, 64]) directly through the Rust binary at three configurations and measured the reconstruction in Python:

| Configuration | Reported MSE | SNR | Correlation |
|---|---:|---:|---:|
| Max fidelity (b=4 vr=1.0 exact) | 7.23e-1 | **9.1×** | **94.4%** |
| Paper default (b=2 vr=0.95 rsvd) | 9.56e-1 | 6.9× | 92.5% |
| v1.2 ACCEPT (b=3 vr=0.95 exact) | 8.55e-1 | 7.7× | 93.3% |

**The codec's reconstruction correlates with the input at only 94.4% even at maximum fidelity.** On a single layer this is a ~13% signal-to-noise ratio in K; compounded through 24 transformer layers the effect on the final logits is catastrophic, which is what the PPL test measures.

## Consequences for the paper

1. **The paper's "ACCEPT" verdict framework is inadequate.** Reconstruction MSE inflation of 1.13× sounds harmless but translates to 77% PPL regression on real next-token prediction --- an order of magnitude worse than the 1% PPL bar the standard LLM-compression thresholds imply.

2. **The paper's central quality claim is empirically false at the currently tested scale.** KakeyaTurbo tier-1 does NOT preserve downstream quality on Qwen2.5-0.5B at ctx=1024. This is not a bit-width issue, not an RSVD issue, and not a tier-1.5 / RoPE-aware-K issue --- even the max-fidelity configuration fails.

3. **The MSE-as-upper-bound-on-attention-logit-perturbation argument in §2.3 is mathematically correct but not tight.** The bound |q^T (k - k̂)| ≤ ||q|| · ||k - k̂|| is a worst-case bound. In practice, the KV cache perturbation interacts nonlinearly with attention softmax, and small per-vector MSE compounds catastrophically through multi-layer attention. This needs to be explicitly stated, not glossed over.

## Consequences for the v1.3 codec

This is NOT a "needs better benchmarks" problem. This is a **"the codec in its current form is not usable as a drop-in KV cache replacement"** finding. Three ways the codec can be repaired:

1. **Abandon the skeleton + residual coding paradigm for very-low-bit-width K** and return to exact-PCA + high-precision residual (e.g. bf16 residual at tiny d_res) on the K stream. This reduces the compression ratio but should recover quality.
2. **Attention-aware fine-tuning of codec parameters per layer**, using the KL divergence of next-token distributions as the optimisation target instead of reconstruction MSE. This aligns the objective with what actually matters.
3. **Abandon training-free compression.** Use the codec as an initialisation for a small amount of fine-tuning that recovers downstream quality. This loses the "post-hoc" selling point.

## Immediate action

- **Do NOT push the v6/v7 paper claims** until the codec is repaired or the paper is honestly rewritten. The current paper claims 6/7 model outperformance at ACCEPT quality; the end-to-end evidence says otherwise.
- **The v1.3 codec as committed is a mathematical framework with a working real-data compression ratio story, but an incomplete and currently broken downstream-quality story.** This needs to be said plainly in the paper.

## What about the GPU / vLLM / SGLang / TensorRT-LLM integration?

Given this finding, GPU integration is premature. There is no point benchmarking the latency of a codec that destroys model output. These tasks are now paused pending codec repair.

## Raw data

- `qwen2_5_0_5b_smoke.json`         — v1.3 paper default (b=2 rsvd)
- `qwen2_5_0_5b_b3.json`             — v1.2 b=3 randomized
- `qwen2_5_0_5b_exact_b3.json`       — v1.2 ACCEPT baseline (b=3 exact)
- `qwen2_5_0_5b_maxfid_b4vr1.json`   — max fidelity (b=4 vr=1.0 exact)
- `qwen2_5_0_5b_maxfid.json`         — max fidelity from earlier iteration

Each JSON contains per-passage metrics and the global aggregate with verdict.

## Reproduction

```bash
# Build Rust binary with --dump-decoded flag
cd kakeyaturbo && cargo build --release --bins && cd ..

# Run harness
python3 benchmarks/e2e_ppl_validation.py \
    --model-path models/Qwen2.5-0.5B-Instruct \
    --model-name qwen2_5_0_5b_default \
    --ctx-len 1024 --n-eval 64 \
    --block-size 512 --bit-width 2 \
    --pca-method randomized --variance-ratio 0.95 \
    --n-passages 2 \
    --out-dir reports/v1_3_rsvd_rope/e2e_ppl_smoke
```
