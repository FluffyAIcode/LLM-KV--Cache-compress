# v1.4 KakeyaLattice — 128k KV Storage Report

**Date**: 2026-04-23
**Branch**: `AgentMemory/v1-4-kv-128k-report-c478`
**Harness**: `benchmarks/multimodel_v14_kv_128k_report.py`
**Environment**: vast.ai H200 · CUDA 13.0 · vLLM `0.19.2rc1.dev100` · transformers 5.5.2
**Raw data**: `reports/v1_4_release/kv_128k_report/*.json` + `*.log`

All numbers are measured per-token (ctx=2048, n_eval=64, 4 WikiText-103 passages)
and scaled linearly to 128k tokens.  Strict-GPU, no mock / no simplification /
no fallback / no overfit.

"Kakeya KV (bf16 store)" = **v1.4 KakeyaLattice applied to both K and V** at all
non-boundary layers (first 2 + last 2 kept bf16 for stability — same constant
for every model in this table).

---

## Operating point #1 — Aggressive (v1.4 Q=10, ~3.6× per-head K/V compression)

| Model | 128k Baseline KV | 128k Kakeya KV (bf16 store) | **Total ratio** |
|:------|-----------------:|----------------------------:|----------------:|
| Qwen/Qwen3-4B                                 | 18.00 GiB | 6.50 GiB | **2.77×** |
| zai-org/GLM-4-9B-Chat                         |  5.00 GiB | 1.77 GiB | **2.83×** |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     |  3.50 GiB | 1.34 GiB | **2.60×** |
| google/gemma-4-E4B                            |  6.00 GiB | 2.37 GiB | **2.53×** |

Accuracy at this point:

| Model | \|Δppl\| | top-1 pair | K rel-MSE | V rel-MSE |
|:------|---------:|-----------:|----------:|----------:|
| Qwen/Qwen3-4B                                 | 1.45 %   | 94.92 %    | 7.67e-3 | 8.71e-3 |
| zai-org/GLM-4-9B-Chat                         | 6.52 %   | 85.55 %    | 7.94e-3 | 8.74e-3 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     | 2.22 %   | 91.02 %    | 8.37e-3 | 8.73e-3 |
| google/gemma-4-E4B                            | 0.33 %   | 98.83 %    | 1.03e-2 | 1.05e-2 |

Q=10 is the **most aggressive compression** the codec supports before quality
degrades rapidly — GLM-4-9B-Chat shows it's at or past the edge for this model
(+6.5 % Δppl, 85.55 % top-1).  Safe for Gemma 4; noticeable for Qwen3-4B /
DeepSeek; unsafe for GLM-4-9B-Chat.

---

## Operating point #2 — Balanced (v1.4 Q=38, ~2.5× per-head K/V compression)

| Model | 128k Baseline KV | 128k Kakeya KV (bf16 store) | **Total ratio** |
|:------|-----------------:|----------------------------:|----------------:|
| Qwen/Qwen3-4B                                 | 18.00 GiB | 8.50 GiB | **2.12×** |
| zai-org/GLM-4-9B-Chat                         |  5.00 GiB | 2.33 GiB | **2.15×** |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     |  3.50 GiB | 1.72 GiB | **2.04×** |
| google/gemma-4-E4B                            |  6.00 GiB | 2.99 GiB | **2.01×** |

Accuracy at this point:

| Model | \|Δppl\| | top-1 pair | K rel-MSE | V rel-MSE |
|:------|---------:|-----------:|----------:|----------:|
| Qwen/Qwen3-4B                                 | 0.65 %   | 100.00 %   | 5.31e-4 | 6.03e-4 |
| zai-org/GLM-4-9B-Chat                         | 2.51 %   | 95.31 %    | 5.50e-4 | 6.05e-4 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     | 0.45 %   | 97.27 %    | 5.79e-4 | 6.05e-4 |
| google/gemma-4-E4B                            | 0.15 %   | 99.22 %    | 7.16e-4 | 7.26e-4 |

**Balanced point.  Deployment-recommended.**  \|Δppl\| < 1 % for all models
except GLM-4-9B-Chat (2.5 %); top-1 ≥ 95 % universally.

---

## Operating point #3 — Near-lossless (v1.4 Q=152, ~1.9× per-head K/V compression)

| Model | 128k Baseline KV | 128k Kakeya KV (bf16 store) | **Total ratio** |
|:------|-----------------:|----------------------------:|----------------:|
| Qwen/Qwen3-4B                                 | 18.00 GiB | 10.50 GiB | **1.71×** |
| zai-org/GLM-4-9B-Chat                         |  5.00 GiB |  2.89 GiB | **1.73×** |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     |  3.50 GiB |  2.09 GiB | **1.67×** |
| google/gemma-4-E4B                            |  6.00 GiB |  3.62 GiB | **1.66×** |

Accuracy at this point:

| Model | \|Δppl\| | top-1 pair | K rel-MSE | V rel-MSE |
|:------|---------:|-----------:|----------:|----------:|
| Qwen/Qwen3-4B                                 | 0.48 %   |  99.61 %   | 3.32e-5 | 3.77e-5 |
| zai-org/GLM-4-9B-Chat                         | 0.90 %   |  98.05 %   | 3.44e-5 | 3.79e-5 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     | 0.28 %   |  98.05 %   | 3.63e-5 | 3.78e-5 |
| google/gemma-4-E4B                            | 0.32 %   |  99.22 %   | 4.48e-5 | 4.54e-5 |

Near-lossless.  \|Δppl\| < 1 % on every model; top-1 ≥ 98 %.  Recommended when
quality is paramount and ~1.7× is acceptable.

---

## Model config (for reference)

| Model | Layers | head_dim | KV heads | Boundary skip |
|:------|-------:|---------:|---------:|:--------------|
| Qwen/Qwen3-4B                                 | 36 | 128 | 8 | {0,1,34,35} |
| zai-org/GLM-4-9B-Chat                         | 40 | 128 | 2 | {0,1,38,39} |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     | 28 | 128 | 2 | {0,1,26,27} |
| google/gemma-4-E4B                            | 24 | 256 | 2 | {0,1,22,23} |

---

## Methodology

For each (model, Q-value, passage):

1. **Capture**: run a real vLLM prefill @ ctx=2048 with `KAKEYA_SNAPSHOT_QWEN3=1`.
   Post-QK/V-norm, pre-RoPE K and V are captured strict-GPU (fp32 tensors on device;
   `assert X.is_cuda`).
2. **Encode + decode** every non-boundary layer's K AND V through
   `V14KakeyaZamirLatticeGPU(D=head_dim, q_range=Q)`.  Bits-per-token-per-head
   is whatever the codec actually writes (same value for K and V since both use the
   same D4-lattice encoder at the same Q).
3. **Replace** K and V inside the same vLLM alt-forward pass.  A fire-count
   guard aborts any channel where the hook silently passes through live K/V
   (no hidden fallback).
4. **Measure**: \|Δppl\|, top-1 pair agreement vs the bf16 capture pass,
   per-tensor rel-MSE, cos.  Codec wall-time recorded.

**Per-token storage**:
```
raw_bytes_per_head_per_token = 16 · head_dim / 8 = head_dim · 2  (bf16)
Baseline 128k KV = L · kv_h · (raw_bytes_K + raw_bytes_V) · 128k
                 = L · kv_h · 4 · head_dim · 128k

Kakeya 128k KV   = L_bdry · kv_h · (raw_bytes_K + raw_bytes_V) · 128k      [bf16 boundary]
                 + L_comp · kv_h · (K_bits/8 + V_bits/8) · 128k             [v1.4 compressed]

Total ratio      = Baseline / Kakeya
```

`K_bits == V_bits` at Q=10 / 38 / 152 → 576 / 832 / 1088 bits (for hd=128);
1120 / 1632 / 2144 bits (for hd=256).

## Compliance with ban list

- **No mock**: real vLLM, real HF weights, real WikiText-103 passages, real FA
  bf16 forward.
- **No simplification**: codec code path identical across models
  (`V14KakeyaZamirLatticeGPU`); only the capture-side hook differs by model
  family (Qwen3 / Qwen2 / Gemma4 / GLM).
- **No fallback**: head_dim not divisible by 4 → `raise ValueError` (by
  design).  Hook silent passthrough → channel marked fatal, not silently
  skipped.
- **No overfit**: boundary policy (first-2 + last-2) and the three Q points
  are the SAME constant across every model.

## Caveats

1. **128k is a linear scale** of the directly-measured per-token cost at
   ctx=2048.  The codec's per-token bit cost is independent of context length
   (each token's K and V are encoded independently).
2. \|Δppl\| was measured on n_eval=64 tokens per passage × 4 passages = 256
   target positions per channel.  A larger eval would tighten the \|Δppl\|
   noise floor, especially at near-lossless Q=152.
3. V-MSE is slightly worse than K-MSE in every row.  V is captured post-v_norm
   for Gemma4 and direct-from-projection for the other three; the codec itself
   is symmetric.  This suggests V has a slightly different statistical shape
   than K (heavier tails), which is consistent with prior findings that V's
   information density per coordinate is marginally higher.
4. GLM-4-9B-Chat has the tightest \|Δppl\| tolerance.  The aggressive Q=10
   point (6.5 % Δppl) is NOT recommended for GLM deployment; Q=38 or Q=152
   are.
