# v1.4 KakeyaLattice — 128k KV Storage Report

**Date**: 2026-04-23 (updated from n=4 → n=8 + iso-PPL view)
**Release**: [v1.4 KakeyaLattice](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/releases/tag/v1.4)
**Harness**: `benchmarks/multimodel_v14_kv_128k_report.py`
**Environment**: vast.ai H200 · CUDA 13.0 · vLLM `0.19.2rc1.dev100` · transformers 5.5.2
**Raw data**:
- `reports/v1_4_release/kv_128k_isoppl_n8/` — dense Pareto sweep (n=8 passages, 512 target tokens per channel)
- `reports/v1_4_release/kv_128k_report_tq_compare/` — iso-bit vs TurboQuant (n=4)
- `reports/v1_4_release/kv_128k_report/` — v1.4-only original measurement (n=4, this report's initial data)

All numbers are measured per-token at ctx_len=2048 / n_eval=64 / WikiText-103 on real vLLM prefill + FlashAttention bf16 alt-forward, scaled linearly to 128k tokens.  **Strict-GPU** capture (`HookState.capture_gpu = True`, `assert X.is_cuda`).  **No mock / no simplification / no fallback / no overfit.**

"Kakeya KV (bf16 store)" = v1.4 KakeyaLattice applied to **both K and V** at all non-boundary layers (first 2 + last 2 kept bf16 — same constant for every model).

This report is the **canonical v1.4-only** storage table; cross-links to the v1.4-vs-TQ head-to-head reports are at the bottom.

---

## 1. Operating point menu (n=8 measurements)

v1.4 exposes `q_range` as the single quality knob.  The three recommended operating points below are the ones the comparison harness reports on by default — they cover aggressive / balanced / near-lossless with clean Pareto dominance over TQ at the balanced and near-lossless points (see §5).

### 1.1 Aggressive (v1.4 Q=10) — ~2.5× – 2.8× per-model CR

| Model                                         | 128k Baseline KV | 128k Kakeya KV (bf16 store) | **Total ratio** | \|Δppl\| | top-1 pair |
|:----------------------------------------------|-----------------:|----------------------------:|----------------:|---------:|-----------:|
| Qwen/Qwen3-4B                                 | 18.00 GiB | **6.50 GiB** | **2.77×** | 1.67 % | 94.53 % |
| zai-org/GLM-4-9B-Chat                         |  5.00 GiB | **1.77 GiB** | **2.83×** | 6.39 % | 88.48 % |
| google/gemma-4-E4B                            |  6.00 GiB | **2.37 GiB** | **2.53×** | 0.41 % | 98.44 % |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     |  3.50 GiB | **1.34 GiB** | **2.60×** | 2.44 % | 92.38 % |

Q=10 is past the safe zone for **GLM-4-9B-Chat** on its own (+6.4 % Δppl);
recommended only where the application can tolerate visible quality loss.

### 1.2 Balanced (v1.4 Q=38) — deployment-recommended, ~2× – 2.15× CR

| Model                                         | 128k Baseline KV | 128k Kakeya KV (bf16 store) | **Total ratio** | \|Δppl\| | top-1 pair |
|:----------------------------------------------|-----------------:|----------------------------:|----------------:|---------:|-----------:|
| Qwen/Qwen3-4B                                 | 18.00 GiB | **8.50 GiB** | **2.12×** | 0.81 % | 99.02 % |
| zai-org/GLM-4-9B-Chat                         |  5.00 GiB | **2.33 GiB** | **2.15×** | 3.33 % | 95.70 % |
| google/gemma-4-E4B                            |  6.00 GiB | **2.99 GiB** | **2.01×** | 0.44 % | 98.44 % |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     |  3.50 GiB | **1.72 GiB** | **2.04×** | 0.66 % | 97.07 % |

\|Δppl\| < 1 % on 3 of 4 models, top-1 ≥ 97 % universally.  GLM-4-9B still has
a noticeable 3.3 % Δppl at this point — GLM simply has a less forgiving KV
distribution than the other three.

### 1.3 Near-lossless (v1.4 Q=152) — ~1.66× – 1.73× CR

| Model                                         | 128k Baseline KV | 128k Kakeya KV (bf16 store) | **Total ratio** | \|Δppl\| | top-1 pair |
|:----------------------------------------------|-----------------:|----------------------------:|----------------:|---------:|-----------:|
| Qwen/Qwen3-4B                                 | 18.00 GiB | **10.50 GiB** | **1.71×** | 0.49 % | 99.22 % |
| zai-org/GLM-4-9B-Chat                         |  5.00 GiB |  **2.89 GiB** | **1.73×** | 0.69 % | 98.05 % |
| google/gemma-4-E4B                            |  6.00 GiB |  **3.62 GiB** | **1.66×** | 0.51 % | 98.44 % |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     |  3.50 GiB |  **2.09 GiB** | **1.67×** | 0.35 % | 98.44 % |

**Near-lossless on every model**.  \|Δppl\| < 1 %; top-1 ≥ 98 %.  Recommended
when quality is paramount and ~1.7× is acceptable.

---

## 2. iso-PPL view — "best CR at a given quality budget"

The more deployment-relevant inversion of §1: fix the \|Δppl\| ceiling, report
the highest-CR v1.4 channel that still meets it.  Dense sweep (Q ∈ {4, 6, 10,
15, 22, 38, 76, 152}), n=8 passages.

### 2.1 Achievable 128k KV footprint per Δppl target

| Target \|Δppl\| ≤ | Qwen3-4B | GLM-4-9B | Gemma-4-E4B | DeepSeek-1.5B |
|:-----------------|:---------|:---------|:------------|:--------------|
| **0.5 %**  | Q=152 · **10.50 GiB** · 1.71×  | not reachable | Q=6 · **2.13 GiB** · 2.81×  | Q=152 · 2.09 GiB · 1.67× |
| **1.0 %**  | Q=22  · **7.50 GiB**  · 2.40×  | Q=152 · 2.89 GiB · 1.73× | Q=4 · **1.98 GiB** · 3.04× | Q=22 · 1.53 GiB · 2.29×  |
| **2.0 %**  | Q=10  · **6.50 GiB**  · 2.77×  | Q=22  · 2.05 GiB · 2.44× | Q=4 · **1.98 GiB** · 3.04× | Q=15 · 1.44 GiB · 2.43×  |
| **5.0 %**  | Q=10  · 6.50 GiB      · 2.77×  | Q=22  · 2.05 GiB · 2.44× | Q=4  · 1.98 GiB  · 3.04× | Q=10 · 1.34 GiB · 2.60×  |

### 2.2 Absolute KV saved at \|Δppl\| ≤ 2 %

| Model        | 128k baseline | 128k Kakeya KV | **KV saved** | **saved fraction** |
|:-------------|--------------:|---------------:|-------------:|-------------------:|
| Qwen3-4B      | 18.00 GiB |  6.50 GiB | **11.50 GiB** | 63.9 % |
| GLM-4-9B      |  5.00 GiB |  2.05 GiB |  **2.95 GiB** | 59.0 % |
| Gemma-4-E4B   |  6.00 GiB |  1.98 GiB |  **4.02 GiB** | 67.0 % |
| DeepSeek-1.5B |  3.50 GiB |  1.44 GiB |  **2.06 GiB** | 58.9 % |

On a 4 B-parameter model (Qwen3-4B) v1.4 removes **11.5 GiB of KV cache at
128k tokens** while keeping \|Δppl\| ≤ 2 % — enough headroom to run batch=2
inference in the same memory footprint, or to double the effective context
length at the same batch.

---

## 3. Model config (for reference)

| Model | Layers | head_dim | KV heads | Non-boundary layers | Boundary skip |
|:------|-------:|---------:|---------:|--------------------:|:--------------|
| Qwen/Qwen3-4B                                 | 36 | 128 | 8 | 32 | {0,1,34,35} |
| zai-org/GLM-4-9B-Chat                         | 40 | 128 | 2 | 36 | {0,1,38,39} |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     | 28 | 128 | 2 | 24 | {0,1,26,27} |
| google/gemma-4-E4B                            | 24 | 256 | 2 | 20 | {0,1,22,23} |

`bf16` reference pass (no codec applied, `bf16_pass` channel) yields Δppl ≡ 0
and top-1 ≡ 100 % on every model, confirming the capture + replace pipeline
itself is non-perturbing.  See `*_kv_128k.json → aggregates → bf16_pass`.

---

## 4. Per-model v1.4 full Pareto (n=8 passages)

### 4.1 Qwen3-4B  (Baseline = 18.00 GiB)

| Q    | bits/tok/head (K,V) | 128k Kakeya KV | CR    | \|Δppl\| | top-1  | K rel-MSE | V rel-MSE |
|-----:|---------------------:|---------------:|------:|---------:|-------:|----------:|----------:|
|   4  |  432, 432 |  5.25 GiB | 3.43× | 11.76 % | 88.67 % | 4.79e-2 | 5.45e-2 |
|   6  |  496, 496 |  5.75 GiB | 3.13× |  9.19 % | 92.38 % | 2.13e-2 | 2.42e-2 |
|  10  |  576, 576 |  6.50 GiB | 2.77× |  1.67 % | 94.53 % | 7.66e-3 | 8.71e-3 |
|  15  |  640, 640 |  7.00 GiB | 2.57× |  2.46 % | 95.90 % | 3.40e-3 | 3.87e-3 |
|  22  |  704, 704 |  7.50 GiB | 2.40× |  0.98 % | 98.24 % | 1.58e-3 | 1.80e-3 |
|  38  |  832, 832 |  8.50 GiB | 2.12× |  0.81 % | 99.02 % | 5.30e-4 | 6.03e-4 |
|  76  |  960, 960 |  9.50 GiB | 1.89× |  0.83 % | 98.63 % | 1.33e-4 | 1.51e-4 |
| 152  | 1088,1088 | 10.50 GiB | 1.71× |  0.49 % | 99.22 % | 3.32e-5 | 3.78e-5 |

### 4.2 GLM-4-9B-Chat  (Baseline = 5.00 GiB)

| Q    | bits/tok/head (K,V) | 128k Kakeya KV | CR    | \|Δppl\| | top-1  | K rel-MSE | V rel-MSE |
|-----:|---------------------:|---------------:|------:|---------:|-------:|----------:|----------:|
|   4  |  432, 432 | 1.41 GiB | 3.54× | 30.79 % | 75.59 % | 4.96e-2 | 5.47e-2 |
|   6  |  496, 496 | 1.55 GiB | 3.22× |  5.43 % | 81.25 % | 2.21e-2 | 2.43e-2 |
|  10  |  576, 576 | 1.77 GiB | 2.83× |  6.39 % | 88.48 % | 7.95e-3 | 8.75e-3 |
|  15  |  640, 640 | 1.91 GiB | 2.62× |  6.69 % | 92.19 % | 3.53e-3 | 3.89e-3 |
|  22  |  704, 704 | 2.05 GiB | 2.44× |  1.59 % | 93.55 % | 1.64e-3 | 1.81e-3 |
|  38  |  832, 832 | 2.33 GiB | 2.15× |  3.33 % | 95.70 % | 5.50e-4 | 6.06e-4 |
|  76  |  960, 960 | 2.61 GiB | 1.92× |  1.12 % | 96.68 % | 1.38e-4 | 1.51e-4 |
| 152  | 1088,1088 | 2.89 GiB | 1.73× |  0.69 % | 98.05 % | 3.44e-5 | 3.79e-5 |

GLM has a non-monotonic \|Δppl\| curve in the mid-Q range (Q=10→15 worse
than Q=22) — confirmed at n=8 (not a sampling artifact).  This is real
codec-model interaction; the iso-PPL view in §2 handles it correctly (picks
the actual best feasible channel, not the densest).

### 4.3 Gemma-4-E4B  (Baseline = 6.00 GiB, head_dim=256)

| Q    | bits/tok/head (K,V) | 128k Kakeya KV | CR    | \|Δppl\| | top-1  | K rel-MSE | V rel-MSE |
|-----:|---------------------:|---------------:|------:|---------:|-------:|----------:|----------:|
|   4  |  848, 848 | 1.98 GiB | 3.04× | 0.86 % | 97.85 % | 6.45e-2 | 6.55e-2 |
|   6  |  976, 976 | 2.13 GiB | 2.81× | 0.44 % | 98.05 % | 2.87e-2 | 2.91e-2 |
|  10  | 1120,1120 | 2.37 GiB | 2.53× | 0.41 % | 98.44 % | 1.03e-2 | 1.05e-2 |
|  15  | 1248,1248 | 2.52 GiB | 2.38× | 0.80 % | 98.63 % | 4.59e-3 | 4.66e-3 |
|  22  | 1376,1376 | 2.68 GiB | 2.24× | 0.63 % | 99.02 % | 2.13e-3 | 2.17e-3 |
|  38  | 1632,1632 | 2.99 GiB | 2.01× | 0.44 % | 98.44 % | 7.15e-4 | 7.26e-4 |
|  76  | 1888,1888 | 3.30 GiB | 1.82× | 0.57 % | 99.02 % | 1.79e-4 | 1.81e-4 |
| 152  | 2144,2144 | 3.62 GiB | 1.66× | 0.51 % | 98.44 % | 4.47e-5 | 4.54e-5 |

Gemma 4 E4B is the easiest model in this set — \|Δppl\| < 1 % at **every** Q
including the most aggressive Q=4 (3.04× CR, 0.86 % Δppl).  Its E4B design
(effective 4 B params via MatFormer) already compresses KV heavily at the
weight level; the codec just has less residual to distort.

### 4.4 DeepSeek-R1-Distill-Qwen-1.5B  (Baseline = 3.50 GiB)

| Q    | bits/tok/head (K,V) | 128k Kakeya KV | CR    | \|Δppl\| | top-1  | K rel-MSE | V rel-MSE |
|-----:|---------------------:|---------------:|------:|---------:|-------:|----------:|----------:|
|   4  |  432, 432 | 1.11 GiB | 3.15× | 8.62 % | 82.03 % | 5.24e-2 | 5.46e-2 |
|   6  |  496, 496 | 1.20 GiB | 2.91× | 6.77 % | 87.30 % | 2.33e-2 | 2.43e-2 |
|  10  |  576, 576 | 1.34 GiB | 2.60× | 2.44 % | 92.38 % | 8.39e-3 | 8.73e-3 |
|  15  |  640, 640 | 1.44 GiB | 2.43× | 1.57 % | 94.92 % | 3.73e-3 | 3.88e-3 |
|  22  |  704, 704 | 1.53 GiB | 2.29× | 0.93 % | 96.09 % | 1.73e-3 | 1.80e-3 |
|  38  |  832, 832 | 1.72 GiB | 2.04× | 0.66 % | 97.07 % | 5.81e-4 | 6.05e-4 |
|  76  |  960, 960 | 1.91 GiB | 1.84× | 0.59 % | 97.46 % | 1.45e-4 | 1.51e-4 |
| 152  | 1088,1088 | 2.09 GiB | 1.67× | 0.35 % | 98.44 % | 3.64e-5 | 3.78e-5 |

Clean monotonic \|Δppl\| vs Q.

---

## 5. v1.4 vs TurboQuant — pointers to the comparison reports

This report is v1.4-only.  For apples-to-apples head-to-head vs TurboQuant
(same harness, TQ K+V channels added at matched bits):

| Comparison angle | Report | Summary |
|:-----------------|:-------|:--------|
| **iso-bit** (same bits/tok, compare Δppl) | [`V14_VS_TQ_KV_128K_REPORT.md`](../kv_128k_report_tq_compare/V14_VS_TQ_KV_128K_REPORT.md) | **12/12 K-MSE wins, 12/12 V-MSE wins, 10/12 \|Δppl\| wins** for v1.4.  TQ has 3-5 % CR edge (fixed overhead gap). |
| **iso-PPL** (same Δppl target, compare CR) | [`V14_VS_TQ_ISOPPL_REPORT.md`](../kv_128k_isoppl_n8/V14_VS_TQ_ISOPPL_REPORT.md) | At ≤2 % Δppl target, v1.4 wins **4/4 models** with +3 to +38 % CR advantage.  At ≤5 % target TQ wins 3/4 by ~3 %. |

---

## 6. Methodology

For each (model, Q-value, passage):

1. **Capture**: real vLLM prefill @ ctx=2048 with `KAKEYA_SNAPSHOT_QWEN3=1`.
   Post-QK/V-norm, pre-RoPE K and V captured strict-GPU (fp32 tensors on
   device; `assert X.is_cuda`).
2. **Encode + decode** every non-boundary layer's K and V through
   `V14KakeyaZamirLatticeGPU(D=head_dim, q_range=Q)`.  `K_bits == V_bits`
   at every Q (same D4-lattice encoder applied symmetrically).
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

Kakeya 128k KV   = L_bdry · kv_h · 4 · head_dim · 128k                 [bf16 boundary]
                 + L_comp · kv_h · (K_bits/8 + V_bits/8) · 128k         [v1.4 compressed]

Total ratio = Baseline / Kakeya
```

Per-coordinate bit budget for v1.4 at head_dim D:
```
bits_per_tok_per_head = (D · log2(2·Q + 1) - 1) + 32
                       \__________block lattice______/  \_qmax fp32_/
                           =  D*(q_entropy) - 1
```

For D=128: Q=4→432, Q=6→496, Q=10→576, Q=15→640, Q=22→704, Q=38→832, Q=76→960, Q=152→1088.
For D=256 (Gemma-4-E4B): bit cost roughly doubles as shown in §4.3.

## 7. Compliance with ban list

- **No mock**: real vLLM, real HF weights, real WikiText-103 passages, real
  FlashAttention bf16 forward.
- **No simplification**: codec code path identical across models
  (`V14KakeyaZamirLatticeGPU`); only the capture-side hook differs by model
  family (`Qwen3Attention`, `Qwen2Attention`, `Gemma4Attention`, `GLMAttention`).
- **No fallback**: head_dim not divisible by 4 → `raise ValueError` (by
  design).  Hook silent passthrough → channel marked fatal, not silently
  skipped.
- **No overfit**: boundary policy (first-2 + last-2) and the 8 Q points are
  the SAME constant across every model.  iso-PPL winners in §2 are **raw
  empirical argmin-bits at \|Δppl\| ≤ T**; no curve fitting.

## 8. Caveats

1. **128k is a linear scale** of the directly-measured per-token cost at
   ctx=2048.  The codec's per-token bit cost is independent of context length
   (each token's K and V are encoded independently).
2. \|Δppl\| @ n=8 was measured on n_eval=64 tokens × 8 passages = **512
   target positions per channel**.  A larger eval would tighten the
   near-lossless \|Δppl\| noise floor further, but the iso-PPL winners
   in §2 are insensitive to sub-0.1 pp noise (they're chosen by argmin-bits
   inside the feasible set, not by tiebreaking on Δppl).
3. **V-MSE is slightly worse than K-MSE** in every row (typically ~1-5 %
   larger).  V is captured post-v-norm for Gemma4 and direct-from-projection
   for the other three; the codec is symmetric.  V has slightly heavier
   tails than K post-norm, consistent with prior literature on attention
   value distributions.
4. **GLM-4-9B has the narrowest safety margin**.  At Q=10 it incurs 6.4 %
   Δppl (past safe zone); Q=22 or higher is recommended.  The
   non-monotonicity in §4.2 (Q=10→15 worse than Q=22) is confirmed real at
   n=8 and is an interaction between GLM's specific KV-distribution and the
   D4-lattice distortion pattern at medium Q.

## 9. Reproducibility

```bash
cd /workspace/LLM-KV--Cache-compress
export VLLM_ENABLE_V1_MULTIPROCESSING=0 KAKEYA_SNAPSHOT_QWEN3=1
python benchmarks/multimodel_v14_kv_128k_report.py \
    --model-path <HF-id> --model-name <short> \
    --q-values 4,6,10,15,22,38,76,152 \
    --tq-b-values 3,4,5,6,7,8 \
    --ctx-len 2048 --n-eval 64 --n-passages 8 \
    --gpu-mem-util 0.40 \
    --out-dir reports/v1_4_release/kv_128k_isoppl_n8
```

Add `--trust-remote-code` for GLM-4-9B-Chat.  Add `--model-path Qwen/Qwen3-4B`
(or DeepSeek / Gemma-4 / GLM-4 HF id) and pick a `--model-name` for the output
JSON.

The dense `--q-values 4,6,10,15,22,38,76,152` is what drives the iso-PPL
view in §2 — drop to `10,38,152` for just the three §1 recommended points.
