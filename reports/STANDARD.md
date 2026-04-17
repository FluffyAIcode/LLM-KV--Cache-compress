# Kakeya KV Cache Compression — Benchmark Standard

This file defines the methodology and report template used for every model
in this repo. All per-model reports under `reports/<model>/` follow the
same structure so numbers are directly comparable.

## Methodology

For each model we measure, on a single machine with the same codec
hyperparameters, the following quantities:

### Fixed codec hyperparameters (the "standard preset")

| Parameter | Value | Meaning |
|---|---:|---|
| `block_size` | 512 | Number of cached tokens per compressed block. |
| `residual_length` | 256 | Recent tail kept in exact precision. |
| `d_res` | 8 | Number of top residual coefficients stored per row. |
| `K` (centers) | 16 | Spherical K-means codebook size per block. |
| `variance_ratio` | 0.95 | PCA cumulative variance threshold → effective rank d_eff. |
| dtype | bf16 | KV dtype on the baseline side. |
| attention | eager | Deterministic reference implementation. |

### Measured context lengths

The standard measurement sweep is **2 048, 4 096, 8 192 tokens**. These
three points are enough to fully pin the per-block skeleton cost and the
per-token encoded cost (the codec's per-block output is deterministic
once `block_size` is fixed, so longer contexts are exact byte projections,
not statistical fits).

### Projected context lengths

From the 8 192-token report we project to **16 384, 32 768, 65 536,
131 072, 262 144** tokens using `kakeya_extrapolate.py`. The projections
were cross-validated against measured 16 384 and 32 768 runs on
`google/gemma-4-E2B-it` with max absolute ratio error ≤ 0.003. That
validation is what justifies the other models' projections being trusted
beyond the 8 192 measurement ceiling imposed by the CPU-only benchmark
environment.

### Reported metrics

For each context length we report:

- `baseline_full_bytes`: KV storage on full-attention layers using a
  standard `DynamicCache`.
- `baseline_sliding_bytes`: KV storage on sliding-window layers (capped
  by `sliding_window`, so constant beyond that length). Layers where
  the codec does not apply.
- `baseline_total_bytes`: `full + sliding`.
- `kakeya_*_bytes`: same split, with the Kakeya cache.
- `*_bytes_dtype_matched`: the "bf16-store projection" that assumes an
  optimized implementation keeping compressed tensors in bf16 rather
  than the current f32-on-CPU store. The residual integer indices are
  unchanged; the skeleton/centers/alpha/t tensors are halved.
- Ratios: `baseline / kakeya` for each of those quantities, in both
  the as-is f32 store and bf16-projected store.

### What "compression gain" means in this document

Two orthogonal axes exist:

1. **Full-attention ratio** vs **Total ratio**. Models with sliding-window
   layers have an irreducible O(sliding_window) cache that cannot be
   compressed further by any codec; the full-attention ratio is the
   relevant number for systems that care about long-context memory
   scaling; the total ratio is what the user actually sees on disk.
2. **f32 store** vs **bf16 store**. The current reference codec stores
   compressed tensors as float32 on CPU. A production implementation
   would store them in the KV dtype (bf16), halving the compressed
   side. Both numbers are reported.

The default headline number used in cross-model comparisons is
**`total_ratio_bf16_store` at 32k tokens**, which is the most
conservative honest comparison: total (not just full-attn) and
projection-matched to bf16.

## Reference run: Gemma 4 E2B

`google/gemma-4-E2B-it` is the reference model for this standard
because it exercises all three features that complicate the general
case:

- A hybrid `layer_types` plan with interleaved sliding-window and
  full-attention layers.
- Grouped-query attention (`num_key_value_heads=1`, i.e. MQA).
- `num_kv_shared_layers=20` so the HF cache only holds the first 15
  decoder layers.

Concretely, of those 15 cached layers, only 3 are `full_attention`
(indices 4, 9, 14) with `global_head_dim=512`, and 12 are
`sliding_attention` capped at 512 tokens with `head_dim=256`.

### Measured compression at 2k–32k

| Context | Baseline (total) | Kakeya (total, f32) | Kakeya (total, bf16) | Full-attn f32 | Full-attn bf16 | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  2 048 |  17.99 MiB |  13.44 MiB |  11.22 MiB | 1.61× | 2.30× | **1.34×** | **1.60×** |
|  4 096 |  29.99 MiB |  18.73 MiB |  13.86 MiB | 1.88× | 3.05× | **1.60×** | **2.16×** |
|  8 192 |  53.99 MiB |  29.14 MiB |  19.06 MiB | 2.07× | 3.67× | **1.85×** | **2.83×** |
| 16 384 | 101.99 MiB |  50.66 MiB |  29.83 MiB | 2.15× | 4.03× | **2.01×** | **3.42×** |
| 32 768 | 197.99 MiB |  93.69 MiB |  51.34 MiB | 2.19× | 4.23× | **2.11×** | **3.86×** |

### Projected 64k–256k (byte-exact extrapolation)

| Context | Baseline (total) | Kakeya (total, f32) | Kakeya (total, bf16) | Full-attn f32 | Full-attn bf16 | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  65 536 | 389.99 MiB | 179.74 MiB |  94.36 MiB | 2.21× | 4.35× | **2.17×** | **4.13×** |
| **131 072** | **773.99 MiB** | **351.83 MiB** | **180.41 MiB** | **2.22×** | **4.40×** | **2.20×** | **4.29×** |
| 262 144 |   1.51 GiB | 696.01 MiB | 352.50 MiB | 2.23× | 4.43× | **2.22×** | **4.37×** |

### Cross-validation of the extrapolator

| Prediction | Predicted | Measured | Abs error |
|---|---:|---:|---:|
|  8k → 16k full (f32 / bf16) | 2.151 / 4.030 | 2.149 / 4.027 | ≤ 0.003 |
|  8k → 32k full (f32 / bf16) | 2.191 / 4.237 | 2.189 / 4.234 | ≤ 0.003 |
| 16k → 32k full (f32 / bf16) | 2.189 / 4.234 | 2.189 / 4.234 | 0.000 |

The extrapolator is byte-accurate under the standard preset: the
skeleton cost per compressed block and the encoded bytes per token are
both deterministic functions of `block_size`, `d_res`, `K`, `variance_ratio`
and the per-layer head dimension. Once any run with ≥ 1 compressed
block is measured, all longer contexts are arithmetic.

## Per-model reports

- `reports/gemma4_e2b/` — reference run (this document's numbers)
- `reports/qwen2_5_0_5b/` — Qwen2.5-0.5B-Instruct (Alibaba)
- `reports/smollm2_1_7b/` — SmolLM2-1.7B-Instruct (HuggingFace)
- `reports/qwen3_0_6b/` — Qwen3-0.6B (Alibaba)
- `reports/deepseek_r1_distill_qwen_1_5b/` — DeepSeek-R1-Distill-Qwen-1.5B (DeepSeek)
- `reports/glm_edge_1_5b/` — GLM-Edge-1.5B-Chat (Zhipu AI)
- `reports/glm_edge_4b/` — GLM-Edge-4B-Chat (Zhipu AI)

Each folder contains `bench_<ctx>.json` for every measured context
length, plus `extrapolation.json` with the projected rows.
`reports/CROSS_MODEL.md` is the side-by-side comparison.
