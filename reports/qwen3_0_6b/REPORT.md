# Qwen3-0.6B — Kakeya KV Cache Compression Report

**Model:** `Qwen/Qwen3-0.6B` (BF16, 1.5 GB)
**Architecture:** pure full-attention GQA, 28 decoder layers
**Cached layers:** 28 (all full-attention)
**KV heads:** GQA, `num_key_value_heads=8` of 16 attention heads
**Head dim:** 128

## Codec preset

Same as the Gemma 4 reference: `block_size=512`, `residual_length=256`,
`d_res=8`, `K=16`, `variance_ratio=0.95`.

## Measured (2k – 8k tokens)

| Context | Baseline (total) | Kakeya (f32 store) | Kakeya (bf16 store) | **Full-attn f32** | **Full-attn bf16** | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  2 048 | 224.00 MiB | 128.48 MiB |  92.24 MiB | 1.74× | 2.43× | **1.74×** | **2.43×** |
|  4 096 | 448.00 MiB | 226.35 MiB | 141.17 MiB | 1.98× | 3.17× | **1.98×** | **3.17×** |
|  8 192 | 896.00 MiB | 423.17 MiB | 239.59 MiB | 2.12× | 3.74× | **2.12×** | **3.74×** |

## Projected (16k – 128k tokens)

| Context | Baseline (total) | Kakeya (f32 store) | Kakeya (bf16 store) | **Full-attn f32** | **Full-attn bf16** | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 384 |   1.75 GiB | 814.83 MiB | 435.41 MiB | 2.20× | 4.12× | **2.20×** | **4.12×** |
| 32 768 |   3.50 GiB |   1.56 GiB | 827.07 MiB | 2.24× | 4.33× | **2.24×** | **4.33×** |
| 65 536 |   7.00 GiB |   3.09 GiB |   1.57 GiB | 2.27× | 4.45× | **2.27×** | **4.45×** |
| **131 072** |  **14.00 GiB** |   **6.15 GiB** |   **3.10 GiB** | **2.28×** | **4.51×** | **2.28×** | **4.51×** |

## Observations specific to Qwen3-0.6B

- **Best overall ratio in this benchmark suite.** At 128k tokens, the
  dtype-matched Kakeya cache is 3.10 GiB vs. a 14.00 GiB baseline —
  4.51× compression on every layer.
- The combination that makes Qwen3 a sweet spot: `head_dim=128`
  (large enough for PCA to actually discard redundant components) +
  GQA ratio 16:8 (enough heads to pool for stable K-means but not so
  many that the skeleton cost dominates) + every layer being full
  attention (no sliding-window freebie that skips compression).
- The f32-store column (2.28×) is also the highest of the four models;
  Qwen3 is the only one where the codec's as-is representation is
  already substantially smaller than the baseline without any dtype
  optimization, because the head_dim=128 PCA has enough slack to
  absorb the float32 blowup of the skeleton.
