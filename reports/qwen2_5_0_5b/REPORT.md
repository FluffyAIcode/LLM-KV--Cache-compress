# Qwen2.5-0.5B-Instruct — Kakeya KV Cache Compression Report

**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (BF16, 942 MB)
**Architecture:** pure full-attention GQA, 24 decoder layers
**Cached layers:** 24 (all full-attention)
**KV heads:** GQA, `num_key_value_heads=2` of 14 attention heads
**Head dim:** 64

## Codec preset

Same as the Gemma 4 reference: `block_size=512`, `residual_length=256`,
`d_res=8`, `K=16`, `variance_ratio=0.95`.

## Measured (2k – 8k tokens)

| Context | Baseline (total) | Kakeya (f32 store) | Kakeya (bf16 store) | **Full-attn f32** | **Full-attn bf16** | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  2 048 |  24.00 MiB |  22.66 MiB |  14.33 MiB | 1.06× | 1.68× | **1.06×** | **1.68×** |
|  4 096 |  48.00 MiB |  44.86 MiB |  25.43 MiB | 1.07× | 1.89× | **1.07×** | **1.89×** |
|  8 192 |  96.00 MiB |  89.25 MiB |  47.62 MiB | 1.08× | 2.02× | **1.08×** | **2.02×** |

## Projected (16k – 128k tokens)

| Context | Baseline (total) | Kakeya (f32 store) | Kakeya (bf16 store) | **Full-attn f32** | **Full-attn bf16** | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 384 | 192.00 MiB | 178.04 MiB |  92.02 MiB | 1.08× | 2.09× | **1.08×** | **2.09×** |
| 32 768 | 384.00 MiB | 355.64 MiB | 180.82 MiB | 1.08× | 2.12× | **1.08×** | **2.12×** |
| 65 536 | 768.00 MiB | 710.83 MiB | 358.41 MiB | 1.08× | 2.14× | **1.08×** | **2.14×** |
| **131 072** |   **1.50 GiB** |   **1.39 GiB** | **713.60 MiB** | **1.08×** | **2.15×** | **1.08×** | **2.15×** |

## Observations specific to Qwen2.5-0.5B

- All 24 layers are cached, so the gain is unmuted by any sliding
  layers. Full-attn ratio = total ratio.
- The small `head_dim=64` is the main cap on f32-store gain: the
  codec's skeleton (basis + mean + t_dir + centers) scales with
  head_dim, but small head_dim means the skeleton is cheap in absolute
  terms **and** the tail of PCA has less information to discard.
- The bf16-projected number (2.15× at 128k) is healthy; the f32-store
  number (1.08×) is dominated by the codec's float32 CPU representation
  of compressed tensors vs the bf16 baseline.
- This is a good canary for "what the codec does on a tiny generic
  GQA transformer with no architectural tricks."
