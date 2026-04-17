# Gemma 4 E2B — Kakeya KV Cache Compression Report

**Model:** `google/gemma-4-E2B-it` (BF16, 10.2 GB)
**Architecture:** hybrid sliding + full attention, 35 decoder layers, `num_kv_shared_layers=20`
**Cached layers:** 15 (12 sliding · 3 full-attention at indices 4, 9, 14)
**KV heads:** MQA, `num_key_value_heads=1`
**Head dim:** `head_dim=256` (sliding), `global_head_dim=512` (full)

This is the reference run for the benchmark standard (see `reports/STANDARD.md`).

## Codec preset

`block_size=512`, `residual_length=256`, `d_res=8`, `K=16`, `variance_ratio=0.95`.

## Measured (2k – 32k tokens)

| Context | Baseline (total) | Kakeya (f32 store) | Kakeya (bf16 store) | Full-attn f32 | Full-attn bf16 | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  2 048 |  17.99 MiB |  13.44 MiB |  11.22 MiB | 1.61× | 2.30× | **1.34×** | **1.60×** |
|  4 096 |  29.99 MiB |  18.73 MiB |  13.86 MiB | 1.88× | 3.05× | **1.60×** | **2.16×** |
|  8 192 |  53.99 MiB |  29.14 MiB |  19.06 MiB | 2.07× | 3.67× | **1.85×** | **2.83×** |
| 16 384 | 101.99 MiB |  50.66 MiB |  29.83 MiB | 2.15× | 4.03× | **2.01×** | **3.42×** |
| 32 768 | 197.99 MiB |  93.69 MiB |  51.34 MiB | 2.19× | 4.23× | **2.11×** | **3.86×** |

## Projected (64k – 256k tokens)

| Context | Baseline (total) | Kakeya (f32 store) | Kakeya (bf16 store) | Full-attn f32 | Full-attn bf16 | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  65 536 | 389.99 MiB | 179.74 MiB |  94.36 MiB | 2.21× | 4.35× | **2.17×** | **4.13×** |
| **131 072** | **773.99 MiB** | **351.83 MiB** | **180.41 MiB** | **2.22×** | **4.40×** | **2.20×** | **4.29×** |
| 262 144 |   1.51 GiB | 696.01 MiB | 352.50 MiB | 2.23× | 4.43× | **2.22×** | **4.37×** |

## Observations specific to Gemma 4 E2B

- Because only 3 of 15 cached layers are `full_attention`, raw compression
  gain is limited at short contexts (2k: 1.34× total) but quickly grows
  as those layers start to dominate the KV footprint.
- Sliding-window layers contribute a fixed ~6 MiB total for contexts ≥ 512
  tokens, so the "full vs total" gap narrows as context grows.
- MQA + `global_head_dim=512` means each full-attention layer stores
  `2 * 1 * seq * 512 * 2 bytes = 2 KiB/token`, a relatively small KV
  footprint per token. This is why the absolute byte savings at 128k
  (≈ 593 MiB) are modest compared to the dense 32-head SmolLM2.

## Generation sanity

At 2k context with greedy decode, Kakeya cache produced **identical 12
tokens** to the `DynamicCache` baseline (see `bench_2048.json` →
`generation.kakeya.text == generation.baseline.text`).
