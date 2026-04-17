# SmolLM2-1.7B-Instruct — Kakeya KV Cache Compression Report

**Model:** `HuggingFaceTB/SmolLM2-1.7B-Instruct` (BF16, 3.4 GB)
**Architecture:** pure full-attention MHA (Llama-style), 24 decoder layers
**Cached layers:** 24 (all full-attention, no `layer_types` on config)
**KV heads:** MHA, `num_key_value_heads=32` = `num_attention_heads`
**Head dim:** 64

## Codec preset

Same as the Gemma 4 reference. Note that at 8k context the prefill
needs `--prefill-chunk 1024` because the MHA KV cache already weighs
1.5 GiB and a single full-sequence forward pass saturates the 15 GiB
CPU RAM budget on the benchmark box.

## Measured (2k – 8k tokens)

| Context | Baseline (total) | Kakeya (f32 store) | Kakeya (bf16 store) | **Full-attn f32** | **Full-attn bf16** | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  2 048 | 384.00 MiB | 350.41 MiB | 223.21 MiB | 1.10× | 1.72× | **1.10×** | **1.72×** |
|  4 096 | 768.00 MiB | 689.69 MiB | 392.84 MiB | 1.11× | 1.96× | **1.11×** | **1.96×** |
|  8 192 |   1.50 GiB |   1.34 GiB | 732.08 MiB | 1.12× | 2.10× | **1.12×** | **2.10×** |

## Projected (16k – 128k tokens)

| Context | Baseline (total) | Kakeya (f32 store) | Kakeya (bf16 store) | **Full-attn f32** | **Full-attn bf16** | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 384 |   3.00 GiB |   2.66 GiB |   1.38 GiB | 1.13× | 2.18× | **1.13×** | **2.18×** |
| 32 768 |   6.00 GiB |   5.31 GiB |   2.70 GiB | 1.13× | 2.22× | **1.13×** | **2.22×** |
| 65 536 |  12.00 GiB |  10.61 GiB |   5.35 GiB | 1.13× | 2.24× | **1.13×** | **2.24×** |
| **131 072** |  **24.00 GiB** |  **21.21 GiB** |  **10.65 GiB** | **1.13×** | **2.25×** | **1.13×** | **2.25×** |

## Observations specific to SmolLM2

- The **absolute bytes saved** is the most dramatic of the four models:
  at 128k, the bf16-projected Kakeya cache is 10.65 GiB vs. a 24 GiB
  baseline — a 13 GiB absolute savings per sequence. MHA with 32 KV
  heads makes every token 32× more expensive than a MQA token, so
  even a modest 2.25× ratio translates into huge wall-clock memory.
- The bf16-projected total ratio (2.25× at 128k) is slightly higher
  than Qwen2.5 (2.15×) because SmolLM2 has more redundancy across its
  32 KV heads — pooling them together in the PCA gives a fatter tail
  that the codec discards.
- This run is the strongest argument for finishing the bf16 store
  optimization in the codec: the absolute delta between f32 (1.13×)
  and bf16 (2.25×) stores is an entire 10 GiB per 128k sequence.
