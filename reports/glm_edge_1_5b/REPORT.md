# GLM-Edge-1.5B-Chat — Kakeya KV Cache Compression Report

**Model:** `THUDM/glm-edge-1.5b-chat` (BF16, 3.19 GB)
**Architecture:** GLM (Zhipu AI's on-device transformer), 28 decoder layers
**Cached layers:** 28 (all full-attention, no `layer_types`, no sliding)
**KV heads:** GQA, `num_key_value_heads=4` of 16 attention heads
**Head dim:** 128
**Max context length:** 8 192 tokens (the small GLM-Edge variant is a
short-context on-device model; the projected rows below are the ratio
the codec **would** produce if the model were extended to longer
contexts — the model itself cannot process 16k+ inputs).

This is the smallest release in the GLM-Edge family (ZhipuAI's
on-device line), released on HuggingFace without any gate.

## Codec preset

Same as the Gemma 4 reference: `block_size=512`, `residual_length=256`,
`d_res=8`, `K=16`, `variance_ratio=0.95`.

## Measured (2k – 8k tokens)

| Context | Baseline (total) | Kakeya (f32 store) | Kakeya (bf16 store) | **Full-attn f32** | **Full-attn bf16** | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  2 048 | 112.00 MiB |  72.93 MiB |  50.46 MiB | 1.54× | 2.22× | **1.54×** | **2.22×** |
|  4 096 | 224.00 MiB | 132.83 MiB |  80.44 MiB | 1.69× | 2.79× | **1.69×** | **2.79×** |
|  8 192 | 448.00 MiB | 252.86 MiB | 140.39 MiB | 1.77× | 3.19× | **1.77×** | **3.19×** |

## Projected (16k – 128k tokens, codec-only extrapolation)

These rows tell you what the **codec** would produce at longer
contexts. Note that `glm-edge-1.5b-chat` itself caps at 8 192 tokens;
the 16k+ projections are for reference only (e.g., as a baseline
expectation if this architecture were extended via YaRN / RoPE scaling).

| Context | Baseline (total) | Kakeya (f32 store) | Kakeya (bf16 store) | **Full-attn f32** | **Full-attn bf16** | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 384 | 896.00 MiB | 492.55 MiB | 260.27 MiB | 1.82× | 3.44× | **1.82×** | **3.44×** |
| 32 768 |   1.75 GiB | 972.08 MiB | 500.04 MiB | 1.84× | 3.58× | **1.84×** | **3.58×** |
| 65 536 |   3.50 GiB |   1.89 GiB | 979.58 MiB | 1.86× | 3.66× | **1.86×** | **3.66×** |
| **131 072** |   **7.00 GiB** |   **3.76 GiB** |   **1.89 GiB** | **1.86×** | **3.70×** | **1.86×** | **3.70×** |

## Observations

- Very similar compression profile to the DeepSeek distilled Qwen and
  to Qwen3-0.6B — which makes sense architecturally: they all share
  the "GQA + head_dim=128 + ~28 dense full-attention layers" recipe
  even though the weights are trained by three different teams.
- GLM-Edge has slightly higher compression ratios than
  DeepSeek-R1-Distill-Qwen-1.5B (3.70× vs 3.41× at 128k bf16) because
  GQA is wider here (4 KV heads vs 2), giving K-means more rows per
  block to fit stably.
