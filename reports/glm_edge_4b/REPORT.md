# GLM-Edge-4B-Chat — Kakeya KV Cache Compression Report

**Model:** `THUDM/glm-edge-4b-chat` (BF16, 8.66 GB)
**Architecture:** GLM (Zhipu AI's on-device transformer), 40 decoder layers
**Cached layers:** 40 (all full-attention)
**KV heads:** GQA, `num_key_value_heads=6` of 24 attention heads
**Head dim:** 128
**Max context length:** 8 192 tokens

The 4B variant of GLM-Edge: deeper (40 layers vs 28), wider (24 heads
vs 16), same GQA ratio (4:1). Good stress test for how the codec
behaves on a model with a substantially larger absolute KV footprint
per token but no architectural changes.

## Codec preset

Same as the Gemma 4 reference: `block_size=512`, `residual_length=256`,
`d_res=8`, `K=16`, `variance_ratio=0.95`.

Note: the 8k measurement requires `--skip-generation --prefill-chunk 1024`
because the combination of 40 layers, 6 KV heads, and the intermediate
activations during the 8k forward pass exceeds 15 GiB without it.

## Measured (2k – 8k tokens)

| Context | Baseline (total) | Kakeya (f32 store) | Kakeya (bf16 store) | **Full-attn f32** | **Full-attn bf16** | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  2 048 | 240.00 MiB | 151.34 MiB | 105.67 MiB | 1.59× | 2.27× | **1.59×** | **2.27×** |
|  4 096 | 480.00 MiB | 273.32 MiB | 166.69 MiB | 1.76× | 2.88× | **1.76×** | **2.88×** |
|  8 192 | 960.00 MiB | 518.75 MiB | 289.34 MiB | 1.85× | 3.32× | **1.85×** | **3.32×** |

## Projected (16k – 128k tokens, codec-only extrapolation)

| Context | Baseline (total) | Kakeya (f32 store) | Kakeya (bf16 store) | **Full-attn f32** | **Full-attn bf16** | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 384 |   1.88 GiB |  1008 MiB | 533.96 MiB | 1.91× | 3.60× | **1.91×** | **3.60×** |
| 32 768 |   3.75 GiB |   1.94 GiB |   1.00 GiB | 1.93× | 3.75× | **1.93×** | **3.75×** |
| 65 536 |   7.50 GiB |   3.85 GiB |   1.95 GiB | 1.95× | 3.84× | **1.95×** | **3.84×** |
| **131 072** |  **15.00 GiB** |   **7.67 GiB** |   **3.87 GiB** | **1.95×** | **3.88×** | **1.95×** | **3.88×** |

## Observations

- **3.88× at 128k bf16** — the highest bf16 ratio among the dense
  models tested that is not Qwen3. The combination of 40 layers × 6
  KV heads × 128 head_dim gives each compressed block a large,
  well-conditioned row matrix, which lets PCA + K-means land deeper.
- Absolute saving at 128k is 11.13 GiB per sequence (15.00 → 3.87 GiB),
  close to SmolLM2-1.7B's 13.35 GiB saving despite GLM-Edge-4B being
  MHA-free (it's GQA 4:1). The more-layers-but-still-GQA profile
  ends up in the same per-sequence KV pressure regime as MHA.
- The f32-store ratio (1.95×) is already substantially > 1, meaning
  the codec is a net win even without the bf16 storage optimization.

## Architecture remark

GLM-Edge's `head_dim=128` gives the codec plenty of room to discard
components via PCA (`variance_ratio=0.95` typically lands `d_eff` in
the 40-80 range for this head_dim), which is the main reason it
outperforms, e.g., the smaller-head Qwen2.5-0.5B (`head_dim=64`).
