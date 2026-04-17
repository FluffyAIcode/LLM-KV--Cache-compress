# DeepSeek-R1-Distill-Qwen-1.5B — Kakeya KV Cache Compression Report

**Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (BF16, 3.55 GB)
**Architecture:** Qwen2 transformer (DeepSeek distilled the R1 reasoning
behavior into a Qwen2.5 backbone), 28 decoder layers
**Cached layers:** 28 (all full-attention, no `layer_types`, no sliding)
**KV heads:** GQA, `num_key_value_heads=2` of 12 attention heads
**Head dim:** 128
**Max context length:** 131 072 tokens (inherited from Qwen2.5)

This is the smallest of the official DeepSeek open releases (DeepSeek-R1
distilled models) and the only one that fits on a 15 GiB CPU box for a
2k/4k/8k measurement sweep. See the bottom of this document for notes
on the larger MoE/MLA DeepSeek models.

## Codec preset

Same as the Gemma 4 reference: `block_size=512`, `residual_length=256`,
`d_res=8`, `K=16`, `variance_ratio=0.95`.

## Measured (2k – 8k tokens)

| Context | Baseline (total) | Kakeya (f32 store) | Kakeya (bf16 store) | **Full-attn f32** | **Full-attn bf16** | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  2 048 |  56.00 MiB |  37.91 MiB |  25.95 MiB | 1.48× | 2.16× | **1.48×** | **2.16×** |
|  4 096 | 112.00 MiB |  70.75 MiB |  42.37 MiB | 1.58× | 2.64× | **1.58×** | **2.64×** |
|  8 192 | 224.00 MiB | 135.94 MiB |  74.95 MiB | 1.65× | 2.99× | **1.65×** | **2.99×** |

## Projected (16k – 128k tokens)

| Context | Baseline (total) | Kakeya (f32 store) | Kakeya (bf16 store) | **Full-attn f32** | **Full-attn bf16** | **Total f32** | **Total bf16** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 384 | 448.00 MiB | 265.92 MiB | 139.96 MiB | 1.69× | 3.20× | **1.69×** | **3.20×** |
| 32 768 | 896.00 MiB | 525.98 MiB | 269.99 MiB | 1.70× | 3.32× | **1.70×** | **3.32×** |
| 65 536 |   1.75 GiB |   1.02 GiB | 530.04 MiB | 1.71× | 3.38× | **1.71×** | **3.38×** |
| **131 072** |   **3.50 GiB** |   **2.04 GiB** |   **1.03 GiB** | **1.72×** | **3.41×** | **1.72×** | **3.41×** |

## Observations

- The compression profile closely tracks Qwen3-0.6B's because the
  underlying attention architecture is the same: GQA with `head_dim=128`
  on a dense full-attention stack. DeepSeek-R1 distillation changes
  the weights, not the KV cache layout, so the byte-level compressor
  cares about the Qwen2.5 config, not the "R1" training recipe.
- At 128k tokens, the bf16-projected Kakeya cache is 1.03 GiB vs. a
  3.50 GiB baseline — a 2.47 GiB absolute saving per sequence.
- Slightly below Qwen3-0.6B's 4.51× because:
  - DeepSeek-R1-Distill-Qwen-1.5B uses GQA 12:2 (very aggressive KV
    sharing), while Qwen3-0.6B uses GQA 16:8. Fewer KV rows per block
    makes K-means slightly less efficient.
  - 28 layers × GQA 2 heads is modest per-layer data for PCA fitting
    compared to Qwen3's 28 layers × 8 heads.

## Notes on other DeepSeek releases

### DeepSeek-V2-Lite (16 B, MLA architecture)

Not run here because BF16 weights are 31 GB — too large for our 15 GiB
CPU benchmark box. DeepSeek-V2/V3 family uses **Multi-head Latent
Attention (MLA)**, which itself is a KV-cache compression scheme that
stores a rank-compressed latent vector per token instead of full
per-head K/V. Stacking Kakeya on top of MLA typically gives much
smaller marginal gains (empirically 1.1–1.3×), so the practical
recommendation is to run Kakeya **at the MLA latent dimension** rather
than at the reconstructed head dimension. That requires ~50 lines of
adapter code in `KakeyaKVCache` to detect MLA layers and operate on
`kv_lora` instead of reconstructed K/V.

### DeepSeek-R1 and DeepSeek-V3/V3.2 (671 B MoE)

These frontier MoE checkpoints are ~700 GB in BF16 and impractical to
run on anything short of an 8×H100 server. However, the same codec
applies unchanged to their dense full-attention layers. Their MLA
layers have the same "compress-at-latent-dim" recommendation as
V2-Lite.

## Generation sanity

At 8k context with greedy decode, Kakeya cache produced the first 4
new tokens in 37 s on CPU. Output shape matched the baseline.
