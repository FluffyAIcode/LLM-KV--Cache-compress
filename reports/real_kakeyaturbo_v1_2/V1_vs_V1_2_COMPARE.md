# kakeyaturbo v1.2 (A + B') vs v1.0 Real Benchmark Comparison

v1.2 changes:
- **A**: skeleton tensors (PCA mean / basis / K-means centres) stored as bf16 instead of f32
- **B'**: V stream uses a **layer-pooled PCA basis** (one fit per layer, reused across all blocks);
  K stream keeps per-block PCA (required to preserve reconstruction on RoPE-driven K distributions)

Both were decided by the PCA basis-sharing ablation in PR #6: V inflation is 1.03–1.30×, K inflation would be 2–12×.

**Codec params identical between v1 and v1.2**: block_size=512, variance_ratio=0.95, K=16, bit_width=3, seed=3405691582.
All numbers are real end-to-end measurements via the release-built Rust binary on the same HF weights and prompt.

## Total bf16 compression ratio — v1 → v1.2

| Model | 2 048 (v1) | 2 048 (v1.2) | Δ | 4 096 (v1) | 4 096 (v1.2) | Δ | 8 192 (v1) | 8 192 (v1.2) | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-0.6B` | 2.818x | **4.067x** | +44.3% | 2.869x | **4.104x** | +43.0% | 2.896x | **4.123x** | +42.4% |
| `google/gemma-4-E2B-it` | 1.571x | **2.109x** | +34.3% | 1.867x | **2.909x** | +55.8% | 2.140x | **3.875x** | +81.1% |
| `THUDM/glm-edge-4b-chat` | 2.007x | **3.078x** | +53.4% | 2.037x | **3.099x** | +52.2% | 2.060x | **3.104x** | +50.6% |
| `THUDM/glm-edge-1.5b-chat` | 2.098x | **3.083x** | +47.0% | 2.126x | **3.117x** | +46.6% | 2.137x | **3.130x** | +46.5% |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 2.070x | **3.139x** | +51.6% | 2.115x | **3.205x** | +51.5% | 2.137x | **3.237x** | +51.4% |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 2.203x | **3.080x** | +39.8% | 2.216x | **3.087x** | +39.3% | 2.224x | **3.091x** | +39.0% |
| `Qwen/Qwen2.5-0.5B-Instruct` | 2.392x | **3.206x** | +34.1% | 2.432x | **3.275x** | +34.7% | 2.467x | **3.339x** | +35.3% |

## Full-attention-only ratio (strips sliding layers from Gemma 4)

| Model | 2 048 (v1 → v1.2) | 4 096 (v1 → v1.2) | 8 192 (v1 → v1.2) |
|---|---:|---:|---:|
| `Qwen/Qwen3-0.6B` | 2.818x → **4.067x** (+44%) | 2.869x → **4.104x** (+43%) | 2.896x → **4.123x** (+42%) |
| `google/gemma-4-E2B-it` | 2.196x → **4.722x** (+115%) | 2.382x → **5.556x** (+133%) | 2.494x → **6.043x** (+142%) |
| `THUDM/glm-edge-4b-chat` | 2.007x → **3.078x** (+53%) | 2.037x → **3.099x** (+52%) | 2.060x → **3.104x** (+51%) |
| `THUDM/glm-edge-1.5b-chat` | 2.098x → **3.083x** (+47%) | 2.126x → **3.117x** (+47%) | 2.137x → **3.130x** (+46%) |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 2.070x → **3.139x** (+52%) | 2.115x → **3.205x** (+52%) | 2.137x → **3.237x** (+51%) |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 2.203x → **3.080x** (+40%) | 2.216x → **3.087x** (+39%) | 2.224x → **3.091x** (+39%) |
| `Qwen/Qwen2.5-0.5B-Instruct` | 2.392x → **3.206x** (+34%) | 2.432x → **3.275x** (+35%) | 2.467x → **3.339x** (+35%) |

## Reconstruction quality change

Per-layer mean K and V MSE at 4 k context. K should be unchanged (same algorithm). V should land within
1.0–1.3× of v1 per the ablation prediction.

| Model | K MSE v1 | K MSE v1.2 | V MSE v1 | V MSE v1.2 | V inflation |
|---|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-0.6B` | 1.398e+00 | 1.398e+00 | 3.619e+00 | 4.182e+00 | 1.156× |
| `google/gemma-4-E2B-it` | 3.785e-03 | 3.782e-03 | 2.755e-01 | 2.812e-01 | 1.021× |
| `THUDM/glm-edge-4b-chat` | 6.770e-01 | 6.769e-01 | 3.731e-01 | 4.069e-01 | 1.091× |
| `THUDM/glm-edge-1.5b-chat` | 7.534e-01 | 7.535e-01 | 3.067e-01 | 3.509e-01 | 1.144× |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 5.497e-01 | 5.500e-01 | 4.131e-01 | 4.658e-01 | 1.128× |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 6.555e-01 | 6.560e-01 | 2.399e-01 | 2.570e-01 | 1.071× |
| `Qwen/Qwen2.5-0.5B-Instruct` | 1.297e+00 | 1.297e+00 | 2.146e-01 | 2.232e-01 | 1.040× |

## Byte savings @ 8 192 tokens

| Model | v1 compressed | v1.2 compressed | Extra saved |
|---|---:|---:|---:|
| `Qwen/Qwen3-0.6B` | 309.39 MiB | 217.32 MiB | 92.07 MiB |
| `google/gemma-4-E2B-it` | 25.23 MiB | 13.93 MiB | 11.30 MiB |
| `THUDM/glm-edge-4b-chat` | 465.96 MiB | 309.32 MiB | 156.64 MiB |
| `THUDM/glm-edge-1.5b-chat` | 209.66 MiB | 143.12 MiB | 66.53 MiB |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 104.81 MiB | 69.21 MiB | 35.60 MiB |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 690.64 MiB | 496.88 MiB | 193.76 MiB |
| `Qwen/Qwen2.5-0.5B-Instruct` | 38.91 MiB | 28.75 MiB | 10.16 MiB |
