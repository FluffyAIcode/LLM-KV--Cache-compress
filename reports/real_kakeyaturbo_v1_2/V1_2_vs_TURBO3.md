# kakeyaturbo v1.2 vs TurboQuant turbo3 — Real Benchmark Comparison

Both sets of numbers are real end-to-end measurements on real HF KV
cache tensors in bf16 (`attn_implementation="eager"`), on the same
machine, with the same prompt. No projections, no bf16-store-assumed
asymptote, no simplification.

- **kakeyaturbo v1.2** runs the `kakeyaturbo-bench` Rust release binary
  with `--share-basis` on V streams and per-block PCA on K streams.
  Preset: `block_size=512, variance_ratio=0.95, K=16, bit_width=3`.
- **TurboQuant turbo3** runs the official `turboquant_plus` Python
  reference implementation (3-bit `PolarQuant` on both K and V).

## Headline: total bf16 KV compression ratio

| Model | ctx | v1.2 | turbo3 | ratio v1.2/turbo3 | turbo2 (for context) | turbo4 (for context) |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-0.6B` | 2048 | **4.067×** | 5.120× | **0.79×** | 7.529× | 3.879× |
| `Qwen/Qwen3-0.6B` | 4096 | **4.104×** | 5.120× | **0.80×** | 7.529× | 3.879× |
| `Qwen/Qwen3-0.6B` | 8192 | **4.123×** | 5.120× | **0.81×** | 7.529× | 3.879× |
| `google/gemma-4-E2B-it` | 2048 | **2.109×** | 5.260× | **0.40×** | 7.837× | 3.959× |
| `google/gemma-4-E2B-it` | 4096 | **2.909×** | 5.268× | **0.55×** | 7.853× | 3.963× |
| `google/gemma-4-E2B-it` | 8192 | **3.875×** | 5.272× | **0.74×** | 7.864× | 3.966× |
| `THUDM/glm-edge-4b-chat` | 2048 | **3.078×** | 5.120× | **0.60×** | 7.529× | 3.879× |
| `THUDM/glm-edge-4b-chat` | 4096 | **3.099×** | 5.120× | **0.61×** | 7.529× | 3.879× |
| `THUDM/glm-edge-4b-chat` | 8192 | **3.104×** | 5.120× | **0.61×** | 7.529× | 3.879× |
| `THUDM/glm-edge-1.5b-chat` | 2048 | **3.083×** | 5.120× | **0.60×** | 7.529× | 3.879× |
| `THUDM/glm-edge-1.5b-chat` | 4096 | **3.117×** | 5.120× | **0.61×** | 7.529× | 3.879× |
| `THUDM/glm-edge-1.5b-chat` | 8192 | **3.130×** | 5.120× | **0.61×** | 7.529× | 3.879× |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 2048 | **3.139×** | 5.120× | **0.61×** | 7.529× | 3.879× |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 4096 | **3.205×** | 5.120× | **0.63×** | 7.529× | 3.879× |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 8192 | **3.237×** | 5.120× | **0.63×** | 7.529× | 3.879× |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 2048 | **3.080×** | 4.923× | **0.63×** | 7.111× | 3.765× |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 4096 | **3.087×** | 4.923× | **0.63×** | 7.111× | 3.765× |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 8192 | **3.091×** | 4.923× | **0.63×** | 7.111× | 3.765× |
| `Qwen/Qwen2.5-0.5B-Instruct` | 2048 | **3.206×** | 4.923× | **0.65×** | 7.111× | 3.765× |
| `Qwen/Qwen2.5-0.5B-Instruct` | 4096 | **3.275×** | 4.923× | **0.67×** | 7.111× | 3.765× |
| `Qwen/Qwen2.5-0.5B-Instruct` | 8192 | **3.339×** | 4.923× | **0.68×** | 7.111× | 3.765× |

## Full-attention-only ratio (Gemma 4: compressible layers only)

| Model | ctx | v1.2 full-attn | turbo3 full-attn | ratio |
|---|---:|---:|---:|---:|
| `Qwen/Qwen3-0.6B` | 2048 | **4.067×** | 5.120× | 0.79× |
| `Qwen/Qwen3-0.6B` | 4096 | **4.104×** | 5.120× | 0.80× |
| `Qwen/Qwen3-0.6B` | 8192 | **4.123×** | 5.120× | 0.81× |
| `google/gemma-4-E2B-it` | 2048 | **4.722×** | 5.278× | 0.89× |
| `google/gemma-4-E2B-it` | 4096 | **5.556×** | 5.278× | 1.05× |
| `google/gemma-4-E2B-it` | 8192 | **6.043×** | 5.278× | 1.14× |
| `THUDM/glm-edge-4b-chat` | 2048 | **3.078×** | 5.120× | 0.60× |
| `THUDM/glm-edge-4b-chat` | 4096 | **3.099×** | 5.120× | 0.61× |
| `THUDM/glm-edge-4b-chat` | 8192 | **3.104×** | 5.120× | 0.61× |
| `THUDM/glm-edge-1.5b-chat` | 2048 | **3.083×** | 5.120× | 0.60× |
| `THUDM/glm-edge-1.5b-chat` | 4096 | **3.117×** | 5.120× | 0.61× |
| `THUDM/glm-edge-1.5b-chat` | 8192 | **3.130×** | 5.120× | 0.61× |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 2048 | **3.139×** | 5.120× | 0.61× |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 4096 | **3.205×** | 5.120× | 0.63× |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 8192 | **3.237×** | 5.120× | 0.63× |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 2048 | **3.080×** | 4.923× | 0.63× |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 4096 | **3.087×** | 4.923× | 0.63× |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 8192 | **3.091×** | 4.923× | 0.63× |
| `Qwen/Qwen2.5-0.5B-Instruct` | 2048 | **3.206×** | 4.923× | 0.65× |
| `Qwen/Qwen2.5-0.5B-Instruct` | 4096 | **3.275×** | 4.923× | 0.67× |
| `Qwen/Qwen2.5-0.5B-Instruct` | 8192 | **3.339×** | 4.923× | 0.68× |

## Cross-context trajectory (v1.2 @ 8k vs turbo3)

turbo3's ratio is context-independent by construction; v1.2 grows with context.

| Model | v1.2 @ 2k | v1.2 @ 4k | v1.2 @ 8k | turbo3 (flat) | first-cross-over? |
|---|---:|---:|---:|---:|---|
| `Qwen/Qwen3-0.6B` | 4.067× | 4.104× | 4.123× | 5.120× | not yet @ 8k |
| `google/gemma-4-E2B-it` | 2.109× | 2.909× | 3.875× | 5.272× | not yet @ 8k |
| `THUDM/glm-edge-4b-chat` | 3.078× | 3.099× | 3.104× | 5.120× | not yet @ 8k |
| `THUDM/glm-edge-1.5b-chat` | 3.083× | 3.117× | 3.130× | 5.120× | not yet @ 8k |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 3.139× | 3.205× | 3.237× | 5.120× | not yet @ 8k |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 3.080× | 3.087× | 3.091× | 4.923× | not yet @ 8k |
| `Qwen/Qwen2.5-0.5B-Instruct` | 3.206× | 3.275× | 3.339× | 4.923× | not yet @ 8k |

## Absolute compressed bytes @ 8 192 tokens

| Model | baseline (bf16) | v1.2 | turbo3 | v1.2 / turbo3 |
|---|---:|---:|---:|---:|
| `Qwen/Qwen3-0.6B` | 896.0 MiB | 217.3 MiB | 175.0 MiB | 1.24× |
| `google/gemma-4-E2B-it` | 54.0 MiB | 13.9 MiB | 10.2 MiB | 1.36× |
| `THUDM/glm-edge-4b-chat` | 960.0 MiB | 309.3 MiB | 187.5 MiB | 1.65× |
| `THUDM/glm-edge-1.5b-chat` | 448.0 MiB | 143.1 MiB | 87.5 MiB | 1.64× |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 224.0 MiB | 69.2 MiB | 43.8 MiB | 1.58× |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.50 GiB | 496.9 MiB | 312.0 MiB | 1.59× |
| `Qwen/Qwen2.5-0.5B-Instruct` | 96.0 MiB | 28.8 MiB | 19.5 MiB | 1.47× |

## Data-source notes

All turbo3 measurements were produced by `compare_kakeya_vs_turboquant.py`
via the official `turboquant_plus` Python prototype (stored on the same
captured KV tensors that were fed into kakeyaturbo v1.2).
Baselines match to < 1% across all 21 cells (different HF forward passes
produce bit-identical KV tensors for the same seed / prompt / dtype).