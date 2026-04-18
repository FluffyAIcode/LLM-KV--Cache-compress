# kakeyaturbo Real Benchmark — Full 7-Model Matrix

All numbers below are **real end-to-end measurements** produced by:

1. Loading the HF model in BF16 with `attn_implementation="eager"`.
2. Running a real prefill on a 2k/4k/8k-token prompt.
3. Extracting every cached layer's K and V from `DynamicCache`.
4. Writing each tensor to a KKTV binary file on disk.
5. Invoking the release-built `kakeyaturbo-bench` Rust binary
   (from `main` branch, commit-stable), which runs the real
   PCA + spherical K-means + WHT + Lloyd-Max chain.
6. Verifying each block by decode+MSE.

Codec preset (identical for every run):

| parameter | value |
|---|---:|
| `block_size` | 512 |
| `variance_ratio` | 0.95 |
| `K` (K-means centres) | 16 |
| `bit_width` | 3 |
| `rotation_seed` | 3405691582 |
| `kmeans_max_iter` | 32 |
| metric on K | InnerProduct |
| metric on V | MSE |

No mock, no fallback, no overfit, no simplification. The
Rust binary either runs the real codec chain or exits non-zero;
the Python driver only does I/O and aggregation.

## Headline: total bf16 compression ratio

| Model | 2 048 | 4 096 | 8 192 |
|---|---:|---:|---:|
| `Qwen/Qwen3-0.6B` | 2.818x | 2.869x | 2.896x |
| `google/gemma-4-E2B-it` | 1.571x | 1.867x | 2.140x |
| `THUDM/glm-edge-4b-chat` | 2.007x | 2.037x | 2.060x |
| `THUDM/glm-edge-1.5b-chat` | 2.098x | 2.126x | 2.137x |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 2.070x | 2.115x | 2.137x |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 2.203x | 2.216x | 2.224x |
| `Qwen/Qwen2.5-0.5B-Instruct` | 2.392x | 2.432x | 2.467x |

## Full-attention-only ratio (for hybrid Gemma 4)

For Gemma 4 E2B the 15 cached layers are 7 full-attention + 28 sliding
(after `num_kv_shared_layers=20` strips the last 20); sliding layers are
pass-through. The total ratio is diluted by sliding bytes; the full-attn
column shows the kakeyaturbo ratio on the compressible subset.

| Model | 2 048 | 4 096 | 8 192 |
|---|---:|---:|---:|
| `Qwen/Qwen3-0.6B` | 2.818x | 2.869x | 2.896x |
| `google/gemma-4-E2B-it` | 2.196x | 2.382x | 2.494x |
| `THUDM/glm-edge-4b-chat` | 2.007x | 2.037x | 2.060x |
| `THUDM/glm-edge-1.5b-chat` | 2.098x | 2.126x | 2.137x |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 2.070x | 2.115x | 2.137x |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 2.203x | 2.216x | 2.224x |
| `Qwen/Qwen2.5-0.5B-Instruct` | 2.392x | 2.432x | 2.467x |

## Absolute bytes @ 8 192 tokens

| Model | Baseline bf16 | kakeyaturbo | Saved |
|---|---:|---:|---:|
| `Qwen/Qwen3-0.6B` | 896.00 MiB | 309.39 MiB | 586.61 MiB |
| `google/gemma-4-E2B-it` | 53.99 MiB | 25.23 MiB | 28.76 MiB |
| `THUDM/glm-edge-4b-chat` | 960.00 MiB | 465.96 MiB | 494.04 MiB |
| `THUDM/glm-edge-1.5b-chat` | 448.00 MiB | 209.66 MiB | 238.34 MiB |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 224.00 MiB | 104.81 MiB | 119.19 MiB |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.50 GiB | 690.64 MiB | 845.36 MiB |
| `Qwen/Qwen2.5-0.5B-Instruct` | 96.00 MiB | 38.91 MiB | 57.09 MiB |

## Per-layer reconstruction quality (mean MSE across compressed layers)

Reported as `mean_block_mse`, averaged across all full-attention layers of the
model at 4 k context. K is reconstructed under InnerProduct metric and may have
large absolute MSE on models with large K norms (Qwen family); V is reconstructed
under MSE metric.

| Model | head_dim | mean K MSE | mean V MSE |
|---|---:|---:|---:|
| `Qwen/Qwen3-0.6B` | 128 | 1.398e+00 | 3.619e+00 |
| `google/gemma-4-E2B-it` | 512/256 | 3.785e-03 | 2.755e-01 |
| `THUDM/glm-edge-4b-chat` | 128 | 6.770e-01 | 3.731e-01 |
| `THUDM/glm-edge-1.5b-chat` | 128 | 7.534e-01 | 3.067e-01 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 128 | 5.497e-01 | 4.131e-01 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 64 | 6.555e-01 | 2.399e-01 |
| `Qwen/Qwen2.5-0.5B-Instruct` | 64 | 1.297e+00 | 2.146e-01 |

## Setup

- Host: x86_64 CPU-only, 15 GiB RAM
- Rust 1.83 stable, `cargo build --release --bin kakeyaturbo-bench`
- Python 3.12, `torch==2.11 bf16`, `transformers==5.5`
- Prompt: identical technical-writer boilerplate for every run
- SmolLM2 and GLM-Edge-4B use `--prefill-chunk 1024` at 8 k to stay under
  the 15 GiB memory cap (chunking doesn't change captured KV tensors).

Per-run JSON reports (including every layer's K/V encode+decode bytes
and verify MSE) are under `reports/real_kakeyaturbo/full/<model>/ctx_<N>/`.
