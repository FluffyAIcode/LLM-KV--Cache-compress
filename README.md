# Kakeya KV Cache Compression

A Kakeya-like set-compression codec applied to transformer KV caches as
a drop-in replacement for `transformers.cache_utils.DynamicCache`. Works
out of the box with any modern HF decoder-only model (Gemma, Llama,
Mistral, Qwen, SmolLM2, Cohere2, etc.) without changing the model code.

## Quick start

```bash
pip install -U torch transformers accelerate huggingface_hub
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kakeya_kv_codec import build_kakeya_cache

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="bfloat16")

inputs = tokenizer("Your long prompt here...", return_tensors="pt")
cache = build_kakeya_cache(model)  # model-agnostic factory

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    use_cache=True,
    past_key_values=cache,
)
```

## What you get

Under the standard codec preset (`block_size=512, residual_length=256,
d_res=8, K=16, variance_ratio=0.95`), the total KV cache compresses to
roughly **2.2–4.5×** of the uncompressed bf16 baseline at 128k tokens,
depending on the model architecture. Concretely (see
`reports/CROSS_MODEL.md` for details):

| Model | 128k Baseline KV | 128k Kakeya KV (bf16 store) | Total ratio |
|---|---:|---:|---:|
| Qwen/Qwen3-0.6B | 14.00 GiB | 3.10 GiB | **4.51×** |
| google/gemma-4-E2B-it | 774 MiB | 180 MiB | **4.29×** |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 24.00 GiB | 10.65 GiB | **2.25×** |
| Qwen/Qwen2.5-0.5B-Instruct | 1.50 GiB | 714 MiB | **2.15×** |

Full measured + projected tables for every model are under
`reports/<model>/REPORT.md`.

## Repository layout

```
kakeya_kv_codec.py       # the codec + drop-in Cache (~600 lines)
kakeya_benchmark.py      # end-to-end benchmark harness (any HF model)
kakeya_extrapolate.py    # byte-exact projection to longer contexts
run_all_benchmarks.sh    # orchestrator for a full 2k/4k/8k sweep
smoke_test.py            # 30-second self-test of the codec
gemma4_inference_test.py # Gemma-4-specific reference runner (legacy CLI)

reports/
  STANDARD.md            # benchmark methodology + Gemma 4 reference numbers
  CROSS_MODEL.md         # side-by-side comparison across 4 models
  gemma4_e2b/            # Gemma 4 E2B: bench_*.json + REPORT.md
  qwen2_5_0_5b/          # Qwen2.5-0.5B
  smollm2_1_7b/          # SmolLM2-1.7B
  qwen3_0_6b/            # Qwen3-0.6B
```

## How the codec works (one paragraph)

Each KV cache layer is split into two zones: a recent tail kept in
exact precision (`residual_length` tokens), and older tokens grouped
into compressed blocks of `block_size` tokens. When a block is sealed:

1. PCA (`variance_ratio` → `d_eff`) projects rows onto a low-rank basis.
2. A mean "time direction" `t_dir` is separated out of the block.
3. Spherical K-means with `K` centers clusters the perpendicular
   component; each row gets a segment id + scalar projection.
4. The top-`d_res` residual coefficients are kept as a sparse sub-vector.

Decode reverses these steps on demand, producing an approximate
reconstruction of the original `[bsz, n_kv, block, head_dim]` tensor.
The model is never told that its KV cache is compressed — attention
operates on the reconstructed tensor exactly as with `DynamicCache`.

## Model support

The `build_kakeya_cache(model)` factory auto-detects the layer plan from
`model.config`:

| Config pattern | Handling |
|---|---|
| `layer_types` exists (Gemma 2/3/4, Cohere 2, SmolLM3, Qwen3 hybrid) | Per-layer dispatch: `full_attention` → Kakeya, `sliding_attention` / `chunked_attention` → `DynamicSlidingWindowLayer`. |
| `num_kv_shared_layers > 0` (Gemma 4) | Only non-shared layers are cached (HF convention). |
| No `layer_types`, no `sliding_window` (Llama, Mistral, Qwen2, SmolLM2) | Every layer is full attention → Kakeya on every layer. |
| `sliding_window` only (rare) | Treated as all-sliding, codec is a pass-through. |
| `attention_chunk_size` (Llama 4 etc.) | Treated as sliding. |

For notes on what a cross-model port requires (MLA, SSM, encoder-decoder, etc.) see
the PR description on GitHub.

## Status

This is a research-grade codec. The CPU store of compressed tensors is
currently float32; moving it to bf16 halves the compressed side (see
the "bf16 store" columns in the reports). Generation correctness has
been validated by matching greedy decode against the `DynamicCache`
baseline on Gemma 4 E2B (first 12 tokens identical at 2k context).

## License

See `LICENSE`.
