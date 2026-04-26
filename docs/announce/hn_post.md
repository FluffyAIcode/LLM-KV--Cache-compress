# Hacker News submission

**Title:** `KakeyaLattice: 2.77× KV cache compression for Qwen3-4B with 1.67 % perplexity loss`

**URL:** `https://github.com/FluffyAIcode/LLM-KV--Cache-compress`

> **Do not** post both a URL and text. HN lets you post *either* a URL
> *or* a Show HN text post. Use the URL form with the title above; then
> post this body as the **first comment** on your own thread.

---

## First-comment body

Author here. This is the code + paper + raw benchmark data for
KakeyaLattice, a drop-in subclass of `transformers.DynamicCache` that
compresses KV caches using nested D4 / E8 lattice quantisation.

**The numbers** (real vLLM prefill + real FlashAttention bf16 on
NVIDIA H200, 128 k context, WikiText-103, n=8 passages × 64 eval
positions per passage):

At |Δppl| ≤ 1 %:
- Qwen3-4B: **2.40× CR** (TurboQuant: 1.95×, +23.3 %)
- GLM-4-9B-Chat: **1.73× CR** (TurboQuant: out of range)
- DeepSeek-R1-Distill-Qwen-1.5B: **2.29× CR** (TurboQuant: 2.09×, +9.2 %)
- Gemma-4-E4B: **3.04× CR** (tied with TurboQuant)

At |Δppl| ≤ 2 % the Qwen3-4B CR becomes 2.77× (+26.9 % vs TurboQuant)
and GLM-4-9B-Chat becomes 2.44× (+37.8 %).

The headline advantage over per-channel scalar quantisers comes from
two places:

1. **Sylvester–Hadamard rotation** gaussianises heavy-tailed,
   non-isotropic KV distributions before quantisation. Real LLM KV
   activations are not Gaussian and not aligned to the canonical basis;
   per-channel quantisers waste bits on the worst-case channel.
2. **Nested D4 / E8 lattices** are the densest lattices in dimensions
   4 and 8. Quantising a rotated, gaussianised 8-D block with E8 beats
   any scalar quantisation of the same eight channels.

**Integration**:

```python
from kakeyalattice.hf import KakeyaLatticeCache
cache = KakeyaLatticeCache(
    variant="e8", q_range=38,
    num_hidden_layers=model.config.num_hidden_layers,
    head_dim=model.config.head_dim,
    device="cuda",
)
out = model.generate(**inputs, past_key_values=cache, use_cache=True)
```

That's the full API. The codec is stateless per-vector, so no
calibration, no warm-up, and streaming / online decode is supported by
construction.

**What's NOT in this release**:

- Today the reference `KakeyaLatticeCache` round-trips through the
  codec and stores the **reconstructed** tensor in the model's KV
  dtype. The on-paper CR is a reconstruction-quality number, not an
  HBM byte-count number. A native vLLM integration that stores
  lattice indices directly is the next step — PR pending GPU
  validation.
- We have not tested every open-source LLM. Qwen3, Llama-3, DeepSeek,
  GLM-4, Gemma-4, Qwen2, and Phi-3 family models should work
  transparently; please open an issue if a `head_dim ∈ {64, 128, 256}`
  model fails.

**Live demo**:
<https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress>.
Qwen3-0.6B side-by-side with bf16, Q=10, Q=38, Q=152 on a free CPU tier.

**Paper + raw data**:
- <https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/reports/paper/kakeyalattice.pdf>
- <https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/v1_4_release/kv_128k_isoppl_n8>

Every number in this post reconciles with the raw JSON and a 3-line
reproducer (`benchmarks/extract_iso_ppl_table.py`). Happy to answer
questions about method, limitations, or how this compares to KIVI /
HQQ / QuantoQuantizedCache / SnapKV.

---

## Timing

- Post Tuesday / Wednesday / Thursday, **08:30 ET** (13:30 UTC) — HN's
  empirically best window for technical launches.
- Don't post from a throwaway. Low-karma authors get auto-flagged.
- Don't post on a major-news day (OpenAI / Anthropic launches,
  macro events). Check <https://news.ycombinator.com> for obvious
  front-page saturation before pulling the trigger.

## Expected failure modes and pre-prepared replies

Post these as follow-up comments to your own first comment if
questioned — keeps the thread coherent.

### "Why not just compare to KIVI?"

KIVI is 2-bit per-value scalar + per-token grouping. KakeyaLattice at
`q_range=10` (E8, 128-D head) is ~3.2 effective bits/value and gives
|Δppl| < 2 % on Qwen3-4B. Direct head-to-head is planned but we
prioritised TurboQuant because it is the strongest published
scalar-quant baseline and shares the same code path
(`benchmarks/multimodel_v14_kv_128k_report.py --tq-b-values`).

### "HBM savings claim is misleading if you decode to bf16"

Fair — and we explicitly call this out. The reference impl tests
reconstruction quality. Real HBM savings need a native vLLM (or
SGLang / TRT-LLM) integration that stores lattice indices. That PR is
scaffolded and waiting on GPU validation. The lattice-index format is
~14 bits per 8-D block at Q=38 which directly gives 2.3× HBM ratio.

### "n=8 passages is small"

Each passage is 2048 tokens × 64 evaluation positions = 512 per
channel. Total eval positions per codec operating point = 4096 (8 × 512)
across passages from 8 independent WikiText-103 documents. The paper's
appendix has a variance analysis showing n=8 × 64 is sufficient for the
CR differences reported here to exceed 2σ.

### "Streaming latency 0.25 ms — measured how?"

`benchmarks/v14_streaming_latency.py`. H200, CUDA 13.0, one-token
decode, all layers × all KV heads, averaged over 512 steps after 32
warm-up. Per-step wall-clock delta between codec-on and codec-off.
Raw log under `reports/v1_4_release/streaming/`.
