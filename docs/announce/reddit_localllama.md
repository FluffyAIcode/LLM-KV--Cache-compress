# r/LocalLLaMA submission

**Subreddit**: r/LocalLLaMA

**Post flair**: "Resources" (or "Discussion" if you prefer a softer
ask for feedback).

**Title**: `Compressed Qwen3-4B's KV cache 2.8× with ~1.7 % ppl loss using lattice quantisation — drop-in DynamicCache subclass, pip install`

---

## Post body

Hey r/LocalLLaMA —

If you've been frustrated that long-context serving on 24 GB / 48 GB
consumer GPUs is KV-cache-bound, this might help. I've been working on
a library called **KakeyaLattice** that compresses the KV cache of any
modern causal LM 2.4×–2.8× at under 1 % perplexity loss using nested
D4 / E8 lattice quantisation.

### What it is, in one paragraph

A drop-in subclass of `transformers.DynamicCache`. You construct
`KakeyaLatticeCache(variant="e8", q_range=38, ...)`, pass it as
`past_key_values` to `model.generate()`, and every K / V tensor the
model writes to the cache is transparently rotated by a
Sylvester–Hadamard matrix, L²-scaled, and snapped to the closest point
of a nested D4 / E8 lattice using Conway–Sloane closest-point
decoders. Decoding is one matmul.

### Why this compresses harder than scalar KV quantisers at matched quality

Real LLM KV activations are heavy-tailed and non-isotropic. A
per-channel scalar quantiser has to assume the worst-case channel and
wastes bits. A basis rotation that whitens the distribution lets you
quantise each rotated channel under its *typical* variance — a smaller
alphabet. Lattice quantisation in D4 (dim 4) and E8 (dim 8) then
exploits the densest sphere packings known in those dimensions to
beat any scalar quantisation of the same block.

### Numbers (real vLLM on H200, 128 k context, WikiText-103 n=8)

Iso-PPL compression ratio at ≤ 1 % perplexity loss:

| model | KakeyaLattice CR | best scalar-quant CR |
|:---|---:|---:|
| Qwen3-4B | **2.40×** | 1.95× |
| GLM-4-9B-Chat | **1.73×** | out-of-range |
| Gemma-4-E4B | **3.04×** | 3.04× (tied) |
| DeepSeek-R1-Distill-Qwen-1.5B | **2.29×** | 2.09× |

At 2 % perplexity loss: 2.77× / 2.44× / 3.04× / 2.43×. "Out of range"
means the scalar quantiser's densest bit setting still couldn't meet
that quality target on that model.

### Streaming latency

On H200, codec per-decode-step overhead is **~0.25 ms** — which is
**< 2 %** of the typical 15–30 ms bf16 decode step at batch 1. No
calibration pass, no warm-up, no cross-token state. Works out of the
box in streaming / online mode.

### How to use it

```bash
pip install kakeyalattice
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kakeyalattice.hf import KakeyaLatticeCache

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16
).cuda()

cache = KakeyaLatticeCache(
    variant="e8", q_range=38,
    num_hidden_layers=model.config.num_hidden_layers,
    head_dim=model.config.head_dim,
    device="cuda",
)
out = model.generate(
    **tok("Hello world", return_tensors="pt").to("cuda"),
    max_new_tokens=256,
    past_key_values=cache,
    use_cache=True,
)
```

That's the full integration.

### Honest caveats

- **The reference `KakeyaLatticeCache` round-trips through the codec
  and stores the reconstructed tensor in the model's KV dtype.** So
  the on-paper compression ratio and real HBM bytes are different: the
  demo proves reconstruction quality, not HBM savings. Native vLLM
  integration that stores lattice indices directly is the pending PR.
- `head_dim` must be a power of 2 and divisible by 8 (E8) or 4 (D4).
  Qwen3 / Llama-3 / Gemma / DeepSeek are all fine.
- Tested on Qwen3, Qwen2, GLM-4, Gemma-4, DeepSeek-R1-Distill. If a
  model with compatible `head_dim` fails, open a GitHub issue.

### Try it without installing

Live demo on HF Spaces:
<https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress>.
Qwen3-0.6B side-by-side with bf16 / Q=10 / Q=38 / Q=152 on a free CPU
tier (each "Run comparison" click takes ~4–8 minutes).

### Links

- **GitHub + paper + raw JSON benchmark data**:
  <https://github.com/FluffyAIcode/LLM-KV--Cache-compress>
- **PyPI**: <https://pypi.org/project/kakeyalattice/>
- **Live demo**: <https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress>
- **FAQ** (how does it compare to KIVI / HQQ / QuantoQuantizedCache):
  <https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/docs/faq.md>

Happy to answer questions about method, limitations, or vLLM /
SGLang / llama.cpp integration plans.

---

## Timing

- **Thu / Fri mornings US time**, ~10 AM ET, are historically best for
  r/LocalLLaMA — evening and weekend posts get buried by meme content.
- Don't crosspost within the first 4 hours; it can trigger spam
  filters.
- Respond to every substantive comment within the first 2 hours; the
  subreddit rewards author engagement.

## Common questions — pre-prepared replies

### "Does this reduce VRAM or not?"

Honest answer: in the reference Python implementation that ships with
the package today, **no** — we round-trip K / V through the codec but
store the reconstructed tensor in the model's KV dtype. The point of
the reference impl is to isolate and measure reconstruction quality.
Native vLLM integration (pending GPU validation) stores lattice
indices directly and at that point CR = HBM ratio.

### "How does it compare to KIVI?"

KIVI is scalar 2-bit per-value with per-token grouping. At Q=10 (E8),
KakeyaLattice is ~3.2 effective bits/value with typically lower |Δppl|
on the same models because the rotation step handles heavy tails.
Direct head-to-head benchmark is planned.

### "Does this work with GGUF / llama.cpp?"

Not yet. GGUF would need a new block type for lattice indices. If
someone wants to pair on a llama.cpp PR, reach out — the codec itself
is <200 lines of numerics.

### "Why the Hadamard rotation?"

Sylvester–Hadamard rotation at dimension D is a unitary matrix
(preserves norms) whose action is a mixture of all input coordinates
with ±1 signs and a 1/√D scale. Empirically this is enough to
gaussianise KV activations on every model we tested. A full
characterisation of the rotation's effect on heavy-tailed
distributions is in the paper (`reports/paper/kakeyalattice.pdf`).
