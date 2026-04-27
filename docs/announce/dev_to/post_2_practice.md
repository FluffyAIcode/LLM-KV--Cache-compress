---
title: Qwen3 KV cache compression in 10 lines of Python
published: false
description: A drop-in transformers.DynamicCache subclass that compresses Qwen3's KV cache 2.4-2.8x at under 1% perplexity loss. Three operating points, one pip install, no calibration.
tags: python, llm, transformers, huggingface
cover_image: https://raw.githubusercontent.com/FluffyAIcode/LLM-KV--Cache-compress/main/assets/hero_pareto.png
canonical_url: https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/blog/2026-04-kakeyalattice-v1-5.md
---

## TL;DR

A 10-line integration to compress your Qwen3 / Llama-3 / DeepSeek /
GLM-4 / Gemma-4 KV cache 2.4×–2.8× at under 1% perplexity loss.
Works with any HF `transformers` model whose `head_dim` is a
power of 2 divisible by 4 or 8. No calibration, no warm-up,
streaming-safe. This post is the practice-first companion to
[the theory post][post1].

## The setup

You've built an LLM inference service. It was fine until a customer
asked for a 128k context and your GPU melted. KV cache turns out to
be the biggest memory consumer by far — more than the model weights
at long contexts. Compressing the KV cache 2-3× at no quality cost
would immediately let you fit twice as many concurrent users on the
same hardware.

`QuantoQuantizedCache` in HF transformers does 2× at small quality
cost. TurboQuant does a bit better (published
[arXiv:2406.17005](https://arxiv.org/abs/2406.17005)). KIVI pushes to
4× with 2-bit per-value ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750))
but the |Δppl| grows.

`kakeyalattice` lands between them: **2.4-2.8× CR at under 1% perplexity
loss across four open-source model families**, measured on real vLLM
with real FlashAttention on NVIDIA H200. Drop-in
`transformers.DynamicCache` subclass.

Let's ship it.

## Install

```bash
pip install kakeyalattice
```

## The 10-line integration

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kakeyalattice.hf import KakeyaLatticeCache

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16
).cuda()

cache = KakeyaLatticeCache(
    variant="e8", q_range=38,   # balanced default: ~2.3x CR, <1% |Δppl|
    num_hidden_layers=model.config.num_hidden_layers,
    head_dim=model.config.head_dim,
    device="cuda",
)

inputs = tok("Explain quantisation in one paragraph:", return_tensors="pt").to("cuda")
out = model.generate(
    **inputs,
    max_new_tokens=256,
    past_key_values=cache,      # <-- that's the whole integration
    use_cache=True,
)
print(tok.decode(out[0], skip_special_tokens=True))
```

The `past_key_values=cache` argument is where the `transformers`
library replaces its default `DynamicCache` with our subclass. From
that point on, every K and V the model writes to the cache is
transparently rotated, scaled, and lattice-quantized; every read
decodes one matmul + one unscale.

## Three operating points

`q_range` tunes the aggressiveness of the lattice snap. Higher Q =
more lattice codepoints per dimension = higher quality, less
compression. Lower Q = fewer codepoints = more compression, more
error.

| config             | q_range | bits/vec @ head_dim=128 | typical \|Δppl\| on Qwen3 | use when |
|:-------------------|--------:|------------------------:|-------------------------:|:---------|
| aggressive         |      10 | 640 (−69 %)             | 1.5–2.5%                 | memory is the hard constraint |
| **balanced**       |  **38** | **880 (−57 %)**         | **0.5–1.0%**             | **default — production serving** |
| near-lossless      |     152 | 1920 (−6 %)             | <0.1%                    | quality-sensitive, last-resort deployments |

D4 variant (`variant="d4"`) works for head_dim divisible by 4 only
(e.g. Qwen2-0.5B's head_dim=64) and gives roughly half the
compression-gain of E8 at the same |Δppl|.

## What you get on each model

Real numbers from real vLLM prefill on NVIDIA H200, WikiText-103,
n=8 passages × 64 eval positions per passage = 512 target positions
per channel. Raw JSON under
[`reports/v1_4_release/kv_128k_isoppl_n8/`][raw] in the GitHub repo.

Iso-PPL compression ratio at ≤1% perplexity loss:

| model | CR |
|:------|---:|
| Qwen3-4B                      | **2.40×** |
| GLM-4-9B-Chat                 | **1.73×** |
| Gemma-4-E4B                   | **3.04×** |
| DeepSeek-R1-Distill-Qwen-1.5B | **2.29×** |

At ≤2%: 2.77× / 2.44× / 3.04× / 2.43×.

## Streaming-safe by construction

Unlike calibration-based quantizers (KIVI, SmoothQuant), KakeyaLattice
is **stateless per-vector**. The codec does not look across tokens,
does not collect statistics, does not need a warm-up pass. The first
token you decode is compressed identically to the millionth. This
means:

- Works with `model.generate(..., streaming=True)` out of the box.
- No calibration script to run before deployment.
- No surprising quality drift between different batch sizes or
  different input distributions.

On NVIDIA H200 the codec adds **~0.25 ms per decode step** — under
2% of a typical 15-30 ms bf16 decode step at batch size 1. You won't
see it on a wall-clock profile unless you're specifically hunting it.

## Operational checklist

Before you deploy:

- [ ] `head_dim` of your model is a power of 2 and divisible by 8
      (E8) or 4 (D4). Check `model.config.head_dim` — almost all
      modern LLMs pass.
- [ ] You are on `transformers >= 4.51`. Qwen3 support landed there.
- [ ] You have `torch >= 2.1` (GPU) or `torch >= 2.1` CPU build
      for development.
- [ ] You measured `|Δppl|` on your own eval set at `q_range=38`
      before shipping. Our numbers are on WikiText-103; your domain
      may differ by ±0.5%.
- [ ] You have a rollback plan (`past_key_values=DynamicCache()` is
      a one-line revert).

## When not to ship KakeyaLattice

Be honest with yourself:

- **Short-context serving (≤4k).** KV cache is small at short
  contexts; compression overhead is not worth it.
- **Real-time voice / sub-second latency budgets.** Codec overhead
  is small but non-zero; measure it on your stack.
- **Regulatory review.** A new library, even MIT-licensed and
  open-source, is a procurement hurdle. If HQQ + `DynamicCache`
  already meets your quality target, don't add code.
- **Model with `head_dim ∉ {64, 128, 256}`.** A handful of older
  models (some early Llama variants, some research MoEs) have
  `head_dim=96` or `head_dim=176`, which is not lattice-compatible.
  You can still use KakeyaLattice but only D4, not E8, and the
  numerical advantage is smaller.

## Try it without installing

The [HF Space][space] runs Qwen3-0.6B live, side-by-side with bf16
baseline at all three operating points. Click "Run comparison" and
you'll see four generated paragraphs at increasing compression ratios
— text quality degrades smoothly from essentially-identical (Q=152)
to slightly-different (Q=38) to noticeably-different-but-coherent
(Q=10).

## Links + cite

- Live demo (no install): <https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress>
- GitHub (MIT-licensed): <https://github.com/FluffyAIcode/LLM-KV--Cache-compress>
- PyPI: <https://pypi.org/project/kakeyalattice/>
- Paper draft: [`reports/paper/kakeyalattice.pdf`][paper]
  (arXiv submission pending)
- FAQ with KIVI / HQQ / Quanto / SmoothQuant comparison:
  [`docs/faq.md`][faq]
- Cite: GitHub's sidebar "Cite this repository" widget, sourced from
  [`CITATION.cff`][cite]

The companion theory post is [here][post1] — read it if you want to
understand *why* the rotation-plus-lattice trick works. If you just
want to ship faster inference, you're already done.

[post1]: https://dev.to/<your-handle>/e8-lattice-kv-cache-compression-from-first-principles-<slug>
[space]: https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress
[paper]: https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/reports/paper/kakeyalattice.pdf
[faq]: https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/docs/faq.md
[cite]: https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/CITATION.cff
[raw]: https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/v1_4_release/kv_128k_isoppl_n8
