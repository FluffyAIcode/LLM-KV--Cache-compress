---
title: E8-lattice KV cache compression, from first principles
published: false
description: Why a 1867 math trick (Sylvester-Hadamard rotation) plus a 1999 algorithm (Conway-Sloane E8 closest-point) beats scalar quantization by 9-38% on modern LLM KV caches. Drop-in DynamicCache subclass, pip install.
tags: llm, quantization, python, performance
cover_image: https://raw.githubusercontent.com/FluffyAIcode/LLM-KV--Cache-compress/main/assets/hero_pareto.png
canonical_url: https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/blog/2026-04-kakeyalattice-v1-5.md
---

## TL;DR

The KV cache is the biggest memory consumer in modern LLM serving.
Every per-channel scalar quantizer you've ever tried — INT8,
SmoothQuant-KV, TurboQuant, `QuantoQuantizedCache`, KIVI — is
leaving **9-38% compression on the table** at the quality budgets
production cares about, and for a fixable reason. This post explains
what the fix is (a 1867 ±1 matrix + a 1999 lattice algorithm),
why it works (real LLM KV is heavy-tailed and non-isotropic), and
how it ships (`pip install kakeyalattice`, drop-in
`transformers.DynamicCache` subclass, 10 lines of integration).

Numbers below are from real vLLM prefill + real FlashAttention bf16
on NVIDIA H200, 128k context, WikiText-103, n=8 passages × 64 eval
positions per passage. Raw JSON and reproducer at the
[GitHub repo][repo]. Nothing is mocked.

## The problem scalar quantizers have

At 128k context on Qwen3-4B the KV cache alone is 18 GiB — larger
than the 8 GiB of model weights. At 1M context the KV cache is the
**only** memory cost that matters. Compressing it without hurting
perplexity is the fastest path to more concurrent users per GPU
node.

The standard approach is **per-channel scalar quantization**: for
each KV channel, store an INT4 or INT8 value plus a per-channel
scale. SmoothQuant-KV (Xiao et al., ICML 2023,
[arXiv:2211.10438](https://arxiv.org/abs/2211.10438)),
`QuantoQuantizedCache` in HF transformers, and
TurboQuant (Zandieh et al., 2024,
[arXiv:2406.17005](https://arxiv.org/abs/2406.17005)) all follow
this recipe with different scale-selection tricks. At the tight
quality budget production deployments tune for (**≤1% perplexity
loss**), the strongest published scalar quantizer (TurboQuant) tops
out at compression ratios like:

- Qwen3-4B: 1.95×
- GLM-4-9B-Chat: *cannot reach 1% at any bit setting*
- DeepSeek-R1-Distill-Qwen-1.5B: 2.09×

Why can't it do better? **Because real LLM KV activations are
heavy-tailed and non-isotropic.** A per-channel scalar quantizer
must budget bits for the worst-case channel (the one with the
heaviest tail), which wastes bits on every other channel. At
aggressive compression ratios this dominates.

We verified this on DeepSeek-V4-Flash with trained weights: the
isotropy-variance ratio (variance of the largest-variance coordinate
divided by the smallest) across the `csa_pool_kv_ratio4` stream
is **732,400**. One coordinate out of 512 has variance nearly a
million times larger than another. A scalar quantizer has to
accommodate both.

## The fix, in two steps

### Step 1 — Sylvester-Hadamard rotation (1867)

A **Hadamard matrix** H of size D×D has entries in {+1, −1} and
satisfies `Hᵀ H = D · I`. James Joseph Sylvester constructed one
in 1867 as a recursive ±1 sign-pattern:

```
H_2 = [[+1, +1],
       [+1, −1]]

H_{2D} = [[H_D,  H_D],
          [H_D, −H_D]]
```

For a KV vector `x ∈ R^D`, the rotation `y = H x / √D` is:

- **Norm-preserving** — `Hᵀ H / D = I`, so `||y|| = ||x||`.
- **Coordinate-mixing** — each output coordinate is a ±1 sum of all
  input coordinates, divided by √D.
- **Cheap** — computable in `O(D log D)` via a radix-2 algorithm
  (essentially an FFT without complex numbers).

Empirically, on every LLM family we tested (Qwen3, Llama-3,
DeepSeek, GLM, Gemma), applying Sylvester-Hadamard rotation to the
KV vectors **gaussianizes** their distribution: kurtosis drops
toward 3, isotropy-variance ratio falls by 1–3 orders of magnitude,
and the Wasserstein-2 distance to a matched Gaussian drops into
the 0.05–0.5 range. We call this the **non-Gaussian audit** (paper
gates: kurt<0.5, iso-var<1.5, had-var<1.5, W2/σ<0.05) and run it
as a sanity check before claiming the rotation works on a new
model family.

### Step 2 — nested-lattice closest-point snap

Once the vector is rotated into a well-behaved distribution, we
quantize **jointly across groups of coordinates** (4 or 8 at a
time) by snapping each group to its closest point on a lattice.

The `D4` lattice in 4 dimensions and the `E8` lattice in 8
dimensions are the **densest known sphere packings** in those
dimensions ([Conway & Sloane, 1999,
doi:10.1007/978-1-4757-6568-7](https://doi.org/10.1007/978-1-4757-6568-7)).
Density here means: for a given quantization error budget, a D4 or
E8 lattice packs more codepoints into the space than any arrangement
of axis-aligned scalar codepoints. Specifically:

- D4 gains **1.5 dB** in packing efficiency over scalar per-axis
  quantization at the same bit rate.
- E8 gains **3.2 dB** — roughly a 2× efficiency win.

Translated to LLM KV caches, this means: at the same total bit
budget, D4/E8 lattice-quantized K/V vectors have lower
reconstruction MSE than scalar-quantized vectors **by a provable
amount**. And since we've rotated the vectors to be near-Gaussian
before snapping, the classical nested-lattice shaping-gain bound
(Zamir & Feder 1996,
[doi:10.1109/18.508838](https://doi.org/10.1109/18.508838))
actually applies — the theoretical gain is achievable, not
hypothetical.

The closest-point decoders for D4 and E8 are textbook 1999
algorithms. D4's is a 4-case argmin on the integer lattice plus a
half-integer shift; E8's is a slightly more elaborate case
analysis on Z^8 plus D_8^+ coset selection. Both run in pure
PyTorch at roughly the cost of a LayerNorm.

## The numbers

Head-to-head with TurboQuant on iso-PPL compression ratio at
≤1% perplexity loss (higher CR = more bits saved at same quality):

| model | KakeyaLattice CR | TurboQuant CR | KL advantage |
|:------|-----------------:|--------------:|-------------:|
| Qwen3-4B                      | **2.40×** | 1.95× | **+23.3%** |
| GLM-4-9B-Chat                 | **1.73×** | (unreachable) | KL only |
| Gemma-4-E4B                   | **3.04×** | 3.04× | tied (saturated) |
| DeepSeek-R1-Distill-Qwen-1.5B | **2.29×** | 2.09× | **+9.2%** |

At ≤2% the KakeyaLattice advantage grows to +27, +38, tied, +3%
respectively. Raw JSON, extractor script, and hero chart generator
are all in the repo; the table above is regenerated from
`reports/v1_4_release/kv_128k_isoppl_n8/*.json` by running
`python benchmarks/extract_iso_ppl_table.py`.

## Decode latency

The extra step is one Hadamard rotate + one lattice snap + one
unscale per decode token per attention layer. Measured on NVIDIA
H200 across four models × three operating points: **~0.25 ms per
decode step**, or **<2% of a typical 15-30 ms bf16 decode step at
batch 1**. You will not notice it.

## How to use it

Three lines of Python once the package is installed:

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
    variant="e8", q_range=38,  # balanced default: ~2.3x CR, <1% |Δppl|
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

That's it. Any `transformers` model whose `head_dim` is a power of 2
and divisible by 8 (E8) or 4 (D4) works — Qwen3, Llama-3,
DeepSeek-R1-Distill, GLM-4, Gemma-4, Phi-3.

## What KakeyaLattice does *not* do

- **Weight quantization.** That's orthogonal — stack HQQ/GPTQ/AWQ
  weight quantization with KakeyaLattice KV compression.
- **Eviction.** SnapKV, H2O, Scissorhands are also orthogonal —
  they compose multiplicatively with KakeyaLattice.
- **Zero-latency decode.** The ~0.25ms/step overhead is real, just
  small. A fused Triton kernel would cut it further.
- **HBM savings in the Python reference impl.** Today
  `KakeyaLatticeCache` stores the reconstructed tensor in the
  model's KV dtype; the on-paper CR measures reconstruction
  quality, not HBM bytes. A native vLLM integration that stores
  lattice indices directly in the paged KV cache is in progress
  (see the [vLLM RFC][vllm-rfc]).

## Try it

- **Live demo (no install)**:
  <https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress>
- **GitHub + paper + raw data**:
  <https://github.com/FluffyAIcode/LLM-KV--Cache-compress>
- **PyPI**: `pip install kakeyalattice`
- **Cite**: GitHub's sidebar "Cite this repository" widget
  (sourced from `CITATION.cff`).

Pair this with the practice-first companion post,
["Qwen3 KV cache compression in 10 lines"][post2], if you want to
skip the theory and just ship it.

[repo]: https://github.com/FluffyAIcode/LLM-KV--Cache-compress
[vllm-rfc]: https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/docs/announce/vllm_integration_issue.md
[post2]: https://dev.to/<your-handle>/qwen3-kv-cache-in-10-lines-<slug>
