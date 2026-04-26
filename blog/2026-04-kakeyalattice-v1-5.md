# KV-cache compression in 2026: where KakeyaLattice fits in the landscape

**TL;DR** — This post is a landscape survey of LLM KV-cache
compression methods as they stand in April 2026, followed by a
placement of KakeyaLattice inside that landscape with real numbers.
If you're picking a KV compressor today you will not (and should
not) pick based on a single benchmark; the method you want is the
one that fits your bit budget, your quality tolerance, your serving
stack, and what you're already doing for weights and eviction.

All numeric claims in this post reconcile 1:1 with the public
benchmark JSON under
[`reports/v1_4_release/kv_128k_isoppl_n8/`](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/v1_4_release/kv_128k_isoppl_n8)
— no hand-tuned or curated-best numbers. Peer-method citations are
carried through to full arXiv / DOI entries in
[`ACKNOWLEDGMENTS.md`](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/ACKNOWLEDGMENTS.md).

---

## The four things you can do to a KV cache

KV cache is the dominant HBM consumer of transformer inference at
context lengths past a few thousand tokens. At 128 k context on
Qwen3-4B the KV cache alone is 18 GiB — more than double the 8 GiB
of model weights. At 1 M context it is the *only* thing that matters.

Four orthogonal axes exist to reduce it:

| axis | what it does | example |
|:---|:---|:---|
| **weight quantisation** | shrinks what the attention heads multiply against | GPTQ, AWQ, HQQ |
| **eviction** | drops or summarises KV entries you decide you don't need | SnapKV, H2O, Scissorhands |
| **KV quantisation** | stores each KV entry in fewer bits | QuantoQuantizedCache, TurboQuant, KIVI, KakeyaLattice |
| **attention sparsification** | changes which KV entries attention reads | Native Sparse Attention, YOCO |

KakeyaLattice is on the third axis. The other three are **orthogonal
and composable** with it — HQQ-weight + SnapKV-eviction +
KakeyaLattice-KV + native-sparse-attention is a perfectly sensible
4-way stack, and no part of it makes any of the others harder. Most
of the rest of this post is therefore *only* about the KV
quantisation axis.

## KV quantisation, four generations

### Generation 1 (2018–2022): "bf16 KV is fine"

Attention KV was stored in the same dtype as the model weights
(typically fp16 or bf16) and nobody questioned it. Contexts were
short enough (≤ 8 k) that KV was a small fraction of HBM. This
changed in late 2022 when long-context serving started landing.

### Generation 2 (2023–2024): per-channel scalar quantisation

The first wave of KV quantisers applied per-channel scalar
quantisation (INT8 or INT4 with a per-channel scale and zero-point):

- **SmoothQuant** ([Xiao et al., ICML 2023](https://arxiv.org/abs/2211.10438))
  pioneered the "migrate difficulty between weights and activations"
  framing. The KV variant extended it to the K and V tensors.
- **QuantoQuantizedCache** and **HQQQuantizedCache** landed in
  Hugging Face [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)
  as in-tree `DynamicCache` alternatives. They are what most
  non-specialists will compare any new KV quantiser to first.
- **TurboQuant** ([Zandieh et al., 2024, arXiv:2406.17005](https://arxiv.org/abs/2406.17005))
  is the strongest published per-channel scalar baseline, using
  learned scale migration and careful grouping to push the
  achievable CR at matched quality up.

**What this generation does well**: simple code path, fits on
existing kernels, usually under 5 % |Δppl| at 2× compression.
**Where it struggles**: real LLM KV activations are **not Gaussian
and not isotropic**. A per-channel scalar quantiser has to budget
bits for the worst-case channel, wasting bits on everything else.
At aggressive settings (≥ 3× CR) this starts to dominate.

### Generation 3 (2024–2025): low-bit per-token grouping

A second wave tried to push the bit budget below 4 bits/value:

- **KIVI** ([Liu et al., 2024, arXiv:2402.02750](https://arxiv.org/abs/2402.02750))
  groups K per-channel and V per-token, quantises both to 2 bits
  asymmetrically, keeps a small residual in full precision. Tuning-
  free, which is a big deal.
- **MiKV**, **WKVQuant**, and adjacent variants played with mixed
  precision across heads or layers.

**What this generation does well**: reaches 4× CR and beyond.
**Where it struggles**: the per-channel / per-token grouping still
can't neutralise the *joint* heavy tails across multiple channels.
At very low bit budgets the quantisation noise starts to correlate
with structure the attention step can detect, and |Δppl| climbs
non-linearly.

### Generation 4 (2025–2026): basis rotation + lattice

The insight that drives KakeyaLattice is older than either
generation above and comes from the Zamir–Feder line of work
([Zamir & Feder, 1996, doi:10.1109/18.508838](https://doi.org/10.1109/18.508838)):

> A well-chosen basis rotation can **gaussianise and isotropise** a
> heavy-tailed, non-isotropic source. Once gaussianised, a
> **nested-lattice quantiser** achieves compression provably close
> to the entropy bound, and strictly better than any scalar
> quantiser at the same bit budget.

For LLM KV activations the rotation is a Sylvester–Hadamard matrix
(1867 vintage — no training, no calibration, just a ±1 sign-pattern
matrix scaled by 1/√D). The lattice is D4 or E8, the densest known
in dimensions 4 and 8; the closest-point decoders are Conway–Sloane
classics ([Conway & Sloane, 1999, doi:10.1007/978-1-4757-6568-7](https://doi.org/10.1007/978-1-4757-6568-7)).

**KakeyaLattice** ships this construction as a `DynamicCache`
subclass in the `kakeyalattice` PyPI package. The codec is
**stateless per-vector**, so no calibration, no warm-up, and
streaming / online decode is supported out of the box. Head-to-head
iso-PPL numbers follow.

## KakeyaLattice vs TurboQuant — iso-PPL, four models, n=8

All numbers below are measured with
[`benchmarks/multimodel_v14_kv_128k_report.py`](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/benchmarks/multimodel_v14_kv_128k_report.py)
on an NVIDIA H200, real vLLM prefill, real FlashAttention bf16
forward, 128 k context, WikiText-103, n=8 passages × 64 eval positions
per passage = 512 target positions per channel. Raw JSON at
[`reports/v1_4_release/kv_128k_isoppl_n8/`](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/v1_4_release/kv_128k_isoppl_n8)
— reproducible end-to-end from
[`benchmarks/extract_iso_ppl_table.py`](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/benchmarks/extract_iso_ppl_table.py).

| model                          | ≤ 1 % \|Δppl\|           | ≤ 2 % \|Δppl\|           |
|:-------------------------------|:-------------------------|:-------------------------|
| Qwen3-4B                       | **2.40× vs 1.95×** (+23.3 %) | **2.77× vs 2.18×** (+26.9 %) |
| GLM-4-9B-Chat                  | **1.73× vs oor** (KL only)   | **2.44× vs 1.77×** (+37.8 %) |
| Gemma-4-E4B                    | **3.04× vs 3.04×** (tied)    | **3.04× vs 3.04×** (tied)    |
| DeepSeek-R1-Distill-Qwen-1.5B  | **2.29× vs 2.09×** (+9.2 %)  | **2.43× vs 2.36×** (+3.3 %)  |

*"oor" = out of range: TurboQuant's densest bit setting (b=8) could
not meet the ≤ 1 % quality target on GLM-4-9B-Chat. KakeyaLattice
reaches that target at 1.73× CR.*

**Pattern.** KakeyaLattice wins by 9 %–38 % compression ratio at
the 1–2 % |Δppl| band that production deployments tune for. At very
loose quality budgets (≥ 5 % |Δppl|) scalar quantisers catch up
because the 32-bit-per-block `qmax` overhead starts to dominate the
nested-lattice rate. At very tight budgets (≤ 0.5 %) on most models
KakeyaLattice is the only codec that reaches the target.

Gemma-4-E4B is the easiest model in this set — effective-parameter
design plus Q/K norms have already aggressively normalised its KV.
Both codecs saturate at 3.04× at the densest bit settings and tie
structurally above that.

## Streaming latency

Codec per-decode-step overhead on H200, measured across all four
models × three operating points (Q=10 aggressive, Q=38 balanced,
Q=152 near-lossless): **~0.25 ms**. Bf16 decode step at batch 1 is
15–30 ms. Codec is **< 2 % of decode latency** — invisible unless
you are specifically hunting for it. Raw log at
[`reports/v1_4_release/streaming/`](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/v1_4_release/streaming).

## The DeepSeek-V4-Flash addendum

V4-Flash is interesting because it uses its own FP8-E4M3 per-64-block
KV scheme as the production baseline, not bf16. We measured
KakeyaLattice E8 Q=38 against V4-Flash's own FP8 on 8 semantically
diverse passages × three representative V4 attention layer types
(0/SWA, 2/c4a, 3/c128a) with trained V4-Flash weights on 2 × H200
(total compute cost under USD 0.05):

> **22 % bit reduction per KV vector with non-regressive quality at
> 95 % confidence (layer-weighted rel-MSE vs hardware FP8 = 0.959,
> 95 % CI [0.935, 0.983]; n=8).**

Bit saving is an **exact codec-arithmetic constant** (3296 vs 4224
bits/vec) that does not depend on passages or layers. Quality is a
layer-weighted win across V4-Flash's 43-layer mix (3 SWA + 20 c4a +
20 c128a). Full per-passage tables at
[`reports/v1_5_release/dsv4_stage075/FINDINGS_N8.md`](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/reports/v1_5_release/dsv4_stage075/FINDINGS_N8.md).

## When not to use KakeyaLattice

- If your `head_dim` is not a power of 2 and divisible by 4 or 8
  (very rare in modern LLMs — essentially all of Qwen3, Llama-3,
  DeepSeek-R1-Distill, GLM-4, Gemma-4 are fine).
- If you are compressing *weights* (use HQQ / GPTQ / AWQ instead —
  orthogonal problem).
- If your quality budget is loose enough that a scalar KV quantiser
  already meets it at your target CR. No point adding code.
- If your serving stack is a custom CUDA path with no
  `DynamicCache` hook and you cannot patch it. Today we ship a
  `transformers.DynamicCache` subclass and a vLLM capture plugin;
  native vLLM integration (lattice-index pages rather than round-
  trip through KV dtype) is the next PR.

## The one thing you should read after this post

If you are evaluating KV compressors and care about rigor, read
[Zandieh, Han, Karbasi & Mirrokni, 2024 (TurboQuant, arXiv:2406.17005)](https://arxiv.org/abs/2406.17005).
It's the clearest write-up of the per-channel scalar state of the
art and it's what KakeyaLattice had to beat at matched quality to
justify existing.

## Try it

- **Browser demo (no install)**:
  <https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress>
- **PyPI**: `pip install kakeyalattice`
- **Code, paper, raw benchmark data**:
  <https://github.com/FluffyAIcode/LLM-KV--Cache-compress>
- **Drop your deployment on the public list**:
  [`DEPLOYMENTS.md`](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/DEPLOYMENTS.md)
- **Cite**: GitHub's sidebar "Cite this repository" widget, sourced
  from
  [`CITATION.cff`](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/CITATION.cff).

---

*Thanks to the authors of TurboQuant, KIVI, SmoothQuant, HQQ,
QuantoQuantizedCache, SnapKV, H2O, Scissorhands, vLLM,
FlashAttention, and transformers — the comparisons above are only
possible because you open-sourced your implementations and published
your numbers.*
