# KakeyaLattice — Frequently Asked Questions

A short, direct answer to the 15 questions we get most often. If your
question is not here, open an issue at
<https://github.com/FluffyAIcode/LLM-KV--Cache-compress/issues> or ask on
the HF Space community tab.

- [What is KakeyaLattice?](#what-is-kakeyalattice)
- [What compression ratios can I actually expect?](#what-compression-ratios-can-i-actually-expect)
- [How does it compare to KIVI, HQQ, QuantoQuantizedCache, and SmoothQuant-KV?](#comparisons)
- [Does it work with vLLM / SGLang / TensorRT-LLM / llama.cpp?](#vllm-integration)
- [What models and `head_dim` values are supported?](#supported-models)
- [Does it require a calibration pass or warm-up?](#calibration)
- [Does the codec work in streaming / online generation mode?](#streaming)
- [How much runtime overhead does the codec add?](#runtime-overhead)
- [What hardware is required?](#hardware)
- [How do I choose `q_range`?](#choosing-q_range)
- [Does compression reduce real HBM usage, or only reconstruction error?](#real-hbm-savings)
- [Is perplexity the only metric you optimise for?](#beyond-perplexity)
- [Does KakeyaLattice work with quantised (INT4 / INT8) models?](#quantised-models)
- [How do I cite KakeyaLattice?](#citation)
- [Where can I try it without installing anything?](#browser-demo)

---

## What is KakeyaLattice?

KakeyaLattice is a Python library that compresses the KV cache of
transformer language models at inference time. It provides a single class,
`KakeyaLatticeCache`, that is a drop-in subclass of
`transformers.DynamicCache`. Every K and V tensor the model writes to the
cache is rotated by a Sylvester–Hadamard matrix, adaptively L²-scaled, and
snapped to the closest point of a nested D4 (dim 4) or E8 (dim 8) lattice.
Decoding is a single matmul.

The result is **2.4× to 2.8× KV compression at <1 % perplexity loss** on
Qwen3, Llama-3, DeepSeek, GLM, and Gemma — verified in real vLLM with real
FlashAttention bf16 forward on H200.

## What compression ratios can I actually expect?

At the 1 % perplexity-loss budget that typical production deployments
target, we measure (128 k context, WikiText-103, n=8 passages × 64 positions
each, real vLLM + FlashAttention on H200):

| model                          | KakeyaLattice CR | best scalar-quant baseline |
|:-------------------------------|-----------------:|---------------------------:|
| Qwen3-4B                       | **2.40×**        | TurboQuant 1.95×           |
| GLM-4-9B-Chat                  | **1.73×**        | TurboQuant out-of-range    |
| Gemma-4-E4B                    | **3.04×**        | TurboQuant 3.04× (tied)    |
| DeepSeek-R1-Distill-Qwen-1.5B  | **2.29×**        | TurboQuant 2.09×           |

At 2 % perplexity loss the numbers go to 2.77× / 2.44× / 3.04× / 2.43×.
See the [full table in the README](../README.md#headline-numbers) for
0.5 % / 1 % / 2 % targets side-by-side with TurboQuant.

## <a id="comparisons"></a>How does it compare to KIVI, HQQ, QuantoQuantizedCache, and SmoothQuant-KV?

Short answer, by category. Full citations and links are in
[`ACKNOWLEDGMENTS.md`](../ACKNOWLEDGMENTS.md#peer-methods-we-compare-against);
we carry the one-line reference next to each entry here for
self-containment.

- **vs per-channel scalar quantisers** — **TurboQuant** ([Zandieh et al.,
  2024, arXiv:2406.17005](https://arxiv.org/abs/2406.17005)),
  **SmoothQuant-KV** ([Xiao et al., ICML 2023,
  arXiv:2211.10438](https://arxiv.org/abs/2211.10438)), and
  **`QuantoQuantizedCache`** (Hugging Face transformers in-tree).
  KakeyaLattice wins at tight quality targets (≤ 1 % |Δppl|) by
  **9 %–38 %** compression ratio across the four models we measure
  (see [Headline numbers](../README.md#headline-numbers)). At very
  loose targets (≥ 5 % |Δppl|) scalar quantisers catch up because
  the 32-bit-per-block `qmax` overhead starts to dominate the
  nested-lattice rate. Our benchmark harness
  ([`benchmarks/multimodel_v14_kv_128k_report.py`](../benchmarks/multimodel_v14_kv_128k_report.py))
  sweeps KakeyaLattice's `q_range` and TurboQuant's `b` on the same
  code path so the comparison is iso-harness.
- **vs low-bit KV quantisation — KIVI** ([Liu et al., 2024,
  arXiv:2402.02750](https://arxiv.org/abs/2402.02750)) — KIVI is 2-bit
  per-value scalar with per-token grouping. KakeyaLattice at
  `variant="e8", q_range=10` sits at a comparable effective bit
  budget (~3.2 bits/value on head_dim=128) but typically achieves a
  lower |Δppl| because the Sylvester–Hadamard rotation gaussianises
  heavy-tailed KV distributions before quantisation — scalar KIVI
  cannot. A direct iso-bit head-to-head on Qwen3 / DeepSeek-R1-Distill
  is on the roadmap.
- **vs HQQ** ([Badri & Shaji, 2023](https://mobiusml.github.io/hqq_blog/))
  — HQQ is a *weight* quantiser, not a KV quantiser. Orthogonal.
  You can (and should) stack them: HQQ-quantised weights +
  KakeyaLattice KV cache.
- **vs eviction methods** — **SnapKV** ([Li et al., 2024,
  arXiv:2404.14469](https://arxiv.org/abs/2404.14469)),
  **H2O** ([Zhang et al., NeurIPS 2023,
  arXiv:2306.14048](https://arxiv.org/abs/2306.14048)),
  **Scissorhands** ([Liu et al., NeurIPS 2023,
  arXiv:2305.17118](https://arxiv.org/abs/2305.17118)) — these are
  eviction (which KV to keep), not quantisation (how to store).
  Orthogonal to KakeyaLattice; eviction + KakeyaLattice compose
  multiplicatively.

See
[`reports/v1_4_release/kv_128k_isoppl_n8/V14_VS_TQ_ISOPPL_REPORT.md`](../reports/v1_4_release/kv_128k_isoppl_n8/V14_VS_TQ_ISOPPL_REPORT.md)
for the full iso-PPL head-to-head vs TurboQuant.

## <a id="vllm-integration"></a>Does it work with vLLM / SGLang / TensorRT-LLM / llama.cpp?

- **vLLM**: yes, via the `vllm_backend/kakeya_v1_4_snapshot/` plugin, which
  installs as a `vllm.general_plugins` entry point and monkey-patches the
  Attention path to capture post-QK/V-norm, pre-RoPE tensors. See the
  plugin's README. Full native KV-dtype integration (no roundtrip) is the
  subject of a forthcoming PR and requires GPU validation.
- **SGLang**: not yet. The KV cache abstraction is similar to vLLM's; a
  port is planned.
- **TensorRT-LLM**: not yet. TRT-LLM's KV cache is custom CUDA; integration
  would require a rewritten codec kernel.
- **llama.cpp**: not yet. llama.cpp uses GGUF block formats; a KakeyaLattice
  GGUF block type would be a natural fit but is not implemented yet. Issue
  open at the llama.cpp repo once we have a working prototype.

## <a id="supported-models"></a>What models and `head_dim` values are supported?

KakeyaLattice requires `head_dim` to be a power of 2 and:

- divisible by **4** for the D4 variant (`variant="d4"`), or
- divisible by **8** for the E8 variant (`variant="e8"`).

Verified real-vLLM on H200:

| family          | example           | head_dim | variants    |
|:----------------|:------------------|---------:|:------------|
| Qwen3           | Qwen/Qwen3-0.6B   | 128      | d4 / e8     |
| Qwen3           | Qwen/Qwen3-4B     | 128      | d4 / e8     |
| Qwen2           | Qwen/Qwen2-0.5B   | 64       | d4          |
| Llama-3.2       | meta-llama/Llama-3.2-1B | 64  | d4          |
| DeepSeek-R1-Distill | DeepSeek-R1-Distill-Qwen-1.5B | 128 | d4 / e8 |
| GLM-4-9B-Chat   | THUDM/glm-4-9b-chat | 128    | d4 / e8     |
| Gemma-4-E4B     | google/gemma-4-e4b | 256    | d4 / e8     |

Any model outside this list with a compatible `head_dim` is expected to
work transparently. Report failures as issues with the model name and
config JSON.

## <a id="calibration"></a>Does it require a calibration pass or warm-up?

No. KakeyaLattice is a **stateless per-vector codec**. No calibration set,
no offline statistics, no warm-up tokens. The first token you decode is
compressed identically to the millionth. This is what lets the codec run
in streaming / online mode without any algorithmic changes.

## <a id="streaming"></a>Does the codec work in streaming / online generation mode?

Yes — by construction. Every step of `V14KakeyaZamirLatticeGPU.roundtrip`
is a pure per-vector function. There is no cross-token state. Measured
per-decode-step codec latency on H200 is **~0.25 ms**, which is **< 2 %**
of a typical 15–30 ms bf16 decode step at batch size 1. See
`reports/v1_4_release/streaming/V14_STREAMING_REPORT.md`.

## <a id="runtime-overhead"></a>How much runtime overhead does the codec add?

- **Prefill**: <5 % of total prefill time on H200, measured across all
  four benchmark models.
- **Decode**: ~0.25 ms per decode step per KV head per layer, or
  <2 % of the total bf16 decode step cost at batch 1.
- **Memory**: the current reference implementation round-trips K and V
  through the codec and stores the reconstructed tensors in the model's
  KV dtype, so *HBM bytes on paper* are unchanged. A forthcoming
  vLLM-native PR stores the lattice indices directly, at which point the
  compression ratio becomes the HBM ratio. The reference impl's purpose
  is reconstruction-quality proof, not HBM savings.

## <a id="hardware"></a>What hardware is required?

- **To run `kakeyalattice` on a toy model (Qwen2-0.5B / Qwen3-0.6B)**:
  any CPU with 4+ GB RAM works, though slowly. The HF Space demo runs
  on a free 2-core CPU instance.
- **To reproduce the headline 128 k iso-PPL numbers**: one NVIDIA H100
  or H200 (we used H200 on vast.ai). A100 / L40S will work but will take
  longer per sweep.
- **To use the vLLM snapshot plugin**: any CUDA-capable GPU vLLM supports.

## <a id="choosing-q_range"></a>How do I choose `q_range`?

`q_range` is the scalar-alphabet size per lattice dimension; bits scale
as `D * log2(q_range) + 32` where D = head_dim. Practical settings
(E8 variant, head_dim = 128):

| q_range | bits/vec | CR vs bf16 | typical |Δppl| on Qwen3 | when to use |
|--------:|---------:|-----------:|-----------------------:|:---|
|   4     |  288     |  7.1×      | 8–12 %                | mostly useful for upper-bound sanity checks |
|  10     |  640     |  3.2×      | 1.5–2.5 %             | aggressive — memory-bound inference |
|  22     |  736     |  2.8×      | 0.8–1.2 %             | strong quality + aggressive CR |
|  38     |  880     |  2.3×      | 0.5–1.0 %             | **recommended default** |
|  76     | 1344     |  1.5×      | 0.1–0.3 %             | quality-critical serving |
| 152     | 1920     |  1.1×      | < 0.1 %               | near-lossless — use as reference |

Rule of thumb: start at 38, drop to 22 if you have headroom on quality,
climb to 76 if you cannot afford any user-visible quality regression.

## <a id="real-hbm-savings"></a>Does compression reduce real HBM usage, or only reconstruction error?

Today, in the `KakeyaLatticeCache` reference implementation, we
round-trip through the codec and store the **reconstructed** tensor in
the model's KV dtype. So the on-paper compression ratio and the HBM byte
count are different: the demo proves *reconstruction quality*, not HBM
savings.

The forthcoming `vllm_backend` native integration (PR pending) stores
the lattice indices directly in the vLLM KV cache page format, at
which point the compression ratio *is* the HBM ratio. Track progress in
the
[vLLM integration meta-issue](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/issues).

## <a id="beyond-perplexity"></a>Is perplexity the only metric you optimise for?

No. The paper (`reports/paper/kakeyalattice.pdf`) also reports:

- Needle-in-a-haystack (NIAH) retrieval at 128 k context.
- Top-1 token match rate between bf16 and compressed generation.
- Per-layer K- and V-MSE and cosine similarity.

NIAH results (`reports/v1_4_release/niah/`) show KakeyaLattice preserves
long-context retrieval reliability at the same bit budgets where
TurboQuant starts dropping retrieved-needle accuracy.

## <a id="quantised-models"></a>Does KakeyaLattice work with quantised (INT4 / INT8) models?

Yes. KakeyaLattice compresses activations (K and V tensors), not weights.
It is orthogonal to weight quantisation. Stack HQQ- or GPTQ-quantised
weights with a KakeyaLattice KV cache to get multiplicative memory
savings. We have tested this informally; the paper's primary numbers
use bf16 weights for clean ablation.

## <a id="citation"></a>How do I cite KakeyaLattice?

GitHub surfaces a **"Cite this repository"** widget in the sidebar
that exports BibTeX / APA / CFF from
[`CITATION.cff`](../CITATION.cff). For manual citation:

```bibtex
@software{kakeyalattice2026,
  title     = {KakeyaLattice: Nested-Lattice KV-Cache Compression for Large Language Models},
  author    = {Li, Allen},
  year      = {2026},
  version   = {1.5.0},
  month     = apr,
  license   = {MIT},
  url       = {https://github.com/FluffyAIcode/LLM-KV--Cache-compress},
  note      = {See reports/paper/kakeyalattice.pdf for the companion technical report.},
}
```

If you use KakeyaLattice in academic work, please also cite the theory
lineage we build on (Zamir & Feder 1996; Conway & Sloane 1999) — full
entries in [`ACKNOWLEDGMENTS.md`](../ACKNOWLEDGMENTS.md#theoretical-foundations).

## <a id="browser-demo"></a>Where can I try it without installing anything?

<https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress> — runs
Qwen3-0.6B side-by-side with bf16, E8 Q=10, Q=38, and Q=152 on a free
CPU tier (each "Run comparison" click takes ~4–8 minutes on 2 cores).
