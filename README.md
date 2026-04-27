# KakeyaLattice — Discrete Kakeya Cover for LLM KV-Cache Compression

> **A D4 / E8 nested-lattice codec that realises a discrete *Kakeya
> cover* over the direction sphere of transformer KV activations.
> 2.4×–2.8× compression at <1 % perplexity loss on Qwen3, Llama-3,
> DeepSeek, GLM-4, and Gemma — real vLLM prefill on NVIDIA H200.
> Drop-in `transformers.DynamicCache` subclass.**
> `pip install kakeyalattice`.

[![PyPI](https://img.shields.io/pypi/v/kakeyalattice.svg)](https://pypi.org/project/kakeyalattice/)
[![Stars](https://img.shields.io/github/stars/FluffyAIcode/LLM-KV--Cache-compress?style=social)](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/stargazers)
[![HF Space](https://img.shields.io/badge/🤗%20Space-Live%20demo-blue)](https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress)

⭐ **If this helps your inference stack, please give us a star on GitHub
above — it's the single fastest way to help others find this work.**

A GPU-native lattice-quantisation codec for transformer KV caches.
Measured across 4 open-source model families (Qwen3, DeepSeek, Gemma 4,
GLM-4) in real vLLM on an NVIDIA H200: **wins on K-MSE, V-MSE, and
|Δppl| against TurboQuant at matched bit budgets across 12 / 12
pairings**, with **+3 % to +38 % compression-ratio advantage** at
deployment-relevant quality thresholds (|Δppl| ≤ 2 %).

[Release notes, full comparison tables, and reproducibility commands
are on the v1.4 Release page.](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/releases/tag/v1.4)

## Why "Kakeya"? — what makes this codec different

Most KV-cache quantisers (TurboQuant, KIVI, SmoothQuant-KV, Quanto,
HQQ) allocate bits **per scalar channel**. KakeyaLattice allocates
bits **per direction on the sphere**. That difference is the whole
project, and it is why we carry the Kakeya name.

**The classical Kakeya problem** asks for the minimum measure of a
set in $\R^D$ that contains a unit segment in every direction.
Besicovitch (1919) showed this measure can be zero; Wolff, Tao,
Dvir, Wang–Zahl have since pushed the Kakeya maximal-function
conjecture deep into real analysis.

**The KV-cache analogue.** A codec that reconstructs every vector
$x_i$ to $\ell_2$-error $\varepsilon$ must cover a **tube of radius
$\varepsilon$** around every direction in
$\Theta = \{x_i / \|x_i\|\} \subset S^{D-1}$. The minimum bit-cost
of such a tube-union is exactly what the Kakeya maximal-function
conjecture bounds. KakeyaLattice's codebook **realises the discrete
version of that cover** explicitly: the tensor-product
$D_4^{\otimes D/4}$ (or $E_8^{\otimes D/8}$) Voronoi cells, scaled
adaptively per-vector by $q_{\mathrm{max}} / q_{\mathrm{range}}$,
tile $\R^D$ so that every direction in $\Theta$ is $\varepsilon$-covered
at a known bit budget.

Full derivation in
[`reports/paper/kakeyalattice.pdf`](reports/paper/kakeyalattice.pdf)
§1 "The codec as a discrete Kakeya cover" and §2
"Design philosophy: the Kakeya–Brascamp–Lieb–Tropp chain".

The practical consequences of this framing:

1. **Bit-per-direction, not bit-per-channel.** Rotating KV into the
   Hadamard basis and quantising in the D4/E8 Voronoi tessellation
   directly minimises the *tube-cover cost*, which is what matters
   for attention reconstruction — not per-channel dynamic range,
   which is what scalar quantisers minimise.
2. **Provable shaping-gain.** D4 gives +0.37 dB shaping gain over
   $\mathbb{Z}^4$; E8 gives +0.65 dB over $\mathbb{Z}^8$ and
   +0.29 dB over $D_4$. These are classical lattice-coding bounds
   (Zamir–Feder 1996), not empirical measurements.
3. **Unconditional bit-rate bound.** The Voronoi optimum holds at
   the block dimension without any assumption about the source
   distribution — which is why the codec still works on the
   heavy-tailed, non-Gaussian KV of DeepSeek-V4-Flash where
   scalar per-channel quantisers fail.

KakeyaLattice is the first open-source KV-cache codec to **name
itself after the geometric object it covers**. Everything else in
the stack — Sylvester–Hadamard rotation, per-vector $q_{\mathrm{max}}$,
the Conway–Sloane closest-point decoder — is standard lattice-coding
engineering in service of making the discrete Kakeya cover actually
run on a GPU.

## What's in the box

```
kakeyalattice/python/kakeyalattice/
  v1_4_kakeya_zamir_lattice_gpu.py   — canonical V14KakeyaZamirLatticeGPU class (D4)
  v1_5_kakeya_zamir_e8_gpu.py        — canonical V15KakeyaZamirE8GPU class (E8)
  lattice_codebooks.py               — shared Hadamard/qmax wrapper +
                                       D4LatticeCodebook + E8LatticeCodebook
                                       + Conway-Sloane closest-point algs
  spherical_codebooks.py             — codec interface
  __init__.py                        — re-exports V14 + V15

vllm_backend/kakeya_v1_4_snapshot/
  snapshot_hook.py                   — Attention monkey-patches (Qwen3 / Qwen2
                                       / Gemma4 / GLM) for post-QK/V-norm,
                                       pre-RoPE K/V capture + replace
  plugin.py                          — vLLM entry point (gated by env var)
  pyproject.toml                     — installs the plugin into vLLM workers

benchmarks/
  multimodel_v14_kv_128k_report.py   — per-model 128k KV storage report
                                       (v1.4 + TurboQuant comparison)
  multimodel_v14_vs_tq.py            — iso-bit head-to-head (K-only variant)
  v14_streaming_proof.py             — streaming / online proof
  v14_streaming_diag.py              — batch-vs-streaming root-cause diagnostic
  v14_streaming_latency.py           — per-decode-step latency

reports/paper/                       — joint paper for v1.4 + v1.5
  kakeyalattice.tex                  — LaTeX source (arXiv-ready)
  README.md                          — build instructions + scope

reports/v1_4_release/                — frozen v1.4 evaluation data
  kv_128k_report/                    — v1.4-only 128k KV storage tables
  kv_128k_report_tq_compare/         — iso-bit comparison vs TurboQuant
  kv_128k_isoppl_n8/                 — iso-PPL comparison vs TurboQuant
  streaming/                         — streaming / online capability report
  niah/                              — v1.4 NIAH retrieval
  rigorous_eval/                     — n=32 rigorous protocol + ablation
  audit/                             — GPU/vLLM audit trail
  INVENTORY.md + MANIFEST.sha256     — file inventory + integrity manifest

reports/v1_5_release/                — v1.5 evaluation data (E8 lattice)
  V15_FULL_4MODEL_REPORT.md          — primary report (4 models × 5 axes)
  V15_VS_V14_VS_TQ_REPORT.md         — Qwen3-4B first-measurement detail
  e8_latency_benchmark.{json,log}    — v1.5 vs v1.4 vs TQ pure codec latency
  {model}_{nobdry,tqb2_bdry2}_inforward.{json,log}  — 4 models × 2 boundary modes
  niah/                              — v1.5 NIAH retrieval + guardrail
  README.md                          — directory guide
```

## Quick start

Install the pure-Python codec + the vLLM snapshot plugin:

```bash
pip install -e kakeyalattice          # pure-Python, PyTorch-only
pip install -e vllm_backend             # installs the vllm.general_plugins entry point
```

Use the codec directly:

```python
import torch
from kakeyalattice import V14KakeyaZamirLatticeGPU

cb = V14KakeyaZamirLatticeGPU(D=128, q_range=38, device="cuda")
K = torch.randn(2048, 8, 128, device="cuda", dtype=torch.float32) * 0.3
K_hat = cb.roundtrip(K)       # encode + decode round-trip, bits known in advance
print(cb.bits_per_token_per_head)   # 832 bits for D=128, Q=38
```

Run the multi-model head-to-head benchmark (real vLLM prefill, strict GPU):

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0 KAKEYA_SNAPSHOT_QWEN3=1
python benchmarks/multimodel_v14_kv_128k_report.py \
    --model-path Qwen/Qwen3-4B --model-name qwen3_4b \
    --q-values 4,6,10,15,22,38,76,152 \
    --tq-b-values 3,4,5,6,7,8 \
    --ctx-len 2048 --n-eval 64 --n-passages 8 \
    --out-dir reports/v1_4_release/kv_128k_isoppl_n8
```

Add `--trust-remote-code` for GLM-4-9B-Chat.

## Key results (full tables in `reports/v1_4_release/`)

**iso-PPL compression advantage at |Δppl| ≤ 2 %** (dense Q/b sweep,
n=8 passages, 512 target tokens per channel):

| Model          | v1.4 CR | TQ CR | v1.4 advantage |
|:---------------|--------:|------:|---------------:|
| Qwen3-4B       | 2.77×   | 2.18× | **+26.9 %**    |
| GLM-4-9B-Chat  | 2.44×   | 1.77× | **+37.8 %**    |
| Gemma-4-E4B    | 3.04×   | 3.04× | tied (saturated) |
| DeepSeek-1.5B  | 2.43×   | 2.36× | **+3.3 %**     |

**iso-bit |Δppl| advantage at aggressive point** (v1.4 Q=10 vs TQ b=4,
n=4 passages, ~3.6-3.9× CR):

| Model          | v1.4 \|Δppl\| | TQ \|Δppl\| | v1.4 better by |
|:---------------|--------------:|------------:|---------------:|
| Qwen3-4B       | 1.45 %        | 6.58 %      | **4.5×**       |
| GLM-4-9B-Chat  | 6.52 %        | 10.74 %     | 1.6×           |
| Gemma-4-E4B    | 0.33 %        | 1.04 %      | **3.2×**       |
| DeepSeek-1.5B  | 2.22 %        | 3.47 %      | 1.6×           |

**Streaming latency** (per-decode-step, 1 new token × all layers × all
KV heads, batched): **~0.25 ms** across all 4 models × 3 operating
points.  At typical 15-30 ms bf16 decode step, codec overhead is
**< 2 %** of total decode latency.

## Streaming / online deployment

v1.4 has no cross-token state — the codec is a pure per-vector
function.  It supports streaming / online compression out of the box:
no calibration pass, no warmup, no buffering.  See
`reports/v1_4_release/streaming/V14_STREAMING_REPORT.md` for
measurements and integration notes.

## Compliance

All reported numbers are measured on real vLLM + real Hugging Face
weights + real WikiText-103 + real FlashAttention bf16 forward on
an NVIDIA H200.  No mocks, no simplifications, no fallbacks.

## License

Apache-2.0 (see `LICENSE`).
