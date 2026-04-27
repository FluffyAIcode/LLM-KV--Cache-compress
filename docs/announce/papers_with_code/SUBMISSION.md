# Papers with Code submission

Papers with Code (PwC) is the canonical "benchmarks-by-method" index
for ML and gets crawled daily by Google Scholar, Semantic Scholar,
Connected Papers, and AI answer engines. Entries under
<https://paperswithcode.com/task/kv-cache-compression> are the
first result on queries like "KV cache compression benchmark" on all
four of those retrievers.

## Prerequisites

- arXiv ID minted (see [`../arxiv/SUBMISSION.md`](../arxiv/SUBMISSION.md)).
  **PwC requires an arXiv link** for paper submissions. The benchmark
  submission below is possible without arXiv but works better with it.
- GitHub repo publicly visible (it is).
- PyPI package published (it is, v1.5.0).

## What to submit

Two entries, filed separately:

1. **Paper submission** — creates the canonical PwC page for
   KakeyaLattice.
2. **Benchmark submission** — four benchmark rows on the
   KV-cache-compression task, one per model we measure.

## 1 — Paper submission

Form: <https://paperswithcode.com/paper/submit>

### Metadata to paste

- **Title**: `KakeyaLattice: Nested-Lattice KV-Cache Compression for Large Language Models`
- **Abstract**: paste from the arXiv abstract.
- **arXiv URL**: `https://arxiv.org/abs/<ID>` (fill in once minted).
- **PDF URL**: `https://arxiv.org/pdf/<ID>.pdf` (auto-populated from
  arXiv URL in most cases).
- **Tasks**: add `KV Cache Compression`, `Language Modelling`,
  `Quantization`.
- **Methods**: add `Nested-Lattice Quantization`, `Sylvester-Hadamard
  Rotation`, `E8 Lattice`, `D4 Lattice`. PwC will show a "new method"
  prompt — accept and fill in the short method description below.
- **Source code**:
  `https://github.com/FluffyAIcode/LLM-KV--Cache-compress`
- **Framework**: `PyTorch`, `Hugging Face transformers`, `vLLM`.

### Method description (to paste into the "New method" dialog)

```
KakeyaLattice is a nested-lattice quantiser for the KV cache of
transformer language models. Each K or V vector is rotated by a
Sylvester-Hadamard matrix H/sqrt(D), scaled adaptively by its L2
norm, and snapped to the closest point of a nested D4 (dim 4) or E8
(dim 8) lattice using Conway-Sloane closest-point decoders. The
rotation gaussianises the heavy-tailed, non-isotropic KV activations
real LLMs produce; the lattice snap then exploits the densest known
sphere packings in dimensions 4 and 8 to beat any per-channel scalar
quantiser at the same bit budget. The codec is stateless per-vector,
so it supports streaming / online decode without calibration or
warm-up.
```

## 2 — Benchmark submissions

Four rows under
<https://paperswithcode.com/task/kv-cache-compression>. One row per
model we measure, each citing the same arXiv paper.

### Row template

The PwC benchmark form asks for: dataset, metric, value, extra
info, model name, link to paper. Fill in per the table below; the
"extra info" slot is where we disclose the quality target and CI
protocol.

| model                          | dataset       | metric                        | value   | extra info                                       |
|:-------------------------------|:--------------|:------------------------------|:--------|:-------------------------------------------------|
| Qwen3-4B                       | WikiText-103  | KV compression ratio @ ≤2% \|Δppl\| | **2.77×** | 128k ctx, n=8 passages × 64 eval pos, H200, vLLM bf16 |
| GLM-4-9B-Chat                  | WikiText-103  | KV compression ratio @ ≤2% \|Δppl\| | **2.44×** | 128k ctx, n=8 passages × 64 eval pos, H200, vLLM bf16 |
| Gemma-4-E4B                    | WikiText-103  | KV compression ratio @ ≤2% \|Δppl\| | **3.04×** | 128k ctx, n=8 passages × 64 eval pos, H200, vLLM bf16 (tied with TurboQuant at saturation) |
| DeepSeek-R1-Distill-Qwen-1.5B  | WikiText-103  | KV compression ratio @ ≤2% \|Δppl\| | **2.43×** | 128k ctx, n=8 passages × 64 eval pos, H200, vLLM bf16 |

Numbers taken directly from
`reports/v1_4_release/kv_128k_isoppl_n8/V14_VS_TQ_ISOPPL_REPORT.md`.
Reproducible via `benchmarks/extract_iso_ppl_table.py` — the PR body
can link to the reproducer so PwC reviewers can check.

### DeepSeek-V4-Flash (separate task entry)

Also file a row under
<https://paperswithcode.com/task/model-compression> (or a new custom
task if V4-Flash is not already listed):

| model              | dataset         | metric                              | value      | extra info                                         |
|:-------------------|:----------------|:------------------------------------|:-----------|:---------------------------------------------------|
| DeepSeek-V4-Flash  | WikiText-style  | KV bit reduction vs FP8 @ matched quality | **−22.0 %** | n=8 H200, 3/43 SWA + 20/43 c4a + 20/43 c128a layers, layer-weighted rel-MSE 0.959 ± 0.024 (95 % CI) vs hardware FP8 per-64-block |

Cites `reports/v1_5_release/dsv4_stage075/FINDINGS_N8.md`.

## After submission

1. The PwC entry lives at `https://paperswithcode.com/paper/kakeyalattice`.
2. Add the PwC URL to `README.md` as an additional badge:
   ```markdown
   [![PapersWithCode](https://img.shields.io/badge/Papers%20with%20Code-kakeyalattice-21caf5.svg)](https://paperswithcode.com/paper/kakeyalattice)
   ```
3. Add the PwC URL to `CITATION.cff` under `identifiers`.

## Why PwC matters for GEO

PwC ranks disproportionately well on **benchmark-comparison queries**,
which is what procurement-stage decision-makers actually search for.
A query like `"KV cache compression benchmark 2026"` returns the PwC
leaderboard first; having two rows there named KakeyaLattice puts us
in front of every reader of that page. The NexusQuant precedent
confirms this: their PwC page has been cited in three independent
papers since landing, entirely through organic discovery.
