# arXiv / preprint package — KakeyaLattice (v1.4 + v1.5)

**Title**: KakeyaLattice: $D_4$ / $E_8$ Nested-Lattice KV-Cache Compression
with Kakeya-Style Discrete Codebooks — Extended: v1.4 ($D_4$) + v1.5 ($E_8$)
Joint Release

**Author**: Allen Li (individual researcher, AllenL329@gmail.com)

**Date**: April 24, 2026

**arXiv category suggestion**: `cs.LG` (primary), `cs.CL`, `cs.IT`, `cs.DS` (secondary)

## Scope

This paper covers **two KakeyaLattice releases** jointly:

- **v1.4** (released 2026-04-23, tag [`v1.4`](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/releases/tag/v1.4)): the baseline $D_4$ nested-lattice codec plus the four engineering levers (unit-norm factorisation, Sylvester-Hadamard rotation, per-vector adaptive $q_\mathrm{max}$, and inverse-of-the-above at decode). The $D_4$ shaping gain $+0.37\,$dB is the structural lever on top of $\Z^4$ scalar quantisation.
- **v1.5** (released 2026-04-24, tag [`v1.5`](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/releases/tag/v1.5)): substitutes $E_8$ for $D_4$ as the shaping lattice, keeping every other pipeline step unchanged. Adds a measured $+1.3$ to $+2.0\,$dB per-layer $K$-MSE gain on four open-source models ($4$–$6\times$ the theoretical $+0.29\,$dB minimum) and $28$–$53\%$ $|\Delta\mathrm{ppl}|$ reduction in real vLLM in-forward no-boundary at $n=32$.

## What changed between v1.4 and v1.5

| Axis | v1.4 | **v1.5** |
|:---|:---|:---|
| Shaping lattice | $D_4$ (Conway-Sloane 1982 Alg. 4) | $E_8$ (Conway-Sloane 1982 Alg. 5) |
| Block dim | $4$ | $8$ |
| $G(\Lambda)$ | $0.0766$ | $0.0717$ |
| Shaping gain vs $\Z^n$ | $+0.37\,$dB | $+0.66\,$dB (**$+0.29\,$dB over v1.4**) |
| Measured per-layer $K$-MSE gain (4-model avg) | — | **$+1.8\,$dB** (6× theory, due to two-coset outlier handling) |
| $|\Delta\mathrm{ppl}|$ reduction vs v1.4 | — | **$-31.5\%$ at Q=10, $-53.4\%$ at Q=4** (Qwen3-4B) |
| Encode/decode latency | $330$--$354\,\mu$s | $551$--$613\,\mu$s ($1.56$--$1.80\times$) |
| Bits per vector at Q=10 (head\_dim=128) | $576$ | $608$ ($+32$ overhead) |
| Compression-ratio cost at iso-Q | baseline | $-5$ to $-7\%$ |
| NIAH retrieval (Qwen3, Gemma) | 100% | **100%** |
| NIAH retrieval (GLM-4-9B at Q=10) | 100% | $89\%$ (1 cell of 27 degrades) |

## Five measurement axes covered

1. **PPL** (`$|\Delta\mathrm{ppl}|$`) at $n=32$ passages with $95\%$ CI (Student-$t$), live vLLM in-forward
2. **MSE** (per-layer $K$/$V$ rel-MSE, matches Gaussian theory prediction)
3. **CR** (compression ratio at $128$k-token KV cache)
4. **Latency** (pure codec wall time on H200, $500$ iters, p50/p99)
5. **NIAH** (Needle-in-a-Haystack retrieval at ctx $\in\{4,8,16\}$k $\times$ depth $\in\{0.1,0.5,0.9\}$, $n_\mathrm{trials}=3$)

All axes measured across four open-source model families: Qwen/Qwen3-4B, deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, google/gemma-4-E4B, zai-org/GLM-4-9B-Chat.

## Directory layout

```
reports/
├── paper/                          ← this directory (paper source)
│   ├── kakeyalattice.tex           ← LaTeX source (arXiv-ready)
│   ├── README.md                   ← this file
│   └── (PDF regeneratable via `latexmk -pdf kakeyalattice.tex`)
├── v1_4_release/                   ← frozen v1.4 evidence (tag v1.4)
└── v1_5_release/                   ← v1.5 evidence (tag v1.5)
```

The PDF is no longer committed; regenerate from `.tex` for the current version. The paper references `reports/v1_4_release/` and `reports/v1_5_release/` directly for raw data tables.

## Build instructions

```bash
cd reports/paper
latexmk -pdf kakeyalattice.tex
# or:
pdflatex kakeyalattice.tex && pdflatex kakeyalattice.tex
```

Requires: standard LaTeX distribution with `amsmath`, `amssymb`, `booktabs`,
`multirow`, `xcolor`, `hyperref`, `algorithm`, `algpseudocode`,
`enumitem`, `caption`.

## Compliance statement in paper

All numbers: live vLLM `0.19.2rc1.dev100+gf946659ff`, strict-GPU on NVIDIA H200, no mock / no simplification / no fallback / no overfit / no deferred. Bit-level regression protection via `benchmarks/e8_parity_and_smoke.py` (8 `sha256` hashes pinned in `benchmarks/frozen_parity.json`).

## Citation

If you use this work, please cite:

```bibtex
@misc{kakeyalattice2026,
  author       = {Li, Allen},
  title        = {KakeyaLattice: $D_4$ / $E_8$ Nested-Lattice KV-Cache
                  Compression with Kakeya-Style Discrete Codebooks
                  (v1.4 + v1.5 Joint Release)},
  year         = {2026},
  howpublished = {\url{https://github.com/FluffyAIcode/LLM-KV--Cache-compress}},
  note         = {Tags v1.4 (2026-04-23) and v1.5 (2026-04-24)}
}
```
