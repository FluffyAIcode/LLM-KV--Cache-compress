# arXiv submission package

**Title**: Kakeya Skeleton with Residual Turbo Compression: A Rate--Distortion Framework for LLM KV Cache Compression

**Authors**: KakeyaTurbo Contributors

**Date**: April 19, 2026

**arXiv category suggestion**: `cs.LG` (primary), `cs.CL`, `cs.DS` (secondary)

## Files

| File | Size | Purpose |
|---|---:|---|
| `kakeyaturbo.tex` | ~64 KB | single-file LaTeX source (arXiv-compatible, no custom style files) |
| `kakeyaturbo.pdf` | ~399 KB | compiled 18-page PDF for preview |

## Reproducibility

This paper presents benchmark results measured on the `cursor/v1-3-rsvd-rope-aware-12f5` branch of the `FluffyAIcode/LLM-KV--Cache-compress` repository. All code, tests, and raw measurement JSON for every table in the paper are released there under Apache-2.0.

Specifically:

- The v1.3 codec implementation: `kakeyaturbo/src/` (Rust, 153 passing tests)
- The real-data benchmark driver: `benchmarks/kakeyaturbo_v1_2_real_bench.py`
- The inverse-RoPE POC: `benchmarks/rope_aware_k_poc.py`
- Per-model per-layer JSON reports: `reports/v1_3_rsvd_rope/{bench,rope_poc}/`
- The release decision document: `reports/v1_3_rsvd_rope/DECISION.md`
- The flagship projection methodology: `reports/v1_3_rsvd_rope/FLAGSHIP_COMPARISON.md`
- The SmolLM2 capability note: `reports/v1_3_rsvd_rope/SMOLLM2_CAPABILITY.md`
- The v1.4 roadmap: `reports/v1_3_rsvd_rope/V1_4_V1_5_ROADMAP.md`

## Compilation

The paper uses only standard arXiv-supported LaTeX packages (`amsmath`, `graphicx`, `hyperref`, `booktabs`, `algorithm`, `algpseudocode`). Compile with:

```bash
pdflatex kakeyaturbo.tex
pdflatex kakeyaturbo.tex    # second pass for cross-references
pdflatex kakeyaturbo.tex    # third pass for bibliography + TOC
```

No `bibtex` step is required — the bibliography is embedded in `thebibliography`.

## Known remaining overfull boxes

The compiled PDF has a few small overfull-hbox warnings (2–33pt) on long English words and URLs. These are cosmetic and do not affect arXiv acceptance. The paper compiles cleanly with zero errors.
