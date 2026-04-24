# arXiv submission package — KakeyaLattice v1.4

**Title**: KakeyaLattice: D4 Nested-Lattice KV-Cache Compression with Kakeya-Style Discrete Codebooks

**Author**: Allen Li (individual researcher, AllenL329@gmail.com)

**Date**: April 23, 2026

**arXiv category suggestion**: `cs.LG` (primary), `cs.CL`, `cs.IT`, `cs.DS` (secondary)

## What this paper is

A complete rewrite of the v1.3 Kakeya paper
(`reports/v1_3_rsvd_rope/paper/kakeyaturbo.tex`) based on the
**v1.4 KakeyaLattice** release. The v1.3 paper was a low-rank
PCA + Lloyd-Max codec framed against a conditional Kakeya lower
bound; this paper is a $D_4$ nested-lattice codec framed against
an unconditional lattice-density bound, with head-to-head wins
on real vLLM across four open-source model families.

## What changed from v1.3 paper

| Axis | v1.3 paper | v1.4 paper (this) |
|:---|:---|:---|
| Codec | PCA + RSVD + WHT + Lloyd-Max + inverse-RoPE | **D4 nested lattice + TQ engineering stack** |
| Baseline | TurboQuant Python reference impl (unfair) | **TurboQuant vLLM merged (PR #38479)** |
| Measurement | CPU-only, no live inference | **Live vLLM on H200, strict-GPU capture + replace** |
| Models | 7 models, MSE only | 4 models, MSE + $\|\Delta$ppl$\|$ + top-1 + 128k KV |
| Lower bound | Conditional on Kakeya conjecture $D\geq 4$ | Unconditional via $D_4$ lattice density |
| Headline result | $5.4$–$6.7\times$ CR at ACCEPT MSE | **12/12 K-MSE wins over TQ, 9/12 $\|\Delta$ppl$\|$ wins** |

## Deployment positioning

The paper's deliberate positioning is as an **~8% improvement to
the upstream TurboQuant kernel**, not a standalone codec. The $D_4$
shaping gain is $+0.37\,$dB ($\times 0.92$ K-MSE) over $\Z^4$; the
engineering stack contributes ~1000× K-MSE on top. v1.4 slots into
TurboQuant's Hadamard + qmax wrapper as a drop-in replacement for
the int·per-coord scalar quantiser. Proposed upstream PR:
`turboquant_d4v4` preset beside `turboquant_k8v4`.

## Files

| File | Size | Purpose |
|:---|---:|:---|
| `kakeyalattice.tex` | ~60 KB | single-file LaTeX source (arXiv-compatible) |
| `kakeyalattice.pdf` | — | compile locally via `pdflatex` ×3 |

## Compilation

Standard arXiv-supported packages only (`amsmath`, `graphicx`,
`hyperref`, `booktabs`, `algorithm`, `algpseudocode`, `enumitem`,
`multirow`). No `bibtex` step (`thebibliography` embedded):

```bash
pdflatex kakeyalattice.tex
pdflatex kakeyalattice.tex    # cross-references
pdflatex kakeyalattice.tex    # TOC + bibliography
```

## Data sources cited

Every numerical claim in the paper is traceable to a committed
report file on the `main` branch at tag `v1.4`:

- **Multi-model head-to-head**:
  `reports/v1_3_ppl/snapshot_mode_qwen3/multimodel/FINDINGS_MULTIMODEL.md`
  (+ `*_multimodel.json`, `*.log` per model)
- **Full Qwen3-4B Pareto sweep**:
  `reports/v1_3_ppl/snapshot_mode_qwen3/bridges_abc/qwen3_4b_compression_sweep_gpu.json`
- **Three-bridge research trail**:
  `reports/v1_3_ppl/snapshot_mode_qwen3/SESSION_KAKEYA_RESEARCH.md`
- **Non-Gaussianity measurements**:
  `reports/v1_3_ppl/snapshot_mode_qwen3/non_gaussianity/qwen3_4b_k_non_gaussianity.json`
- **128k KV storage**:
  `reports/v1_4_release/kv_128k_report/V14_KV_128K_REPORT.md`
  (+ `*_kv_128k.json` per model)
- **Codec implementation**:
  `kakeyaturbo-py/python/kakeyaturbo_py/v1_4_kakeya_zamir_lattice_gpu.py`
  (canonical class) delegates to
  `kakeyaturbo-py/python/kakeyaturbo_py/lattice_codebooks.py::D4LatticeCodebook`
  (implementation core).  In the v1.4 release tag this was
  `bridge_b2_d4_tq_style.py::D4TQStyleCodebook`; the two produce
  bit-identical output (verified by `benchmarks/e8_parity_and_smoke.py`
  against a frozen sha256 snapshot).

## Reproducibility

```bash
cd /workspace/LLM-KV--Cache-compress
git checkout v1.4
export VLLM_ENABLE_V1_MULTIPROCESSING=0 KAKEYA_SNAPSHOT_QWEN3=1
python benchmarks/multimodel_v14_vs_tq.py \
    --model-path Qwen/Qwen3-4B --model-name qwen3_4b \
    --ctx-len 2048 --n-eval 64 --n-passages 4 \
    --out-dir reports/v1_3_ppl/snapshot_mode_qwen3/multimodel
```

Replace `--model-path` / `--model-name` for the other three models.
Add `--trust-remote-code` for GLM-4-9B-Chat. Requires H200-class
GPU, vLLM 0.19.2+, transformers 5.5.2+, CUDA 12.x/13.x.

## Known remaining items

- Small cosmetic overfull hboxes on long URLs and the single-word
  class names in `\texttt{}` blocks. Not an arXiv-acceptance issue.
- No `*.pdf` committed in this directory; compile locally
  (self-contained, no external assets).
