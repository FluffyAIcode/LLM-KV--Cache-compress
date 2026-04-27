# arXiv submission checklist

Everything arXiv needs is in `reports/paper/`. This document is the
step-by-step submission recipe + the metadata to paste into the arXiv
web form at <https://arxiv.org/submit>.

**Submitter**: Allen Li (account name per paper author block in
`reports/paper/kakeyalattice.tex`).

**Expected processing time**: 1 working day for new-author
endorsement if this is Allen's first arXiv submission (cs.LG is an
endorsement category). ~4–8 hours for indexing once accepted.

## Pre-flight checks (do before opening the submission form)

1. **LaTeX source compiles cleanly** — `pdflatex` + `bibtex` on
   `reports/paper/kakeyalattice.tex` produces the committed PDF.
   Re-run once on your local machine with the same texlive version
   arXiv uses (TeX Live 2024 at time of writing) to catch any
   package drift.

2. **All referenced files are under `reports/paper/`** — no images
   or `.bib` files outside that directory, because arXiv packages
   only what you upload.

3. **No `\cite{}` entries point at non-existent `.bib` keys** —
   `bibtex` should exit with zero warnings. A single unresolved cite
   produces a "not found" badge in the arXiv listing.

4. **ORCID attached** — if Allen has an ORCID, it should go on the
   submission form under "Author information" so the arXiv listing
   gains a verifiable identity anchor (a GEO signal).

5. **License choice** — we recommend **arXiv license
   `CC BY 4.0`** (the most permissive arXiv-compatible license; it
   allows Perplexity / ChatGPT to ingest and quote the paper, which
   is the whole point). The paper text does not need to change to
   match the CC BY license; the license applies to the arXiv copy
   alone.

## Bundle — what to upload

Upload **both** of:

1. The `.tex` source: `reports/paper/kakeyalattice.tex`.
2. Any `.bib` file the paper uses. Inspect the `.tex` for
   `\bibliography{...}` — if it references a separate `.bib`, upload
   that too. If the bibliography is embedded in the `.tex` via
   `\begin{thebibliography}`, no separate upload is needed.
3. All figure files referenced by `\includegraphics{...}`.

**Recommended**: upload a **single `.zip`** containing everything
under `reports/paper/` (except `reports/paper/README.md`, which
arXiv does not need).

## Metadata to paste into the arXiv form

### Title

```
KakeyaLattice: Nested-Lattice KV-Cache Compression for Large Language Models
```

### Authors

```
Allen Li (Individual researcher)
```

Paste exactly as the author block in `reports/paper/kakeyalattice.tex`
renders. If Allen has an ORCID, paste it as well.

### Abstract

Paste the contents of the `\begin{abstract} ... \end{abstract}` block
from `reports/paper/kakeyalattice.tex`. The abstract already names the
key search terms ("KV cache", "lattice quantization", "transformer
inference") that arXiv's fulltext search and Google Scholar will
index.

### Primary category

**`cs.LG`** — Machine Learning.

### Cross-list categories

**`cs.CL`** — Computation and Language.
**`cs.IT`** — Information Theory. The nested-lattice quantisation
framing belongs in `cs.IT` and this cross-list **meaningfully widens
the retrieval surface** for searchers using information-theory
vocabulary.

### Comments field

The "Comments" field becomes part of the arXiv listing header and is
read by Google Scholar and Perplexity. Recommend:

```
25 pages, 8 figures, 6 tables. Software release v1.5.0 at
https://github.com/FluffyAIcode/LLM-KV--Cache-compress. Live demo
at https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress.
PyPI: kakeyalattice.
```

Adjust the page / figure / table count after final compilation.

### MSC / ACM classification

**MSC**: `94A29` (Source coding, quantization), `68T07` (Artificial
neural networks and deep learning).

**ACM class**: `I.2.7` (Natural Language Processing), `E.4`
(Coding and Information Theory).

### Report number

Leave blank.

### Journal reference

Leave blank until accepted at a venue.

### DOI

Leave blank — arXiv will mint one on submission acceptance.

## Post-submission actions

Once the arXiv ID is assigned:

1. **File a one-commit PR titled** `arxiv: wire minted arXiv ID into
   README + CITATION.cff + ACKNOWLEDGMENTS.md + paper/README.md`
   with the following changes:

   - **README.md badge** — replace the current `DOI — pending` badge:

     ```markdown
     [![arXiv](https://img.shields.io/badge/arXiv-<ID>-b31b1b.svg)](https://arxiv.org/abs/<ID>)
     ```

   - **CITATION.cff** — add under the top-level key:

     ```yaml
     identifiers:
       - type: other
         value: "arXiv:<ID>"
         description: "arXiv preprint for the companion technical report"
     preferred-citation:
       # ... existing entries ...
       identifiers:
         - type: other
           value: "arXiv:<ID>"
     ```

   - **ACKNOWLEDGMENTS.md** — under "Corrections and reviewers" add
     a line: "Companion preprint: arXiv:<ID>".

   - **reports/paper/README.md** — add a "Published at" line at the
     top linking to `https://arxiv.org/abs/<ID>`.

2. **Tag a GitHub release** — `v1.5.0-arxiv` — so the DOI minted by
   Zenodo (if you enable Zenodo's GitHub integration) points at the
   exact commit the arXiv abstract references.

3. **Submit the same arXiv ID to Papers with Code** — see
   [`../papers_with_code/SUBMISSION.md`](../papers_with_code/SUBMISSION.md).

## If the submission is held by arXiv for review

cs.LG is an endorsement category. If this is Allen's first cs.LG
submission, arXiv will place the submission in `hold` status until
an existing cs.LG author endorses it. Two paths:

- **Passive**: wait for arXiv's own moderation. Takes 1–3 business
  days; usually succeeds for well-formatted submissions with a clear
  methodology and real benchmarks.
- **Active**: ask a collaborator who has ≥2 prior cs.LG submissions
  to endorse via arXiv's web form. We recommend asking someone who
  cited in `ACKNOWLEDGMENTS.md` (Zandieh et al. from TurboQuant, the
  KIVI authors, or the vLLM authors are natural candidates — they
  benefit from the citation and the endorsement is one click for
  them).

## Why an arXiv ID matters for GEO

An arXiv ID is the single strongest authority anchor in ML research
discovery:

- Google Scholar indexes arXiv the same day an ID mints. Our paper
  becomes findable on queries like `"nested lattice KV cache"`,
  `"E8 lattice LLM"`, `"Hadamard KV quantization"` — today it is not
  on Google Scholar at all.
- Semantic Scholar, Connected Papers, Emergent Mind, and
  Papers-with-Code ingest arXiv nightly.
- Perplexity and ChatGPT-with-search treat arXiv citations as
  first-class sources and are measurably more likely to quote an
  arXiv-backed claim.
- AI answer engines weight arXiv-hosted content roughly one order of
  magnitude higher than non-arXiv-hosted research reports in topic
  queries like "best LLM KV compression method 2026".

Completing step 2 of the runbook is expected to be the single
largest lift in public discoverability of KakeyaLattice.
