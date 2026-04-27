# Discovery runbook — making KakeyaLattice findable

Single runbook for getting KakeyaLattice cited by search engines,
AI answer engines (ChatGPT / Perplexity / Claude), and the Python /
LLM-inference developer community. Each section is self-contained:
owner, inputs, steps, done-when.

Reference reading that motivates this runbook: the NexusQuant launch
strategy (three DEV.to posts + arXiv + HF Space + vLLM discussion
+ Papers with Code + GitHub topics) currently dominates the "E8 KV
compression" / "lattice KV quantisation" query space on Google,
Perplexity, and ChatGPT with search. We are following the same
template with the same number of hits but with our own positioning.

## Priority and timing

Run in this order. Each step unlocks retrieval signal for the next.

| # | step | owner | difficulty | search-engine payoff |
|---|:-----|:------|:-----------|:---------------------|
| 1 | GitHub topics | repo admin | trivial | immediate — GitHub topic pages are indexed within hours |
| 2 | arXiv submission | paper author (Allen Li) | 1–2 steps | highest single-action payoff — an arXiv ID is the strongest authority anchor in ML |
| 3 | vLLM integration issue | repo author + vLLM community | discussion-thread | high — vLLM issues are crawled aggressively by AI engines |
| 4 | HF Space back-link + Qwen3/Llama-3 model-card link-backs | repo author | trivial | medium — HF pages are high-authority but many indexed already |
| 5 | Papers with Code entry | paper author | form-fill | medium — PwC is the canonical "benchmarks by method" index |
| 6 | DEV.to posts × 2 | repo author | writing | high — DEV.to posts rank unusually well on specialist search queries |

Tasks 1, 4, 6 are low-effort and should land first. Tasks 2, 3, 5
are higher-friction but have the best long-term SEO / GEO return.

## 1 — GitHub topics

**Owner**: whoever has write access to the repo on
`github.com/FluffyAIcode/LLM-KV--Cache-compress`.

**Topics to set** (8 terms, derived from actual query patterns
observed in the NexusQuant launch and from our own
`docs/faq.md` question set):

```
kv-cache
kv-cache-compression
quantization
vllm
lattice-quantization
llm-inference
long-context
e8-lattice
```

Additional topics to consider adding after the first six land (not
required for initial indexing, but they widen the retrieval surface):

```
d4-lattice
transformers
huggingface
deepseek-v4
qwen3
flashattention
pytorch
arxiv
```

### One-click script (requires `gh` CLI with a repo-write token)

The cloud-agent `gh` CLI is read-only. Run this locally as the repo
admin:

```bash
gh repo edit FluffyAIcode/LLM-KV--Cache-compress \
  --add-topic kv-cache \
  --add-topic kv-cache-compression \
  --add-topic quantization \
  --add-topic vllm \
  --add-topic lattice-quantization \
  --add-topic llm-inference \
  --add-topic long-context \
  --add-topic e8-lattice
```

### UI alternative

1. Open <https://github.com/FluffyAIcode/LLM-KV--Cache-compress>.
2. Click the ⚙ icon next to "About" in the repo sidebar.
3. In the "Topics" field, paste the eight topics above, one at a time
   (GitHub autocompletes existing topics, which is what we want — hitting
   an existing high-traffic topic adds us to that topic's discovery page).
4. Save.

**Done when**: the repo's About sidebar shows all eight topics AND
`https://github.com/topics/e8-lattice` lists this repo in the sort
(may take 1-6 hours for GitHub to index).

## 2 — arXiv submission

**Owner**: Allen Li (paper author per `reports/paper/kakeyalattice.tex`).

**Inputs**: `reports/paper/kakeyalattice.tex`,
`reports/paper/kakeyalattice.pdf`, the figures referenced in the `.tex`
(already committed in the paper directory).

### Submission bundle

Everything arXiv needs is already in `reports/paper/`. See
[`docs/announce/arxiv/SUBMISSION.md`](arxiv/SUBMISSION.md) for the
checklist (metadata, categories, abstract, comments field, license).

### Target categories

- Primary: **`cs.LG`** (Machine Learning)
- Cross-list: **`cs.CL`** (Computation and Language)
- Optional cross-list: **`cs.IT`** (Information Theory) — the
  nested-lattice quantisation framing sits naturally in `cs.IT`
  and this cross-list significantly widens the retrieval surface
  for "lattice quantization" searchers.

### After the arXiv ID mints

Open a one-commit PR titled `arxiv: replace 'DOI — pending' badges
with the minted arXiv ID` that:

- Replaces the `DOI — pending` badge in `README.md` with
  `[![arXiv](https://img.shields.io/badge/arXiv-<ID>-b31b1b.svg)](https://arxiv.org/abs/<ID>)`.
- Adds the arXiv URL to `CITATION.cff` as an `identifiers` entry.
- Adds the arXiv URL to the `ACKNOWLEDGMENTS.md` infrastructure
  section.
- Updates `reports/paper/README.md` to point at the public arXiv page.

A follow-up agent session can run this PR once you paste the arXiv ID
into a new message.

**Done when**: the paper is listed at `https://arxiv.org/abs/<ID>`
AND the README badge is live.

## 3 — vLLM integration issue / discussion

**Owner**: whoever files issues under the FluffyAIcode identity.

**Target repo**: `vllm-project/vllm`.

**Pre-written body + title + labels**: see
[`docs/announce/vllm_integration_issue.md`](vllm_integration_issue.md).

The NexusQuant analogue is vLLM issue #16047 (filed 2025-02, ongoing
discussion). We aim for the same class of reception: a maintainer
engages in the thread, we end up with an "official" integration path
even if the implementation is gated on a follow-up PR.

**Done when**: issue is filed with the labels `new-feature-proposal`
and `kv-cache`, and has at least one maintainer acknowledgement.

## 4 — HF back-links

**Owner**: repo author (has HF token).

### HF Space README (already live)

Confirmed on 2026-04-25: the Space
`huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress` has a
"Links" section that points back to the GitHub repo, the PyPI
package, and the paper directory.

If you want to tighten this further, see
[`docs/announce/hf_space_backlinks.md`](hf_space_backlinks.md) for
two safe incremental edits (add arXiv badge once minted; pin the
Space via Collections).

### Model-card link-backs

Opening PRs on the Qwen3 / Llama-3 model cards to add
KakeyaLattice to a "Related projects" section is a legitimate
discovery tactic but the success rate depends on the model author's
review latency. See
[`docs/announce/model_card_backlinks.md`](model_card_backlinks.md)
for per-model PR drafts.

**Done when**: at least two high-traffic HF model cards carry a
KakeyaLattice "Related projects" entry.

## 5 — Papers with Code

**Owner**: paper author.

**Target**: create entries under
<https://paperswithcode.com/task/kv-cache-compression> (the task page
already exists from KIVI / TurboQuant / H2O entries). We add two
benchmark rows:

1. `KakeyaLattice (D4)` — iso-PPL @ 128 k, WikiText-103, n=8
2. `KakeyaLattice (E8)` — iso-PPL @ 128 k, WikiText-103, n=8

**Pre-filled submission**: see
[`docs/announce/papers_with_code/SUBMISSION.md`](papers_with_code/SUBMISSION.md).

**Done when**: the KakeyaLattice entry is live at
`https://paperswithcode.com/paper/kakeyalattice` AND the method is
listed under the KV-cache-compression task leaderboard with our
four-model numbers.

## 6 — DEV.to posts ×2

**Owner**: repo author (drafts written by the cloud agent; you
copy-paste + publish under your DEV.to identity).

Two posts, deliberately different in tone and target query:

- [`docs/announce/dev_to/post_1_theory.md`](dev_to/post_1_theory.md)
  — "E8-lattice KV cache compression, from first principles"
  (~1200 words). Targets searchers looking for *why* E8 beats scalar
  KV quantisation. Ranks on queries like "nested lattice vs scalar
  quantisation", "E8 lattice KV compression", "Hadamard rotation
  for LLM activations".

- [`docs/announce/dev_to/post_2_practice.md`](dev_to/post_2_practice.md)
  — "Qwen3 KV cache in 10 lines of Python" (~1000 words). Targets
  searchers looking for *how to use*. Ranks on queries like
  "transformers DynamicCache compression", "compress Qwen3 KV cache",
  "KakeyaLattice tutorial".

### DEV.to front-matter

Both posts include the DEV.to-specific front-matter block (title,
published, tags, cover_image, canonical_url). The `canonical_url`
field is **important**: it points to the GitHub blog path so
DEV.to's SEO juice credits our repo as the source-of-truth, not
DEV.to itself.

**Done when**: both posts live at dev.to/<your-handle>/<slug>,
both show ≥3 tags, and both have the canonical_url back to the
repo's blog directory.

## Tracking table

After each step lands, update this table. PRs on top of this file
are welcome.

| step | done? | date | notes |
|:-----|:------|:-----|:------|
| 1. GitHub topics            | ☐ | | |
| 2. arXiv submission         | ☐ | | |
| 3. vLLM issue               | ☐ | | |
| 4. HF Space back-links      | ☑ | 2026-04-25 | SPACE_README.md has Links section pointing back; comparison paragraph in place |
| 4. Model-card back-links    | ☐ | | |
| 5. Papers with Code         | ☐ | | |
| 6a. DEV.to post 1 (theory)  | ☐ | | |
| 6b. DEV.to post 2 (practice)| ☐ | | |
