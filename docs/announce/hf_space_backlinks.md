# HF Space + model-card back-links

HF pages are high-authority. Two kinds of back-links matter:

1. **Outbound** from the KakeyaLattice HF Space to the GitHub repo,
   the PyPI page, and (once minted) the arXiv abstract.
2. **Inbound** from high-traffic model cards (Qwen3 family, Llama-3
   family, DeepSeek-R1-Distill, GLM-4, Gemma-4) that mention the
   KakeyaLattice Space under a "Related projects" section.

## 1 — Space outbound links

### Status

The Space `huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress`
already carries outbound links via its `README.md` (sourced from
`demos/hf_llama_kakeyalattice/SPACE_README.md` in this repo):

- GitHub: <https://github.com/FluffyAIcode/LLM-KV--Cache-compress>
- PyPI: <https://pypi.org/project/kakeyalattice/>
- Paper directory: `reports/paper/` (inside the GitHub repo)
- Stage 0.75 DSv4 findings: `reports/v1_5_release/dsv4_stage075/FINDINGS.md`
  (inside the GitHub repo)

### Suggested tightening (to be applied once arXiv ID mints)

Replace the "Paper: `reports/paper/`" line with:

```markdown
- Paper: [arXiv:<ID>](https://arxiv.org/abs/<ID>) · [PDF in repo](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/reports/paper/kakeyalattice.pdf)
```

This adds the arXiv URL as a second authority anchor; Perplexity /
ChatGPT both follow arXiv URLs and boost content that includes one.

### Suggested tightening (now, independent of arXiv)

Add an arXiv-style badge block at the top of the Space README so a
human visitor sees the stack of attestations immediately:

```markdown
[![PyPI](https://img.shields.io/pypi/v/kakeyalattice.svg)](https://pypi.org/project/kakeyalattice/)
[![GitHub](https://img.shields.io/badge/GitHub-source-181717?logo=github)](https://github.com/FluffyAIcode/LLM-KV--Cache-compress)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/LICENSE)
```

This is a 3-line edit to `demos/hf_llama_kakeyalattice/SPACE_README.md`
plus a push to the Space via `huggingface_hub.HfApi.create_commit(...)`.
Can be done in the same session as the next Space push (e.g. when the
arXiv ID lands).

### Collection pin

Create an HF Collection titled **"KV-cache compression"** (user-scope)
containing:

1. The Space `FluffyAIcode/LLM-KA-Cache-Compress`.
2. Any future paper page on HF (once the arXiv ID is registered via
   HF's [Papers](https://huggingface.co/papers) system).
3. External papers we compare against (TurboQuant's HF page if it
   exists; KIVI's; etc.) for topical clustering.

Collections are a moderate GEO signal because they show up on each
member's sidebar; a KV-cache-compression Collection that names
adjacent methods strengthens our authority graph position.

## 2 — Model-card inbound link-backs

### Rationale

A PR on a popular model card's `README.md` that adds a line like

> **Related projects**: [KakeyaLattice](...) — drop-in KV-cache
> compression for this model, 2.4×–2.8× CR at <1 % ppl loss.

…is a high-leverage move **when it lands**. Model cards are among
the highest-authority pages on Hugging Face and are indexed
aggressively by AI answer engines. The success rate depends entirely
on the model author's review latency.

### Candidates (ordered by expected ROI)

| model card | reviewer | expected decision time | notes |
|:-----------|:---------|:-----------------------|:------|
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Alibaba / Qwen team | medium (they review community PRs) | natural fit — the Space uses Qwen3-0.6B as the default |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Alibaba / Qwen team | medium | our strongest benchmark number (2.77× @ 2 % \|Δppl\|) uses this |
| [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | Meta | low (gated community edits) | try but do not expect acceptance |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | DeepSeek team | medium | we have a full benchmark row |
| [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) | Zhipu team | medium | +37.8 % compression advantage over TurboQuant @ 2 % \|Δppl\| — our biggest per-model win |
| [google/gemma-4-e4b](https://huggingface.co/google/gemma-4-e4b) | Google team | low (gated community edits) | both codecs saturate at 3.04× so we look good but not differentiated |

### Suggested PR body (per model card)

Adapt per model card. This template is for Qwen3-4B; rewrite the
compression-ratio sentence to cite the actual measured number per
model.

---

**Title**: `docs: add KakeyaLattice to Related projects`

**Body**:

```
Adding a single line to the "Related projects" section linking to
KakeyaLattice, a drop-in `transformers.DynamicCache` subclass that
compresses the KV cache of Qwen3-4B **2.40× at ≤ 1 % perplexity
loss** and **2.77× at ≤ 2 %** (real vLLM prefill + real FlashAttention
bf16 on NVIDIA H200, WikiText-103 n=8 × 64 evaluation positions per
passage = 512 positions per channel; raw JSON at
https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/v1_4_release/kv_128k_isoppl_n8).

Usage is three lines:

```python
from kakeyalattice.hf import KakeyaLatticeCache
cache = KakeyaLatticeCache(
    variant="e8", q_range=38,
    num_hidden_layers=model.config.num_hidden_layers,
    head_dim=model.config.head_dim,
)
out = model.generate(**inputs, past_key_values=cache, use_cache=True)
```

- Repo: https://github.com/FluffyAIcode/LLM-KV--Cache-compress
- PyPI: https://pypi.org/project/kakeyalattice/
- Live demo: https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress
- Citation: https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/CITATION.cff

The KakeyaLattice compare table in the README cites Qwen3-4B
alongside Qwen3-0.6B, GLM-4-9B-Chat, Gemma-4-E4B, and
DeepSeek-R1-Distill-Qwen-1.5B. No change to this model card's
numeric claims or recommended usage — this is a pointer-only edit.
```

**Diff to propose** (adapt path if the model card uses different
section names):

```diff
+ ## Related projects
+
+ - [KakeyaLattice](https://github.com/FluffyAIcode/LLM-KV--Cache-compress)
+   — drop-in `transformers.DynamicCache` subclass, 2.4×–2.8× KV cache
+   compression at under 1 % perplexity loss on this model.
```

If the card already has a "Related projects" or "Community extensions"
section, insert a single bullet there instead.

### How to file

Each HF repo supports **"Community" tab → "New discussion"** for a
soft approach before a PR, and **"Community" tab → "Pull request"**
for the PR itself. For model authors who have never interacted
publicly, a discussion first is polite; for authors who merge
community PRs regularly (Alibaba-Qwen, HF's own models), a PR is
faster.

## Done when

- Space README carries PyPI + GitHub + License badges at the top.
- At least two of the six model cards above carry a "Related projects"
  entry linking back to this repo.
- The HF Collection "KV-cache compression" is live and pinned to the
  Space.
