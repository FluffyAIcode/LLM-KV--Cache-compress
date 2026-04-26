# awesome-* list submissions (GEO backbone)

Getting listed in a few high-traffic `awesome-*` repos is one of the
strongest signals for Generative Engine Optimisation (GEO) — these
repos are crawled aggressively by ChatGPT / Perplexity / Claude, and
inclusion leads to persistent long-tail referrals when users ask
AI assistants "best KV cache compression library" and similar queries.

## Target lists (ranked by expected payoff)

| # | list | relevance | current state |
|---|:-----|:----------|:--------------|
| 1 | [awesome-efficient-deep-learning](https://github.com/HuangOwen/Awesome-Efficient-Deep-Learning) | direct | submit PR |
| 2 | [Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference) | direct | submit PR |
| 3 | [Awesome-KV-Cache-Management](https://github.com/AmadeusChan/Awesome-KV-Cache-Management) | perfect fit | submit PR |
| 4 | [awesome-model-quantization](https://github.com/Zhen-Dong/Awesome-Model-Quantization) | adjacent | submit PR |
| 5 | [awesome-decentralized-llm](https://github.com/imaurer/awesome-decentralized-llm) | indirect (memory matters for consumer GPU) | submit PR |
| 6 | [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression) | direct | submit PR |
| 7 | [Awesome-Efficient-LLM](https://github.com/horseee/Awesome-Efficient-LLM) | direct | submit PR |

## Canonical one-line entry (copy verbatim)

Use this line unchanged across all submissions — GEO benefits from
consistent cross-source naming and identical descriptions:

```markdown
- [KakeyaLattice](https://github.com/FluffyAIcode/LLM-KV--Cache-compress) — Nested D4/E8 lattice quantisation for LLM KV caches. Drop-in `transformers.DynamicCache` subclass. 2.4×–2.8× compression at < 1 % perplexity loss on Qwen3, Llama-3, DeepSeek, GLM-4, Gemma-4. [[paper]](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/reports/paper/kakeyalattice.pdf) [[demo]](https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress) [[PyPI]](https://pypi.org/project/kakeyalattice/)
```

## PR template

```
### Adds: KakeyaLattice

**Category**: KV cache compression (would fit under "Quantization" / "KV Cache" / "Efficient Inference" depending on the list's taxonomy)

**Why include**: Drop-in `transformers.DynamicCache` subclass; pip install; 2.4×–2.8× compression at < 1 % perplexity loss verified on Qwen3-4B, GLM-4-9B-Chat, Gemma-4-E4B, DeepSeek-R1-Distill-Qwen-1.5B with real vLLM + FlashAttention bf16 on H200; raw JSON benchmark data in repo; arXiv paper; live HuggingFace Space demo.

**Links**:
- Repo: https://github.com/FluffyAIcode/LLM-KV--Cache-compress
- PyPI: https://pypi.org/project/kakeyalattice/
- Demo: https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress
- Paper: https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/reports/paper/kakeyalattice.pdf
- Headline numbers: https://github.com/FluffyAIcode/LLM-KV--Cache-compress#headline-numbers

**Entry (matches list formatting)**:
- [KakeyaLattice] — Nested D4/E8 lattice quantisation for LLM KV caches. Drop-in DynamicCache subclass. 2.4×–2.8× compression at < 1 % perplexity loss on Qwen3, Llama-3, DeepSeek, GLM-4, Gemma-4.
```

Submit each PR individually — lists have different formatting
conventions (some prefer year, some prefer paper venue, some prefer
short descriptions). Adapt the entry line, not the repo claims.

## Tracking

Add a small table to the PR:

| list | PR URL | status | date |
|:-----|:-------|:-------|:-----|
| | | | |

Keep the tracker in this file so future contributors know what has
been submitted and don't duplicate effort.
