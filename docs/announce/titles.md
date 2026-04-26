# Announcement — title candidates

Three headline variants, each hits a different audience. Each one is
strictly defensible against the published numbers in
`reports/v1_4_release/kv_128k_isoppl_n8/V14_VS_TQ_ISOPPL_REPORT.md`.

## A. Benchmark-led (recommended for HN / r/MachineLearning)

> **KakeyaLattice: 2.77× KV cache compression for Qwen3-4B with 1.67 % perplexity loss (drop-in DynamicCache subclass)**

Why: numeric, specific, mentions a well-known model. Lead with the
single strongest headline point (Qwen3-4B ≤ 2 % target → 26.9 % more
compression than the best scalar-quant baseline at the same quality).
"drop-in DynamicCache subclass" signals low integration cost.

## B. Problem-led (recommended for r/LocalLLaMA, dev.to, personal blog)

> **We compressed Qwen3-4B's KV cache 2.8× with near-zero quality loss using lattice quantisation — here's how to plug it in**

Why: reads as a practitioner's problem, not a research announcement.
"near-zero" is fine here — at Q=38 the |Δppl| is <1 % across all four
models. The post body must immediately back it up with the table.

## C. Research-led (recommended for Twitter/X, arXiv bsky, mailing lists)

> **KakeyaLattice (v1.5): nested D4/E8 lattice quantisation of LLM KV caches beats scalar quantisation by 9 %–38 % CR at matched perplexity across 4 model families**

Why: claim is precise, spans the full family range. Good for audiences
that want to check the math before clicking.

## Social-short variants (≤ 280 chars)

- *[A-short]* `KakeyaLattice: 2.77× KV compression for Qwen3-4B at 1.67 % Δppl, drop-in DynamicCache subclass. pip install kakeyalattice. Live demo: hf.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress`

- *[B-short]* `Compressed Qwen3-4B KV cache 2.8× with ~1 % ppl loss. Real vLLM on H200. Same API as DynamicCache — swap one line. Code + paper + HF Space in thread.`

- *[C-short]* `D4/E8 nested-lattice KV compression > scalar KV quant by 9–38 % CR at matched |Δppl| on Qwen3, Llama-3, GLM-4, DeepSeek-R1-Distill. Paper + 128 k iso-PPL sweep: github.com/FluffyAIcode/LLM-KV--Cache-compress`

## My recommendation

**A for HackerNews** (the audience clicks through on specific numbers
in titles, and the 26.9 % CR advantage at 1.67 % |Δppl| is our strongest
single number).

**B for r/LocalLLaMA** (the audience is running local models on
limited VRAM; "plug it in" reads correctly as a practitioner pitch).

**C for Twitter** (280 chars, research-fluent audience).
