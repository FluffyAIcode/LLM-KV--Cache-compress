# Twitter / X thread

A 6-tweet thread. Each numbered unit goes in a separate tweet. Keep
under 280 chars per tweet (lines below are within that limit).

---

## Tweet 1 (the hook)

D4/E8 nested-lattice KV compression > scalar KV quant by 9–38% CR at matched |Δppl| on Qwen3, Llama-3, GLM-4, DeepSeek-R1-Distill.

Drop-in DynamicCache subclass. pip install kakeyalattice.

Thread ↓

## Tweet 2 (the headline numbers)

Iso-PPL KV compression ratio at ≤ 1 % perplexity loss, real vLLM on H200, 128 k context, WikiText-103 n=8:

• Qwen3-4B: 2.40× (TQ 1.95×)
• GLM-4-9B-Chat: 1.73× (TQ oor)
• Gemma-4-E4B: 3.04× (tied)
• DeepSeek-R1-Distill-Qwen-1.5B: 2.29× (TQ 2.09×)

## Tweet 3 (method, plain English)

Why it compresses harder than scalar KV quantisers:

1. Sylvester–Hadamard rotation gaussianises heavy-tailed KV activations
2. Nested D4 / E8 lattice snap uses the densest packings in dim 4 / 8
3. No calibration, no warm-up — stateless per-vector codec

## Tweet 4 (integration — show, don't tell)

The full integration:

```python
from kakeyalattice.hf import KakeyaLatticeCache
cache = KakeyaLatticeCache(
    variant="e8", q_range=38,
    num_hidden_layers=...,
    head_dim=...,
)
out = model.generate(..., past_key_values=cache)
```

## Tweet 5 (honest caveat)

Honest caveat: today the reference impl round-trips K/V through the codec and stores reconstructed tensors in the model's KV dtype. So HBM bytes are unchanged in the Python reference.

Native vLLM integration (CR = HBM ratio) is the next PR, pending GPU validation.

## Tweet 6 (links)

Try it in a browser (no install): https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress

Code + paper + raw benchmark data: https://github.com/FluffyAIcode/LLM-KV--Cache-compress

PyPI: https://pypi.org/project/kakeyalattice/

---

## Alt-image suggestion

Attach the `assets/hero_pareto.png` (4-panel Pareto front) to
Tweet 2 — that's the visual cue the HN / Reddit version wants readers
to scroll for. Twitter's image preview crops to 2:1-ish, so set the
alt text to "KakeyaLattice vs TurboQuant iso-PPL Pareto front across
Qwen3-4B, GLM-4-9B-Chat, Gemma-4-E4B, and DeepSeek-R1-Distill-Qwen-1.5B".

## Timing

- Wednesday or Thursday, **09:00–10:00 ET** (14:00–15:00 UTC), is
  historically the highest-engagement window for ML research-adjacent
  accounts.
- Schedule Tweet 6 with the links at the end — Twitter's algorithm
  deprioritises threads whose first tweet contains external links.
