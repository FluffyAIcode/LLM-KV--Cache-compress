---
title: KakeyaLattice KV-cache compression
emoji: 📐
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# KakeyaLattice KV-cache compression

Side-by-side comparison of **bf16 DynamicCache** vs **KakeyaLattice E8**
compression at three quality levels (Q=10 aggressive, Q=38 balanced,
Q=152 near-lossless) on a small HuggingFace causal LM.

Default model: `Qwen/Qwen3-0.6B` (head_dim=128, GQA 16/8 — the same
attention shape as modern production LLMs, so the codec numbers are
representative). Runs on the free CPU tier (each "Run comparison"
click takes ~4–8 minutes on 2 cores). Override `KAKEYA_DEMO_MODEL`
env var to use a larger model on a GPU Space (`Qwen/Qwen3-1.7B`,
`Qwen/Qwen3-4B`).

## How it works

`KakeyaLatticeCache` is a drop-in subclass of `transformers.DynamicCache`
that applies a Zamir-Feder nested-lattice codec roundtrip (encode +
decode) to every K and V written into the cache.

```python
from transformers import AutoModelForCausalLM
from kakeyalattice.hf import KakeyaLatticeCache

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
cache = KakeyaLatticeCache(
    variant="e8", q_range=38,
    num_hidden_layers=model.config.num_hidden_layers,
    head_dim=model.config.head_dim,
)
out = model.generate(input_ids, max_new_tokens=200, past_key_values=cache)
```

## What you'll see in the demo

For each prompt, the app generates four times (bits/vec here assume
head_dim=128 → bf16 baseline is 2048 bits/vec; exact numbers for other
head_dims scale proportionally):

| config                   | bits/vec (head_dim=128) | expected quality                  |
| ------------------------ | ----------------------- | --------------------------------- |
| bf16 DynamicCache        | 2048 (reference)        | identical to reference            |
| E8 Q=152 near-lossless   | ~1920 (-6%)             | essentially identical             |
| E8 Q=38 balanced         | ~880 (-57%)             | ~1% deviation in ppl              |
| E8 Q=10 aggressive       | ~640 (-69%)             | noticeably different but coherent |

(The percentage savings `-6% / -57% / -69%` are what matter — they are
fixed by the E8 codec design and do not depend on head_dim.)

Wall-clock latency per config is also reported.

## Caveats

- The cache roundtrips K/V but stores the reconstructed tensor in the
  model's KV dtype. Real HBM bytes saved are **nominal** — the demo's
  value is showing reconstruction quality, not memory savings.
- Decode is ~1.3-2× slower than bf16 because the codec runs as pure
  PyTorch ops. A fused Triton kernel would close this gap.
- Head-dim must be a power of 2 and divisible by 4 (D4) or 8 (E8).
  Most modern LLMs satisfy this.

## Links

- Package: https://pypi.org/project/kakeyalattice/
- Repo: https://github.com/FluffyAIcode/LLM-KV--Cache-compress
- Paper: `reports/paper/`
- DeepSeek-V4-Flash Stage 0.75 findings: `reports/v1_5_release/dsv4_stage075/FINDINGS.md`
