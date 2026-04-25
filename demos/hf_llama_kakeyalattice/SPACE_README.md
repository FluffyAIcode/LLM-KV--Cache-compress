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

# KakeyaLattice KV-cache compression demo

Side-by-side comparison of **bf16 DynamicCache** vs **KakeyaLattice E8**
compression at three quality levels (Q=10 aggressive, Q=38 balanced,
Q=152 near-lossless) on a small HuggingFace causal LM.

Default model: `Qwen/Qwen2-0.5B` (head_dim=64, E8-compatible, runs on
free CPU tier). Override `KAKEYA_DEMO_MODEL` env var to use a larger
model on a GPU Space.

## How it works

`KakeyaLatticeCache` is a drop-in subclass of `transformers.DynamicCache`
that applies a Zamir-Feder nested-lattice codec roundtrip (encode +
decode) to every K and V written into the cache.

```python
from transformers import AutoModelForCausalLM
from kakeyalattice.hf import KakeyaLatticeCache

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
cache = KakeyaLatticeCache(
    variant="e8", q_range=38,
    num_hidden_layers=model.config.num_hidden_layers,
    head_dim=model.config.head_dim,
)
out = model.generate(input_ids, max_new_tokens=200, past_key_values=cache)
```

## What you'll see in the demo

For each prompt, the app generates four times:

| config                   | bits/token        | expected quality                       |
| ------------------------ | ----------------- | -------------------------------------- |
| bf16 DynamicCache        | 1024 (reference)  | identical to reference                 |
| E8 Q=152 near-lossless   | ~960 (-6%)        | essentially identical                  |
| E8 Q=38 balanced         | ~440 (-57%)       | ~1% deviation in ppl                   |
| E8 Q=10 aggressive       | ~320 (-69%)       | noticeably different but coherent      |

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
