---
title: KakeyaLattice KV-cache compression
emoji: 📐
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
---

# KakeyaLattice KV-cache compression

A Gradio Space that compares bf16 baseline vs KakeyaLattice E8 at three
compression levels on a small HuggingFace causal LM.

## Local run

```bash
pip install kakeyalattice[hf] gradio
python app.py
```

## HF Space deployment

1. Create a new Space on https://huggingface.co/new-space
2. Choose Gradio SDK, Python 3.10+
3. Select CPU-Basic (free) or T4-small (GPU) depending on which model
   you want to serve
4. Copy `app.py` and this `README.md` to the Space repo
5. Push: `git push`

The default model is `Qwen/Qwen3-0.6B` (head_dim=128, GQA 16/8,
E8-compatible, fits on free HF Space CPU tier).
Override with:

```bash
KAKEYA_DEMO_MODEL=Qwen/Qwen3-1.7B python app.py       # GPU recommended
KAKEYA_DEMO_MODEL=Qwen/Qwen3-4B python app.py         # GPU required
KAKEYA_DEMO_MODEL=meta-llama/Llama-3.2-1B python app.py
```

## What the demo shows

For each prompt, the Space runs generation four times:

1. **bf16 DynamicCache** — reference (uncompressed)
2. **E8 Q=10 aggressive** — ~3.6× KV compression per layer
3. **E8 Q=38 balanced** — ~2.5× KV compression per layer (recommended)
4. **E8 Q=152 near-lossless** — ~1.9× KV compression per layer

For each run it reports the generated text, wall-clock latency, and
bits-per-token-per-head. Text quality degrades gracefully from near-
identical to baseline (Q=152) to mildly different (Q=38) to noticeably
different but still coherent (Q=10).

## Caveats

- Current KakeyaLatticeCache **roundtrips** K/V (encode + decode) before
  storing in bf16. So HBM savings in this demo are **nominal** — the
  point is to show reconstruction quality, not memory savings. Real HBM
  savings require a custom attention kernel that stores lattice indices
  natively (see the vLLM backend in the main repo).
- Decode latency is ~1.3–2× slower than bf16 baseline because the codec
  runs as pure PyTorch ops per layer per token. A fused Triton kernel
  would close most of this gap.
- Head-dim must be a power of 2 AND divisible by 8 (for E8) or 4 (for
  D4). Most LLMs satisfy this (head_dim ∈ {64, 128, 256}); old models
  with 96 or 176 do not.

## Links

- GitHub: https://github.com/FluffyAIcode/LLM-KV--Cache-compress
- PyPI: https://pypi.org/project/kakeyalattice/
- Paper: [KakeyaLattice (reports/paper/)](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/paper)
