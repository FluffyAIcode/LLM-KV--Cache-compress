# vLLM integration discussion — pre-written issue body

Paste the contents of this file into a **new GitHub Discussion or Issue**
at <https://github.com/vllm-project/vllm>. The discussion / issue
format is deliberate — we do not yet have a PR to show, and vLLM
maintainers prefer a discussion when the proposal needs scoping before
code.

## Where to file

- **Primary choice**: `https://github.com/vllm-project/vllm/discussions`
  under the category **"RFC"** (Request For Comment).
- **Fallback**: `https://github.com/vllm-project/vllm/issues/new/choose`
  → "Feature request". Use this only if Discussions are disabled for
  your account or the maintainers redirect you.

## Suggested title

```
[RFC] Third-party KV-cache quantiser plugin: KakeyaLattice (nested D4/E8 lattice, 2.4–2.8× CR at <1% |Δppl|)
```

## Suggested labels

Ask the maintainers to add (one of these at filing time is plenty —
they will add the rest):

- `new-feature-proposal`
- `kv-cache`
- `quantization`
- `RFC`

## Body (paste verbatim from here)

```markdown
## Proposal

Ship a third-party plugin path for a new KV-cache quantiser,
**KakeyaLattice**, a nested D4 / E8 lattice codec for transformer
KV activations that lands via vLLM's existing `general_plugins` entry
point. Code already exists at
<https://github.com/FluffyAIcode/LLM-KV--Cache-compress> and is on
PyPI as `kakeyalattice` (v1.5.0, MIT-licensed).

I am filing this as an RFC rather than a PR because the code I have
works today as a capture / replace monkey-patch and is **not yet a
clean integration into vLLM's paged KV manager**. I want to align on
the integration path before writing that bridge, to avoid a duplicate
of the `QuantoQuantizedCache` / `HQQQuantizedCache` work in
`transformers` that landed piecemeal.

## What KakeyaLattice does

- **Input**: a `[seq, heads, head_dim]` K or V tensor from any
  transformer attention layer.
- **Pipeline**: Sylvester–Hadamard rotate → per-vector adaptive L²
  scale → nested D4 (dim-4) or E8 (dim-8) lattice closest-point
  encode → store indices.
- **Decode**: one matmul + one unscale.
- **Operating points**: three canonical `q_range` settings
  (10 aggressive, 38 balanced, 152 near-lossless).

It is a **stateless per-vector function** — no calibration, no
warm-up, no cross-token state. Streaming / online decode is
supported by construction.

## Why another KV quantiser

vLLM already ships `--kv-cache-dtype fp8` and interoperates with
`transformers`'s `QuantoQuantizedCache` / `HQQQuantizedCache`
classes. What changes:

At the **tight quality budget most production deployments tune for**
(≤ 1 % |Δppl|), KakeyaLattice compresses **9 %–38 % harder** than
TurboQuant (the strongest published per-channel scalar baseline) on
four open-source model families. Real vLLM prefill + real
FlashAttention bf16 forward on NVIDIA H200, WikiText-103, n=8
passages × 64 eval positions per passage, 128 k context:

| model                          | KakeyaLattice CR | TurboQuant CR | advantage |
|:-------------------------------|-----------------:|--------------:|----------:|
| Qwen3-4B                       |        **2.40×** |         1.95× | +23.3 %   |
| GLM-4-9B-Chat                  |        **1.73×** | out of range  | KL only   |
| Gemma-4-E4B                    |        **3.04×** |         3.04× | tied      |
| DeepSeek-R1-Distill-Qwen-1.5B  |        **2.29×** |         2.09× | +9.2 %    |

At ≤ 2 % |Δppl| the advantage grows to +27, +38, tied, +3 %
respectively. Raw JSON + reproducer at
<https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/v1_4_release/kv_128k_isoppl_n8>.

The mechanism: real LLM KV activations are **heavy-tailed and
non-isotropic**. Per-channel scalar quantisers allocate bits for the
worst-case channel. A Sylvester–Hadamard rotation empirically
gaussianises the distribution (see the non-Gaussian audit in
<https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/reports/v1_5_release/dsv4_stage075/FINDINGS_N8.md>),
after which D4/E8 lattice quantisation exploits the densest sphere
packings in those dimensions.

## What already exists

In the [kakeyalattice repo](https://github.com/FluffyAIcode/LLM-KV--Cache-compress):

1. **`kakeyalattice.hf.KakeyaLatticeCache`** — a drop-in subclass
   of `transformers.DynamicCache`. Works today with any
   `model.generate(past_key_values=cache, ...)` call. The
   [HF Space](https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress)
   uses this on Qwen3-0.6B.
2. **`vllm_backend/kakeya_v1_4_snapshot/`** — a
   `vllm.general_plugins` entry point that monkey-patches the
   Attention path on Qwen2/3, Gemma4, GLM to capture
   post-QK/V-norm, pre-RoPE K and V and replace them with the
   roundtripped versions. **This is the "capture and replace" mode
   used to generate the 128 k iso-PPL tables above**. It works
   today on vLLM `0.19.2rc1.dev100` with `transformers` 5.5.2.

The monkey-patch mode is **not** a real memory-saving integration —
it stores the reconstructed tensor in the model's KV dtype. That
is the gap I want to close with vLLM's help.

## Integration path I'd like to RFC

Three possible landing points, from lowest to highest invasiveness:

### Path A — register as a `KVCacheQuantConfig` backend

`vllm/config.py` has a `KVCacheQuantConfig` enum and a registry in
`vllm/kv_transfer/`. Add a `"lattice"` value that dispatches to a
`KakeyaLatticeKVManager` implementing the existing
`KVCacheManagerBase` protocol. The manager would:

- On prefill / decode: encode K and V blocks via E8 closest-point,
  store lattice indices in the paged KV buffer instead of bf16 /
  fp8 values.
- On attention read: one matmul to decode (per 8-D block), then
  the existing FlashAttention path runs unchanged.

Pros: uses vLLM's own page-allocator; no kernel changes.
Cons: decode overhead per attention read is ~0.25 ms on H200 today
(< 2 % of bf16 decode step at batch 1); on smaller GPUs this might
be worse.

### Path B — fused decode in the attention kernel

Write a Triton kernel that reads lattice indices + fused-unscales
during the QK⁻¹ step of FlashAttention. Faster but **invasive** and
requires maintaining a codec-aware variant of FlashAttention.

### Path C — compressed cache in `nextn`-style hot tier only

Store bf16 for the last ~1 k tokens (active decode window), encode
the rest via E8. Trades a small HBM win for zero decode-path
complexity.

**My default proposal is Path A** because it matches what vLLM
already does for INT8 / FP8 and keeps the blast radius small.

## What I'd like from the vLLM maintainers

1. Agreement on **Path A** (or a pointer to Path B / C if you see a
   better fit).
2. Pointer to the **exact interface** you want a new KV-cache backend
   to implement. I read `vllm/worker/model_runner.py`,
   `vllm/kv_transfer/`, and `vllm/core/block/block_manager.py`, but
   the canonical integration point has moved around in the 0.6 → 1.0
   transition.
3. A **`vllm-plugin-kakeyalattice`** naming convention, if you'd
   prefer the plugin live under a `vllm-project/*-plugin-*` naming
   scheme rather than in my own namespace.

Happy to open the PR as soon as we've agreed on Path and interface.
Paper draft at
<https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/reports/paper/kakeyalattice.pdf>
(arXiv submission pending).

## Compliance note

All numbers above come from real vLLM prefill + real FlashAttention
bf16 forward on NVIDIA H200. No mocks, no fallbacks. The `reports/`
tree of the kakeyalattice repo carries a SHA-256 manifest so claims
are reproducible end-to-end from the committed JSON.
```

## How to follow up

After filing:

1. Post a one-line cross-reference from the kakeyalattice repo —
   either on the `AgentMemory/discovery-runbook-c478` PR (this PR) or
   as a new issue titled `discovery: vLLM RFC filed at vllm-project/vllm#<N>`.

2. When a vLLM maintainer engages, reply **within 4 hours** during
   Pacific business hours. This is the single highest-leverage
   engagement of the six discovery tasks; the thread's visibility
   drops sharply once the first reply goes stale.

3. If the RFC gets closed without a maintainer reply within 7 days,
   file it as an issue (not a discussion) tagged `feature-request`
   + `kv-cache` and @-mention a maintainer who has touched
   `vllm/core/block/block_manager.py` in the last 90 days. A
   `git log` + email-on-commits query can find the right handle.

## If vLLM wants numbers beyond what we have

Two likely maintainer asks, and the one-sentence response:

- **"Have you measured real HBM savings, not just rel-MSE?"** — No,
  that's exactly why Path A is the RFC. The reference impl round-trips
  K/V through the codec. Path A is the first integration where CR
  equals HBM ratio.

- **"Have you benchmarked against KIVI on Qwen3?"** — Not yet, direct
  iso-bit head-to-head vs KIVI is on the roadmap. Our current
  baseline is TurboQuant (the strongest published scalar KV quantiser
  at our bit budgets) and the Paper with Code submission at
  <https://paperswithcode.com/paper/kakeyalattice> will carry the
  KIVI comparison once the arXiv ID is minted.
