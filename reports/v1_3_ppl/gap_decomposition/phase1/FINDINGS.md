# Phase 1 + 5 — cross-engine baseline PPL and logit KL

**Question.** With codec OFF, do HF transformers (eager) and vLLM
0.7.3 (FLASH_ATTN, bf16) give the same logits on the same tokens?
If not, the "HF +7.82 %" and "vLLM +35.33 %" numbers sit on
different baselines and part of the 27 pp gap is a baseline-mismatch
artifact.

**Setup.** DS-Distill-Qwen-1.5B, 4 WikiText-103 test passages of
2112 tokens each, eval window `[ctx=2048, 2112)`. HF loaded with
`torch_dtype=bfloat16`, `attn_implementation="eager"`. vLLM 0.7.3,
FLASH_ATTN, bf16, enforce_eager. Same token ids fed to both.

## Result

| Passage | ppl_hf  | ppl_vllm | (vllm−hf)/hf | top-1 agree | KL (top-20) | mean \|Δ log P(true)\| |
|:-:|--:|--:|--:|--:|--:|--:|
| 1 | 155.29 | 124.88 | **−19.59 %** | 79.7 % | 0.173 | 0.467 |
| 2 |  37.00 |  33.00 | **−10.81 %** | 78.1 % | 0.145 | 0.343 |
| 3 |   9.73 |   8.54 | **−12.25 %** | 84.4 % | 0.130 | 0.282 |
| 4 |  26.17 |  25.36 |  −3.13 %   | 85.9 % | 0.134 | 0.297 |

**Aggregate**: mean PPL rel gap **−11.44 %**, mean top-1 agreement
**82.0 %**, mean symmetric KL on top-20 **0.145**, mean |Δlogp(true)|
**0.35 nats**.

## Finding — HF and vLLM disagree on the clean model

At codec OFF on the same tokens the two engines produce:

- **PPL that differs by 11 % on average** (HF consistently higher).
- **KL ≈ 0.15 nats** on the top-20 logprob bucket — an order of
  magnitude larger than bf16 rounding noise; 1 in 5 positions has
  a different top-1 prediction between engines.
- **\|Δ log P(true_token)\| ≈ 0.35 nats** per position — a material
  logit-level disagreement.

This is the baseline mismatch. It means:

1. The production cell's relative Δppl is measured against different
   reference PPLs on each engine. A codec that leaves the ALT run
   statistically close to the REF run on *its own* engine will report
   a smaller Δppl on the engine with the more peaked baseline.
2. "HF eager" is not a neutral oracle to compare vLLM to. Both
   engines are approximations; neither is the ground truth.
3. A substantial fraction of the 27 pp HF↔vLLM Δppl gap (+7.82 % vs
   +35.33 %) is "the two Δppl numbers are answering slightly
   different questions about slightly different logit distributions".

## What this does NOT say

- It does not say the codec is fine. Even if we corrected for
  baseline mismatch, +35 % on vLLM is well above the ~+8 % of HF.
  But the right framing is now "how much of the 27 pp is pure
  engine baseline shift vs how much is the codec genuinely doing
  worse on vLLM?" — a question Phase 2 (noise sensitivity curves)
  is designed to answer.
- It does not implicate Flash-Attention. The KL is measured on clean
  logits; FA's bf16 softmax choice affects these numbers, but so
  does HF eager's own f32 vs bf16 accumulation. Only that they
  DISAGREE at this magnitude has been established.

## Caveat (logged)

HF transformers emits the warning
`"Sliding Window Attention is enabled but not implemented for 'eager';
unexpected results may be encountered."` on DS-Distill-Qwen-1.5B
config. DS-Distill's config has `sliding_window: null`, so this is a
spurious warning, but the eager attention path may be taking a slower
numeric route. vLLM's FLASH_ATTN backend does not share that path.
This is a concrete contributor to the KL we are observing.

## Data

`ds_distill_qwen_1_5b_engine_baseline.json` contains per-passage
`lps_true` only (top-K map is not stored — too large).
