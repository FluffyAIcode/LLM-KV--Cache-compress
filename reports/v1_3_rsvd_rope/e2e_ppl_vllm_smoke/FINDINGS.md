# v1.3 end-to-end PPL validation on vLLM (smoke)

## Setup

- **Engine**: vLLM 0.7.3 (Flash Attention backend, V0 engine, `enforce_eager=True`)
- **Torch / CUDA**: PyTorch 2.5.1 + CUDA 12.4, NVIDIA H200 80 GB (via Vast.ai)
- **Model**: `Qwen/Qwen2.5-0.5B`, bf16 weights
- **Data**: WikiText-103 test split, first 2 passages tokenising to
  `>= ctx_len + n_eval = 1088` tokens
- **Evaluation window**: `[ctx_len, ctx_len + n_eval) = [1024, 1088)` — 64
  teacher-forced next-token positions per passage
- **Codec config**: v1.3 (bit_width=2, block_size=512, randomized PCA,
  variance_ratio=0.95, rsvd_target_rank = head_dim / 2)
- **K/V routing** (asymmetric, matching the HF harness from PR #12):
  - K: `metric = inner_product`, `share_basis = False`
  - V: `metric = mse`, `share_basis = True`

## How the codec is attached

The Rust codec is not inlined into the CUDA attention kernel. Instead,
`benchmarks/e2e_ppl_validation_vllm.py` monkey-patches
`vllm.attention.layer.Attention.forward` **before** constructing `LLM`.
When a module-level `CodecState.active` flag is True, the patched
forward:

1. Receives the incoming `key` and `value` tensors
   (`[num_tokens, num_kv_heads, head_size]`)
2. Reshapes each to `[N, head_size]`
3. For each K/V call, writes the tensor to a KKTV temp file and
   invokes `kakeyaturbo-bench --dump-decoded` in a subprocess (release
   build)
4. Reads back the decoded tensor and substitutes it for the original
   K / V before delegating to the underlying paged-attention op

This is a quality probe, not a performance integration — the
subprocess round-trip dominates `t_alt` below (~4.5 s per passage for
48 K+V calls vs. ~0.1 s for the reference run).

## Result

| Passage | ppl_ref | ppl_alt | Δppl       | top-1 | KV calls |
|:-------:|--------:|--------:|-----------:|------:|---------:|
| 1       | 21.09   | 61.67   | **+192.3 %** | 43.8 % | 48       |
| 2       | 9.36    | 45.99   | **+391.5 %** | 50.0 % | 48       |

**Aggregate**: `mean Δppl = +291.9 %`, `mean top-1 agreement = 46.9 %`
→ **VERDICT: REJECT** (standard ACCEPT threshold: `|Δppl| ≤ 1 % AND
top-1 ≥ 95 %`; MARGINAL: `|Δppl| ≤ 3 % AND top-1 ≥ 85 %`).

## Cross-engine comparison with PR #12 (HF-transformers harness)

| Engine | Config                          | Δppl mean | top-1  | Verdict |
|-------:|:--------------------------------|----------:|-------:|:-------:|
| HF     | v1.3 default b=2 rsvd r=D/2     | +29 086 % | 23.0 % | REJECT  |
| HF     | v1.2 default b=3 randomized     | +11 030 % | 24.0 % | REJECT  |
| HF     | Max fidelity b=4 vr=1.0 exact   | +24 310 % | 19.8 % | REJECT  |
| **vLLM** | **v1.3 default b=2 rsvd r=D/2** | **+292 %** | **47 %** | **REJECT** |

Two observations:

1. **Sign agrees, magnitude differs**. Both engines REJECT the codec
   on real downstream PPL at the tested configuration. vLLM shows a
   smaller quality hit (+292 % vs HF's +29 000 %), which is consistent
   with the two engines writing the KV tensor at slightly different
   points of the attention pipeline — vLLM's Flash-Attention backend
   reshapes `key` / `value` with the RoPE already applied, whereas the
   HF eager path runs the codec on the same tensors in a different
   internal layout. The result is qualitatively the same: v1.3 at its
   tier-1 setting fails the ACCEPT bar on downstream PPL under the
   production inference engine, not just under HF eager.

2. **Top-1 agreement is higher on vLLM (47 %) than on HF (23 %)**.
   That points at the same underlying codec distortion being less
   destructive when applied to the Flash-attention-era KV layout
   than to the HF eager layout. This is an interesting diagnostic
   signal but does not change the ACCEPT/REJECT verdict.

## What this unblocks / does not unblock

- **Unblocked**: the codec now has a reproducible vLLM-engine PPL
  measurement path. Any future codec change can be sanity-checked on
  vLLM in ~30 s on an H200 before the full HF matrix run.
- **Not unblocked**: the paper's "preserves downstream quality" claim
  still does not survive end-to-end PPL on either engine. PR #12's
  call to either repair the codec or rewrite the quality claims is
  unchanged.

## Reproduce

```bash
# On a machine with CUDA + vLLM installed (or Vast.ai H200; see
# agent's setup_kk.sh for a one-shot installer):
cd LLM-KV--Cache-compress
git checkout AgentMemory/v1-3-ppl-vllm-integration-102e
./benchmarks/run_v1_3_ppl_vllm.sh
# → reports/v1_3_rsvd_rope/e2e_ppl_vllm_smoke/qwen2_5_0_5b_vllm.json
```

## Artifact

`qwen2_5_0_5b_vllm.json` in this directory contains the full
per-passage metrics.
