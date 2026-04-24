# Stage 0.5 — DeepSeek-V4-Flash architecture probe

**Status**: scaffold, awaiting H200 run.

## What this is

The smallest honest experiment we can run to learn whether **KakeyaLattice's
five engineering levers + D4/E8 shaping gain are still relevant on
DeepSeek-V4-Flash's KV cache**, without requiring

  * a 284 B / 150 GB multi-node V4 checkpoint,
  * vLLM's still-missing `DeepseekV4Attention` support, or
  * the tilelang kernels (`sparse_attn`, `fp8_gemm`, `fp4_gemm`,
    `hc_split_sinkhorn`) needed for the full inference stack.

What we **do** run: a pure-PyTorch port of V4-Flash's KV-write path ——
`Attention.wkv -> kv_norm -> RoPE -> FP8(nope-only)` and
`Compressor.forward` with the gated-pool + overlap-transform + RoPE +
FP8 postamble —— fed **real LLM hidden states** (from Qwen3-4B or any
other host model on hand). KakeyaLattice Q=10 / Q=38 roundtrips on the
resulting three KV streams (sliding window, CSA-ratio-4 pool,
HCA-ratio-128 pool), compared against V4's own FP8 baseline.

## Honesty caveats, up front

1. **Weights are random Gaussian-init.** The V4-Flash Compressor's
   `wkv`, `wgate`, and `ape` parameters are trained FP8 tensors; we
   replace them with `std=hidden^-0.5` random inits. This experiment
   measures *architectural* distribution shape, not the exact numerical
   values a trained V4-Flash would produce in that position.
2. **Three layers only, not all 43.** The V4-Flash `compress_ratios`
   list is `[0, 0, 4, 128, 4, 128, ..., 0]`. We capture one
   representative of each: sliding-window-only (ratio 0),
   CSA-with-Indexer (ratio 4), HCA-without-Indexer (ratio 128). The
   ratio-4 and ratio-128 layers alternate down the stack; we believe
   single-layer statistics are representative, but this is untested.
3. **No Indexer.** The Indexer (inference/model.py:380-433) is a side
   path producing per-query top-k selection indices, not KV values
   landing in the main cache. We omit it because KakeyaLattice operates
   on stored KV tensors, not on selection indices. The Indexer's own
   output (indices) doesn't enter the KV cache at all.
4. **No Hyper-Connections.** V4 uses 4-copy residuals + Sinkhorn mixing
   (`Block.hc_pre`/`hc_post`). Our harness feeds the host model's
   post-embedding hidden states directly into the KV projection, i.e.
   we bypass HC. Real V4 input to `Attention.wkv` would be
   `attn_norm(hc_pre(x))`, an HC-mixed tensor. HC is a learned linear
   rebalancing and should preserve sub-Gaussian / heavy-tail character,
   but we don't verify this.
5. **Single passage**. Our non-Gaussian audit is computed over one
   WikiText-style passage × batch × 2048 tokens, giving ~2k–16k vectors
   per stream. The non-Gaussian gates (kurtosis, isotropy, Hadamard
   variance ratio) are computed to the same definitions as the paper
   (`§1.3`), so these numbers are directly comparable.

## What we claim from this

  **If** the non-Gaussian audit fires on the CSA / HCA / sliding streams
  with roughly the same strength as Qwen3-4B post-QK-norm K
  (§1.3 reports: excess kurtosis 0.84, RMS W2/σ 0.65, variance ratio
  4.71), **then** the five engineering levers are motivated on V4-arch
  KV. Otherwise the V4 Compressor's own pooling has already flattened
  the relevant distribution features and KakeyaLattice's headroom shrinks
  toward the pure D4 / E8 shaping-gain asymptote.

  **If** KakeyaLattice Q=10 (576 bits/vector at D=128, scaling to 2304
  bits at D=512) achieves rel-MSE ≤ FP8 baseline at the same or fewer
  bits, that's a positive signal. V4's FP8 is ~10 bits per vector-dim
  = 5120 bits for D=512; KakeyaLattice Q=10 at D=512 is ~2304 bits +
  overhead, so we're comparing **~2× compression** *on top of* V4's
  internal FP8. This is the headroom question the paper can't answer
  without this run.

## What we do NOT claim

  * End-to-end Δppl impact on V4 outputs. That requires the full 43-
    layer stack + trained weights + MoE, which is out of scope.
  * Latency parity with V4's tilelang kernels. Our FP8 simulation is
    portable PyTorch with a fake-quant round-trip via
    `float8_e4m3fn.to()`.
  * Exact RoPE-phase match with a trained V4. Random weights produce
    random post-projection phases; RoPE frequencies and block structure
    are, however, bit-exact ports of `precompute_freqs_cis` and
    `apply_rotary_emb` from `inference/model.py:199-244`.

## Files in this directory

After a run, this directory will contain:

| File | Contents |
| --- | --- |
| `dsv4_stage0_5_report.json` | Structured output: per-stream non-Gaussian audit, per-codec rel-MSE / cosine / wall-time, config echo. |

## How to run

```bash
# Unit tests (CPU-friendly, no weights needed):
cd benchmarks/dsv4_stage0_5
python test_dsv4_generator.py

# Rigorous run on H200 with real host model:
cd benchmarks/dsv4_stage0_5
python run_dsv4_stage0_5.py \
    --host-model qwen3-4b \
    --seqlen 2048 \
    --batch-size 2 \
    --q-values 10,38 \
    --enable-e8 \
    --out ../../reports/v1_5_release/dsv4_stage0_5/dsv4_stage0_5_report.json
```

## Reproducing the V4 architecture port

Operator-level port from `deepseek-ai/DeepSeek-V4-Flash/inference/model.py`
(commit `6e76323`, 2026-04-24):

| V4-Flash reference | Our port |
| --- | --- |
| `Compressor.forward:316-377` (prefill branch) | `DSV4Compressor.forward` |
| `Compressor.overlap_transform:307-314` | `DSV4Compressor._overlap_transform` |
| `Attention.forward:502-506` (wkv + norm + RoPE + FP8 on nope) | `DSV4MainKVProjection.forward` |
| `precompute_freqs_cis:199-229` | `precompute_freqs_cis` (verbatim) |
| `apply_rotary_emb:232-244` | `apply_rotary_emb` (verbatim) |
| `RMSNorm:183-196` | `RMSNorm` (verbatim) |
| `kernel.py:act_quant` (FP8 in-place) | `_simulate_fp8_block_quant_dequant` (portable PyTorch approx) |
| `kernel.py:sparse_attn_kernel` | (not ported — attention is out of scope for Stage 0.5) |
| `kernel.py:hc_split_sinkhorn` | (not ported — HC is bypassed, see caveat 4) |

## Next step (Stage 1)

Once vLLM lands `DeepseekV4Attention` natively — pipeline: see PR #42
body's "Stage 1" bullet for the full plan — we replace this reference
generator with vLLM's live attention hook, run the same three streams
under `rigorous_eval.py` + `niah_eval.py` on actual V4-Flash weights
(on a 2–4× H200 NVLink node), and compare against the Stage 0.5
numbers here as a calibration.
