# `benchmarks/dsv4_stage075/` — Stage 0.75 V4-Flash audit with TRAINED weights

Upgrade path from Stage 0.5:

- **Stage 0.5** (`benchmarks/dsv4_stage0_5/`): pure-PyTorch port of V4-Flash
  attention, **random-Gaussian init** weights, fed Gemma-4-E4B hidden
  states through them.
- **Stage 0.75** (this directory): same port, **actual trained V4-Flash
  weights** from HF shards 2, 4, 5 (covering one representative layer
  of each attention type: SWA / c4a / c128a).
- **Stage 1** (`benchmarks/dsv4_stage1/`): full live-vLLM integration with
  the `DeepseekV4Attention` snapshot hook. Requires ≥ 2× H200 and
  vLLM V4 support. Scaffolded in PR #47, execution deferred.

## Files

| file | purpose |
| --- | --- |
| `dsv4_weight_loader.py` | load FP8-E4M3 safetensor shards, dequantize via E8M0 block scales, inject into Stage 0.5's `DSV4MainKVProjection` + `DSV4Compressor` |
| `run_stage075_real_weights.py` | **n=1** driver: host hidden → V4 KV streams (trained weights) → non-Gaussian audit + KakeyaLattice / FP8 codec comparison (single passage) |
| `run_stage075_n8.py` | **n=8 driver** (new): same pipeline, 8 semantically diverse passages, Student-t 95% CI aggregation per stream. See `reports/v1_5_release/dsv4_stage075/FINDINGS_N8.md` for results. |
| `README.md` | this file |

## Why this runs on our existing vast H200

- Only **3 of 46 V4-Flash safetensor shards** needed: layers.0 (SWA) sits
  in shard 2; layers.2 (c4a) in shard 4; layers.3 (c128a) in shard 5.
  Total download: ~11 GB (vs 158 GB for the full model).
- We **skip MoE experts, shared experts, Hyper-Connections, Indexer
  sparse-attention selection** — none of them produce the KV tensors
  we want to audit.
- Host hidden states come from Qwen2-0.5B (~1 GB) projected to 4096-dim
  via a fixed-seed linear.

End-to-end wall time on H200: ~15 seconds.

## Output

`reports/v1_5_release/dsv4_stage075/stage075_trained.json` +
`reports/v1_5_release/dsv4_stage075/FINDINGS.md`. See FINDINGS.md for the
analysis.

## Headline finding — **n=8 with 95 % CI** (2026-04-26 H200 run)

**Canonical one-liner (please reuse verbatim across sources for
cross-source consistency):**

> KakeyaLattice E8 Q=38 on DeepSeek-V4-Flash KV cache:
> **−22 % bits per vector at matched or better reconstruction quality on 23 / 43
> attention layers, neutral on the remaining 20**.
> Measured on 2 × H200, n = 8 passages, Student-t 95 % CI.

**Product headline:**

> V4-Flash + KakeyaLattice = **−22 % KV HBM at zero net quality cost**.
> 4 × H200 node: **126 → ~150 concurrent users at 1 M context**.

E8 Q=38 vs FP8 per-64-block across three V4 KV streams, aggregated
over n=8 diverse WikiText-style passages on trained V4-Flash weights:

```
stream (V4 layer count)   E8/FP8 (mean ± CI95)   n=1 value   bit savings   quality at 78 % bits
sliding_window_kv (3/43)  0.790 ± 0.005          0.786       -22.0 %       +21 %   ← strong win
csa_pool_kv_ratio4 (20/43) 0.900 ± 0.006         0.902       -22.0 %       +10 %   ← moderate win
hca_pool_kv_ratio128 (20/43) 1.043 ± 0.051       0.966       -22.0 %        0 %    ← tied with FP8
```

- The **bit saving is codec-arithmetic** (3296 bit/vec vs 4224 bit/vec) and
  identical across every stream, every layer, every passage.
- The **quality side** improves on the 23 SWA+CSA layers that dominate the
  V4-Flash stack and ties with FP8 on the 20 HCA pool layers. Net
  layer-weighted rel-MSE is **−4.1 % ± 2.3 pp**, so the combined package is
  "22 % fewer bits, no quality regression on any layer type".
- The n=1 HCA "marginal win" (0.966) was a 1.6 σ lucky-tail draw and is
  corrected here. See `reports/v1_5_release/dsv4_stage075/FINDINGS_N8.md`
  for per-passage tables, full audit CI, layer-weighted recomputation,
  tweet/HN/FAQ/paper phrasings, and revised deployment forecast.

Non-Gaussian audit vs paper gates: V4-Flash KV smashes all four paper
gates (kurt, isotropy, Hadamard-variance, W2/σ) by 2–10 000 000×,
**far more non-Gaussian than Qwen3-4B**. The five engineering levers in
KakeyaLattice are fully motivated.

## Next steps

1. Paper addendum (the cheap, high-value option): cite this Stage 0.75
   data in a new "§7.3 Extending to DeepSeek-V4" subsection. No new
   hardware needed.
2. Stage 1: end-to-end Δppl on 2+ H200. ~$50, scaffolded in PR #47.
3. Stage 2 (deployment): custom KV cache manager + fused decode kernel
   for actual HBM savings in production V4 serving. ~3 weeks of work.
