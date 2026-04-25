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
| `run_stage075_real_weights.py` | end-to-end driver: host hidden → V4 KV streams (trained weights) → non-Gaussian audit + KakeyaLattice / FP8 codec comparison |
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

## Headline finding (2026-04-25 H200 run, TRAINED V4-Flash weights)

E8 Q=38 vs FP8 per-64-block across three V4 KV streams:

```
stream                  E8/FP8 rel-MSE   bit savings
sliding_window_kv       0.786            -22.0%       ← strong Pareto win
csa_pool_kv_ratio4      0.902            -22.0%       ← moderate Pareto win
hca_pool_kv_ratio128    0.966            -22.0%       ← marginal Pareto win
mean                    0.884            -22.0%
```

**~22% bit savings with 12% lower MSE on average.** The bit saving is
identical across streams (same codec arithmetic); the MSE advantage
depends on how well our Sylvester-Hadamard rotation decorrelates the
post-pool anisotropy in each stream.

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
