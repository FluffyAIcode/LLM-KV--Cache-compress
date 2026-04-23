# Head-to-head: Kakeya v1.3 PPL vs TurboQuant k8v4 on Qwen3-4B + vLLM

All numbers on H200 80GB, vLLM v1 engine (`VLLM_ENABLE_V1_MULTIPROCESSING=0`),
bf16 weights, WikiText-103 test split, ctx=2048, n_eval=64, 4 passages,
same passage-selection + same eval window as
`benchmarks/e2e_ppl_validation_vllm_snapshot_qwen3.py`.

PPL is `mean(exp(−logp[gold]))` over the 64-position eval window.
Both bf16 and TQ run the harness in
`benchmarks/snapshot_protocol_tq_vs_bf16.py` which drives a single
prompt_logprobs forward per passage.  Kakeya runs the existing
snapshot harness (two-pass: clean prefill → offline codec → alt
prefill); its `ppl_ref` per passage matches bf16 exactly
(e.g. `median_ppl = 7.933` both sides), confirming the protocols
are aligned.

## Summary table

| Metric                            | bf16 baseline | TurboQuant k8v4 | `v1.3-GPU-snapA` | `v1.3-GPU-snapB` |
|:----------------------------------|:-------------:|:---------------:|:---------------:|:---------------:|
| PPL (mean over passages)          |       11.836  |         11.847  |         22.982  |         22.537  |
| PPL (median over passages)        |        7.933  |          7.909  |         10.954  |         12.395  |
| Δppl vs bf16 (mean)               |           —   |       **+0.09%** |       +94.17%   |       +90.41%   |
| Δppl vs bf16 (paired snapshot)    |           —   |       **+0.09%** |       +61.84%   |       +65.98%   |
| top-1 agreement                   |        100%   |        ~100%    |        79.30%   |       **81.64%** |
| K reconstruction MSE (abs)        |           0   |      **0.0048** |        0.503    |        0.503    |
| K relative MSE (MSE / var(K))     |           0   |     **0.0002**  |        ~0.02    |        ~0.02    |
| V reconstruction MSE (abs)        |           0   |       **0.0525** |        0.0517   |        0.0517   |
| V relative MSE (MSE / var(V))     |           0   |       **0.0206** |        ~0.02    |        ~0.02    |
| Compression ratio (per token)     |        1.00×  |       **2.61×** |         2.20×   |         2.00×   |
| Compression (blended w/ 14-layer bf16 skip) |  1.00×  |       **2.61×** |         1.50×   |         1.44×   |
| Forward time per passage (pass-2) |       0.134 s |       0.148 s  |       **0.064 s** |     0.064 s   |
| Pass-1 (clean forward)            |           —   |          —     |       0.342 s   |     0.342 s   |
| Offline codec time                |           —   |          —     |      **18.08 s** |    18.10 s   |
| Peak GPU memory (all 4 passages)  |      56.65 GiB |      56.63 GiB |      ~56.7 GiB  |    ~56.7 GiB  |

## Kakeya configuration definitions

`v1.3-GPU-Qwen-snap-bK64-bdry14` (spoken: **`v1.3-GPU-snapA`**,
Δppl-optimal):
```
--bit-width-k 4 --k-kmeans-k 64 --rsvd-target-rank-factor 0.75
--bit-width-v 2 --v-kmeans-k 16
--boundary-skip-layers 0 1 2 3 4 5 6 29 30 31 32 33 34 35
--gpu-codec --no-share-basis-v
--disable-q-precond --disable-centroids --disable-outlier
```

`v1.3-GPU-Qwen-snap-bK128-bdry14` (spoken: **`v1.3-GPU-snapB`**,
top-1-optimal): same as snapA but with `--k-kmeans-k 128`.

See `NAMING.md` in this directory for the canonical-name schema
and the full alias table.

## Reading

**1. TurboQuant k8v4 is effectively lossless on this model +
workload.**  Δppl = +0.09 % is at 4-passage variance level.  K
int8 per-token quantisation keeps >99.98 % of K variance (rel
MSE = 2e−4).  V int4 loses ~2 % of variance but this doesn't
propagate to visible PPL cost — softmax de-amplifies V error.

**2. `v1.3-GPU-snapA / snapB` lose 60-94 pp Δppl** and have K
reconstruction MSE **two orders of magnitude worse** than TQ
(0.503 vs 0.0048).  V MSE is comparable (0.052 vs 0.053).

**3. Compression-wise TQ wins, not Kakeya.**  TQ's `1024 + 512 +
32 bytes = 2.61×` beats Kakeya's best `2.20×`.  After 14-layer
boundary skip the blended Kakeya number drops to 1.50× — we're
paying 2 pp of PPL-recovery dollars by keeping 39% of layers
entirely bf16.

**4. Wall-clock: apples-to-oranges.**  TQ does per-token
quantisation in the CUDA attention kernel — ~14 ms additional per
passage (0.148 vs 0.134 = 10 %).  Kakeya's 18 s offline codec is
a Scenario-A cost (post-prefill cache compression); it doesn't
pay again at decode time.  For a workload with long decode after
a single prefill, Kakeya's one-time codec amortises; for many
short prompts, TQ wins overwhelmingly.

**5. Memory: head-room undetectable at ctx=2048.**  Both dtypes
sit near the 56 GiB pre-set `gpu_memory_utilization=0.40` ceiling
because of model weights and prefill activations.  The compression
delta only shows up at long contexts (the cache budget starts
mattering past a few × 10k tokens).

## Bottom line on Qwen3-4B

**TurboQuant k8v4 dominates the current Kakeya v1.3 PPL recipe on
every axis we measured except top-1 agreement** (`v1.3-GPU-snapB`'s
81.64% is 17 pp behind TQ's ~100%, so this isn't actually a win
either).  For this model, TQ k8v4 is the correct production
choice.

Where Kakeya is *supposed* to matter is at **more aggressive
compression ratios (4×+)** where per-token quantisation
collapses.  Our current recipes sit at 2.0–2.6×, right in TQ's
comfort zone.  A fair comparison of the two codecs would pin both
at 4×+ compression and measure Δppl there — Kakeya's RSVD
skeleton + Lloyd-Max residual is designed for that regime, not
the 2× one we've been tuning in.

## Report sources

- `headtohead/auto_snapshot_protocol.json` — bf16 PPL on the
  snapshot protocol.
- `headtohead/turboquant_k8v4_snapshot_protocol.json` — TQ PPL +
  forward time on the same protocol.
- `budget_sweep/qwen3_4b_budget_k64_bK4_deff96_vllm_snapshot.json`
  — `v1.3-GPU-snapA` (kakeya best Δppl).
- `budget_sweep/qwen3_4b_budget_kK128_vllm_snapshot.json` —
  `v1.3-GPU-snapB` (kakeya best top-1).
