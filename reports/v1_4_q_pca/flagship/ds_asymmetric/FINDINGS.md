# Asymmetric K/V + systematic boundary skip — v1.4 Sprint 3

**Date.** 2026-04-17
**Branch.** `cursor/v1-3-rsvd-rope-aware-12f5`
**Model.** DeepSeek-R1-Distill-Qwen-1.5B (D=128, 28 layers, flagship proxy)
**Config.** pre-RoPE cache, `ctx=2048`, `n_eval=64`, 2 WikiText passages.
**Streaming-safe:** exact PCA, per_block (no `share_basis`), `bs=1024`,
`skip_layers=[0, 1, 26, 27]` (systematic first-2 + last-2 boundary policy),
`skeleton_dtype=fp16`, Q-preconditioning ON for K only (V has no static
Σ equivalent).

## The 4-cell asymmetric grid + symmetric reference

| b_K | b_V | total ratio | Δppl     | top-1    | KL     | verdict  |
|----:|----:|------------:|---------:|---------:|-------:|:--------:|
| 3   | 2   | 2.97×       | +4.72 %  | 86.51 %  | —      | MARGINAL |
| 3   | 3   | 2.72×       | +2.33 %  | 84.13 %  | 0.051  | MARGINAL |
| **4** | **2** | **2.72×** | **+2.44 %** | **89.68 %** | **0.026** | **ACCEPT** |
| 4   | 3   | 2.50×       | +2.07 %  | 95.24 %  | 0.020  | ACCEPT   |
| **4** | **4** | **2.32×** | **−1.54 %** | **88.89 %** | 0.026 | ACCEPT (ref) |

**New best ACCEPT cell: `b_K=4 b_V=2` at 2.72× compression with
Δppl = +2.44 %, top-1 = 89.68 %.**

This is a **+17 % compression gain over the previous symmetric
best (2.32×)**, at the same ACCEPT quality level.  `b_K=4 b_V=3` gives a
safer top-1 (95.24 %) but only 2.50× compression.

## Streaming-safety

All five cells use **per-block PCA** (no `--share-basis`), so they
respect the paper's §1 streaming contract: each block's PCA fit
triggers as soon as the block is full, no dependency on future blocks.
The K stream's Q-preconditioning adds a `L` (Cholesky factor) matmul
before encode and `L⁻¹` matmul after decode, which is per-vector —
also streaming-safe.  The systematic `skip_layers=[0, 1, 26, 27]`
boundary policy is a compile-time decision, not runtime-dependent.

## Where the gain comes from — and where it stops

Per-stream byte breakdown at `bs=1024, bw=2`:

```
  b=2 V-stream bytes:  skeleton = 145 KB (46 %) + codes = 168 KB (54 %) → 313 KB → 3.27x
  b=3 V-stream bytes:  skeleton = 145 KB (38 %) + codes = 232 KB (62 %) → 377 KB → 2.72x
  b=4 V-stream bytes:  skeleton = 145 KB (33 %) + codes = 296 KB (67 %) → 441 KB → 2.32x
```

At `b_V = 2`, **skeleton bytes are 46 % of the V stream**.  No matter
how aggressive the residual quantizer gets, the per-block PCA mean + basis
(stored in fp16) is a floor that the current architecture cannot cross.

TurboQuant, by contrast, has zero skeleton — it's a per-vector
quantizer (WHT rotation + Lloyd-Max on coordinates, no PCA).  Its V
stream at `b=2` gets 7.11× per-vector vs our 3.27× per-stream.  The gap
is almost entirely skeleton bytes.

## The two levers still untouched

### 1. Cross-layer skeleton sharing for V
At ctx = 2048, V has 4 blocks per layer (for `bs=512`) or 2 blocks per
layer (`bs=1024`).  With 28 layers the total V corpus is 64 × 28 = 1792
vectors per kv-head across all layers, but we fit 28 independent PCAs.
A layer-pooled PCA fit for V could amortise the 145 KB skeleton across
all 28 layers, saving ~4 MB per kv-head in a 128k-ctx deployment.

**Streaming interaction:** requires freezing the V basis at prefill-end
or doing an online updating basis. Not pure per-block but decode-time
is O(1) (pick already-computed basis).

### 2. Replace V codec with TurboQuant
V has **no softmax non-linearity** — attention consumes V as
`p^T V` with fixed p.  Per-vector MSE noise on V adds linearly to the
attention output, it does not get amplified by softmax saturation.
This is why the TurboQuant+ production finding "V compression is free"
exists.  Our per-vector noise measurement on V (from the earlier
TurboQuant probe) showed rel_err ≈ 20 % at b=3, which is modest and
bounded.

If we route V through `turboquant_v_roundtrip` (PolarQuant only,
per-vector, no PCA, no skeleton), V at b=2 gets ~7.1× compression.
Combined with K at b_K=4 + Q-precond + skip + KakeyaTurbo
(2.32× on K stream), total ratio would be:

```
bf16 total = 2 × 1024 KB = 2048 KB
K (KakeyaTurbo b=4) = 441 KB
V (TurboQuant b=2)  = 144 KB (per-vector compression on 4096 × 128 vectors)
total              = 585 KB
total ratio         ≈ 3.50×
```

**Projected 3.5× ACCEPT** on DeepSeek D=128, assuming V-only TurboQuant
doesn't cause unexpected K/V cross-talk.  This needs to be measured.

## Verdict

v1.4 Sprint 3 (asymmetric K/V + systematic boundary skip) delivers an
honest streaming-safe ACCEPT cell at **2.72× compression, Δppl = +2.44 %,
top-1 = 89.68 %** on the flagship proxy.  This is a genuine Pareto
expansion over the 2.32× symmetric reference.

The remaining gap to TurboQuant+ production's 4.6× is driven almost
entirely by our V-stream skeleton overhead (46 % of V bytes at b=2).
The two architectural levers above (cross-layer V skeleton sharing,
TurboQuant V) would close that gap.

## Files

- `benchmarks/e2e_ppl_pre_rope.py` — `--bit-width-v` CLI flag (asymmetric K/V)
- `reports/v1_4_q_pca/flagship/ds_asymmetric/` — 4 per-cell JSONs

## Reproduce (any one cell)

```bash
python3 benchmarks/e2e_ppl_pre_rope.py \
    --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
    --model-name ds_asym_bK4_bV2 \
    --ctx-len 2048 --n-eval 64 --n-passages 2 \
    --block-size 1024 --bit-width 4 --bit-width-v 2 \
    --pca-method exact --variance-ratio 1.0 \
    --skeleton-dtype fp16 --compress kv \
    --q-precondition reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors \
    --q-precond-skip-layers 0 1 26 27 \
    --prefill-chunk 1024 --skip-sanity \
    --out-dir reports/v1_4_q_pca/flagship/ds_asymmetric
```
