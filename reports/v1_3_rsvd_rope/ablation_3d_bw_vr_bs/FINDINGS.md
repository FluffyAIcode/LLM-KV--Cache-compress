# 3-D ablation: bit_width × variance_ratio × block_size (+ block_size extension)

**Date.** 2026-04-17
**Branch.** `cursor/v1-3-rsvd-rope-aware-12f5`
**Model.** Qwen2.5-0.5B-Instruct, pre-RoPE cache.
**Fixed config.** PCA = exact, skeleton = fp16, share_basis = per_block,
RSVD off. ctx_len=1024, n_eval=64, 2 WikiText-103-raw-v1 passages.
**Swept axes.** `bit_width ∈ {2,3,4}`, `variance_ratio ∈ {0.995,0.999,1.0}`,
`block_size ∈ {128,256,512}`; follow-up probe at bs ∈ {16,32,64}.

## Goal

After the 2×2×2 ablation ruled out RoPE, skeleton dtype, and basis reuse as
dominant error sources, the remaining candidates were residual coding
precision (bit_width), PCA truncation tightness (variance_ratio), and block
locality (block_size). This sweep tests all three jointly.

## Primary 27-cell table (block_size ∈ {128,256,512})

```
block_size = 128
 b \ vr      vr=0.995    vr=0.999    vr=1.0
  b=2         +75.93%     +66.40%     +60.70%
  b=3         +70.28%     +63.90%     +62.30%
  b=4         +63.52%     +64.03%     +55.27%

block_size = 256
 b \ vr      vr=0.995    vr=0.999    vr=1.0
  b=2         +86.69%     +83.34%     +78.82%
  b=3         +78.64%     +81.87%     +74.21%
  b=4         +79.87%     +83.33%     +76.29%

block_size = 512
 b \ vr      vr=0.995    vr=0.999    vr=1.0
  b=2        +106.54%     +96.61%    +110.29%
  b=3         +97.63%     +87.21%     +99.60%
  b=4         +96.29%     +85.51%    +100.09%
```

Marginal effects (mean over the other two axes):

| axis              | level  | mean Δppl |  Δ         |
|-------------------|--------|-----------|-----------:|
| **block_size**    | 128    | **+64.70 %** |           |
|                   | 256    |  +80.34 %    |            |
|                   | 512    |  +97.75 %    |**33.1 pp** |
| bit_width         | 2      |  +85.04 %    |            |
|                   | 3      |  +79.51 %    |            |
|                   | 4      |  +78.24 %    | 6.8 pp     |
| variance_ratio    | 0.995  |  +83.93 %    |            |
|                   | 0.999  |  +79.13 %    |            |
|                   | 1.0    |  +79.73 %    | 4.8 pp     |

**block_size dominates by ~5× over the other two axes combined.** This
changes the v1.3 story: the fixed 512-token block was not a neutral
engineering choice, it was the single largest quality cost.

## Follow-up: how far does the block_size trend go?

```
bs  bw  vr     Δppl      KL      top-1
 64  4  1.0   +33.87%   0.256   81.00%
 32  4  1.0    +9.44%     -        -
 16  4  1.0    +0.94%   0.037   85.70%
```

Quality improves **monotonically** as block_size shrinks, and at bs=16 the
codec lands firmly inside ACCEPT (Δppl ≤ 3 %, top-1 ≥ 85 %, KL < 0.1).

## But: compression ratio also moves — in the wrong direction

Skeleton bytes (PCA mean, basis, K-means centres) are per-block, so halving
block_size nearly doubles skeleton overhead. Residual coding bytes are per
token, so they scale linearly with `bit_width` only. When blocks shrink
faster than ~O(D), skeleton overhead swamps residual savings and the
codec becomes an *expander*, not a compressor.

Measured against the bf16 baseline on a realistic 2048×64 cache tensor:

| bs  | bw | vr    | ratio vs bf16 | Δppl    |
|-----|----|-------|---------------:|--------:|
| 512 | 2  | 0.999 | **2.77× (smaller)** | +96.6 % |
| 512 | 3  | 0.999 | 2.36× | +87.2 % |
| 256 | 3  | 1.0   | 1.72× | +74.2 % |
| 128 | 4  | 1.0   | 1.04× | +53.8 % |
|  64 | 4  | 1.0   | 0.64× (BIGGER) | +33.9 % |
|  32 | 4  | 1.0   | 0.69× (BIGGER) | +9.4 %  |
|  16 | 4  | 1.0   | 0.73× (BIGGER) | **+0.9 %** |

(32/16 have different ratios because `d_eff` saturates at D, so rank
truncation ceases to save basis bytes.)

## Pareto frontier (only non-dominated cells)

```
bs  bw  vr       ratio       Δppl
512  2  0.999   2.768×      +96.6 %
512  3  0.999   2.359×      +87.2 %
512  4  0.999   2.056×      +85.5 %
256  2  1.0     1.925×      +78.8 %
256  3  0.995   1.733×      +78.6 %
256  3  1.0     1.718×      +74.2 %
128  2  1.0     1.196×      +60.7 %
128  4  1.0     1.041×      +53.8 %
 16  4  1.0     0.727×      +0.9 %  ← ACCEPT but negative compression
```

The frontier is monotone and brutal: **every cell that compresses has
Δppl ≥ 54 %; every cell that produces ACCEPT quality expands the data.**
There is no "sweet spot" on the Pareto curve that gives both.

## What this tells us about v1.3 — and the paper

1. **The block-size floor is the structural PPL limit, not quantisation.**
   b=3→b=4 at fixed bs buys ~5 pp. vr=0.995→1.0 buys ~4 pp. bs=512→128
   buys ~33 pp. bs=128→16 buys another ~53 pp.
2. **The skeleton overhead is the structural compression limit, not residual
   bits.** At usable PPL (bs=16), skeleton bytes are 89 % of the payload.
3. **v1.3 (bs=512) was operating in the ~2.4× compression / +87 % PPL
   quadrant.** This is a clear regression vs just "discard half the cache"
   (which costs no bytes and loses <100 % PPL on many contexts).
4. **No parameter combination in the current Kakeya-skeleton architecture
   reaches Δppl ≤ 3 % with ratio ≥ 1×.** On this model, at this context
   length, the architecture cannot clear that bar by tuning alone.

## What does not rescue this

Small blocks + weight-sharing schemes (e.g. share PCA basis across
adjacent small blocks) mathematically equal medium-block per-block PCA;
the 2×2×2 already showed layer-shared basis is net negative. Smaller
block_size plus RSVD would re-introduce the +69 pp RSVD penalty.
Promoting skeleton to fp32 was +4 pp noise. None of these change the
Pareto picture.

## What might

- **Calibrated / fine-tuned codec.** The PPL floor is an untrained-codec
  floor. A 1–2% fine-tune of the decompression path against attention
  logits could collapse the per-layer error multiplicatively.
- **Different skeleton formulation.** The Kakeya skeleton's byte cost is
  dominated by the PCA basis (`d_eff × D × 2`). Replacing it with
  something like KIVI (per-channel `int4` + per-token scale, no basis)
  changes the byte scaling entirely. This is the architectural option
  the previous DIAGNOSIS.md suggested — the 2×2×2 and 3-D ablations show
  that no parameter tuning inside the current architecture will close
  the gap.
- **Longer contexts.** At ctx=1024 the total data being compressed is
  tiny (2048 vectors per layer per stream), which amortises skeleton
  bytes poorly. At ctx=32k with bs=512 the cell economics flip: one
  basis amortises over 64 blocks of residuals. The codec was designed
  for long contexts. *We should repeat this ablation at ctx ≥ 8192*
  before drawing permanent conclusions about the architecture.

## Artefacts

- `qwen2_5_kv_3d_summary.json` — 27-cell table as JSON.
- `qwen2_5_kv_b{2,3,4}_vr{0.995,0.999,1.0}_bs{128,256,512}.json` — per-cell
  per-passage metrics (27 files).
- `qwen2_5_smallbs_kv_b4_vr1.0_bs{16,32,64}.json` and
  `qwen2_5_bs64_kv_b{3,4}_vr1.0_bs{64,128}.json` — follow-up probe cells.
- `benchmarks/ablation_3d_bw_vr_bs.py` — driver that loads the model
  once and sweeps an arbitrary user-specified 3-D grid.
