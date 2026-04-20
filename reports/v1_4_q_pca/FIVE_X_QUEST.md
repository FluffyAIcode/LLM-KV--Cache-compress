# Pursuing ≥5× compression at ACCEPT quality (Qwen2.5-0.5B, D=64)

**Date.** 2026-04-17
**Branch.** `cursor/v1-3-rsvd-rope-aware-12f5`

## Starting point

After v1.4 Sprint 1 (Q-preconditioned PCA), the best ACCEPT cell was
2.06× compression at Δppl = −0.56 %. The request: push to the v1.3
paper's flagship 5.8× while keeping ACCEPT quality.

## Where the v1.3 paper's 5.8× comes from

`reports/v1_3_rsvd_rope/FLAGSHIP_COMPARISON.md`: the 5.8× number was
measured on **128k-context × head_dim≥128 flagship models**
(Qwen3-235B, DeepSeek-V3.1, GLM-4.6, etc.) under tier-1 recipe:
`b=2, RSVD target_rank=D/2, share_basis`. Three variables stacked:

1. Long context (128k) amortises skeleton bytes across hundreds of
   blocks per layer.
2. Large head dim (D=128-192) reduces skeleton cost as a fraction of
   per-block codes.
3. Aggressive recipe: 2-bit residuals, RSVD rank cap, shared basis.

At Qwen2.5-0.5B (D=64, ctx=1024) the ratio ceiling is structurally
lower, and the rank cap has much more room to damage quality.

## Exploration (all runs on Qwen2.5-0.5B, ctx=1024, 2 WikiText passages)

### 1. The v1.3 tier-1 recipe with Q-precondition

| recipe                                    | ratio | Δppl    | top-1   |
|-------------------------------------------|------:|--------:|--------:|
| bs=512 b=2 RSVD r=32 share Q-precond OFF  | 5.80× | +228.89 %| 50.00 % |
| bs=512 b=2 RSVD r=32 share Q-precond ON   | 5.80× |**+105.47 %**| 59.52 % |

Q-precond halves the PPL inflation, but REJECT remains at the flagship
recipe.

### 2. Aggressive-recipe ablation (RSVD + share, 18 cells × OFF/ON)

Mean Δppl over the grid:
- **OFF: +175.05 %**
- **ON:  +79.13 %**  (−95.9 pp improvement)

Q-precond dramatically reduces the damage of aggressive recipes, but
the minimum ON Δppl is +49.80 % at bs=256 b=4 vr=1.0 share — still
REJECT.

### 3. Decomposing the aggression axes at (bs=512, b=3, vr=1.0, Q-precond ON)

| pca          | share       | Δppl      |
|--------------|-------------|----------:|
| exact        | per_block   | +14.62 %  |
| exact        | layer-shared| **+8.26 %**  |
| randomized   | per_block   | +41.51 %  |
| randomized   | layer-shared| +66.97 %  |

**Under Q-precond, layer-shared basis flips from net negative to net
positive** (previously +20 pp in v1.3, now −6 pp). Rationale: after
whitening, every block of a layer lives in the same Σ_q-correct
coordinate frame; shared basis now picks up a genuinely shared signal
rather than averaging heterogeneous blocks.

**RSVD remains a heavy penalty (+33 pp to +59 pp vs exact)** even under
Q-precond. Q-preconditioning the input doesn't rescue RSVD's rank-cap
approximation error.

### 4. Tight variance-ratio (forcing low rank with exact PCA)

Sweep vr ∈ {0.3, 0.5, 0.7, 0.95} at bs ∈ {512, 1024}, b ∈ {2, 3, 4},
exact + share + Q-precond ON. Every tight-vr cell is catastrophic:

```
bs=512 b=3 vr=0.3 share Q-on  →  Δppl = +954.56%
bs=512 b=3 vr=0.5 share Q-on  →  Δppl = +1303.22%
bs=512 b=3 vr=0.7 share Q-on  →  Δppl = +279.02%
bs=512 b=3 vr=0.95 share Q-on →  Δppl = +218.50%
bs=512 b=3 vr=1.0 share Q-on  →  Δppl = +11.32%
```

The variance-ratio trick that works on synthetic Gaussian data (where
spectrum is flat) does not work on real K cache (where spectrum is
heavy-tailed, so small vr drops crucial directions).

### 5. Hard exact rank cap (NEW: `--exact-rank-cap N` CLI flag)

The v1.3 5.8× was achieved by RSVD's `target_rank=32`. What if we use
the **same rank budget with exact PCA** (no RSVD approximation)?

Added `CodecParams::exact_rank_cap: Option<usize>` + CLI flag
`--exact-rank-cap`. At bs=512, b=2, rank_cap=32, share, Q-on:
**ratio=5.80×, Δppl=+91.21 %**.

At bs=512, b=3, rank_cap=32, share, Q-on:
**ratio=4.91×, Δppl=+45.80 %**.

At bs=512, b=3, rank_cap=24, share, Q-on:
**ratio=5.11×, Δppl=+118.58 %**.

**Exact PCA with hard rank cap reproduces RSVD's ratios** but incurs
the same quality damage, confirming the quality loss is from the rank
cap itself, not from RSVD's approximation error.

## Why D=64 has a structural ceiling

For typical K-cache singular-value spectra decaying like $\sigma_k \sim 1/k$:

| variance captured | d_eff needed |
|-------------------|-------------:|
| 90.0 %            |  6           |
| 95.0 %            | 10           |
| 99.0 %            | 31           |
| 99.5 %            | 42           |
| **99.9 %**        | **58**       |
| **~100 %**        | **64**       |

At D=64, **preserving the attention-salient 99.9 % of the spectrum
requires d_eff ≥ 58** — 90 % of the full rank. Any rank cap below ~50
excises long-tail directions that collectively contribute materially
to attention logits.

By contrast, at D=128-192 (flagship models), the same 99.9 % threshold
needs d_eff ≈ 0.9·D — absolutely more directions, but the fractional
rank cost is lower because the skeleton-byte denominator is larger.
This is why 5.8× compression is reachable at flagship scale but has a
hard ceiling at Qwen2.5-0.5B scale.

## Honest Pareto frontier (Qwen2.5-0.5B, pre-RoPE, Q-precond ON)

```
  recipe                                   ratio     Δppl      top-1
  ─────────────────────────────────────    ──────    ──────    ──────
  exact, per_block, b=4 vr=1.0 bs=512      2.06×    −0.56 %   92.86 %  ← ACCEPT
  exact, per_block, b=3 vr=1.0 bs=512      2.36×    +3.32 %   84.92 %  (1 pp over ACCEPT, MARGINAL)
  exact, per_block, b=2 vr=1.0 bs=512      2.77×   +35.40 %   73.02 %  REJECT
  exact, share,     b=3 vr=1.0 bs=512      3.04×    +8.26 %   85.71 %  MARGINAL
  exact, share,     b=2 vr=1.0 bs=512      3.76×   +36.89 %   73.81 %  REJECT
  exact, rank_cap=32, share, b=2 bs=512    5.80×   +91.21 %   61.11 %  REJECT
  RSVD r=32, share, b=2 bs=512             5.80×  +105.47 %   59.52 %  REJECT
```

**Best ACCEPT cell on this model: 2.06× at Δppl=−0.56 %.**
**Best MARGINAL cell: 3.04× at Δppl=+8.26 % (Q-precond + share).**
**5× with ACCEPT is not reachable at D=64 under Q-preconditioned Kakeya.**

## Two paths to 5× ACCEPT on this model

Both are architecture-level, not parameter-tuning:

### Path A: Calibration / fine-tuning
As sketched in the prior "Tier 1/2/3" design note. Concretely for the
rank-cap regime: add a per-layer, per-block residual corrector
trained with MSE loss after the rank-cap PCA.  A 32×64 fp16 correction
matrix per block would add ~4 KB per block (negligible at bs=512) and
should recover most of the rank-truncation damage since the target is
purely linear.  Cost: ~30 min offline calibration on ctx=4k data,
zero Rust changes. Expected effect: Δppl at bs=512 b=2 rank_cap=32
from +91 % to ~+10 % (projecting from KIVI/KVQuant literature).

### Path B: Test at flagship scale
The architecture *is* designed for D≥128 and long context, and the
published 5.8× on flagship models was measured at that scale with the
current code.  We haven't re-tested flagship models with Q-precond.
If the 5.8× + Q-precond combo is close to ACCEPT on D=128+ models
(which is the more plausible scenario given the d_eff arithmetic
above), then the "ACCEPT at 5×" claim is a scale claim, not a
parameter claim.

## Recommendation

Neither tuning path within the current architecture gets Qwen2.5-0.5B
to 5× ACCEPT — this is structural for D=64. The choices are:

1. **Accept 2× ACCEPT at D=64** and pitch the architecture as
   scaling to 5-6× at D≥128 (which is where production deployment
   actually happens — e.g. Qwen3-235B, DeepSeek, etc.)
2. **Do Path A** (affine corrector post rank_cap). This is the
   cheapest high-impact direction within the "no weight changes"
   invariant.
3. **Test at flagship scale directly** to establish whether the
   5.8× + Q-precond combination is already ACCEPT at D=128+.

Path 3 is the cheapest and highest-information next step.

## Artefacts

- `benchmarks/e2e_ppl_pre_rope.py` — `--exact-rank-cap N` CLI flag
- `kakeyaturbo/src/{pca,codec}.rs` — `exact_rank_cap: Option<usize>`
  in `CodecParams`, honoured by `fit_weighted_pca_with_storage_capped`
- `kakeyaturbo/src/bin/kakeyaturbo-bench.rs` — `--exact-rank-cap` flag
- `reports/v1_4_q_pca/aggressive_ablation/` — full 18×OFF/ON grid
- `reports/v1_4_q_pca/ratio5x_search/` — small-vr search (every cell REJECT)
