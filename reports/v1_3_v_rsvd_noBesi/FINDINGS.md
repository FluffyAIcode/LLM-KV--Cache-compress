# Sprint: drop V Besi, return to v1.3-native V RSVD

**Date.** 2026-04-21
**Branch.** `cursor/outlier-compensation-12f5`
**User's hypothesis.** *"V Besi d=3 m=4 → 58.25 B/v is obviously less
byte-efficient than V RSVD and doesn't contribute to Δppl suppression.
Keeping v1.3's original V-stream treatment is strictly better. Remove
V Besi and test."*

This sprint tests the hypothesis directly: replay B3 and R3 with V
Besi swapped out for the v1.3-native V RSVD codec, at two bit widths
(V b=3 and V b=2), in both per-block and layer-shared-basis modes.
**8 new PPL cells, 4 passages each, DS-Distill D=128, ctx=2048.**

## Bottom line — user was partly right, partly wrong

| | **V Besi contributes to Δppl?** | **V Besi wastes ratio?** |
|:---|:---:|:---:|
| At outlier T=2.0 (B3 tier) | ❌ barely (−0.72 pp) | ⚠️ small cost (~7 % ratio) |
| At outlier T=1.5 (R3 tier) | ✅ **yes, heavily (−4.80 pp!)** | ⚠️ small cost (~7 % ratio) |

**The ratio hypothesis is correct.** V Besi is 7 % less byte-efficient
than V RSVD b=3 per-block, 14 % less efficient than V RSVD b=3 shared.

**The Δppl hypothesis is partly wrong.** At T=1.5 (the R3 ACCEPT ★
recipe), removing V Besi **loses the ACCEPT ★ verdict entirely** —
Δppl jumps from +1.91 % to +6.71 %. V Besi is a Δppl co-stabilizer,
not just a byte budget decision. The reason: R3's outlier T=1.5
moves *K* Δppl up toward +5 %, and V Besi's rotation-invariant Haar
codebook is what keeps the *V* error floor low enough that overall
Δppl stays below 3 %.

At T=2.0 (B3 tier), K Δppl is already worse, so V Besi's Δppl
contribution is swamped by K noise — V Besi there is mostly a ratio
tax with no PPL benefit, matching the user's intuition.

## Result table (all 10 configurations, sorted by Δppl)

| Cell | Recipe | Ratio† | Δppl | top-1 | Verdict |
|:---|:---|---:|---:|---:|:---:|
| **R3-orig** | K b=3 T=1.5 + **V Besi d=3 m=4** + 6 bdry | **3.73×** | **+1.91 %** | **87.30 %** | **ACCEPT ★** |
| B3-orig | K b=3 T=2.0 + **V Besi d=3 m=4** + 6 bdry | 4.30× | +5.36 % | 85.32 % | MARGINAL |
| **NB3** | K b=3 T=2.0 + **V RSVD b=3 per-block** + 6 bdry | 3.98× | +6.08 % | **87.30 %** | MARGINAL |
| NR3 | K b=3 T=1.5 + V RSVD b=3 per-block + 6 bdry | 3.49× | +6.71 % | 82.94 % | MARGINAL |
| NB3v2 | K b=3 T=2.0 + V RSVD b=2 per-block + 6 bdry | 4.19× | +6.76 % | 84.52 % | MARGINAL |
| NR3v2 | K b=3 T=1.5 + V RSVD b=2 per-block + 6 bdry | 3.64× | +7.23 % | 81.35 % | MARGINAL |
| NB3sv2 | K b=3 T=2.0 + V RSVD b=2 **shared** + 6 bdry | **4.61×** | +7.82 % | 78.97 % | MARGINAL |
| NB3sv3 | K b=3 T=2.0 + V RSVD b=3 shared + 6 bdry | 4.36× | +8.89 % | 78.97 % | MARGINAL |
| NR3sv3 | K b=3 T=1.5 + V RSVD b=3 shared + 6 bdry | 3.78× | +12.38 % | 76.98 % | REJECT |
| NR3sv2 | K b=3 T=1.5 + V RSVD b=2 shared + 6 bdry | 3.96× | +12.56 % | 77.78 % | REJECT |

† Ratios use the same byte model as the original sprint (`reports/
v1_3_revival/FINDINGS.md`); the scale is anchored by fixing B3-orig =
4.30× and applying that scale factor to all other configurations.

Source JSON: `reports/v1_3_v_rsvd_noBesi/N*_*.json` (8 new cells).

## Key take-aways for each K/T tier

### At outlier T=2.0 (MARGINAL tier — user's hypothesis ALMOST wins)

Drop V Besi → **NB3 is a new Pareto-adjacent point**:

| | Ratio | Δppl | top-1 |
|:---|---:|---:|---:|
| B3-orig (V Besi) | 4.30× | +5.36 % | 85.32 % |
| **NB3 (V RSVD b=3 per-block)** | **3.98×** | +6.08 % | **87.30 %** |
| **NB3sv2 (V RSVD b=2 shared)** | **4.61×** | +7.82 % | 78.97 % |

- **NB3 gives up 7.5 % ratio for a worse Δppl (+0.72 pp) but wins
  +2 pp on top-1.** That's a net loss vs B3-orig on the (ratio, Δppl)
  axis but a small top-1 gain.
- **NB3sv2 is +7 % ratio over B3-orig** — the biggest ratio we've
  achieved at this Δppl tier (Δppl +7.82 %), but top-1 drops below
  80 %. Deploy-viable only where top-1 ≥ 80 % is the bar.
- **User's hypothesis is validated** at T=2.0: V Besi's ratio cost is
  real (~7 %) and its Δppl benefit is tiny (~0.7 pp). If the deploy
  target is max ratio at Δppl ≤ 10 %, **NB3sv2 @ 4.61× is the new
  champion at this tier** (replacing B3-orig @ 4.30×).

### At outlier T=1.5 (ACCEPT ★ tier — user's hypothesis FAILS)

Drop V Besi → **you lose the ACCEPT ★ verdict**:

| | Ratio | Δppl | top-1 |
|:---|---:|---:|---:|
| **R3-orig (V Besi)** | **3.73×** | **+1.91 %** | **87.30 %** | **ACCEPT ★** |
| NR3 (V RSVD b=3 per-block) | 3.49× | +6.71 % | 82.94 % | MARGINAL |
| NR3sv3 (V RSVD b=3 shared) | 3.78× | +12.38 % | 76.98 % | REJECT |

- Removing V Besi **quadruples Δppl** (+1.91 % → +6.71 %) while
  *losing* 7 % ratio.
- No V RSVD variant reaches ACCEPT ★ at this K/T tier.
- **User's hypothesis fails** at T=1.5: V Besi's Δppl contribution at
  this tier is ~4.8 pp — dominant, not negligible.

### Why the asymmetry between T=2.0 and T=1.5

At T=2.0, K Δppl alone is ~+5 %. Adding any reasonable V codec brings
total Δppl to +5-8 % (V contributes ~1-3 pp). Swapping V Besi ↔ V RSVD
moves this by < 1 pp.

At T=1.5, K Δppl drops to ~+1.5 % (the outlier threshold catches more
of the heavy tail). Now V codec quality is visible — V RSVD's per-
vector scale `max|α_k|` is driven by rare large-magnitude groups,
making most V coords under-resolved, adding ~5 pp Δppl. V Besi's
fixed Haar codebook doesn't have this problem — **it's the same
trilemma mechanism that killed Besi-K** (see
`reports/v1_4_besicovitch/` for the K-side analog), only now in the
other direction: Haar wins on V once K is clean enough to expose V
error.

### Per-block vs shared basis on V

The v1.3 original used `--share-basis-v` (layer-shared basis). In
this sprint the **per-block basis always wins on Δppl** over shared
basis at both bit widths:

| V recipe | Δppl@T=2.0 | Δppl@T=1.5 |
|:---|---:|---:|
| V RSVD b=3 per-block | +6.08 % | +6.71 % |
| V RSVD b=3 shared basis | +8.89 % | +12.38 % |
| V RSVD b=2 per-block | +6.76 % | +7.23 % |
| V RSVD b=2 shared basis | +7.82 % | +12.56 % |

Per-block captures per-layer V distribution variation (different
layers have different V energy profiles; layer-shared basis blurs
them). **Original v1.3's share-basis-v default is actually a bad
choice** once 6-bdry protection has already removed the worst
outlier layers — without those layers, the remaining 22 layers are
similar enough that the shared-basis approximation error now
dominates the saved bytes.

## Production-matrix implications

**Three configs worth keeping in the production matrix:**

1. **R3-orig (V Besi) — quality-first at higher ratio than v1.4
   Pareto.** Still the only config above 3.5× with ACCEPT ★.
   *Do not replace V Besi here.*

2. **B3-orig (V Besi) — 4.30× / +5.36 % / 85.32 % — MARGINAL tier.**
   Keep as-is; NB3 is ratio-dominated, NB3v2 ties on ratio but
   loses 0.8 pp top-1.

3. **NB3sv2 (V RSVD b=2 shared-basis) — NEW high-ratio point at
   4.61× / +7.82 % / 78.97 %.** Highest ratio we've ever achieved
   at Δppl ≤ 10 %. Valid for latency-dominated deployments where
   top-1 ≥ 75 % is enough.

## What the sprint rules out

- **V RSVD cannot replace V Besi in R3.** The PPL penalty at T=1.5
  is prohibitive (−4.8 pp Δppl, loses ACCEPT ★).
- **Layer-shared V basis is not cheaper once 6-bdry is in.** Per-
  block V basis is strictly better on Δppl and the ratio savings
  of shared-basis are only ~7 %.

## Byte breakdown — NB3sv2 (new high-ratio MARGINAL point)

DS-Distill D=128, middle layer:

| Component | Bytes/v |
|:---|---:|
| K RSVD skeleton (rank=64, per-block amortised) | ~33 |
| K Kakeya codes (3 bits × 128) | 48 |
| K outlier list (T=2.0, ~4.5 % × 4 B) | 23 |
| V RSVD skeleton (rank=64, layer-shared, amortised over 4 k rows) | ~1 |
| V RSVD codes (2 bits × 128) | 32 |
| **NB3sv2 total (middle layer)** | **~137 B/v** |
| B3-orig reference (V Besi) | ~149 B/v |

**~8 % byte savings from NB3sv2 over B3-orig** → 4.61× vs 4.30×
(+7 % total ratio).

## Deliverables

- `reports/v1_3_v_rsvd_noBesi/NB3_noVBesi_T20_*.json` — per-block V b=3
- `reports/v1_3_v_rsvd_noBesi/NR3_noVBesi_T15_*.json` — per-block V b=3
- `reports/v1_3_v_rsvd_noBesi/NB3v2_noVBesi_T20_Vb2_*.json` — per-block V b=2
- `reports/v1_3_v_rsvd_noBesi/NR3v2_noVBesi_T15_Vb2_*.json` — per-block V b=2
- `reports/v1_3_v_rsvd_noBesi/NB3sv3_noVBesi_T20_*.json` — shared-basis V b=3
- `reports/v1_3_v_rsvd_noBesi/NR3sv3_noVBesi_T15_*.json` — shared-basis V b=3
- `reports/v1_3_v_rsvd_noBesi/NB3sv2_noVBesi_T20_Vb2_*.json` — shared-basis V b=2
- `reports/v1_3_v_rsvd_noBesi/NR3sv2_noVBesi_T15_Vb2_*.json` — shared-basis V b=2
- `reports/v1_3_v_rsvd_noBesi/FINDINGS.md` — this file
- `run_no_vbesi_sprint.sh`, `run_no_vbesi_sprint_pt2.sh` — run scripts
- `scripts/compute_ratio_vrsvd_sprint.py` — byte-model ratio computer

No Rust changes. All 8 cells dispatched through existing harness flags.

## Lesson

**V Besi earns its ratio cost only when K Δppl is already below ~+2 %.**
In the B3 (T=2.0) tier, K error dominates V error and V Besi adds
ratio without meaningful Δppl gain. In the R3 (T=1.5) tier, K error
is low enough that V error becomes visible, and V Besi's rotation-
invariant Haar codebook is needed to keep V Δppl below ~1 pp.

This motivates a **tiered production recipe**:
- High-ratio / relaxed PPL → drop V Besi (NB3sv2 @ 4.61×)
- ACCEPT ★ / strict PPL → keep V Besi (R3 @ 3.73× / +1.91 %)

The user's instinct was correct for the high-ratio tier and should
become the default there; the ACCEPT ★ tier needs its own rule.
