# Bit-packed KakeyaLattice (D4 & E8) vs TurboQuant — REAL-byte iso-ppl
# Qwen3-4B on NVIDIA H200, 2026-06-15

> **Comparison standard (v1.6.1):** all codec-vs-codec comparisons are
> **(1) bit-packed** (real `kv_storage_bytes`, not int8) **and (2) iso-quality**
> — i.e. each codec is taken at the operating point that meets a fixed |Δppl|
> threshold, then we compare real bytes. **Never rank codecs by raw CR at
> unmatched bit budgets** (e.g. "TurboQuant b=4 = 3.77× vs E8 Q=38 = 2.37×" is
> meaningless — b=4 has |Δppl| ≈ 4.8 % vs ≈ 0.2 %). The iso-ppl Pareto below is
> the canonical result; the per-fixed-point table in `qwen3_4b_packed_e2e.json`
> is only an end-to-end sanity check, not a head-to-head.

## What this answers
Does the README's iso-ppl compression advantage of KakeyaLattice over TurboQuant
(bit-rate numbers, e.g. Qwen3-4B **+26.9%** at |Δppl|≤2%) **survive when measured
as REAL bit-packed bytes**, with both codecs packed identically?

**Verdict: the *direction* holds, but the *magnitude* shrinks a lot.** With real
bit-packing on real prose, at the production-relevant 1–2% |Δppl| band
KakeyaLattice still compresses more than TurboQuant — **E8 +7.7%, D4 +5.0%** —
not the +27% of the bit-rate table. **E8 (v1.5) beats TurboQuant at every quality
threshold** tested; D4 (v1.4) wins in the tight/mid band and loses slightly at the
loose 5% target. The big bit-rate headline does **not** transfer 1:1 to real HBM.

## What changed in the code (so this is measurable)
1. **Cache layout fix** (`quantized_cache.py`): KV is now one **contiguous
   per-layer buffer** grown by a single `cat` per `update()` (was a growing
   Python list re-`cat`-ed in full every step — O(N²) and not a contiguous
   buffer). Returned K/V are contiguous → directly SDPA-feedable.
2. **`bitpack.py`**: GPU-native fixed-width packer + **D4 and E8 block codes**
   that hit the codec's per-block bit budget exactly, with a tiny **exception
   side-channel** for the ~1% of blocks the codec's defensive `clamp` knocks out
   of the lattice (keeps packing **lossless**).
3. **`turboquant.py`**: `TurboQuantCodec` = the repo's scalar-quantise baseline,
   parameterised by bits/coord.
4. **`packed_cache.py`**: end-to-end `KakeyaLatticePackedCache` (D4/E8) and
   `TurboQuantPackedCache` — contiguous, decode-on-read, `kv_storage_bytes()`
   reports the **real bit-packed** footprint; pack→unpack verified lossless.

Real packed CR at Q=38 (vs the old int8 1.94×): **D4 2.46×, E8 2.37×** (E2E on
Qwen3-4B, `packed_e2e.json`). TurboQuant b=4 = 3.76× (but lower quality).

## Method
- Model Qwen3-4B (36 layers, head_dim 128), bf16, H200.
- Text: real Gutenberg prose (Pride & Prejudice), 4 passages × 2048 tokens →
  **bf16 ref ppl = 18.51** (≫1, so |Δppl| is discriminating, unlike the demo's
  repetitive text where ppl≈1).
- Each operating point run with its **packed cache** as `past_key_values` in one
  teacher-forcing forward: perplexity reflects the compression; `kv_storage_bytes`
  gives the real packed footprint. `real CR = bf16 DynamicCache bytes / packed`.
- All codecs applied to **K and V on all layers** (boundary=0) — a fair
  per-vector codec-vs-codec comparison (differs from the README table's
  boundary=2 whole-cache CR; absolute numbers therefore aren't directly
  comparable to that table — the valid comparison is D4/E8/TQ measured *here*).

## Full sweep (real bit-packed CR)
| codec | param | ppl | \|Δppl\| | real CR |
|---|---:|---:|---:|---:|
| bf16 | — | 18.51 | — | 1.00× |
| D4 | Q=10 | 18.96 | 2.73% | 3.56× |
| D4 | Q=15 | 18.42 | 0.84% | 3.20× |
| D4 | Q=22 | 18.64 | 1.67% | 2.91× |
| D4 | Q=38 | 18.54 | 0.73% | 2.46× |
| D4 | Q=76 | 18.50 | 0.12% | 2.13× |
| E8 | Q=10 | 18.50 | 0.96% | 3.28× |
| E8 | Q=15 | 18.62 | 0.99% | 2.98× |
| E8 | Q=22 | 18.56 | 1.18% | 2.73× |
| E8 | Q=38 | 18.53 | 0.18% | 2.37× |
| E8 | Q=76 | 18.47 | 0.33% | 2.07× |
| TQ | b=4 | 19.38 | 4.78% | 3.77× |
| TQ | b=5 | 18.37 | 0.88% | 3.05× |
| TQ | b=6 | 18.52 | 0.57% | 2.56× |
| TQ | b=7 | 18.50 | 0.30% | 2.21× |
| TQ | b=8 | 18.49 | 0.31% | 1.94× |

(Aggressive points: D4 Q4 4.92×/37%, Q6 4.27×/9.4%; E8 Q4 4.40×/18.7%, Q6 3.88×/
4.7%; TQ b3 4.92×/77%. High-Q low-CR: D4 Q152 1.88×, E8 Q152 1.56× via fallback.)

## Iso-ppl Pareto — REAL-byte compression ratio (best feasible CR per codec)
Each cell is the densest operating point whose mean |Δppl| ≤ the row threshold;
the winning **TurboQuant b** (and D4/E8 Q) is annotated in parentheses.

| \|Δppl\| ≤ | D4 | E8 | TurboQuant | D4 vs TQ | E8 vs TQ |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 2.13× (Q=76) | 2.37× (Q=38) | 2.21× (**b=7**) | −3.3% | **+7.6%** |
| 1.0% | 3.20× (Q=15) | 3.28× (Q=10) | 3.05× (**b=5**) | **+5.0%** | **+7.7%** |
| 2.0% | 3.20× (Q=15) | 3.28× (Q=10) | 3.05× (**b=5**) | **+5.0%** | **+7.7%** |
| 5.0% | 3.56× (Q=10) | 3.88× (Q=6)  | 3.77× (**b=4**) | −5.6% | **+2.9%** |

**TurboQuant bit budget used:** the sweep covered **b ∈ {3,4,5,6,7,8}**. The
iso-ppl winner is **b=7** at ≤0.5%, **b=5** at the 1–2% production band, and
**b=4** only at the loose 5% target. **b=2 (and b=3) are excluded** as
non-competitive: TurboQuant at b≤3 is catastrophic for KV (b=3 here gives
|Δppl|=77% at ppl 32.6; archived repo data shows b=2 |Δppl| in the tens-of-
thousands of %), so they never appear on the iso-ppl Pareto.

## Reading the result
- **Conclusion holds in direction, shrinks in size.** At the deployment-relevant
  1–2% band, KakeyaLattice still wins in *real bytes*: **E8 +7.7%, D4 +5.0%** over
  TurboQuant. But this is far from the bit-rate table's +26.9% — real bit-packing
  narrows the gap because TurboQuant's scalar codes also pack densely; the
  lattice's per-bit distortion advantage converts to a **modest** real-byte edge.
- **E8 ≥ TurboQuant everywhere** (+2.9% … +7.7%) and is the strongest codec.
- **D4 loses at loose (5%) and tightest (0.5%) targets** (−5.6%, −3.3%) — matches
  the original report's crossing analysis (scalar packs denser when distortion is
  allowed to be large; lattice overhead bites at the extremes).
- **Why smaller than the README's +27%:** different protocol (boundary=0 vs 2,
  real Gutenberg prose vs WikiText snapshot-mode vLLM, n=4, per-vector real packed
  bytes incl. fp16 overhead + ~1% E8 exceptions). The headline +9–38% are
  *bit-rate* ratios under a different harness; they should be read as the codec's
  information-theoretic edge, not the realized HBM edge.

## Caveats
- n=4 passages → some |Δppl| sampling noise (non-monotonic in Q at sub-1% levels);
  a larger n would smooth the Pareto. Direction is stable.
- D4 Q=152 and E8 Q≥76 use the per-coordinate / int16 fallback (low-CR region,
  outside the interesting band).
- Single model (Qwen3-4B), head_dim=128, per user scope.

## Files
- `qwen3_4b_real_cr.json` — full sweep + Pareto.
- `packed_e2e.json` — end-to-end packed-cache real CR (D4 2.46×, E8 2.37×, TQ 3.76×).
- Code: `kakeyalattice/python/kakeyalattice/hf/{bitpack,turboquant,packed_cache}.py`,
  tests `hf/test_bitpack.py`; harness `benchmarks/bitpack_vs_tq/`.

## Reproduce
```bash
python3 -m pytest kakeyalattice/python/kakeyalattice/hf/test_bitpack.py
python3 benchmarks/bitpack_vs_tq/verify_packed_e2e.py --model Qwen/Qwen3-4B
python3 benchmarks/bitpack_vs_tq/compare_real_cr.py --model Qwen/Qwen3-4B \
    --ctx-len 2048 --n-passages 4
```
