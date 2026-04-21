# Sprint Close-out — PR #13 consolidation

**Date.** 2026-04-17 → 2026-04-21
**Purpose.** Single source of truth for what PR #13 actually
established. The production recipe is **one codec path** — v1.3
KakeyaTurbo plus four PPL stabilization guardrails (= "v1.3 PPL").
Quality ↔ ratio trade-off is tuned by K / V bit width on that same
recipe; no additional codec paths are needed.

## Production recipe: **v1.3 PPL**

The v1.3 RSVD codec (K b=3 + V b=2 with `--share-basis-v`, the original
v1.3 defaults) wrapped by the four PPL stabilization guardrails:

1. **Q-preconditioning** (Chol Σ_q) on K
2. **Calibrated Lloyd-Max** K residual codebook
3. **6-layer boundary protection** `[0, 1, 7, 14, 26, 27]` kept bf16
4. **Outlier compensation** T=2.0 on K residual (~4.5 % of coords → f16)

## Production cell (DS-Distill D=128, WikiText-103, ctx=2048, 4 passages)

| Config | Ratio | Δppl | top-1 | Verdict |
|:---|---:|---:|---:|:---:|
| **v1.3 PPL** (K b=3, V b=2, T=2.0, 6 bdry) | **4.61×** | **+7.82 %** | **78.97 %** | **MARGINAL 🎯** |

Source JSON: `reports/v1_3_ppl/v1_3_ppl_prerope_kv_b3_randomized_fp16_sk0_sv1.json`
Sprint write-up: `reports/v1_3_ppl/FINDINGS.md`

**Highest-ratio point measured in this PR at Δppl ≤ 10 %.** It is the
v1.3 KV-cache compression codec, unchanged, with a PPL-stabilization
layer around it — not a new algorithm.

## Quality ↔ ratio tuning (same recipe, adjust bit widths)

If a deployment needs higher quality (ACCEPT-level), **raise K and/or
V bit width on the same v1.3 PPL recipe**:

- K b=3 → K b=4 (raise K residual fidelity)
- V b=2 → V b=3 or V b=4 (raise V residual fidelity)
- RSVD rank-factor 0.5 → 0.75 (larger skeleton, smaller residual)

These are all existing harness flags (`--bit-width`, `--bit-width-v`,
`--rsvd-rank-factor`) on the v1.3 PPL recipe. No alternative codec
paths (Besicovitch, Riemann-Besi, asymmetric K/V with a Haar codebook,
etc.) are needed for deployable quality tiers.

## Evidence ladder — how v1.3 PPL was built

(`reports/v1_3_revival/FINDINGS.md` documents each step.)

### b=2 ladder (progressive guardrails)

| Step | Added guardrail | Δppl | top-1 |
|:----:|:---|-----:|------:|
| V0 | **BARE** v1.3 RSVD b=2 | **+355.62 %** | 42.46 % |
| V1 | + Q-precondition + 4 bdry | **+37.91 %** | 73.02 % |
| V2 | + K calibrated Lloyd-Max codebook | +36.53 % | 68.25 % |
| V3 | + 6 bdry (add L=7, L=14) + V cal | **+25.18 %** | 71.43 % |

Source: `reports/v1_3_revival/V{0-3}_*.json`

**V0 → V3: Δppl +355 % → +25 %, top-1 42 % → 71 %.** Q-precond alone
is worth 317 pp Δppl.

### b=3 ladder

| Step | Added guardrail | Δppl | top-1 |
|:----:|:---|-----:|------:|
| B0 | **BARE** v1.3 RSVD b=3 | +374.90 % | 41.27 % |
| B1 | + all guardrails except outlier | +15.73 % | 76.98 % |
| v1.3 PPL | + outlier T=2.0 (on B1) | **+7.82 %** | **78.97 %** |

Source: `reports/v1_3_revival/B{0,1}_*.json`, `reports/v1_3_ppl/`

**B0 → v1.3 PPL: Δppl +374 % → +7.82 %, top-1 41 % → 79 %.**

## Four architectural conclusions (established with experimental data)

1. **Q-preconditioning** (Chol Σ_q) is the single biggest guardrail.
   Bare v1.3 RSVD b=2: +355 % Δppl. + Q-precond: +37.9 % Δppl.
   The whitening *is* the Σ_q-metric Riemannian → Euclidean isometry —
   no separate "Riemann codec" is needed.

2. **Calibrated Lloyd-Max codebook** helps primarily at b=2. Pooled
   codebook trained on 25 M α samples, saved in
   `reports/v1_4_q_pca/calibrated_codebook/ds_K_b{2,3}_centroids.f32`.
   Does NOT help at b=4 (degrades slightly there).

3. **Boundary expansion to 6 layers** (add L=7, L=14 — the two worst
   per-layer MSE on DS-Distill) is the second-biggest guardrail.
   8 boundaries starts to hurt (extra compression burden on the
   remaining layers).

4. **Outlier compensation T=2.0** (~4.5 % of K residual coords stored
   as sparse f16) is the step that lifts Δppl below +10 % on b=3.
   Tighter thresholds trade ratio for Δppl monotonically — adjust
   via `--k-outlier-threshold`.

## Established negative results (save time — don't re-explore)

1. **Besicovitch skeleton replacing PCA on K**: loses 18 pp Δppl at
   matched MSE. Besi is rotation-invariant (Haar codebook is a
   mathematical symmetry), so attention-weighted rotation is moot
   on K. See `reports/v1_4_besicovitch/` and
   `reports/v1_4_k_besi_attention_weighted/`.

2. **Besi + Q-precond + quantized magnitude**: trilemma. Per-vector
   scale `max|α_k|` gets driven by the one whitening-amplified
   outlier group, destroying everything else's quantization.
   +700 % Δppl disaster.

3. **K-residual Besicovitch (replace Lloyd-Max only)**: 3× worse MSE
   than Lloyd-Max at matched bits. WHT + Lloyd-Max is a joint design;
   replacing only the quantizer breaks it. See
   `reports/v1_4_besicovitch_k_residual/`.

4. **Asymmetric V Besicovitch**: adds 7–14 % ratio cost for a Δppl
   benefit that is only visible when K Δppl is already ≤ +2 %. For
   the v1.3 PPL production tier, Besicovitch on V is net-negative.
   Raise K / V bit width instead.

5. **Perron-tree / attention-weighted direction density**: oracle MSE
   gain only 0.1 % on real data. Real V groups have median λ₁/λ₂ =
   1.26 (near-isotropic). Haar codebook is already near-optimal
   for V. See `reports/v1_4_perron_tree_analysis/`.

6. **8-boundary expansion**: strictly worse than 6-boundary.

7. **Alternative "Riemann codec" with its own scale/magnitude
   calibration**: Q-precond already provides the Σ_q-metric Euclidean
   frame; no separate path is needed.

## Multi-model + long-context status

- **DS-Distill 1.5B (Qwen2 family)**: primary validation model for
  v1.3 PPL.
- **GLM-edge 1.5b**: v1.3-family recipe transferred with similar Δppl
  behaviour (`reports/v1_4_multi_model/`).
- **Qwen3-0.6B**: K compression structurally incompatible (baseline
  already +39 % Δppl). Only V-only path works. Not recommended for
  aggressive compression.
- **Long context** (ctx=4k, 8k, 16k): v1.3 PPL not yet re-tested at
  long context — skeleton amortisation should improve ratio.

## Infrastructure summary

### Rust codec path (kakeyaturbo/) — single production binary

- `kakeyaturbo-bench --pca-method randomized --bit-width B
  --rsvd-target-rank R --outlier-threshold T --centroids-file F` —
  the entire v1.3 PPL recipe on K.
- V stream uses the same binary at its own bit width
  (`--bit-width-v`) with `--share-basis-v`.

### Python harness (benchmarks/)

- `e2e_ppl_pre_rope.py` — 4-passage WikiText-103 PPL validation
- `q_precondition.py`, `q_calibration.py` — Σ_q Cholesky and K cal
- `lloyd_max_calibration.py` — Lloyd-Max codebook fitting

### Calibration artifacts

- `reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors` — Σ_q Cholesky
- `reports/v1_4_q_pca/calibrated_codebook/ds_K_b{2,3}_centroids.f32` — K Lloyd-Max
- `reports/v1_4_q_pca/calibrated_codebook/ds_V_b2_centroids.f32` — V Lloyd-Max

### Test status

- Rust: 178 unit tests pass
- Python harness: syntax-check clean
- v1.3 PPL measured on 4 WikiText passages; progressive ladder cells
  retained in `reports/v1_3_revival/` for the guardrail-contribution
  evidence trail.

## Open questions (next session)

1. **v1.3 PPL at higher K / V bit width** — sweep K ∈ {3, 4} and
   V ∈ {2, 3, 4} on DS-Distill to populate the quality ↔ ratio
   frontier on the single recipe.

2. **v1.3 PPL on GLM-edge and Qwen3** (multi-model validation).

3. **v1.3 PPL at ctx ∈ {4k, 8k, 16k}** (skeleton amortisation at
   long context should further improve ratio).

4. **Honest TurboQuant C++ comparison** on matched (model, ctx,
   bits) — the reference Python impl comparison in earlier sprints
   was apples-to-oranges and has been removed.

## Lessons

- **One codec path beats a zoo.** The five prior skeleton-replacement
  sprints (Besi-K, V-only Besi, K-residual Besi, Riemann K-Besi,
  Perron-tree) produced negative or tier-limited results. The v1.3
  RSVD skeleton with guardrails dominates them all.
- **Baseline rehabilitation before codec redesign.** A broken baseline
  (+355 % Δppl) almost always needs guardrails, not a new codec.
- **Tune by bit width, not by codec.** Quality ↔ ratio is a K/V bit
  width choice on the v1.3 PPL recipe — not a reason to add a new
  codec path.
