# Sprint Close-out — PR #13 consolidation

**Date.** 2026-04-17
**Purpose.** Single source of truth for what PR #13 actually
established. Use this as the starting point for the next session; the
long assistant context has developed inconsistencies that need a
clean slate.

## Established Pareto frontier (DS-Distill D=128, WikiText-103, 4 passages, ctx=2048)

All numbers verified against the per-cell JSON files under `reports/`.
All percentages are **Δppl vs bf16 reference**, top-1 vs bf16
reference.

| ID | Config | Ratio | Δppl | top-1 | Verdict |
|:--|:---|---:|---:|---:|:---:|
| v1.4 Pareto | K Kakeya exact b=4 + V Besi d=3 m=4 + 4 bdry | 2.97× | **−2.04 %** | **91.27 %** | **ACCEPT ★** |
| R3 | RSVD b=3 + K cal + outlier T=1.5 + V Besi + 6 bdry | **3.74×** | **+1.91 %** | **87.30 %** | **ACCEPT ★** |
| B3 | RSVD b=3 + K cal + outlier T=2.0 + V Besi + 6 bdry | 4.30× | +5.36 % | 85.32 % | MARGINAL |
| R1 | RSVD b=2 + K cal + outlier T=2.0 + V Besi + 6 bdry | 4.54× | +7.09 % | 82.54 % | MARGINAL |
| R2 | RSVD b=2 + K cal + outlier T=1.5 + V Besi + 6 bdry | 3.92× | +3.88 % | 84.13 % | MARGINAL |

Source JSON files:
- v1.4 Pareto: `reports/v1_4_besicovitch_v_only/ds_kakeya_vbesi_d3m4q_prerope_kv_b4_exact_fp16_sk0_sv0.json`
- R1/R2/R3: `reports/v1_3_riemann_b2/R{1,2,3}_*.json`
- B3: `reports/v1_3_revival/B3_rsvd_b3_outlier_Vbesi_prerope_kv_b3_randomized_fp16_sk0_sv0.json`

## Key architectural conclusions (established with experimental data)

1. **Q-preconditioning** (Cholesky of Σ_q) is the single biggest guardrail.
   Bare v1.3 RSVD b=2: +355 % Δppl. + Q-precond: +37.9 % Δppl.
   This IS the "put it in Riemannian space" operation — no separate
   Riemannian codec needed.

2. **Calibrated Lloyd-Max codebook** helps primarily at b=2. Pooled
   codebook trained on 25M α samples, saved in
   `reports/v1_4_q_pca/calibrated_codebook/ds_K_b{2,3}_centroids.f32`.
   Does NOT help at b=4 (degrades slightly there).

3. **Boundary expansion to 6 layers** (add L=7, L=14 — the two worst
   per-layer MSE on DS-Distill) is the second-biggest guardrail.
   +7.18 % → +1.60 % Δppl on d=6 m=4 Riemann configs. 8 boundaries
   starts to hurt (extra compression burden on remaining layers).

4. **Asymmetric K/V** (V stream uses Besicovitch d=3 m=4 +mean while K
   uses Kakeya-PCA): +1-2 pp Δppl, slight ratio cost. V benefits
   because V's distortion metric is plain MSE (Besi Haar codebook is
   RD-optimal); K benefits from PCA's per-block data adaptivity.

5. **Outlier compensation** T=1.5 (13.4 % of coords patched as f16)
   vs T=2.0 (4.5 % of coords): −3.3 pp Δppl, costs ~13 % ratio. On
   b=3 this crosses the ACCEPT threshold.

## Established negative results (save time — don't re-explore)

1. **Besicovitch skeleton replacing PCA on K**: loses 18 pp Δppl at
   matched MSE. Besi is rotation-invariant (Haar codebook is a
   mathematical symmetry), so attention-weighted rotation is moot on
   K. See `reports/v1_4_besicovitch/` and `reports/v1_4_k_besi_attention_weighted/`.

2. **Besi + Q-precond + quantized magnitude**: trilemma. Per-vector
   scale `max|α_k|` gets driven by the one whitening-amplified
   outlier group, destroying everything else's quantization.
   +700 % Δppl disaster. Use per-(layer, group) offline scale
   (Riemann variant) to escape.

3. **K-residual Besicovitch (replace Lloyd-Max only)**: 3× worse MSE
   than Lloyd-Max at matched bits. WHT + Lloyd-Max is a joint design;
   replacing only the quantizer breaks it. See
   `reports/v1_4_besicovitch_k_residual/`.

4. **Perron-tree / attention-weighted direction density**: oracle MSE
   gain only 0.1 % on real data. Real V groups have median λ₁/λ₂ =
   1.26 (near-isotropic). Haar codebook is already near-optimal
   for V. See `reports/v1_4_perron_tree_analysis/`.

5. **8-boundary expansion on Riemann K-Besi**: worse than 6-boundary.
   See `reports/v1_4_riemann_k_besi_enhanced/` for empirical check.

## TurboQuant comparison — honest takeaway

TurboQuant README's "5.12× turbo3" is **V-stream-only** at
`block_size=128`. Their actual **shipped** KV config is asymmetric:
`q8_0-K (8.5 bits) + turbo3-V (3.125 bits) + Boundary V`:

- Average bits/val ≈ 6.2 → **total KV compression ~2.58×** (not 5.12×)

Our R3 (3.74×) vs TurboQuant shipped (2.58×) at matched Δppl tier:
**+45 % ratio advantage**. Details in `reports/v1_3_riemann_b2/FAIR_VS_TURBOQUANT.md`.

TurboQuant Python **reference impl** test with +19000 % Δppl
(`reports/v1_3_riemann_b2/T{1,2,3}_*.json`) reflects the un-optimized
raw algorithm, NOT TurboQuant's shipped C++ performance. Do NOT use
those T-cell numbers to claim "10,000× better Δppl". That comparison
is apples-to-oranges.

## Multi-model + long-context status

- **DS-Distill 1.5B (Qwen2 family)**: primary validation model, all
  Pareto points above measured here.
- **GLM-edge 1.5b**: Pareto config confirmed (+1.47 % Δppl, 90.48 %
  top-1 at v1.4 Pareto recipe). See `reports/v1_4_multi_model/`.
- **Qwen3-0.6B**: K compression structurally incompatible (baseline
  already +39 % Δppl). Only V-only Besi works (1.73× @ −0.25 % Δppl
  ACCEPT). Not recommended for aggressive compression.
- **Long context** (ctx=4k, 8k, 16k): v1.4 Pareto holds, B3/R3 not
  re-tested at long context.

## Infrastructure summary (what's in the repo)

### Rust codec paths (kakeyaturbo/)
- `--codec kakeyaturbo --pca-method {exact, randomized}` — Kakeya-PCA (v1.2) + Kakeya-RSVD (v1.3)
- `--codec besicovitch` — Besi codec (v1.4 Besi sprint)
- `--outlier-threshold T` — sparse f16 outlier patching
- `--residual-besi-*` — K-residual Besi (negative result, kept for future)

### Python codec paths (benchmarks/)
- `--codec riemann_besi` — Riemannian Besi K codec (Python-only)
- TurboQuant reference impl via `turboquant_roundtrip.py`
- Q-precondition via `q_precondition.py`, calibration via `q_calibration.py`

### Calibration artifacts
- `reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors` — Σ_q Cholesky
- `reports/v1_4_q_pca/calibrated_codebook/ds_K_b{2,3}_centroids.f32` — Lloyd-Max
- `reports/v1_4_q_pca/calibrated_codebook/ds_V_b2_centroids.f32` — V calibration

### Test status
- Rust: 178 unit tests pass (+5 outlier, +11 besicovitch, +3 residual-besi)
- Python harness: syntax-check clean
- 75+ end-to-end PPL cells measured across 10+ sprints

## Open questions (fresh-context next session should tackle)

1. **Can R3 be pushed to ACCEPT ★ quality tier?**
   R3 is MARGINAL-edge at Δppl=+1.91 % / top-1=87.30 %. The 3 %
   threshold is met but top-1 is 4 pp below v1.4 Pareto's 91.27 %.
   Candidates to test:
   - R3 with exact PCA instead of RSVD (+encode cost for ~2 pp Δppl
     improvement)
   - R3 + per-layer calibrated codebooks (24 codebooks instead of 1)
   - R3 with longer context amortization (skeleton bytes decrease)

2. **Can R3 be validated on GLM/Qwen3?**
   Single-model Pareto claims are weak. GLM + R3 recipe not yet
   tested.

3. **Honest TurboQuant C++ comparison.**
   Integrating their Metal kernel or finding a public
   apples-to-apples benchmark on wikitext-103 ctx=2048 would settle
   the "+45 % ratio" claim rigorously.

4. **Long-context R3/B3.**
   At ctx ≥ 8k, skeleton bytes amortize better; ratios might push
   beyond 5× at ACCEPT.

## What went well / what went wrong in this sprint

### Went well
- User's sequential corrections kept sprints converging to real wins:
  - "v1.3 + guardrails" suggestion → R3 (new Pareto champion)
  - "fair comparison" correction → FAIR_VS_TURBOQUANT.md
  - "ratio bug" catch → corrected B3 from 3.71× to 4.30×

### Went wrong
- I went down 4 skeleton-replacement rabbit holes (Besi K, Besi V
  alone, K-residual Besi, Riemann K-Besi) before realizing the
  baseline needed rehabilitation first. That's 3-4 sprints of Δppl
  reporting that could have been one sprint of "apply guardrails to
  v1.3". User called this out and was right.
- Ratio accounting bug (exact vs RSVD boundary cost) went undetected
  until user spotted the 3.71× / 5.8× inconsistency.
- Compared our fully-guardrailed config to TurboQuant's
  un-guardrailed Python ref — produced 3-4 orders-of-magnitude Δppl
  gap that was misleading. User called this out and was right.

### Take into next session
- **Stop comparing to un-guardrailed reference impls.** Always
  compare matched-fence configs.
- **Recompute ratios from scratch** when configs change.
- **Baseline rehabilitation before codec redesign.** If a baseline
  is broken (+355 % Δppl), first try guardrails on that baseline.
