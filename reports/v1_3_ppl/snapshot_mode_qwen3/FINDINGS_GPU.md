# v1.3 PPL on vLLM GPU-codec — Qwen3-4B snapshot-mode findings

Companion to `reports/v1_3_ppl/FINDINGS.md` (which covered the CPU Rust
codec + DS-R1-Distill-Qwen-1.5B) and
`reports/v1_3_ppl/snapshot_mode_qwen3/QWEN3_SCENARIO_A_REPORT.md`
(which established the snapshot harness plumbing for Qwen3-4B).
This doc records the ablation matrix we ran on top of the
GPU-per-head codec path
(`kakeyaturbo_py.gpu_skeleton.fit_skeleton_batched` +
`kakeyaturbo_py.gpu_encode.encode_and_pack_batched`) and the
harness-level boundary-skip flag.

**Setup.** vLLM v1 engine with `VLLM_ENABLE_V1_MULTIPROCESSING=0`
(InprocClient, so `HookState` is reachable from the Qwen3Attention
forward monkey-patch), `enforce_eager=True`, bf16.  Model:
`Qwen/Qwen3-4B` (36 layers, 8 kv-heads, head_dim=128, per-head
qk-norm).  GPU: NVIDIA H200 80 GB (Vast.ai).  4 WikiText-103 test
passages, ctx=2048, evaluate positions `[2048, 2112)` (64
teacher-forced next tokens per passage).  Shared reference logprobs
per passage — all rows below are strictly paired.

**Harness.** `benchmarks/e2e_ppl_validation_vllm_snapshot_qwen3.py`
with `--gpu-codec --no-share-basis-v --disable-q-precond
--disable-centroids --disable-outlier` as the common base; only the
column under test varies per row.

**Codec config.** K bit_width=3, V bit_width=2, RSVD rank = D/2 = 64,
K-means k=16, `pca_method=randomized` oversample=8 power_iters=2,
Gaussian-default Lloyd-Max centroids (no M2 table), outliers off,
Σ_q whitening off.  These are deliberately the **bare** v1.3 codec
— every guardrail that did not add value was peeled off during the
ablation below.

## Baseline anchors (WikiText-103, 4 passages, 8k token window)

| Row                                  | Δppl (mean) | top-1 (mean) | Verdict |
|:-------------------------------------|------------:|-------------:|:-------:|
| bf16 baseline (no codec)             |    **0.00%** |     **100%** |  PASS   |
| CPU pooled-heads + full recipe       |   +8 858.93% |       31.25% | REJECT  |
| CPU pooled-heads + bare codec        |     +619.42% |       55.08% | REJECT  |
| **GPU per-head + bare codec (4-layer bdry)** | **+202.75%** |   **66.02%** | REJECT  |

## Knobs that helped (3 of the 10 we tried)

| Change                                                           | Before | After | Δ            | Mechanism |
|:-----------------------------------------------------------------|-------:|------:|:-------------|:----------|
| (a) **Per-head GPU codec** (1 head × n_tokens per block) vs CPU pooled-heads (8 heads × 64 tokens interleaved per block) |  +619.42% |   +202.75% | **−417 pp**  | Pooled-heads PCA has to compromise across 8 different per-head qk-norm distributions. Per-head PCA fits each head's own spectrum. Side-by-side decoded L2 rel-err: 0.3766 (pooled) → 0.1667 (per-head), 2.26× tighter. |
| (b) **Symmetric boundary skip, depth = 6–14** vs 4-layer `[0,1,34,35]` |  +202.75% | best **+96.86%** @ 14 layers | **−106 pp**  | Boundary layers stay bf16 — biggest single lever after (a). Monotonic in depth on symmetric patterns (6→10→12→14). See sweep table. |
| (c) **Σ_q whitening OFF** vs ON (raw cond=252 or Tikhonov reg50) | +8 858.93% / +492.37% |   +202.75% | **−290–8600 pp** | M2 Σ_q bundle calibrated on pre-RoPE Q is orthogonal to the post-qk-norm K distribution seen by the codec; L⁻¹ amplifies codec noise by up to cond(L). Regularisation tightens cond(L) but doesn't fix the **direction** mismatch. |

(a)+(b)+(c) combined is the **current best on Qwen3-4B**: mean Δppl
= **+96.86 %**, mean top-1 = **76.56 %**, verdict REJECT (below
`≤+20%` MARGINAL bar, but top-1 is above the 70% that PR #17 saw on
MARGINAL DeepSeek-1.5B).

### Symmetric boundary-skip depth sweep

All rows: `--disable-q-precond --disable-centroids --disable-outlier
--gpu-codec --no-share-basis-v`, 4 passages each.

| Depth | Pattern                                              | Δppl (mean) | top-1 (mean) | % of layers skipped |
|:-----:|:-----------------------------------------------------|------------:|-------------:|--------------------:|
|   4   | `[0,1,34,35]`                                         |   +202.75 % |      66.02 % |               11.1% |
|   6   | `[0,1,2,33,34,35]`                                    |   +169.36 % |      67.97 % |               16.7% |
|   7   | `[0,1,2,3,33,34,35]` (asymmetric)                     |   +169.51 % |      68.36 % |               19.4% |
|   8   | `[0,1,2,3,32,33,34,35]`                               |   +200.55 % |      66.41 % |               22.2% |
|  10   | `[0,1,2,3,4,31,32,33,34,35]`                          |   +145.79 % |      71.48 % |               27.8% |
|  12   | `[0,1,2,3,4,5,30,31,32,33,34,35]`                     |   +134.71 % |      75.39 % |               33.3% |
|  14   | `[0,1,2,3,4,5,6,29,30,31,32,33,34,35]`                |   **+96.86 %** |  **76.56 %** |              38.9% |

The 8-layer row breaks monotonicity (+200.55 % vs the 6-layer
+169.36 % and 10-layer +145.79 % neighbours).  We don't have a
causal explanation yet — possibly a single-passage variance
outlier, possibly layer-3 or layer-32 interacts badly with the
residual codec in its specific residual-stream role.  The overall
trend from 6 → 14 is otherwise cleanly monotonic improving.

### PR #17 interior-skip patterns don't transfer

PR #17's DS-1.5B recipe used `[0,1,7,14,26,27]` (interior skip at
layers 7 = 1/4, 14 = 1/2).  Scaled to Qwen3-4B's 36 layers = `9,
18`:

| Pattern                                    | Δppl (mean) | top-1 (mean) | Reading |
|:-------------------------------------------|------------:|-------------:|:--------|
| `[0,1,9,18,34,35]` (6 layers, PR-#17 style) |   +196.96 % |      66.02 % | Interior skip alone barely helps vs 4-layer (+202.75 %). |
| `[0,1,2,9,18,33,34,35]` (8 layers, mixed)   |   +158.62 % |      70.70 % | Better than plain 8-layer `[0,1,2,3,32,33,34,35]` +200.55 %, suggesting the interior-skip is contributing ≈ −40 pp; but still worse than pure symmetric 10-layer +145.79 %. |
| `[0,1,2,3,9,18,32,33,34,35]` (10 layers, mixed) |   +193.81 % |      66.80 % | Adding layers 3, 32 on top of PR-style pattern regresses. |

Reading: interior skip is **model-specific** to DS-1.5B; on
Qwen3-4B the better ROI is **more boundary depth**, not interior
skip.

## Knobs that did **not** help (7 distinct experiments)

Each row was measured against a 6-layer-boundary base (the best
symmetric baseline at the time these ablations were run; the
symmetric-depth sweep table above later showed 14-layer is better,
but the guardrail ablations are paired against 6-layer).

| Change                                                           | Result                   | Reading |
|:-----------------------------------------------------------------|:-------------------------|:--------|
| (d) **Σ_q Tikhonov regularisation** `cond ≤ 50` (tool: `benchmarks/q_regularize_sigma.py`) | +492.37 % (3× worse than off) | Any non-identity L rotates K away from its own principal axes. Condition number is a symptom, not the cause. |
| (e) **Calibrated Lloyd-Max centroids** (M2 `qwen3_4b_lloyd_max_{K,V}_b{3,2}.f32`) | same ± noise as Gaussian default | M2 centroids were fit on pre-RoPE K residuals; post-qk-norm residual distribution doesn't match. |
| (f) **Outlier compensation T=2.0** on top of 6-layer bdry        | +169.60 % (vs +169.36 % off, noise-level) | Post-qk-norm K residuals don't have the heavy tail the outlier side-buffer was designed to catch. |
| (g) **Approach B: query-subspace codec** (replace Σ_q with hard projection onto top-r eigenvectors of Q) at r ∈ {32, 64, 80, 96, 112, 128} | r=32: +17 758 %; r=64: +926 %; r=128 (identity): +292 % — all worse than bare +202 % | Qwen3-4B post-qk-norm Q is NOT low-rank (r=32 → 83 % energy; r=64 → 93 %; need r≥112 for 99 %). Projection steals codec budget from K's natural spectrum. |
| (h) **Share_basis on V stream** (V falls back to CPU Rust pooled-heads path that supports share_basis; K stays on GPU per-head) | +226.79 % vs nearly-paired-config 6-layer bare +169.36 % | Share_basis in CPU Rust has to pool across the whole layer; combined with pooled-heads layout it loses more V quality than share_basis saves. |
| (i) **K-stream: disable stage 4b Walsh-Hadamard rotation** (encode + decode symmetric skip via `--k-disable-wht`) | +95.88 % vs baseline +96.86 % (noise-level, 4-passage) | Residual after K-means already has a particular structure; WHT smears it uniformly for Lloyd-Max; skipping it neither helps nor hurts on post-qk-norm Qwen3-4B. |
| (j) **K-stream: SRHT sketch matrix** (Besicovitch-ish structured Ω = (1/√n) D · H[:, cols] replacing iid-Gaussian) via `--k-sketch-kind=srht` | +96.23 % (alone); +93.75 % combined with `--k-disable-wht` (noise-level) | HMT's SRHT error-bound advantage shows at large n; our n=512 is small enough that Gaussian's constants dominate.  Combined with WHT-off gives a 3 pp improvement that's within 4-passage variance band. |

## Architectural reading

- On Qwen3-4B's **post-qk-norm K**, all three **algorithmic**
  guardrails the CPU DeepSeek path leaned on (Σ_q, calibrated
  centroids, outlier compensation) are **neutral or harmful**.  The
  architectural reason is specific to qk-norm's independent per-head
  `g_q` / `g_k` weights breaking the Σ_q / Σ_k alignment that let
  Q-preconditioning win on DS-1.5B.  See
  `QWEN3_SCENARIO_A_REPORT.md` → "What this means for Scenario A".
- **Boundary-layer skip is the only PR-#17 guardrail that transfers
  cleanly**, and scales with depth: 4 → 14 layers is cleanly
  monotonic from +202.75 % to **+96.86 %** Δppl.  The 8-layer
  symmetric breakpoint (+200.55 %) is the one exception; we lack a
  causal explanation but the monotonic trend resumes at 10 layers.
  Interior skip (PR #17's 1/4, 1/2 positions) does not transfer.
  The win is mechanical: boundary layers hit bf16, skipping them
  caps the codec-error-amplification path.
- **The information-theoretic floor is NOT +169 %** — current best
  is **+61.84 %** at 14-layer skip + `b_K=4, k=64, d_eff=96` (top-1
  = 79.30 %, above the PR #17 DS-1.5B MARGINAL top-1 of ~74 %).
  Reaching MARGINAL (≤+20 % Δppl) from here likely needs a
  structurally different lever (e.g., codec-aware fine-tuning,
  block_size increase, or Q-recalibration on post-qk-norm data).

## Codec-budget sweep (4 passages each, 14-layer boundary base)

Three codec-budget knobs independently tested, then best combined.
Same base as above: `--disable-q-precond --disable-centroids
--disable-outlier --gpu-codec --no-share-basis-v
--boundary-skip-layers 0..6 29..35`.

| b_K | k_kmeans | d_eff | Δppl       | top-1    | Note |
|:---:|:--------:|:-----:|-----------:|---------:|:-----|
| 3   | 16       | 64    | +96.86 %   | 76.56 %  | baseline |
| 4   | 16       | 64    | +96.49 %   | 77.73 %  | b_K alone: noise-level Δppl, +1 pp top-1 |
| 3   | 32       | 64    | **+74.10 %** | 75.39 %  | **k alone: −22.8 pp** |
| 3   | 16       | 96    | +100.04 %  | 74.22 %  | d_eff alone: worse (more basis dims, same Lloyd-Max bits → lower per-dim SNR) |
| 4   | 32       | 64    | +73.87 %   | 75.78 %  | b_K + k: −0.2 pp vs k alone |
| 4   | 16       | 96    | +98.61 %   | 73.83 %  | b_K + d_eff: noise-level Δppl |
| 4   | 32       | 96    | +61.80 %   | 78.12 %  | **all three: −35.1 pp** |
| 4   | 64       | 96    | **+61.84 %** | **79.30 %** | **best: k=64 holds Δppl, +1.2 pp top-1** |
| 4   | 64       | 128   | +69.78 %   | 79.69 %  | d_eff=full-rank regresses on Δppl |

Reading: **K-means cluster count `k` is the dominant budget knob**
(`k=16→32` alone → −22.8 pp Δppl; `k=32→64` → plateau but top-1 still
gains).  `d_eff` and `b_K` in isolation are either noise-level or
counter-productive, but together with larger `k` they compound to
**−35 pp Δppl vs baseline**.  Further pushing `d_eff=96→128`
regresses — the codec is PCA-saturated beyond d_eff=96 on
Qwen3-4B's post-qk-norm K.

The best-measured configuration — canonical name
**`v1.3-GPU-Qwen-snap-bK64-bdry14`** (spoken: **`v1.3-GPU-snapA`**) —
is `b_K=4, k_K=64, d_eff=96` at `--boundary-skip-layers 0..6 29..35`
(V-stream kept at defaults `b_V=2, k_V=16`):

  * Δppl = **+61.84 %**
  * top-1 = **79.30 %**
  * Verdict REJECT (Δppl bar is ≤+20 % for MARGINAL) but top-1 is
    above PR #17 DS-1.5B production-cell MARGINAL top-1 = 74.22 %.
  * Relative to baseline `b_K=3, k=16, d_eff=64`: slot bytes rise
    by ~1.7 × (larger k-means cluster table + fp16 centroids ×
    more clusters + larger basis).  Exact slot-size impact can be
    read off `KakeyaV13PPLConfig.slot_size_bytes` for the config.

### V-stream budget is NOT symmetric with K

The "bigger-is-better" K-stream observation does NOT extend to V.
On the same `b_K=4, k_K=64, d_eff=96` base, pushing V's k / b in
the K direction REGRESSES on Δppl:

| b_V | k_V | Δppl       | top-1    | Δ vs baseline |
|:---:|:---:|-----------:|---------:|--------------:|
| 2   | 16  | **+61.84 %** | 79.30 %  | baseline |
| 2   | 32  | +70.71 %   | 79.30 %  | +8.9 pp worse |
| 2   | 64  | +68.03 %   | **80.47 %** | +6.2 pp worse |
| 3   | 16  | +66.52 %   | 79.30 %  | +4.7 pp worse |
| 3   | 64  | +68.16 %   | 78.91 %  | +6.3 pp worse |
| 4   | 64  | +65.63 %   | 80.47 %  | +3.8 pp worse |

Reading:
* **Attention-score sensitivity**: K error is exponentially
  amplified by softmax, V error is linearly weighted and dampened.
  Per-bit ROI on K ≫ per-bit ROI on V.
* **Qwen3's qk-norm** only normalises Q/K, not V.  V's spectrum
  is flatter and has less compressible structure; larger `d_eff`
  (shared K/V) is *already* providing all the dim-budget V can
  usefully absorb.
* **Fixed `b_V=2` residual budget**: `ceil(log2(k))` seg_id bits
  grow with `k_V`.  At `b_V=2`, going `k_V=16→32` eats one extra
  seg_id bit per token with only 4 Lloyd-Max levels of residual
  precision left to compensate, so residual precision is degraded.
* **Top-1 does rise by ~1 pp at `k_V=64`** (independent of the
  Δppl drop), suggesting the structured V-codebook helps the
  model pick the right top prediction more often even when its
  logit spread is noisier.  Not enough to change the verdict.

If further V work is worthwhile, the right direction is the
*opposite* of K: **reduce** V's slot budget (e.g. `b_V=1` or
`k_V=8`) and shift the saved bytes to K.  Or implement GPU
`share_basis_v=True` (the PR #17 DS-1.5B recipe, which PR #17
confirmed helps the V-stream via pooled-across-blocks basis) —
currently only the CPU path supports it, and mixing GPU-K with
CPU-V-share_basis was tested and regresses (+226.79 %).

#### V shrink + K uplift measurements

Trying to push in the opposite direction — shrink V to fund
larger K — also doesn't win:

| b_V | k_V | b_K | k_K | Δppl       | top-1    | Δ vs baseline |
|:---:|:---:|:---:|:---:|-----------:|---------:|--------------:|
| 2   | 16  | 4   | 64  | +61.84 %   | 79.30 %  | baseline |
| 1   | 16  | 4   | 64  | +79.84 %   | 76.56 %  | +18.0 pp worse |
| 2   | 8   | 4   | 64  | +69.24 %   | 78.52 %  | +7.4 pp worse  |
| 1   | 8   | 4   | 64  | +92.80 %   | 74.22 %  | +31.0 pp worse |
| 2   | 16  | 4   | 128 | +65.98 %   | **81.64 %** | +4.1 pp Δppl worse, **+2.3 pp top-1 better** |
| 2   | 8   | 4   | 128 | +67.95 %   | 80.08 %  | +6.1 pp worse  |

Reading:
* **V is already on the Pareto frontier** at `b_V=2, k_V=16` —
  every shrink costs more Δppl than the K-side uplift can recover.
  `b_V=1` alone costs 18 pp; `b_V=1 + k_V=8` combined costs 31 pp.
* **`k_K=128` is the top-1 winner**: 81.64 % top-1 (new high for
  Qwen3-4B snapshot-mode) at the cost of 4.1 pp more Δppl.
  This is a **deliberate Pareto point**, not a regression — for
  *decoding-based* downstream evals (argmax / multiple-choice)
  the top-1 matters more than the logprob spread.
* **Pareto-incompatible trade**: saving V bytes (shrink)
  combined with spending them on K (uplift) gives strictly worse
  results on both axes (+6.1 pp Δppl, −1.56 pp top-1).  V is
  paying in Δppl more than K is earning.

### Two operational recipes on Qwen3-4B

Depending on downstream task:

**`v1.3-GPU-Qwen-snap-bK64-bdry14`** (spoken: **`v1.3-GPU-snapA`**)
— Δppl-optimal, for perplexity / LM-eval tasks:
```
--bit-width-k 4 --k-kmeans-k 64 --rsvd-target-rank-factor 0.75
--bit-width-v 2 --v-kmeans-k 16
--boundary-skip-layers 0 1 2 3 4 5 6 29 30 31 32 33 34 35
--gpu-codec --no-share-basis-v
--disable-q-precond --disable-centroids --disable-outlier
```
→ Δppl = **+61.84 %**, top-1 = 79.30 %

**`v1.3-GPU-Qwen-snap-bK128-bdry14`** (spoken: **`v1.3-GPU-snapB`**)
— top-1-optimal, for argmax / MMLU-style evals.
Same as snapA but with `--k-kmeans-k 128`.
→ Δppl = +65.98 %, top-1 = **81.64 %**

See `NAMING.md` in this directory for the full canonical-name
schema and the table of legacy aliases (Recipe A / best-K /
`qwen3_4b_budget_k64_bK4_deff96` / …).

Neither hits the MARGINAL Δppl bar (≤+20 %), but both have top-1
above the PR #17 DS-1.5B MARGINAL (74.22 %).

### Knobs that did NOT help — continued

#### (h) Exact PCA instead of RSVD: **NOT the K-MSE bottleneck**
(`v1.3-GPU-Qwen-snap-bK64-bdry14-pcaExact`, spoken `v1.3-GPU-snapC`)

Prior diagnosis claimed snapA's K-MSE = 0.503 was dominated by RSVD
failing on Qwen3-4B's flat K spectrum, and that exact PCA would
cut it by ~44×.  We implemented `pca_kind="exact"` in
`kakeyaturbo_py.gpu_skeleton.fit_skeleton_batched`
(`torch.linalg.svd` on the centred data matrix; V stream stays on
RSVD) and ran the 4-passage snapshot comparison at the snapA
budget.  Result (paired across the same 4 passages, same refs):

| Configuration | Δppl (paired) | top-1 | K-MSE (mean non-bdry layers) |
|:--------------|--------------:|------:|----------:|
| snapA (RSVD)  | +61.84 % | 79.30 % | 0.5030 |
| snapC (exact) | +74.20 % | 79.30 % | 0.5020 |
| Δ (C − A)     | **+12.36 pp worse** | +0.00 pp | −0.0010 (0.2 %) |

Per-passage paired: snapC is worse or tied on 3 of 4 passages
(P1 Δalt +6.44 ppl, P2 Δalt +0.82 ppl, P3 Δalt −0.04 ppl, P4
Δalt +1.31 ppl).  Top-1 pairwise identical at 79.30 % (with
per-passage movement of ±3.1 pp averaging to zero).

**What actually dominates K-MSE** — direct decomposition on a
representative mid-stack non-boundary layer (layer 17), 4 blocks
of 512 tokens, 8 kv-heads, same K tensor fed to both paths:

| Stage                                    | K-MSE (RSVD) | K-MSE (exact) |
|:-----------------------------------------|-------------:|--------------:|
| (1) bf16 only                            | 0.00000      | 0.00000       |
| (2) + mean subtract                      | 0.00000      | 0.00000       |
| (3) + PCA truncate d_eff = 96 of 128     | **0.00572**  | **0.00564**   |
| (4) + spherical K-means nearest centre (k=64) | 0.78073 | 0.78144 |
| Full codec (WHT + Lloyd-Max + residual), from json | 0.5595 | 0.5579 |

RSVD and exact PCA give **identical** reconstruction at stage (3)
— 1.4 % difference, matches the synthetic smoke test.  On
Qwen3-4B's measured K spectrum, RSVD with power_iters=2 is
already pinned to within ~1 % of the SVD floor.  The entire
~44× gap reported earlier was measured against a synthetic
flat-spectrum tensor that does NOT represent Qwen3-4B's real K,
where the PCA stage's angular error is essentially zero.

**The real K-MSE bottleneck is the spherical K-means
nearest-centre projection** (stage 3 → 4: K-MSE leaps from 0.006
to 0.78, a 135× amplification).  WHT + Lloyd-Max + outlier
residual recovers some of that loss back down to the measured
0.56, but the structural limit on this recipe is the K-means
clustering quality on the PCA-projected coeffs.  Future K-MSE
reductions must target:
  * the K-means stage itself (k uplift, seeding strategy, or a
    non-spherical assignment), or
  * more Lloyd-Max precision (bit_width_k uplift), or
  * a smaller residual-driven error path that cannot benefit from
    the centre-nearest projection.

PCA kind is not a lever.  We leave the `pca_kind="exact"` code
path in for completeness (it's functionally correct, just not a
productive knob on this model), but snapC is **not adopted** as
a recipe.  `v1.3-GPU-snapA` remains the Δppl-optimal and
`v1.3-GPU-snapB` the top-1-optimal operational recipe.

JSON: `reports/v1_3_ppl/snapshot_mode_qwen3/pcaExact/qwen3_4b_snap_pcaExact_vllm_snapshot.json`.
Decomposition probe: `/tmp/decompose_k_mse.py` (in-session only;
not checked in).

## Report index (JSONs in this directory)

Baselines (historical, pre-ablation):

- `qwen3_4b_snap_skipall_vllm_snapshot.json` — identity replace
  sanity (Δppl = 0.00%, plumbing OK).
- `qwen3_4b_snap_smoke_vllm_snapshot.json` — full recipe 1p smoke
  (Σ_q + centroids + outlier, CPU): +8859%.
- `qwen3_4b_snapshot_bare_vllm_snapshot.json` — CPU bare codec
  (4p): +619%.
- `qwen3_4b_snapshot_gpu_all_vllm_snapshot.json` — GPU bare codec
  4-layer bdry (4p): +202.75%.

Codec-budget sweep (`budget_sweep/` subfolder, all 4-passage, on
14-layer boundary base):

- `qwen3_4b_budget_default_vllm_snapshot.json` — b_K=3, k=16, d_eff=64:
  +96.86 %, 76.56 % (baseline).
- `qwen3_4b_budget_bK4_vllm_snapshot.json` — b_K=4: +96.49 %, 77.73 %.
- `qwen3_4b_budget_k32_vllm_snapshot.json` — k=32: **+74.10 %, 75.39 %**.
- `qwen3_4b_budget_deff96_vllm_snapshot.json` — d_eff=96: +100.04 %, 74.22 %.
- `qwen3_4b_budget_k32_bK4_vllm_snapshot.json` — k=32+b_K=4: +73.87 %, 75.78 %.
- `qwen3_4b_budget_deff96_bK4_vllm_snapshot.json` — d_eff=96+b_K=4: +98.61 %, 73.83 %.
- `qwen3_4b_budget_all3_vllm_snapshot.json` — all 3: +61.80 %, 78.12 %.
- `qwen3_4b_budget_k64_bK4_deff96_vllm_snapshot.json` — **best: +61.84 %, 79.30 %**.
- `qwen3_4b_budget_k64_bK4_deff128_vllm_snapshot.json` — d_eff=128 (regresses): +69.78 %, 79.69 %.
- `qwen3_4b_budget_bestK_Vdefault_vllm_snapshot.json` — V defaults pinned, best-K base (= re-run of the best config): +61.84 %, 79.30 %.
- `qwen3_4b_budget_bestK_Vk32_vllm_snapshot.json` — V k=32: +70.71 %, 79.30 % (regresses 8.9 pp).
- `qwen3_4b_budget_bestK_Vk64_vllm_snapshot.json` — V k=64: +68.03 %, 80.47 % (Δppl regresses 6.2 pp; top-1 +1.17 pp).
- `qwen3_4b_budget_bestK_VbV3_vllm_snapshot.json` — V b=3: +66.52 %, 79.30 % (regresses 4.7 pp).
- `qwen3_4b_budget_bestK_Vk64_bV3_vllm_snapshot.json` — V k=64 + b=3: +68.16 %, 78.91 %.
- `qwen3_4b_budget_bestK_Vk64_bV4_vllm_snapshot.json` — V k=64 + b=4 (full K/V symmetry): +65.63 %, 80.47 %.
- `qwen3_4b_budget_bestK_VbV1_vllm_snapshot.json` — V b=1: +79.84 %, 76.56 %.
- `qwen3_4b_budget_bestK_Vk8_vllm_snapshot.json` — V k=8: +69.24 %, 78.52 %.
- `qwen3_4b_budget_bestK_Vk8_bV1_vllm_snapshot.json` — V k=8 + b=1: +92.80 %, 74.22 % (worst).
- `qwen3_4b_budget_kK128_vllm_snapshot.json` — **K k=128** (= `v1.3-GPU-snapB`): +65.98 %, **81.64 %** (top-1 new high).
- `qwen3_4b_budget_kK128_Vk8_vllm_snapshot.json` — K k=128 + V k=8 (Pareto-incompatible shrink): +67.95 %, 80.08 %.

Boundary-skip sweep (`bdry_sweep/` subfolder, all 4-passage):

- `qwen3_4b_bare_bdry_0_1_2_33_34_35_vllm_snapshot.json` — 6-layer
  symmetric: +169.36 %, 67.97 % top-1.
- `qwen3_4b_bare_bdry_0_1_2_3_33_34_35_vllm_snapshot.json` — 7-layer
  asymmetric: +169.51 %, 68.36 %.
- `qwen3_4b_bare_bdry_0_1_2_3_32_33_34_35_vllm_snapshot.json` —
  8-layer symmetric, non-monotonic regression: +200.55 %.
- `qwen3_4b_bare_bdry_0_1_2_3_4_31_32_33_34_35_vllm_snapshot.json` —
  10-layer symmetric: +145.79 %, 71.48 %.
- `qwen3_4b_bare_bdry_0_1_2_3_4_5_30_31_32_33_34_35_vllm_snapshot.json`
  — 12-layer symmetric: +134.71 %, 75.39 %.
- `qwen3_4b_bare_bdry_0_1_2_3_4_5_6_29_30_31_32_33_34_35_vllm_snapshot.json`
  — **14-layer symmetric, current best**: +96.86 %, 76.56 %.
- `qwen3_4b_bare_bdry_0_1_9_18_34_35_vllm_snapshot.json` — PR#17
  interior-skip pattern (6 layers): +196.96 %.
- `qwen3_4b_bare_bdry_0_1_2_9_18_33_34_35_vllm_snapshot.json` —
  6-layer + 2 interior (= 8 total): +158.62 %.
- `qwen3_4b_bare_bdry_0_1_2_3_9_18_32_33_34_35_vllm_snapshot.json` —
  8-layer + 2 interior (= 10 total): +193.81 %.
- `qwen3_4b_bdry6_outlier_vllm_snapshot.json` — 6-layer + T=2.0
  outlier on: +169.60 % (noise-level diff from outlier-off).
- `qwen3_4b_bdry6_mixed_kgpu_vcpu_vllm_snapshot.json` — K GPU +
  V CPU share_basis on (6-layer): +226.79 % (share_basis hurts).
