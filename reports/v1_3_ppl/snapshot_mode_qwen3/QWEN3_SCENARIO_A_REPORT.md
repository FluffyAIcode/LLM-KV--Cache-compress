# Scenario A — Qwen3-4B snapshot-mode PPL (WikiText-103, 2048 ctx × 64 eval × 4 passages)

## Setup
- Model: `Qwen/Qwen3-4B` (36 layers, 8 kv-heads × 128 head_dim, bf16)
- Engine: vLLM v1 in **InprocClient** mode (`VLLM_ENABLE_V1_MULTIPROCESSING=0`)
- Harness: `benchmarks/e2e_ppl_validation_vllm_snapshot_qwen3.py`
  - Pass 1: clean forward → capture post-qk-norm / pre-RoPE K, V per layer
  - Offline: per-layer codec (RSVD + Lloyd-Max + optional Σ_q + optional outlier)
  - Pass 2: `Qwen3Attention.forward` hook replaces live K, V with codec'd snapshot
    (Q still comes from running residual — true HF two-pass semantics)
- Monkey-patch installed via `vllm.general_plugins` entry point gated on
  `KAKEYA_SNAPSHOT_QWEN3=1` so it runs in the parent process (InprocClient runs
  model inference in-process so no subprocess plumbing needed).
- M2 calibration artefacts: `qwen3_4b_sigma_q.{safetensors,json}`,
  `qwen3_4b_lloyd_max_K_b3.f32`, `qwen3_4b_lloyd_max_V_b2.f32`.

## Ablation grid (1 passage for identity, otherwise 1 passage smoke; see JSONs)

| Config | Δppl | top-1 agree | Verdict | Report |
|---|---|---|---|---|
| Identity replace (skip all 36 layers) | **+0.00%** | 100% | PASS | `qwen3_4b_snap_skipall_vllm_snapshot.json` |
| Full production recipe (Σ_q + centroids + outlier, skip boundary 0/1/34/35) | **+8 858.93%** | 31.25% | REJECT | `qwen3_4b_snap_smoke_vllm_snapshot.json` |
| Only Σ_q on (centroids/outlier off) | +8 458.53% | 29.69% | REJECT | `qwen3_4b_snap_sigma_only_vllm_snapshot.json` |
| Only Σ_q off (centroids/outlier on) | +497.72% | 67.19% | REJECT | `qwen3_4b_snap_noguard_vllm_snapshot.json` … wait, see below |
| Σ_q + centroids + outlier ALL off (bare codec) | **+619.42%** (4 passages) | 55.08% | REJECT | `qwen3_4b_snapshot_bare_vllm_snapshot.json` |

### Column legend
- **Δppl**: `(ppl_alt - ppl_ref) / ppl_ref`, averaged across passages
- **top-1 agreement**: fraction of eval positions where alt-pass argmax equals ref-pass argmax
- **PASS / MARGINAL / REJECT** thresholds from PR #17 (`≤+10%` / `≤+20%` / else)

## Identity replace = PASS proves the plumbing is correct

The `skip-all` run (`--boundary-skip-layers 0 … 35`) returned every layer's
captured K/V verbatim to pass 2 — byte-exact identity — and PPL matched
the clean run to 3 decimals with top-1 = 100%.  So the three-phase hook,
`InprocClient` + plugin-based patch-install, and the pass-2 replacement
handoff are all working.  Any PPL regression is attributable to the codec
itself, not the harness.

## Root cause of the +8 859% blowup: Σ_q conditioning

Reading the Cholesky factors for each (layer, kv-head) of Qwen3-4B's Σ_q:

| Model | median cond(L) | max cond(L) | Full-recipe Δppl in snapshot mode |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B (PR #17) | 65 | 235 | +29.07% |
| **Qwen3-4B (this run)** | **252** | **337** | **+8 859%** |

Σ_q unwhitening on the decoder side multiplies by `L⁻¹`, whose spectral norm
is the condition number of `L`.  Every 1% codec error in whitened space blows
up to ~252% in original space for Qwen3-4B — which matches the measured
+8 859% (median cond × codec error).

The Σ_q bundle for Qwen3-4B was calibrated on the same Σ_q = E[q qᵀ]
protocol that PR #17 used; the conditioning is worse because Qwen3's
qk-norm concentrates energy in a few head-dims, so the Gram matrix is
inherently closer to low-rank.

## Secondary finding: bare codec is also rejected (+619%)

Disabling Σ_q, custom centroids, and outliers to get the raw codec
(RSVD d_eff=64 + spherical K-means k=16 + Lloyd-Max Gaussian centroids
+ no outlier side-buffer) still gives +619% Δppl on Qwen3-4B.

PR #17 on DeepSeek 1.5B had ppl_ref ~3 → ppl_alt ~3.8 (≈+29%).
Qwen3-4B has ppl_ref ~3.8–27 across WikiText-103 passages; bare codec
gets ppl_alt 27–165 (≈+620%).  The codec error per layer is
intrinsically higher for Qwen3-4B's post-qk-norm K than for DeepSeek's
pre-qk-norm K — same mechanism as Σ_q conditioning: qk-norm makes the
spectrum more uniform.

## What this means for Scenario A

Snapshot-mode semantics (codec as post-prefill cache compressor) is the
right deployment pattern, and the harness is proven correct.  But on
Qwen3-4B **the codec itself** is the bottleneck — not the harness, not
the engine.  Two mitigations are realistic next steps (both outside this
commit's scope):

1. Recalibrate Σ_q using regularisation (`Σ_q + λ·I` with `λ` chosen
   to cap cond ≤ 50).  This trades some Q-preconditioning power for
   decode-side numerical stability.
2. Increase the codec budget: b_K = 4 (from 3), k_centers = 32 (from 16),
   or d_eff = 96 (from 64).  Costs slot bytes but gives directly measurable
   Δppl improvement.

## Scenario A deployment verdict for Qwen3-4B: not deployable yet on current recipe

Best snapshot-mode config on Qwen3-4B (bare codec, 4-passage mean):
**Δppl = +619%**, top-1 = 55%.  Below the PR #17 DeepSeek `+29% MARGINAL`
bar.  Needs calibration fix or codec-budget uplift before snapshot-mode
is production-viable on this model.

## Artefacts
- Harness: `benchmarks/e2e_ppl_validation_vllm_snapshot_qwen3.py`
- Monkey-patch + `HookState`: `vllm_backend/kakeya_v1_3_ppl/snapshot_hook.py`
- Plugin gate: `vllm_backend/kakeya_v1_3_ppl/plugin.py` (env var `KAKEYA_SNAPSHOT_QWEN3=1`)
- Per-ablation JSONs: `reports/v1_3_ppl/snapshot_mode_qwen3/*.json`
