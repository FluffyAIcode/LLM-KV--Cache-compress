# Qwen3-4B snapshot-mode codec config naming

This document pins the canonical name for the current best
measured Qwen3-4B + vLLM codec configuration, plus its sibling
(top-1-optimal) variant.  All reports, commit messages, and
follow-up conversation should use these names from now on to
avoid the naming drift that accumulated during the PR #17 →
Scenario A → Path D → budget sweep → head-to-head rollout.

## Canonical name

**`v1.3-GPU-Qwen-snap-bK64-bdry14`** — the current best Δppl
recipe.

Decoded:

| token              | meaning |
|:-------------------|:--------|
| `v1.3`             | codec family (RSVD skeleton + spherical K-means + WHT + Lloyd-Max + outlier + PR #17 guardrails).  Matches `kakeyaturbo-bench` defaults aside from the deltas below. |
| `GPU`              | runs `kakeyaturbo_py.gpu_skeleton.fit_skeleton_batched` + `kakeyaturbo_py.gpu_encode.encode_and_pack_batched` (per-kv-head batched RSVD on CUDA), not the Rust-CLI CPU pooled-heads path. |
| `Qwen`             | model family the knobs were tuned for.  Qwen2 / Qwen3 / QwenVL all share the per-head qk-norm architecture that makes K compress differently from DeepSeek-style pre-qk-norm K. |
| `snap`             | two-pass snapshot harness (clean prefill → offline codec → replay).  Matches `benchmarks/e2e_ppl_validation_vllm_snapshot_qwen3.py`.  The in-forward harness is the `vllm_backend/kakeya_v1_3_ppl` AttentionImpl path and has separate numbers. |
| `bK64`             | K-stream K-means cluster count = 64.  Δppl-optimal point on the K budget sweep.  (The `bK` prefix is unfortunate legacy; it stands for "K-stream K-means cluster count", not "K bit width".  See the alias row below.) |
| `bdry14`           | 14-layer symmetric boundary skip: `[0..6, 29..35]` of Qwen3-4B's 36 decoder layers remain bf16 (no codec). |

The FULL parameter set under this name (everything else is
inherited from defaults; change these and the name changes):

```
--bit-width-k 4 --k-kmeans-k 64 --rsvd-target-rank-factor 0.75
--bit-width-v 2 --v-kmeans-k 16
--boundary-skip-layers 0 1 2 3 4 5 6 29 30 31 32 33 34 35
--gpu-codec --no-share-basis-v
--disable-q-precond --disable-centroids --disable-outlier
```

Measured (WikiText-103 test, ctx=2048, n_eval=64, 4 passages, H200):

* Δppl (paired, per-passage) = **+61.84 %**
* Δppl (pooled, mean PPL ratio) = **+94.17 %**
* top-1 pair agreement = **79.30 %**
* Mean ppl_ref = 11.836, mean ppl_alt = 22.982.
* One-time codec overhead = **18.08 s / passage**; steady-state
  prefill = 0.064 s / passage; peak GPU mem ≈ 56.7 GiB.
* Compression ratio: 2.20× per token on non-boundary layers,
  1.50× blended after the 14-layer bf16 boundary.

Verdict **REJECT** (MARGINAL bar is Δppl ≤ +20 %); top-1 is above
the PR #17 DS-1.5B production-cell MARGINAL top-1 of 74.22 %.

## Spoken-form alias

In conversation, Slack, code comments, and diff summaries use
**`v1.3-GPU-snapA`** as the short form of
`v1.3-GPU-Qwen-Snap-bK64-bdry14`.  The `-Qwen-*-bdry14` suffix
is the default context; only spell it out when comparing across
models or boundary depths.

## Sibling: `v1.3-GPU-Qwen-snap-rvq-4x16` (alias `snapF`)

2-level Residual VQ applied to the K-stream; same snapA base
(b_K=4, d_eff=96, bdry14, V b_V=2, k_V=16) but replaces flat
K-means (k=64) with cascaded K-means (k1=4, k2=16, k_eff=64).
Snapshot-only — implemented in `kakeyaturbo_py.gpu_encode.
roundtrip_residual_vq` (not in the vLLM slot path).

* Δppl (paired) = **+52.37 %** (−9.47 pp vs snapA, lowest Δppl
  on Qwen3-4B snapshot to date)
* top-1 = **83.98 %** (+4.68 pp vs snapA — **new high-water mark
  for Qwen3-4B snapshot**)
* K reconstruction MSE = **0.053** (10× lower than snapA's 0.503,
  but still 10× higher than TurboQuant k8v4's 0.0048)
* Centroid storage per (block, head): **3 840 B** vs snapA's
  12 288 B — **3.2× reduction** on the centroid table itself.
  If ported to the vLLM slot path, the full field-by-field
  calculation (see `FINDINGS_GPU.md`, "Theoretical slot-port
  compression ratio") gives:
  * Non-boundary compression: **2.04× → 2.31×** (**+13.2 %**)
  * Blended with 14-layer bf16 boundary: 1.45× → 1.53× (+5.5 %)
  * Net per-token-per-head: 125.3 B → 110.8 B (−14.5 B)
  * At 128 K context: **~328 MiB saved / sequence** on the
    non-boundary layers.

  Half of the centroid saving is spent on a doubled `t` field
  (RVQ stores `t₁` and `t₂` per token instead of snapA's single
  `t`), which the earlier back-of-envelope "2.10×" estimate did
  not account for; the correct projection is **2.31×**.

Honest attribution: **−8.43 pp of the −9.47 pp Δppl advantage**
vs snapA comes from the snapshot-only decode path using "absorbed"
scale semantics (always applying `‖residual‖`) whereas the slot
decoder uses "explicit" mode (sets `inv_scale=1`) for
`inner_product` metric.  Only −1.04 pp is attributable to the
RVQ structure itself (measured against the same-path k2=1 baseline).
See `FINDINGS_GPU.md` "Perron-tree-inspired Residual VQ" for the
full three-effect decomposition.

Canonical parameter set:

```
--bit-width-k 4 --k-kmeans-k 4 --k-rvq-level2 16
--rsvd-target-rank-factor 0.75
--bit-width-v 2 --v-kmeans-k 16
--boundary-skip-layers 0 1 2 3 4 5 6 29 30 31 32 33 34 35
--gpu-codec --no-share-basis-v
--disable-q-precond --disable-centroids --disable-outlier
```

Deployment caveat: snapF's advantages are snapshot-harness only.
Porting to production vLLM requires changing the slot decode path
(`inner_product`-mode to carry `‖residual‖` through the `norm`
field), which is a real slot-format change.  Recorded here for
later productionisation.

## Historical aliases (DO NOT use in new docs)

The following legacy names all refer to **`v1.3-GPU-snapA`** and
appear in existing commits / JSONs.  Kept for back-reference only:

| Legacy name                                         | Context |
|:----------------------------------------------------|:--------|
| `Recipe A`                                          | `HEADTOHEAD_vs_TQ.md`, V-shrink commit |
| "best-K config" / "best config"                     | Budget-sweep commit messages |
| "14-layer boundary + no guardrails"                 | Conversation, post-snapshot-mode refactor |
| "Qwen3-4B GPU per-head optimum"                     | Conversation, before budget sweep |
| `qwen3_4b_budget_k64_bK4_deff96_vllm_snapshot.json` | JSON file name under `budget_sweep/` |
| "Scenario A best"                                   | Paper trail from Scenario A planning docs |

## Rules going forward

1. New reports, commit messages, and conversation use
   `v1.3-GPU-snapA` or `v1.3-GPU-snapF` (or the full canonical
   name).  `snapB` has been retired (removed from documentation
   and recipe list); the JSON that used to be labelled `snapB` is
   deleted.  `snapC` (exact PCA) and `snapD` / `snapE` (extended-k
   sweep) were reverted as negative results; see repo git history
   for details.
2. Legacy JSON file names are **not** renamed — `git mv` is
   cheap but it breaks `git blame` and existing URLs in earlier
   commits.  Instead, a new report that references old data
   should name the file by its current git path and the
   canonical name in the prose.
3. When a new budget / layout is measured, it gets a NEW
   canonical name (e.g. `v1.3-GPU-Qwen-snap-bK64-bdry14-bV1` if
   we later flip V bit-width).  The spoken alias is bumped
   (`snapC`, …) in order of measurement time.  This file
   registers each alias as it's introduced.
