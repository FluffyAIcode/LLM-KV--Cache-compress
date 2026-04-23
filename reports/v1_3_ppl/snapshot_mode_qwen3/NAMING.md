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

## Sibling: `v1.3-GPU-Qwen-snap-bK128-bdry14` (alias `snapB`)

Same recipe but `--k-kmeans-k 128`.  This is the top-1-agreement
optimum on the K budget sweep:

* Δppl (paired) = +65.98 %
* Δppl (pooled) = +90.41 %
* top-1 = **81.64 %** (+2.34 pp vs snapA)
* Compression ratio: 2.00×, blended 1.44×.

Use snapB when the downstream eval is argmax-based (MMLU-style,
GSM8K extraction, etc.) and top-1 trumps logprob spread.  Use
snapA when it's perplexity or LM-eval-harness.

## Sibling: `v1.3-GPU-Qwen-snap-bK64-bdry14-pcaExact` (alias `snapC`)

Same budget as snapA (`bK64`, `bdry14`, `bit_width_k=4`,
`rsvd_target_rank_factor=0.75`) but swaps the K-stream PCA stage
from RSVD (HMT 2011 Alg 4.4) to **exact PCA** via
`torch.linalg.svd` on the centred K matrix.

Motivation: snapA's K-reconstruction MSE is 0.503 — 100× worse
than TurboQuant k8v4 at matching compression ratio.  The K-MSE
origin is **not** the Lloyd-Max residual layer, the WHT stage,
or Σ_q; it is the PCA skeleton.  RSVD with Gaussian Ω has a
well-known failure mode on flat-spectrum inputs: when
σ_{r+1} / σ_1 is close to 1, the Gaussian sketch loses angular
discrimination and the recovered top-`r` basis drifts far from
the true PCA subspace.  Qwen3-4B's post-qk-norm K has exactly
that property: on a representative layer, top-96 PCA captures
only 97 % of the variance and σ_{97..128} are ≈3 % of σ_1.
Measured at this recipe:

| PCA stage                | K-MSE   | Ratio vs TQ k8v4 (0.0048) |
|:-------------------------|:--------|:--------------------------|
| RSVD (snapA)             | 0.5030  | 104× worse                |
| exact SVD (snapC ceiling)| 0.0115  | 2.4× worse                |

Exact PCA removes the RSVD approximation error entirely and
closes ~44× of the K-MSE gap, at the cost of replacing the
O(n·d·r) sketch with an O(n·d·min(n,d)) thin-SVD.  At Qwen3-4B's
block_size=512 and head_dim=128 this is still comfortably under
a millisecond per block on H200 — exact SVD is the efficient
choice whenever n ≲ 1024.

**Relationship between exact SVD and exact PCA:** exact PCA on a
centred design matrix A ∈ ℝ^{n×d} is defined as the eigendecomp
of Σ = AᵀA/n.  Thin-SVD of A gives A = UΣV^⊤; the columns of V
are the eigenvectors of AᵀA, and the PCA eigenvalues are σ_i²/n.
Since we only need the right singular vectors and V is the
top-d_eff rows of Vh from `torch.linalg.svd(A, full_matrices=False)`,
**"exact PCA via SVD" and "exact SVD on centred data" describe
the same operation** — no approximation either way, just two
ways of naming it.  The `pca_kind="exact"` code path uses the
SVD form because it's numerically more stable than forming AᵀA.

Canonical parameter set:

```
--bit-width-k 4 --k-kmeans-k 64 --rsvd-target-rank-factor 0.75
--bit-width-v 2 --v-kmeans-k 16
--boundary-skip-layers 0 1 2 3 4 5 6 29 30 31 32 33 34 35
--gpu-codec --no-share-basis-v
--disable-q-precond --disable-centroids --disable-outlier
--k-pca-kind exact
```

Expected outcome of switching snapA → snapC: K-MSE drops from
0.503 to ~0.012; Δppl and top-1 gated on whether that K-MSE
reduction propagates to the attention output (pooled Δppl is
expected to drop meaningfully, but the exact magnitude is
measured, not predicted).  The measured row is filled in by the
4-passage run logged in `reports/v1_3_ppl/snapshot_mode_qwen3/
FINDINGS_GPU.md` under "pcaExact ablation".

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

And for **`v1.3-GPU-snapB`**:

| Legacy name                                   | Context |
|:----------------------------------------------|:--------|
| `Recipe B`                                    | V-shrink commit |
| "top-1 optimum" / "top-1-best"                | Conversation |
| `qwen3_4b_budget_kK128_vllm_snapshot.json`    | JSON file name |

## Rules going forward

1. New reports, commit messages, and conversation use
   `v1.3-GPU-snapA` / `snapB` or the full canonical name.
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
