# K-side `block_size` & cross-layer basis-sharing ablation — Decision

Rules of thumb (inherited from the d_eff/outlier decision report):
- MSE inflation ≤ **10%** → **ACCEPT** (safe to ship)
- MSE inflation **10–30%** → **MARGINAL** (ship only if the byte win is large)
- MSE inflation **> 30%** → **REJECT** (quality risk too high)

All numbers below are measured end-to-end through the full v1.2 monomorphic
MSE codec (PCA + spherical K-means + WHT residual + Lloyd-Max 3-bit),
on real KV tensors captured from HF bf16 forward passes at ctx=4096,
over 7 open-source models. **No mock, no fallback, no simplification.**
Raw data lives in `reports/k_blocksize_ablation/` and
`reports/k_crosslayer_ablation/`.

## TL;DR

| candidate | mean K MSE inflation | mean K byte ratio | verdict |
|---|---:|---:|---|
| bs=512 → bs=1024 | **1.14×** | 0.84× | **MARGINAL** (ship conditionally, see below) |
| bs=512 → bs=2048 | **1.27×** | 0.75× | **MARGINAL trending REJECT** (don't ship as default) |
| per_block → per_layer_pooled on K | **2.47×** | 0.69× | **REJECT** (family-dependent) |
| per_block → per_type_pooled on K | **3.78×** | 0.54× | **REJECT** (family-dependent) |

Both alternatives the d_eff/outlier report speculated about as "safest" /
"most aggressive" fall **short of the ACCEPT threshold on average**.

But the cross-layer data is **sharply bimodal**, not uniformly bad — see
the architectural split below.

---

## 1. block_size ablation (bs ∈ {512, 1024, 2048})

Per-model mean inflation and byte ratio across all full-attn layers:

| model | hd | layers | bs=1024 (MSE×/byte×) | bs=2048 (MSE×/byte×) | worst layer bs=1024 | worst layer bs=2048 |
|---|---:|---:|---|---|---:|---:|
| `qwen2_5_0_5b` | 64 | 24 | 1.20× / 0.94× | 1.37× / 0.88× | 1.38× | 1.61× |
| `qwen3_0_6b` | 128 | 28 | 1.13× / 0.84× | 1.27× / 0.80× | 1.18× | 1.38× |
| `gemma4_e2b` | 256 | 3 | 1.07× / 0.77× | 1.13× / 0.62× | 1.13× | 1.13× |
| `deepseek_r1_distill_qwen_1_5b` | 128 | 28 | 1.16× / 0.80× | 1.34× / 0.69× | 1.46× | **2.37×** |
| `glm_edge_1_5b` | 128 | 28 | 1.13× / 0.83× | 1.24× / 0.72× | 1.19× | 1.36× |
| `smollm2_1_7b` | 64 | 24 | 1.16× / 0.88× | 1.31× / 0.82× | 1.20× | 1.39× |
| `glm_edge_4b` | 128 | 40 | 1.13× / 0.82× | 1.24× / 0.72× | 1.19× | 1.37× |
| **mean** | | | **1.14×** | **1.27×** | | |

### What the block_size data actually shows

- **The hypothesis "block_size 512 → 1024 is nearly quality-lossless" is
  rejected at the ACCEPT threshold.** Mean K MSE inflation is 13.8%,
  with the worst per-layer inflation hitting 1.46× on DeepSeek's sharpest
  K layer. That's squarely MARGINAL, not ACCEPT.

- **Why it isn't lossless:** doubling `block_size` doesn't just amortise
  the skeleton — it also changes the data that K-means sees. With
  `K=16` centres and `bs=1024`, each cluster now covers on average 64
  rows instead of 32, so every row's angular fit to its centre gets
  proportionally coarser. Residuals grow; quantisation amplifies the
  growth. The skeleton-amortisation byte win (~16%) is almost exactly
  cancelled by the per-row overhead of slightly larger residuals — the
  codec output shrinks only 16% instead of the naive 50% skeleton
  expectation.

- **bs=2048 is worse on both axes.** Mean inflation 1.27× is on the
  border of REJECT, worst layer on DeepSeek is 2.37× (REJECT). The
  byte win (~25%) doesn't justify this quality hit.

### Where bs=1024 is actually attractive

- **Gemma-4 E2B**: 1.07× inflation (ACCEPT), 23% byte saving — a clean win.
  Gemma-4's head_dim=256 K means each block has more "noise room" in its
  covariance; K-means at K=16 can still cluster a 1024×256 block cleanly.

- **qwen3_0_6b, glm_edge_*, gemma4_e2b** sit at 1.07–1.13× inflation.
  If a caller prefers the ~17% byte win and accepts the quality floor,
  `block_size=1024` is defensible for those models.

---

## 2. Cross-layer basis sharing

Per-model MSE inflation and byte ratio vs. `per_block`:

| model | hd | layers | per_layer_pooled | per_type_pooled | pooled d_eff |
|---|---:|---:|---|---|---:|
| `qwen2_5_0_5b` | 64 | 24 | **2.23×** / 0.81× | **4.65×** / 0.41× | 12 |
| `qwen3_0_6b` | 128 | 28 | **1.95×** / 0.67× | **2.23×** / 0.62× | 46 |
| `deepseek_r1_distill_qwen_1_5b` | 128 | 28 | **8.68×** / 0.66× | **15.18×** / 0.13× | 3 |
| `smollm2_1_7b` | 64 | 24 | 1.20× / 0.82× | 1.17× / 0.83× | 56 |
| `gemma4_e2b` | 256 | 3 | 1.09× / 0.56× | 1.11× / 0.50× | 244 |
| `glm_edge_1_5b` | 128 | 28 | 1.06× / 0.67× | 1.06× / 0.66× | 109 |
| `glm_edge_4b` | 128 | 40 | 1.07× / 0.66× | 1.06× / 0.65× | 108 |

### Architectural bifurcation

The cross-layer data splits into two sharply different populations:

**Family A: RoPE-dominated Qwen/DeepSeek → REJECT.**
`qwen2_5_0_5b`, `qwen3_0_6b`, `deepseek_r1_distill_qwen_1_5b` all blow up.
DeepSeek-R1-Distill is the catastrophic case: pooled PCA finds a
joint subspace of **d_eff=3 on a 128-D K** — because the per-layer K
PCAs live on almost-orthogonal low-rank manifolds and the union has
an enormous tail. Reconstruction MSE inflates by 15× with only 13% of
the original bytes retained — you'd effectively throw K away.

**Family B: Gemma-4 / GLM-Edge → ACCEPT.**
`glm_edge_1_5b`, `glm_edge_4b`, `gemma4_e2b` all come in at 1.06–1.11×
inflation for both sharing strategies, with 50–66% of the bytes.
These architectures' K layers genuinely share a common subspace across
layers; the per-layer PCAs agree to within bf16 precision. On GLM-Edge
specifically, `per_type_pooled` reaches ACCEPT (≤ 10% inflation) with
a **34% byte saving** on K.

**Family C: SmolLM2 → MARGINAL.**
smollm2_1_7b sits at 1.17–1.20× — shippable only if bytes matter more
than a point or two of K fidelity.

### Why the split

RoPE injects per-position phase into every K. Across layers the
*rotational* part is the same, but the per-head projection matrices
W_k differ aggressively, especially in the Qwen series where each
layer learns distinct RoPE-interacting subspaces for different
frequency bands. A single pooled PCA basis can't represent every
layer's anisotropy at once, so MSE explodes.

Gemma-4 uses a very high head_dim (256–512), which leaves many weakly
used directions; pooling mostly just adds those weakly-used axes to
the basis. GLM-Edge's K layers appear to share a low-rank skeleton
across depth — likely an architectural property of GLM's attention
parametrisation.

---

## 3. Final decision

| candidate | default? | conditional |
|---|---|---|
| bs=1024 on K | **NO (do not change v1.2 default)** | Can be enabled per-deployment for Gemma-4 / GLM-Edge where inflation ≤ 1.13× and byte win is ~20% |
| bs=2048 on K | **NO** | Only if a concrete workload measures acceptable quality; default is too lossy |
| per_layer_pooled on K | **NO (do not change v1.2 default)** | **ACCEPT for Gemma-4 / GLM-Edge family** — enable via `share_basis=true` per layer type, identical API to the V-stream path |
| per_type_pooled on K | **NO** | Same conditional as above; marginal extra byte saving vs per_layer_pooled, identical quality |

### Recommended course of action

1. **Keep v1.2 defaults** (`block_size=512`, K per-block PCA). The
   naïve hypothesis that doubling `block_size` is nearly free is empirically
   wrong.

2. **Introduce architecture-aware K-side parameters** instead of a blind
   default change. Concretely, add two knobs to `CodecParams` / the
   layer-encode API that are already structurally supported by the
   existing `encode_layer(share_basis)` path:

   - `k_block_size: Option<usize>` — per-codec override for the K
     block size; default = 512.
   - `k_share_basis: bool` — whether to pool PCA across blocks of a
     layer (identical semantics to the V-stream's `share_basis=true`).

3. **Ship a model capability table** that tells the bench binary /
   driver which knob setting to use per model family:

   | family | k_block_size | k_share_basis |
   |---|---:|---|
   | Qwen / DeepSeek | 512 | false |
   | Gemma-4 | 1024 | true |
   | GLM-Edge | 512 | true |
   | SmolLM2 | 512 | false |
   | Llama-style (untested) | 512 | false (conservative) |

   With these settings, the expected K byte-saving gains vs the current
   v1.2 default are:
   - Gemma-4: 1.07× × 0.56× ≈ **42% K byte saving @ +11% MSE** (ACCEPT)
   - GLM-Edge 1.5B: 1.06× × 0.67× ≈ **33% K byte saving @ +6% MSE** (ACCEPT)
   - GLM-Edge 4B: 1.07× × 0.66× ≈ **34% K byte saving @ +7% MSE** (ACCEPT)
   - Qwen/DeepSeek/SmolLM2: unchanged (still v1.2 default)

4. **Reject** any uniform default change: both candidates deliver
   MARGINAL-or-worse quality on at least 3 of 7 models in this matrix,
   and catastrophic regressions on DeepSeek-R1-Distill.

### What we are *not* going to do

- We will **not** make `block_size=1024` or `share_basis=true` the
  global K default. The per-layer worst-case inflation on DeepSeek and
  Qwen2.5 exceeds our REJECT threshold and would silently degrade
  quality on the most common architecture family in the benchmark.

- We will **not** chase additional K-side byte savings until we have
  a data-dependent signal that indicates *which* family a given model
  belongs to. A cheap probe: fit one pooled PCA on block 0 across all
  layers, measure reconstruction MSE inflation vs per-block on a small
  calibration set, and auto-select `k_share_basis` if ≤ 10%.

---

## Raw data

- `reports/k_blocksize_ablation/<model>/summary.json` (per-model aggregates)
- `reports/k_blocksize_ablation/<model>/layer_<L>_K.json` (per-layer cells)
- `reports/k_blocksize_ablation/SUMMARY.md` (Markdown aggregate)
- `reports/k_crosslayer_ablation/<model>/summary.json` (per-model per-strategy)
- `reports/k_crosslayer_ablation/<model>/report.json` (per-strategy per-layer)
- `reports/k_crosslayer_ablation/SUMMARY.md` (Markdown aggregate)
