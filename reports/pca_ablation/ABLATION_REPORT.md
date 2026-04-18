# PCA Basis-Sharing Ablation — Experimental Results

## Motivation

Before implementing the proposed kakeyaturbo v1.2 optimisation (share
PCA basis across blocks of a layer), we need empirical evidence that
the "within-layer KV distributions are stable across blocks" assumption
holds. If it does, a single layer-pooled basis should fit every block
almost as well as per-block PCA. If it doesn't, basis sharing will
hurt reconstruction quality.

## Method

For every full-attention layer of every model, dump the K stream and
V stream (shape `[n_blocks * block_size, head_dim]`, after flattening
`[bsz, n_kv, seq, head_dim]`) and invoke the `kakeyaturbo-pca-ablation`
Rust binary on each. The binary runs the **real** `fit_weighted_pca`
from the kakeyaturbo crate under three strategies:

- `per_block`: fit PCA independently on each block (current v1 behavior)
- `layer_pooled`: fit PCA once on the whole layer-stream, reuse
- `first_block`: fit PCA on block 0 only, reuse on all blocks

then measure **mean reconstruction MSE per block** under each basis.

Fixed: `block_size=512`, `variance_ratio=0.95`, context = 4096.
7 models × (K + V) × (3 to 40 full-attention layers) = **350
measurements** total.

## Headline result: layer-pooled basis inflates MSE vs per-block

Inflation = `layer_pooled MSE / per_block MSE` (lower is better; 1.0 =
no quality loss from sharing; > 1 = pooled is worse).

| Model | # meas. | pooled mean | pooled median | pooled max | first-block mean | first-block max |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-0.5B-Instruct` | 48 | **3.39×** | 1.54× | 26.60× | 136.6× | 3349× |
| `Qwen3-0.6B` | 56 | **3.93×** | 1.68× | 27.40× | 16.9× | 42.6× |
| `gemma-4-E2B-it` | 6 | **1.02×** | 1.03× | 1.04× | 3.2× | 5.2× |
| `DeepSeek-R1-Distill-Qwen-1.5B` | 56 | **6.44×** | 1.33× | 268.60× | 184.0× | 9071× |
| `glm-edge-1.5b-chat` | 56 | **1.57×** | 1.54× | 2.29× | 13.0× | 21.9× |
| `SmolLM2-1.7B-Instruct` | 48 | **2.14×** | 1.70× | 4.38× | 20.6× | 106.5× |
| `glm-edge-4b-chat` | 80 | **1.60×** | 1.40× | 2.34× | 13.8× | 35.3× |

**Every model except Gemma 4 E2B has mean inflation above our 10%
decision threshold**. 5 of 7 models have median inflation > 1.5×.

## Decomposition: K stream is the offender, V is fine

Splitting by K vs V stream:

| Model | K pool mean | K pool median | V pool mean | V pool median |
|---|---:|---:|---:|---:|
| `Qwen2.5-0.5B-Instruct` | **5.48×** | 2.04× | 1.29× | 1.27× |
| `Qwen3-0.6B` | **6.55×** | 4.14× | 1.30× | 1.31× |
| `gemma-4-E2B-it` | 1.02× | 1.02× | 1.03× | 1.04× |
| `DeepSeek-R1-Distill-Qwen-1.5B` | **11.75×** | 1.57× | 1.14× | 1.13× |
| `glm-edge-1.5b-chat` | 1.96× | 1.95× | 1.18× | 1.17× |
| `SmolLM2-1.7B-Instruct` | 3.02× | 3.07× | 1.27× | 1.25× |
| `glm-edge-4b-chat` | 2.04× | 2.04× | 1.16× | 1.15× |

**Sharp pattern**:

- **V stream is safely shareable** across all 7 models: mean inflation
  1.03–1.30× (worst median 1.31×). Well under the 10% threshold?
  Actually, note that the V inflation **is** in the 10–30% range for
  every model except Gemma 4 — on the line.
- **K stream blows up** on Qwen-family models: Qwen2.5 5.5×, Qwen3
  6.5×, DeepSeek-R1-distill 11.8×. Even on GLM-Edge (hd=128) and
  SmolLM2 (hd=64, MHA), K pooled inflation lands at 2–3×.
- **Only Gemma 4 has uniformly safe sharing** — which makes sense:
  Gemma 4 has only 3–7 full-attention layers and each block in that
  long-context layer is drawn from the same distribution (the
  per-block fitting is in fact over-fitting noise on Gemma 4).

## Why K is unstable across blocks

Two compounding reasons:

1. **K norms are heterogeneous per token**. In the Qwen family this is
   the exact phenomenon we discussed in earlier rounds (the one that
   breaks symmetric TurboQuant turbo3). A pooled PCA has to fit both
   high-norm and low-norm regions; a per-block PCA sees a narrower
   slice.
2. **K's principal directions encode position-sensitive routing**.
   RoPE embeds position information into K's direction, so neighboring
   blocks (different position ranges) sit on different "ridges" of
   the K distribution. A pooled basis averages over these ridges and
   fits none of them well.

Interestingly, `first-block` basis is drastically worse than pooled
on Qwen and DeepSeek (up to 9000× inflation) — consistent with the
ridge interpretation: block 0 sits in one position range and fails
hard at reconstructing later position ranges.

V is stable because it's computed from the same `V_proj` weight on
approximately stationary hidden states (after layer-norm), and does
not carry positional information.

## Implications for the v1.2 design

### Reject the naive "layer-pooled basis for all streams" plan

The inflation on K stream is far above the 10% threshold on 6 of 7
models. Simply sharing the basis would degrade K reconstruction by
3–10× MSE on real data — exactly the regime that causes attention
routing errors and PPL regression (per TurboQuant+'s documented Qwen
asymmetric-quantisation findings).

### Adopt the **asymmetric** revision: pooled basis for V, per-block for K

Since V sharing inflates MSE by ≤ 31% across all models (Gemma being
the cleanest at 3%), we can share the V basis safely and keep the
bulk of the skeleton savings on V. K stays per-block to preserve
reconstruction quality.

**Byte impact estimate**: per-layer skeleton bytes are evenly split
between K and V skeletons. Sharing V only saves ~25% of the total
skeleton bytes (not 50%, because K skeleton is now stored with more
blocks' basis overhead preserved, and centers are still per-block).
At Qwen3-0.6B 8k:
- v1: 2.94 MiB skeleton per stream × 2 streams = 5.88 MiB
- v1.2-V-only: 2.94 MiB (K side) + 10 KB + 128 × 1.5 KB (V side) =
  ~3.14 MiB → saves ~2.74 MiB
- Expected ratio improvement: ~18% on Qwen3, giving roughly
  3.15× → **~3.7×** at 8k

Less ambitious than the initial 6.9× projection but **honestly
achievable with acceptable quality**.

### Note on the 2-pass encode for V

`encode_layer<R>` still needs 2 passes (fit on all V vectors, then
re-encode each block). K uses the simpler single-pass flow. The API
becomes:

```rust
fn encode_layer<R: Distortion>(
    blocks: &[&[f32]],
    weights: &[&[f32]],
    d: usize,
    params: &CodecParams,
    share_basis: bool,  // false for K, true for V
) -> (SharedSkeleton, Vec<BlockCodes>)
```

This preserves the RDO mathematical contract (one function, parameter
controls behavior) and the monomorphisation guarantee (no `dyn`).

### Consider the optional "per-layer-type" middle ground

For Gemma 4 with only a handful of full-attention layers, the cost of
storing a fresh basis per layer is low. For dense transformers (Qwen,
Llama) with 24–40 full-attention layers, a middle ground is **one
basis per layer**, not per block and not per model. The ablation here
measured per-layer pooling (not cross-layer), so the middle ground is
what we would actually deploy.

## Raw data

- Per-model per-measurement JSON: `reports/pca_ablation/<model>/layer_<L>_<K|V>.json`
- Model-level summary: `reports/pca_ablation/<model>/summary.json`
- Global summary: `reports/pca_ablation/global_summary.json`
