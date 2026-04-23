# v1.3 PPL as a production vLLM KV-cache backend — **Option C** (zero-simplification)

**Scope statement.** Deliver a deployable vLLM attention backend
`--kv-cache-dtype kakeya_v1_3_ppl` that implements the **complete**
v1.3 PPL algorithm with zero steps dropped or mocked:

  * Q-preconditioning (Σ_q Cholesky whiten)
  * Per-block PCA rank reduction
  * Per-block spherical K-means residual clustering
  * Walsh-Hadamard rotation
  * Per-coord Lloyd-Max quantization (calibrated centroids)
  * Outlier compensation T=2.0 (sparse overrides)
  * 6-layer boundary skip

Any step that an Option-A or Option-B plan would drop must be
present in the final kernel. No fallback paths. No "mock" modes.
No overfit-to-this-calibration hacks.

## Reality check: what makes Option C hard

TurboQuant PR #38479's V1 `TQFullAttentionSpec` assumes each token
occupies a **fixed** number of bytes in its paged-cache slot. That
fits K/V = "WHT rotated, then per-token quantized" cleanly: slot
size = `(bits_per_coord × head_dim + 7) / 8` bytes per token per
head, plus metadata constants per layer.

Our per-block PCA + K-means step produces **per-block state that is
NOT per-token**:

  * **PCA basis matrix** `B_l_h^{block}`: `[d_eff, D]` fp32 per block
    per (layer, KV-head). With `D = 128, d_eff = 64`, this is
    64·128·4 = **32 KB per block**.
  * **PCA mean vector** `μ_l_h^{block}`: `[D]` fp32 per block =
    **512 B per block**.
  * **K-means centroid table** `C_l_h^{block}`: `[K=16, d_eff]` fp32
    per block = **4 KB per block**.
  * **K-means weights / counts**: negligible.

For `block_size = 512` vectors this block-level skeleton is
`~37 KB / 512 ≈ 72 bytes amortised per vector` — non-negligible,
comparable to the per-vector residual itself.

The residual per vector is:
  * **K-means index**: `log2(K) = 4 bits` (which centroid this vector is
    closest to).
  * **Post-WHT Lloyd-Max residual**: `d_eff × bits` = `64 × 3 = 192 bits = 24 bytes`.
  * **Outlier overrides** (K only): ~5 % of coords as `(u16 idx, f16 val)` = 4 bytes × 0.05 × 64 ≈ 13 bytes avg for K stream.
  * **Total per K vector**: 4 + 24 + 13 = **41 B**.
  * **Total per V vector** (b=2, no outliers): 4 + 64·2/8 = **20 B**.

Baseline bf16 per vector per stream: `128 × 2 = 256 B`.

## The key design decision: where the block skeleton lives

vLLM V1 paged cache is a single `[2, num_blocks, block_size, num_kv_heads, head_size]` tensor per layer, where
`(block_size, num_kv_heads, head_size)` are **vLLM block geometry**, not our codec block geometry. vLLM's `block_size` is set by the cache config (default 16 tokens per cache block). vLLM's `head_size` here is the per-head byte footprint.

We need **two** conceptual block systems:

  1. **vLLM cache block** (vLLM's `block_size_vllm`, e.g. 16 tokens per cache block). This controls paging / block table.
  2. **Codec block** (our `block_size_codec`, e.g. 512 tokens). This is the unit over which PCA / K-means are fit.

These must be aligned. **We require `block_size_codec` to be a multiple of `block_size_vllm`**. Easiest: set vLLM `block_size = 512` via `--block-size 512`, so one vLLM cache block = one codec block. This is a supported setting; we simply require it.

With that alignment, **one vLLM cache block hosts exactly one codec block's worth of data**:

```
vLLM cache block (physical layout, proposed):

  [ HEADER        48  B  ]  ← PCA rank d_eff, K-means K, outlier count, reserved
  [ PCA basis     d_eff·D·2 B (bf16) ]   # bf16 skeleton per SPRINT_CLOSEOUT
  [ PCA mean      D·2 B ]
  [ K-means cent  K·d_eff·2 B ]
  [ Per-vector:   block_size × (4-bit km_idx + d_eff·3-bit residual) ]
  [ Outlier side-buffer (K only, variable length) ]
```

For a codec block of 512 tokens at D=128, d_eff=64, K=16, b_K=3, b_V=2 (no V outlier), the Q-preconditioned, outlier-compensated K block fits in:

| Component     | K block bytes | V block bytes |
|:--------------|--------------:|--------------:|
| Header        | 48            | 48            |
| PCA basis     | 64·128·2 = 16 384 | 16 384 |
| PCA mean      | 256           | 256            |
| K-means cent  | 16·64·2 = 2 048 | 2 048        |
| K-means idx   | 512 · 0.5 = 256 | 256 |
| Residual      | 512·64·3/8 = 12 288 | 512·64·2/8 = 8 192 |
| Outlier       | ~4 %·512·64·4 = ~5 242  | 0 |
| **Total**     | **36 522 B**  | **27 184 B**   |

Per-vector: K ≈ 71 B; V ≈ 53 B. Slot sum = **124 B**.
Baseline bf16 slot sum: 256 + 256 = 512 B.
**Compression ratio: 512 / 124 = 4.13×.**

Outlier count is data-dependent → variable-length slot. TurboQuant's V1 spec assumes **fixed** slot size per layer. We handle this by **reserving a worst-case outlier budget** (upper-bounded at 8 % of coords per block for K) so the slot size becomes a fixed upper bound. Wasted bytes when actual outlier count < budget are zero-filled. Net compression after this budget padding: ~**3.8× – 4.0×** on K+V combined, at worst-case outlier distribution.

## Cache layout interface contract

We subclass `FullAttentionSpec` with a new `KakeyaV13FullAttentionSpec` that:

  * Declares `real_page_size_bytes = slot_budget_bytes` per layer (matching the layout above).
  * Allocates the cache as a raw `torch.uint8` tensor of shape
    `[2, num_blocks, slot_budget_bytes]`. Note: **not** the standard `[2, num_blocks, block_size, n_kv_heads, head_size]` because our layout is per-block, not per-(token, head).
  * Provides `pack_into(slot: uint8[..., slot_budget], k_block: bf16[block, n_kv_heads, head_size])` — the Triton STORE kernel.
  * Provides `unpack_from(slot, slot_indices, out_k: bf16[num_tokens, n_kv_heads, head_size])` — the Triton DECODE kernel.

vLLM's attention backends call `write_to_paged_cache(key, value, kv_cache, slot_mapping, …)` at store time. We override that callsite to buffer tokens and only write when a full codec block is ready. This requires either:

  * **Buffering tokens until the codec block boundary**, then running the full PCA+K-means+WHT+Lloyd-Max pipeline on the accumulated block (the natural fit for prefill, where all prompt tokens arrive at once).
  * **During decode** (tokens arrive one at a time), we must use a different path: tokens that fall into an *already-sealed* block would need to go through the same compressed block (which is fitted). The natural solution is to **never mutate a sealed block**: once written, it stays compressed; the current decode step's token goes into a *partial-block staging area* that is kept bf16 until the next full block fills up.

**Consequence**: the paged cache has two slot types per layer:

  * **Sealed codec blocks** (compressed, full codec pipeline applied).
  * **Trailing partial block** (bf16, < `block_size_codec` tokens).

Decode reads from both: attention kernel fuses "unpack sealed block + attention" with "direct bf16 read from partial block". This is **the** core complication of Option C and **not** present in TurboQuant (TurboQuant quantizes each token independently at store time, so there's no concept of a partial block).

## Files this branch will add

```
vllm_backend/
  kakeya_v1_3_ppl/
    __init__.py                    # backend registration with vLLM's attention selector
    backend.py                     # KakeyaV13PPLAttentionBackend
    spec.py                        # KakeyaV13FullAttentionSpec (slot layout)
    calibration.py                 # Σ_q + PCA basis + K-means + Lloyd-Max (fitted offline, loaded into kernel constants)
    kernels/
      store.py                     # Triton STORE kernel (encode path)
      decode.py                    # Triton DECODE kernel (attention path)
      partial.py                   # Triton kernel for the bf16 trailing partial block
      reference.py                 # PyTorch reference for correctness tests
    tests/
      test_store_roundtrip.py      # bit-exact vs Rust reference codec
      test_decode.py               # dequant → reconstructed K ≈ original K within Lloyd-Max noise
      test_attention.py            # end-to-end attention output parity
```

And the calibration script:

```
benchmarks/
  qwen3_4b_calibration.py          # runs Σ_q + per-block PCA + K-means + LM calibration; emits a single .safetensors with all per-(layer, kv-head) constants
```

## Offline calibration deliverable

For Qwen3-4B at `block_size_codec = 512`:

| Constant                        | Shape                           | Dtype | Bytes per layer per kv-head |
|:--------------------------------|:--------------------------------|:------|:---------------------------|
| Σ_q Cholesky factor L           | [128, 128]                      | fp32  | 65 536                     |
| Σ_q Cholesky inverse L⁻¹        | [128, 128]                      | fp32  | 65 536                     |
| PCA basis per block             | [num_calib_blocks, 64, 128]     | bf16  | variable; kept as a per-block table indexed by block hash during kernel execution |
| PCA mean per block              | [num_calib_blocks, 128]         | bf16  | ditto                      |
| K-means centroids per block     | [num_calib_blocks, 16, 64]      | bf16  | ditto                      |
| Lloyd-Max centroid table (K)    | [2^b_K]                         | fp32  | 32 (for b=3)               |
| Lloyd-Max centroid table (V)    | [2^b_V]                         | fp32  | 16 (for b=2)               |

**IMPORTANT**: the per-block PCA basis, PCA mean, and K-means centroids are **runtime-fit per codec block during prefill**, not loaded from calibration. Calibration only provides Σ_q L, Σ_q L⁻¹, and the two Lloyd-Max centroid tables (K and V). The per-block statistics are recomputed in-kernel from the live K/V stream. This is identical to the HF harness semantics and to the behaviour of `kakeyaturbo/src/codec.rs`.

## Correctness gating (before any benchmark)

Nothing runs to benchmark until **bit-exact correctness** is established:

  * For ≥ 1000 random `(block, Σ_q_L, b_K, b_V, outlier_T, centroids_K, centroids_V)` triples, the Triton STORE+DECODE pipeline must produce **the same** reconstructed block (within fp32 rounding tolerance, ≤ 1e-5 relative error) as the Rust reference codec in `kakeyaturbo/src/codec.rs`.
  * Failure mode = ban: if any correctness test diverges, we FIX the kernel, we do NOT relax the tolerance, we do NOT fall back to a slower Python path, we do NOT skip the test.

## Milestones (revised for Option C)

| # | Milestone | Exit criterion |
|:-:|:----------|:---------------|
| M0 | This plan document | Written, committed |
| M1 | vLLM upgrade + Qwen3-4B download + TQ-k8v4 baseline reproduction | TQ k8v4 GSM8K ≥ 0.80 and TPOT within 10 % of published numbers on our H200 |
| M2 | Offline calibration on Qwen3-4B (Σ_q L + Lloyd-Max centroid tables only) | `.safetensors` produced, `whiten ∘ unwhiten = I` within 2e-5 |
| M3 | Rust reference codec refactored into an in-process library (no subprocess, no disk), exposed via pyo3 or cffi | Old HF harness (PR #15) uses the library, reproduces the +35.33 % Δppl it had before — proves semantic parity |
| M4 | Triton STORE kernel (encode path, all 5 steps + outlier) | Bit-exact vs Rust reference on 1000+ random triples |
| M5 | Triton DECODE kernel (full reconstruction, including partial-block path) | Bit-exact vs Rust reference; attention output matches |
| M6 | `KakeyaV13PPLAttentionBackend` + spec + registration | `vllm serve Qwen/Qwen3-4B --kv-cache-dtype kakeya_v1_3_ppl --block-size 512` produces coherent output |
| M7 | Head-to-head benchmark | One artifact table with baseline / TQ-k8v4 / TQ-4bit_nc / ours on throughput + GSM8K + NIAH + WikiText PPL + peak GPU memory |
| M8 | PR #18 opened | The table lands in the PR body |

## Non-negotiables (per user)

  1. **No simplification** of the algorithm. All 5 codec steps + 4 guardrails stay in the kernel.
  2. **No fallback paths**. If the Triton kernel fails on a case, fix the kernel, don't re-route to PyTorch/Rust CPU.
  3. **No mocking**. No "pretend you compressed it" stubs. The paged cache must actually contain the compressed bytes and the attention kernel must actually read them.
  4. **No overfitting**. Calibration data must be disjoint from the benchmark data. The calibrated centroids + Σ_q must generalize (verified by cross-split re-calibration producing ≤ 5 % drift on Δppl).

## Open engineering questions (to be resolved in M0 design sessions)

  1. **Runtime vs calibration-time PCA/K-means**: runtime per-block PCA inside a Triton kernel is nontrivial (eigendecomposition). Options: (a) Jacobi sweep in Triton (doable, roughly 10 sweeps on 128×128 SPD matrix, ~4 µs), (b) randomised PCA via HMT in Triton (one matmul + QR), (c) a separate prep kernel launched before STORE. **Decision: use (b) — randomised PCA matches `kakeyaturbo/src/pca.rs` `fit_weighted_pca_randomized` — as that is what the v1.3 codec does today. Identical math, different memory layout.** Zero semantic change.
  2. **K-means inside a Triton kernel**: 16 centroids, 64-dim, 512 vectors, 32 Lloyd iterations. ~260 K flops per block. One kernel launch per codec block is fine. Same decision: port `kakeyaturbo/src/kmeans.rs` semantics exactly, no approximations.
  3. **Outlier side-buffer with fixed slot budget**: we pad to 8 % worst-case and zero-fill unused. Alternative is a packed variable-length side-buffer — rejected because it breaks fixed-slot assumption. 8 % budget is a compression-ratio tax, not a correctness tax.
  4. **Partial-block bf16 trailing segment**: during decode, tokens accumulate in a bf16 staging buffer. When that buffer fills to `block_size_codec`, the codec kernel runs and the block is sealed. This is the ONLY situation in the codec where tokens are ever seen in "un-codec'd" form during decode — and it is bit-exactly what the HF 2-pass harness does (the teacher-force step's eval-window K/V in the HF harness are in fact bf16, not codec'd, because the codec was only applied to the prefill cache). Semantic parity preserved.

## Ban-list

Anything in this list means we've failed Option C and need to re-open the conversation with the user:

  * `if sigma < ε: return identity_codec()`
  * `# TODO: implement outlier path — stub returns zero-length side-buffer`
  * `# Note: Σ_q whitening skipped for this test — assumes Σ_q = I`
  * `# Fallback to bf16 because Triton kernel not yet implemented`
  * Calibration data that overlaps the benchmark data
  * Separate calibration per benchmark prompt
  * `ratio` reported as 4.6× when the actually-stored bytes-per-vector implies 2.6×
