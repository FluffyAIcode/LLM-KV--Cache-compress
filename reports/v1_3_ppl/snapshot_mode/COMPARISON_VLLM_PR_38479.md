# Comparison: PR #17 (ours) vs vLLM upstream PR #38479 (TurboQuant)

**Sources.**
- vLLM PR: https://github.com/vllm-project/vllm/pull/38479 (merged 2026-04-15)
- Our PR #17: snapshot-mode on vLLM 0.7.3, DS-Distill-Qwen-1.5B, v1.3 PPL recipe.

## TL;DR

1. **PR #38479's PR-description numbers and its actual community-verified numbers are very different.** The headline table claims `tq3` ≈ 4.9× with 0.72 GSM8K; multiple independent reproductions (Neural Magic, Finland Verda cloud, a community contributor on A4000) report `tq3` defaults → **near-zero** on GSM8K (0.009), coherent output only with `TQ_VALUE_BITS=8`, which drops compression to ~2×.
2. The **reliable** operating point of PR #38479 is **`k8v4`** (FP8 keys + 4-bit uniform values) at **2.6× compression, ~96 % GSM8K retention**. That point is one tier *less aggressive* than our v1.3 PPL 4.61× target, and it uses substantially more bits per K coordinate (FP8 = 8 bits) than our K b=3.
3. Their codec math is **different from ours** in ways that matter: no PCA rank reduction, no Q-preconditioning, no outlier compensation, but with norm-correction and per-head quantization, and **fused Triton kernels that keep the dequant → attention path numerically tight** (no fp32↔bf16 CPU round-trip).

## Side-by-side table

| Axis | vLLM #38479 (merged)                            | Our PR #17 (snapshot-mode)                        |
|:-----|:------------------------------------------------|:--------------------------------------------------|
| Quality metric reported    | GSM8K accuracy, NIAH probes     | WikiText-103 PPL Δ, top-1 agreement |
| Model tested               | Qwen3-4B (head_dim=128)         | DS-Distill-Qwen-1.5B (head_dim=128) |
| Context length             | 4K – 32K (NIAH probes)          | 2048 + 64 (teacher-force PPL window) |
| Compression location       | **Online, store-time fused Triton kernel** in FA v1 backend | Offline snapshot via Rust CPU subprocess, substituted back via hook |
| Decompression location     | **In-Triton-kernel**, fused with attention kernel | Full fp32 decode from CPU, returned as dense bf16 tensor |
| K path                     | WHT rotation on raw K → Lloyd-Max scalar quant (Gaussian centroids) → bit-pack | PCA rank=D/2 + WHT + Lloyd-Max (real-data calibrated centroids) + outlier T=2.0 |
| V path                     | Uniform scalar quant + norm correction | PCA + WHT + Lloyd-Max share_basis_v (real-data calibrated centroids) |
| Per-head quantization      | **Yes** — store is per-(token, head, bit)         | **No** — we reshape `[num_tokens × num_kv_heads, head_dim]` and pool all heads in the same 512-row PCA block |
| Q-preconditioning (Chol Σ_q) | ❌ Not present                  | ✅ Present (K-only, whitening before codec)        |
| Outlier compensation       | ❌ Not present                  | ✅ T=2.0, ~4.5 % K residual coords stored as sparse f16 |
| Norm correction (NC)       | ✅ Present (centroid renorm before inverse rotation) | ❌ Not present |
| Boundary-layer skip        | ✅ `--kv-cache-dtype-skip-layers 0,1,L-2,L-1` | ✅ 6-layer skip `[0,1,7,14,26,27]` (DS-Distill) |
| Calibration                | ✅ Offline Lloyd-Max centroids fit on pooled real data |
| Gaussian defaults          | ✅ Default Lloyd-Max from N(0,1) — no calibration step |
| Asymmetric K/V precision   | ✅ Supported via presets         | ✅ K b=3 / V b=2                                  |
| vLLM integration           | Full V1 attention-backend subclass (`TurboQuantAttentionBackend`) + CUDAGraph fixes + stream overlap | Research harness via `Qwen2Attention.forward` monkey-patch, V0 engine, `enforce_eager=True` |
| Throughput cost            | **79 % – 100 % of FP16 baseline tok/s** (k8v4), reported on RTX PRO 6000 | Research measurement, not optimized for throughput |

## Community-verified quality numbers (from PR #38479 comments)

| Config | Keys | Values | Compression | Reported GSM8K | Comment |
|:-------|:----:|:------:|:----------:|:---------------:|:--------|
| baseline | FP16 | FP16 | 1.0× | 0.900 | |
| **`k8v4`** | **FP8 E4M3** | **4-bit uniform** | **2.6×** | **0.860** | **community-verified stable** |
| `4bit_nc` | 4-bit MSE+NC | 4-bit uniform+NC | 3.8× | 0.840 | varjoranta: "4-bit MSE pack path broken in `tq4`, garbage output" |
| `k3v4_nc` | 3-bit MSE+NC | 4-bit uniform+NC | 4.3× | 0.780 | N/A |
| `3bit_nc` | 3-bit MSE+NC | 3-bit uniform+NC | 4.9× | 0.720 | mgoin: **independent repro scored 0.009 on GSM8K**, tq4 crashes |
| `tq3` + `TQ_VALUE_BITS=8` | 2-bit MSE | FP8 | ~2× | **100 % baseline** | MidasMining / varjoranta: this is "the safe shipping default" |

**Interpretation.** The usable high-compression point in PR #38479 is still around `k8v4` at 2.6× — which makes K carry ~8 bits (FP8) of per-coord precision rather than 3. When community reproductions pushed to 3-bit K + 2-bit V (matching our recipe's compression ratio), quality collapsed just as badly as ours on vLLM does.

## Our PR #17 quality (same production recipe across engines)

| Harness | Engine | Δppl | top-1 | Verdict | Compression |
|:--------|:------|-----:|------:|:-------:|:----------:|
| HF 2-pass DynamicCache | HF eager | **+7.82 %** | 78.97 % | MARGINAL | 4.61× |
| **vLLM snapshot-mode (PR #17)** | vLLM 0.7.3 FA | **+29.07 %** | 74.22 % | REJECT | 4.61× |
| vLLM in-forward (PR #15) | vLLM 0.7.3 FA | +35.33 % | 59.38 % | REJECT | 4.61× |

## Why their "good" numbers are not directly comparable to ours

1. **Different quality metric.** GSM8K accuracy is a *downstream-task* metric; WikiText-103 teacher-force PPL is a *next-token-distribution* metric. A codec that hurts the 95th-percentile logit tail can leave GSM8K nearly untouched (because the model still picks the right intermediate steps) while Δppl looks catastrophic. **GSM8K @ 96 % is NOT a claim of "PPL is preserved"**; it is a weaker claim that the answer chain-of-thought is preserved.
2. **Different compression ratios.** Their verified-stable config is 2.6×. Our target is 4.61×. A head-to-head comparison at matched compression would put theirs at `3bit_nc` (4.9×) — which the community couldn't get to work — or our codec at K b=4 + V b=4 which we measured at **+27.3 % Δppl** on vLLM, roughly matching their failure point.
3. **Different model.** Qwen3-4B is their test model; DS-Distill-Qwen-1.5B is ours. DS-Distill is a heavily distilled-from-DeepSeek model with unusual logit distributions; Qwen3-4B is closer to a standard instruction-tuned transformer.

## Techniques in PR #38479 that we don't have, and their likely impact on our PPL gap

### 1. **Fused in-kernel decompression** (probably the biggest win)

PR #38479's decode path is a **single Triton kernel** that:
```
cache → unpack K → dequant → Q·K scores → softmax → ··· → output
```
bf16 → quant → bit-pack happens in one GPU kernel at store; dequant → FA math happens in one GPU kernel at decode. **No fp32 intermediates leak into the residual stream.**

Our PR #17 snapshot path:
```
capture pass: GPU bf16 → .cpu().to(fp32).numpy() → Rust process (disk KKTV I/O) → codec → disk → numpy → .cuda().to(fp32)
replace pass: substitute the reconstructed fp32 tensors as K/V input to FA
```
Each layer's replacement tensor enters FA **in fp32** (we cast back to bf16 inside the hook). Even with the hook's fp32→bf16 right before FA, the **internal path through FA's bf16 softmax** has to integrate the codec's error against that bf16-accumulated attention score. The merged PR argues, and our noise-sensitivity curve from Phase 2 is consistent with, that **bf16 softmax has a rougher response to structured error than the "noise" test suggests**, so keeping dequant in-kernel and fusing into the Q·K kernel avoids some of that roughness.

**Concrete hypothesis:** Porting our codec into a fused Triton decode kernel would likely move our PR #17 Δppl closer to HF's +8 % without any algorithm changes.

### 2. **Per-head quantization**

Their store kernel quantizes per (token, head, stream) — each head has its own Lloyd-Max bucketing. Ours pools `[num_tokens, num_kv_heads]` into a single `(N, head_dim)` stream, so the PCA basis and the K-means codebook are shared across heads within one block. DS-Distill has 2 KV heads (GQA), so only 2 heads share a basis — less harmful than a model with 32 heads, but the effect isn't zero. Per-head quantization preserves distinct distributions of head statistics (which the codec eats into the shared basis otherwise).

**Concrete hypothesis:** Per-head PCA + codec would close 1-3 pp of Δppl on DS-Distill and more on models with more KV heads.

### 3. **Norm correction (NC)**

After inverse rotation, quantization has shifted the codebook vectors' norms. NC re-normalizes reconstructed centroid vectors to match the original pre-quantization norm. The PR claims ~0.8 % PPL improvement at 4-bit. We don't do this; our inverse-WHT preserves the rotation's orthogonality but not the quantizer's norm preservation.

**Concrete hypothesis:** +1-2 pp on Δppl.

### 4. **No PCA rank reduction**

PR #38479 keeps all `head_dim = 128` dimensions (WHT is `128 → 128`). We PCA-reduce to `d_eff = head_dim × variance_ratio` or `head_dim / 2`. Rank reduction throws away some fraction of the K energy at the skeleton level. At high compression that matters.

**Concrete hypothesis:** +3-5 pp on Δppl, cost: compression ratio drops from 4.6× to ~3.5×.

### 5. **Calibrated Lloyd-Max centroids — we have this, they don't**

Ironically, this is a feature of OUR codec that their verified-stable config lacks. Their `k8v4` uses FP8 (not Lloyd-Max at all) for K, and their MSE presets use Gaussian-default centroids. We fit real-data Lloyd-Max offline. The empirical Lloyd-Max improvement over Gaussian is ~1.47× MSE gain at K b=2 (PR #13 data).

### 6. **Q-preconditioning — we have this, they don't**

Our Chol Σ_q pre-whitening of K aligns the codec's MSE metric with the Σ_q-weighted distortion that attention actually "sees". Their WHT-only rotation does not. Phase 2 of our gap decomposition showed Q-precond helps significantly (~4× reduction in Δppl relative to "no Q-precond" at b=2).

### 7. **Outlier compensation — we have this, they don't**

Our K residual outlier path (T=2.0, sparse f16 overrides on the ~4.5 % of coords that exceed the threshold). SPRINT_CLOSEOUT's evidence is that this closes ~8 pp of Δppl on the HF v1.3 PPL cell. They use uniform / Lloyd-Max without any outlier path.

## What PR #38479 would look like if ported our algorithm

Adding our algorithmic pieces to their engineering:

1. Our **Q-preconditioning** → their K store kernel
2. Our **real-data Lloyd-Max centroids** → their centroid table
3. Our **outlier compensation** → their K path
4. Keep their **per-head quantization + NC + fused Triton**

This would combine the **numerical tightness of their store/decode path** with the **algorithmic guardrails** of our v1.3 PPL recipe. At matched 4.6× compression, the expected Δppl on vLLM would likely be somewhere between HF's +7.82 % and our current +29 %, though I don't have an empirical number without running it.

## What PR #17 would look like if it used their engineering

Adding their engineering pieces to our algorithm:

1. Our full v1.3 PPL algorithm **in a fused Triton store/decode kernel**
2. Per-head quantization (eliminate our `[tokens × n_kv]` block pooling)
3. Add norm correction after inverse WHT

This would likely:
- Close the ~11 pp "intrinsic engine" bucket identified in PR #17's revised decomposition (most of it is the CPU round-trip + fp32/bf16 boundary issue)
- Close a few more pp from per-head quantization and NC

The resulting number on vLLM at 4.6× might reach **Δppl ~+10 %, top-1 > 78 %**, closing the HF-vs-vLLM gap substantially — at the engineering cost of writing Triton kernels.

## Recommended next steps for this branch

Ranked by cost/benefit:

1. **Add norm correction to our snapshot-mode harness** (small change). Re-run PR #17. Expected: ~1-2 pp Δppl improvement. (Low risk, easy win.)
2. **Switch to per-head quantization** (medium change to Python side; no Rust change required). Re-run PR #17. Expected: 1-3 pp Δppl improvement.
3. **Fuse dequant into vLLM's FA decode path (Triton rewrite)**. Large engineering project; would require cloning TurboQuantAttentionBackend structure and substituting our codec math. Expected: most of the remaining engine-level bucket.
4. **Port our algorithm into PR #38479's backend** — i.e. submit a follow-up PR upstream that adds Q-precond + calibrated Lloyd-Max + outlier compensation to TurboQuant. That's where the two efforts converge: their engineering + our algorithm.

Option 4 is probably the right long-term play for the project, since PR #38479 has already landed upstream (merged April 15) and is the surface on which vLLM KV quantization will evolve. Options 1 and 2 can be done on this branch to characterise their delta before attempting option 3 or 4.
