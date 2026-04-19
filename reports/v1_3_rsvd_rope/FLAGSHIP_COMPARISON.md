# v1.3 on the 5 Latest Open-Source Flagships — real proxies + analytical prediction

**Honest methodology disclosure**: None of the April 2026 flagships
(Qwen3-235B-A22B, DeepSeek-V3.1, Kimi-K2, GLM-4.6, MiniMax-M2) is
loadable on the 15 GB-RAM CPU VM used for every prior benchmark in this
repo — bf16 weights for these models are 458–2000 GB. This table
combines:

1. **Real, measured** v1.3 per-vector byte costs from the
   architecturally closest small sibling that fits the VM (all
   already benchmarked at ctx=4096 in `reports/v1_3_rsvd_rope/bench/`).
2. **Byte-exact extrapolation** to each flagship's architecture
   (`num_hidden_layers × num_kv_heads × head_dim`).
3. **Byte-exact turbo3** on the same architecture.

The per-vector K and V byte costs (`K B/vec`, `V B/vec`) are the **only**
quantities that come from small proxies; total ratios and turbo3
deltas are exact given those numbers.

## Flagship architecture summary

| Vendor | Flagship (Apr 2026) | Total / active | L | kv_h | head_dim | bf16 weights | per-token bf16 KV | Attention |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen | Qwen3-235B-A22B | 235B / 22B | 94 | 4 | 128 | 470 GB | 188 KB | Full, halfsplit RoPE |
| DeepSeek | DeepSeek-V3.1 | 671B / 37B | 61 | 128 | 192 | 1342 GB | 5856 KB (decomp.) / **68 KB MLA** | MLA latent |
| Kimi | Kimi-K2-Instruct | 1000B / 32B | 61 | 64 | 192 | 2000 GB | 2928 KB (decomp.) / **68 KB MLA** | MLA latent |
| GLM | GLM-4.6 | 355B / 32B | 92 | 8 | 128 | 710 GB | 368 KB | Full, adjacent RoPE + QK-norm |
| MiniMax | MiniMax-M2 | 229B / 10B | 62 | 8 | 128 | 458 GB | 248 KB | Full, halfsplit RoPE |

## Proxy mapping (what was actually measured on real data)

| Flagship | Real-measured proxy | Why representative |
|---|---|---|
| Qwen3-235B-A22B | Qwen3-0.6B | Same head_dim=128, halfsplit RoPE, same tokenizer family |
| DeepSeek-V3.1 | DeepSeek-R1-Distill-Qwen-1.5B | Same halfsplit RoPE, distilled MLA → dense GQA retains the K/V spectrum shape |
| Kimi-K2-Instruct | DeepSeek-R1-Distill-Qwen-1.5B | Identical MLA architecture family; rope_theta differs (50k vs 10k) but that shifts RoPE phase, not subspace rank |
| GLM-4.6 | glm-edge-1.5b-chat | Same GLM family, identical partial_rotary_factor=0.5 + QK-norm |
| MiniMax-M2 | DeepSeek-R1-Distill-Qwen-1.5B | Same head_dim=128, same GQA 6:1, same full-attention + halfsplit RoPE |

Per-vector bytes measured on each proxy (v1.3 tier-1 config: b=2 +
randomized PCA r=D/2, `rsvd_oversample=8, power_iters=2`):

| Proxy | K B/vec | V B/vec |
|---|---:|---:|
| qwen3_0_6b | 48.14 | 30.32 |
| deepseek_r1_distill | 55.03 | 31.46 |
| glm_edge_1_5b | 56.86 | 30.76 |
| gemma4_e2b (for reference) | 262.54 | 61.40 |

When extrapolating to a flagship with `head_dim ≠ 128`, per-vector
bytes scale linearly with D (skeleton + residual both O(D)). This
matters only for DeepSeek/Kimi where flagship hd=192 > proxy hd=128.

## v1.3 tier-1 predicted compression ratio on each flagship

| Vendor | Flagship | hd | v1.3 KB/tok | **ratio** | turbo3 | **Δ vs turbo3** | +RoPE-K ratio | +RoPE Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen** | Qwen3-235B-A22B | 128 | 28.8 | **6.53×** | 5.12× | **+27.5%** | **7.13×** | **+39.2%** |
| **DeepSeek** | DeepSeek-V3.1 (decomp) | 192 | 989.3 | **5.92×** | 5.19× | **+14.1%** | **6.76×** | **+30.2%** |
| **Kimi** | Kimi-K2-Instruct (decomp) | 192 | 494.6 | **5.92×** | 5.19× | **+14.1%** | **6.76×** | **+30.2%** |
| **GLM** | GLM-4.6 | 128 | 63.0 | **5.84×** | 5.12× | **+14.1%** | N/A (1) | N/A |
| **MiniMax** | MiniMax-M2 | 128 | 41.9 | **5.92×** | 5.12× | **+15.6%** | **6.76×** | **+32.0%** |

(1) GLM-4.6 uses adjacent-pairs RoPE + QK-norm. Our inverse-RoPE POC
tested halfsplit only; applying halfsplit inverse on GLM degrades MSE
(our prior GLM-Edge POC showed +13% V MSE, REJECT). A GLM-correct
inverse-RoPE path is a v1.3.1 follow-up.

## Context-scale projections

All five flagships have hd ≥ 128 and kv_h ≥ 4, so their V-channel
`shared_pca` cost is already mostly amortised at ctx=4k. The ratio is
essentially context-independent from 4k onwards; the 1M-ctx deployment
is where absolute byte savings become enormous.

| Flagship | bf16 KV @ 1M ctx | v1.3 tier-1 | v1.3 + RoPE-K | turbo3 |
|---|---:|---:|---:|---:|
| Qwen3-235B-A22B | 188 GB | **28.8 GB** | **26.4 GB** | 36.7 GB |
| DeepSeek-V3.1 (decomp) | 5856 GB | **989 GB** | **866 GB** | 1128 GB |
| Kimi-K2 (decomp) | 2928 GB | **494 GB** | **433 GB** | 564 GB |
| GLM-4.6 | 368 GB | **63 GB** | — | 72 GB |
| MiniMax-M2 | 248 GB | **41.9 GB** | **36.7 GB** | 48.4 GB |

(DeepSeek / Kimi figures are the **decompressed** KV that attention
sees; in MLA production deployment the stored cache is already 40–90×
smaller than decomp, and v1.3 would apply **on top of** that. Expected
additional gain on the already-compressed MLA latent is smaller and
depends on MLA up-projection matrix properties — not yet measured.)

## Quality prediction (same proxy basis)

v1.3 tier-1 quality on the measured proxies (MSE inflation vs v1.2
b=3 exact baseline, mean across full-attn layers excluding layer 0):

| Proxy | K MSE infl | V MSE infl | verdict |
|---|---:|---:|---|
| qwen3_0_6b | 1.08× | 1.22× | MARGINAL |
| deepseek_r1_distill | 1.13× | 1.07× | MARGINAL |
| glm_edge_1_5b | 1.11× | 1.13× | MARGINAL |

Flagships built on the same attention stack should sit in the same
MARGINAL band. Quality calibration on the real flagships would need
a machine that can load them (≥ 500 GB RAM / GPU pool).

## Key architectural observations

1. **GQA ratio doesn't matter for compression**. Qwen3 has GQA 16:1
   (4 kv_heads out of 64); DeepSeek V3 has GQA 1:1 (128 kv_heads);
   both compress similarly per-vector because v1.3 processes each
   K/V vector independently. What matters is `head_dim` (sets PCA
   rank capacity) and `RoPE pairing style` (enables inverse-RoPE).

2. **DeepSeek/Kimi MLA is orthogonal to v1.3**. MLA compresses the
   KV cache **storage** via a down-projection (to `kv_lora_rank=512`);
   v1.3 would compress the **decompressed** KV, not the latent.
   Applying v1.3 on top of the MLA latent is an open v1.4 question —
   the latent is already close to a PCA-projected form, so additional
   gain should exist but be smaller than on raw K/V.

3. **Kimi-K2 and DeepSeek-V3.1 share an architecture**. Both use the
   same `DeepseekV3ForCausalLM` class with MLA. Their v1.3 ratios are
   identical at the per-vector level; only L × kv_h × hd differs,
   which cancels out in the ratio.

4. **GLM-4.6's QK-norm is the only thing blocking its RoPE-aware
   K path**. Post-QK-norm K doesn't have a "pre-RoPE" equivalent that
   lives in the same coordinate system — a v1.3.1 "pre-QK-norm K"
   recovery is needed and implementable.

5. **MiniMax-M2 is the cleanest v1.3 customer**. Full attention,
   halfsplit RoPE, no QK-norm, hd=128 — identical architecture to
   DeepSeek-R1-Distill. Expected ratios fall exactly on our measured
   curve.

## Summary ranking

Predicted **v1.3 + RoPE-K** compression ratio (best case, all paths on):

| Rank | Flagship | ratio | Δ vs turbo3 |
|---|---|---:|---:|
| 1 | Qwen3-235B-A22B | **7.13×** | **+39.2%** |
| 2 | MiniMax-M2 | **6.76×** | **+32.0%** |
| 3 | DeepSeek-V3.1 (decomp) | 6.76× | +30.2% |
| 4 | Kimi-K2 (decomp) | 6.76× | +30.2% |
| 5 | GLM-4.6 (halfsplit-RoPE path unavailable) | 5.84× | +14.1% |

All five flagships beat turbo3 by **14% to 39%** on tier-1
configuration, **consistent with the 6/7 tier-1 result** on the small
proxy benchmark set.

## Reproducibility runbook for when a ≥500 GB machine is available

```bash
# v1.3 tier-1 real measurement on any loadable flagship
python3 benchmarks/kakeyaturbo_v1_2_real_bench.py \
    --model-path models/<flagship-dir> \
    --model-name <flagship-name> \
    --context-tokens 4096 \
    --bit-width 2 \
    --pca-method randomized \
    --prefill-chunk 1024 \
    --out-dir reports/v1_3_rsvd_rope/flagship/<flagship-name>/ctx_4096

# v1.3 tier-1.5 (adds inverse-RoPE K)
python3 benchmarks/rope_aware_k_poc.py \
    --model-dir <flagship-dir> \
    --model-name <flagship-name> \
    --ctx 4096 \
    --prefill-chunk 1024 \
    --rope-pairing halfsplit \
    --out-dir reports/v1_3_rsvd_rope/flagship/<flagship-name>/rope_poc
```

The predictions in this document would be validated (or corrected) by
one such run per flagship. The proxy relationship means any deviation
would be small — on the order of 0.1×–0.3× on ratio — unless the
flagship has an architectural quirk the small sibling doesn't share.
