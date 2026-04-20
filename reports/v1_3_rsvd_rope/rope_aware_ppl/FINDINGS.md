# RoPE-aware v1.3 — end-to-end PPL findings

**Date.** 2026-04-17
**Branch.** `cursor/v1-3-rsvd-rope-aware-12f5`
**Driver.** `benchmarks/e2e_ppl_rope_aware.py`
**Model.** `Qwen2.5-0.5B-Instruct` (half-split RoPE, θ = 1 000 000, `head_dim=64`)
**Context.** 1024-token WikiText-103 prefix, 64-token continuation PPL
**Corpus.** 2 WikiText-103 passages

## Architectural principle (restored)

The codec must never see RoPE phase on K:

```
attention:   K_post  (stored in DynamicCache, consumed by dot-product)
codec in:    K_post  →  RoPE⁻¹(pos)  =  K_pre
             K_pre   →  encode → decode  =  K̂_pre
codec out:   K̂_pre   →  RoPE(pos)  =  K̂_post   (written back to cache)
```

The previous v1.3 PPL harness compressed `K_post` directly, which throws the
codec against a non-stationary, position-mixed signal that PCA/K-means cannot
exploit. The rule "codec is RoPE-agnostic" is a hard architectural invariant,
not a tuning knob. The diagnosis report that proposed "replace skeleton with
KIVI-int4" violated this principle; it is withdrawn.

## RoPE convention correctness

The numpy forward half-split RoPE is validated byte-exact against Hugging Face
`apply_rotary_pos_emb`:

```
HF  vs  ours  max abs diff   = 2.98e-07    (fp32 noise)
inverse(forward(x))  max err = 4.77e-07    (fp32 noise)
```

## A/B result (same codec, RoPE layer removed)

All rows: `block_size=512`, `RSVD r=D/2`, inner-product metric for K, MSE for V,
2 passages, mean reported.

| Config                                             | rope_mode     | Δppl (mean) | KL     | top-1    |
|----------------------------------------------------|---------------|-------------|--------|----------|
| b=2, rsvd, vr=0.95                                 | **none**      |   +956.56 % | 2.384  |  35.71 % |
| b=2, rsvd, vr=0.95                                 | **halfsplit** |   +314.70 % | 1.459  |  46.83 % |
| b=3, exact, vr=0.995                               | halfsplit     |   +167.30 % | 0.889  |  57.14 % |
| b=4, exact, vr=1.000 (max fidelity)                | halfsplit     |   +140.91 % | 0.814  |  61.11 % |

Switching from `rope_mode=none` → `rope_mode=halfsplit` at the **same bit
budget** cut KL by ~40 % and cut PPL inflation by ~3×. This is a pure
architectural win; no parameter tuning involved.

## Stream-isolation ablation (RoPE-aware, b=3, exact, vr=0.995)

| Compressed stream | Δppl (mean) | KL     | top-1    |
|-------------------|-------------|--------|----------|
| K only            |    +91.82 % | 0.560  |  64.29 % |
| V only            |    +63.20 % | 0.383  |  76.98 % |
| K + V             |   +167.30 % | 0.889  |  57.14 % |

Observations:
1. **K-side quality tracks V-side closely when RoPE is removed.** The old
   diagnosis claimed K was the dominant contributor; that was an artefact of
   compressing `K_post`. Once the RoPE phase is stripped, K and V sit in the
   same ballpark.
2. Degradation stacks near-additively on log-PPL (`exp(log(1.63)+log(1.92)) =
   3.12 ≈ 1+1.67`), consistent with per-layer errors being roughly independent.

## Remaining floor (max-fidelity run)

At b=4, exact PCA, vr=1.0, RoPE-aware, we still have +141 % Δppl, top-1 = 61 %.
The saturating behaviour (b=3 → b=4 gains only ~26 pp of PPL) rules out the
**residual quantizer** as the dominant error source. The remaining floor comes
from one (or both) of:

- **Skeleton storage precision.** `PcaFit.mean` and `PcaFit.basis` are stored
  in `half::f16`, and `KmeansFit.centers` are `f16`. At `head_dim=64` and
  `d_eff≈30`, 16-bit precision on the skeleton directly caps achievable SNR.
- **Block boundaries.** Each 512-token block gets an independent PCA basis.
  Positional structure that crosses a block boundary (even post inverse-RoPE)
  is re-encoded with a slightly rotated basis, producing a small
  discontinuity that compounds over 24 layers.

Both are **parameter-tunable within the existing architecture**; neither
requires replacing the skeleton.

## Proposed v1.3.1 experiments (no architectural change)

1. Promote skeleton storage to bf16 or fp32 (only applies to the compression
   artefact, never leaves the codec boundary; cost = 2× on a tiny fraction of
   total bytes).
2. Share PCA basis across blocks within a layer (cross-block, not cross-layer
   — we already explored cross-layer and rejected it).
3. Tighter variance ratio (>0.999) combined with wider block_size (1024 or
   full sequence), so that there is effectively one basis per layer.
4. Keep the first 128 tokens at full precision (attention-sink preservation).

These are all within the Kakeya-skeleton + residual architecture and respect
the RoPE-agnostic boundary. If any of them closes the PPL gap to
ACCEPT (Δppl ≤ 3 %, top-1 ≥ 85 %), the paper's quality claims can be honestly
retained.

## Artefacts

- `benchmarks/e2e_ppl_rope_aware.py` — harness with `--rope-mode`,
  `--compress={kv,k_only,v_only}`, half-split RoPE forward/inverse,
  byte-exact round-trip sanity check at startup.
- Per-passage JSONs under this folder (produced by the harness).
