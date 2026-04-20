# Pre-RoPE cache — end-to-end PPL findings

**Date.** 2026-04-17
**Branch.** `cursor/v1-3-rsvd-rope-aware-12f5`
**Driver.** `benchmarks/e2e_ppl_pre_rope.py` + `benchmarks/pre_rope_cache.py`
**Model.** `Qwen2.5-0.5B-Instruct`, 1024-token WikiText-103 prefix, 64-token PPL

## What changed from the earlier wrapper approach

Earlier we applied `RoPE⁻¹` on K_post before encoding and `RoPE` on K̂_pre
after decoding — mathematically correct but architecturally wrong, because
production inference engines (vLLM, SGLang, TRT-LLM) never store post-RoPE K:
they store pre-RoPE K and apply RoPE inside the attention kernel at read
time. The inverse-RoPE wrapper simulated a property the real system should
have natively.

This revision actually implements that property:

```python
forward:
    k_pre = k_proj(h)                        # never rotated
    cache.update(k_pre, v)                   # cache stores PRE-RoPE K
    k_pre_all = cache[layer].keys
    cos_all, sin_all = rotary(seq_total)
    k_post_all = apply_rotary(k_pre_all, cos_all, sin_all)
    q_post     = apply_rotary(q_pre,     cos_new, sin_new)
    attn(q_post, k_post_all, v)
```

The codec operates on `cache[layer].keys` directly. It never sees RoPE phase;
there is no forward or inverse RoPE step on the codec side, because there is
no RoPE step to undo.

## Correctness sanity checks

1. Patched model vs stock model (no codec):
   - fp32, 128-token prompt: **max |Δlogits| = 0.0e+00**, top-1 = 100.00%
   - fp32, prefill 100 tokens + decode 32 tokens: **max |Δlogits| = 0.0e+00**,
     top-1 = 100.00%
2. Cache content check: patched cache `K_pre` vs stock cache `K_post` at
   layer 5: max absolute elementwise difference = **9.57** (i.e. they are
   the right tensors — different, both norms identical).

## Pre-RoPE PPL sweep (Qwen2.5-0.5B, 2 passages, compress=full)

| Config                                      | Δppl        | KL     | top-1    |
|---------------------------------------------|-------------|--------|----------|
| kv, b=2, rsvd r=D/2, vr=0.95 (default)      | +313.32 %   | 1.370  | 50.79 %  |
| kv, b=3, exact, vr=0.995                    | +161.33 %   | 0.890  | 57.94 %  |
| kv, b=4, exact, vr=1.000 (max fidelity)     | +139.90 %   | 0.819  | 61.11 %  |
| **k_only**, b=3, exact, vr=0.995            |  +94.23 %   | 0.569  | 65.87 %  |
| **v_only**, b=3, exact, vr=0.995            |  +56.60 %   | 0.372  | 76.98 %  |

## Cross-check: pre-RoPE cache vs inverse-RoPE wrapper

Same codec, same configs, two architectures:

| Config                 | inverse-RoPE wrapper | pre-RoPE cache |
|------------------------|----------------------|----------------|
| b=2 rsvd vr=0.95       | +315 %               | **+313 %**     |
| b=3 exact vr=0.995     | +167 %               | **+161 %**     |
| b=4 exact vr=1.0 (max) | +141 %               | **+140 %**     |
| K-only b=3             |  +92 %               |  **+94 %**     |
| V-only b=3             |  +63 %               |  **+57 %**     |

Numerically equivalent, as required by linearity. Architecturally correct
only in the right-hand column.

## What this tells us

1. **RoPE is no longer a variable in the codec quality equation.** At max
   fidelity (b=4, exact, vr=1.0), the codec causes +140 % PPL inflation with
   zero RoPE on the data path. Whatever is left is pure codec loss.
2. **The "compound across 24 layers" diagnosis is confirmed** but its driver
   is different from the earlier KIVI-style hypothesis. Per-layer K and V
   reconstruction errors — independent of RoPE — stack multiplicatively on
   log-PPL.
3. **V alone inflates PPL by +57 %.** This alone is disqualifying: V has no
   RoPE, no positional encoding, is a pure linear projection of hidden
   states, and is already being given the codec's most generous regime
   (MSE metric, shared basis across blocks, b=3, exact PCA, vr=0.995).
   So the bottleneck isn't K-specific, it's the skeleton quantizer.
4. **Saturation from b=3 → b=4 is only ~21 pp PPL and ~3 pp top-1.**
   Doubling the residual bit budget barely moves the needle. This rules out
   the Lloyd-Max path as the dominant error source; the dominant source is
   on the skeleton side: `f16` storage of the PCA mean, basis, and K-means
   centres, plus block-boundary basis-switching.

## Implications for v1.3.1

All within-architecture, no skeleton replacement:

1. Promote skeleton storage to `bf16` or `fp32`. Cost: 2× on the skeleton
   bytes only (a few tens of KB per layer at typical D=64,
   block_size=512), i.e. negligible against the hot byte budget.
2. Share PCA basis **within a layer** across all blocks (we already rejected
   cross-layer sharing). This removes block-boundary basis discontinuity.
3. Tighter `variance_ratio` (0.9995+) combined with larger `block_size`
   (1024 or 2048), so there is effectively one basis per layer.
4. Keep the first 128 tokens (attention sink) and the last ~64 tokens
   (active tail) in full precision; compress the middle. This is a feature
   paged-attention backends support natively.

Target: **Δppl ≤ 3 %, top-1 ≥ 85 %** = ACCEPT. If none of the above closes
the gap, the paper must be honestly rewritten as "works end-to-end only for
V-only compression + bf16-skeleton K" — which is still meaningful but
needs to be stated.

## Artefacts

- `benchmarks/pre_rope_cache.py` — `install(model)` monkey-patch for
  Qwen2/Qwen3 family that stores pre-RoPE K in `DynamicCache`.
- `benchmarks/e2e_ppl_pre_rope.py` — PPL driver using the patched model.
- Per-config JSONs under this folder.
