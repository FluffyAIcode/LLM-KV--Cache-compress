# Stage 1 Pre-Execution Projection

**Date**: 2026-04-25
**Status**: paper projection, not measured on trained V4 weights
**Purpose**: scope what KakeyaLattice can realistically add on top of DeepSeek-V4-Flash / V4-Pro's already-highly-compressed KV cache, so we know what to expect from a Stage 1 run and whether the hardware cost is justified.

## Ground rules for this projection

- **Every numerical claim is labelled** either `[MEASURED]` (from our H200 Stage 0.5 JSON, the paper, or the vLLM V4 blog), `[DERIVED]` (direct arithmetic on measured anchors), or `[EXTRAPOLATED]` (would need a Stage 1 run to confirm; we flag the assumption explicitly).
- **Every projection has a failure mode**: the condition under which the projected gain would evaporate is named.
- **No claim is stronger than its weakest caveat**.

## Measured anchors

### A1 — Stage 0.5 H200 rel-MSE vs FP8 per-64-block, on V4-arch KV streams `[MEASURED]`

Source: `reports/v1_5_release/dsv4_stage0_5/dsv4_stage0_5_gemma4_e4b.json` (2026-04-24 H200 run, Gemma-4-E4B host hidden states, real hardware `float8_e4m3fn` quantisation).

| V4 KV stream | FP8 bits | FP8 rel-MSE | E8 Q=38 bits | E8 Q=38 rel-MSE | E8/FP8 ratio |
| --- | --- | --- | --- | --- | --- |
| sliding_window_kv | 4224 | 7.27e-04 | 3296 | **6.17e-04** | **0.849×** |
| csa_pool_kv_ratio4 | 4224 | 9.03e-04 | 3296 | **7.84e-04** | **0.868×** |
| hca_pool_kv_ratio128 | 4224 | 1.12e-03 | 3296 | **9.15e-04** | **0.820×** |
| **mean** | **4224** | **9.2e-04** | **3296** | **7.72e-04** | **0.846×** |

Bit savings: **3296 / 4224 = 0.780**, i.e. E8 Q=38 uses **22.0% fewer bits** than FP8 per-64 and delivers **15% lower rel-MSE** (mean across the three streams).

### A2 — Paper: K-MSE agrees with theoretical D4 shaping gain to ~1% `[MEASURED]`

Source: `reports/paper/kakeyalattice.tex` §5.3 Table 4. On Qwen3-4B at $q_\mathrm{range}=152$: predicted ratio $G(D_4)/G(\Z^4)=0.919$; measured 0.911–0.913 in three independent environments. The paper's section 6 reports the E8 variant adds another $+1.3$ to $+2.0\,$dB K-MSE over D4 in-forward at $n=32$ with 95% CI.

### A3 — vLLM V4 blog: absolute KV footprint at 1M context `[MEASURED]`

Source: [vllm.ai/blog/deepseek-v4](https://vllm.ai/blog/deepseek-v4) appendix "Arithmetic behind the 8.7× savings".

| model | bf16 KV @ 1M | production FP4-indexer + FP8-attention @ 1M |
| --- | --- | --- |
| V4-Pro (61 layers) | **9.62 GiB** | ~4.8 GiB |
| V4-Flash (43 layers)\* | **~6.7 GiB** (derived, see A3b) | **~3.4 GiB** |
| V3.2 (61 layers) | 83.9 GiB | ~42 GiB |

\*The blog publishes Pro's number only. V4-Flash is derived below.

### A3b — V4-Flash bf16 KV @ 1M `[DERIVED from A3 arithmetic + V4-Flash config]`

V4-Flash `compress_ratios` (44 entries for 43 layers + 1 MTP, from `config.json`): `[0, 0, 4, 128, 4, 128, …, 4, 128, 4, 0]`. Counting:

- 2 SWA layers (positions 0, 1) + 1 MTP layer (position 43) = 3 SWA-type layers
- Layers 2–42 alternating: positions 2, 4, …, 42 → 21 `c4a` layers; positions 3, 5, …, 41 → 20 `c128a` layers

Per-layer bf16 KV at 1M context (from the vLLM blog arithmetic, adapting the Pro template to Flash dims):

- **c4a layer**: shared-KV 512 dim × 2 B × (1 048 576 / 4) = 256 MiB; + indexer cache 128 dim × 2 B × (1 048 576 / 4) = 64 MiB → **320 MiB**
- **c128a layer**: 512 × 2 × (1 048 576 / 128) = 8 MiB (no indexer)
- **SWA layer**: 512 × 2 × 128 = 128 KiB (negligible)

V4-Flash total: 21 × 320 MiB + 20 × 8 MiB + 3 × 128 KiB ≈ **6.72 GiB bf16 @ 1M**.

Cross-check against Pro: 30 c4a × 320 MiB + 31 c128a × 8 MiB = 9.36 GiB + 0.24 GiB = **9.60 GiB** vs blog's stated 9.62 GiB. **Match**. ✓

### A4 — Paper: encode-latency cost `[MEASURED]`

`reports/paper/kakeyalattice.tex` Table 6 (D=128):

| codec | latency p50 (μs/2048 tokens, 1 layer) |
| --- | --- |
| FP8 per-64 scalar (reference) | 140–210 |
| D4 Q=10/38/152 | 331–354 |
| E8 Q=10/38/152 | 551–596 |

Ratio: E8/FP8 ≈ **3.0×** at D=128. At V4's D=512 the block count falls (16 E8 blocks vs 32 D4 blocks), so per-vector E8 wall-time is only ~1.56–1.80× D4 (paper §6.2), but **still multiplicatively slower than FP8**.

## Projection B1 — KakeyaLattice as "on top of FP8" fidelity improvement `[EXTRAPOLATED from A1]`

**Claim**: at near-lossless Q=38, E8 saves 22% bits while reducing K-MSE by 15% on Gemma-hidden-driven V4-arch streams. On trained V4-Flash weights the *ratio* is expected to hold within ±20% in either direction (i.e. 18–26% bit savings with comparable or slightly different MSE).

**Why the ratio should roughly carry over**:
1. Stage 0.5 showed all three audit gates fire on V4-arch KV (kurt 0.95–1.10, iso 16–2515, W2/σ 0.24–0.47) at significantly stronger magnitudes than Qwen3-4B's (0.84 / 4.71 / 0.65). **More non-Gaussian input → more headroom for Hadamard + per-vec qmax preprocessing**.
2. The D4 shaping-gain ratio is a *lattice* property, so it transfers independent of the input distribution (A2).
3. E8's super-linear amplification under cross-layer accumulation (paper §6.1) is most active at aggressive Q; at Q=38 we're in the "linear propagation" regime where it's a pure per-layer gain.

**Failure modes (any of these would invalidate the projection)**:
1. **Trained V4 KV is less non-Gaussian than random-init V4 arch**: the V4 Compressor's learned `wkv` + `wgate` weights could regularize the KV to near-Gaussian, flattening the four audit gates and removing the 5-lever headroom. Stage 0.5 uses random-Gaussian init by design; trained might look different. **This is the single biggest risk**.
2. **V4's downstream fused RMSNorm + RoPE + FP8 quant pipeline is non-idempotent on our reconstructed latent**: our hook splices the codec-roundtripped latent BEFORE that pipeline, so any numerical drift in the downstream kernel would cost us some of the advantage.
3. **Hopper-only FlashMLA fallback degrades FP8 baseline more than E8**: if we end up running on H200 not Blackwell, V4's FP8 path is degraded (no native FP4 indexer), which could artificially *inflate* our "beats FP8" signal — the comparison becomes less fair. The measured A1 numbers were on H200 with simulated FP8 (native `float8_e4m3fn` cast with round-trip), so this risk is quantifiable and small, but real Stage 1 must be run with the same FP8 baseline configuration as the V4 production stack.

## Projection B2 — Absolute KV memory savings per user at 1M context `[DERIVED from A3 + B1]`

Applying the 22% bit saving from Projection B1 to the FP8-attention portion of V4's KV (NOT the indexer, which is already FP4 and not a KakeyaLattice target):

V4-Flash @ 1M:

```
bf16 KV total:                              6.72 GiB
FP8 attention + FP4 indexer (V4 prod):     ≈ 3.4 GiB
  of which FP8 attention (compressible):   ≈ 2.7 GiB
  of which FP4 indexer (not compressible): ≈ 0.7 GiB

with E8 Q=38 replacing FP8 attention:
  attention portion: 2.7 * 0.78         ≈ 2.1 GiB
  indexer portion unchanged:             ≈ 0.7 GiB
  total:                                 ≈ 2.8 GiB

Savings per user @ 1M: ~0.6 GiB (18% of V4 production baseline)
```

V4-Pro @ 1M:

```
bf16 KV total:                              9.62 GiB
FP8 + FP4 (V4 prod):                       ≈ 4.8 GiB
  FP8 attention (compressible):            ≈ 3.8 GiB
  FP4 indexer:                             ≈ 1.0 GiB

with E8 Q=38:
  attention: 3.8 * 0.78                  ≈ 3.0 GiB
  indexer unchanged:                     ≈ 1.0 GiB
  total:                                 ≈ 4.0 GiB

Savings per user @ 1M: ~0.8 GiB (17% of V4 production baseline)
```

**This is the ceiling**. Real deployment savings can only equal this if we swap V4's internal FP8 cache for our compressed cache (see Projection C).

## Projection B3 — Aggregate savings per-GPU at deployment scale `[DERIVED]`

Reference deployment: 8× H200 SXM 141 GiB = **1128 GiB aggregate HBM** for V4-Pro, minus ~350 GiB for weights → **~780 GiB usable for KV + workspace**. Per-user 1M-context KV at 4.0 GiB (with E8 Q=38) vs 4.8 GiB (V4 prod baseline).

| baseline | users/node @ 1M | additional users from 18% saving |
| --- | --- | --- |
| V4 prod (4.8 GiB/user) | 780 / 4.8 ≈ **162** | — |
| V4 + E8 Q=38 (4.0 GiB/user) | 780 / 4.0 ≈ **195** | **+33 users (+20%)** |

For V4-Flash on 4×H200 (564 GiB, ~430 GiB usable after weights):

| baseline | users/node @ 1M | additional users |
| --- | --- | --- |
| V4 prod (3.4 GiB/user) | 430 / 3.4 ≈ **126** | — |
| V4 + E8 Q=38 (2.8 GiB/user) | 430 / 2.8 ≈ **153** | **+27 users (+21%)** |

**Caveat**: numerator is "usable HBM after weights & workspace", which is a rough estimate. Real vLLM reserves additional buffers for CUDA graphs, activation workspace, prefix cache, etc. A 20% multiplier on user count translates to roughly **18% more throughput at fixed latency**, assuming attention-bound workload.

## Projection C — The deployment gap `[EXTRAPOLATED, biggest open question]`

**The arithmetic above assumes KakeyaLattice *replaces* V4's FP8 KV storage.** Our current scaffold only *intercepts* and *reconstructs* — the hook applies codec to the latent, then passes the reconstructed latent BACK to V4's FP8 kernel which stores it as FP8 bytes in cache. **Memory savings are therefore zero at the current scaffold level**; only MSE-accuracy delta is measurable.

To realize the projected **~18% per-user HBM savings**, we'd need:

1. **Custom KV cache manager**: store encoded lattice indices instead of FP8 bytes. Requires integrating with vLLM's block allocator (`vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py`, 563 lines). Non-trivial.
2. **Fused decode kernel**: the V4 FlashMLA + sparse attention kernels need to accept lattice-encoded inputs, dequantise in-register, and run attention. Currently they only accept FP8/BF16. Requires extending `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` (477 lines).
3. **New `--kv-cache-dtype`**: register `kakeyalattice_e8` alongside `fp8` in vLLM's kv-cache dtype enum. Follow-up upstream PR per paper §8.5.

**Stage 1 as currently scaffolded only validates the accuracy delta** (K-MSE and, with trained weights, Δppl). The accuracy result is what we need to justify the custom cache manager work. The memory savings come in Stage 2.

## Projection D — Throughput cost `[DERIVED from A4]`

Per-layer codec wall time on V4's D=512 latent at Q=38:

- Stage 0.5 H200 measured: **E8 Q=38 on [1, 2048, 512] ≈ 0.57 ms** (`dsv4_stage0_5_gemma4_e4b.json` → `v15_e8_Q38` → `wall_time_sec: 5.7e-4`)
- Over 43 V4-Flash layers: **~24.5 ms extra per decode forward pass**
- V4 production decode step: 10–30 ms (vLLM blog)
- **Projected slowdown: 1.8× to 3.4× on a single decode step** if codec is run naively in Python

**Failure mode**: this is **unacceptable for production**. The scaffold measures *accuracy* under in-forward hook but cannot be deployed as-is. A fused Triton / CUDA implementation of the Conway–Sloane E8 closest-point algorithm is a prerequisite for any production deployment.

**Where it can still be useful as-is**: offline evaluation and paper validation. Running 32 passages × 10 codec-Q values × 3 models = 960 decode passes. At ~60 ms extra per pass that's only ~1 minute of overhead per configuration. Trivial.

## Projection E — Iso-PPL operating point on V4 `[EXTRAPOLATED, requires Stage 1 run]`

Paper §6.1 reports on Qwen3-4B (D=128): at matched Q, E8 reduces K-MSE by 0.65× on average and |Δppl| by 28–53% in-forward at $n=32$. At D=512 the E8 block count halves from 32 to 16 — fewer blocks, less aggressive scaling, potentially **smaller** relative K-MSE advantage vs D=128 (depends on how the per-block quantisation noise composes across dimensions).

**Best guess from current data**:
- E8 Q=38 on V4: |Δppl| reduction in the **15–30% range** relative to V4's FP8 baseline
- E8 Q=10 on V4: |Δppl| reduction in the **20–40% range** but with absolute |Δppl| numbers possibly already past the "usable" threshold for critical workloads

**Stage 1 must measure this; we do not project a specific Δppl number.**

## Projection F — What Stage 1 would actually deliver `[EXTRAPOLATED timeline]`

Assuming user provisions 2× H200 SXM 141 GiB on vast.ai (~$8/hr) for a 6-hour session:

| hour | deliverable |
| --- | --- |
| 0–1 | Install vllm-openai:deepseekv4-cu130 docker image; download V4-Flash weights (~150 GB); verify fwd pass works |
| 1–2 | Install kakeyalattice + install_all_snapshot_patches_dsv4_aware; run `test_dsv4_snapshot_hook.py`'s `test_patched_attribute_marker` on live wheel |
| 2–4 | First rigorous_eval run: Qwen-WikiText-103, ctx=4096, n=32, q-values={4, 10, 38}, `--kv-stream-filter=all`; ~30 min per configuration × 4 Q-values × 2 protocols (snap + in-forward) |
| 4–5 | Per-stream breakdown: re-aggregate with `--kv-stream-filter={swa, c4a, c128a}` (requires the ≤ 20-line patch to `rigorous_eval.py` flagged in `README.md` Open Items) |
| 5–6 | Write FINDINGS.md; commit + push + PR |

**Projected output**: a `reports/v1_5_release/dsv4_stage1/FINDINGS.md` document mirroring `dsv4_stage0_5/FINDINGS.md`, with trained-weight K-MSE tables + in-forward Δppl tables + per-stream breakdown. Cost estimate: **~$50 of compute** for a clean result.

If the Stage 1 K-MSE numbers fall within the projected 18–26% bit-saving range, **that's the paper addendum**. If they fall outside, **that's a finding** — either trained V4 KV is much more Gaussian than we expect (failure mode B1.1) or V4's FP8 pipeline has different noise characteristics than our per-64 simulation.

## Bottom line

**What we expect from Stage 1**:

1. **Accuracy side** (measurable with scaffold as-is): E8 Q=38 reduces V4 K-MSE by ~15% and uses ~22% fewer bits than V4's internal FP8 per-64-block. **High confidence**, anchored on Stage 0.5 measured numbers; risk concentrated in whether trained V4 KV is as non-Gaussian as random-weight V4 arch.

2. **Δppl side** (requires Stage 1 to measure): |Δppl| reduction of 15–30% at near-lossless Q=38 vs V4's FP8 baseline on three deployable models. **Medium confidence**; could go to 28–53% if super-linear amplification transfers cleanly (paper §6.1 pattern), could go to zero if V4's FP8 noise is already below threshold.

3. **Deployment side** (requires Stage 2 cache-manager + fused kernel): ~18% per-user HBM savings at 1M context, translating to ~20% more concurrent users per GPU at attention-bound load. **Gated on a substantial custom-cache-manager PR**, not on Stage 1.

**What Stage 1 is worth**: the accuracy delta is the claim we'd add to the paper as an "extending KakeyaLattice to next-gen MoE + hybrid-attention architectures" section. The $50 spend is justified if and only if we expect to publish this. If the plan is to stop at the current paper (which already has strong headline results on Qwen3 / DeepSeek-1.5B / Gemma-4 / GLM), Stage 1 is deferrable until a reviewer asks.

**What Stage 2 (deployment savings) is worth**: a concrete ~20% HBM saving on V4 is genuinely useful to hyperscalers. But it requires ~3 weeks of custom-cache-manager + fused-kernel work. Not recommended unless a hyperscaler commits funding.

## Appendix — re-deriving the per-stream bit budgets (all arithmetic checked against Python)

For D=512 head dim (V4's shared-KV latent size):

```
D4 blocks: D/4 = 128
D4 bits/block (exact):   4·log2(2Q+1) − 1 = 32.01 @ Q=152, 16.57 @ Q=10, 24.07 @ Q=38
D4 bits/block (packed):  ceil(exact)     =   33   @ Q=152,    17  @ Q=10,    25  @ Q=38
D4 bits/vec  (packed):   128·ceil + 32 overhead = 4256 @ Q=152, 2208 @ Q=10, 3232 @ Q=38

E8 blocks: D/8 = 64
E8 bits/block (exact):   8·log2(2Q+1)     = 66.02 @ Q=152, 35.14 @ Q=10, 50.13 @ Q=38
E8 bits/block (packed):  ceil             =   67   @ Q=152,    36  @ Q=10,    51  @ Q=38
E8 bits/vec  (packed):   64·ceil + 32 overhead  = 4320 @ Q=152, 2336 @ Q=10, 3296 @ Q=38

FP8 per-64-block (V4 reference at D=512):
  8 bits/coord × 512 = 4096
  per-64 scale: ceil(512/64) = 8 scales × 16-bit (fp16) = 128
  total: 4224 bits/vec

Ratios at Q=38:
  E8/FP8 = 3296/4224 = 0.780 (22.0% bit savings)
  E8/D4  = 3296/3232 = 1.020 (E8 pays +2% over D4 at matched Q)

Ratios at Q=10:
  E8/FP8 = 2336/4224 = 0.553 (44.7% bit savings, but ~10× worse MSE)
  D4/FP8 = 2208/4224 = 0.523 (47.7% bit savings, but ~20× worse MSE)

Ratios at Q=152 (near-lossless):
  E8/FP8 = 4320/4224 = 1.023 (E8 costs 2.3% MORE bits than FP8 at near-lossless)
  D4/FP8 = 4256/4224 = 1.008 (D4 costs 0.8% more bits than FP8 at near-lossless)

At the near-lossless end of the curve KakeyaLattice does NOT beat V4's
FP8 on bits; the deployment operating point where we have headroom is
Q=38 (22% saved with 15% lower MSE).  Q=10 is the aggressive-
compression tail (45% saved but ~10× worse MSE); its value is in
specialty workloads where KV memory is the binding constraint and
modest quality loss is tolerable.
```

These match the `bits_per_vector` fields in the Stage 0.5 JSON exactly.
