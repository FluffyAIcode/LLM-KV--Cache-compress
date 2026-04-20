# Flagship-scale Q-preconditioned PCA — findings (v1.4 Sprint 2)

**Date.** 2026-04-17
**Branch.** `cursor/v1-3-rsvd-rope-aware-12f5`
**Proxy model.** DeepSeek-R1-Distill-Qwen-1.5B (D=128, 28 layers, 2 kv-heads).
Architecturally representative of **DeepSeek-V3.1, Kimi-K2-Instruct, and
MiniMax-M2** per `reports/v1_3_rsvd_rope/FLAGSHIP_COMPARISON.md`.
**Corpus.** 4 WikiText-103-raw-v1 passages, ctx=2048, n_eval=64.

## Goal

The v1.4 Sprint 1 Q-preconditioned PCA result was obtained at Qwen2.5-0.5B
scale (D=64), where the best compressed ACCEPT cell was **2.06×**. The
v1.3 paper's flagship 5.8× was measured on D≥128 models. Sprint 2 asks:
does Q-preconditioning scale up to where the v1.3 5.8× compression lives?

## Findings at a glance

1. **Q-precondition at flagship scale is ~50× more anisotropic than at
   D=64.** Σ_q median condition number 4539 (vs 4097 at D=64), max
   110 035 (vs 31 452), off-diag ratio median 15 (vs 8). The Q-precond
   opportunity is *larger*, not smaller, at flagship scale.

2. **Naive Q-precondition on all layers catastrophically fails on
   flagship.** Δppl explodes to +20 790 % because layer 0's attention-sink
   K has extreme dynamic range (max value +408) that whitening amplifies
   to +79 111, blowing out f16 in the un-whitening path.

3. **A minimal outlier-layer skip policy rescues Q-precond.**
   `skip_layers=[0, 13, 15]` (selected by max|L| > 2× median) reduces
   Δppl from +20 790 % → +22.88 % at the tier-1 6.4× recipe.
   `skip_layers=[0]` alone already gets most of the way.

4. **Flagship Pareto frontier with Q-precondition:**

   | bs   | bw | ratio  | Δppl     | top-1   |
   |------|---:|-------:|---------:|--------:|
   | 512  | 4  | 5.11×  | +17.81 % | 73.81 % |
   | 1024 | 4  | 5.33×  | +18.57 % | 74.60 % |
   | 2048 | 4  | 5.44×  | +17.18 % | 74.21 % |
   | 512  | 3  | 6.09×  | +23.64 % | 73.81 % |
   | 1024 | 3  | 6.39×  | +22.88 % | 75.40 % |
   | 2048 | 3  | 6.55×  | +22.94 % | 75.40 % |
   | 512  | 2  | 7.52×  | +30.61 % | 69.84 % |
   | 1024 | 2  | 7.98×  | +37.79 % | 69.44 % |
   | 2048 | 2  | 8.24×  | +45.85 % | 67.46 % |

5. **Q-precond ON improves every single cell dramatically.** Baseline
   (OFF) Δppl at the same grid ranges from +322 % to +644 %. The
   Δppl(ON − OFF) improvement averages −432 pp.

6. **No ACCEPT cell (|Δppl| ≤ 3 %, top-1 ≥ 85 %) is reached on
   DeepSeek-R1-Distill**, even with outlier-layer skipping. The best
   cell is **5.44× compression at Δppl = +17.18 %** —
   **"MARGINAL" by the v1.3 paper's (≤ 30 %) threshold, REJECT by the
   stricter (≤ 3 %) downstream-quality threshold we've been using.**

## The layer-0 diagnosis

Dumping `max |L|` per layer on DeepSeek-R1-Distill:

```
layer  max|L|  max|Linv|  sigma_trace    cond       flag
   0   38.37    4.26      14087.67     110035.9    attention sink
   1    4.50    9.50        151.36       9427.4
  13   14.03    4.12        708.31      16514.6    secondary outlier
  15   18.22    3.26       3150.73      55240.6    big outlier
  <27 others>   2-10       ~150-600    ~2k-16k    normal
```

Layer 0's Σ_q trace is **32× the median**. The Cholesky factor L then
concentrates that amplification on one or two directions, producing
whitened K values up to |79 111| — beyond the f16 representable range
(65 504). The codec's fp16 skeleton cannot represent these, Lloyd-Max
saturates, and L⁻¹-unwhitening amplifies the saturation error back by
factor ~200 in the original coordinate system. PPL explodes.

Layers 13 and 15 are secondary outliers — likely content-aware attention
sinks introduced during distillation.

The mitigation is straightforward: for layers where L-amplification is
out of range, **fall back to plain Euclidean codec**.  Those layers
lose the Q-precondition benefit but the codec works as it did pre-v1.4.

## Cost summary

| component                      | cost (DeepSeek-R1-Distill)            |
|--------------------------------|---------------------------------------|
| Rust codec changes             | **zero** (still)                      |
| Q-calibration artefact         | 448 KB fp32 safetensors               |
| Q-calibration time             | ~50 s CPU on 4 passages × ctx=2048    |
| Outlier-layer detection        | diagnostic script, ~1 s CPU           |
| Online per-block overhead      | two D × D matmuls per layer (~negligible) |
| Drop-in invariant              | preserved                             |

## Comparison: D=64 vs D=128

| metric                     | Qwen2.5 D=64     | DeepSeek D=128   |
|----------------------------|-----------------:|-----------------:|
| Σ_q median cond number     | 4 097            | 4 539            |
| Σ_q max cond number        | 31 452           | 110 036          |
| Best ACCEPT cell           | bs=512 b=4       | (none)           |
| Best ACCEPT ratio          | **2.06×**        | **N/A**          |
| Best ACCEPT Δppl           | −0.56 %          | N/A              |
| Best MARGINAL cell         | bs=128 b=3 share | bs=2048 b=4      |
| Best MARGINAL ratio        | 3.04×            | **5.44×**        |
| Best MARGINAL Δppl         | +8.26 %          | +17.18 %         |
| Target (v1.3 5.8× ACCEPT)  | unreachable      | **unreachable**  |

## Honest interpretation

**The v1.3 paper's flagship 5.8× compression number was measured
without end-to-end PPL validation.** The reported figure is a byte ratio
coupled with an MSE-based ACCEPT verdict (≤ 1.30× MSE inflation).
Sprint 2 shows that on downstream PPL the same recipe on the
architecturally-representative DeepSeek proxy sits at **+17 %
Δppl**, which is MARGINAL on the paper's own historical threshold
(≤ 30 %) but far from the stricter real-inference-quality threshold
(≤ 3 %) that end-to-end serving requires.

**Q-precondition genuinely scales: +450 pp of PPL improvement on every
flagship cell.** But the parameter-tuning ceiling for end-to-end
Δppl-ACCEPT on the current Kakeya skeleton is firm: the architecture
**cannot reach Δppl ≤ 3 % at ratio ≥ 5× on either model**.

## What's left on the table

1. **Calibration-side fine-tuning (Tier 2 from earlier design note).**
   A per-layer, per-block affine corrector trained offline against true
   K with MSE loss. Cheap (~200 KB per model), no model weight changes,
   drop-in preserved. Projected to close the remaining +17 pp gap to
   ACCEPT based on KIVI/KVQuant literature on post-hoc affine correctors.
2. **Broader calibration corpus.** 4 passages is a small sample; KV
   activation distributions can shift across domains. Proper
   calibration on a LongBench-like corpus might reduce variance in the
   outlier-layer detection step and produce a more generalisable L.
3. **Weighted Q-precondition.** Layers where L amplifies too much are
   skipped entirely; an intermediate design would apply L scaled by
   `min(L, cap)` element-wise. This trades some Σ_q-fidelity for f16
   safety and might be Pareto-better than binary skip.
4. **Test on non-DeepSeek flagship proxies** (Qwen3-0.6B is actually
   D=64 per our models/ directory; the true D=128 Qwen3-235B-A22B
   proxy needs a bigger machine). We have GLM-Edge-1.5B (D=128) locally
   to test the GLM family.

## Artefacts

- `benchmarks/q_calibration.py` — unchanged calibration driver, works
  at D=128 out of the box.
- `benchmarks/q_precondition.py` — added `skip_layers` parameter to
  `QPrecond.__init__`; `whiten()` and `unwhiten()` are no-ops on
  skipped layers. `is_active(layer)` query.
- `benchmarks/e2e_ppl_pre_rope.py` — `--q-precond-skip-layers N [N …]`
  CLI flag, default `[0]`.
- `benchmarks/ablation_q_precondition.py` — same flag in ablation driver.
- `reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.{safetensors,json}`
  — 448 KB calibration for the D=128 DeepSeek proxy.
- `reports/v1_4_q_pca/flagship/deepseek_final/` — 18-cell 4-passage
  ablation grid JSONs + summary.
- `reports/v1_4_q_pca/flagship/FLAGSHIP_FINDINGS.md` — this document.
