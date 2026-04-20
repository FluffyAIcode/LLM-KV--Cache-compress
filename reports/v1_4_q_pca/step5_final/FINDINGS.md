# Step 5 — Scenario 3 full PPL validation

**Date.** 2026-04-17
**Branch.** `cursor/v1-3-rsvd-rope-aware-12f5`
**Model.** DeepSeek-R1-Distill-Qwen-1.5B (D=128, flagship proxy)
**Setup.** pre-RoPE cache, ctx=2048, n_eval=64, **4 WikiText-103 passages**
(previously 2 passages for exploration; 4 for validation).
Streaming-safe: exact PCA per-block K, V share per-layer, Q-precond
(skip=[0,1,26,27]), conservative boundary protection (b=4 on layers
[0,1,26,27]) with calibrated Lloyd-Max K-stream codebook.

## The 5-step plan outcome

| Step | Status | Outcome |
|------|--------|---------|
| 1. V cross-layer rank cap diagnostic | ✅ done | Block 3b not recoverable; V must stay full-rank |
| 2. Path A (cross-layer V skeleton share) | ⏸ deferred | Projected +5% ratio, PPL-neutral; implementation left as polish |
| 3. Calibrated Lloyd-Max codebook | ✅ done | K b=2 MSE 1.47× better than Gaussian; Rust + Python tooling complete |
| 4. Systematic boundary protection | ✅ done | Two modes (bf16 / conservative); conservative b=4 is the practical winner |
| 5. Scenario 3 + b_K=2 PPL validation | ✅ done | **REJECT on 4-passage measurement** |

## Headline PPL results (4 passages, WikiText-103)

| b_K | b_V | boundary | cal. codebook | **Δppl** | top-1   | verdict  |
|----:|----:|:--------:|:-------------:|---------:|--------:|:--------:|
| 4   | 2   |    ✗     |      ✗        | **−3.56 %** | **87.30 %** | **ACCEPT** (Sprint 3.5 baseline) |
| 3   | 3   |    ✓     |      ✓        |  +9.09 % | 79.37 % | REJECT   |
| 3   | 2   |    ✓     |      ✓        |  +9.77 % | 82.54 % | REJECT   |
| 2   | 3   |    ✓     |      ✓        | +11.92 % | 75.79 % | REJECT   |
| 2   | 2   |    ✓     |      ✓        | +18.52 % | 78.17 % | REJECT   |

## The 2-passage → 4-passage revision

On 2 passages, b_K=2 + full guardrails showed Δppl = +2.17 % (ACCEPT).
At 4 passages, the same config shows +18.52 % (REJECT). The 2-passage
result was a small-sample artefact: passage 2 happened to cooperate
with the codec's error structure; passage 1 + 2 more new passages
reveal the true distribution is far more punishing.

**Sprint 3.5 (K b=4 + V b=2 share, no calibration, no boundary
protection) held up: −3.56 % @ top-1 87.30 %** on the same 4-passage
measurement. This was NOT optimised with Steps 3 or 4 — the calibration
and boundary protection we added this sprint are applied *on top* of
the b_K=2 recipe, which still REJECTs.

## Why K b=2 fails end-to-end on this architecture

The theoretical argument was:
1. Q-precondition whitens K → residual is attention-orthogonal
2. At `d_eff = D`, skeleton preserves full K space
3. Therefore only residual quantisation error remains, and K b=2 should
   behave like TurboQuant+ turbo2 (+6.48 % on Qwen2.5-1.5B, MARGINAL)

The theory is structurally correct, but empirically the **gap between
MARGINAL and ACCEPT** ended up larger than estimated:
- TurboQuant+'s +6.48 % @ b=2 is already MARGINAL not ACCEPT
- KakeyaTurbo's b=2 with full guardrails lands at +9-18 % @ 4 passages,
  i.e. somewhere between TurboQuant+'s turbo2 (+6.48 %) and turbo3
  (+1.06 %) — MARGINAL regime, not ACCEPT

The +6.48 % TurboQuant+ number was an MSE-aware-codebook Qwen2.5-1.5B
measurement on wikitext-2 at ctx=512. Our +9-18 % number is
WikiText-103 at ctx=2048 with a different (smaller) model.
Apples-to-apples parity would need running TurboQuant+ production
(turbo2 + norm correction + boundary protection + their Lloyd-Max)
under our harness.

## What actually worked this sprint

Steps 3 and 4 independently are genuine wins — they just don't save
b=2:

**Step 3 alone (calibrated Lloyd-Max), at b=2:**
- Same compression (4.19×), Δppl basically unchanged, **top-1 +6.4 pp**
- Not enough to rescue b=2

**Step 4 alone (conservative boundary protection), at b=2:**
- Δppl improvement from +10 % → +3 % on 2-passage test (with cal codebook)
- But 4-passage shows this was optimistic; true Δppl is +11-18 %

**Steps 3+4 combined at b_K=3 (conservative fallback):**
- **Sprint 3.5 baseline (K b=4 + V b=2 share)**: 3.12× @ −3.56 % @ top-1 87.30 %
- This remains the best streaming-safe ACCEPT operating point we've found.

## Honest final Pareto on DS D=128 (streaming-safe, 4-passage PPL)

```
recipe                                   ratio    Δppl      top-1     verdict
Sprint 3.5 (K b=4 + V b=2 share)         3.12×    −3.56%    87.30%    ACCEPT  ★
+ Step 3 (cal K codebook b=4 unchanged)  3.12×   (+top-1 hint, not tested)  
+ Step 4 conservative boundary [0,1,26,27]
      with middle K b=2 V b=2            3.85×   +18.52%   78.17%    REJECT
+ Step 4 conservative boundary
      with middle K b=3 V b=3            ~3.2×   +9.09%    79.37%    MARGINAL
```

## What's left on the table

The theoretical ceiling at K b=2 remains attractive (3.85× vs 3.12×,
+23 % ratio). What we haven't yet tried:

1. **Per-layer calibrated codebooks**: each layer has its own Lloyd-Max
   centroids instead of one pooled codebook. Addresses the fact that
   layer-specific residual distributions differ.
2. **Per-block calibrated codebooks**: even more fine-grained. Byte
   cost (~32 bytes/block) is negligible; the question is whether the
   offline calibration can converge with only 512-1024 residual
   samples per block.
3. **Q-precondition refitting on residuals**: the diagnostic 2-pass
   Q-precondition was fit on K *pre-PCA*. A second Σ_q fit on
   *post-residual-quantization* reconstructed K could catch the
   remaining attention-importance misalignment.
4. **Scenario 3 at longer context (ctx=8192+)**: the skeleton byte
   amortisation favours long context, so V-share and boundary
   protection's relative cost changes.

## Conclusion

**Sprint 3.5 (3.12× ACCEPT) remains the deployable streaming-safe
operating point on DS D=128**.  The 5-step plan's attempt to push K
to b=2 under Q-precond + calibrated codebook + boundary protection did
not clear the ACCEPT bar under rigorous 4-passage WikiText-103
measurement.  The 2-passage intermediate result was a small-sample
artefact.

Steps 3 and 4 produced useful infrastructure (calibrated Lloyd-Max
tooling, boundary protection modes) that can be layered into future
exploration, but do not unlock b_K=2 ACCEPT within the current
Kakeya-skeleton architecture.

## Artefacts

- `benchmarks/lloyd_max_calibration.py` — offline codebook calibrator
- `benchmarks/e2e_ppl_pre_rope.py` — `--k-centroids-file`,
  `--v-centroids-file`, `--boundary-skip-layers`, `--boundary-mode`,
  `--boundary-bit-width`
- `kakeyaturbo/src/quantize.rs` — `quantize_vector_with_centroids` /
  `dequantize_vector_with_centroids` + 4 new tests
- `kakeyaturbo/src/codec.rs` — `CodecParams.custom_centroids` field
- `kakeyaturbo/src/bin/kakeyaturbo-bench.rs` — `--centroids-file` flag
- `reports/v1_4_q_pca/calibrated_codebook/*.f32` — DS calibrated codebooks
- This FINDINGS.md
