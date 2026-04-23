# Mechanism note — why HF and vLLM harness behave differently

**Status.** This is a *hypothesis + partial-evidence record*, not a
fully verified conclusion. It was drafted in conversation to
explain the +39 pp "cross-layer non-linear compounding" estimate
that PR #16 Phase 6 derived from single-layer attribution data,
**before** PR #17 actually measured the snapshot-mode number and
found it removes only ~6 pp instead of the ~39 pp the hypothesis
predicted. The mechanism logic is useful on its own (it is the
reason we built PR #17 in the first place) but it should NOT be
cited as proven — PR #17 evidence contradicts its quantitative
prediction.

The purpose of this file is audit trail: keep the reasoning so
future agents can recheck it against new evidence (e.g. the
Option C backend's fully in-kernel path), rather than relying on
reconstructed chat memory.

## Summary of the claim (draft)

- HF 2-pass harness: clean prefill → codec applied to DynamicCache
  snapshot once → teacher-force eval. Each layer's per-cache error
  `ε_l` is **independent across layers** (all `K_l^clean` came from
  the same clean prefill pass). Joint Δppl ≈ Σ_l Var(ε_l) — linear
  composition.
- vLLM in-forward harness: at every layer, K/V are projected from
  a residual that already carries the codec errors of all earlier
  layers, then codec'd again. Same `ε_l` appears **twice**: once
  as the direct perturbation of layer l's attention output, and
  once as the indirect pre-codec input shift for layer l+1 via
  `W_k · δ_l`. Because both copies of `ε_l` have the same sign and
  direction, they **correlate across layers**, and the variance
  grows as `(N σ)²` rather than `N σ²` — a factor of `√N` above
  independent composition.
- For the 22 non-boundary layers, `√22 ≈ 4.7×` extra amplification
  over the linear sum.
- bf16 residual-stream accumulation aggravates the correlation
  because the engine cannot cancel opposite-sign sub-ULP errors —
  correlated errors add in phase, uncorrelated errors that would
  nearly cancel in fp32 end up rounded apart under bf16.
- HF eager's `eager_attention_forward` explicitly upcasts softmax
  to fp32 (`softmax(..., dtype=torch.float32).to(query_states.dtype)`);
  vLLM's FA kernel keeps softmax in bf16. This further widens the
  gap for correlated errors that pass through `exp(score)`.

## Quantitative prediction derived from this claim

- Running vLLM in snapshot mode (= HF-like semantic, independent
  per-layer `ε_l`) should remove the `~√22 ≈ 4.7×` amplification
  factor, cutting roughly 39 pp off the 45 pp total — landing
  around HF's +7.82 % MARGINAL number.

## Evidence against the quantitative prediction

PR #17 `reports/v1_3_ppl/snapshot_mode/` measured the snapshot-mode
cell directly:

| Mode | Δppl | top-1 |
|:---|--:|--:|
| HF 2-pass (reference)                   | +7.82 %  | 78.97 % |
| vLLM snapshot-mode (= HF semantic)      | +29.07 % | 74.22 % |
| vLLM in-forward                         | +35.33 % | 59.38 % |

- Δppl drop from in-forward → snapshot: **only ~6 pp**, not ~39 pp.
- Top-1 agreement DID jump 15 pp (59 → 74), consistent with the
  "in-forward pollution changes argmax choice" part of the
  hypothesis, but Δppl stayed far from HF.

So the hypothesis correctly identified a real mechanism
(in-forward coherent error has a measurable effect — top-1
recovery of ~15 pp and Δppl reduction of ~6 pp) but **over-
estimated its magnitude by ~6×**. The dominant residual bucket
(~11 pp) lives in intrinsic engine numerics (FA bf16 softmax vs
HF eager fp32 softmax, plus ~10 pp of clean-model logit
mismatch from Phase 1+5).

## Evidence FOR pieces of the claim

- Phase 4 (`reports/v1_3_ppl/gap_decomposition/phase4/` on PR #16
  branch): codec residuals themselves are engine-agnostic —
  codec MSE ratio vLLM/HF is 1.01 median. Rules out "codec
  produces different errors" as a cause.
- Phase 2: HF is MORE sensitive to independent `σ·randn` noise
  per unit σ. This is consistent with the bf16-rounding argument
  above — independent noise gets rounded out under bf16, so vLLM
  tolerates random noise better; correlated noise (what codec
  actually produces) does NOT get rounded out because its sign
  is aligned across layers.
- PR #17 top-1 recovery (+15 pp) is the strongest direct evidence
  that in-forward pollution is real; the one-best prediction is
  exactly what residual-stream shift should disturb.

## What the Option C (full in-kernel) backend experiment should
actually settle

The snapshot-mode harness applies the codec to a **fp32 snapshot
on CPU** and substitutes fp32 tensors back (cast to bf16 at the
FA boundary). The Option C backend applies the codec **inside a
Triton kernel, bf16 all the way, stored as compressed bytes in
the paged cache**. Two possibilities:

1. If Option C Δppl ≈ +8 % (HF parity) → the ~11 pp intrinsic-
   engine bucket from PR #17's decomposition was actually the
   CPU-fp32 round-trip noise, not FA bf16 softmax. Snapshot-mode
   underestimated vLLM's true quality ceiling.
2. If Option C Δppl ≈ +20-30 % → FA bf16 softmax is the real
   root cause, and moving the codec into the engine doesn't help.
   The path forward would be to modify the FA kernel itself (or
   accept the Δppl penalty).

Option C is a single-point discriminator between the remaining
hypotheses. It's the experiment the team is now executing on the
new CUDA-13 H200.

## Why this file lives here and not inside a specific phase dir

- `snapshot_mode/FINDINGS.md` reports the actual PR #17 measured
  result and correctly notes the +39 pp prediction was wrong.
- `gap_decomposition/FINDINGS.md` (on PR #16 branch) reports the
  bucket decomposition from Phases 1–6, with Phase 3's per-layer
  attribution as input.
- The **mechanism-level narrative** (why in-forward could amplify
  at all, sign-correlated composition, bf16 ULP argument) wasn't
  written down anywhere — it was only in chat. This file fixes
  that.
- Keep it at the `reports/v1_3_ppl/` root so it's discoverable
  from any phase/experiment subdirectory.

## Corrections needed if Option C contradicts this file

If Option C ends up with Δppl close to HF (+8 %), update:

- PR #17 `snapshot_mode/FINDINGS.md`: revise "intrinsic engine
  ~11 pp" bucket — it was actually the CPU round-trip, not the
  engine.
- This file: mark section "Quantitative prediction derived from
  this claim" as **partially correct** — the snapshot-mode number
  was just noisy because of the harness, and the true in-forward
  vs snapshot gap on vLLM is much smaller than snapshot-mode
  suggested.
