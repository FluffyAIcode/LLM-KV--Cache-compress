# Scenario A — snapshot-mode KV compression on vLLM

**Question.** PR #16 decomposed the 27 pp HF↔vLLM Δppl gap and
attributed up to +39 pp to "cross-layer non-linear compounding"
caused by the production harness applying the codec inside the
forward at every layer. If that attribution is right, switching
vLLM to HF's two-pass snapshot semantics (clean prefill → codec
snapshot → teacher-force against codec'd cache) should land Δppl
close to HF's **+7.82 %** MARGINAL number.

**Setup.** Same codec recipe and same 4 WikiText-103 test passages
as every prior run (DS-Distill-Qwen-1.5B, ctx=2048, n_eval=64,
K b=3 + V b=2 + Q-precond + calibrated Lloyd-Max + outlier T=2.0
+ 6-layer boundary skip). vLLM 0.7.3, FLASH_ATTN, bf16, H200.

## Implementation

New harness `benchmarks/e2e_ppl_validation_vllm_snapshot.py` runs
each passage through vLLM twice:

1. **capture pass** — `Qwen2Attention.forward` hook records the
   per-layer pre-RoPE K, V for all 2112 prompt tokens while the
   codec is OFF. The forward is otherwise untouched so the ref
   PPL is a true clean pass.
2. **offline codec** — runs the production v1.3 codec on every
   captured (layer, stream) snapshot (Q-precond → Rust codec →
   un-whiten; boundary layers skipped; K outlier T=2.0 on).
3. **replace pass** — second forward through the same engine. The
   hook **substitutes** each layer's projected `k, v` with the
   pre-codec'd tensor from step 2 instead of letting the layer
   project from its current (potentially shifted) residual. Q
   still comes from the running residual, matching HF's
   teacher-force flow.

The net effect mirrors HF's DynamicCache pattern: every layer's
cache K/V depends only on the codec'd clean snapshot, not on what
earlier layers did to the residual during this forward.

## Result

| Passage | ppl_ref | ppl_alt | Δppl       | top-1   |
|:-:|--:|--:|--:|--:|
| 1 | 124.88 | 139.26 | **+11.52 %** | 75.00 % |
| 2 |  33.00 |  37.30 | **+13.01 %** | 71.88 % |
| 3 |   8.54 |  11.57 | **+35.59 %** | 73.44 % |
| 4 |  25.36 |  39.60 | **+56.17 %** | 76.56 % |

**Aggregate**: `Δppl +29.07 %`, `top-1 74.22 %`, **REJECT**.

Codec time per passage: ~18 s (offline CPU). Alt forward: ~0.17 s
(GPU-only, no in-forward subprocess).

## Cross-mode comparison

| Mode | Harness | Δppl | top-1 | Verdict |
|:---|:---|--:|--:|:-:|
| HF 2-pass DynamicCache (SPRINT_CLOSEOUT) | HF eager | **+7.82 %** | 78.97 % | **MARGINAL** |
| **vLLM snapshot-mode (this run)**         | vLLM FA | **+29.07 %** | **74.22 %** | REJECT |
| vLLM in-forward (PR #15 production)       | vLLM FA | +35.33 % | 59.38 % | REJECT |

## Finding — the +39 pp compounding estimate was wrong

PR #16 Phase 6 predicted snapshot-mode vLLM would land near HF's
+8 % because the sum of 22 non-boundary single-layer Δppl
contributions was −3.9 % and the "extra +39 pp" was attributed to
in-forward cross-layer compounding.

**The actual snapshot-mode run removes only ~6 pp** (+35.33 →
+29.07). The top-1 agreement does jump substantially (59.38 % →
74.22 %, nearly reaching HF's 78.97 %), confirming that the
in-forward harness WAS polluting the one-best prediction — the
codec-shifted residual was changing which token the model argmaxed
at each position. But the **Δppl** stays much higher than HF's
+7.82 %, so the in-forward pollution accounts for only a small
fraction of the HF↔vLLM gap.

### Re-decomposition of the 27 pp HF↔vLLM Δppl gap

| Bucket (revised) | Δppl attribution |
|:---|:---:|
| in-forward vs snapshot (harness integration) | **~6 pp** (of the 27 pp) |
| engine baseline shift (Phase 1, clean model) | ~10 pp |
| residual "intrinsic engine" (FA bf16 attention + softmax + residual stream in bf16) | **~11 pp** |

**The dominant term is actually the engine itself**, not the harness.
Phase 6's Phase-3-based estimate of a +39 pp compounding term was
a miscalculation: the per-layer singletons summed to −3.9 %, but
the joint forward **also had contributions from the snapshot-mode
error compounding** that we attributed to "harness-only". Running
snapshot-mode lets us separate them cleanly.

## Deployment implications

**Scenario A is better, but still rejects.** Using the codec as a
post-prefill cache compressor on vLLM gives +29 pp Δppl / 74 %
top-1 — better than the in-forward harness (+35 / 59 %) but far
from HF's MARGINAL +8 %. Scenario A is still the correct semantics
to deploy (it corresponds to the realistic "compress already-filled
paged cache" use case), but on this model / codec config it does
not reach quality parity with HF.

### Where the remaining ~21 pp actually lives

With the harness-integration term now bounded at ~6 pp, the
remaining ~21 pp vs HF is split between:

- **clean-model baseline mismatch** (~10 pp) — codec-OFF HF and
  vLLM disagree on logits by KL 0.145. Nothing the codec can do
  changes this.
- **intrinsic engine compounding** (~11 pp) — how FA's bf16
  attention kernel propagates identical codec residuals through
  28 layers differs from HF's eager path with fp32-accumulate
  softmax. This is fundamental to the engine and cannot be fixed
  from the harness side.

Top-1 at 74.22 % (within 4.75 pp of HF's 78.97 %) is the first
positive datapoint on vLLM at this codec config. It suggests the
codec's argmax-preserving property IS recoverable when the codec
runs on a clean snapshot; the Δppl gap that remains is in the
logit **distribution**, not in the top-1 choice.

## Artifacts

- `ds_distill_qwen_1_5b_snapshot_vllm_snapshot.json` — per-passage
  metrics, including codec-offline timing per passage.
