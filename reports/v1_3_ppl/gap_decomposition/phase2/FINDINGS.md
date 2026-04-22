# Phase 2 — Δppl(σ) noise-sensitivity curves

**Question.** If we replace the codec with a pure Gaussian
perturbation of a controlled magnitude σ (in units of per-tensor RMS
of K/V), how does Δppl depend on σ on each engine? If the curves
coincide, the engines are equally sensitive to the same relative
noise, and the codec-cell Δppl gap must come from the codec's actual
error distribution differing. If the curves diverge, the engines
intrinsically amplify the same noise by different factors.

**Setup.** DS-Distill-Qwen-1.5B, 4 WikiText-103 test passages, ctx
2048, n_eval 64. Pre-RoPE hook on each engine. Seeded RNG counter
so both engines see statistically-equivalent noise at each σ; the
noise is `sigma * rms(K/V) * randn`, applied to both streams at the
same σ.

## Result

| σ       | **vLLM Δppl** | **HF Δppl** | vLLM \|Δlogp\| | HF \|Δlogp\| |
|:-:|:-:|:-:|:-:|:-:|
| 0.001   | **−0.57 %** (~0)   | **+1.74 %**    | 0.0387 | 0.1027 |
| 0.010   | **+12.37 %**       | **+33.50 %**   | 0.5038 | 0.6134 |
| 0.030   | +2645 %            | +2464 %        | 3.56   | 3.47   |
| 0.100   | +18 460 %          | +33 103 %      | 5.33   | 5.88   |
| 0.300   | +20 997 %          | +40 949 %      | 5.48   | 5.82   |

## Finding — in the linear regime HF is MORE sensitive, not less

At σ = 0.001 and σ = 0.010 we are in the small-perturbation regime.
Here:

- At σ=0.001, vLLM's Δppl is within passage noise (−0.57 %),
  while HF already gives +1.74 %. HF is more sensitive.
- At σ=0.010, vLLM gives +12.4 % vs HF's +33.5 %. Ratio ≈ 2.7×;
  HF is **more** sensitive per unit σ in the linear regime.

Above σ=0.03 both engines are in the saturation regime (Δppl > 2000 %,
|Δlogp| > 3 nats) and the curves crisscross irregularly — the signal
is lost to saturation. The only meaningful comparison is at σ ≤ 0.01.

## What this contradicts — and what it explains

Before Phase 2 the "working theory" (PR #15) was that FA's bf16
softmax **amplifies** residuals more than HF eager. Phase 2 says the
opposite: on a controlled noise injection, **HF's eager forward
amplifies relative-RMS noise more, not less**, per unit σ.

The only way this is consistent with the measured codec cell
(**HF +7.82 % vs vLLM +35.33 %**) is if either:

1. **HF and vLLM see very different effective σ under the production
   codec**, so that HF operates at a smaller σ on its own engine
   than vLLM does on its own engine. This is plausible because the
   codec is applied to pre-RoPE K in both harnesses, but HF reads
   the reconstructed K from a DynamicCache tensor that might be
   upcast to fp32 somewhere in the attention path, while vLLM keeps
   bf16 all the way. Phase 4 showed the codec itself produces the
   same fp32 residuals; what differs is what each engine does
   **after** that fp32 → bf16 transition inside attention.
2. **The codec's error is not well-modelled by isotropic Gaussian
   noise.** Lloyd-Max + WHT + outlier-T codec produces a structured
   residual whose projection onto the attention metric is different
   from random noise's projection. If that structure happens to
   align with directions HF's eager attention suppresses (but vLLM's
   FA doesn't), the engines swap their sensitivity order.

Either way, Phase 2 rules out the simple story "FA bf16 softmax
amplifies noise more than HF eager f32". The sensitivity difference
is either opposite in sign at matched σ, or it is irrelevant because
the codec's error is *not* σ×randn.

## Implications for the gap decomposition

| Bucket | Attribution |
|:---|:---|
| Phase 1 — engine baseline shift (clean model)             | −11 % PPL rel, 0.145 KL, 18 % top-1 disagreement even at σ=0 |
| Phase 4 — codec residual magnitude (input-output identity) | ~0 (codec errors are engine-agnostic at ≤1% resolution) |
| Phase 2 — engine noise sensitivity (controlled σ)          | HF **more** sensitive per σ in the linear regime; does not explain vLLM being worse |

The 27 pp gap (HF +7.82 % vs vLLM +35.33 %) therefore most likely
comes from the **codec error structure** interacting with each
engine's attention numerics in directions that pure noise does not
probe. Phase 3 (per-layer codec attribution) is the next sharpest
knife for that: if the "extra" vLLM damage is concentrated in a
small layer set, it suggests a structural-noise explanation; if it
is uniform, it points at a systematic per-layer numeric path
difference instead.

## Data

- `ds_distill_qwen_1_5b_vllm_both.json` — vLLM σ-sweep.
- `ds_distill_qwen_1_5b_hf_both.json`   — HF eager σ-sweep.
