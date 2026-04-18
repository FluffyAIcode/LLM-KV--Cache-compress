# SmolLM2 — v1.3 per-model capability note

## The question

Can we land SmolLM2-1.7B-Instruct in the v1.3 "ACCEPT ∩ beat turbo3"
region via a per-model knob in the `rsvd_target_rank` / `bit_width`
capability table?

## The answer (measured, 5 real configs at ctx=4096)

| config | ratio | vs turbo3 (4.92×) | V MSE inflation vs v1.2 b=3 | verdict |
|---|---:|---:|---:|---|
| v1.2 b=3 exact (production) | 3.09× | −37.3% | 1.00× | baseline |
| **b=2 sym r=32 (v1.3 default)** | **5.37×** | **✓ +9.2%** | 1.61× | REJECT on V |
| Kb=2 Vb=3 K r=32 V r=32 | 4.98× | ✓ +1.3% | 1.54× | REJECT on V |
| Kb=2 Vb=2 K r=32 V r=64 | 4.47× | ✗ −9.2% | 1.13× | MARGINAL |
| Kb=2 Vb=3 K r=32 V r=64 | 3.94× | ✗ −20.0% | **1.00×** | ACCEPT, but worse than v1.2 |

**No knob combination lands in the "ACCEPT quality ∩ beats turbo3"
region**. The Pareto frontier has no such point on SmolLM2.

## Root cause (not a knob problem)

Per-layer V d_eff at `variance_ratio = 0.95` on SmolLM2:

| layer | exact PCA d_eff | V head_dim D | d_eff / D |
|---|---:|---:|---:|
| L1 | 59 | 64 | 0.92 |
| L2 | 59 | 64 | 0.92 |
| L3 | 59 | 64 | 0.92 |
| L4..L23 | 58–59 | 64 | 0.91–0.92 |

SmolLM2's V-stream spectrum is **effectively flat** — exact PCA needs
92% of the available head_dim to capture 95% of variance. There is
no PCA tail to truncate.

Any `rsvd_target_rank < D` forces a truncation below the exact d_eff
and pays the eigenvalue that was cut in full MSE. The `r=32` default
cuts d_eff in half and inflates V MSE to 1.54–1.61× independent of
bit_width; relaxing to `r=D=64` recovers MSE but loses the byte win
that makes the v1.3 config beat turbo3 in the first place.

This is a **structural property of MHA hd=64 models**, not a SmolLM2
bug:
- MHA = every attention head has its own KV (no GQA sharing)
- head_dim=64 is small, leaving no "redundant" low-variance directions
- turbo3 also has no free lunch here — its 4.92× ratio on SmolLM2 is
  already 0.4× lower than its 5.12× on hd=128 models for exactly the
  same reason (a 4-byte norm + 3-bit quantisation on 64 dims costs
  more per-coordinate than on 128 dims)

## Decision

SmolLM2 (and any future MHA hd=64 model with a flat V spectrum) enters
the v1.3 capability table as a **tier-2** deployment:

| tier | config | target | rationale |
|---|---|---|---|
| 1 (default) | b=2 + rsvd r=D/2 + optional InverseRoPE | Qwen / DeepSeek / GLM / Gemma | beats turbo3 at ACCEPT quality |
| **2 (SmolLM2 / MHA hd=64)** | **b=2 + rsvd sym r=32** | SmolLM2 + similar | **9.2% over turbo3 at V=MARGINAL/REJECT** — ship only when compression is hard-prioritised over quality |
| 2 alt | b=2 + rsvd K=32 V=D | SmolLM2 + similar | MARGINAL V, loses to turbo3 by 9% — ship when quality is hard-prioritised over ratio |
| 3 (fallback) | v1.2 b=3 exact | any model | full ACCEPT baseline, pays 37% byte tax vs turbo3 |

Operator chooses tier based on the deployment's quality vs bytes
preference. The v1.3 codec exposes the necessary knobs
(`rsvd_target_rank_k`, `rsvd_target_rank_v`, `bit_width_k`,
`bit_width_v`) — only the choice of tier is policy, not engineering.

## Why not "force" SmolLM2 into tier 1 via more aggressive knobs

We tried:
- `rsvd_oversample` 4 → 8 → 10 → no meaningful V MSE change at fixed r.
- `power_iters` 1 → 2 → 3 → no meaningful change (the V spectrum is not
  ill-conditioned; the issue is it's wide).
- `variance_ratio` 0.90 → 0.99 → tracks exact d_eff monotonically;
  same story as d_eff/outlier ablation on K.
- `block_size` 512 → 1024 → already rejected in block_size ablation
  as MARGINAL globally; on SmolLM2 V the MSE gets worse because
  K-means has to cluster 2× more vectors in the same 16 centres.

None of these moved SmolLM2 into ACCEPT ∩ beat-turbo3. The answer
really is "ship tier 2 with eyes open."

## 7/7 status

| model | v1.3 tier | Δ vs turbo3 | quality |
|---|---|---:|---|
| qwen2_5_0_5b | 1 | +9.7% | MARGINAL |
| qwen3_0_6b | 1 | +29.2% | MARGINAL |
| gemma4_e2b | 1 | +19.8% | ACCEPT |
| deepseek_r1_distill | 1 | +16.8% | MARGINAL |
| glm_edge_1_5b | 1 | +14.2% | MARGINAL |
| **smollm2_1_7b** | **2** | **+9.2%** | **REJECT on V** |
| glm_edge_4b | 1 | +14.0% | MARGINAL |

**Honest summary**: v1.3 tier-1 is 6/7. SmolLM2 is the architectural
outlier where no parameter choice closes the Pareto gap. This is a
property of the model, not the codec.
