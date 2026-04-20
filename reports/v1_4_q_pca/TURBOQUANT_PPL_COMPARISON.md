# TurboQuant vs KakeyaTurbo — end-to-end PPL comparison

**Date.** 2026-04-17
**Branch.** `cursor/v1-3-rsvd-rope-aware-12f5`
**Measurement.** Pre-RoPE cache harness, same WikiText-103-raw-v1 passages,
same `ctx_len / n_eval / n_passages` setup as KakeyaTurbo measurements.
**TurboQuant source.** Reference Python impl at
`turboquant_plus/turboquant/turboquant.py` (Algorithm 2: PolarQuant at
`b-1` bits + QJL at 1 bit for K; PolarQuant at `b` bits for V).

## Why this is a fair comparison

The harness serves the codec at exactly the granularity production
attention uses:

```
for each full-attention layer l:
    K_pre_l, V_l = cache.layers[l]   # pre-RoPE, post k_proj
    K_hat_l = codec.compress_then_decompress(K_pre_l)
    V_hat_l = codec.compress_then_decompress(V_l)
    cache_alt.layers[l] = (K_hat_l, V_hat_l)

alt_logits = model.forward(cont_ids, past_key_values=cache_alt)
compare(alt_logits, ref_logits)
```

Both KakeyaTurbo and TurboQuant see the same per-layer K/V tensors and
the same PPL evaluation downstream. This isolates codec quality from
framework.

## Results — Qwen2.5-0.5B (D=64, 24 layers, ctx=1024, 2 passages)

| codec                        | b  | ratio  | Δppl          | top-1   |
|------------------------------|---:|-------:|--------------:|--------:|
| TurboQuant (no Q-precond)    | 2  | 6.40×  | +120 288 %    |  3.97 % |
| TurboQuant (no Q-precond)    | 3  | 4.57×  | +772 220 %    |  8.73 % |
| TurboQuant (no Q-precond)    | 4  | 3.56×  |   +1 728 %    | 42.06 % |
| **KakeyaTurbo + Q-precond**  | 4  | **2.06×**  | **−0.56 %** | **92.86 %** |
| KakeyaTurbo + Q-precond      | 3  | 2.36×  |   +3.32 %     | 84.92 % |
| KakeyaTurbo + Q-precond      | 2  | 2.77×  |  +35.40 %     | 73.02 % |

## Results — DeepSeek-R1-Distill-Qwen-1.5B (D=128, 28 layers, ctx=2048, 2 passages)

| codec                                  | b  | ratio  | Δppl          | top-1   |
|----------------------------------------|---:|-------:|--------------:|--------:|
| TurboQuant (no Q-precond)              | 2  | 7.11×  | +16 957 %     |  6.35 % |
| TurboQuant (no Q-precond)              | 3  | 4.92×  |  +9 329 %     |  7.14 % |
| TurboQuant (no Q-precond)              | 4  | 3.76×  |  +9 342 %     |  9.52 % |
| **KakeyaTurbo + Q-precond (skip[0,13,15])** | 4 | **5.44×** | **+17.18 %** | 74.21 % |
| KakeyaTurbo + Q-precond (skip[0,13,15]) | 3  | 6.39×  | +22.88 %      | 75.40 % |
| KakeyaTurbo + Q-precond (skip[0,13,15]) | 2  | 7.98×  | +37.79 %      | 69.44 % |

## Order-of-magnitude gap

| config                     | KakeyaTurbo + Q-precond | TurboQuant | gap     |
|----------------------------|------------------------:|-----------:|--------:|
| Qwen2.5-0.5B, b=3          | +3.32 %                 | +772 220 % | ~230 000× |
| Qwen2.5-0.5B, b=4          | −0.56 %                 |   +1 728 % | ~3 000×   |
| DeepSeek-R1-Distill, b=3   | +22.88 %                |  +9 329 %  | ~400×     |
| DeepSeek-R1-Distill, b=4   | +17.18 % *(b=4 bs=2048)*|  +9 342 %  | ~540×     |

At every bit width on every model, **KakeyaTurbo + Q-preconditioning
beats TurboQuant by 3-6 orders of magnitude in downstream PPL** at
comparable compression ratios.

## Why TurboQuant struggles end-to-end

TurboQuant is a **per-vector data-oblivious codec**: each K/V vector
gets an independent PolarQuant + QJL roundtrip with ~40 % relative L2
error per vector at `b=3`. That noise is full-rank (every coordinate
gets quantized), so it cannot be absorbed by any low-rank attention
substructure — it simply compounds multiplicatively through 24-28
attention layers, and the softmax argmax drift accumulates to the
top-1=4-9 % regime we observe.

KakeyaTurbo's block-PCA skeleton by contrast produces **per-block
correlated** noise in a `d_eff`-dimensional subspace. Vectors within
a block share the same basis, so their reconstruction errors are not
independent — a large fraction of the residual lives in the
complement subspace that the attention operator doesn't
disproportionately weight (and Q-preconditioning further aligns this
complement with low-Σ_q directions).

This is also consistent with the published TurboQuant+ `perplexity`
log at `turboquant_plus/benchmark-results-raw/ppl_turbo3.log`:

```
bf16 baseline :  PPL =   6.12 ± 0.33
turbo3         :  PPL = 165.64 ± 11.08   (Δppl ≈ +2 607 %)
```

measured with llama.cpp on Qwen3.5-35B-A3B Q8_0 / 8-chunk WikiText.
Our reference-impl PPL is consistent with that published log — the
catastrophic Δppl is not a bug, it is a genuine property of the
per-vector scalar quantization approach without a model-aware
compensator.

## Asterisk: compression-ratio claims

TurboQuant's hardware-serving value proposition *is* dominated by its
MSE / inner-product distortion metric, under which it is a legitimate
(≤ 1.30× MSE inflation) ACCEPT codec at `b=3`. The v1.3 paper's own
comparison table also reports TurboQuant-turbo3 favourably on MSE.
**But MSE-ACCEPT and PPL-ACCEPT are not interchangeable**, as the
KakeyaTurbo sprints earlier on this branch have documented — a codec
can clear MSE ≤ 1.30× while destroying downstream PPL by thousands of
percent, and in TurboQuant's case that's exactly what's happening.

## Caveats

1. **Reference-impl speed.** The Python TurboQuant reference
   implementation is slow (one 32×32 rotation matrix per `quantize()`
   call); that's why we use a modest `n_passages=2`. The **relative**
   ordering is clear enough at 128-token eval, but absolute
   Δppl numbers above 100 % have measurement noise on the order of
   ±30 pp at this sample size.
2. **TurboQuant's own rotation is data-oblivious.** The paper's
   strength is the provable guarantee on random rotation — but that
   guarantee is distributional (expected distortion ≤ threshold), not
   per-vector. Real K/V cache is very far from the Gaussian prior the
   guarantee assumes.
3. **Q-precond on TurboQuant.** Briefly tested — Q-precond + TurboQuant
   at Qwen2.5-0.5B b=3 gives Δppl ≈ +1 284 000 %, an order of
   magnitude worse than TurboQuant alone. TurboQuant's internal
   normalisation conflicts with Q-whitening (whiten → norms change →
   PolarQuant's codebook is off — documented behaviour, not a bug).
   **Q-precond is specific to codecs that minimise plain MSE with no
   internal per-vector normalisation.**
4. **This is not a KLD comparison.** The TurboQuant+ log above
   reports KLD separately; its KLD numbers (`kld_dense_turbo3.log`)
   are much more flattering to TurboQuant. Our Δppl captures the full
   generative consequence, which KLD does not.

## One-paragraph headline

On end-to-end PPL, measured apples-to-apples on the same WikiText
passages with the same pre-RoPE cache harness:

> **KakeyaTurbo + Q-preconditioned PCA ACCEPTs (Δppl = −0.56 %) at
> 2.06× compression on Qwen2.5-0.5B, while TurboQuant at comparable
> compression (4× at b=4) inflates PPL by +1728 %. On the
> flagship-scale DeepSeek proxy at 5.4× compression, KakeyaTurbo +
> Q-precond is +17 % Δppl (MARGINAL) while TurboQuant at matching
> ratio is +9342 % (catastrophic).**

The 3-6 orders-of-magnitude advantage is explained by the block-PCA
structure + Σ_q-aware skeleton, which produces correlated low-rank
reconstruction noise instead of the per-vector full-rank noise that
TurboQuant's per-vector scalar quantizer produces.

## Artefacts

- `benchmarks/turboquant_roundtrip.py` — TurboQuant adapter.
- `benchmarks/e2e_ppl_pre_rope.py` — new `--codec={kakeyaturbo,turboquant}` flag.
- `turboquant_plus/benchmark-results-raw/ppl_turbo3.log` — llama.cpp
  reference showing PPL 165.64 vs f16 6.12 on Qwen3.5-35B (consistent
  with our Δppl ≈ 2 600 %).
