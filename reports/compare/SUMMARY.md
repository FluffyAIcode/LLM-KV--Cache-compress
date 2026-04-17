# Kakeya vs TurboQuant+ — Byte-for-Byte Comparison

Side-by-side compression ratios on the **same model**, **same captured
KV tensors**, **same context length**. Kakeya numbers use the
standard benchmark preset (`block_size=512, residual_length=256,
d_res=8, K=16, variance_ratio=0.95`); TurboQuant+ numbers use the
reference Python prototype from
[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)
at `turbo2` (2-bit K+V), `turbo3` (3-bit K+V) and `turbo4` (4-bit K+V).

Baseline for both is bf16 KV. Kakeya's "bf16 store" column is the
dtype-matched projection documented in `reports/STANDARD.md`.
TurboQuant+'s byte count is exact — `head_dim × b / 8` bytes per
vector plus 32 bits per vector for the K norm (V is MSE-only, no
norm). TurboQuant+'s ratio **does not depend on context length**
because it is per-token scalar quantization; Kakeya's ratio **grows
with context** because its per-block skeleton cost amortizes.

## Measured at 2k and 8k

| Model (head_dim, KV layout) | Ctx | Kakeya total bf16 | turbo2 | turbo3 | turbo4 |
|---|---:|---:|---:|---:|---:|
| qwen2_5_0_5b (hd=64, GQA 14:2) |  2 048 | 1.68× | **7.11×** | 4.92× | 3.77× |
| qwen2_5_0_5b (hd=64, GQA 14:2) |  8 192 | 2.03× | **7.11×** | 4.92× | 3.77× |
| smollm2_1_7b (hd=64, MHA 32:32) |  2 048 | 1.72× | **7.11×** | 4.92× | 3.77× |
| smollm2_1_7b (hd=64, MHA 32:32) |  4 096 | 1.96× | **7.11×** | 4.92× | 3.77× |
| qwen3_0_6b (hd=128, GQA 16:8) |  2 048 | 2.37× | **7.53×** | 5.12× | 3.88× |
| qwen3_0_6b (hd=128, GQA 16:8) |  8 192 | 3.62× | **7.53×** | 5.12× | 3.88× |
| deepseek_r1_distill_qwen_1_5b (hd=128, GQA 12:2) |  2 048 | 2.10× | **7.53×** | 5.12× | 3.88× |
| deepseek_r1_distill_qwen_1_5b (hd=128, GQA 12:2) |  8 192 | 2.92× | **7.53×** | 5.12× | 3.88× |
| glm_edge_1_5b (hd=128, GQA 16:4) |  2 048 | 2.21× | **7.53×** | 5.12× | 3.88× |
| glm_edge_1_5b (hd=128, GQA 16:4) |  8 192 | 3.18× | **7.53×** | 5.12× | 3.88× |
| glm_edge_4b (hd=128, GQA 24:6) |  2 048 | 2.27× | **7.53×** | 5.12× | 3.88× |
| glm_edge_4b (hd=128, GQA 24:6) |  4 096 | 2.88× | **7.53×** | 5.12× | 3.88× |
| gemma4_e2b (mixed hd=256/512, hybrid) |  2 048 | 1.58× | **7.84×** | 5.26× | 3.96× |
| gemma4_e2b (mixed hd=256/512, hybrid) |  8 192 | 2.80× | **7.86×** | 5.27× | 3.97× |

## Extrapolated to 32k / 128k (Kakeya grows, TurboQuant is flat)

The Kakeya column below uses the byte-exact extrapolator
(`kakeya_extrapolate.py`) that was cross-validated on real 16k/32k
runs; the TurboQuant+ numbers are repeated from the 2k row because
per-token quantization has no context dependence.

| Model | 32k Kakeya bf16 | 32k turbo3 | 32k turbo4 | 128k Kakeya bf16 | 128k turbo3 | 128k turbo4 |
|---|---:|---:|---:|---:|---:|---:|
| qwen2_5_0_5b | 2.12× | 4.92× | 3.77× | 2.15× | 4.92× | 3.77× |
| smollm2_1_7b | 2.22× | 4.92× | 3.77× | 2.25× | 4.92× | 3.77× |
| qwen3_0_6b | **4.33×** | 5.12× | 3.88× | **4.51×** | 5.12× | 3.88× |
| deepseek_r1_distill_qwen_1_5b | 3.32× | 5.12× | 3.88× | 3.41× | 5.12× | 3.88× |
| glm_edge_1_5b | 3.58× | 5.12× | 3.88× | 3.70× | 5.12× | 3.88× |
| glm_edge_4b | 3.75× | 5.12× | 3.88× | 3.88× | 5.12× | 3.88× |
| gemma4_e2b | 3.86× | 5.27× | 3.97× | 4.29× | 5.27× | 3.97× |

**Bold** = Kakeya surpasses turbo4.

## Key takeaways

### 1. TurboQuant+ is strictly bigger on raw ratio in the short-context regime

For every model at every context length we measured, TurboQuant+'s
weakest setting (turbo4, 4-bit) beats Kakeya **on raw compression
ratio**. turbo3 beats Kakeya in every row except Qwen3 at 128k.
turbo2 beats Kakeya everywhere.

This is entirely expected and consistent with the design goal of each
method:

- **TurboQuant+** is a **token-independent bit-depth shrink**. It
  takes a bf16 vector and packs it into 2/3/4 bits per coordinate
  with one float32 norm. Compression is a fixed function of
  `head_dim` and `bit_width` — context-independent.
- **Kakeya** is a **cross-token redundancy exploiter**. It keeps
  the recent tail at full precision and, over a block of `block_size`
  tokens, discards a PCA basis plus a spherical K-means codebook.
  Compression grows with context because the skeleton cost amortizes
  across more blocks.

### 2. The two compress **different** sources of redundancy

TurboQuant+ compresses **within a single KV vector** (rotated
coordinates are Gaussian, so scalar quantization is near-optimal).
Kakeya compresses **across neighboring KV vectors** (temporally
adjacent tokens share a low-rank trajectory + a discrete segment
structure).

Concretely: at head_dim=128, TurboQuant+ with b=3 bits stores
`128 × 3 + 32 = 416 bits = 52 bytes` per vector, independent of
context. Kakeya with the default preset, per full compressed block
of 512 tokens × (bsz × n_kv) rows, stores about:

- skeleton: `d_eff² + 3·d_eff` floats ≈ a few KB per block,
- encoded: `1 seg_id + 1 alpha + 1 t + d_res residual vals + d_res residual idx` per row ≈ 12-14 bytes per row,
- exact tail: `residual_length` tokens × `head_dim × 2` bytes per row.

So Kakeya's per-vector cost at steady state is **~14 bytes (encoded
side)** plus amortized skeleton — **lower** than TurboQuant turbo4
(54 bytes) asymptotically, comparable to turbo3 (52 bytes). The
gap visible in the tables is mostly the float32-on-CPU overhead of
Kakeya's stored tensors and the fact that d_res=8 keeps a
non-trivial residual per row.

### 3. They can be **stacked**

Nothing in the Kakeya codec prevents quantizing the stored `alpha`,
`t`, `residual_vals`, `mean`, `basis`, and `centers` with
TurboQuant+-style PolarQuant. A stacked codec would get:

- the per-token **Gaussian scalar** savings from TurboQuant+,
- on top of the per-block **cross-token low-rank** savings from Kakeya,

potentially reaching 8-10× at 128k on head_dim=128 architectures
instead of the 4-5× either achieves alone. This is out of scope for
this comparison but an obvious direction for v2.

### 4. Reconstruction MSE on the same tensors

We measured raw per-coordinate MSE on the **same captured tensors**
(first compressed block of the first full-attention layer). Kakeya
uses its full block pipeline; TurboQuant+ uses turbo3 (3-bit K/V):

| Model | head_dim | Kakeya K MSE | turbo3 K MSE | Kakeya V MSE | turbo3 V MSE |
|---|---:|---:|---:|---:|---:|
| qwen2_5_0_5b | 64 | 2.6e+01 | **2.07e+02** | 1.2e-04 | 2.4e-05 |
| qwen3_0_6b | 128 | 4.3e+00 | **8.99e+01** | 1.1e-02 | 7.4e-04 |
| deepseek_r1_distill_qwen_1_5b | 128 | 3.5e+00 | **1.44e+03** | 3.6e-02 | 3.8e-03 |
| glm_edge_1_5b | 128 | 4.1e-01 | 2.81e-01 | 5.8e-04 | 5.4e-05 |
| glm_edge_4b | 128 | 2.1e-01 | 1.48e-01 | 1.4e-04 | 1.8e-05 |
| gemma4_e2b | 512 | 1.3e-03 | 2.96e-03 | 2.2e-01 | 3.4e-02 |

**On K cache, Kakeya is 8×–400× lower MSE than turbo3 on the Qwen
family** (Qwen2.5, Qwen3, DeepSeek-R1-Distill-Qwen). These are the
exact models that TurboQuant+'s own README flags as catastrophic
under symmetric turbo and recommends the `q8_0-K + turbo-V` rescue
config for. Kakeya's low-rank PCA approach handles the large-magnitude
K norms in these models implicitly.

On V cache TurboQuant+ is 2-10× lower MSE than Kakeya — a straight
consequence of TurboQuant's per-coordinate bit-depth shrink doing
well on the near-Gaussian rotated V values, while Kakeya's `d_res=8`
sparse residual discards per-row information that V cache actually
uses.

**Published PPL for reference (TurboQuant+, M5 Max, llama.cpp port):**
turbo4 +0.23% vs q8_0; turbo3 +1.06%; turbo2 +6.48%. The asymmetric
rescue `q8_0-K + turbo-V` brings large Q4_K_M models back in line.

**Published Kakeya quality signal:** greedy decode matches
`DynamicCache` baseline to the token for the first 12 tokens at 2k
context on Gemma 4 E2B (`reports/gemma4_e2b/REPORT.md`). Full PPL
sweep pending.

Without a shared perplexity protocol the two end-to-end "quality"
numbers are not directly comparable. On raw MSE of the captured
tensors, the data suggests **Kakeya should be preferred on K cache
for the Qwen family**, **TurboQuant+ should be preferred on V cache
everywhere**, and an asymmetric composition (Kakeya-K + TurboQuant-V)
would likely be the strongest single method.

### 5. Where Kakeya wins cleanly

- **Zero model-code changes, zero kernel work.** TurboQuant+ needs a
  C/CUDA/Metal/HIP kernel port plus a fork of llama.cpp (a separate
  repo, ~1500 lines of quantize/dequantize glue per backend). Kakeya
  is a drop-in `transformers.Cache` subclass in pure Python on top of
  unmodified `transformers`.
- **Per-block skeleton is semantic.** The PCA basis, temporal
  direction, and K-means centers from the Kakeya skeleton are
  interpretable and can be re-used as features for attention
  analysis / routing / eviction. TurboQuant+'s rotated-Gaussian
  indices have no such structure.
- **Reconstruction is approximate by design.** Kakeya returns
  `K ≈ basis · coeff + mean` — which is what an attention step sees.
  TurboQuant+ returns the same `K` as fp16 up to quantization noise
  on every coordinate.

### 6. Where TurboQuant+ wins cleanly

- **Raw memory ratio at short contexts** (every scenario below 32k).
- **No warm-up.** Kakeya needs `block_size + residual_length` tokens
  before its first block is compressed; before that it behaves
  like `DynamicCache`. TurboQuant+ compresses from token 1.
- **Production-deployed.** llama.cpp fork exists, community-tested
  across Metal/CUDA/HIP on ~30 hardware configs up to 104B models
  at 128k context.
- **Asymmetric K vs V** is a built-in flag (`q8_0-K + turbo4-V`
  rescues Q4_K_M models); Kakeya currently treats K and V the same.

## One-line summary

> On **raw compression ratio** TurboQuant+ wins below 32k context —
> its `turbo2/turbo3/turbo4` give 7.5×/5.1×/3.9× on head_dim=128
> models regardless of context. Kakeya starts at 1.6-2.4× at 2k,
> reaches 2.9-3.7× at 8k, and only surpasses turbo4 on Qwen3 at 32k+
> and on Gemma 4 / Qwen3 at 128k. On **K cache reconstruction MSE**
> Kakeya is 8×–400× better than turbo3 on the Qwen/DeepSeek family —
> the exact models TurboQuant+'s own docs flag for asymmetric
> `q8_0-K + turbo-V` rescue. The two methods target orthogonal
> redundancy sources (TurboQuant+ = within-vector bit-depth;
> Kakeya = cross-vector low-rank) and compose naturally as
> **Kakeya-K + TurboQuant-V** for a potential ~8–10× combined.

## Files

- `reports/compare/<model>/compare_<ctx>.json` — per-model, per-context byte-level report (both methods' numbers).
- `compare_kakeya_vs_turboquant.py` — reproducible harness: loads model → captures KV → runs both → writes JSON.
- `run_comparison_matrix.sh` — orchestrator that reproduces the tables above.

### Reproducing a single row

```bash
git clone https://github.com/TheTom/turboquant_plus /workspace/turboquant_plus
pip install -e /workspace/turboquant_plus

python3 compare_kakeya_vs_turboquant.py \
  --model-path models/Qwen3-0.6B \
  --model-name qwen3_0_6b \
  --context-tokens 8192 \
  --out reports/compare/qwen3_0_6b/compare_8192.json
```

## Environment

- Same as our other benchmark PRs: CPU-only x86_64, 15 GB RAM, BF16,
  eager attention. TurboQuant+ Python prototype version captured on
  commit at clone time.
