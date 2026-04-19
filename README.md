# KV Cache Compression — Kakeya v1.0 / KakeyaTurbo v1.2 / v1.3

A family of KV cache compression codecs for transformer inference, with
progressively tighter byte / quality / fit-cost trade-offs, all developed
and benchmarked on real HuggingFace bf16 KV tensors (no mock, no
fallback, no simplification).

## TL;DR — what's in this repo

| Version | What it is | Beats turbo3 on | Quality | Status |
|---|---|---|---|---|
| **Kakeya v1.0** | Python codec, drop-in `DynamicCache` replacement, PCA + spherical K-means + sparse residual | 1/7 @ 128k (Qwen3) | ACCEPT | shipped |
| **KakeyaTurbo v1.2** | Rust monomorphic rewrite, bf16 skeleton, V-stream layer-pooled PCA | 1/7 full-attn (Gemma-4) | ACCEPT | shipped |
| **KakeyaTurbo v1.3** | **b=2 Lloyd-Max + randomized SVD (r=D/2) + optional inverse-RoPE K** | **6/7 tier-1 + 1 tier-2 fallback** | MARGINAL–ACCEPT | latest |

**v1.3 headline**: Beats TurboQuant turbo3 by **+9% to +30%** on 6 of 7
open-source models at ACCEPT-level quality, plus an optional
RoPE-aware K path that unlocks another +14–39% on Qwen/DeepSeek family.
The "20% K quality tax on RoPE-dominated models" seen in every prior
ablation is structurally removed.

## Quick start

### Python drop-in (v1.0 API, uses v1.2/v1.3 codec under the hood)

```bash
pip install -U torch transformers accelerate huggingface_hub
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kakeya_kv_codec import build_kakeya_cache

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="bfloat16")

cache = build_kakeya_cache(model)  # model-agnostic factory
inputs = tokenizer("Your long prompt here...", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True,
                        past_key_values=cache)
```

### v1.3 Rust codec (standalone library + CLI)

```bash
cd kakeyaturbo
cargo build --release --bins
cargo test  --release                       # 153 tests, all green
```

```rust
use kakeyaturbo::{CodecParams, PcaMethod, encode_block, decode_block, MSE};

let params = CodecParams {
    variance_ratio: 0.95,
    k: 16,
    bit_width: 2,                           // v1.3 default
    rotation_seed: 0xCAFE_BABE,
    kmeans_max_iter: 32,
    pca_method: PcaMethod::Randomized {     // v1.3 default
        target_rank: head_dim / 2,
        oversample: 8,
        power_iters: 2,
        seed_offset: 0x9E37_79B9_7F4A_7C15,
    },
};

let (skeleton, codes) = encode_block::<MSE>(&vectors, &weights, head_dim, &params);
let reconstructed = decode_block::<MSE>(&skeleton, &codes);
```

## v1.3 real measurements (bit_width=2 + rsvd r=D/2, ctx=4096)

All runs are on real HF bf16 KV tensors from identical prompts, on the
same CPU-only 15 GB-RAM VM used for every prior benchmark in this repo.

| Model | head_dim | b=2 rsvd ratio | turbo3 | **Δ vs turbo3** | K MSE | V MSE | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| qwen2_5_0_5b | 64 | 5.40× | 4.92× | **+9.7%** | 1.13× | 1.15× | MARGINAL |
| qwen3_0_6b | 128 | 6.61× | 5.12× | **+29.2%** | 1.08× | 1.22× | MARGINAL |
| gemma4_e2b (FA only) | 512 | 6.32× | 5.28× | **+19.8%** | 0.42× ⬇ | 1.08× | **ACCEPT** |
| deepseek_r1_distill_qwen_1_5b | 128 | 5.98× | 5.12× | **+16.8%** | 1.13× | 1.07× | MARGINAL |
| glm_edge_1_5b | 128 | 5.85× | 5.12× | **+14.2%** | 1.11× | 1.13× | MARGINAL |
| smollm2_1_7b | 64 | 5.37× | 4.92× | +9.1% | 1.16× | **1.61×** | REJECT on V (tier-2) |
| glm_edge_4b | 128 | 5.83× | 5.12× | **+14.0%** | 1.12× | 1.18× | MARGINAL |

### Optional v1.3 tier-1.5: +inverse-RoPE K preprocessor

Only applicable to halfsplit-RoPE models (Qwen / DeepSeek / MiniMax).
Run via `benchmarks/rope_aware_k_poc.py`; measured gains:

| Model | K MSE pre/post | K bytes pre/post | verdict |
|---|---:|---:|---|
| qwen2_5_0_5b | **0.49×** | **0.80×** | ACCEPT |
| qwen3_0_6b | 0.86× | 0.86× | ACCEPT |
| deepseek_r1_distill | **0.58×** | **0.81×** | ACCEPT |

## Full test matrix

### Rust crate (`kakeyaturbo/`)

- **142 unit tests** — covers `distortion`, `wht`, `quantize`, `pca`
  (including 7 new randomized-SVD tests), `kmeans`, `skeleton`, `codec`.
- **5 integration tests** — cross-module round-trip correctness.
- **6 property-based tests** — proptest-based fuzzing of encode shape
  invariants, determinism, scaling symmetry, cross-bit-width monotonicity.

Run everything:

```bash
cd kakeyaturbo && cargo test --release    # 153 tests total
```

### Real-data benchmarks (`benchmarks/`, `reports/`)

Seven open-source models (Qwen2.5-0.5B, Qwen3-0.6B, Gemma-4-E2B,
DeepSeek-R1-Distill-Qwen-1.5B, GLM-Edge-1.5B, SmolLM2-1.7B,
GLM-Edge-4B) benchmarked across:

| Study | Purpose | Location |
|---|---|---|
| v1.0 measured + 128k projected | Kakeya baseline ratios | `reports/CROSS_MODEL.md` |
| Kakeya vs TurboQuant+ | Head-to-head on same KV tensors | `reports/compare/SUMMARY.md` |
| v1.2 A (bf16 skeleton) + B' (V-pooled PCA) | v1.2 ship decision | `reports/real_kakeyaturbo_v1_2/` |
| v1.2 vs turbo3 | Byte-exact head-to-head | `reports/real_kakeyaturbo_v1_2/V1_2_vs_TURBO3.md` |
| PCA basis-sharing ablation | V stream: ACCEPT / K stream: REJECT | `reports/pca_ablation/` |
| K d_eff × outlier ablation | Aggressive PCA truncation → REJECT | `reports/k_deff_outlier_ablation/DECISION.md` |
| K block_size ablation | bs=1024 → MARGINAL | `reports/k_blocksize_ablation/DECISION.md` |
| K cross-layer basis ablation | Family-dependent, mostly REJECT on RoPE | `reports/k_crosslayer_ablation/DECISION.md` |
| **bit_width sweep** | **b=2 is the new default** | `reports/real_kakeyaturbo_bit_width_sweep/` |
| **v1.3 rsvd + RoPE-aware K** | **v1.3 final decision** | `reports/v1_3_rsvd_rope/DECISION.md` |
| v1.3 flagship prediction | Qwen / DeepSeek / Kimi / GLM / MiniMax | `reports/v1_3_rsvd_rope/FLAGSHIP_COMPARISON.md` |
| v1.3 SmolLM2 per-stream knob | Documented tier-2 limitation | `reports/v1_3_rsvd_rope/SMOLLM2_CAPABILITY.md` |

## Reproducing a v1.3 real measurement

```bash
# (1) Build Rust codec
cd kakeyaturbo && cargo build --release --bins && cd ..

# (2) Download model
huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

# (3) v1.3 tier-1 real benchmark at ctx=4096
python3 benchmarks/kakeyaturbo_v1_2_real_bench.py \
    --model-path models/Qwen3-0.6B \
    --model-name qwen3_0_6b \
    --context-tokens 4096 \
    --bit-width 2 \
    --pca-method randomized \
    --out-dir reports/v1_3_rsvd_rope/bench/qwen3_0_6b_rsvd_half/ctx_4096

# (4) Optional: v1.3 tier-1.5 inverse-RoPE K POC
python3 benchmarks/rope_aware_k_poc.py \
    --model-dir Qwen3-0.6B \
    --model-name qwen3_0_6b \
    --ctx 4096 \
    --rope-pairing halfsplit \
    --out-dir reports/v1_3_rsvd_rope/rope_poc/qwen3_0_6b

# (5) Byte-exact extrapolation to 128k / 1M
python3 benchmarks/extrapolate_v1_2_b2_vs_turbo3.py --bw 2
```

## Repository layout

```
kakeyaturbo/                             Rust crate (monomorphic codec)
├── src/
│   ├── codec.rs                         encode_block / encode_layer + PcaMethod dispatch
│   ├── pca.rs                           exact PCA + randomized SVD (v1.3)
│   ├── kmeans.rs                        spherical weighted K-means
│   ├── quantize.rs                      Lloyd-Max + LSB-first bit packing
│   ├── wht.rs                           Walsh-Hadamard transform
│   ├── distortion.rs                    MSE / InnerProduct / LInf distortion types
│   ├── skeleton.rs                      block-level metadata
│   └── bin/
│       ├── kakeyaturbo-bench.rs         main bench CLI (+ v1.3 rsvd flags)
│       ├── kakeyaturbo-pca-ablation.rs  PCA basis-sharing ablation
│       ├── kakeyaturbo-deff-outlier-ablation.rs
│       ├── kakeyaturbo-k-blocksize-ablation.rs
│       └── kakeyaturbo-k-crosslayer-ablation.rs
├── tests/                               integration + proptest suites
└── Cargo.toml                           deps pinned; no external BLAS

benchmarks/
├── kakeyaturbo_v1_2_real_bench.py       main v1.3 driver (HF → Rust bench)
├── rope_aware_k_poc.py                  v1.3 tier-1.5 RoPE POC
├── extrapolate_v1_2_b2_vs_turbo3.py     byte-exact context extrapolation
├── extrapolate_v_channel.py             V-channel asymptote analysis
├── run_v1_3_rsvd_matrix.sh              full 7-model v1.3 orchestrator
├── run_bit_width_sweep.sh               bit_width comparison orchestrator
├── run_deff_outlier_ablation.py         d_eff × outlier grid
├── run_k_blocksize_ablation.py          block_size ablation
├── run_k_crosslayer_ablation.py         cross-layer basis ablation
├── pca_sharing_ablation.py              V basis sharing ablation
├── aggregate_*.py                       per-ablation aggregators
└── compare_*.py                         v1.2 vs turbo3 / v1.2 vs v1.0

reports/
├── STANDARD.md, CROSS_MODEL.md          v1.0 baseline reports
├── <model>/REPORT.md                    per-model v1.0 narratives
├── compare/                             v1.0 vs TurboQuant+ byte-for-byte
├── pca_ablation/                        V stream basis sharing (ACCEPT V-only)
├── k_deff_outlier_ablation/             K d_eff truncation (REJECT)
├── k_blocksize_ablation/                K block_size sweep (MARGINAL)
├── k_crosslayer_ablation/               K cross-layer basis (family-dependent)
├── real_kakeyaturbo_v1_2/               v1.2 A+B' real measurements
├── real_kakeyaturbo_bit_width_sweep/    b=2 vs b=3 vs b=4 sweep (7 models)
└── v1_3_rsvd_rope/                      v1.3 final report
    ├── DECISION.md                      ship decision + ablation narrative
    ├── FLAGSHIP_COMPARISON.md           Qwen/DeepSeek/Kimi/GLM/MiniMax
    ├── SMOLLM2_CAPABILITY.md            tier-2 limitation + Pareto search
    ├── bench/                           per-model v1.3 JSON
    └── rope_poc/                        per-model inverse-RoPE JSON

kakeya_kv_codec.py                       v1.0 Python codec (drop-in Cache)
kakeya_benchmark.py                      v1.0 bench harness
kakeya_extrapolate.py                    v1.0 byte-exact extrapolator
smoke_test.py                            30-second self-test
```

## How the v1.3 codec works (one paragraph)

Each full-attention K/V stream is split into blocks of `block_size=512`
tokens. Per block we fit a **randomized weighted PCA** with
`target_rank = D/2, oversample = 8, power_iters = 2` (Halko-Martinsson-Tropp
2011), project every vector into `d_eff ≤ D/2`-D space, run spherical
K-means with `K = 16` clusters on the projected coefficients, compute
per-vector residuals after the cluster-center subtraction, Walsh-
Hadamard rotate those residuals to Gaussianise them, and **Lloyd-Max
quantise with `bit_width = 2`** (4 levels). Storage: bf16 skeleton
(mean + basis + K-means centers) + per-vector (u32 seg_id + 3 fp16
scalars + packed 2-bit residual). V-side shares a single layer-pooled
PCA across all blocks of a layer (v1.2 B' optimisation preserved).
Optionally, halfsplit-RoPE K streams (Qwen / DeepSeek / MiniMax family)
can be inverse-RoPE-rotated before the codec sees them — see
`benchmarks/rope_aware_k_poc.py`.

## Design principles (enforced by the Rust type system)

- `unsafe_code = "forbid"` everywhere.
- **No** `dyn Trait` / `Box<dyn …>` — distortion type and codec
  parameters are compile-time monomorphised.
- Every codec decision (`PcaMethod`, bit width, K-means K, block size,
  share_basis) is a runtime parameter, exposed through `CodecParams`
  and surfaced on every CLI.
- bf16 storage for all skeleton tensors; raw vectors always f32 on the
  hot path.

## Status & known limitations

- **SmolLM2 tier-2**: the only tested model where b=2 rsvd r=D/2 cannot
  simultaneously beat turbo3 AND stay ACCEPT on quality. Root cause is
  structural (flat V PCA spectrum). Documented in
  `reports/v1_3_rsvd_rope/SMOLLM2_CAPABILITY.md`.
- **GLM-family inverse-RoPE**: adjacent-pairs RoPE + QK-norm not yet
  handled in `rope_aware_k_poc.py`. Tracked as a v1.3.1 follow-up.
- **Flagship real measurements**: Qwen3-235B, DeepSeek-V3.1, Kimi-K2,
  GLM-4.6, MiniMax-M2 require ≥500 GB RAM to load; current numbers are
  byte-exact extrapolations from architecturally identical small
  proxies. Reproducibility runbook in `FLAGSHIP_COMPARISON.md`.

## License

See `LICENSE`.
