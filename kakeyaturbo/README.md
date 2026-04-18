# kakeyaturbo

A monomorphic Rust implementation of the **KakeyaTurbo** compression framework.

## Design

Under Shannon's weighted rate-distortion optimisation, the entire
"Kakeya + TurboQuant" pipeline collapses to one function:

```
encode_block::<R>(vectors, weights, d, params) -> (Skeleton, Vec<Code>)
decode_block::<R>(skeleton, codes)             -> Vec<f32>
```

where `R: Distortion` is a zero-sized type chosen at the call site and
`weights: &[f32]` is a runtime vector.

**All "attention-awareness" (metric choice, per-vector weighting,
norm-storage mode) is expressed as parameters `(ρ, w)` of this single
function** — not as plugins, not as extension points, not as separate
code paths. Each `(R, N, D, d_eff, K, B)` combination compiles to its
own specialised machine-code function; no runtime dispatch ever.

## Quality gates

| gate | status |
|---|---|
| `cargo build` (no warnings on stable) | ✅ |
| `cargo test` | ✅ 136/136 |
| `cargo clippy --all-targets` | ✅ 0 errors, style warnings only |
| `cargo llvm-cov` line coverage | **99.76 %** (100 % of production code) |
| `cargo llvm-cov` function coverage | **100.00 %** |
| `#![forbid(unsafe_code)]` | ✅ |
| `grep -rn "dyn " src/` | ✅ no occurrences in production |
| `grep -rn "Box<" src/` | ✅ no occurrences in production |

The only uncovered lines are assertion-failure messages inside tests
(`assert!(cond, "...")`) that the tests never hit.

## Modules

- `distortion` — `Distortion` trait + zero-sized types `MSE`, `InnerProduct`, `LInf`
- `wht` — Walsh-Hadamard transform + seeded sign flips
- `quantize` — Lloyd-Max codebooks (Normal source) + bit packing
- `pca` — weighted PCA truncated at `d_eff` via `variance_ratio`
- `kmeans` — weighted spherical K-means on perpendicular directions
- `skeleton` — block-level metadata container
- `codec` — `encode_block` / `decode_block` top-level kernel

## Example

```rust
use kakeyaturbo::{encode_block, decode_block, CodecParams, MSE};

let n = 64;
let d = 32;
let block: Vec<f32> = /* your n×d data, row-major */;
let weights = vec![1.0_f32; n];

let params = CodecParams {
    variance_ratio: 0.95,
    k: 8,
    bit_width: 3,
    rotation_seed: 0xCAFE_BABE,
    kmeans_max_iter: 32,
};

let (skeleton, codes) = encode_block::<MSE>(&block, &weights, d, &params);
let recovered = decode_block::<MSE>(&skeleton, &codes);
```

Change `MSE` to `InnerProduct` for attention-K-style inner-product-
preserving compression, or to `LInf` for bounded-error scientific use.
Change `weights` per row to express boundary-layer emphasis (Gemma-style
L4), attention-weight sparsity (L5), or any custom importance profile
— without changing a line of the codec.
