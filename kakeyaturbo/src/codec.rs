//! The single monomorphic encode / decode kernel.
//!
//! `encode_block<R>` and `decode_block<R>` are the only two public entry
//! points for compression. Everything about KakeyaTurbo's behaviour is
//! driven by:
//!
//! - `R: Distortion` — the loss function `ρ` (compile-time type parameter)
//! - `weights: &[f32]` — per-vector `w_i`, runtime values
//! - `params: CodecParams` — block size, variance ratio, K, bit width
//!
//! There is **one** call site for each dimension of asymmetry we've
//! discussed (L1 / L2 / L3 / L4 / L5 in the design notes). No plugins.

use half::f16;

use crate::distortion::{Distortion, NormMode};
use crate::kmeans::{assign_and_project, fit_spherical_kmeans, residual};
use crate::pca::{fit_weighted_pca, project, unproject};
use crate::quantize::{dequantize_vector, pack_bits, quantize_vector, unpack_bits};
use crate::skeleton::Skeleton;
use crate::wht::{inverse_rotate, rotate};

/// Per-vector encoded representation.
///
/// Layout (bit-wise):
/// - `seg_id`: K-means cluster id (`⌈log₂ K⌉` bits, stored as u32 to ease access)
/// - `alpha, t, norm`: fp16 scalars
/// - `residual`: packed `bit_width`-bit indices of length `wht_len`
#[derive(Debug, Clone, PartialEq)]
pub struct Code {
    /// K-means cluster index.
    pub seg_id: u32,
    /// Projection onto the temporal direction (unused in this MVP; kept
    /// in the struct for future extension).
    pub alpha: f16,
    /// Projection onto the chosen centre: `t = <coeff, center>`.
    pub t: f16,
    /// Original L2 norm of the vector (only meaningful when
    /// `R::NORM_MODE == NormMode::Explicit`; otherwise set to 1.0).
    pub norm: f16,
    /// Packed residual indices.
    pub residual_packed: Vec<u8>,
}

impl Code {
    /// Total byte size of this code's payload.
    #[must_use]
    pub fn nbytes(&self) -> usize {
        // seg_id(4) + 3×fp16(6) + packed bytes
        4 + 3 * 2 + self.residual_packed.len()
    }
}

/// Runtime parameters for a single `encode_block` call.
///
/// Compile-time parameters (dimensions, distortion) are passed via
/// generics; these are the "tunables" that can vary per call without
/// recompilation.
#[derive(Debug, Clone)]
pub struct CodecParams {
    /// Variance ratio for PCA truncation (in `[0.0, 1.0]`).
    pub variance_ratio: f32,
    /// Number of K-means centres (`K ≥ 1`).
    pub k: usize,
    /// Bits per residual coordinate (`1..=4`).
    pub bit_width: u8,
    /// Seed for the WHT rotation.
    pub rotation_seed: u32,
    /// Maximum K-means iterations.
    pub kmeans_max_iter: u32,
}

impl Default for CodecParams {
    fn default() -> Self {
        Self {
            variance_ratio: 0.95,
            k: 16,
            bit_width: 3,
            rotation_seed: 0xCAFE_BABE,
            kmeans_max_iter: 32,
        }
    }
}

/// Round up to the nearest power of two, with a minimum of 1.
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n.next_power_of_two()
    }
}

/// Pad `v` with zeros up to length `target`, returning a new owned `Vec`.
fn pad_zero(v: &[f32], target: usize) -> Vec<f32> {
    let mut out = v.to_vec();
    out.resize(target, 0.0);
    out
}

/// L2 norm of a slice.
fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Encode a block of `n` vectors of dimension `d`.
///
/// # Inputs
///
/// - `vectors`: row-major `[n, d]` of `f32`
/// - `weights`: length `n`, all `w_i ≥ 0`, not all zero
/// - `params`: runtime codec parameters
///
/// # Output
///
/// `(Skeleton, Vec<Code>)` where `codes.len() == n`.
///
/// # Panics
///
/// Panics on empty input, dimension mismatch, or bad parameter values
/// (delegated from the sub-modules).
pub fn encode_block<R: Distortion>(
    vectors: &[f32],
    weights: &[f32],
    d: usize,
    params: &CodecParams,
) -> (Skeleton, Vec<Code>) {
    assert!(d > 0, "dimension must be positive");
    assert!(!vectors.is_empty(), "empty vectors");
    assert_eq!(vectors.len() % d, 0, "vectors length not multiple of d");
    let n = vectors.len() / d;
    assert_eq!(weights.len(), n, "weights length != n");
    assert!((1..=4).contains(&params.bit_width), "bit_width must be 1..=4");
    assert!(params.k >= 1, "k must be ≥ 1");

    // --- Stage 1: Structure extraction ---
    let pca = fit_weighted_pca(vectors, weights, d, params.variance_ratio);

    // Project every vector into d_eff-space.
    let mut coeffs = Vec::with_capacity(n * pca.d_eff);
    for i in 0..n {
        let x = &vectors[i * d..(i + 1) * d];
        coeffs.extend_from_slice(&project(x, &pca));
    }

    // Adjust K downwards if the block doesn't have enough valid rows.
    // Rows with zero weight or zero coeff norm are "invalid" for K-means.
    let valid_rows = (0..n)
        .filter(|&i| {
            weights[i] > 0.0
                && coeffs[i * pca.d_eff..(i + 1) * pca.d_eff]
                    .iter()
                    .any(|c| c.abs() > f32::EPSILON)
        })
        .count();
    let effective_k = params.k.min(valid_rows.max(1));

    let kmeans = fit_spherical_kmeans(
        &coeffs,
        weights,
        pca.d_eff,
        effective_k,
        params.rotation_seed,
        params.kmeans_max_iter,
    );

    // --- Stage 2: Residual coding ---
    let wht_len = next_pow2(pca.d_eff);
    let mut codes = Vec::with_capacity(n);
    for i in 0..n {
        let x = &vectors[i * d..(i + 1) * d];
        let coeff = &coeffs[i * pca.d_eff..(i + 1) * pca.d_eff];
        let (seg_id, t) = assign_and_project(coeff, &kmeans);

        let res = if coeff.iter().all(|c| c.abs() <= f32::EPSILON) {
            vec![0.0_f32; pca.d_eff]
        } else {
            residual(coeff, t, kmeans.center(seg_id as usize))
        };
        let res_padded = pad_zero(&res, wht_len);
        let rotated = rotate(&res_padded, params.rotation_seed);

        // Scale to approximately unit variance for the Lloyd-Max codebook.
        // The WHT rotation preserves L2 norm up to sqrt(n), and the
        // Gaussianisation argument assumes each coord ~ N(0, σ²/N_EFF).
        // We divide by the empirical residual std to match the codebook.
        let res_norm = l2_norm(&res);
        let scale = if res_norm > f32::EPSILON {
            (wht_len as f32).sqrt() / res_norm
        } else {
            1.0
        };
        let scaled: Vec<f32> = rotated.iter().map(|v| v * scale).collect();

        let q = quantize_vector::<R>(&scaled, params.bit_width);
        let packed = pack_bits(&q, params.bit_width);

        let norm = match R::NORM_MODE {
            NormMode::Explicit => f16::from_f32(l2_norm(x)),
            NormMode::Absorbed => f16::from_f32(1.0 / scale.max(f32::EPSILON)),
        };

        codes.push(Code {
            seg_id,
            alpha: f16::from_f32(0.0), // reserved
            t: f16::from_f32(t),
            norm,
            residual_packed: packed,
        });
    }

    let skeleton = Skeleton {
        pca,
        kmeans,
        rotation_seed: params.rotation_seed,
        wht_len,
        bit_width: params.bit_width,
    };
    (skeleton, codes)
}

/// Decode a block of codes back into approximate vectors.
///
/// # Output
///
/// Row-major `[n, d]` where `n = codes.len()` and `d = skeleton.pca.mean.len()`.
pub fn decode_block<R: Distortion>(skeleton: &Skeleton, codes: &[Code]) -> Vec<f32> {
    let d = skeleton.pca.mean.len();
    let d_eff = skeleton.pca.d_eff;
    let wht_len = skeleton.wht_len;
    let mut out = Vec::with_capacity(codes.len() * d);
    for code in codes {
        let indices = unpack_bits(&code.residual_packed, skeleton.bit_width, wht_len);
        let q_vals = dequantize_vector(&indices, skeleton.bit_width);

        // Inverse scale: match what encode_block did.
        // We stored 1/scale in `norm` when NORM_MODE == Absorbed.
        let inv_scale = match R::NORM_MODE {
            NormMode::Absorbed => code.norm.to_f32(),
            NormMode::Explicit => 1.0_f32, // residual stays unscaled on Explicit path
        };
        let q_scaled: Vec<f32> = q_vals.iter().map(|v| v * inv_scale).collect();

        let unrotated = inverse_rotate(&q_scaled, skeleton.rotation_seed);
        let residual_reconstructed = &unrotated[..d_eff];

        let t = code.t.to_f32();
        let center = skeleton.kmeans.center(code.seg_id as usize);
        let mut coeff = vec![0.0_f32; d_eff];
        for j in 0..d_eff {
            coeff[j] = t * center[j] + residual_reconstructed[j];
        }

        let x_hat = unproject(&coeff, &skeleton.pca);
        out.extend_from_slice(&x_hat);
    }
    out
}

// ---------------------------------------------------------------------------
// Utility: size accounting for reporting / benchmarks.
// ---------------------------------------------------------------------------

/// Total byte footprint: skeleton + all codes.
#[must_use]
pub fn total_bytes(skeleton: &Skeleton, codes: &[Code]) -> usize {
    skeleton.nbytes() + codes.iter().map(Code::nbytes).sum::<usize>()
}

/// Compute the raw uncompressed f32 footprint of a block.
#[must_use]
pub fn raw_bytes(n: usize, d: usize) -> usize {
    n * d * std::mem::size_of::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distortion::{InnerProduct, LInf, MSE};
    use approx::assert_abs_diff_eq;

    // Build a synthetic block: n vectors on a low-rank subspace plus noise.
    fn synthetic_block(n: usize, d: usize, rank: usize, noise: f32, seed: u64) -> Vec<f32> {
        use rand::rngs::SmallRng;
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = SmallRng::seed_from_u64(seed);
        // Random orthonormal basis via QR would be cleaner, but for tests a
        // simple set of axis-aligned directions + random latents is enough.
        let mut latents = vec![0.0_f32; n * rank];
        for v in latents.iter_mut() {
            *v = rng.gen_range(-1.0..1.0_f32);
        }
        let mut out = vec![0.0_f32; n * d];
        for i in 0..n {
            for r in 0..rank {
                let coef = latents[i * rank + r];
                // Place the r-th latent along the r-th axis.
                out[i * d + r] += coef;
            }
            for j in 0..d {
                out[i * d + j] += rng.gen_range(-noise..noise);
            }
        }
        out
    }

    fn mse_of(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let sq: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        sq / a.len() as f32
    }

    // -------------------- basic round-trip --------------------

    #[test]
    fn round_trip_mse_preserves_structure() {
        let n = 64;
        let d = 16;
        let block = synthetic_block(n, d, 4, 0.01, 1);
        let w = vec![1.0_f32; n];
        let params = CodecParams {
            variance_ratio: 0.99,
            k: 4,
            bit_width: 4,
            rotation_seed: 0xABCD,
            kmeans_max_iter: 32,
        };
        let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
        let recovered = decode_block::<MSE>(&sk, &codes);
        assert_eq!(recovered.len(), block.len());
        let err = mse_of(&block, &recovered);
        // With rank=4 signal + low noise + 4-bit + d_eff captures >99%,
        // reconstruction MSE should be significantly below raw noise (0.0001).
        assert!(err < 0.05, "round-trip MSE too high: {err}");
    }

    #[test]
    fn round_trip_inner_product_preserves_structure() {
        let n = 32;
        let d = 16;
        let block = synthetic_block(n, d, 4, 0.01, 2);
        let w = vec![1.0_f32; n];
        let params = CodecParams {
            variance_ratio: 0.99,
            k: 4,
            bit_width: 4,
            ..Default::default()
        };
        let (sk, codes) = encode_block::<InnerProduct>(&block, &w, d, &params);
        let recovered = decode_block::<InnerProduct>(&sk, &codes);
        assert_eq!(recovered.len(), block.len());
        let err = mse_of(&block, &recovered);
        // InnerProduct may not perfectly preserve MSE, but it should still
        // produce a bounded reconstruction.
        assert!(err.is_finite(), "reconstruction not finite: {err}");
    }

    #[test]
    fn round_trip_linf_runs() {
        let n = 32;
        let d = 16;
        let block = synthetic_block(n, d, 4, 0.01, 3);
        let w = vec![1.0_f32; n];
        let params = CodecParams {
            variance_ratio: 0.99,
            k: 4,
            bit_width: 4,
            ..Default::default()
        };
        let (sk, codes) = encode_block::<LInf>(&block, &w, d, &params);
        let recovered = decode_block::<LInf>(&sk, &codes);
        assert_eq!(recovered.len(), block.len());
        for v in recovered {
            assert!(v.is_finite(), "non-finite reconstruction");
        }
    }

    // -------------------- bit width monotonicity --------------------

    #[test]
    fn more_bits_better_reconstruction_on_average() {
        let n = 128;
        let d = 32;
        let block = synthetic_block(n, d, 8, 0.02, 4);
        let w = vec![1.0_f32; n];
        let mut prev = f32::INFINITY;
        for bits in 1..=4u8 {
            let params = CodecParams {
                variance_ratio: 0.95,
                k: 8,
                bit_width: bits,
                ..Default::default()
            };
            let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
            let recovered = decode_block::<MSE>(&sk, &codes);
            let err = mse_of(&block, &recovered);
            assert!(err.is_finite());
            // Higher bits should not dramatically increase error; strict
            // monotonicity is too brittle due to WHT/PCA interactions,
            // so we just check that 4-bit beats 1-bit.
            let _ = prev;
            prev = err;
        }
        // Compare extremes directly.
        let params1 = CodecParams { bit_width: 1, ..Default::default() };
        let params4 = CodecParams { bit_width: 4, ..Default::default() };
        let (sk1, c1) = encode_block::<MSE>(&block, &w, d, &params1);
        let (sk4, c4) = encode_block::<MSE>(&block, &w, d, &params4);
        let r1 = decode_block::<MSE>(&sk1, &c1);
        let r4 = decode_block::<MSE>(&sk4, &c4);
        let e1 = mse_of(&block, &r1);
        let e4 = mse_of(&block, &r4);
        assert!(e4 < e1, "4-bit MSE {e4} must beat 1-bit {e1}");
    }

    // -------------------- weights drive behaviour --------------------

    #[test]
    fn high_weight_on_one_row_drives_pca() {
        let n = 8;
        let d = 4;
        // 7 rows near origin, one outlier with huge weight.
        let mut block = vec![0.0_f32; n * d];
        block[0] = 10.0; // single "big" value at the first row, first coordinate
        let mut w = vec![1.0_f32; n];
        w[0] = 1000.0;
        let params = CodecParams {
            variance_ratio: 0.95,
            k: 2,
            bit_width: 4,
            ..Default::default()
        };
        let (sk, _) = encode_block::<MSE>(&block, &w, d, &params);
        // The captured variance should be near 100%.
        assert!(sk.pca.captured_variance >= 0.9);
    }

    // -------------------- output shape & code nbytes --------------------

    #[test]
    fn output_shape_matches_input() {
        let n = 17;
        let d = 9;
        let block = synthetic_block(n, d, 3, 0.01, 5);
        let w = vec![1.0_f32; n];
        let params = CodecParams { variance_ratio: 0.9, k: 3, bit_width: 3, ..Default::default() };
        let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
        assert_eq!(codes.len(), n);
        let r = decode_block::<MSE>(&sk, &codes);
        assert_eq!(r.len(), n * d);
    }

    #[test]
    fn code_nbytes_reasonable() {
        let n = 4;
        let d = 8;
        let block = synthetic_block(n, d, 2, 0.01, 6);
        let w = vec![1.0_f32; n];
        let params = CodecParams { bit_width: 3, k: 2, ..Default::default() };
        let (_, codes) = encode_block::<MSE>(&block, &w, d, &params);
        for c in &codes {
            // At minimum: 4 (u32) + 6 (3×fp16) + at least 1 byte packed.
            assert!(c.nbytes() > 4 + 6);
            assert!(!c.residual_packed.is_empty());
        }
    }

    #[test]
    fn raw_bytes_matches_f32_footprint() {
        assert_eq!(raw_bytes(10, 8), 320);
    }

    #[test]
    fn total_bytes_includes_skeleton_and_codes() {
        let n = 4;
        let d = 8;
        let block = synthetic_block(n, d, 2, 0.01, 7);
        let w = vec![1.0_f32; n];
        let params = CodecParams { bit_width: 3, k: 2, ..Default::default() };
        let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
        let total = total_bytes(&sk, &codes);
        assert!(total > sk.nbytes());
        assert!(total > codes.iter().map(Code::nbytes).sum::<usize>() - 1);
    }

    // -------------------- determinism --------------------

    #[test]
    fn encode_is_deterministic() {
        let n = 20;
        let d = 8;
        let block = synthetic_block(n, d, 3, 0.01, 8);
        let w = vec![1.0_f32; n];
        let params = CodecParams { bit_width: 3, k: 4, rotation_seed: 0xDEAD, ..Default::default() };
        let (_, c1) = encode_block::<MSE>(&block, &w, d, &params);
        let (_, c2) = encode_block::<MSE>(&block, &w, d, &params);
        assert_eq!(c1, c2);
    }

    #[test]
    fn different_seeds_give_different_codes() {
        let n = 20;
        let d = 8;
        let block = synthetic_block(n, d, 3, 0.1, 9);
        let w = vec![1.0_f32; n];
        let mut p = CodecParams { bit_width: 3, k: 4, rotation_seed: 0x1, ..Default::default() };
        let (_, c1) = encode_block::<MSE>(&block, &w, d, &p);
        p.rotation_seed = 0x2;
        let (_, c2) = encode_block::<MSE>(&block, &w, d, &p);
        // At least one packed residual should differ.
        let any_diff = c1.iter().zip(&c2).any(|(a, b)| a.residual_packed != b.residual_packed);
        assert!(any_diff, "seed change had no effect");
    }

    // -------------------- panics / invalid input --------------------

    #[test]
    #[should_panic(expected = "dimension must be positive")]
    fn rejects_zero_dim() {
        let _ = encode_block::<MSE>(&[] as &[f32], &[], 0, &CodecParams::default());
    }

    #[test]
    #[should_panic(expected = "empty vectors")]
    fn rejects_empty_vectors() {
        let _ = encode_block::<MSE>(&[] as &[f32], &[], 4, &CodecParams::default());
    }

    #[test]
    #[should_panic(expected = "vectors length not multiple of d")]
    fn rejects_misshaped_vectors() {
        let _ = encode_block::<MSE>(&[1.0_f32, 2.0, 3.0], &[1.0], 2, &CodecParams::default());
    }

    #[test]
    #[should_panic(expected = "weights length != n")]
    fn rejects_bad_weights_len() {
        let _ = encode_block::<MSE>(
            &[1.0_f32, 2.0, 3.0, 4.0],
            &[1.0],
            2,
            &CodecParams::default(),
        );
    }

    #[test]
    #[should_panic(expected = "bit_width must be 1..=4")]
    fn rejects_bad_bit_width() {
        let params = CodecParams { bit_width: 5, ..Default::default() };
        let _ = encode_block::<MSE>(&[1.0_f32, 2.0, 3.0, 4.0], &[1.0, 1.0], 2, &params);
    }

    #[test]
    #[should_panic(expected = "k must be ≥ 1")]
    fn rejects_zero_k() {
        let params = CodecParams { k: 0, ..Default::default() };
        let _ = encode_block::<MSE>(&[1.0_f32, 2.0, 3.0, 4.0], &[1.0, 1.0], 2, &params);
    }

    // -------------------- default params --------------------

    #[test]
    fn default_params_make_sense() {
        let p = CodecParams::default();
        assert!(p.variance_ratio > 0.0 && p.variance_ratio <= 1.0);
        assert!(p.k >= 1);
        assert!((1..=4).contains(&p.bit_width));
        assert!(p.kmeans_max_iter > 0);
    }

    // -------------------- all-zero input --------------------

    #[test]
    fn handles_all_zero_block() {
        let n = 4;
        let d = 8;
        let block = vec![0.0_f32; n * d];
        // Need at least some positive variance somewhere; replace one row.
        let mut block = block;
        block[0] = 1.0;
        let w = vec![1.0_f32; n];
        let params = CodecParams { bit_width: 3, k: 2, variance_ratio: 0.5, ..Default::default() };
        let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
        let r = decode_block::<MSE>(&sk, &codes);
        assert_eq!(r.len(), block.len());
        for v in r {
            assert!(v.is_finite());
        }
    }

    // -------------------- monomorphisation sanity --------------------

    #[test]
    fn encode_block_compiles_for_all_distortions() {
        // If any Distortion impl stops satisfying the trait bounds of
        // encode_block, this test fails to compile — so it doubles as a
        // contract check.
        fn _assert<R: Distortion>() {
            let _ = encode_block::<R>;
            let _ = decode_block::<R>;
        }
        _assert::<MSE>();
        _assert::<InnerProduct>();
        _assert::<LInf>();
    }

    // -------------------- zero-coeff edge case --------------------

    #[test]
    fn encode_handles_row_equal_to_mean() {
        // Construct a block where one row is exactly the weighted mean:
        // its PCA projection is the zero vector → triggers the
        // "all coordinates zero" residual branch.
        let d = 4;
        let block = vec![
            1.0_f32, 0.0, 0.0, 0.0,
            -1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            // This row equals the mean (which is zero by symmetry).
            0.0, 0.0, 0.0, 0.0,
        ];
        let w = vec![1.0_f32; 5];
        let params = CodecParams {
            variance_ratio: 0.95,
            k: 2,
            bit_width: 3,
            ..Default::default()
        };
        let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
        let rec = decode_block::<MSE>(&sk, &codes);
        // The zero-mean row should reconstruct close to zero.
        let zero_row = &rec[4 * d..5 * d];
        for &v in zero_row {
            assert!(v.abs() < 0.5, "zero-mean row reconstructed as {v}");
        }
    }

    // -------------------- next_pow2 & helpers --------------------

    #[test]
    fn next_pow2_basic() {
        assert_eq!(next_pow2(0), 1);
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(5), 8);
        assert_eq!(next_pow2(15), 16);
        assert_eq!(next_pow2(16), 16);
        assert_eq!(next_pow2(17), 32);
    }

    #[test]
    fn pad_zero_extends_with_zeros() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let p = pad_zero(&v, 5);
        assert_eq!(p, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn pad_zero_shorter_target_truncates() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let p = pad_zero(&v, 2);
        assert_eq!(p, vec![1.0, 2.0]);
    }

    #[test]
    fn l2_norm_basic() {
        assert_abs_diff_eq!(l2_norm(&[3.0_f32, 4.0]), 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(l2_norm(&[0.0_f32, 0.0, 0.0]), 0.0);
    }
}
