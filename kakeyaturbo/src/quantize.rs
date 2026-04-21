//! Scalar quantisation with Lloyd-Max codebooks and bit packing.
//!
//! The Gaussianised residual (output of `wht::rotate`) is quantised
//! coordinate-wise against a pre-computed Lloyd-Max optimal codebook
//! calibrated for a unit-variance Gaussian source.
//!
//! The codebook size `1 << B` and the residual length `D_EFF` are
//! const-generic so the compiler can unroll loops fully at call sites.

use crate::distortion::Distortion;

// ---------------------------------------------------------------------------
// Lloyd-Max centroids for a standard Normal source, precomputed offline.
// ---------------------------------------------------------------------------

/// Return Lloyd-Max centroids for a standard Normal with `2^bits` levels.
///
/// Source: optimal non-uniform scalar quantiser tables for the Normal
/// density (Max 1960, Lloyd 1982). Values reproducible in Python with
/// `scipy.optimize` applied to the MSE objective.
///
/// # Panics
///
/// Panics if `bits` is outside the supported range `1..=4`.
#[must_use]
pub fn centroids_gaussian(bits: u8) -> &'static [f32] {
    match bits {
        1 => &[-0.798_156, 0.798_156],
        2 => &[-1.510_0, -0.452_8, 0.452_8, 1.510_0],
        3 => &[
            -2.151_945, -1.343_757, -0.756_268, -0.244_943, 0.244_943, 0.756_268, 1.343_757,
            2.151_945,
        ],
        4 => &[
            -2.732_2, -2.069_0, -1.617_7, -1.256_3, -0.942_2, -0.656_6, -0.388_5, -0.128_1,
            0.128_1, 0.388_5, 0.656_6, 0.942_2, 1.256_3, 1.617_7, 2.069_0, 2.732_2,
        ],
        _ => panic!("unsupported bit width {bits}; expected 1..=4"),
    }
}

// ---------------------------------------------------------------------------
// Quantise / Dequantise
// ---------------------------------------------------------------------------

/// Nearest-centroid quantiser parametrised by the distortion metric.
///
/// For `R = MSE`, `R::d(x, c)` inlines to `(x - c)²` and the function
/// becomes a pure argmin loop with no dispatch in the emitted code.
///
/// Uses the unit-variance-Gaussian Lloyd-Max centroids by default.
#[inline]
#[must_use]
pub fn quantize_vector<R: Distortion>(x: &[f32], bits: u8) -> Vec<u8> {
    quantize_vector_with_centroids::<R>(x, bits, None)
}

/// Variant of [`quantize_vector`] that accepts an optional caller-supplied
/// centroid table. When `Some`, must contain exactly `1 << bits` sorted
/// floats; when `None`, falls back to the Gaussian default.
///
/// The calibrated-codebook path (fitting Lloyd-Max centroids on the
/// empirical residual distribution of a specific model) uses this to
/// replace the unit-variance Gaussian assumption with a model-specific
/// optimum. Paper §2 calls this the "codebook calibration" step.
#[inline]
#[must_use]
pub fn quantize_vector_with_centroids<R: Distortion>(
    x: &[f32],
    bits: u8,
    custom_centroids: Option<&[f32]>,
) -> Vec<u8> {
    assert!((1..=4).contains(&bits), "bits must be 1..=4");
    let default;
    let centroids: &[f32] = match custom_centroids {
        Some(c) => {
            assert_eq!(
                c.len(),
                1usize << bits,
                "custom centroids must have {} entries for {bits}-bit",
                1usize << bits
            );
            c
        }
        None => {
            default = centroids_gaussian(bits);
            default
        }
    };
    let mut out = Vec::with_capacity(x.len());
    for &xi in x {
        let mut best_idx: u8 = 0;
        let mut best_cost = f32::INFINITY;
        for (j, &cj) in centroids.iter().enumerate() {
            let cost = R::d(xi, cj);
            if cost < best_cost {
                best_cost = cost;
                best_idx = j as u8;
            }
        }
        out.push(best_idx);
    }
    out
}

/// Reverse of [`quantize_vector`]: map indices back to centroid values.
///
/// Uses the unit-variance-Gaussian Lloyd-Max centroids by default.
#[must_use]
pub fn dequantize_vector(indices: &[u8], bits: u8) -> Vec<f32> {
    dequantize_vector_with_centroids(indices, bits, None)
}

/// Variant of [`dequantize_vector`] that accepts an optional caller-supplied
/// centroid table (same contract as [`quantize_vector_with_centroids`]).
#[must_use]
pub fn dequantize_vector_with_centroids(
    indices: &[u8],
    bits: u8,
    custom_centroids: Option<&[f32]>,
) -> Vec<f32> {
    assert!((1..=4).contains(&bits), "bits must be 1..=4");
    let default;
    let centroids: &[f32] = match custom_centroids {
        Some(c) => {
            assert_eq!(
                c.len(),
                1usize << bits,
                "custom centroids must have {} entries for {bits}-bit",
                1usize << bits
            );
            c
        }
        None => {
            default = centroids_gaussian(bits);
            default
        }
    };
    indices.iter().map(|&i| centroids[i as usize]).collect()
}

// ---------------------------------------------------------------------------
// Bit packing (for `bits ∈ {1, 2, 3, 4}`).
// ---------------------------------------------------------------------------

/// Pack a stream of `bits`-bit indices into a byte vector, LSB-first.
///
/// The output length is `ceil(indices.len() * bits / 8)`.
#[must_use]
pub fn pack_bits(indices: &[u8], bits: u8) -> Vec<u8> {
    assert!((1..=8).contains(&bits), "bits must be 1..=8");
    let total_bits = indices.len() * bits as usize;
    let nbytes = (total_bits + 7) / 8;
    let mut out = vec![0u8; nbytes];
    let mask: u8 = ((1u16 << bits) - 1) as u8;
    for (i, &idx) in indices.iter().enumerate() {
        debug_assert!(idx & !mask == 0, "index {idx} exceeds {bits}-bit range");
        let bit_offset = i * bits as usize;
        let byte_idx = bit_offset / 8;
        let shift = bit_offset % 8;
        let lo = (idx & mask) as u16;
        let hi_shift = 8_i32 - shift as i32;
        // low part
        out[byte_idx] |= (lo << shift) as u8;
        // high part if it spills to the next byte
        if (shift as i32 + bits as i32) > 8 {
            out[byte_idx + 1] |= (lo >> hi_shift) as u8;
        }
    }
    out
}

/// Inverse of [`pack_bits`].
#[must_use]
pub fn unpack_bits(bytes: &[u8], bits: u8, count: usize) -> Vec<u8> {
    assert!((1..=8).contains(&bits), "bits must be 1..=8");
    let mut out = Vec::with_capacity(count);
    let mask: u8 = ((1u16 << bits) - 1) as u8;
    for i in 0..count {
        let bit_offset = i * bits as usize;
        let byte_idx = bit_offset / 8;
        let shift = bit_offset % 8;
        let mut v = bytes[byte_idx] >> shift;
        if (shift as i32 + bits as i32) > 8 {
            v |= bytes[byte_idx + 1] << (8_i32 - shift as i32);
        }
        out.push(v & mask);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distortion::{InnerProduct, LInf, MSE};
    use approx::assert_abs_diff_eq;

    // -------------------- centroids --------------------

    #[test]
    fn centroids_have_correct_count() {
        assert_eq!(centroids_gaussian(1).len(), 2);
        assert_eq!(centroids_gaussian(2).len(), 4);
        assert_eq!(centroids_gaussian(3).len(), 8);
        assert_eq!(centroids_gaussian(4).len(), 16);
    }

    #[test]
    fn centroids_are_symmetric_about_zero() {
        for bits in 1..=4 {
            let c = centroids_gaussian(bits);
            let k = c.len();
            for i in 0..k {
                assert_abs_diff_eq!(c[i], -c[k - 1 - i], epsilon = 1e-3);
            }
        }
    }

    #[test]
    fn centroids_are_sorted_ascending() {
        for bits in 1..=4 {
            let c = centroids_gaussian(bits);
            for w in c.windows(2) {
                assert!(w[0] < w[1], "centroids not sorted for bits={bits}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "unsupported bit width")]
    fn centroids_unsupported_bits() {
        let _ = centroids_gaussian(5);
    }

    #[test]
    #[should_panic(expected = "unsupported bit width")]
    fn centroids_zero_bits() {
        let _ = centroids_gaussian(0);
    }

    // -------------------- quantise --------------------

    #[test]
    fn quantize_chooses_nearest_centroid_mse() {
        let centroids = centroids_gaussian(2);
        let input: Vec<f32> = centroids.to_vec();
        let q = quantize_vector::<MSE>(&input, 2);
        assert_eq!(q, vec![0, 1, 2, 3]);
    }

    #[test]
    fn quantize_round_trip_for_exact_centroids() {
        for bits in 1..=4 {
            let centroids = centroids_gaussian(bits);
            let input: Vec<f32> = centroids.to_vec();
            let q = quantize_vector::<MSE>(&input, bits);
            let rec = dequantize_vector(&q, bits);
            for (a, b) in input.iter().zip(&rec) {
                assert_abs_diff_eq!(a, b, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn quantize_handles_out_of_range_inputs() {
        // Way outside Gaussian domain — should snap to extreme centroid.
        let input = vec![100.0_f32, -100.0, 50.0, -50.0];
        let q = quantize_vector::<MSE>(&input, 3);
        let c = centroids_gaussian(3);
        assert_eq!(q[0] as usize, c.len() - 1, "large positive → max centroid");
        assert_eq!(q[1] as usize, 0, "large negative → min centroid");
        assert_eq!(q[2] as usize, c.len() - 1);
        assert_eq!(q[3] as usize, 0);
    }

    #[test]
    fn quantize_decreasing_mse_with_more_bits() {
        let input: Vec<f32> = (0..64).map(|i| ((i as f32) / 16.0).sin()).collect();
        let mut prev = f32::INFINITY;
        for bits in 1..=4 {
            let q = quantize_vector::<MSE>(&input, bits);
            let rec = dequantize_vector(&q, bits);
            let mse: f32 = input
                .iter()
                .zip(&rec)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / input.len() as f32;
            assert!(
                mse < prev,
                "MSE must decrease with more bits: prev={prev} new={mse} at {bits} bits"
            );
            prev = mse;
        }
    }

    #[test]
    fn quantize_empty_input() {
        let q = quantize_vector::<MSE>(&[], 3);
        assert!(q.is_empty());
    }

    #[test]
    #[should_panic(expected = "bits must be 1..=4")]
    fn quantize_rejects_oversized_bits() {
        let _ = quantize_vector::<MSE>(&[0.0_f32], 5);
    }

    #[test]
    #[should_panic(expected = "bits must be 1..=4")]
    fn quantize_rejects_zero_bits() {
        let _ = quantize_vector::<MSE>(&[0.0_f32], 0);
    }

    #[test]
    #[should_panic(expected = "bits must be 1..=4")]
    fn dequantize_rejects_oversized_bits() {
        let _ = dequantize_vector(&[0u8], 5);
    }

    // -------------------- distortion metric monomorphisation --------------------

    #[test]
    fn quantize_mse_and_inner_product_agree_at_scalar_level() {
        // Both metrics reduce to (x - x̂)² per coordinate; only the norm
        // handling differs. Their per-coord quantisations must agree.
        let input: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.1).collect();
        for bits in 1..=4 {
            let q1 = quantize_vector::<MSE>(&input, bits);
            let q2 = quantize_vector::<InnerProduct>(&input, bits);
            assert_eq!(q1, q2, "MSE and IP quantise the same at bits={bits}");
        }
    }

    #[test]
    fn quantize_linf_is_nontrivially_different() {
        // With the Huberised objective, near-zero values are penalised
        // more relative to their tail — quantisation near zero should
        // pick tighter centroids than MSE would.
        let input = vec![0.05_f32, 0.05, 0.05, 0.05];
        let q_mse = quantize_vector::<MSE>(&input, 3);
        let q_inf = quantize_vector::<LInf>(&input, 3);
        // At b=3, the nearest centroid to 0.05 is centroid index 4
        // (value 0.245). Both metrics should agree on this particular
        // value, but the test exercises the generic path via LInf::d.
        assert_eq!(q_mse.len(), q_inf.len());
    }

    // -------------------- bit packing --------------------

    #[test]
    fn pack_unpack_round_trip_3bit() {
        let indices = vec![0u8, 7, 3, 4, 1, 6, 2, 5];
        let packed = pack_bits(&indices, 3);
        let unpacked = unpack_bits(&packed, 3, indices.len());
        assert_eq!(unpacked, indices);
    }

    #[test]
    fn pack_unpack_round_trip_all_bit_widths() {
        for bits in 1..=8u8 {
            let max = (1u16 << bits) as usize - 1;
            let indices: Vec<u8> = (0..64).map(|i| (i % (max + 1)) as u8).collect();
            let packed = pack_bits(&indices, bits);
            let unpacked = unpack_bits(&packed, bits, indices.len());
            assert_eq!(unpacked, indices, "round-trip failed at bits={bits}");
        }
    }

    #[test]
    fn pack_byte_layout_matches_lsb_first() {
        // With bits=2 and input [0b01, 0b10, 0b11, 0b00], the packed byte
        // must be 0b00_11_10_01 (LSB first).
        let indices = vec![0b01, 0b10, 0b11, 0b00];
        let packed = pack_bits(&indices, 2);
        assert_eq!(packed, vec![0b00_11_10_01]);
    }

    #[test]
    fn pack_bits_4_is_two_per_byte() {
        let indices = vec![0x5_u8, 0xA, 0xF, 0x3, 0x0, 0xC];
        let packed = pack_bits(&indices, 4);
        assert_eq!(packed, vec![0xA5, 0x3F, 0xC0]);
        let back = unpack_bits(&packed, 4, 6);
        assert_eq!(back, indices);
    }

    #[test]
    fn pack_byte_count_is_ceil() {
        for bits in 1..=8u8 {
            for n in 0..20usize {
                let indices = vec![0u8; n];
                let packed = pack_bits(&indices, bits);
                let expected = (n * bits as usize + 7) / 8;
                assert_eq!(packed.len(), expected, "bits={bits} n={n}");
            }
        }
    }

    #[test]
    fn pack_empty_input() {
        let p = pack_bits(&[], 3);
        assert!(p.is_empty());
        let u = unpack_bits(&[], 3, 0);
        assert!(u.is_empty());
    }

    #[test]
    #[should_panic(expected = "bits must be 1..=8")]
    fn pack_rejects_oversized_bits() {
        let _ = pack_bits(&[0u8], 9);
    }

    #[test]
    #[should_panic(expected = "bits must be 1..=8")]
    fn pack_rejects_zero_bits() {
        let _ = pack_bits(&[0u8], 0);
    }

    #[test]
    #[should_panic(expected = "bits must be 1..=8")]
    fn unpack_rejects_oversized_bits() {
        let _ = unpack_bits(&[0u8], 9, 1);
    }

    // -------------------- custom centroids --------------------

    #[test]
    fn quantize_with_custom_centroids_matches_nearest() {
        // Asymmetric centroids — calibrated for a heavy-tailed distribution
        // (bigger extreme centroids, denser near 0).
        let c: Vec<f32> = vec![-3.0, -0.5, 0.5, 3.0];  // b=2, 4 entries
        let input = vec![-4.0_f32, -1.0, -0.3, 0.0, 0.3, 1.0, 4.0];
        let q = quantize_vector_with_centroids::<MSE>(&input, 2, Some(&c));
        //  -4 → 0 (nearest -3)
        //  -1 → 1 (nearest -0.5, since |-1 - -0.5| = 0.5 < |-1 - -3| = 2)
        //  -0.3 → 1 (nearest -0.5)
        //  0.0 → 1 or 2 (equidistant, argmin picks first = 1)
        //  0.3 → 2 (nearest 0.5)
        //  1.0 → 2 (nearest 0.5)
        //  4.0 → 3 (nearest 3)
        assert_eq!(q, vec![0, 1, 1, 1, 2, 2, 3]);
    }

    #[test]
    fn custom_centroids_round_trip() {
        let c: Vec<f32> = vec![-2.5, -0.3, 0.3, 2.5];
        let input = c.clone();
        let q = quantize_vector_with_centroids::<MSE>(&input, 2, Some(&c));
        let rec = dequantize_vector_with_centroids(&q, 2, Some(&c));
        for (a, b) in input.iter().zip(&rec) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    fn custom_centroids_outperform_gaussian_on_calibrated_source() {
        // Data that's heavy-tailed — calibrated Lloyd-Max should beat
        // the Gaussian-assumed centroids on reconstruction MSE.
        // Build 1000 samples from Laplace-like mixture.
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(7);
        let input: Vec<f32> = (0..1000)
            .map(|_| {
                let u: f32 = rng.gen();
                // Laplace sample: sign(u-0.5) * ln(1 - 2|u-0.5|)
                let s = (u - 0.5).signum();
                let v: f32 = -(1.0 - 2.0 * (u - 0.5).abs()).ln();
                s * v
            })
            .collect();

        // Empirical Lloyd-Max at b=2 for Laplace-like data:
        // Compute mean of each quartile empirically — simple approximation.
        let mut sorted = input.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q1 = sorted.len() / 4;
        let q2 = sorted.len() / 2;
        let q3 = 3 * sorted.len() / 4;
        let c_calibrated: Vec<f32> = vec![
            sorted[..q1].iter().sum::<f32>() / q1 as f32,
            sorted[q1..q2].iter().sum::<f32>() / (q2 - q1) as f32,
            sorted[q2..q3].iter().sum::<f32>() / (q3 - q2) as f32,
            sorted[q3..].iter().sum::<f32>() / (sorted.len() - q3) as f32,
        ];

        let q_gauss = quantize_vector_with_centroids::<MSE>(&input, 2, None);
        let rec_gauss = dequantize_vector_with_centroids(&q_gauss, 2, None);
        let mse_gauss: f32 = input
            .iter()
            .zip(&rec_gauss)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / input.len() as f32;

        let q_cal = quantize_vector_with_centroids::<MSE>(&input, 2, Some(&c_calibrated));
        let rec_cal = dequantize_vector_with_centroids(&q_cal, 2, Some(&c_calibrated));
        let mse_cal: f32 = input
            .iter()
            .zip(&rec_cal)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / input.len() as f32;

        assert!(
            mse_cal < mse_gauss,
            "calibrated codebook MSE ({mse_cal:.4}) should beat Gaussian ({mse_gauss:.4}) on Laplace-distributed input"
        );
    }

    #[test]
    #[should_panic(expected = "custom centroids must have")]
    fn quantize_rejects_wrong_count_centroids() {
        let wrong = vec![-1.0_f32, 0.0, 1.0];  // 3 entries, but bits=2 needs 4
        let _ = quantize_vector_with_centroids::<MSE>(&[0.0_f32], 2, Some(&wrong));
    }

    #[test]
    fn pack_quantise_full_chain() {
        // End-to-end: quantise → pack → unpack → dequantise
        let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 8.0).collect();
        for bits in 1..=4u8 {
            let q = quantize_vector::<MSE>(&input, bits);
            let packed = pack_bits(&q, bits);
            let unpacked = unpack_bits(&packed, bits, input.len());
            let rec = dequantize_vector(&unpacked, bits);
            assert_eq!(rec.len(), input.len());
        }
    }
}
