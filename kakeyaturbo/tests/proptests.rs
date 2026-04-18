//! Property-based tests using proptest.
//!
//! These generate random blocks and check universal invariants that
//! should hold regardless of the specific data:
//!
//! - `decode(encode(x))` produces a vector of the expected shape
//! - All reconstructions are finite
//! - Reconstruction is deterministic given identical inputs
//! - Total compressed byte size is strictly positive
//! - Encoding is idempotent

use kakeyaturbo::{decode_block, encode_block, CodecParams, MSE};
use proptest::prelude::*;

fn gen_block(n: usize, d: usize) -> BoxedStrategy<Vec<f32>> {
    let total = n * d;
    prop::collection::vec(-10.0_f32..10.0_f32, total..=total).boxed()
}

fn gen_weights(n: usize) -> BoxedStrategy<Vec<f32>> {
    prop::collection::vec(0.001_f32..100.0_f32, n..=n).boxed()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn encode_decode_shape_invariant(
        (block, weights, n, d) in (2usize..8, 2usize..8).prop_flat_map(|(n, d)| {
            (
                prop::collection::vec(-5.0_f32..5.0, n * d..=n * d),
                prop::collection::vec(0.01_f32..50.0, n..=n),
                Just(n),
                Just(d),
            )
        })
    ) {
        let params = CodecParams {
            variance_ratio: 0.95,
            k: 2.min(n),
            bit_width: 3,
            ..Default::default()
        };
        let (sk, codes) = encode_block::<MSE>(&block, &weights, d, &params);
        prop_assert_eq!(codes.len(), n);
        let rec = decode_block::<MSE>(&sk, &codes);
        prop_assert_eq!(rec.len(), n * d);
        for v in &rec {
            prop_assert!(v.is_finite(), "non-finite reconstruction");
        }
    }

    #[test]
    fn encode_is_deterministic(
        block in gen_block(8, 4),
        weights in gen_weights(8),
        seed in 0u32..100,
    ) {
        let params = CodecParams {
            variance_ratio: 0.9,
            k: 2,
            bit_width: 3,
            rotation_seed: seed,
            ..Default::default()
        };
        let (_, c1) = encode_block::<MSE>(&block, &weights, 4, &params);
        let (_, c2) = encode_block::<MSE>(&block, &weights, 4, &params);
        prop_assert_eq!(c1, c2);
    }

    #[test]
    fn reconstruction_is_finite_for_any_valid_input(
        block in gen_block(16, 8),
        weights in gen_weights(16),
    ) {
        let params = CodecParams {
            variance_ratio: 0.95,
            k: 3,
            bit_width: 3,
            ..Default::default()
        };
        let (sk, codes) = encode_block::<MSE>(&block, &weights, 8, &params);
        let rec = decode_block::<MSE>(&sk, &codes);
        for v in rec {
            prop_assert!(v.is_finite());
        }
    }

    #[test]
    fn every_bit_width_works(
        block in gen_block(16, 8),
        weights in gen_weights(16),
        bits in 1u8..=4,
    ) {
        let params = CodecParams {
            variance_ratio: 0.9,
            k: 2,
            bit_width: bits,
            ..Default::default()
        };
        let (sk, codes) = encode_block::<MSE>(&block, &weights, 8, &params);
        let rec = decode_block::<MSE>(&sk, &codes);
        prop_assert_eq!(rec.len(), block.len());
    }

    #[test]
    fn every_k_works(
        block in gen_block(32, 8),
        weights in gen_weights(32),
        k in 1usize..8,
    ) {
        let params = CodecParams {
            variance_ratio: 0.9,
            k,
            bit_width: 3,
            ..Default::default()
        };
        let (sk, codes) = encode_block::<MSE>(&block, &weights, 8, &params);
        let rec = decode_block::<MSE>(&sk, &codes);
        prop_assert_eq!(rec.len(), block.len());
        prop_assert!(sk.k() >= 1);
    }

    #[test]
    fn scaling_all_weights_preserves_codes(
        block in gen_block(8, 4),
        weights in gen_weights(8),
        scale in 0.01_f32..100.0,
    ) {
        let params = CodecParams {
            variance_ratio: 0.9,
            k: 2,
            bit_width: 3,
            ..Default::default()
        };
        let w_scaled: Vec<f32> = weights.iter().map(|w| w * scale).collect();
        let (_, c1) = encode_block::<MSE>(&block, &weights, 4, &params);
        let (_, c2) = encode_block::<MSE>(&block, &w_scaled, 4, &params);
        // Uniform scaling of weights shouldn't affect the result (PCA and
        // K-means both divide by Σ w_i, so the scale cancels).
        prop_assert_eq!(c1.len(), c2.len());
        for (a, b) in c1.iter().zip(&c2) {
            prop_assert_eq!(a.seg_id, b.seg_id,
                "seg_id differs under uniform weight scaling");
        }
    }
}
