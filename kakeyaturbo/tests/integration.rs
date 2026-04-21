//! Integration tests: realistic end-to-end encode → decode flows.

use kakeyaturbo::{decode_block, encode_block, CodecParams, InnerProduct, MSE};

fn synthetic(n: usize, d: usize, rank: usize, noise: f32, seed: u64) -> Vec<f32> {
    use rand::rngs::SmallRng;
    use rand::Rng;
    use rand::SeedableRng;

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut latents = vec![0.0_f32; n * rank];
    for v in latents.iter_mut() {
        *v = rng.gen_range(-1.0..1.0_f32);
    }
    let mut out = vec![0.0_f32; n * d];
    for i in 0..n {
        for r in 0..rank {
            out[i * d + r] += latents[i * rank + r];
        }
        for j in 0..d {
            out[i * d + j] += rng.gen_range(-noise..noise);
        }
    }
    out
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sq: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    sq / a.len() as f32
}

#[test]
fn end_to_end_mse_block_64x32() {
    let n = 64;
    let d = 32;
    let block = synthetic(n, d, 8, 0.02, 101);
    let w = vec![1.0_f32; n];
    let params = CodecParams {
        variance_ratio: 0.95,
        k: 8,
        bit_width: 4,
        rotation_seed: 0xFEED,
        kmeans_max_iter: 64,
        pca_method: kakeyaturbo::PcaMethod::Exact,
    };
    let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
    let recovered = decode_block::<MSE>(&sk, &codes);
    let err = mse(&block, &recovered);
    assert!(err < 0.05, "end-to-end MSE too high: {err}");
}

#[test]
fn compression_ratio_is_bounded() {
    let n = 64;
    let d = 32;
    let block = synthetic(n, d, 6, 0.01, 202);
    let w = vec![1.0_f32; n];
    let params = CodecParams {
        variance_ratio: 0.95,
        k: 4,
        bit_width: 3,
        ..Default::default()
    };
    let raw = n * d * std::mem::size_of::<f32>();
    let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
    let compressed = sk.nbytes() + codes.iter().map(|c| c.nbytes()).sum::<usize>();
    // Under a 3-bit residual we expect meaningful compression when d is not tiny
    // relative to skeleton overhead. For n=64, d=32 this should clearly win.
    assert!(compressed < raw, "no compression: raw={raw} comp={compressed}");
}

#[test]
fn weights_influence_reconstruction() {
    // One row has weight 1000×; reconstruction for that row must be better.
    let n = 16;
    let d = 16;
    let block = synthetic(n, d, 4, 0.05, 303);
    let mut w = vec![1.0_f32; n];
    w[5] = 1000.0;
    let params = CodecParams {
        variance_ratio: 0.95,
        k: 4,
        bit_width: 3,
        ..Default::default()
    };
    let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
    let rec = decode_block::<MSE>(&sk, &codes);

    let err_heavy: f32 =
        (0..d).map(|j| (block[5 * d + j] - rec[5 * d + j]).powi(2)).sum();

    // Compare against the average non-heavy row.
    let mut err_other_sum = 0.0_f32;
    let mut other_count = 0;
    for i in 0..n {
        if i == 5 {
            continue;
        }
        let e: f32 = (0..d)
            .map(|j| (block[i * d + j] - rec[i * d + j]).powi(2))
            .sum();
        err_other_sum += e;
        other_count += 1;
    }
    let err_other_avg = err_other_sum / other_count as f32;

    // The heavy row must be reconstructed at least as well as average.
    assert!(
        err_heavy <= err_other_avg * 1.1,
        "heavy row not prioritised: heavy={err_heavy} avg_other={err_other_avg}"
    );
}

#[test]
fn inner_product_preservation_is_bounded() {
    let n = 32;
    let d = 16;
    let block = synthetic(n, d, 4, 0.02, 404);
    let w = vec![1.0_f32; n];
    let params = CodecParams {
        variance_ratio: 0.99,
        k: 4,
        bit_width: 4,
        ..Default::default()
    };
    let (sk, codes) = encode_block::<InnerProduct>(&block, &w, d, &params);
    let rec = decode_block::<InnerProduct>(&sk, &codes);

    // A random query vector.
    let query: Vec<f32> = (0..d).map(|j| (j as f32 * 0.37).sin()).collect();
    let mut max_abs_err = 0.0_f32;
    for i in 0..n {
        let x = &block[i * d..(i + 1) * d];
        let x_hat = &rec[i * d..(i + 1) * d];
        let ip = x.iter().zip(&query).map(|(a, b)| a * b).sum::<f32>();
        let ip_hat = x_hat.iter().zip(&query).map(|(a, b)| a * b).sum::<f32>();
        max_abs_err = max_abs_err.max((ip - ip_hat).abs());
    }
    // Bounded absolute error — exact value depends on PCA + quantisation.
    assert!(
        max_abs_err < 2.0,
        "inner product error too large: {max_abs_err}"
    );
}

#[test]
fn round_trip_handles_many_block_shapes() {
    for &(n, d, rank) in &[(8, 4, 2), (16, 8, 4), (32, 16, 8), (64, 32, 16), (128, 64, 8)] {
        let block = synthetic(n, d, rank, 0.01, 505 + n as u64);
        let w = vec![1.0_f32; n];
        let params = CodecParams {
            variance_ratio: 0.95,
            k: 4.min(n),
            bit_width: 3,
            ..Default::default()
        };
        let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
        assert_eq!(codes.len(), n);
        let rec = decode_block::<MSE>(&sk, &codes);
        assert_eq!(rec.len(), n * d);
        for v in rec {
            assert!(v.is_finite(), "non-finite at shape ({n}, {d})");
        }
    }
}
