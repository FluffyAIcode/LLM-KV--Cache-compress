//! Walsh-Hadamard Transform with random sign flips.
//!
//! Implements the FJLT (Fast Johnson-Lindenstrauss Transform) construction:
//! `y = H · D · x` where `H` is the Walsh-Hadamard matrix and `D` is a
//! diagonal matrix with random ±1 entries derived deterministically from
//! a `u32` seed.
//!
//! - Forward: `y = H · D · x`
//! - Inverse: `x = Dᵀ · Hᵀ · y / N = D · H · y / N`
//!   (using `Hᵀ = H`, `Dᵀ = D`, and `H² = N · I`)
//!
//! The transform is used to "Gaussianise" the residual so the universal
//! Lloyd-Max codebook becomes near-optimal. The seed is stored (not the
//! matrix) because the ±1 signs are reproducible from the seed alone.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Generate the ±1 diagonal sign pattern for a given seed and length.
/// Returned as a `Vec<f32>` of `+1.0` / `-1.0`.
///
/// # Panics
///
/// Panics if `n` is not a power of two, since the Walsh-Hadamard
/// transform only admits sizes that are powers of two.
#[must_use]
pub fn sign_pattern(seed: u32, n: usize) -> Vec<f32> {
    assert!(
        n.is_power_of_two() && n > 0,
        "WHT size must be a power of two, got {n}"
    );
    let mut rng = SmallRng::seed_from_u64(u64::from(seed));
    (0..n)
        .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
        .collect()
}

/// In-place fast Walsh-Hadamard transform (natural order, unnormalised).
///
/// After calling this, `x[i]` holds the `i`-th Walsh-Hadamard coefficient.
/// Applying it twice yields `N * x`, i.e. `WHT(WHT(x)) = N · x`.
///
/// The classic butterfly: O(N log N) operations, cache-friendly.
pub fn wht_inplace(x: &mut [f32]) {
    let n = x.len();
    assert!(
        n.is_power_of_two() && n > 0,
        "WHT size must be a power of two, got {n}"
    );
    let mut h = 1usize;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
}

/// Apply the randomised Walsh-Hadamard rotation `y = H · D · x`.
#[must_use]
pub fn rotate(x: &[f32], seed: u32) -> Vec<f32> {
    let n = x.len();
    let signs = sign_pattern(seed, n);
    let mut buf: Vec<f32> = x.iter().zip(&signs).map(|(xi, si)| xi * si).collect();
    wht_inplace(&mut buf);
    buf
}

/// Apply the inverse rotation `x = D · H · y / N`.
///
/// Since `H` is self-inverse up to `N` and `D` is its own inverse (±1),
/// this is a single WHT followed by a sign flip and scale.
#[must_use]
pub fn inverse_rotate(y: &[f32], seed: u32) -> Vec<f32> {
    let n = y.len();
    let mut buf: Vec<f32> = y.to_vec();
    wht_inplace(&mut buf);
    let inv_n = 1.0_f32 / (n as f32);
    let signs = sign_pattern(seed, n);
    buf.iter()
        .zip(&signs)
        .map(|(bi, si)| bi * si * inv_n)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // -------------------- Basic properties --------------------

    #[test]
    fn wht_size_1_is_identity() {
        let mut x = vec![3.5];
        wht_inplace(&mut x);
        assert_abs_diff_eq!(x[0], 3.5);
    }

    #[test]
    fn wht_size_2_matches_definition() {
        // H_2 = [[1, 1], [1, -1]]
        let mut x = vec![1.0, 2.0];
        wht_inplace(&mut x);
        assert_abs_diff_eq!(x[0], 3.0);
        assert_abs_diff_eq!(x[1], -1.0);
    }

    #[test]
    fn wht_size_4_matches_definition() {
        // H_4 = [ 1  1  1  1 ]
        //       [ 1 -1  1 -1 ]
        //       [ 1  1 -1 -1 ]
        //       [ 1 -1 -1  1 ]
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        wht_inplace(&mut x);
        assert_abs_diff_eq!(x[0], 10.0); // sum
        assert_abs_diff_eq!(x[1], -2.0); // 1 - 2 + 3 - 4
        assert_abs_diff_eq!(x[2], -4.0); // 1 + 2 - 3 - 4
        assert_abs_diff_eq!(x[3], 0.0); //  1 - 2 - 3 + 4
    }

    #[test]
    fn wht_applied_twice_is_n_times_identity() {
        for log_n in 0..8 {
            let n = 1 << log_n;
            let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.7 + 1.0).collect();
            let mut y = x.clone();
            wht_inplace(&mut y);
            wht_inplace(&mut y);
            // f32 butterfly accumulates O(log n) relative error; bound accordingly.
            let eps = 1e-4_f32 * (n as f32);
            for (i, &xi) in x.iter().enumerate() {
                assert_abs_diff_eq!(y[i], (n as f32) * xi, epsilon = eps);
            }
        }
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn wht_rejects_non_power_of_two() {
        let mut x = vec![1.0_f32, 2.0, 3.0];
        wht_inplace(&mut x);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn wht_rejects_zero() {
        let mut x: Vec<f32> = vec![];
        wht_inplace(&mut x);
    }

    // -------------------- Sign pattern --------------------

    #[test]
    fn sign_pattern_is_deterministic() {
        let a = sign_pattern(42, 16);
        let b = sign_pattern(42, 16);
        assert_eq!(a, b);
    }

    #[test]
    fn sign_pattern_values_are_plus_minus_one() {
        let signs = sign_pattern(123, 64);
        for s in signs {
            assert!(s == 1.0 || s == -1.0, "bad sign {s}");
        }
    }

    #[test]
    fn sign_pattern_different_seeds_give_different_patterns() {
        let a = sign_pattern(0xCAFE, 128);
        let b = sign_pattern(0xBEEF, 128);
        assert_ne!(a, b);
    }

    #[test]
    fn sign_pattern_length_matches() {
        for &n in &[1usize, 2, 4, 16, 256] {
            assert_eq!(sign_pattern(0, n).len(), n);
        }
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn sign_pattern_rejects_non_power_of_two() {
        let _ = sign_pattern(0, 3);
    }

    // -------------------- Rotation + inverse --------------------

    #[test]
    fn rotate_then_inverse_recovers_input() {
        for &n in &[1usize, 2, 4, 8, 16, 32, 64] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32).sin() * 2.0 - 0.3).collect();
            let y = rotate(&x, 0xDEAD_BEEF);
            let recovered = inverse_rotate(&y, 0xDEAD_BEEF);
            for (a, b) in x.iter().zip(&recovered) {
                assert_abs_diff_eq!(a, b, epsilon = 1e-3);
            }
        }
    }

    #[test]
    fn rotate_preserves_l2_norm_up_to_sqrt_n() {
        // ||H · D · x||² = N · ||x||²
        for &n in &[4usize, 16, 64] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.1).collect();
            let y = rotate(&x, 7);
            let x_sq: f32 = x.iter().map(|v| v * v).sum();
            let y_sq: f32 = y.iter().map(|v| v * v).sum();
            assert_abs_diff_eq!(y_sq, (n as f32) * x_sq, epsilon = 1e-2);
        }
    }

    #[test]
    fn rotate_with_different_seeds_gives_different_outputs() {
        let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let a = rotate(&x, 1);
        let b = rotate(&x, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn inverse_rotate_requires_matching_seed() {
        // Decrypting with the wrong seed must not recover the input.
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = rotate(&x, 1);
        let recovered_wrong = inverse_rotate(&y, 2);
        let mut diff = 0.0_f32;
        for (a, b) in x.iter().zip(&recovered_wrong) {
            diff += (a - b).abs();
        }
        assert!(diff > 0.1, "wrong seed accidentally recovered input");
    }

    #[test]
    fn rotate_of_zero_is_zero() {
        let z = vec![0.0_f32; 16];
        let y = rotate(&z, 42);
        for v in y {
            assert_abs_diff_eq!(v, 0.0);
        }
    }

    #[test]
    fn rotate_is_linear() {
        // rotate(αx + βy) = α rotate(x) + β rotate(y)
        let n = 8;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i * 2) as f32 + 1.0).collect();
        let alpha = 0.3_f32;
        let beta = 0.7_f32;
        let lhs_input: Vec<f32> = x
            .iter()
            .zip(&y)
            .map(|(xi, yi)| alpha * xi + beta * yi)
            .collect();
        let lhs = rotate(&lhs_input, 11);
        let rx = rotate(&x, 11);
        let ry = rotate(&y, 11);
        for i in 0..n {
            let rhs = alpha * rx[i] + beta * ry[i];
            assert_abs_diff_eq!(lhs[i], rhs, epsilon = 1e-3);
        }
    }
}
