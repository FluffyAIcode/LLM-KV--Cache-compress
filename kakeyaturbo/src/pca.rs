//! Weighted PCA truncated at an effective rank `d_eff`.
//!
//! Solves the weighted low-rank approximation problem:
//!
//! ```text
//! min_{μ, U}   sum_i w_i · || x_i − μ − U Uᵀ (x_i − μ) ||²
//! subject to   U ∈ ℝ^(D × d_eff),  UᵀU = I
//! ```
//!
//! which is the Stage-1 "structure extraction" step of KakeyaTurbo.
//! The solution is:
//!
//! 1. `μ = Σ w_i x_i / Σ w_i` (weighted mean)
//! 2. Form the weighted covariance `Σ_w = X_c diag(w) X_cᵀ / Σ w`
//! 3. `U` = top-`d_eff` eigenvectors of `Σ_w`
//!
//! `d_eff` is chosen to cover `variance_ratio` of the total weighted
//! variance, clipped to `[1, D]`.

use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// Compute the weighted mean of a set of vectors.
///
/// `vectors` is row-major: shape `[N, D]` as a flat `&[f32]` of length `N*D`.
/// Returns a length-`D` vector.
///
/// # Panics
///
/// Panics if `weights.len() != N`, where `N = vectors.len() / D`,
/// or if weights sum to zero.
#[must_use]
pub fn weighted_mean(vectors: &[f32], weights: &[f32], d: usize) -> Vec<f32> {
    assert!(d > 0, "dimension must be positive");
    assert_eq!(vectors.len() % d, 0, "vector buffer not multiple of D");
    let n = vectors.len() / d;
    assert_eq!(weights.len(), n, "weights length != number of vectors");
    let w_sum: f32 = weights.iter().sum();
    assert!(
        w_sum > f32::EPSILON,
        "weights sum must be positive (got {w_sum})"
    );

    let mut mean = vec![0.0_f32; d];
    for (i, &w) in weights.iter().enumerate() {
        for j in 0..d {
            mean[j] += w * vectors[i * d + j];
        }
    }
    for m in &mut mean {
        *m /= w_sum;
    }
    mean
}

/// Result of a weighted PCA fit.
#[derive(Debug, Clone)]
pub struct PcaFit {
    /// Mean vector, length `D`.
    pub mean: Vec<f32>,
    /// Basis, stored as row-major `[d_eff, D]` (each row is a principal direction).
    pub basis: Vec<f32>,
    /// Number of kept components.
    pub d_eff: usize,
    /// Captured variance ratio (actual, may be ≥ the requested threshold).
    pub captured_variance: f32,
}

/// Weighted PCA truncated by explained-variance ratio.
///
/// Returns a [`PcaFit`] with `d_eff` satisfying
/// `cumulative_variance_ratio[d_eff - 1] ≥ variance_ratio`, clipped
/// to `[1, D]`. If `variance_ratio = 0.0` then `d_eff = 1`;
/// if `variance_ratio ≥ 1.0` then `d_eff = D`.
///
/// # Panics
///
/// Panics on the same conditions as [`weighted_mean`], or if
/// `variance_ratio` is not finite.
#[must_use]
pub fn fit_weighted_pca(vectors: &[f32], weights: &[f32], d: usize, variance_ratio: f32) -> PcaFit {
    assert!(
        variance_ratio.is_finite(),
        "variance_ratio must be finite, got {variance_ratio}"
    );
    assert!(d > 0, "D must be positive");

    let mean = weighted_mean(vectors, weights, d);
    let n = weights.len();
    let w_sum: f32 = weights.iter().sum();

    // Build centred, weighted design matrix Y with rows sqrt(w_i) * (x_i - μ).
    // Then Σ_w = Yᵀ Y / w_sum. We work via SymmetricEigen on the D×D matrix.
    let mut sigma = DMatrix::<f32>::zeros(d, d);
    for i in 0..n {
        let w_i = weights[i];
        if w_i <= 0.0 {
            continue;
        }
        let mut centred = vec![0.0_f32; d];
        for j in 0..d {
            centred[j] = vectors[i * d + j] - mean[j];
        }
        // Accumulate w_i · (x - μ)(x - μ)ᵀ.
        for a in 0..d {
            for b in 0..=a {
                let v = w_i * centred[a] * centred[b];
                sigma[(a, b)] += v;
                if a != b {
                    sigma[(b, a)] += v;
                }
            }
        }
    }
    sigma /= w_sum;

    let eig = SymmetricEigen::new(sigma);
    // nalgebra returns unsorted eigenvalues; sort descending by magnitude.
    let mut pairs: Vec<(f32, DVector<f32>)> = eig
        .eigenvalues
        .iter()
        .copied()
        .zip(eig.eigenvectors.column_iter().map(DVector::from))
        .collect();
    pairs.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let total_var: f32 = pairs.iter().map(|(v, _)| v.max(0.0)).sum();
    let ratio = variance_ratio.clamp(0.0, 1.0);
    let mut cum = 0.0_f32;
    let mut d_eff = d;
    if total_var > f32::EPSILON {
        for (i, (v, _)) in pairs.iter().enumerate() {
            cum += v.max(0.0);
            if cum / total_var >= ratio {
                d_eff = i + 1;
                break;
            }
        }
    } else {
        d_eff = 1;
    }
    d_eff = d_eff.clamp(1, d);

    // Flatten top-d_eff eigenvectors into row-major basis.
    let mut basis = Vec::with_capacity(d_eff * d);
    let mut captured = 0.0_f32;
    for (i, (v, vec_)) in pairs.iter().take(d_eff).enumerate() {
        let _ = i;
        captured += v.max(0.0);
        for j in 0..d {
            basis.push(vec_[j]);
        }
    }
    let captured_variance = if total_var > f32::EPSILON {
        captured / total_var
    } else {
        1.0
    };

    PcaFit {
        mean,
        basis,
        d_eff,
        captured_variance,
    }
}

/// Project a single vector `x` onto the PCA basis: `coeff = U · (x − μ)`.
#[must_use]
pub fn project(x: &[f32], fit: &PcaFit) -> Vec<f32> {
    let d = fit.mean.len();
    assert_eq!(x.len(), d, "x dimension mismatch");
    let mut coeff = vec![0.0_f32; fit.d_eff];
    for k in 0..fit.d_eff {
        let mut acc = 0.0_f32;
        for j in 0..d {
            acc += fit.basis[k * d + j] * (x[j] - fit.mean[j]);
        }
        coeff[k] = acc;
    }
    coeff
}

/// Reverse of [`project`]: reconstruct `x` from its PCA coefficients:
/// `x ≈ μ + Uᵀ · coeff`.
#[must_use]
pub fn unproject(coeff: &[f32], fit: &PcaFit) -> Vec<f32> {
    assert_eq!(coeff.len(), fit.d_eff, "coeff length mismatch");
    let d = fit.mean.len();
    let mut x = fit.mean.clone();
    for k in 0..fit.d_eff {
        let c = coeff[k];
        for j in 0..d {
            x[j] += fit.basis[k * d + j] * c;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // -------------------- weighted_mean --------------------

    #[test]
    fn weighted_mean_uniform_weights() {
        let vecs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = [1.0, 1.0];
        let m = weighted_mean(&vecs, &w, 3);
        assert_abs_diff_eq!(m[0], 2.5);
        assert_abs_diff_eq!(m[1], 3.5);
        assert_abs_diff_eq!(m[2], 4.5);
    }

    #[test]
    fn weighted_mean_with_zero_weight_skips_row() {
        let vecs = [10.0, 10.0, 0.0, 0.0];
        let w = [0.0, 1.0];
        let m = weighted_mean(&vecs, &w, 2);
        assert_abs_diff_eq!(m[0], 0.0);
        assert_abs_diff_eq!(m[1], 0.0);
    }

    #[test]
    fn weighted_mean_nonuniform_weights() {
        let vecs = [0.0, 10.0, 100.0];
        let w = [1.0, 9.0, 0.0];
        let m = weighted_mean(&vecs, &w, 1);
        // (1·0 + 9·10 + 0·100) / 10 = 9
        assert_abs_diff_eq!(m[0], 9.0);
    }

    #[test]
    #[should_panic(expected = "weights sum must be positive")]
    fn weighted_mean_rejects_zero_weights() {
        // 2 vectors of dim 2, both with weight 0 → sum = 0 → panic
        let _ = weighted_mean(&[1.0_f32, 2.0, 3.0, 4.0], &[0.0, 0.0], 2);
    }

    #[test]
    #[should_panic(expected = "weights length")]
    fn weighted_mean_rejects_mismatched_weights() {
        let _ = weighted_mean(&[1.0_f32, 2.0, 3.0, 4.0], &[1.0], 2);
    }

    #[test]
    #[should_panic(expected = "dimension must be positive")]
    fn weighted_mean_rejects_zero_dim() {
        let _ = weighted_mean(&[], &[], 0);
    }

    // -------------------- fit_weighted_pca --------------------

    /// Generate N 2D points on a rotated ellipse (known principal axes).
    fn ellipse_points(n: usize, sigma_major: f32, sigma_minor: f32, theta: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(n * 2);
        let c = theta.cos();
        let s = theta.sin();
        for i in 0..n {
            let u = (i as f32 / n as f32) * std::f32::consts::TAU;
            let a = sigma_major * u.cos();
            let b = sigma_minor * u.sin();
            // Rotate and centre offset 0.
            out.push(a * c - b * s);
            out.push(a * s + b * c);
        }
        out
    }

    #[test]
    fn pca_recovers_major_axis_of_2d_ellipse() {
        let n = 128;
        let vecs = ellipse_points(n, 4.0, 0.5, 0.3);
        let w = vec![1.0_f32; n];
        let fit = fit_weighted_pca(&vecs, &w, 2, 0.99);
        assert_eq!(fit.d_eff, 2);
        // First basis row must point along direction (cos 0.3, sin 0.3).
        let v0 = &fit.basis[0..2];
        let expected = [0.3_f32.cos(), 0.3_f32.sin()];
        // Sign can flip; test |dot product| ≈ 1.
        let dot = (v0[0] * expected[0] + v0[1] * expected[1]).abs();
        assert!(dot > 0.98, "dot = {dot} (v0={v0:?}, expected={expected:?})");
    }

    #[test]
    fn pca_truncates_at_variance_ratio() {
        let n = 64;
        // Very skinny ellipse → first component should dominate.
        let vecs = ellipse_points(n, 10.0, 0.01, 0.0);
        let w = vec![1.0_f32; n];
        let fit = fit_weighted_pca(&vecs, &w, 2, 0.5);
        assert_eq!(fit.d_eff, 1, "should truncate to 1 component");
        assert!(fit.captured_variance > 0.99);
    }

    #[test]
    fn pca_round_trip_with_full_rank_is_identity() {
        let n = 16;
        let vecs = ellipse_points(n, 3.0, 1.2, 0.6);
        let w = vec![1.0_f32; n];
        let fit = fit_weighted_pca(&vecs, &w, 2, 1.0);
        assert_eq!(fit.d_eff, 2);
        for i in 0..n {
            let x = &vecs[i * 2..i * 2 + 2];
            let c = project(x, &fit);
            let r = unproject(&c, &fit);
            assert_abs_diff_eq!(x[0], r[0], epsilon = 1e-4);
            assert_abs_diff_eq!(x[1], r[1], epsilon = 1e-4);
        }
    }

    #[test]
    fn pca_handles_constant_data() {
        // All vectors identical → zero variance in any direction.
        let n = 8;
        let vecs: Vec<f32> = (0..n).flat_map(|_| [5.0_f32, -3.0]).collect();
        let w = vec![1.0_f32; n];
        let fit = fit_weighted_pca(&vecs, &w, 2, 0.95);
        assert!(fit.d_eff >= 1);
        // All points project to the same reconstruction.
        let c = project(&vecs[0..2], &fit);
        let r = unproject(&c, &fit);
        assert_abs_diff_eq!(r[0], 5.0, epsilon = 1e-4);
        assert_abs_diff_eq!(r[1], -3.0, epsilon = 1e-4);
    }

    #[test]
    fn pca_captured_variance_is_monotone_in_threshold() {
        let n = 32;
        let vecs = ellipse_points(n, 3.0, 1.0, 0.7);
        let w = vec![1.0_f32; n];
        let mut prev = 0.0_f32;
        for &r in &[0.3_f32, 0.6, 0.9, 0.99] {
            let fit = fit_weighted_pca(&vecs, &w, 2, r);
            assert!(
                fit.captured_variance + 1e-5 >= prev,
                "captured variance not monotone: prev={prev} new={}",
                fit.captured_variance
            );
            prev = fit.captured_variance;
        }
    }

    #[test]
    fn pca_skips_zero_weight_rows_in_covariance() {
        // Mix a heavy cluster with zero-weight "decoys" that have wildly
        // different statistics. The zero-weight rows must not influence
        // the PCA at all.
        let n = 8;
        let d = 2;
        let mut vecs = Vec::with_capacity(n * d);
        let mut w = Vec::with_capacity(n);
        // Heavy rows: variance along x only.
        for i in 0..4 {
            vecs.push(i as f32 - 1.5);
            vecs.push(0.0);
            w.push(1.0);
        }
        // Decoy rows with zero weight: variance purely along y
        // (would make basis tilt toward y if counted).
        for i in 0..4 {
            vecs.push(0.0);
            vecs.push((i as f32 - 1.5) * 100.0);
            w.push(0.0);
        }
        let fit = fit_weighted_pca(&vecs, &w, d, 0.95);
        let v0 = &fit.basis[0..2];
        assert!(v0[0].abs() > v0[1].abs(), "decoys leaked into basis: {v0:?}");
    }

    #[test]
    fn pca_weighted_emphasises_heavy_points() {
        // Give one point in the x-direction 100× weight compared to
        // a cloud in the y-direction; PCA should align basis to x.
        let mut vecs = vec![10.0_f32, 0.0];
        let mut w = vec![100.0_f32];
        for i in 0..16 {
            let t = (i as f32 / 16.0) * std::f32::consts::TAU;
            vecs.push(0.0);
            vecs.push(t.sin() * 0.1);
            w.push(1.0);
        }
        let fit = fit_weighted_pca(&vecs, &w, 2, 0.9);
        let v0 = &fit.basis[0..2];
        assert!(v0[0].abs() > v0[1].abs(), "basis not x-dominated: {v0:?}");
    }

    #[test]
    #[should_panic(expected = "variance_ratio must be finite")]
    fn pca_rejects_nan_ratio() {
        let _ = fit_weighted_pca(&[0.0_f32, 1.0], &[1.0], 2, f32::NAN);
    }

    #[test]
    fn pca_clips_ratio_above_one() {
        let fit = fit_weighted_pca(
            &[1.0_f32, 0.0, 0.0, 1.0, 0.7, 0.7],
            &[1.0, 1.0, 1.0],
            2,
            5.0,
        );
        assert_eq!(fit.d_eff, 2);
    }

    #[test]
    fn pca_clips_ratio_below_zero() {
        let fit = fit_weighted_pca(
            &[1.0_f32, 0.0, 0.0, 1.0, 0.7, 0.7],
            &[1.0, 1.0, 1.0],
            2,
            -0.5,
        );
        assert!(fit.d_eff >= 1);
    }

    // -------------------- project / unproject --------------------

    #[test]
    #[should_panic(expected = "x dimension mismatch")]
    fn project_rejects_wrong_dim() {
        let fit = PcaFit {
            mean: vec![0.0; 3],
            basis: vec![1.0, 0.0, 0.0],
            d_eff: 1,
            captured_variance: 1.0,
        };
        let _ = project(&[1.0_f32, 2.0], &fit);
    }

    #[test]
    #[should_panic(expected = "coeff length mismatch")]
    fn unproject_rejects_wrong_dim() {
        let fit = PcaFit {
            mean: vec![0.0; 3],
            basis: vec![1.0, 0.0, 0.0],
            d_eff: 1,
            captured_variance: 1.0,
        };
        let _ = unproject(&[1.0_f32, 2.0], &fit);
    }
}
