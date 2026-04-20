//! Weighted spherical K-means on PCA-projected coefficients.
//!
//! After PCA projection, each vector is represented by a `d_eff`-dim
//! coefficient. KakeyaTurbo clusters these coefficients by *direction*
//! (i.e. L2-normalised) to capture angular structure, then stores
//!
//! ```text
//! coeff_i ≈ t_i · center_{seg_i} + residual_i
//! ```
//!
//! This module provides the fit (centres + assignments) and the
//! per-row (seg_id, t) decomposition.

use half::f16;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Result of a spherical K-means fit.
///
/// **Storage contract**: `centers` are stored in **bf16** to halve the
/// per-block K-means footprint (see A optimisation in v1.2). Centres
/// are fit in f32 (iterative); the final result is converted once.
#[derive(Debug, Clone)]
pub struct KmeansFit {
    /// Unit-norm centres row-major `[K, d_eff]`, stored as f16 (empty if fp32 skeleton selected).
    pub centers: Vec<f16>,
    /// Optional fp32 centres buffer — populated iff the caller asked for
    /// `PcaStorage::Fp32` (via the codec's skeleton dtype flag).
    pub centers_fp32: Option<Vec<f32>>,
    /// Number of centres.
    pub k: usize,
    /// Coefficient dimension.
    pub d_eff: usize,
}

impl KmeansFit {
    /// Get a freshly-allocated f32 copy of the `i`-th centre.
    #[must_use]
    pub fn center(&self, i: usize) -> Vec<f32> {
        if let Some(ref c) = self.centers_fp32 {
            return c[i * self.d_eff..(i + 1) * self.d_eff].to_vec();
        }
        self.centers[i * self.d_eff..(i + 1) * self.d_eff]
            .iter()
            .map(|&v| v.to_f32())
            .collect()
    }

    /// Byte footprint of this fit (what the codec stores).
    #[must_use]
    pub fn nbytes(&self) -> usize {
        self.centers.len() * std::mem::size_of::<f16>()
            + self.centers_fp32.as_ref().map(Vec::len).unwrap_or(0)
                * std::mem::size_of::<f32>()
    }

    /// Construct directly from f32 centres (e.g. for tests). Defaults to
    /// f16 skeleton storage for backward compatibility.
    #[must_use]
    pub fn from_f32(centers: Vec<f32>, k: usize, d_eff: usize) -> Self {
        Self {
            centers: centers.iter().map(|&v| f16::from_f32(v)).collect(),
            centers_fp32: None,
            k,
            d_eff,
        }
    }

    /// Construct with fp32 skeleton storage.
    #[must_use]
    pub fn from_f32_skeleton_fp32(centers: Vec<f32>, k: usize, d_eff: usize) -> Self {
        Self {
            centers: Vec::new(),
            centers_fp32: Some(centers),
            k,
            d_eff,
        }
    }
}

/// L2-normalise a vector in place. Returns the original norm.
fn normalise(v: &mut [f32]) -> f32 {
    let sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = sq.sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

/// Compute the dot product of two slices of equal length.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Farthest-first initialisation on the unit sphere (deterministic given a seed).
fn init_farthest_first(dirs: &[f32], d_eff: usize, k: usize, seed: u32) -> Vec<f32> {
    let n = dirs.len() / d_eff;
    assert!(n > 0, "need at least one direction");
    let mut rng = SmallRng::seed_from_u64(u64::from(seed));
    let mut centers = Vec::with_capacity(k * d_eff);
    let first = rng.gen_range(0..n);
    centers.extend_from_slice(&dirs[first * d_eff..(first + 1) * d_eff]);
    for _ in 1..k {
        let mut best_far = 0usize;
        let mut best_dist = f32::NEG_INFINITY;
        for i in 0..n {
            let row = &dirs[i * d_eff..(i + 1) * d_eff];
            // Distance on the sphere ~ 1 - max cos similarity.
            let mut max_cos = f32::NEG_INFINITY;
            let c_count = centers.len() / d_eff;
            for c in 0..c_count {
                let center = &centers[c * d_eff..(c + 1) * d_eff];
                let cos = dot(row, center);
                if cos > max_cos {
                    max_cos = cos;
                }
            }
            let dist = 1.0 - max_cos;
            if dist > best_dist {
                best_dist = dist;
                best_far = i;
            }
        }
        centers.extend_from_slice(&dirs[best_far * d_eff..(best_far + 1) * d_eff]);
    }
    centers
}

/// Fit weighted spherical K-means on the direction vectors of `coeffs`.
///
/// Input `coeffs` is row-major `[n, d_eff]`. Zero-norm rows are dropped
/// before fitting (they don't participate in clustering).
///
/// # Panics
///
/// Panics if `k == 0`, `d_eff == 0`, or `coeffs.len() % d_eff != 0`,
/// or if the number of valid (non-zero-norm, positive-weight) rows
/// is less than `k`.
#[must_use]
pub fn fit_spherical_kmeans(
    coeffs: &[f32],
    weights: &[f32],
    d_eff: usize,
    k: usize,
    seed: u32,
    max_iter: u32,
) -> KmeansFit {
    fit_spherical_kmeans_with_storage(coeffs, weights, d_eff, k, seed, max_iter, false)
}

/// Storage-aware variant: when `fp32_skeleton` is true, keep the fitted
/// centres in full fp32 precision instead of rounding to f16.
#[must_use]
pub fn fit_spherical_kmeans_with_storage(
    coeffs: &[f32],
    weights: &[f32],
    d_eff: usize,
    k: usize,
    seed: u32,
    max_iter: u32,
    fp32_skeleton: bool,
) -> KmeansFit {
    assert!(k > 0, "k must be positive");
    assert!(d_eff > 0, "d_eff must be positive");
    assert_eq!(coeffs.len() % d_eff, 0, "coeffs length not multiple of d_eff");
    let n = coeffs.len() / d_eff;
    assert_eq!(weights.len(), n, "weights length mismatch");

    // Build the set of valid unit-direction rows with their weights.
    let mut dirs = Vec::with_capacity(n * d_eff);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        if weights[i] <= 0.0 {
            continue;
        }
        let mut row = coeffs[i * d_eff..(i + 1) * d_eff].to_vec();
        let norm = normalise(&mut row);
        if norm > f32::EPSILON {
            dirs.extend_from_slice(&row);
            w.push(weights[i]);
        }
    }
    let valid_n = w.len();
    assert!(
        valid_n >= k,
        "need at least {k} non-zero-norm positive-weight rows, got {valid_n}"
    );

    let mut centers = init_farthest_first(&dirs, d_eff, k, seed);

    // Lloyd iterations.
    let mut assignments = vec![0usize; valid_n];
    for _ in 0..max_iter {
        let mut changed = false;

        // Assignment step using |<row, center>| (matches assign_and_project).
        for i in 0..valid_n {
            let row = &dirs[i * d_eff..(i + 1) * d_eff];
            let mut best = 0usize;
            let mut best_abs = f32::NEG_INFINITY;
            for c in 0..k {
                let center = &centers[c * d_eff..(c + 1) * d_eff];
                let abs_cos = dot(row, center).abs();
                if abs_cos > best_abs {
                    best_abs = abs_cos;
                    best = c;
                }
            }
            if assignments[i] != best {
                changed = true;
                assignments[i] = best;
            }
        }

        if !changed {
            break;
        }

        // Update step: weighted mean of assigned directions, re-normalised.
        // Because assignment uses |<row, center>|, rows contribute with a
        // sign so that aligned and anti-aligned rows collaborate rather
        // than cancel.
        let mut new_centers = vec![0.0_f32; k * d_eff];
        let mut cluster_w = vec![0.0_f32; k];
        for i in 0..valid_n {
            let c = assignments[i];
            cluster_w[c] += w[i];
            let row = &dirs[i * d_eff..(i + 1) * d_eff];
            let sign = dot(row, &centers[c * d_eff..(c + 1) * d_eff]).signum();
            let sign = if sign == 0.0 { 1.0 } else { sign };
            for j in 0..d_eff {
                new_centers[c * d_eff + j] += w[i] * sign * row[j];
            }
        }
        for c in 0..k {
            if cluster_w[c] > f32::EPSILON {
                let slice = &mut new_centers[c * d_eff..(c + 1) * d_eff];
                normalise(slice);
            } else {
                // Preserve previous centre if empty cluster.
                let src = &centers[c * d_eff..(c + 1) * d_eff];
                new_centers[c * d_eff..(c + 1) * d_eff].copy_from_slice(src);
            }
        }
        centers = new_centers;
    }

    if fp32_skeleton {
        KmeansFit::from_f32_skeleton_fp32(centers, k, d_eff)
    } else {
        KmeansFit::from_f32(centers, k, d_eff)
    }
}

/// Assign a coefficient row to the centre that minimises the residual
/// norm after projection. Returns `(seg_id, t)` where `t = <coeff, center>`.
///
/// Since `residual = coeff - t · center` and centres are unit-norm,
/// `||residual||² = ||coeff||² - t²`. Minimising this is equivalent
/// to maximising `|t|`, i.e. `argmax_c |<coeff, center_c>|`.
///
/// This is the algorithmically correct criterion for residual coding;
/// naive cosine maximisation (`argmax <coeff, center>`) would fail on
/// anti-aligned inputs (e.g. `[0, -2]` vs centres `[1, 0]`, `[0, 1]`),
/// because `t` absorbs the sign and the two-sided projection is tighter.
#[must_use]
pub fn assign_and_project(coeff: &[f32], fit: &KmeansFit) -> (u32, f32) {
    assert_eq!(coeff.len(), fit.d_eff, "coeff dim mismatch");
    let coeff_norm_sq: f32 = coeff.iter().map(|v| v * v).sum();
    if coeff_norm_sq <= f32::EPSILON {
        return (0, 0.0);
    }
    let mut best = 0u32;
    let mut best_abs_proj = f32::NEG_INFINITY;
    let mut best_t = 0.0_f32;
    for c in 0..fit.k {
        let center = fit.center(c);
        let t = dot(coeff, &center);
        let abs_t = t.abs();
        if abs_t > best_abs_proj {
            best_abs_proj = abs_t;
            best = c as u32;
            best_t = t;
        }
    }
    (best, best_t)
}

/// Compute the residual after subtracting `t · center`.
#[must_use]
pub fn residual(coeff: &[f32], t: f32, center: &[f32]) -> Vec<f32> {
    assert_eq!(coeff.len(), center.len(), "dim mismatch");
    coeff
        .iter()
        .zip(center)
        .map(|(c, cen)| c - t * cen)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // -------------------- normalise / dot helpers --------------------

    #[test]
    fn normalise_unit_vector_leaves_unit_norm() {
        let mut v = vec![3.0_f32, 4.0];
        let n = normalise(&mut v);
        assert_abs_diff_eq!(n, 5.0, epsilon = 1e-4);
        let n2: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_abs_diff_eq!(n2, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn normalise_zero_vector_returns_zero_norm() {
        let mut v = vec![0.0_f32; 4];
        let n = normalise(&mut v);
        assert_abs_diff_eq!(n, 0.0);
        for x in v {
            assert_abs_diff_eq!(x, 0.0);
        }
    }

    #[test]
    fn dot_basic() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        assert_abs_diff_eq!(dot(&a, &b), 32.0);
    }

    // -------------------- kmeans fit --------------------

    #[test]
    fn kmeans_with_k_equal_n_assigns_each_to_itself() {
        // 3 well-separated 2D directions.
        let coeffs: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let w = vec![1.0_f32, 1.0, 1.0];
        let fit = fit_spherical_kmeans(&coeffs, &w, 2, 3, 0, 16);
        assert_eq!(fit.k, 3);
        // Each centre must match one direction up to sign / perm.
        for i in 0..3 {
            let row = &coeffs[i * 2..i * 2 + 2];
            let mut found = false;
            for c in 0..3 {
                let center = fit.center(c);
                let cos = dot(row, &center);
                if cos > 0.95 {
                    found = true;
                    break;
                }
            }
            assert!(found, "row {i} not represented by any centre");
        }
    }

    #[test]
    fn kmeans_centres_are_unit_norm() {
        let coeffs: Vec<f32> = (0..16)
            .flat_map(|i| {
                let t = (i as f32 / 16.0) * std::f32::consts::TAU;
                [t.cos() * (0.5 + i as f32 / 32.0), t.sin() * 1.5]
            })
            .collect();
        let w = vec![1.0_f32; 16];
        let fit = fit_spherical_kmeans(&coeffs, &w, 2, 4, 7, 50);
        for c in 0..fit.k {
            let center = fit.center(c);
            let norm: f32 = center.iter().map(|v| v * v).sum::<f32>().sqrt();
            // bf16 storage gives ~1e-3 relative error on unit norm.
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-2);
        }
    }

    #[test]
    fn kmeans_is_deterministic_given_seed() {
        let coeffs: Vec<f32> = (0..20)
            .flat_map(|i| {
                let t = (i as f32 * 0.5).sin();
                [t, t * 2.0, t * -0.5]
            })
            .collect();
        let w = vec![1.0_f32; 20];
        let fit1 = fit_spherical_kmeans(&coeffs, &w, 3, 4, 42, 50);
        let fit2 = fit_spherical_kmeans(&coeffs, &w, 3, 4, 42, 50);
        for c in 0..fit1.k {
            for j in 0..3 {
                assert_abs_diff_eq!(fit1.center(c)[j], fit2.center(c)[j], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn kmeans_recovers_two_clusters() {
        // Cluster A at (1, 0), cluster B at (-1, 0), 8 noisy points each.
        let mut coeffs = Vec::new();
        let rng_a = [0.02_f32, -0.01, 0.03, 0.0, -0.02, 0.01, 0.005, -0.005];
        let rng_b = [0.01_f32, -0.02, 0.0, 0.02, -0.01, -0.005, 0.005, 0.0];
        for &dy in &rng_a {
            coeffs.push(1.0);
            coeffs.push(dy);
        }
        for &dy in &rng_b {
            coeffs.push(-1.0);
            coeffs.push(dy);
        }
        let w = vec![1.0_f32; 16];
        let fit = fit_spherical_kmeans(&coeffs, &w, 2, 2, 0, 50);
        // One centre ≈ (1, 0), other ≈ (-1, 0).
        let c0 = fit.center(0);
        let c1 = fit.center(1);
        let ok = (c0[0].abs() > 0.99 && c1[0].abs() > 0.99) && c0[0].signum() != c1[0].signum();
        assert!(ok, "clusters not recovered: c0={c0:?}, c1={c1:?}");
    }

    #[test]
    fn kmeans_skips_zero_weight_rows() {
        // 3 valid directions + 2 zero-weight rows; the fit must succeed
        // with k=3 and the zero-weight rows must not affect the centres.
        let coeffs = vec![
            1.0_f32, 0.0,
            0.0, 1.0,
            -1.0, 0.0,
            // zero-weight decoys with strong anti-signal
            100.0, 100.0,
            -100.0, -100.0,
        ];
        let w = vec![1.0_f32, 1.0, 1.0, 0.0, 0.0];
        let fit = fit_spherical_kmeans(&coeffs, &w, 2, 3, 0, 20);
        assert_eq!(fit.k, 3);
        // No centre should align with the (1, 1) decoy direction.
        let decoy = 1.0 / 2.0_f32.sqrt();
        for c in 0..3 {
            let center = fit.center(c);
            let cos = (center[0] * decoy + center[1] * decoy).abs();
            assert!(cos < 0.95, "centre {c} leaked decoy: {center:?}");
        }
    }

    #[test]
    fn kmeans_handles_zero_norm_rows() {
        let coeffs = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let w = vec![1.0_f32; 4];
        let fit = fit_spherical_kmeans(&coeffs, &w, 2, 3, 0, 20);
        assert_eq!(fit.k, 3);
    }

    #[test]
    #[should_panic(expected = "k must be positive")]
    fn kmeans_rejects_k_zero() {
        let _ = fit_spherical_kmeans(&[1.0_f32, 0.0], &[1.0], 2, 0, 0, 10);
    }

    #[test]
    #[should_panic(expected = "d_eff must be positive")]
    fn kmeans_rejects_zero_d_eff() {
        let _ = fit_spherical_kmeans(&[], &[], 0, 1, 0, 10);
    }

    #[test]
    #[should_panic(expected = "coeffs length not multiple")]
    fn kmeans_rejects_misshaped_coeffs() {
        let _ = fit_spherical_kmeans(&[1.0_f32, 2.0, 3.0], &[1.0], 2, 1, 0, 10);
    }

    #[test]
    #[should_panic(expected = "weights length mismatch")]
    fn kmeans_rejects_bad_weights_length() {
        let _ = fit_spherical_kmeans(&[1.0_f32, 0.0, 0.0, 1.0], &[1.0], 2, 1, 0, 10);
    }

    #[test]
    #[should_panic(expected = "need at least")]
    fn kmeans_rejects_insufficient_valid_rows() {
        // k=3 but only 2 valid rows.
        let _ = fit_spherical_kmeans(
            &[0.0_f32, 0.0, 1.0, 0.0, -1.0, 0.0],
            &[1.0, 1.0, 1.0],
            2,
            3,
            0,
            10,
        );
    }

    // -------------------- assign_and_project --------------------

    #[test]
    fn assign_zero_coeff_returns_zero() {
        let fit = KmeansFit::from_f32(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let (seg, t) = assign_and_project(&[0.0_f32, 0.0], &fit);
        assert_eq!(seg, 0);
        assert_abs_diff_eq!(t, 0.0);
    }

    #[test]
    fn assign_finds_best_cosine() {
        let fit = KmeansFit::from_f32(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let (seg, t) = assign_and_project(&[3.0_f32, 0.0], &fit);
        assert_eq!(seg, 0);
        assert_abs_diff_eq!(t, 3.0);

        let (seg, t) = assign_and_project(&[0.0_f32, -2.0], &fit);
        assert_eq!(seg, 1);
        assert_abs_diff_eq!(t, -2.0);
    }

    #[test]
    #[should_panic(expected = "coeff dim mismatch")]
    fn assign_rejects_wrong_dim() {
        let fit = KmeansFit::from_f32(vec![1.0, 0.0], 1, 2);
        let _ = assign_and_project(&[1.0_f32], &fit);
    }

    // -------------------- residual --------------------

    #[test]
    fn residual_subtracts_projection() {
        let coeff = vec![3.0_f32, 4.0];
        let center = vec![1.0_f32, 0.0];
        let t = 3.0;
        let r = residual(&coeff, t, &center);
        assert_abs_diff_eq!(r[0], 0.0);
        assert_abs_diff_eq!(r[1], 4.0);
    }

    #[test]
    #[should_panic(expected = "dim mismatch")]
    fn residual_rejects_mismatched_dims() {
        let _ = residual(&[1.0_f32, 2.0], 1.0, &[1.0]);
    }

    // -------------------- center view --------------------

    #[test]
    fn center_view_returns_correct_slice() {
        let fit = KmeansFit::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        assert_eq!(fit.center(0), vec![1.0_f32, 2.0]);
        assert_eq!(fit.center(1), vec![3.0_f32, 4.0]);
        assert_eq!(fit.center(2), vec![5.0_f32, 6.0]);
    }
}
