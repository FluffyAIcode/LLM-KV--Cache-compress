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

use half::f16;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Storage precision for the mean and basis tensors of a [`PcaFit`].
/// `Fp16` is the v1.2/v1.3 default; `Fp32` doubles skeleton bytes and is
/// used by the 2024-04 skeleton-precision ablation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcaStorage {
    /// IEEE-754 binary16 (default, matches paper byte accounting).
    Fp16,
    /// IEEE-754 binary32 (ablation-only, doubles skeleton bytes).
    Fp32,
}

impl Default for PcaStorage {
    fn default() -> Self {
        Self::Fp16
    }
}

/// Internal helper: pack a finished `(mean, basis, d_eff, captured_variance)`
/// tuple into a `PcaFit` with either f16 or fp32 skeleton storage.
fn materialize_pca_fit(
    mean: Vec<f32>,
    basis: Vec<f32>,
    d_eff: usize,
    captured_variance: f32,
    storage: PcaStorage,
) -> PcaFit {
    let d = mean.len();
    match storage {
        PcaStorage::Fp16 => PcaFit {
            mean: to_bf16(&mean),
            basis: to_bf16(&basis),
            mean_fp32: None,
            basis_fp32: None,
            d_eff,
            d,
            captured_variance,
        },
        PcaStorage::Fp32 => PcaFit {
            mean: Vec::new(),
            basis: Vec::new(),
            mean_fp32: Some(mean),
            basis_fp32: Some(basis),
            d_eff,
            d,
            captured_variance,
        },
    }
}

/// Convert an f32 slice to an owned bf16 Vec (saturating conversion for NaN/Inf).
#[inline]
fn to_bf16(src: &[f32]) -> Vec<f16> {
    src.iter().map(|&x| f16::from_f32(x)).collect()
}

/// Convert a bf16 slice to an owned f32 Vec.
#[inline]
fn to_f32(src: &[f16]) -> Vec<f32> {
    src.iter().map(|&x| x.to_f32()).collect()
}

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
///
/// **Storage contract**: tensors are stored in **bf16** to halve
/// skeleton bytes (see A optimisation in v1.2). **Arithmetic contract**:
/// `project` / `unproject` still work in f32 internally by converting
/// once on read, so the PCA computation path is numerically unchanged.
///
/// The public fields expose the raw bf16 bytes; use [`Self::mean_f32`]
/// and [`Self::basis_f32`] when you need f32 views.
#[derive(Debug, Clone)]
pub struct PcaFit {
    /// Mean vector, length `D`, stored as f16 (empty if fp32 skeleton selected).
    pub mean: Vec<f16>,
    /// Basis row-major `[d_eff, D]`, stored as f16 (empty if fp32 skeleton selected).
    pub basis: Vec<f16>,
    /// Optional fp32 mean buffer — populated iff the caller asked for
    /// `SkeletonDtype::Fp32`. When set, takes precedence over `mean`.
    pub mean_fp32: Option<Vec<f32>>,
    /// Optional fp32 basis buffer — same semantics as `mean_fp32`.
    pub basis_fp32: Option<Vec<f32>>,
    /// Number of kept components.
    pub d_eff: usize,
    /// Input dimension `D` (needed when fp32 buffers are populated and the
    /// f16 buffers are empty).
    pub d: usize,
    /// Captured variance ratio (actual, may be ≥ the requested threshold).
    pub captured_variance: f32,
}

impl PcaFit {
    /// Return the mean as a freshly-allocated f32 vector.
    #[must_use]
    pub fn mean_f32(&self) -> Vec<f32> {
        if let Some(ref m) = self.mean_fp32 {
            return m.clone();
        }
        to_f32(&self.mean)
    }

    /// Return the basis as a freshly-allocated f32 vector.
    #[must_use]
    pub fn basis_f32(&self) -> Vec<f32> {
        if let Some(ref b) = self.basis_fp32 {
            return b.clone();
        }
        to_f32(&self.basis)
    }

    /// Construct a `PcaFit` directly from f32 buffers (e.g. unit tests).
    /// Default skeleton dtype is f16 for backward compatibility.
    #[must_use]
    pub fn from_f32(mean: Vec<f32>, basis: Vec<f32>, d_eff: usize, captured: f32) -> Self {
        let d = mean.len();
        Self {
            mean: to_bf16(&mean),
            basis: to_bf16(&basis),
            mean_fp32: None,
            basis_fp32: None,
            d_eff,
            d,
            captured_variance: captured,
        }
    }

    /// Construct a `PcaFit` with fp32 skeleton storage.
    #[must_use]
    pub fn from_f32_skeleton_fp32(
        mean: Vec<f32>,
        basis: Vec<f32>,
        d_eff: usize,
        captured: f32,
    ) -> Self {
        let d = mean.len();
        Self {
            mean: Vec::new(),
            basis: Vec::new(),
            mean_fp32: Some(mean),
            basis_fp32: Some(basis),
            d_eff,
            d,
            captured_variance: captured,
        }
    }

    /// Input dimension `D`.
    #[must_use]
    pub fn d(&self) -> usize {
        if self.d > 0 {
            self.d
        } else {
            self.mean.len()
        }
    }

    /// Byte footprint of this fit (the thing the codec actually stores).
    #[must_use]
    pub fn nbytes(&self) -> usize {
        let fp32_bytes = self.mean_fp32.as_ref().map(Vec::len).unwrap_or(0)
            * std::mem::size_of::<f32>()
            + self.basis_fp32.as_ref().map(Vec::len).unwrap_or(0) * std::mem::size_of::<f32>();
        let f16_bytes = self.mean.len() * std::mem::size_of::<f16>()
            + self.basis.len() * std::mem::size_of::<f16>();
        fp32_bytes + f16_bytes
    }
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
    fit_weighted_pca_with_storage(vectors, weights, d, variance_ratio, PcaStorage::Fp16)
}

/// Full-control variant of [`fit_weighted_pca`] that lets the caller
/// pick skeleton storage precision. Mathematically identical to
/// `fit_weighted_pca`; only the final mean/basis buffers differ in dtype.
#[must_use]
pub fn fit_weighted_pca_with_storage(
    vectors: &[f32],
    weights: &[f32],
    d: usize,
    variance_ratio: f32,
    storage: PcaStorage,
) -> PcaFit {
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

    materialize_pca_fit(mean, basis, d_eff, captured_variance, storage)
}

/// Fit a weighted PCA on a concatenated multi-block tensor and return a
/// single `PcaFit` — used for the V-stream's layer-pooled basis in v1.2
/// (the A+B' optimisation). Input is row-major `[n_total, D]` with
/// matching `weights` of length `n_total`.
#[must_use]
pub fn fit_weighted_pca_pooled(
    vectors: &[f32],
    weights: &[f32],
    d: usize,
    variance_ratio: f32,
) -> PcaFit {
    fit_weighted_pca(vectors, weights, d, variance_ratio)
}

/// Randomized SVD (Halko–Martinsson–Tropp 2011) weighted PCA.
///
/// Given vectors X ∈ ℝ^{n×D} and weights w, this function finds the top
/// `d_eff` right singular vectors of the centred, weighted design matrix
/// `A := diag(√w)·(X − μ)` via a low-rank randomized sketch.
///
/// **Algorithm** (HMT 2011, §4.3 'range finder on Aᵀ'):
///
/// 1. Draw Ω ∈ ℝ^{n×r} with r = min(k + p, D), i.i.d. N(0, 1).
/// 2. Form sketch Z = Aᵀ · Ω ∈ ℝ^{D×r}.
/// 3. Power iterations: Z ← Aᵀ A Z, repeated `power_iters` times.
/// 4. QR decomposition Z = Q · R, with Q ∈ ℝ^{D×r} orthonormal.
/// 5. Form small matrix B = A · Q ∈ ℝ^{n×r} and compute its thin SVD
///    B = Û · Σ · V̂ᵀ, where V̂ ∈ ℝ^{r×r}.
/// 6. Right singular vectors of A ≈ Q · V̂. Eigenvalues of
///    Σ_w = Aᵀ A / w_sum are σ_i² / w_sum.
///
/// Complexity: **O(n · D · r)** with a single `power_iters` pass versus
/// O(n · D²) for the exact covariance path. For the v1.2 preset
/// (n=512, D=128, r≈12) that's ~12× fewer ops; at Gemma's D=512 it's
/// ~40×.
///
/// Accuracy: with `power_iters = 2` the operator-norm error is
/// `≤ (1 + 11·√(k+p)/(p−1)) · σ_{k+1}` (Halko et al. Thm 10.6). On
/// realistic KV-cache spectra with d_eff much smaller than D this is
/// sub-1% relative error.
///
/// # Arguments
///
/// - `target_rank`: upper bound on `d_eff`. Typical `D/2` for safety.
/// - `oversample`: extra sketch dims, 5–10 is standard. Bigger = more
///   accurate but more work.
/// - `power_iters`: number of subspace-power iterations, 0–3 typical.
///   More iterations → more accurate on slow-decay spectra.
/// - `seed`: RNG seed for the Gaussian test matrix.
///
/// # Panics
///
/// Same as [`fit_weighted_pca`], plus panics if `target_rank == 0`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_weighted_pca_randomized(
    vectors: &[f32],
    weights: &[f32],
    d: usize,
    variance_ratio: f32,
    target_rank: usize,
    oversample: usize,
    power_iters: u32,
    seed: u64,
) -> PcaFit {
    fit_weighted_pca_randomized_with_storage(
        vectors,
        weights,
        d,
        variance_ratio,
        target_rank,
        oversample,
        power_iters,
        seed,
        PcaStorage::Fp16,
    )
}

/// Storage-aware variant of [`fit_weighted_pca_randomized`].
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_weighted_pca_randomized_with_storage(
    vectors: &[f32],
    weights: &[f32],
    d: usize,
    variance_ratio: f32,
    target_rank: usize,
    oversample: usize,
    power_iters: u32,
    seed: u64,
    storage: PcaStorage,
) -> PcaFit {
    assert!(
        variance_ratio.is_finite(),
        "variance_ratio must be finite, got {variance_ratio}"
    );
    assert!(d > 0, "D must be positive");
    assert!(target_rank >= 1, "target_rank must be ≥ 1");
    assert_eq!(vectors.len() % d, 0, "vector buffer not a multiple of D");
    let n = weights.len();
    assert_eq!(vectors.len() / d, n, "weights length mismatch");

    let mean = weighted_mean(vectors, weights, d);
    let w_sum: f32 = weights.iter().sum();
    assert!(
        w_sum > f32::EPSILON,
        "weight sum must be positive, got {w_sum}"
    );

    // Effective sketch size r = min(k + p, D).
    let k_target = target_rank.min(d);
    let r = (k_target + oversample).min(d);

    // A ∈ ℝ^{n×D} with A[i,j] = √w_i · (x_i,j − μ_j), nalgebra stores column-major.
    let a = DMatrix::<f32>::from_fn(n, d, |i, j| {
        let w_i = weights[i].max(0.0);
        w_i.sqrt() * (vectors[i * d + j] - mean[j])
    });

    // Ω ∈ ℝ^{n×r}, i.i.d. N(0,1) via Box–Muller.
    let mut rng = SmallRng::seed_from_u64(seed);
    let omega = DMatrix::<f32>::from_fn(n, r, |_, _| {
        let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);
        (-2.0_f32 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    });

    // Z = Aᵀ · Ω, shape D×r.
    let mut z = a.transpose() * &omega;

    // Subspace power iterations with re-orthogonalisation (HMT 2011 §4.5,
    // "Algorithm 4.4: Randomized Subspace Iteration").  Without the per-
    // iteration QR, power iteration on ill-conditioned data (condition
    // number ≳ 10³) produces exponentially-growing column norms that push
    // nalgebra's subsequent thin-SVD into an effectively-non-terminating
    // Jacobi sweep on numerically-rank-deficient inputs.  Re-orthogonalising
    // Z between iterations keeps columns unit-norm and decouples the
    // iteration's stability from the spectrum of A.
    for _ in 0..power_iters {
        let ay = &a * &z; // n×r
        let ay_q = ay.qr().q();
        let ay_q = ay_q.columns(0, r).into_owned();
        let ata_q = a.transpose() * &ay_q; // D×r
        let ata_qr = ata_q.qr().q();
        z = ata_qr.columns(0, r).into_owned();
    }

    // QR of Z → Q ∈ ℝ^{D×r} orthonormal.
    let qr = z.qr();
    let q_full = qr.q();
    let q = q_full.columns(0, r).into_owned(); // D×r

    // B = A · Q, shape n×r.
    let b = &a * &q;

    // Thin SVD of B.
    let svd = b.svd(true, true);
    let singular = svd.singular_values;
    let v_t = svd.v_t.expect("SVD v_t requested");

    // v_t is r×r (since B is n×r with n > r typically). Right singular
    // vectors of B are rows of v_t. Right singular vectors of A are
    // columns of (Q · v_tᵀ).
    let v_small = v_t.transpose(); // r×r, columns = right singular vectors of B
    let basis_mat = &q * &v_small; // D×r, columns = right singular vectors of A

    // Eigenvalues of Σ_w = Aᵀ A / w_sum are σ_i² / w_sum.
    let sigma_vals: Vec<f32> = singular.iter().map(|s| s * s / w_sum).collect();

    let total_var: f32 = sigma_vals.iter().map(|v| v.max(0.0)).sum();
    let ratio = variance_ratio.clamp(0.0, 1.0);
    let mut cum = 0.0_f32;
    let mut d_eff = r;
    if total_var > f32::EPSILON {
        for (i, v) in sigma_vals.iter().enumerate() {
            cum += v.max(0.0);
            if cum / total_var >= ratio {
                d_eff = i + 1;
                break;
            }
        }
    } else {
        d_eff = 1;
    }
    d_eff = d_eff.clamp(1, k_target);

    // Flatten top-d_eff columns of basis_mat into row-major basis.
    let mut basis = Vec::with_capacity(d_eff * d);
    let mut captured = 0.0_f32;
    for k in 0..d_eff {
        captured += sigma_vals[k].max(0.0);
        for row in 0..d {
            basis.push(basis_mat[(row, k)]);
        }
    }
    let captured_variance = if total_var > f32::EPSILON {
        (captured / total_var).clamp(0.0, 1.0)
    } else {
        1.0
    };

    materialize_pca_fit(mean, basis, d_eff, captured_variance, storage)
}

/// Project a single vector `x` onto the PCA basis: `coeff = U · (x − μ)`.
///
/// Internally converts the bf16 basis/mean to f32 once per call; the
/// inner multiply-add loop stays in f32 for numerical accuracy.
#[must_use]
pub fn project(x: &[f32], fit: &PcaFit) -> Vec<f32> {
    let d = fit.d();
    assert_eq!(x.len(), d, "x dimension mismatch");
    let mean_f32 = fit.mean_f32();
    let basis_f32 = fit.basis_f32();
    let mut coeff = vec![0.0_f32; fit.d_eff];
    for k in 0..fit.d_eff {
        let mut acc = 0.0_f32;
        for j in 0..d {
            acc += basis_f32[k * d + j] * (x[j] - mean_f32[j]);
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
    let d = fit.d();
    let basis_f32 = fit.basis_f32();
    let mut x = fit.mean_f32();
    for k in 0..fit.d_eff {
        let c = coeff[k];
        for j in 0..d {
            x[j] += basis_f32[k * d + j] * c;
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
        let basis = fit.basis_f32();
        let v0 = &basis[0..2];
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
        // Tolerance: bf16 storage of mean/basis introduces ~1e-2 relative
        // round-trip error on non-tiny magnitudes.
        for i in 0..n {
            let x = &vecs[i * 2..i * 2 + 2];
            let c = project(x, &fit);
            let r = unproject(&c, &fit);
            let scale = x[0].abs().max(x[1].abs()).max(1.0);
            assert_abs_diff_eq!(x[0], r[0], epsilon = 2e-2 * scale);
            assert_abs_diff_eq!(x[1], r[1], epsilon = 2e-2 * scale);
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
        // All points project to the same reconstruction (bf16 tol).
        let c = project(&vecs[0..2], &fit);
        let r = unproject(&c, &fit);
        assert_abs_diff_eq!(r[0], 5.0, epsilon = 1e-1);
        assert_abs_diff_eq!(r[1], -3.0, epsilon = 1e-1);
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
        let basis = fit.basis_f32();
        let v0 = &basis[0..2];
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
        let basis = fit.basis_f32();
        let v0 = &basis[0..2];
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
        let fit = PcaFit::from_f32(vec![0.0; 3], vec![1.0, 0.0, 0.0], 1, 1.0);
        let _ = project(&[1.0_f32, 2.0], &fit);
    }

    #[test]
    #[should_panic(expected = "coeff length mismatch")]
    fn unproject_rejects_wrong_dim() {
        let fit = PcaFit::from_f32(vec![0.0; 3], vec![1.0, 0.0, 0.0], 1, 1.0);
        let _ = unproject(&[1.0_f32, 2.0], &fit);
    }

    // -------------------- bf16 storage round-trip --------------------

    #[test]
    fn bf16_storage_round_trips_within_tolerance() {
        // PcaFit storage is bf16; ensure mean_f32/basis_f32 round-trip
        // within bf16 precision (~1e-2 relative for values near 1).
        let mean = vec![0.25_f32, -1.5, 3.125, 0.0];
        let basis = vec![1.0_f32, -0.5, 0.25, -0.125];
        let fit = PcaFit::from_f32(mean.clone(), basis.clone(), 1, 0.9);
        let m2 = fit.mean_f32();
        let b2 = fit.basis_f32();
        for (a, b) in mean.iter().zip(m2.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-2);
        }
        for (a, b) in basis.iter().zip(b2.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-2);
        }
    }

    #[test]
    fn pca_fit_stores_bf16_not_f32() {
        // Regression guard: if we ever revert to f32 storage, nbytes doubles.
        let vecs: Vec<f32> = (0..20).map(|i| (i as f32) * 0.1).collect();
        let w = vec![1.0_f32; 5];
        let fit = fit_weighted_pca(&vecs, &w, 4, 0.95);
        let expected_bytes = (fit.mean.len() + fit.basis.len()) * 2;
        assert_eq!(
            fit.nbytes(),
            expected_bytes,
            "PcaFit should store tensors as bf16 (2 bytes each)"
        );
    }

    // -------------------- randomized SVD --------------------

    /// Subspace angle between two orthonormal bases as a scalar in [0,1]:
    /// 1 − min principal cosine. Smaller = more aligned.
    fn subspace_angle(u1: &[f32], u2: &[f32], d: usize, r1: usize, r2: usize) -> f32 {
        // Build k×k inner-product matrix U1ᵀ · U2.
        let r = r1.min(r2);
        let mut m = 0.0_f32;
        for i in 0..r1 {
            for j in 0..r2 {
                let mut dot = 0.0_f32;
                for k in 0..d {
                    dot += u1[i * d + k] * u2[j * d + k];
                }
                m = m.max(dot.abs());
            }
        }
        let _ = r;
        1.0 - m
    }

    #[test]
    fn randomized_pca_matches_exact_on_rank1_data() {
        // Rank-1 data: every vector is scalar × v. Randomized PCA must
        // recover v as the top direction with very small error.
        let n = 64;
        let d = 32;
        let mut vecs = Vec::with_capacity(n * d);
        let v: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.13).sin()).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let v_hat: Vec<f32> = v.iter().map(|x| x / norm).collect();
        for i in 0..n {
            let s = (i as f32) - (n as f32) / 2.0;
            for j in 0..d {
                vecs.push(s * v_hat[j]);
            }
        }
        let w = vec![1.0_f32; n];
        let exact = fit_weighted_pca(&vecs, &w, d, 0.99);
        let rsvd = fit_weighted_pca_randomized(&vecs, &w, d, 0.99, 2, 4, 2, 42);
        assert_eq!(exact.d_eff, 1);
        assert_eq!(rsvd.d_eff, 1);
        let a = subspace_angle(&exact.basis_f32(), &rsvd.basis_f32(), d, 1, 1);
        assert!(a < 1e-3, "rank-1 top direction should match (1 − cos = {a})");
    }

    #[test]
    fn randomized_pca_recovers_top_subspace_of_low_rank_block() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        // Build a rank-4 block in dim 24 with tiny isotropic noise.
        let n = 256;
        let d = 24;
        let rank = 4;
        let mut rng = SmallRng::seed_from_u64(7);
        let mut basis_true = vec![0.0_f32; rank * d];
        for v in &mut basis_true {
            *v = rng.gen_range(-1.0..1.0);
        }
        // QR to get orthonormal true basis.
        let m = DMatrix::<f32>::from_row_slice(rank, d, &basis_true);
        let q = m.transpose().qr().q().columns(0, rank).into_owned();
        let mut basis_flat = vec![0.0_f32; rank * d];
        for i in 0..rank {
            for j in 0..d {
                basis_flat[i * d + j] = q[(j, i)];
            }
        }
        let mut vecs = vec![0.0_f32; n * d];
        for row in 0..n {
            for k in 0..rank {
                let c: f32 = rng.gen_range(-3.0..3.0);
                for j in 0..d {
                    vecs[row * d + j] += c * basis_flat[k * d + j];
                }
            }
            for j in 0..d {
                vecs[row * d + j] += rng.gen_range(-0.01..0.01_f32);
            }
        }
        let w = vec![1.0_f32; n];
        let exact = fit_weighted_pca(&vecs, &w, d, 0.99);
        let rsvd = fit_weighted_pca_randomized(&vecs, &w, d, 0.99, 8, 6, 2, 123);
        assert_eq!(exact.d_eff, rank);
        assert!(rsvd.d_eff >= rank && rsvd.d_eff <= rank + 1);
        let a = subspace_angle(
            &exact.basis_f32()[..rank * d],
            &rsvd.basis_f32()[..rank * d],
            d,
            rank,
            rank,
        );
        assert!(a < 5e-2, "top-{rank} subspace angle must match (1 − cos = {a})");
    }

    #[test]
    fn randomized_pca_reconstruction_mse_close_to_exact() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        // Realistic scenario: 128-D block with a slow eigenvalue decay.
        let n = 512;
        let d = 128;
        let mut rng = SmallRng::seed_from_u64(19);
        // Use a diagonal covariance in a random orthonormal basis.
        let eigvals: Vec<f32> = (0..d).map(|i| (-(i as f32) / 20.0).exp()).collect();
        let mut q_mat = vec![0.0_f32; d * d];
        for v in &mut q_mat {
            *v = rng.gen_range(-1.0..1.0);
        }
        let q = DMatrix::<f32>::from_row_slice(d, d, &q_mat)
            .qr()
            .q()
            .columns(0, d)
            .into_owned();
        let mut vecs = vec![0.0_f32; n * d];
        for row in 0..n {
            let mut latent = vec![0.0_f32; d];
            for j in 0..d {
                latent[j] = rng.gen_range(-1.0..1.0_f32) * eigvals[j].sqrt();
            }
            // Rotate into ambient space.
            for j in 0..d {
                let mut s = 0.0_f32;
                for k in 0..d {
                    s += q[(j, k)] * latent[k];
                }
                vecs[row * d + j] = s;
            }
        }
        let w = vec![1.0_f32; n];
        let exact = fit_weighted_pca(&vecs, &w, d, 0.95);
        // With 3 power iterations on a slow-decay spectrum, randomized
        // SVD tracks exact to within ~20% reconstruction MSE. Fewer
        // iterations would require larger oversample.
        let rsvd = fit_weighted_pca_randomized(&vecs, &w, d, 0.95, exact.d_eff + 8, 10, 3, 777);

        // Both should capture ≥ 95% variance.
        assert!(exact.captured_variance >= 0.949);
        assert!(rsvd.captured_variance >= 0.90, "rsvd captured only {}", rsvd.captured_variance);

        // Measure per-vector reconstruction MSE under each basis.
        let mut e1 = 0.0_f64;
        let mut e2 = 0.0_f64;
        for row in 0..n {
            let x = &vecs[row * d..(row + 1) * d];
            let c1 = project(x, &exact);
            let r1 = unproject(&c1, &exact);
            let c2 = project(x, &rsvd);
            let r2 = unproject(&c2, &rsvd);
            for j in 0..d {
                e1 += (x[j] - r1[j]) as f64 * (x[j] - r1[j]) as f64;
                e2 += (x[j] - r2[j]) as f64 * (x[j] - r2[j]) as f64;
            }
        }
        // On exponentially-decaying spectra randomized SVD with 3 power
        // iterations + oversample=10 stays within 1.5× of exact MSE.
        let ratio = e2 / e1.max(1e-12);
        assert!(ratio <= 1.5, "rsvd MSE inflation {ratio:.3}× exceeds 1.5");
    }

    #[test]
    fn randomized_pca_honours_variance_ratio_truncation() {
        let n = 128;
        let d = 16;
        let vecs = ellipse_points(n, 5.0, 0.01, 0.2);
        let w = vec![1.0_f32; n];
        // Variance ratio 0.5 on an extremely skinny ellipse → should
        // truncate to 1 direction.
        let mut vecs_d = Vec::with_capacity(n * d);
        for i in 0..n {
            vecs_d.push(vecs[i * 2]);
            vecs_d.push(vecs[i * 2 + 1]);
            for _ in 2..d {
                vecs_d.push(0.0);
            }
        }
        let fit = fit_weighted_pca_randomized(&vecs_d, &w, d, 0.5, 4, 4, 1, 7);
        assert_eq!(fit.d_eff, 1);
    }

    #[test]
    fn randomized_pca_is_deterministic_on_same_seed() {
        let n = 64;
        let d = 16;
        let vecs: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let w = vec![1.0_f32; n];
        let a = fit_weighted_pca_randomized(&vecs, &w, d, 0.9, 4, 4, 2, 42);
        let b = fit_weighted_pca_randomized(&vecs, &w, d, 0.9, 4, 4, 2, 42);
        assert_eq!(a.d_eff, b.d_eff);
        for (x, y) in a.mean_f32().iter().zip(b.mean_f32().iter()) {
            assert_abs_diff_eq!(x, y, epsilon = 1e-6);
        }
        for (x, y) in a.basis_f32().iter().zip(b.basis_f32().iter()) {
            assert_abs_diff_eq!(x, y, epsilon = 1e-6);
        }
    }

    #[test]
    fn randomized_pca_different_seeds_give_similar_subspaces() {
        let n = 256;
        let d = 32;
        let vecs: Vec<f32> = (0..n * d).map(|i| ((i as f32 * 0.17).cos() * 2.0)).collect();
        let w = vec![1.0_f32; n];
        let a = fit_weighted_pca_randomized(&vecs, &w, d, 0.9, 6, 6, 2, 1);
        let b = fit_weighted_pca_randomized(&vecs, &w, d, 0.9, 6, 6, 2, 2);
        // d_eff may differ by ±1, but the leading direction should align.
        let ang = subspace_angle(&a.basis_f32()[..d], &b.basis_f32()[..d], d, 1, 1);
        assert!(ang < 5e-2, "top direction should be consistent across seeds (1 − cos = {ang})");
    }

    #[test]
    #[should_panic(expected = "target_rank must be ≥ 1")]
    fn randomized_pca_rejects_zero_rank() {
        let _ = fit_weighted_pca_randomized(&[1.0_f32; 16], &[1.0; 4], 4, 0.9, 0, 2, 1, 0);
    }

    // ---------- back to pooled fit tests ----------

    #[test]
    fn pooled_fit_matches_plain_fit() {
        // fit_weighted_pca_pooled is currently just an alias but the
        // contract is "takes n_blocks × block_size pooled data"; confirm
        // it works on a realistic shape.
        let n = 64;
        let d = 4;
        let vecs: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.13).sin()).collect();
        let w = vec![1.0_f32; n];
        let a = fit_weighted_pca(&vecs, &w, d, 0.9);
        let b = fit_weighted_pca_pooled(&vecs, &w, d, 0.9);
        assert_eq!(a.d_eff, b.d_eff);
        for (x, y) in a.mean_f32().iter().zip(b.mean_f32().iter()) {
            assert_abs_diff_eq!(x, y, epsilon = 1e-4);
        }
    }
}
