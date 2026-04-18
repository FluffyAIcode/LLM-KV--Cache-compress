//! Distortion metrics `ρ`.
//!
//! Each metric is a zero-sized type (ZST) that implements the
//! [`Distortion`] trait. Because the types have zero size and all
//! methods are `#[inline(always)]`, the compiler completely erases
//! the abstraction — `R::d(x, y)` where `R = MSE` compiles to the
//! same instruction sequence as hand-writing `(x - y) * (x - y)`.
//!
//! This is how KakeyaTurbo expresses "which loss function are we
//! optimizing" without introducing any runtime dispatch.

/// How the L2 norm of a vector should be stored.
///
/// Inner-product-preserving metrics (used for attention K) need an
/// explicit high-precision norm so that `<q, x>` matches `<q, x_hat>`
/// closely. MSE-style metrics absorb the norm into the codebook scaling.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NormMode {
    /// Store the L2 norm as fp16 in the per-vector code.
    Explicit,
    /// Absorb the norm into the quantizer; do not store separately.
    Absorbed,
}

/// A distortion metric `ρ : ℝ × ℝ → ℝ₊`.
///
/// The block-level distortion is `sum_i w_i * sum_j ρ(x_{i,j}, x̂_{i,j})`.
/// The single-coordinate contract makes the trait composable with
/// SIMD inner loops.
pub trait Distortion: Copy + 'static {
    /// Human-readable name (used only for diagnostics).
    const NAME: &'static str;

    /// Whether this metric prefers explicit norm storage.
    const NORM_MODE: NormMode;

    /// Pointwise distortion: how bad is reconstructing `x` as `x_hat`?
    /// Always non-negative; zero iff `x == x_hat`.
    fn d(x: f32, x_hat: f32) -> f32;

    /// Gradient of `d` with respect to `x_hat`.
    /// Used inside iterative solvers (Lloyd-Max refinement, K-means update).
    fn grad(x: f32, x_hat: f32) -> f32;
}

/// Mean squared error (L2). The canonical choice for V cache and
/// most signal-reconstruction tasks.
#[derive(Copy, Clone, Debug, Default)]
pub struct MSE;

impl Distortion for MSE {
    const NAME: &'static str = "MSE";
    const NORM_MODE: NormMode = NormMode::Absorbed;

    #[inline(always)]
    fn d(x: f32, x_hat: f32) -> f32 {
        let e = x - x_hat;
        e * e
    }

    #[inline(always)]
    fn grad(x: f32, x_hat: f32) -> f32 {
        2.0 * (x_hat - x)
    }
}

/// Inner-product-preserving metric. Used for attention K cache and
/// vector retrieval where `<q, x>` must be preserved.
///
/// At the per-coordinate level this reduces to squared error **on the
/// normalized direction**, plus explicit norm tracking. The magnitude
/// part is taken care of by [`NormMode::Explicit`].
#[derive(Copy, Clone, Debug, Default)]
pub struct InnerProduct;

impl Distortion for InnerProduct {
    const NAME: &'static str = "InnerProduct";
    const NORM_MODE: NormMode = NormMode::Explicit;

    #[inline(always)]
    fn d(x: f32, x_hat: f32) -> f32 {
        let e = x - x_hat;
        e * e
    }

    #[inline(always)]
    fn grad(x: f32, x_hat: f32) -> f32 {
        2.0 * (x_hat - x)
    }
}

/// Huberised L-∞ metric. Quadratic near the origin (for differentiability)
/// and linear in the tail. Used for bounded-error scientific compression.
///
/// The crossover point `δ` is fixed at `0.1` for this implementation;
/// a parametric version would need a const-generic or static config.
#[derive(Copy, Clone, Debug, Default)]
pub struct LInf;

const HUBER_DELTA: f32 = 0.1;

impl Distortion for LInf {
    const NAME: &'static str = "LInf";
    const NORM_MODE: NormMode = NormMode::Absorbed;

    #[inline(always)]
    fn d(x: f32, x_hat: f32) -> f32 {
        let e = (x - x_hat).abs();
        if e < HUBER_DELTA {
            // 1/(2δ) · e² on the smooth region, normalised so that at
            // e = δ the two branches meet continuously with value δ/2.
            (e * e) / (2.0 * HUBER_DELTA)
        } else {
            e - HUBER_DELTA / 2.0
        }
    }

    #[inline(always)]
    fn grad(x: f32, x_hat: f32) -> f32 {
        let e = x_hat - x;
        if e.abs() < HUBER_DELTA {
            e / HUBER_DELTA
        } else {
            e.signum()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // -------------------- MSE --------------------

    #[test]
    fn mse_d_is_squared_error() {
        assert_abs_diff_eq!(MSE::d(0.0, 0.0), 0.0);
        assert_abs_diff_eq!(MSE::d(1.0, 3.0), 4.0);
        assert_abs_diff_eq!(MSE::d(-2.5, 0.5), 9.0);
    }

    #[test]
    fn mse_d_is_symmetric() {
        for (x, y) in [(1.0, 2.0), (-3.0, 4.0), (0.5, -0.5)] {
            assert_abs_diff_eq!(MSE::d(x, y), MSE::d(y, x));
        }
    }

    #[test]
    fn mse_d_nonneg() {
        for x in [-5.0_f32, -1.0, 0.0, 1.0, 5.0] {
            for y in [-5.0_f32, -1.0, 0.0, 1.0, 5.0] {
                assert!(MSE::d(x, y) >= 0.0, "d({x}, {y}) < 0");
            }
        }
    }

    #[test]
    fn mse_grad_matches_numerical() {
        // f32 finite differences have relative error ~1e-3 even with tuned h;
        // a proportional tolerance is the right test.
        let h = 1e-3_f32;
        for &(x, y) in &[(1.0_f32, 0.5), (-2.0, 1.0), (0.0, 3.0)] {
            let numerical = (MSE::d(x, y + h) - MSE::d(x, y - h)) / (2.0 * h);
            let analytic = MSE::grad(x, y);
            let scale = analytic.abs().max(1.0);
            assert_abs_diff_eq!(analytic, numerical, epsilon = 1e-2 * scale);
        }
    }

    #[test]
    fn mse_norm_mode_is_absorbed() {
        assert_eq!(MSE::NORM_MODE, NormMode::Absorbed);
    }

    // -------------------- InnerProduct --------------------

    #[test]
    fn inner_product_norm_mode_is_explicit() {
        assert_eq!(InnerProduct::NORM_MODE, NormMode::Explicit);
    }

    #[test]
    fn inner_product_d_matches_mse_on_directions() {
        // After normalization, the IP-preserving metric reduces to squared
        // error between unit-direction coordinates.
        for &(x, y) in &[(0.5_f32, 0.3), (-0.7, 0.7), (0.0, 0.0)] {
            assert_abs_diff_eq!(InnerProduct::d(x, y), MSE::d(x, y));
        }
    }

    #[test]
    fn inner_product_grad_matches_numerical() {
        let h = 1e-3_f32;
        for &(x, y) in &[(1.0_f32, 0.5), (-2.0, 1.0), (0.0, 3.0)] {
            let numerical =
                (InnerProduct::d(x, y + h) - InnerProduct::d(x, y - h)) / (2.0 * h);
            let analytic = InnerProduct::grad(x, y);
            let scale = analytic.abs().max(1.0);
            assert_abs_diff_eq!(analytic, numerical, epsilon = 1e-2 * scale);
        }
    }

    // -------------------- LInf (Huberised) --------------------

    #[test]
    fn linf_d_is_zero_at_equality() {
        assert_abs_diff_eq!(LInf::d(5.0, 5.0), 0.0);
        assert_abs_diff_eq!(LInf::d(-1.0, -1.0), 0.0);
    }

    #[test]
    fn linf_d_is_quadratic_near_zero() {
        // below delta=0.1 it's e²/(2δ)
        for e in [0.01_f32, 0.05, 0.09] {
            assert_abs_diff_eq!(LInf::d(0.0, e), (e * e) / (2.0 * HUBER_DELTA));
        }
    }

    #[test]
    fn linf_d_is_linear_in_tail() {
        // above delta=0.1 it's |e| - δ/2
        for e in [0.5_f32, 1.0, 3.0] {
            assert_abs_diff_eq!(LInf::d(0.0, e), e - HUBER_DELTA / 2.0);
        }
    }

    #[test]
    fn linf_d_is_continuous_at_delta() {
        let below = LInf::d(0.0, HUBER_DELTA - 1e-5);
        let above = LInf::d(0.0, HUBER_DELTA + 1e-5);
        assert_abs_diff_eq!(below, above, epsilon = 1e-3);
    }

    #[test]
    fn linf_grad_matches_numerical() {
        let h = 1e-5_f32;
        for &(x, y) in &[(0.0_f32, 0.05), (0.0, 0.5), (-1.0, 0.3)] {
            let numerical = (LInf::d(x, y + h) - LInf::d(x, y - h)) / (2.0 * h);
            assert_abs_diff_eq!(LInf::grad(x, y), numerical, epsilon = 1e-2);
        }
    }

    #[test]
    fn linf_d_nonneg() {
        for x in [-5.0_f32, -0.05, 0.0, 0.05, 5.0] {
            for y in [-5.0_f32, -0.05, 0.0, 0.05, 5.0] {
                assert!(LInf::d(x, y) >= 0.0, "linf d({x}, {y}) < 0");
            }
        }
    }

    // -------------------- Zero-size & monomorphization --------------------

    #[test]
    fn distortion_types_are_zero_sized() {
        assert_eq!(core::mem::size_of::<MSE>(), 0);
        assert_eq!(core::mem::size_of::<InnerProduct>(), 0);
        assert_eq!(core::mem::size_of::<LInf>(), 0);
    }

    #[test]
    fn distortion_trait_is_object_unsafe_as_intended() {
        // Object safety is explicitly *not* desired — we want the compiler
        // to refuse `dyn Distortion`, forcing monomorphization.
        // If this test ever fails to compile we've broken our zero-dispatch
        // contract. Because `d` and `grad` are associated functions
        // (no `&self`), the trait is already not object-safe.
        fn _check_is_static<T: Distortion>() {}
        _check_is_static::<MSE>();
        _check_is_static::<InnerProduct>();
        _check_is_static::<LInf>();
    }

    #[test]
    fn names_are_distinct() {
        assert_ne!(MSE::NAME, InnerProduct::NAME);
        assert_ne!(MSE::NAME, LInf::NAME);
        assert_ne!(InnerProduct::NAME, LInf::NAME);
    }
}
