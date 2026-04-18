//! Block-level metadata (the "skeleton" in Kakeya terminology).
//!
//! One `Skeleton` is produced per compressed block and must be stored
//! alongside the per-vector codes. It is shared by all `n` vectors
//! in the block, amortising its byte cost.
//!
//! Since v1.2 all PCA/K-means tensors inside are stored as bf16 (see
//! the A optimisation) — the `nbytes` accounting reflects that.

use crate::kmeans::KmeansFit;
use crate::pca::PcaFit;

/// Opaque skeleton. Holds everything the decoder needs apart from
/// per-vector codes.
#[derive(Debug, Clone)]
pub struct Skeleton {
    /// PCA fit: mean, basis, `d_eff`.
    pub pca: PcaFit,
    /// Spherical K-means fit on perpendicular coefficients.
    pub kmeans: KmeansFit,
    /// Rotation seed for the WHT applied to the residual.
    pub rotation_seed: u32,
    /// Length of the residual after WHT (power of two ≥ `d_eff`).
    pub wht_len: usize,
    /// Bits per residual coefficient (1..=4).
    pub bit_width: u8,
}

impl Skeleton {
    /// Total byte size of the skeleton tensors (mean + basis + centres).
    /// Each tensor is bf16, so the accounting goes through each fit's
    /// own `nbytes` method (which knows its storage dtype).
    #[must_use]
    pub fn nbytes(&self) -> usize {
        self.pca.nbytes() + self.kmeans.nbytes()
    }

    /// Effective PCA dimension.
    #[must_use]
    pub fn d_eff(&self) -> usize {
        self.pca.d_eff
    }

    /// Number of K-means centres.
    #[must_use]
    pub fn k(&self) -> usize {
        self.kmeans.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_skeleton() -> Skeleton {
        Skeleton {
            pca: PcaFit::from_f32(vec![0.0; 4], vec![0.0; 8], 2, 0.9),
            kmeans: KmeansFit::from_f32(vec![0.0; 6], 3, 2),
            rotation_seed: 42,
            wht_len: 2,
            bit_width: 3,
        }
    }

    #[test]
    fn skeleton_nbytes_reflects_bf16_storage() {
        let s = mk_skeleton();
        // bf16 = 2 bytes each.
        // PCA: mean 4 + basis 8 = 12 half-floats = 24 bytes.
        // K-means: 6 half-floats = 12 bytes.
        assert_eq!(s.pca.nbytes(), 24);
        assert_eq!(s.kmeans.nbytes(), 12);
        assert_eq!(s.nbytes(), 36);
    }

    #[test]
    fn skeleton_d_eff_and_k() {
        let s = mk_skeleton();
        assert_eq!(s.d_eff(), 2);
        assert_eq!(s.k(), 3);
    }

    #[test]
    fn skeleton_clone_is_equivalent() {
        let s = mk_skeleton();
        let s2 = s.clone();
        assert_eq!(s.nbytes(), s2.nbytes());
        assert_eq!(s.d_eff(), s2.d_eff());
        assert_eq!(s.k(), s2.k());
        assert_eq!(s.rotation_seed, s2.rotation_seed);
        assert_eq!(s.bit_width, s2.bit_width);
    }

    #[test]
    fn skeleton_nbytes_halved_vs_f32() {
        // Regression guard: v1 stored f32 (4 bytes). v1.2 stores bf16 (2 bytes).
        // Any reversal of that change will double this number.
        let s = mk_skeleton();
        let expected_bf16 = (4 + 8 + 6) * 2; // 18 half-floats
        let expected_f32 = (4 + 8 + 6) * 4; // 18 f32s (would have been)
        assert_eq!(s.nbytes(), expected_bf16);
        assert_ne!(s.nbytes(), expected_f32);
    }
}
