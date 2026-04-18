//! Block-level metadata (the "skeleton" in Kakeya terminology).
//!
//! One `Skeleton` is produced per compressed block and must be stored
//! alongside the per-vector codes. It is shared by all `n` vectors
//! in the block, amortising its byte cost.

use crate::kmeans::KmeansFit;
use crate::pca::PcaFit;

/// Opaque skeleton. Holds everything the decoder needs apart from
/// per-vector codes. Cloning is O(size).
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
    /// Fixed-size header (seeds, dims) not counted; this measures the
    /// amortised-per-block cost.
    #[must_use]
    pub fn nbytes(&self) -> usize {
        let mean = self.pca.mean.len() * std::mem::size_of::<f32>();
        let basis = self.pca.basis.len() * std::mem::size_of::<f32>();
        let centers = self.kmeans.centers.len() * std::mem::size_of::<f32>();
        mean + basis + centers
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
            pca: PcaFit {
                mean: vec![0.0; 4],
                basis: vec![0.0; 8], // 2 × 4
                d_eff: 2,
                captured_variance: 0.9,
            },
            kmeans: KmeansFit {
                centers: vec![0.0; 6], // 3 × 2
                k: 3,
                d_eff: 2,
            },
            rotation_seed: 42,
            wht_len: 2,
            bit_width: 3,
        }
    }

    #[test]
    fn skeleton_nbytes_sums_all_tensors() {
        let s = mk_skeleton();
        // 4 (mean) + 8 (basis) + 6 (centers) = 18 f32 = 72 bytes
        assert_eq!(s.nbytes(), 72);
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
}
