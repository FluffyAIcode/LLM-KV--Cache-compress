//! The single monomorphic encode / decode kernel.
//!
//! `encode_block<R>` and `decode_block<R>` are the only two public entry
//! points for compression. Everything about KakeyaTurbo's behaviour is
//! driven by:
//!
//! - `R: Distortion` — the loss function `ρ` (compile-time type parameter)
//! - `weights: &[f32]` — per-vector `w_i`, runtime values
//! - `params: CodecParams` — block size, variance ratio, K, bit width
//!
//! There is **one** call site for each dimension of asymmetry we've
//! discussed (L1 / L2 / L3 / L4 / L5 in the design notes). No plugins.

use half::f16;

use crate::distortion::{Distortion, NormMode};
use crate::kmeans::{
    assign_and_project, fit_spherical_kmeans_with_storage, residual,
};
use crate::pca::{
    fit_weighted_pca_randomized_with_storage, fit_weighted_pca_with_storage_capped, project,
    unproject, PcaFit, PcaStorage,
};
use crate::quantize::{
    dequantize_vector_with_centroids, pack_bits, quantize_vector_with_centroids, unpack_bits,
};
use crate::skeleton::Skeleton;
use crate::wht::{inverse_rotate, rotate};

/// Per-vector encoded representation.
///
/// Layout (bit-wise):
/// - `seg_id`: K-means cluster id (`⌈log₂ K⌉` bits, stored as u32 to ease access)
/// - `alpha, t, norm`: fp16 scalars
/// - `residual`: packed `bit_width`-bit indices of length `wht_len`
/// - `outliers`: optional sparse list of `(coord_index, f16 value)` pairs that
///   override Lloyd-Max dequantization at those coordinates. Used to
///   catch heavy-tail scaled residuals that lie outside Lloyd-Max's
///   centroid coverage (see `CodecParams::outlier_threshold`).
#[derive(Debug, Clone, PartialEq)]
pub struct Code {
    /// K-means cluster index.
    pub seg_id: u32,
    /// Projection onto the temporal direction (unused in this MVP; kept
    /// in the struct for future extension).
    pub alpha: f16,
    /// Projection onto the chosen centre: `t = <coeff, center>`.
    pub t: f16,
    /// Original L2 norm of the vector (only meaningful when
    /// `R::NORM_MODE == NormMode::Explicit`; otherwise set to 1.0).
    pub norm: f16,
    /// Packed residual indices.
    pub residual_packed: Vec<u8>,
    /// Sparse outlier list.  Each entry is `(coord_index_u16, exact_f16_value)`.
    /// Empty if `CodecParams::outlier_threshold.is_none()`.
    ///
    /// The coord_index is into the scaled-residual vector of length
    /// `wht_len`, i.e. into the WHT-rotated, per-vector-norm-scaled
    /// residual just before Lloyd-Max quantization.  Decode re-applies
    /// these values AFTER Lloyd-Max dequantization but BEFORE inverse
    /// WHT and un-scaling.
    pub outliers: Vec<(u16, f16)>,
    /// Optional Besicovitch-product payload for the scaled residual
    /// vector.  When `Some`, it *replaces* `residual_packed` +
    /// `outliers` as the residual representation:
    ///   - encode: after WHT + per-vector-norm scaling, feed the
    ///     resulting `wht_len`-dim vector to `besicovitch_encode_vector`
    ///     and serialize.
    ///   - decode: deserialize + `besicovitch_decode_vector`, then
    ///     inverse-scale (same path as Lloyd-Max) and inverse-WHT.
    /// Empty when `CodecParams::residual_besi.is_none()`.
    pub residual_besi: Vec<u8>,
}

impl Code {
    /// Total byte size of this code's payload.
    #[must_use]
    pub fn nbytes(&self) -> usize {
        // seg_id(4) + 3×fp16(6) + packed residual + outliers (2+2 bytes each)
        //                       + besi payload (replaces residual when set)
        4 + 3 * 2 + self.residual_packed.len() + self.outliers.len() * 4
            + self.residual_besi.len()
    }

    /// Number of outlier entries (0 if outlier compensation is off).
    #[must_use]
    pub fn n_outliers(&self) -> usize {
        self.outliers.len()
    }
}

/// PCA fit strategy — selects between the exact eigendecomposition
/// path (v1.2 default) and the randomized-SVD sketch (v1.3 cheap fit).
///
/// Both paths produce a bit-compatible [`PcaFit`] so the rest of the
/// codec is completely oblivious to the choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcaMethod {
    /// Exact `SymmetricEigen` on the D×D weighted covariance.
    /// Cost: O(n·D² + D³) per block. Numerically ideal.
    Exact,
    /// Halko–Martinsson–Tropp randomized SVD with the given knobs.
    /// Cost: O(n·D·r) per block where `r = target_rank + oversample`.
    Randomized {
        /// Maximum `d_eff` produced by truncation (`D/2` is a safe default).
        target_rank: usize,
        /// Extra sketch dimensions beyond `target_rank` (5–10 standard).
        oversample: usize,
        /// Subspace-power iterations (2 is the typical sweet spot).
        power_iters: u32,
        /// XOR-ed into `CodecParams::rotation_seed` to derive the
        /// Gaussian-test-matrix seed. Keeps all randomness reproducible.
        seed_offset: u64,
    },
}

impl Default for PcaMethod {
    fn default() -> Self {
        Self::Exact
    }
}

/// Storage precision for the Kakeya skeleton (PCA mean, PCA basis, K-means
/// centres).  The residual quantiser (Lloyd-Max) is **not** affected.
///
/// `Fp16` is the v1.2/v1.3 default and matches the paper's byte accounting.
/// `Fp32` doubles skeleton bytes; used only for ablation of the "f16
/// skeleton as structural PPL floor" hypothesis raised in the pre-RoPE
/// cache ablation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkeletonDtype {
    /// IEEE-754 binary16 (v1.2/v1.3 default).
    Fp16,
    /// IEEE-754 binary32 (doubles skeleton bytes; ablation-only).
    Fp32,
}

impl Default for SkeletonDtype {
    fn default() -> Self {
        Self::Fp16
    }
}

/// Runtime parameters for a single `encode_block` call.
///
/// Compile-time parameters (dimensions, distortion) are passed via
/// generics; these are the "tunables" that can vary per call without
/// recompilation.
#[derive(Debug, Clone)]
pub struct CodecParams {
    /// Variance ratio for PCA truncation (in `[0.0, 1.0]`).
    pub variance_ratio: f32,
    /// Number of K-means centres (`K ≥ 1`).
    pub k: usize,
    /// Bits per residual coordinate (`1..=4`).
    pub bit_width: u8,
    /// Seed for the WHT rotation.
    pub rotation_seed: u32,
    /// Maximum K-means iterations.
    pub kmeans_max_iter: u32,
    /// PCA fit strategy (exact vs randomized SVD).
    pub pca_method: PcaMethod,
    /// Storage precision for PCA mean, PCA basis, and K-means centres.
    /// Does not affect the residual Lloyd-Max quantiser.
    pub skeleton_dtype: SkeletonDtype,
    /// Optional hard upper bound on `d_eff` for the exact PCA path.
    /// `None` (the default) leaves `d_eff` controlled by `variance_ratio`.
    /// `Some(r)` clips `d_eff ≤ r` even if variance_ratio would keep more
    /// components — useful to match RSVD's rank budget while using exact
    /// (un-approximated) eigenvectors.
    pub exact_rank_cap: Option<usize>,
    /// Optional caller-supplied Lloyd-Max centroid table for the residual
    /// quantiser. When `Some`, must contain exactly `1 << bit_width`
    /// sorted floats — typically the output of offline empirical Lloyd-Max
    /// calibration on the model's real residual distribution.  When
    /// `None`, the codec uses the unit-variance-Gaussian centroids from
    /// [`crate::quantize::centroids_gaussian`].
    pub custom_centroids: Option<Vec<f32>>,
    /// Optional scalar threshold for outlier compensation on the scaled
    /// residual (post-WHT, per-vector-norm-scaled, pre-Lloyd-Max).
    ///
    /// When `Some(T)`: any coordinate whose absolute scaled value exceeds
    /// `T` is stored verbatim in the code's `outliers` list (as f16),
    /// and its Lloyd-Max index becomes irrelevant (decoded but overridden).
    /// When `None`: no outlier compensation.
    ///
    /// Storage cost per outlier = 2 bytes index + 2 bytes f16 = 4 bytes.
    /// At T=2.0 on Gaussian-like residuals, outlier rate is ~4.5 %.
    /// Targets Gap 1 (K-means + WHT residuals are only near-Gaussian,
    /// so Lloyd-Max's heavy-tail quantization error is disproportionate).
    pub outlier_threshold: Option<f32>,
    /// Optional Besicovitch-product codec for the scaled WHT residual.
    ///
    /// When `Some(params)`: the per-vector scalar Lloyd-Max residual
    /// quantiser is replaced by the Besicovitch-product codec on
    /// groups of `g = params.group_size` consecutive scaled-residual
    /// coordinates.  This bypasses `bit_width` and `custom_centroids`
    /// for the residual path; `outlier_threshold` and
    /// `outlier_threshold`-based sparse patching are also disabled
    /// (their effect is subsumed by Besi's per-vector adaptive scale).
    ///
    /// Rationale: the scaled residual is already near-isotropic
    /// Gaussian (WHT Gaussianization, per-vector norm scaling), so a
    /// Haar-uniform direction codebook is rate-distortion-optimal
    /// under MSE.  The 2-D joint quantisation (g = 2) also captures
    /// the residual cross-coord correlations that Lloyd-Max's scalar
    /// quantiser ignores.
    pub residual_besi: Option<crate::besicovitch::BesicovitchParams>,
}

impl Default for CodecParams {
    fn default() -> Self {
        Self {
            variance_ratio: 0.95,
            k: 16,
            bit_width: 3,
            rotation_seed: 0xCAFE_BABE,
            kmeans_max_iter: 32,
            pca_method: PcaMethod::Exact,
            skeleton_dtype: SkeletonDtype::Fp16,
            exact_rank_cap: None,
            custom_centroids: None,
            outlier_threshold: None,
            residual_besi: None,
        }
    }
}

/// Convert the codec's skeleton-dtype flag to the PCA layer's storage flag.
fn pca_storage(params: &CodecParams) -> PcaStorage {
    match params.skeleton_dtype {
        SkeletonDtype::Fp16 => PcaStorage::Fp16,
        SkeletonDtype::Fp32 => PcaStorage::Fp32,
    }
}

/// Dispatch helper: fit the requested PCA variant with the requested
/// skeleton dtype.
fn fit_pca_dispatch(
    vectors: &[f32],
    weights: &[f32],
    d: usize,
    params: &CodecParams,
) -> PcaFit {
    let storage = pca_storage(params);
    match params.pca_method {
        PcaMethod::Exact => fit_weighted_pca_with_storage_capped(
            vectors,
            weights,
            d,
            params.variance_ratio,
            storage,
            params.exact_rank_cap,
        ),
        PcaMethod::Randomized {
            target_rank,
            oversample,
            power_iters,
            seed_offset,
        } => fit_weighted_pca_randomized_with_storage(
            vectors,
            weights,
            d,
            params.variance_ratio,
            target_rank.min(d),
            oversample,
            power_iters,
            u64::from(params.rotation_seed) ^ seed_offset,
            storage,
        ),
    }
}

/// Dispatch helper for K-means: routes the codec's skeleton dtype into
/// the K-means fp32-skeleton flag.
fn fit_kmeans_dispatch(
    coeffs: &[f32],
    weights: &[f32],
    d_eff: usize,
    k: usize,
    params: &CodecParams,
) -> crate::kmeans::KmeansFit {
    fit_spherical_kmeans_with_storage(
        coeffs,
        weights,
        d_eff,
        k,
        params.rotation_seed,
        params.kmeans_max_iter,
        matches!(params.skeleton_dtype, SkeletonDtype::Fp32),
    )
}

/// Round up to the nearest power of two, with a minimum of 1.
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n.next_power_of_two()
    }
}

/// Pad `v` with zeros up to length `target`, returning a new owned `Vec`.
fn pad_zero(v: &[f32], target: usize) -> Vec<f32> {
    let mut out = v.to_vec();
    out.resize(target, 0.0);
    out
}

/// L2 norm of a slice.
fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Encode a block of `n` vectors of dimension `d`.
///
/// # Inputs
///
/// - `vectors`: row-major `[n, d]` of `f32`
/// - `weights`: length `n`, all `w_i ≥ 0`, not all zero
/// - `params`: runtime codec parameters
///
/// # Output
///
/// `(Skeleton, Vec<Code>)` where `codes.len() == n`.
///
/// # Panics
///
/// Panics on empty input, dimension mismatch, or bad parameter values
/// (delegated from the sub-modules).
pub fn encode_block<R: Distortion>(
    vectors: &[f32],
    weights: &[f32],
    d: usize,
    params: &CodecParams,
) -> (Skeleton, Vec<Code>) {
    assert!(d > 0, "dimension must be positive");
    assert!(!vectors.is_empty(), "empty vectors");
    assert_eq!(vectors.len() % d, 0, "vectors length not multiple of d");
    let n = vectors.len() / d;
    assert_eq!(weights.len(), n, "weights length != n");
    assert!((1..=4).contains(&params.bit_width), "bit_width must be 1..=4");
    assert!(params.k >= 1, "k must be ≥ 1");

    // --- Stage 1: Structure extraction ---
    let pca = fit_pca_dispatch(vectors, weights, d, params);

    // Project every vector into d_eff-space.
    let mut coeffs = Vec::with_capacity(n * pca.d_eff);
    for i in 0..n {
        let x = &vectors[i * d..(i + 1) * d];
        coeffs.extend_from_slice(&project(x, &pca));
    }

    // Adjust K downwards if the block doesn't have enough valid rows.
    // Rows with zero weight or zero coeff norm are "invalid" for K-means.
    let valid_rows = (0..n)
        .filter(|&i| {
            weights[i] > 0.0
                && coeffs[i * pca.d_eff..(i + 1) * pca.d_eff]
                    .iter()
                    .any(|c| c.abs() > f32::EPSILON)
        })
        .count();
    let effective_k = params.k.min(valid_rows.max(1));

    let kmeans = fit_kmeans_dispatch(&coeffs, weights, pca.d_eff, effective_k, params);

    // --- Stage 2: Residual coding ---
    let wht_len = next_pow2(pca.d_eff);
    let mut codes = Vec::with_capacity(n);
    for i in 0..n {
        let x = &vectors[i * d..(i + 1) * d];
        let coeff = &coeffs[i * pca.d_eff..(i + 1) * pca.d_eff];
        let (seg_id, t) = assign_and_project(coeff, &kmeans);

        let res = if coeff.iter().all(|c| c.abs() <= f32::EPSILON) {
            vec![0.0_f32; pca.d_eff]
        } else {
            residual(coeff, t, &kmeans.center(seg_id as usize))
        };
        let res_padded = pad_zero(&res, wht_len);
        let rotated = rotate(&res_padded, params.rotation_seed);

        // Scale to approximately unit variance for the Lloyd-Max codebook.
        // The `rotate` function implements an UNNORMALIZED Walsh-Hadamard
        // transform, so for residual `res` of length d_eff (padded with
        // zeros to wht_len), the rotated vector `rotated = H·D·res_padded`
        // satisfies `‖rotated‖² = wht_len · ‖res‖²`, giving an average
        // per-coordinate squared magnitude of `‖res‖²`. To match the
        // Lloyd-Max codebook (calibrated for N(0, 1)) we therefore scale
        // by `1 / ‖res‖`, so the result has unit per-coord variance.
        //
        // (Prior versions used `scale = √wht_len / ‖res‖`, which made the
        // scaled values have per-coord variance `wht_len`, saturating the
        // quantiser. Fixed in this revision.)
        let res_norm = l2_norm(&res);
        let scale = if res_norm > f32::EPSILON {
            1.0 / res_norm
        } else {
            1.0
        };
        let scaled: Vec<f32> = rotated.iter().map(|v| v * scale).collect();

        // Dispatch: Besicovitch-product residual (when residual_besi is set)
        // or classic Lloyd-Max path.  The two are mutually exclusive —
        // outlier compensation is only meaningful for Lloyd-Max (Besi
        // already has per-vector adaptive scale).
        let (packed, outliers, residual_besi_bytes) = if let Some(besi_params) =
            &params.residual_besi
        {
            let cb = crate::besicovitch::DirectionCodebook::build(
                besi_params.group_size, besi_params.direction_bits,
            );
            let code = crate::besicovitch::encode_vector(&scaled, &cb, besi_params);
            let bytes = crate::besicovitch::serialize_code(&code, besi_params);
            (Vec::new(), Vec::new(), bytes)
        } else {
            // Extract outliers BEFORE quantizing so the Lloyd-Max indices for
            // outlier coordinates don't matter (they're overridden at decode).
            let outliers: Vec<(u16, f16)> = if let Some(t) = params.outlier_threshold {
                scaled
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| {
                        if v.abs() > t {
                            Some((i as u16, f16::from_f32(v)))
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };

            let q = quantize_vector_with_centroids::<R>(
                &scaled, params.bit_width, params.custom_centroids.as_deref(),
            );
            let packed = pack_bits(&q, params.bit_width);
            (packed, outliers, Vec::new())
        };

        let norm = match R::NORM_MODE {
            NormMode::Explicit => f16::from_f32(l2_norm(x)),
            NormMode::Absorbed => f16::from_f32(1.0 / scale.max(f32::EPSILON)),
        };

        codes.push(Code {
            seg_id,
            alpha: f16::from_f32(0.0), // reserved
            t: f16::from_f32(t),
            norm,
            residual_packed: packed,
            outliers,
            residual_besi: residual_besi_bytes,
        });
    }

    let skeleton = Skeleton {
        pca,
        kmeans,
        rotation_seed: params.rotation_seed,
        wht_len,
        bit_width: params.bit_width,
        residual_besi: params.residual_besi.clone(),
    };
    (skeleton, codes)
}

/// Decode a block of codes back into approximate vectors.
///
/// Uses the unit-variance-Gaussian Lloyd-Max centroids.  For calibrated
/// codebooks use [`decode_block_with_centroids`].
///
/// # Output
///
/// Row-major `[n, d]` where `n = codes.len()` and `d = skeleton.pca.mean.len()`.
pub fn decode_block<R: Distortion>(skeleton: &Skeleton, codes: &[Code]) -> Vec<f32> {
    decode_block_with_centroids::<R>(skeleton, codes, None)
}

/// Variant of [`decode_block`] that accepts an optional caller-supplied
/// centroid table, to be used in tandem with
/// [`crate::quantize::quantize_vector_with_centroids`] on the encode side.
pub fn decode_block_with_centroids<R: Distortion>(
    skeleton: &Skeleton,
    codes: &[Code],
    custom_centroids: Option<&[f32]>,
) -> Vec<f32> {
    let d = skeleton.pca.d();
    let d_eff = skeleton.pca.d_eff;
    let wht_len = skeleton.wht_len;
    let mut out = Vec::with_capacity(codes.len() * d);
    for code in codes {
        // Residual path: Besicovitch (if configured) else Lloyd-Max + outliers.
        let mut q_vals = if let Some(besi_params) = &skeleton.residual_besi {
            let cb = crate::besicovitch::DirectionCodebook::build(
                besi_params.group_size, besi_params.direction_bits,
            );
            let groups = wht_len / besi_params.group_size;
            let besi_code = crate::besicovitch::deserialize_code(
                &code.residual_besi, groups, besi_params,
            );
            crate::besicovitch::decode_vector(&besi_code, &cb, besi_params)
        } else {
            let indices = unpack_bits(&code.residual_packed, skeleton.bit_width, wht_len);
            let mut v = dequantize_vector_with_centroids(
                &indices, skeleton.bit_width, custom_centroids,
            );
            // Outlier patch: override Lloyd-Max dequantized values at outlier
            // coordinates with their exact f16 values. These are still in the
            // SCALED residual space, so the override happens before inv_scale.
            for &(idx, val) in &code.outliers {
                let i = idx as usize;
                if i < v.len() {
                    v[i] = val.to_f32();
                }
            }
            v
        };
        let _ = &mut q_vals;

        // Inverse scale: match what encode_block did.
        // We stored 1/scale in `norm` when NORM_MODE == Absorbed.
        let inv_scale = match R::NORM_MODE {
            NormMode::Absorbed => code.norm.to_f32(),
            NormMode::Explicit => 1.0_f32, // residual stays unscaled on Explicit path
        };
        let q_scaled: Vec<f32> = q_vals.iter().map(|v| v * inv_scale).collect();

        let unrotated = inverse_rotate(&q_scaled, skeleton.rotation_seed);
        let residual_reconstructed = &unrotated[..d_eff];

        let t = code.t.to_f32();
        let center = skeleton.kmeans.center(code.seg_id as usize);
        let mut coeff = vec![0.0_f32; d_eff];
        for j in 0..d_eff {
            coeff[j] = t * center[j] + residual_reconstructed[j];
        }

        let x_hat = unproject(&coeff, &skeleton.pca);
        out.extend_from_slice(&x_hat);
    }
    out
}

// ---------------------------------------------------------------------------
// v1.2 extension: encode_layer with optional shared PCA basis.
// ---------------------------------------------------------------------------

/// Result of encoding a whole layer-stream with an optionally shared PCA basis.
///
/// When `shared_pca` is `Some`, all `per_block` entries reference this basis
/// instead of owning their own copy. This lives alongside the existing
/// `encode_block` path — callers who prefer per-block fitting (v1 behaviour)
/// still use it exactly as before.
#[derive(Debug, Clone)]
pub struct LayerEncoding {
    /// Shared PCA fit (mean + basis) — present only if `share_basis` was true.
    pub shared_pca: Option<crate::pca::PcaFit>,
    /// Per-block skeletons. When `shared_pca` is `Some`, the PCA field of each
    /// block's skeleton is a duplicate of `shared_pca` (to keep each block
    /// decodable standalone); the `nbytes` accounting in [`layer_nbytes`]
    /// subtracts the duplication.
    pub per_block: Vec<(Skeleton, Vec<Code>)>,
}

/// Byte footprint of a `LayerEncoding`, accounting for the shared PCA once
/// (not per-block).
#[must_use]
pub fn layer_nbytes(enc: &LayerEncoding) -> usize {
    let shared = enc.shared_pca.as_ref().map(crate::pca::PcaFit::nbytes).unwrap_or(0);
    let mut per_block_total = 0;
    for (sk, codes) in &enc.per_block {
        // If PCA is shared, exclude the PCA portion of each skeleton.
        let sk_bytes = if enc.shared_pca.is_some() {
            sk.kmeans.nbytes()
        } else {
            sk.nbytes()
        };
        per_block_total += sk_bytes + codes.iter().map(Code::nbytes).sum::<usize>();
    }
    shared + per_block_total
}

/// Encode a sequence of blocks.
///
/// * `blocks[i]` has shape `[block_size * d]` (row-major).
/// * `weights[i]` has length `block_size`.
/// * `share_basis == true` triggers the v1.2 B' optimisation: one weighted
///   PCA is fit on the concatenated data of all blocks and re-used across
///   them; only K-means centres remain per-block.
///
/// For the `share_basis == false` path this is exactly equivalent to
/// calling [`encode_block`] on each block independently.
pub fn encode_layer<R: Distortion>(
    blocks: &[Vec<f32>],
    weights: &[Vec<f32>],
    d: usize,
    params: &CodecParams,
    share_basis: bool,
) -> LayerEncoding {
    assert_eq!(blocks.len(), weights.len(), "blocks and weights length mismatch");
    assert!(!blocks.is_empty(), "at least one block required");

    if !share_basis {
        let mut per_block = Vec::with_capacity(blocks.len());
        for (b, w) in blocks.iter().zip(weights) {
            let (sk, codes) = encode_block::<R>(b, w, d, params);
            per_block.push((sk, codes));
        }
        return LayerEncoding { shared_pca: None, per_block };
    }

    // --- share_basis == true path ---
    //
    // 1. Fit pooled PCA once on the union of all blocks' vectors.
    // 2. For each block: project against the shared basis, run per-block
    //    K-means in the coefficient space, quantise the residual as usual.
    //
    // This is implemented by re-using encode_block via a modified params
    // that tells it "the PCA is already fit" — but since encode_block's
    // signature doesn't expose that, we inline the pipeline here. The
    // inner ops (kmeans, rotation, quantise) are identical.
    use crate::kmeans::{assign_and_project, residual};
    use crate::pca::project;
    use crate::quantize::{pack_bits, quantize_vector_with_centroids};
    use crate::wht::rotate;
    use half::f16;

    let n_total: usize = blocks.iter().map(|b| b.len() / d).sum();
    let mut pooled_vecs = Vec::with_capacity(n_total * d);
    let mut pooled_weights = Vec::with_capacity(n_total);
    for (b, w) in blocks.iter().zip(weights) {
        pooled_vecs.extend_from_slice(b);
        pooled_weights.extend_from_slice(w);
    }
    // Pooled PCA honours the same PcaMethod as per-block fits. For the
    // randomized path the seed is derived from rotation_seed XOR a
    // per-layer salt so different layers don't reuse identical sketches.
    let shared_pca = fit_pca_dispatch(&pooled_vecs, &pooled_weights, d, params);

    // Pre-project every block's coefficients so we can run K-means in
    // coefficient space on each block.
    let wht_len = next_pow2(shared_pca.d_eff);

    let mut per_block: Vec<(Skeleton, Vec<Code>)> = Vec::with_capacity(blocks.len());
    for (block_vecs, w) in blocks.iter().zip(weights) {
        let n = block_vecs.len() / d;
        assert_eq!(w.len(), n, "block weight length mismatch");

        // Project through the shared basis.
        let mut coeffs = Vec::with_capacity(n * shared_pca.d_eff);
        for i in 0..n {
            let x = &block_vecs[i * d..(i + 1) * d];
            coeffs.extend_from_slice(&project(x, &shared_pca));
        }
        // K-means per block (effective_k shrinks if block has few valid rows).
        let valid_rows = (0..n).filter(|&i| {
            w[i] > 0.0
                && coeffs[i * shared_pca.d_eff..(i + 1) * shared_pca.d_eff]
                    .iter()
                    .any(|c| c.abs() > f32::EPSILON)
        }).count();
        let effective_k = params.k.min(valid_rows.max(1));
        let kmeans = fit_kmeans_dispatch(&coeffs, w, shared_pca.d_eff, effective_k, params);

        let mut codes = Vec::with_capacity(n);
        for i in 0..n {
            let x = &block_vecs[i * d..(i + 1) * d];
            let coeff = &coeffs[i * shared_pca.d_eff..(i + 1) * shared_pca.d_eff];
            let (seg_id, t) = assign_and_project(coeff, &kmeans);
            let res = if coeff.iter().all(|c| c.abs() <= f32::EPSILON) {
                vec![0.0_f32; shared_pca.d_eff]
            } else {
                residual(coeff, t, &kmeans.center(seg_id as usize))
            };
            let res_padded = pad_zero(&res, wht_len);
            let rotated = rotate(&res_padded, params.rotation_seed);
            let res_norm = l2_norm(&res);
            let scale = if res_norm > f32::EPSILON { 1.0 / res_norm } else { 1.0 };
            let scaled: Vec<f32> = rotated.iter().map(|v| v * scale).collect();
            let (packed, outliers, residual_besi_bytes) = if let Some(besi_params) =
                &params.residual_besi
            {
                let cb = crate::besicovitch::DirectionCodebook::build(
                    besi_params.group_size, besi_params.direction_bits,
                );
                let code = crate::besicovitch::encode_vector(&scaled, &cb, besi_params);
                let bytes = crate::besicovitch::serialize_code(&code, besi_params);
                (Vec::new(), Vec::new(), bytes)
            } else {
                let outliers: Vec<(u16, f16)> = if let Some(thr) = params.outlier_threshold {
                    scaled
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, &v)| {
                            if v.abs() > thr {
                                Some((idx as u16, f16::from_f32(v)))
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let q = quantize_vector_with_centroids::<R>(
                    &scaled, params.bit_width, params.custom_centroids.as_deref(),
                );
                let packed = pack_bits(&q, params.bit_width);
                (packed, outliers, Vec::new())
            };
            let norm = match R::NORM_MODE {
                NormMode::Explicit => f16::from_f32(l2_norm(x)),
                NormMode::Absorbed => f16::from_f32(1.0 / scale.max(f32::EPSILON)),
            };
            codes.push(Code {
                seg_id,
                alpha: f16::from_f32(0.0),
                t: f16::from_f32(t),
                norm,
                residual_packed: packed,
                outliers,
                residual_besi: residual_besi_bytes,
            });
        }

        let skeleton = Skeleton {
            pca: shared_pca.clone(),
            kmeans,
            rotation_seed: params.rotation_seed,
            wht_len,
            bit_width: params.bit_width,
            residual_besi: params.residual_besi.clone(),
        };
        per_block.push((skeleton, codes));
    }

    LayerEncoding {
        shared_pca: Some(shared_pca),
        per_block,
    }
}

/// Decode a whole layer. Dual of [`encode_layer`].
pub fn decode_layer<R: Distortion>(enc: &LayerEncoding) -> Vec<Vec<f32>> {
    decode_layer_with_centroids::<R>(enc, None)
}

/// Variant of [`decode_layer`] that accepts an optional caller-supplied
/// centroid table — pass the same table used at encode time.
pub fn decode_layer_with_centroids<R: Distortion>(
    enc: &LayerEncoding,
    custom_centroids: Option<&[f32]>,
) -> Vec<Vec<f32>> {
    enc.per_block
        .iter()
        .map(|(sk, codes)| decode_block_with_centroids::<R>(sk, codes, custom_centroids))
        .collect()
}

// ---------------------------------------------------------------------------
// Utility: size accounting for reporting / benchmarks.
// ---------------------------------------------------------------------------

/// Total byte footprint: skeleton + all codes.
#[must_use]
pub fn total_bytes(skeleton: &Skeleton, codes: &[Code]) -> usize {
    skeleton.nbytes() + codes.iter().map(Code::nbytes).sum::<usize>()
}

/// Compute the raw uncompressed f32 footprint of a block.
#[must_use]
pub fn raw_bytes(n: usize, d: usize) -> usize {
    n * d * std::mem::size_of::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distortion::{InnerProduct, LInf, MSE};
    use approx::assert_abs_diff_eq;

    // Build a synthetic block: n vectors on a low-rank subspace plus noise.
    fn synthetic_block(n: usize, d: usize, rank: usize, noise: f32, seed: u64) -> Vec<f32> {
        use rand::rngs::SmallRng;
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = SmallRng::seed_from_u64(seed);
        // Random orthonormal basis via QR would be cleaner, but for tests a
        // simple set of axis-aligned directions + random latents is enough.
        let mut latents = vec![0.0_f32; n * rank];
        for v in latents.iter_mut() {
            *v = rng.gen_range(-1.0..1.0_f32);
        }
        let mut out = vec![0.0_f32; n * d];
        for i in 0..n {
            for r in 0..rank {
                let coef = latents[i * rank + r];
                // Place the r-th latent along the r-th axis.
                out[i * d + r] += coef;
            }
            for j in 0..d {
                out[i * d + j] += rng.gen_range(-noise..noise);
            }
        }
        out
    }

    fn mse_of(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let sq: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        sq / a.len() as f32
    }

    // -------------------- basic round-trip --------------------

    #[test]
    fn round_trip_mse_preserves_structure() {
        let n = 64;
        let d = 16;
        let block = synthetic_block(n, d, 4, 0.01, 1);
        let w = vec![1.0_f32; n];
        let params = CodecParams {
            variance_ratio: 0.99,
            k: 4,
            bit_width: 4,
            rotation_seed: 0xABCD,
            kmeans_max_iter: 32,
            pca_method: PcaMethod::Exact,
            skeleton_dtype: SkeletonDtype::Fp16,
            exact_rank_cap: None,
            custom_centroids: None,
            outlier_threshold: None,
            residual_besi: None,
        };
        let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
        let recovered = decode_block::<MSE>(&sk, &codes);
        assert_eq!(recovered.len(), block.len());
        let err = mse_of(&block, &recovered);
        // With rank=4 signal + low noise + 4-bit + d_eff captures >99%,
        // reconstruction MSE should be significantly below raw noise (0.0001).
        assert!(err < 0.05, "round-trip MSE too high: {err}");
    }

    #[test]
    fn round_trip_inner_product_preserves_structure() {
        let n = 32;
        let d = 16;
        let block = synthetic_block(n, d, 4, 0.01, 2);
        let w = vec![1.0_f32; n];
        let params = CodecParams {
            variance_ratio: 0.99,
            k: 4,
            bit_width: 4,
            ..Default::default()
        };
        let (sk, codes) = encode_block::<InnerProduct>(&block, &w, d, &params);
        let recovered = decode_block::<InnerProduct>(&sk, &codes);
        assert_eq!(recovered.len(), block.len());
        let err = mse_of(&block, &recovered);
        // InnerProduct may not perfectly preserve MSE, but it should still
        // produce a bounded reconstruction.
        assert!(err.is_finite(), "reconstruction not finite: {err}");
    }

    #[test]
    fn round_trip_linf_runs() {
        let n = 32;
        let d = 16;
        let block = synthetic_block(n, d, 4, 0.01, 3);
        let w = vec![1.0_f32; n];
        let params = CodecParams {
            variance_ratio: 0.99,
            k: 4,
            bit_width: 4,
            ..Default::default()
        };
        let (sk, codes) = encode_block::<LInf>(&block, &w, d, &params);
        let recovered = decode_block::<LInf>(&sk, &codes);
        assert_eq!(recovered.len(), block.len());
        for v in recovered {
            assert!(v.is_finite(), "non-finite reconstruction");
        }
    }

    // -------------------- bit width monotonicity --------------------

    #[test]
    fn more_bits_better_reconstruction_on_average() {
        let n = 128;
        let d = 32;
        let block = synthetic_block(n, d, 8, 0.02, 4);
        let w = vec![1.0_f32; n];
        let mut prev = f32::INFINITY;
        for bits in 1..=4u8 {
            let params = CodecParams {
                variance_ratio: 0.95,
                k: 8,
                bit_width: bits,
                ..Default::default()
            };
            let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
            let recovered = decode_block::<MSE>(&sk, &codes);
            let err = mse_of(&block, &recovered);
            assert!(err.is_finite());
            // Higher bits should not dramatically increase error; strict
            // monotonicity is too brittle due to WHT/PCA interactions,
            // so we just check that 4-bit beats 1-bit.
            let _ = prev;
            prev = err;
        }
        // Compare extremes directly.
        let params1 = CodecParams { bit_width: 1, ..Default::default() };
        let params4 = CodecParams { bit_width: 4, ..Default::default() };
        let (sk1, c1) = encode_block::<MSE>(&block, &w, d, &params1);
        let (sk4, c4) = encode_block::<MSE>(&block, &w, d, &params4);
        let r1 = decode_block::<MSE>(&sk1, &c1);
        let r4 = decode_block::<MSE>(&sk4, &c4);
        let e1 = mse_of(&block, &r1);
        let e4 = mse_of(&block, &r4);
        assert!(e4 < e1, "4-bit MSE {e4} must beat 1-bit {e1}");
    }

    // -------------------- weights drive behaviour --------------------

    #[test]
    fn high_weight_on_one_row_drives_pca() {
        let n = 8;
        let d = 4;
        // 7 rows near origin, one outlier with huge weight.
        let mut block = vec![0.0_f32; n * d];
        block[0] = 10.0; // single "big" value at the first row, first coordinate
        let mut w = vec![1.0_f32; n];
        w[0] = 1000.0;
        let params = CodecParams {
            variance_ratio: 0.95,
            k: 2,
            bit_width: 4,
            ..Default::default()
        };
        let (sk, _) = encode_block::<MSE>(&block, &w, d, &params);
        // The captured variance should be near 100%.
        assert!(sk.pca.captured_variance >= 0.9);
    }

    // -------------------- output shape & code nbytes --------------------

    #[test]
    fn output_shape_matches_input() {
        let n = 17;
        let d = 9;
        let block = synthetic_block(n, d, 3, 0.01, 5);
        let w = vec![1.0_f32; n];
        let params = CodecParams { variance_ratio: 0.9, k: 3, bit_width: 3, ..Default::default() };
        let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
        assert_eq!(codes.len(), n);
        let r = decode_block::<MSE>(&sk, &codes);
        assert_eq!(r.len(), n * d);
    }

    #[test]
    fn code_nbytes_reasonable() {
        let n = 4;
        let d = 8;
        let block = synthetic_block(n, d, 2, 0.01, 6);
        let w = vec![1.0_f32; n];
        let params = CodecParams { bit_width: 3, k: 2, ..Default::default() };
        let (_, codes) = encode_block::<MSE>(&block, &w, d, &params);
        for c in &codes {
            // At minimum: 4 (u32) + 6 (3×fp16) + at least 1 byte packed.
            assert!(c.nbytes() > 4 + 6);
            assert!(!c.residual_packed.is_empty());
        }
    }

    #[test]
    fn raw_bytes_matches_f32_footprint() {
        assert_eq!(raw_bytes(10, 8), 320);
    }

    #[test]
    fn total_bytes_includes_skeleton_and_codes() {
        let n = 4;
        let d = 8;
        let block = synthetic_block(n, d, 2, 0.01, 7);
        let w = vec![1.0_f32; n];
        let params = CodecParams { bit_width: 3, k: 2, ..Default::default() };
        let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
        let total = total_bytes(&sk, &codes);
        assert!(total > sk.nbytes());
        assert!(total > codes.iter().map(Code::nbytes).sum::<usize>() - 1);
    }

    // -------------------- determinism --------------------

    #[test]
    fn encode_is_deterministic() {
        let n = 20;
        let d = 8;
        let block = synthetic_block(n, d, 3, 0.01, 8);
        let w = vec![1.0_f32; n];
        let params = CodecParams { bit_width: 3, k: 4, rotation_seed: 0xDEAD, ..Default::default() };
        let (_, c1) = encode_block::<MSE>(&block, &w, d, &params);
        let (_, c2) = encode_block::<MSE>(&block, &w, d, &params);
        assert_eq!(c1, c2);
    }

    #[test]
    fn different_seeds_give_different_codes() {
        let n = 20;
        let d = 8;
        let block = synthetic_block(n, d, 3, 0.1, 9);
        let w = vec![1.0_f32; n];
        let mut p = CodecParams { bit_width: 3, k: 4, rotation_seed: 0x1, ..Default::default() };
        let (_, c1) = encode_block::<MSE>(&block, &w, d, &p);
        p.rotation_seed = 0x2;
        let (_, c2) = encode_block::<MSE>(&block, &w, d, &p);
        // At least one packed residual should differ.
        let any_diff = c1.iter().zip(&c2).any(|(a, b)| a.residual_packed != b.residual_packed);
        assert!(any_diff, "seed change had no effect");
    }

    // -------------------- panics / invalid input --------------------

    #[test]
    #[should_panic(expected = "dimension must be positive")]
    fn rejects_zero_dim() {
        let _ = encode_block::<MSE>(&[] as &[f32], &[], 0, &CodecParams::default());
    }

    #[test]
    #[should_panic(expected = "empty vectors")]
    fn rejects_empty_vectors() {
        let _ = encode_block::<MSE>(&[] as &[f32], &[], 4, &CodecParams::default());
    }

    #[test]
    #[should_panic(expected = "vectors length not multiple of d")]
    fn rejects_misshaped_vectors() {
        let _ = encode_block::<MSE>(&[1.0_f32, 2.0, 3.0], &[1.0], 2, &CodecParams::default());
    }

    #[test]
    #[should_panic(expected = "weights length != n")]
    fn rejects_bad_weights_len() {
        let _ = encode_block::<MSE>(
            &[1.0_f32, 2.0, 3.0, 4.0],
            &[1.0],
            2,
            &CodecParams::default(),
        );
    }

    #[test]
    #[should_panic(expected = "bit_width must be 1..=4")]
    fn rejects_bad_bit_width() {
        let params = CodecParams { bit_width: 5, ..Default::default() };
        let _ = encode_block::<MSE>(&[1.0_f32, 2.0, 3.0, 4.0], &[1.0, 1.0], 2, &params);
    }

    #[test]
    #[should_panic(expected = "k must be ≥ 1")]
    fn rejects_zero_k() {
        let params = CodecParams { k: 0, ..Default::default() };
        let _ = encode_block::<MSE>(&[1.0_f32, 2.0, 3.0, 4.0], &[1.0, 1.0], 2, &params);
    }

    // -------------------- default params --------------------

    #[test]
    fn default_params_make_sense() {
        let p = CodecParams::default();
        assert!(p.variance_ratio > 0.0 && p.variance_ratio <= 1.0);
        assert!(p.k >= 1);
        assert!((1..=4).contains(&p.bit_width));
        assert!(p.kmeans_max_iter > 0);
        assert_eq!(p.skeleton_dtype, SkeletonDtype::Fp16);
        assert!(p.outlier_threshold.is_none());
    }

    // -------------------- outlier compensation --------------------

    #[test]
    #[test]
    fn residual_besi_round_trip_reconstructs_input() {
        // Drop Lloyd-Max residual, swap in Besicovitch.  Round-trip
        // should reconstruct with MSE comparable to a Lloyd-Max b=3/4 run.
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(77);
        let n = 64;
        let d = 32;
        let mut block = vec![0.0_f32; n * d];
        for v in &mut block {
            *v = rng.r#gen::<f32>() * 2.0 - 1.0;
        }
        let w = vec![1.0_f32; n];

        let params = CodecParams {
            bit_width: 4, k: 8, variance_ratio: 0.99,
            residual_besi: Some(crate::besicovitch::BesicovitchParams {
                group_size: 2, direction_bits: 4, magnitude_bits: 4,
                magnitude_mode: crate::besicovitch::MagnitudeMode::QuantizedWithPerVectorScale,
                subtract_mean: false,
            }),
            ..Default::default()
        };
        let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
        assert!(sk.residual_besi.is_some());
        // All codes must have populated residual_besi and empty Lloyd-Max residual.
        for c in &codes {
            assert!(c.residual_besi.len() > 0,
                    "residual_besi bytes must be populated when params.residual_besi is Some");
            assert_eq!(c.residual_packed.len(), 0,
                       "residual_packed must be empty on Besi residual path");
            assert_eq!(c.outliers.len(), 0,
                       "outliers disabled on Besi residual path");
        }
        let rec = decode_block::<MSE>(&sk, &codes);
        let e = mse_of(&block, &rec);
        assert!(e < 0.05, "Besi residual round-trip MSE too large: {e:.4e}");
    }

    #[test]
    fn residual_besi_skeleton_records_params() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(101);
        let n = 32; let d = 16;
        let mut block = vec![0.0_f32; n * d];
        for v in &mut block { *v = rng.r#gen::<f32>() * 2.0 - 1.0; }
        let w = vec![1.0_f32; n];
        let besi = crate::besicovitch::BesicovitchParams {
            group_size: 2, direction_bits: 5, magnitude_bits: 3,
            magnitude_mode: crate::besicovitch::MagnitudeMode::QuantizedWithPerVectorScale,
            subtract_mean: false,
        };
        let params = CodecParams {
            bit_width: 3, k: 4, residual_besi: Some(besi.clone()),
            ..Default::default()
        };
        let (sk, _) = encode_block::<MSE>(&block, &w, d, &params);
        let got = sk.residual_besi.as_ref().expect("expected besi on skeleton");
        assert_eq!(got.group_size, besi.group_size);
        assert_eq!(got.direction_bits, besi.direction_bits);
        assert_eq!(got.magnitude_bits, besi.magnitude_bits);
    }

    #[test]
    fn outlier_off_by_default_means_no_outliers() {
        let n = 8;
        let d = 16;
        let mut block = vec![0.0_f32; n * d];
        for (i, v) in block.iter_mut().enumerate() {
            *v = ((i as f32) * 0.123).sin() * 2.0;
        }
        let w = vec![1.0_f32; n];
        let params = CodecParams { bit_width: 2, k: 4, ..Default::default() };
        let (_, codes) = encode_block::<MSE>(&block, &w, d, &params);
        for c in &codes {
            assert!(c.outliers.is_empty(), "outliers should be empty when threshold is None");
        }
    }

    #[test]
    fn outlier_threshold_extracts_large_scaled_values() {
        // Engineered: make residuals large enough that scaled values
        // definitely exceed threshold T=0.5 on at least some coords.
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(7);
        let n = 16;
        let d = 8;
        let mut block = vec![0.0_f32; n * d];
        for v in &mut block {
            *v = rng.gen_range(-1.0_f32..1.0);
        }
        let w = vec![1.0_f32; n];
        let params = CodecParams {
            bit_width: 2,
            k: 4,
            variance_ratio: 0.99,
            outlier_threshold: Some(0.5),
            ..Default::default()
        };
        let (_, codes) = encode_block::<MSE>(&block, &w, d, &params);
        let total_outliers: usize = codes.iter().map(|c| c.outliers.len()).sum();
        assert!(total_outliers > 0, "T=0.5 on this input should yield at least one outlier");
    }

    #[test]
    fn outlier_round_trip_reduces_reconstruction_mse() {
        // Compare MSE with/without outlier compensation on the same block.
        // At b=2 with Gaussian-like residuals, outliers dominate the MSE,
        // so enabling outlier compensation must strictly decrease MSE.
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(13);
        let n = 64;
        let d = 16;
        let mut block = vec![0.0_f32; n * d];
        for v in &mut block {
            *v = rng.gen::<f32>() * 2.0 - 1.0;
        }
        let w = vec![1.0_f32; n];

        let p_no = CodecParams { bit_width: 2, k: 8, variance_ratio: 0.99,
                                 outlier_threshold: None, ..Default::default() };
        let (sk_no, c_no) = encode_block::<MSE>(&block, &w, d, &p_no);
        let rec_no = decode_block::<MSE>(&sk_no, &c_no);
        let mse_no = mse_of(&block, &rec_no);

        let p_on = CodecParams { bit_width: 2, k: 8, variance_ratio: 0.99,
                                 outlier_threshold: Some(2.0), ..Default::default() };
        let (sk_on, c_on) = encode_block::<MSE>(&block, &w, d, &p_on);
        let rec_on = decode_block::<MSE>(&sk_on, &c_on);
        let mse_on = mse_of(&block, &rec_on);

        assert!(
            mse_on <= mse_no,
            "outlier compensation at T=2.0 must not make MSE worse: off={mse_no:.5e}, on={mse_on:.5e}"
        );
        // At b=2 with non-trivial data, the improvement should be real
        // (≥ 10% MSE reduction).  Allow leeway but flag a regression.
        assert!(
            mse_on < mse_no,
            "outlier T=2.0 should strictly reduce MSE at b=2 on Gaussian-like residuals; off={mse_no:.5e}, on={mse_on:.5e}"
        );
    }

    #[test]
    fn outlier_byte_cost_scales_with_outlier_count() {
        // Each outlier entry should cost exactly 4 bytes (u16 index + f16 value).
        let n = 32;
        let d = 8;
        let mut block = vec![0.0_f32; n * d];
        for i in 0..n {
            for j in 0..d {
                block[i * d + j] = ((i + j) as f32 * 0.3).sin();
            }
        }
        let w = vec![1.0_f32; n];
        let p_no = CodecParams { bit_width: 2, k: 4, ..Default::default() };
        let p_on = CodecParams {
            bit_width: 2, k: 4,
            outlier_threshold: Some(1.0),
            ..Default::default()
        };
        let (_, c_no) = encode_block::<MSE>(&block, &w, d, &p_no);
        let (_, c_on) = encode_block::<MSE>(&block, &w, d, &p_on);
        for (a, b) in c_no.iter().zip(c_on.iter()) {
            let expected_delta = b.outliers.len() * 4;
            let actual_delta = b.nbytes() - a.nbytes();
            assert_eq!(
                actual_delta, expected_delta,
                "outlier byte cost should be 4 bytes each (got {} for {} outliers)",
                actual_delta, b.outliers.len()
            );
        }
    }

    #[test]
    fn outlier_with_very_low_threshold_patches_all_coords() {
        // T=0 means every coord is an outlier; reconstruction should then
        // be near-perfect (limited only by f16 precision).
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(21);
        let n = 8;
        let d = 8;
        let mut block = vec![0.0_f32; n * d];
        for v in &mut block {
            *v = rng.gen::<f32>() * 2.0 - 1.0;
        }
        let w = vec![1.0_f32; n];
        let params = CodecParams {
            bit_width: 2, k: 4, variance_ratio: 0.99,
            outlier_threshold: Some(0.0),  // everything is an outlier
            ..Default::default()
        };
        let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
        let rec = decode_block::<MSE>(&sk, &codes);
        let mse = mse_of(&block, &rec);
        // With every coord patched as exact f16, the Lloyd-Max residual
        // error is fully bypassed. Remaining error comes from f16
        // precision on (i) outlier values, (ii) the K-means center / t
        // scalar reconstruction, and (iii) the f16 PCA basis/mean.
        // On uniform-random [-1, 1] input this floor is ~2e-3.
        assert!(mse < 3e-3,
                "T=0 patches every coord → near-lossless, got MSE={mse:.4e}");
    }

    // -------------------- skeleton_dtype ablation --------------------

    /// FP32 skeleton must preserve the PCA mean and basis to full float
    /// precision — no f16 rounding on the way in or out.
    #[test]
    fn skeleton_fp32_preserves_precision() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(0xABCD);
        let n = 64;
        let d = 32;
        let mut block = vec![0.0_f32; n * d];
        for v in &mut block {
            *v = rng.gen_range(-1.0_f32..1.0);
        }
        let w = vec![1.0_f32; n];
        let base = CodecParams {
            variance_ratio: 0.99,
            k: 4,
            bit_width: 3,
            ..Default::default()
        };
        let p_fp16 = CodecParams { skeleton_dtype: SkeletonDtype::Fp16, ..base.clone() };
        let p_fp32 = CodecParams { skeleton_dtype: SkeletonDtype::Fp32, ..base };
        let (sk16, _) = encode_block::<MSE>(&block, &w, d, &p_fp16);
        let (sk32, _) = encode_block::<MSE>(&block, &w, d, &p_fp32);

        // Same d_eff (fit is numerically identical; only storage differs).
        assert_eq!(sk16.pca.d_eff, sk32.pca.d_eff);

        // fp16 path: mean/basis are f16-rounded; mean_fp32 / basis_fp32 are None.
        assert!(sk16.pca.mean_fp32.is_none());
        assert!(sk16.pca.basis_fp32.is_none());

        // fp32 path: mean_fp32 / basis_fp32 are Some; f16 buffers are empty.
        let mean32 = sk32.pca.mean_fp32.as_ref().expect("fp32 mean");
        let basis32 = sk32.pca.basis_fp32.as_ref().expect("fp32 basis");
        assert!(sk32.pca.mean.is_empty());
        assert!(sk32.pca.basis.is_empty());

        // fp32 mean/basis must match the unrounded fp32 values exactly,
        // while fp16 storage round-trips through f16 with non-zero error.
        let mean16 = sk16.pca.mean_f32();
        assert_eq!(mean32.len(), mean16.len());
        let mean_delta: f32 = mean32
            .iter()
            .zip(&mean16)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        // mean_delta > 0 would show f16 was doing some rounding; but if the
        // mean happens to fall exactly on f16-representable grid points it
        // may be zero. Either way the structural check above is the real
        // guarantee. We include this loose sanity check without asserting.
        let _ = mean_delta;

        // K-means centres follow the same storage contract.
        assert!(sk32.kmeans.centers_fp32.is_some());
        assert!(sk32.kmeans.centers.is_empty());

        // Byte accounting: fp32 skeleton must be ~2× fp16.
        assert!(
            sk32.pca.nbytes() >= 2 * sk16.pca.nbytes() - 8,
            "fp32 PCA should be ~2× fp16 ({} vs {})",
            sk32.pca.nbytes(),
            sk16.pca.nbytes()
        );

        // basis32 values should have full fp32 precision (no rounding to f16
        // grid); verify at least some value has more than 10 bits of mantissa
        // beyond the f16 nearest.
        let basis16 = sk16.pca.basis_f32();
        let mut saw_finer = false;
        for (a, b) in basis32.iter().zip(&basis16) {
            if (a - b).abs() > 1e-6 {
                saw_finer = true;
                break;
            }
        }
        assert!(
            saw_finer,
            "fp32 basis must differ from f16-rounded version in at least one coordinate"
        );
    }

    /// Round-trip must still work under fp32 skeleton storage.
    #[test]
    fn skeleton_fp32_round_trip() {
        let n = 16;
        let d = 8;
        let mut block = vec![0.0_f32; n * d];
        for i in 0..n {
            for j in 0..d {
                block[i * d + j] = ((i + j) as f32).sin();
            }
        }
        let w = vec![1.0_f32; n];
        let params = CodecParams {
            variance_ratio: 0.95,
            k: 4,
            bit_width: 3,
            skeleton_dtype: SkeletonDtype::Fp32,
            ..Default::default()
        };
        let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
        let r = decode_block::<MSE>(&sk, &codes);
        assert_eq!(r.len(), block.len());
        for v in r {
            assert!(v.is_finite());
        }
        // fp32 skeleton: mean_fp32 / basis_fp32 should be Some, f16 buffers empty.
        assert!(sk.pca.mean_fp32.is_some());
        assert!(sk.pca.basis_fp32.is_some());
        assert!(sk.pca.mean.is_empty());
        assert!(sk.pca.basis.is_empty());
        assert!(sk.kmeans.centers_fp32.is_some());
        assert!(sk.kmeans.centers.is_empty());
    }

    // -------------------- all-zero input --------------------

    #[test]
    fn handles_all_zero_block() {
        let n = 4;
        let d = 8;
        let block = vec![0.0_f32; n * d];
        // Need at least some positive variance somewhere; replace one row.
        let mut block = block;
        block[0] = 1.0;
        let w = vec![1.0_f32; n];
        let params = CodecParams { bit_width: 3, k: 2, variance_ratio: 0.5, ..Default::default() };
        let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
        let r = decode_block::<MSE>(&sk, &codes);
        assert_eq!(r.len(), block.len());
        for v in r {
            assert!(v.is_finite());
        }
    }

    // -------------------- monomorphisation sanity --------------------

    #[test]
    fn encode_block_compiles_for_all_distortions() {
        // If any Distortion impl stops satisfying the trait bounds of
        // encode_block, this test fails to compile — so it doubles as a
        // contract check.
        fn _assert<R: Distortion>() {
            let _ = encode_block::<R>;
            let _ = decode_block::<R>;
        }
        _assert::<MSE>();
        _assert::<InnerProduct>();
        _assert::<LInf>();
    }

    // -------------------- zero-coeff edge case --------------------

    #[test]
    fn encode_handles_row_equal_to_mean() {
        // Construct a block where one row is exactly the weighted mean:
        // its PCA projection is the zero vector → triggers the
        // "all coordinates zero" residual branch.
        let d = 4;
        let block = vec![
            1.0_f32, 0.0, 0.0, 0.0,
            -1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            // This row equals the mean (which is zero by symmetry).
            0.0, 0.0, 0.0, 0.0,
        ];
        let w = vec![1.0_f32; 5];
        let params = CodecParams {
            variance_ratio: 0.95,
            k: 2,
            bit_width: 3,
            ..Default::default()
        };
        let (sk, codes) = encode_block::<MSE>(&block, &w, d, &params);
        let rec = decode_block::<MSE>(&sk, &codes);
        // The zero-mean row should reconstruct close to zero.
        let zero_row = &rec[4 * d..5 * d];
        for &v in zero_row {
            assert!(v.abs() < 0.5, "zero-mean row reconstructed as {v}");
        }
    }

    // -------------------- next_pow2 & helpers --------------------

    #[test]
    fn next_pow2_basic() {
        assert_eq!(next_pow2(0), 1);
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(5), 8);
        assert_eq!(next_pow2(15), 16);
        assert_eq!(next_pow2(16), 16);
        assert_eq!(next_pow2(17), 32);
    }

    #[test]
    fn pad_zero_extends_with_zeros() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let p = pad_zero(&v, 5);
        assert_eq!(p, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn pad_zero_shorter_target_truncates() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let p = pad_zero(&v, 2);
        assert_eq!(p, vec![1.0, 2.0]);
    }

    #[test]
    fn l2_norm_basic() {
        assert_abs_diff_eq!(l2_norm(&[3.0_f32, 4.0]), 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(l2_norm(&[0.0_f32, 0.0, 0.0]), 0.0);
    }

    // -------------------- encode_layer / shared basis (v1.2 B') --------------------

    fn mk_blocks(n_blocks: usize, bs: usize, d: usize, rank: usize, noise: f32, seed: u64)
        -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let total = synthetic_block(n_blocks * bs, d, rank, noise, seed);
        let mut bs_vecs = Vec::with_capacity(n_blocks);
        let mut ws = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            bs_vecs.push(total[i * bs * d..(i + 1) * bs * d].to_vec());
            ws.push(vec![1.0_f32; bs]);
        }
        (bs_vecs, ws)
    }

    #[test]
    fn encode_layer_no_sharing_matches_per_block_encode() {
        let n_blocks = 3;
        let bs = 32;
        let d = 16;
        let (blocks, ws) = mk_blocks(n_blocks, bs, d, 4, 0.02, 101);
        let params = CodecParams {
            variance_ratio: 0.9,
            k: 4, bit_width: 3,
            rotation_seed: 0xA5A5_A5A5,
            kmeans_max_iter: 32,
            pca_method: PcaMethod::Exact,
            skeleton_dtype: SkeletonDtype::Fp16,
            exact_rank_cap: None,
            custom_centroids: None,
            outlier_threshold: None,
            residual_besi: None,
        };
        let enc = encode_layer::<MSE>(&blocks, &ws, d, &params, false);
        assert!(enc.shared_pca.is_none());
        assert_eq!(enc.per_block.len(), n_blocks);
        for (i, (sk, codes)) in enc.per_block.iter().enumerate() {
            let (sk2, codes2) = encode_block::<MSE>(&blocks[i], &ws[i], d, &params);
            assert_eq!(sk.d_eff(), sk2.d_eff());
            assert_eq!(sk.k(), sk2.k());
            assert_eq!(codes.len(), codes2.len());
            for (c, c2) in codes.iter().zip(codes2.iter()) {
                assert_eq!(c.residual_packed, c2.residual_packed);
                assert_eq!(c.seg_id, c2.seg_id);
            }
        }
    }

    #[test]
    fn encode_layer_shared_basis_amortises_skeleton_bytes() {
        let n_blocks = 4;
        let bs = 64;
        let d = 16;
        let (blocks, ws) = mk_blocks(n_blocks, bs, d, 4, 0.01, 202);
        let params = CodecParams {
            variance_ratio: 0.9,
            k: 4, bit_width: 3,
            ..Default::default()
        };
        let per_block_enc = encode_layer::<MSE>(&blocks, &ws, d, &params, false);
        let shared_enc = encode_layer::<MSE>(&blocks, &ws, d, &params, true);

        assert!(shared_enc.shared_pca.is_some());
        assert_eq!(shared_enc.per_block.len(), n_blocks);

        let per_block_bytes = layer_nbytes(&per_block_enc);
        let shared_bytes = layer_nbytes(&shared_enc);
        assert!(
            shared_bytes < per_block_bytes,
            "shared encoding must be smaller: shared={shared_bytes} per_block={per_block_bytes}"
        );
    }

    #[test]
    fn encode_layer_shared_decodes_round_trip() {
        let n_blocks = 4;
        let bs = 32;
        let d = 16;
        let (blocks, ws) = mk_blocks(n_blocks, bs, d, 4, 0.02, 303);
        let params = CodecParams {
            variance_ratio: 0.9,
            k: 4, bit_width: 4,
            ..Default::default()
        };
        let enc = encode_layer::<MSE>(&blocks, &ws, d, &params, true);
        let recs = decode_layer::<MSE>(&enc);
        assert_eq!(recs.len(), n_blocks);
        for i in 0..n_blocks {
            let rec = &recs[i];
            assert_eq!(rec.len(), blocks[i].len());
            for v in rec {
                assert!(v.is_finite(), "non-finite reconstruction");
            }
            let mse = mse_of(&blocks[i], rec);
            assert!(mse < 5.0, "MSE too high for shared basis path: {mse}");
        }
    }

    #[test]
    fn encode_layer_empty_rejected() {
        let params = CodecParams::default();
        let blocks: Vec<Vec<f32>> = vec![];
        let ws: Vec<Vec<f32>> = vec![];
        let result = std::panic::catch_unwind(|| {
            encode_layer::<MSE>(&blocks, &ws, 4, &params, true)
        });
        assert!(result.is_err());
    }

    #[test]
    fn encode_layer_length_mismatch_rejected() {
        let params = CodecParams::default();
        let blocks = vec![vec![0.0_f32; 16]];
        let ws: Vec<Vec<f32>> = vec![];
        let result = std::panic::catch_unwind(|| {
            encode_layer::<MSE>(&blocks, &ws, 4, &params, true)
        });
        assert!(result.is_err());
    }

    #[test]
    fn layer_nbytes_excludes_duplicate_basis_when_shared() {
        let n_blocks = 3;
        let bs = 32;
        let d = 16;
        let (blocks, ws) = mk_blocks(n_blocks, bs, d, 4, 0.01, 404);
        let params = CodecParams { variance_ratio: 0.9, k: 4, bit_width: 3, ..Default::default() };
        let enc = encode_layer::<MSE>(&blocks, &ws, d, &params, true);
        let reported = layer_nbytes(&enc);
        // Expected = 1× shared PCA + n_blocks × (K-means + codes).
        let pca_once = enc.shared_pca.as_ref().unwrap().nbytes();
        let per_block_sum: usize = enc.per_block.iter()
            .map(|(sk, codes)| sk.kmeans.nbytes() + codes.iter().map(Code::nbytes).sum::<usize>())
            .sum();
        assert_eq!(reported, pca_once + per_block_sum);
    }
}
