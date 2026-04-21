//! # Besicovitch-product codec
//!
//! An alternative skeleton construction for the Kakeya-like set used by
//! the codec.  Instead of fitting a per-block data-adaptive PCA (or RSVD)
//! and storing mean + basis, this codec:
//!
//! 1. Splits each D-dimensional vector into `G = D / g` contiguous groups
//!    of `g` coordinates (`g = 2` by default).
//! 2. Quantizes each group independently against a global, fixed
//!    Besicovitch-like direction codebook of `M = 2^direction_bits`
//!    unit directions on `S^(g−1)`.  For `g = 2` the codebook is the
//!    uniform angular grid on the unit circle.
//! 3. Stores a per-group scalar magnitude `α_k` as f16 (or as a
//!    `magnitude_bits`-bit Lloyd-Max index against a calibrated codebook,
//!    with a separate f16 per-group norm).
//!
//! Reconstruction:
//!     x̂_k = α_k · dir_{id_k}
//! where `dir_{id_k}` is the `id_k`-th entry of the direction codebook
//! for group `k`.
//!
//! ## Why this is a "Besicovitch construction + product"
//!
//! Fix the direction codebook.  The set of all possible reconstructions
//!   B = { concat_{k=1..G} α_k · dir_{id_k} : id_k ∈ [M], α_k ∈ R }
//! is a finite union of `M^G` affine one-dimensional lines through the
//! origin — an unambiguous **Kakeya-like** set of dimension 1 in R^D
//! that contains complete affine lines in each selected direction.
//! In the language of harmonic analysis this is the *product* of a
//! Besicovitch-style low-dimensional Kakeya construction across the G
//! groups.
//!
//! The construction is entirely **non-data-adaptive**: the codebook is
//! fixed globally, so there is *no* per-block skeleton to store.
//! This is the fundamental difference with PCA/RSVD skeletons, which
//! need O(D² + D · d_eff) storage per block for mean + basis.
//!
//! ## Bit budget per vector
//!
//! `per_vec_bits = G · (direction_bits + magnitude_bits_effective)`
//!
//! where `magnitude_bits_effective = 16` in the f16-magnitude path and
//! `= magnitude_bits` in the Lloyd-Max path (plus one shared f16 per
//! vector to carry the residual scale).

#![allow(clippy::module_name_repetitions)]

use half::f16;

/// How a group's magnitude α is encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagnitudeMode {
    /// Store the raw projection `α` as a single f16 per group.
    /// Simple, lossy only at f16 precision, but costs 16 bits/group.
    F16,
    /// Quantize `α` with `magnitude_bits` bits against a global
    /// unit-variance Lloyd-Max codebook, and carry a per-vector f16
    /// scale `s` so that `α_k ≈ s · centroid[idx_k]`.  Cheaper per
    /// group (typically 3–4 bits) at the cost of one extra f16 per
    /// vector and some quantization error on α.
    QuantizedWithPerVectorScale,
}

/// Besicovitch-product codec parameters.
#[derive(Debug, Clone)]
pub struct BesicovitchParams {
    /// Group size `g`.  `D` must be divisible by `g`.  Default `g = 2`
    /// (projection onto the unit circle).
    pub group_size: usize,
    /// Bits per direction index.  Total codebook size `M = 2^direction_bits`.
    pub direction_bits: u8,
    /// Bits per magnitude (only meaningful when
    /// `magnitude_mode == QuantizedWithPerVectorScale`).
    pub magnitude_bits: u8,
    /// Magnitude encoding mode.
    pub magnitude_mode: MagnitudeMode,
    /// If true, subtract the per-block mean before projecting onto
    /// direction codebook and add it back on decode. Costs `D`
    /// f16 floats (= `2D` bytes) of per-block skeleton but recovers
    /// the ability to handle non-zero-mean data (e.g. K-cache layer 0).
    pub subtract_mean: bool,
}

impl Default for BesicovitchParams {
    fn default() -> Self {
        Self {
            group_size: 2,
            direction_bits: 6,
            magnitude_bits: 4,
            magnitude_mode: MagnitudeMode::F16,
            subtract_mean: false,
        }
    }
}

/// Per-vector encoded representation.
#[derive(Debug, Clone, PartialEq)]
pub struct BesicovitchCode {
    /// Direction indices, one per group.  Each fits in `direction_bits`
    /// bits; stored here unpacked for simplicity.  Byte accounting
    /// uses the packed size (see [`BesicovitchCode::nbytes`]).
    pub direction_ids: Vec<u32>,
    /// Encoded magnitudes.  In `F16` mode this is one `f16` per group.
    /// In `QuantizedWithPerVectorScale` mode it is a bit-packed table
    /// of `magnitude_bits` indices.
    pub magnitudes: MagnitudePayload,
}

/// Magnitude payload variants.
#[derive(Debug, Clone, PartialEq)]
pub enum MagnitudePayload {
    /// One f16 per group.
    F16(Vec<f16>),
    /// `magnitude_bits`-bit packed indices + one f16 scale.
    QuantizedWithScale {
        /// Bit-packed `magnitude_bits`-bit indices, one per group.
        packed_indices: Vec<u8>,
        /// Per-vector scale factor `s` (f16).
        scale: f16,
    },
}

impl BesicovitchCode {
    /// Total byte size of this code's payload.
    ///
    /// - `direction_ids`: `⌈G · direction_bits / 8⌉` bytes
    /// - `magnitudes`:
    ///     - `F16` → `G · 2` bytes
    ///     - `QuantizedWithScale` → `⌈G · magnitude_bits / 8⌉ + 2` bytes
    #[must_use]
    pub fn nbytes(&self, direction_bits: u8) -> usize {
        let g = self.direction_ids.len();
        let dir_bits_total = g * direction_bits as usize;
        let dir_bytes = dir_bits_total.div_ceil(8);
        let mag_bytes = match &self.magnitudes {
            MagnitudePayload::F16(v) => v.len() * 2,
            MagnitudePayload::QuantizedWithScale { packed_indices, .. } => {
                packed_indices.len() + 2
            }
        };
        dir_bytes + mag_bytes
    }
}

/// A globally shared, fixed Besicovitch-like direction codebook for one
/// group of size `g`.  For `g = 2` this is the uniform angular grid on
/// `[0, 2π)`; for `g ≥ 3` we use a deterministic pseudo-random
/// equidistribution on the unit sphere (Fibonacci-like).
#[derive(Debug, Clone)]
pub struct DirectionCodebook {
    /// Group size `g`.
    pub g: usize,
    /// Number of directions `M = 2^direction_bits`.
    pub m: usize,
    /// Flattened directions: `M × g` unit vectors stored row-major.
    pub directions: Vec<f32>,
}

impl DirectionCodebook {
    /// Build a codebook for group size `g` and `M = 2^direction_bits`
    /// directions.
    ///
    /// - `g == 2`: uniform angular grid `θ_i = π · i / M` (half circle;
    ///   the other half is covered by magnitude sign).
    /// - `g >= 3`: spherical Fibonacci lattice on the upper hemisphere
    ///   (normals with positive leading coordinate to resolve the ±
    ///   sign ambiguity).
    #[must_use]
    pub fn build(g: usize, direction_bits: u8) -> Self {
        assert!(g >= 2 && g <= 8, "unsupported group_size {g}");
        assert!(direction_bits >= 1 && direction_bits <= 10,
                "direction_bits must be 1..=10, got {direction_bits}");
        let m = 1usize << direction_bits;
        let mut directions = Vec::with_capacity(m * g);
        if g == 2 {
            for i in 0..m {
                let theta = std::f32::consts::PI * (i as f32) / (m as f32);
                directions.push(theta.cos());
                directions.push(theta.sin());
            }
        } else {
            // Spherical Fibonacci on the upper hemisphere in R^g.
            // We generate points on S^(g-1) and force positive leading
            // coordinate.  Not optimal for g ≥ 4 but workable.
            let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
            for i in 0..m {
                let mut coords = vec![0.0_f32; g];
                let t = (i as f32 + 0.5) / m as f32;
                let z = 1.0 - 2.0 * t;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let a = 2.0 * std::f32::consts::PI * (i as f32) / phi;
                coords[0] = z.abs(); // force positive leading coord
                if g >= 2 {
                    coords[1] = r * a.cos();
                }
                if g >= 3 {
                    coords[2] = r * a.sin();
                }
                for k in 3..g {
                    let b = 2.0 * std::f32::consts::PI * (i as f32) * (k as f32 + 1.0) / phi;
                    coords[k] = 0.1 * b.cos();
                }
                // Normalize.
                let n: f32 = coords.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                for c in &mut coords {
                    *c /= n;
                }
                directions.extend_from_slice(&coords);
            }
        }
        Self { g, m, directions }
    }

    /// Return the i-th direction as a slice of length `g`.
    #[must_use]
    pub fn direction(&self, i: usize) -> &[f32] {
        &self.directions[i * self.g..(i + 1) * self.g]
    }

    /// Given a group vector `x ∈ R^g`, find the direction index `i*`
    /// that maximizes `<x, d_i>` in absolute value (signed projection)
    /// and return `(i*, signed_magnitude)` where
    /// `signed_magnitude = <x, d_{i*}>`.
    ///
    /// This is the orthogonal projection onto the nearest line in the
    /// direction codebook (the line through `±d_i`).  The closest-line
    /// criterion minimizes reconstruction MSE over the choice of
    /// direction.
    #[must_use]
    pub fn assign(&self, x: &[f32]) -> (u32, f32) {
        debug_assert_eq!(x.len(), self.g);
        let mut best_i = 0u32;
        let mut best_abs_proj = 0.0_f32;
        let mut best_signed_proj = 0.0_f32;
        for i in 0..self.m {
            let d = self.direction(i);
            let proj: f32 = x.iter().zip(d).map(|(a, b)| a * b).sum();
            let ap = proj.abs();
            if ap > best_abs_proj {
                best_abs_proj = ap;
                best_signed_proj = proj;
                best_i = i as u32;
            }
        }
        (best_i, best_signed_proj)
    }
}

/// Per-block skeleton for the Besicovitch-product codec.
///
/// When `subtract_mean` is enabled, the encoded block carries the
/// block-mean as a `D`-length f16 vector.  Otherwise the skeleton is
/// empty (the direction codebook is deterministic and shared globally).
#[derive(Debug, Clone, PartialEq)]
pub struct BesicovitchSkeleton {
    /// Per-block mean (fp16), empty if `subtract_mean == false`.
    pub mean: Vec<f16>,
}

impl BesicovitchSkeleton {
    /// Byte size.
    #[must_use]
    pub fn nbytes(&self) -> usize {
        self.mean.len() * 2
    }
}

/// Encode a single block of `n` vectors of dimension `D` using the
/// Besicovitch-product skeleton.
///
/// Panics if `D % params.group_size != 0`.
///
/// Returns `(codebook, skeleton, codes)`:
///   - `codebook` is deterministic from `params` (receivers can
///     reconstruct it from params alone).
///   - `skeleton` is per-block (may be empty if `subtract_mean == false`).
pub fn encode_block_full(
    vectors: &[f32],
    d: usize,
    params: &BesicovitchParams,
) -> (DirectionCodebook, BesicovitchSkeleton, Vec<BesicovitchCode>) {
    assert!(d % params.group_size == 0,
            "D={d} must be divisible by group_size={}", params.group_size);
    let n = vectors.len() / d;
    let g = params.group_size;
    let groups = d / g;
    let codebook = DirectionCodebook::build(g, params.direction_bits);

    let mean_f32: Vec<f32> = if params.subtract_mean && n > 0 {
        let mut m = vec![0.0_f32; d];
        for i in 0..n {
            for j in 0..d {
                m[j] += vectors[i * d + j];
            }
        }
        for v in &mut m {
            *v /= n as f32;
        }
        m
    } else {
        Vec::new()
    };

    let skeleton = BesicovitchSkeleton {
        mean: mean_f32.iter().map(|&v| f16::from_f32(v)).collect(),
    };
    // Reconstruct the mean we actually store (post-f16 round-trip) so
    // encode and decode use identical numbers.
    let mean_effective: Vec<f32> = skeleton.mean.iter().map(|v| v.to_f32()).collect();

    let mut codes = Vec::with_capacity(n);
    for i in 0..n {
        let x = &vectors[i * d..(i + 1) * d];
        let mut direction_ids = Vec::with_capacity(groups);
        let mut raw_magnitudes = Vec::with_capacity(groups);
        for k in 0..groups {
            let mut group = [0.0_f32; 8];
            let gs = &mut group[..g];
            for j in 0..g {
                let v = x[k * g + j];
                gs[j] = if params.subtract_mean {
                    v - mean_effective[k * g + j]
                } else {
                    v
                };
            }
            let (id, alpha) = codebook.assign(gs);
            direction_ids.push(id);
            raw_magnitudes.push(alpha);
        }
        let magnitudes = match params.magnitude_mode {
            MagnitudeMode::F16 => MagnitudePayload::F16(
                raw_magnitudes.iter().map(|&a| f16::from_f32(a)).collect(),
            ),
            MagnitudeMode::QuantizedWithPerVectorScale => {
                // Unit-variance Lloyd-Max centroids on the signed
                // magnitudes, with per-vector scale = max |α_k|.
                let scale = raw_magnitudes
                    .iter()
                    .fold(0.0_f32, |m, &v| m.max(v.abs()))
                    .max(1e-12);
                let bits = params.magnitude_bits;
                let centroids = crate::quantize::centroids_gaussian(bits);
                // But our magnitudes aren't unit-Gaussian — they're
                // bounded by `scale`.  Map to [-1, 1] by dividing by
                // `scale`, then snap to nearest centroid (which covers
                // roughly [-3, 3] for Gaussian).  Good enough for
                // bit_width ≥ 3.
                let indices: Vec<u8> = raw_magnitudes
                    .iter()
                    .map(|&a| {
                        let u = a / scale;
                        nearest_centroid_idx(&centroids, u)
                    })
                    .collect();
                let packed = crate::quantize::pack_bits(&indices, bits);
                MagnitudePayload::QuantizedWithScale {
                    packed_indices: packed,
                    scale: f16::from_f32(scale),
                }
            }
        };
        codes.push(BesicovitchCode {
            direction_ids,
            magnitudes,
        });
    }
    (codebook, skeleton, codes)
}

/// Backwards-compatible wrapper: returns just `(codebook, codes)`,
/// using an empty skeleton (no mean subtraction).
pub fn encode_block(
    vectors: &[f32],
    d: usize,
    params: &BesicovitchParams,
) -> (DirectionCodebook, Vec<BesicovitchCode>) {
    let (cb, _sk, codes) = encode_block_full(vectors, d, params);
    (cb, codes)
}

/// Encode a **single** vector of length `d` against the given codebook.
///
/// Used internally by the kakeyaturbo codec to swap Lloyd-Max scalar
/// quantization of the WHT-ed residual for a Besicovitch-product
/// code.  No mean subtraction is performed (the residual is already
/// zero-mean after PCA + k-means).
pub fn encode_vector(
    vector: &[f32],
    codebook: &DirectionCodebook,
    params: &BesicovitchParams,
) -> BesicovitchCode {
    assert_eq!(vector.len() % params.group_size, 0);
    let g = params.group_size;
    let groups = vector.len() / g;
    let mut direction_ids = Vec::with_capacity(groups);
    let mut raw_magnitudes = Vec::with_capacity(groups);
    for k in 0..groups {
        let group = &vector[k * g..(k + 1) * g];
        let (id, alpha) = codebook.assign(group);
        direction_ids.push(id);
        raw_magnitudes.push(alpha);
    }
    let magnitudes = match params.magnitude_mode {
        MagnitudeMode::F16 => {
            MagnitudePayload::F16(raw_magnitudes.iter().map(|&a| f16::from_f32(a)).collect())
        }
        MagnitudeMode::QuantizedWithPerVectorScale => {
            let scale = raw_magnitudes
                .iter()
                .fold(0.0_f32, |m, &v| m.max(v.abs()))
                .max(1e-12);
            let bits = params.magnitude_bits;
            let centroids = crate::quantize::centroids_gaussian(bits);
            let indices: Vec<u8> = raw_magnitudes
                .iter()
                .map(|&a| nearest_centroid_idx(centroids, a / scale))
                .collect();
            let packed = crate::quantize::pack_bits(&indices, bits);
            MagnitudePayload::QuantizedWithScale {
                packed_indices: packed,
                scale: f16::from_f32(scale),
            }
        }
    };
    BesicovitchCode { direction_ids, magnitudes }
}

/// Decode a **single** vector's Besicovitch code back to a vector of
/// length `d = groups * g`.
#[must_use]
pub fn decode_vector(
    code: &BesicovitchCode,
    codebook: &DirectionCodebook,
    params: &BesicovitchParams,
) -> Vec<f32> {
    let g = params.group_size;
    let groups = code.direction_ids.len();
    let alphas: Vec<f32> = match &code.magnitudes {
        MagnitudePayload::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
        MagnitudePayload::QuantizedWithScale { packed_indices, scale } => {
            let bits = params.magnitude_bits;
            let indices = crate::quantize::unpack_bits(packed_indices, bits, groups);
            let centroids = crate::quantize::centroids_gaussian(bits);
            let s = scale.to_f32();
            indices.iter().map(|&i| centroids[i as usize] * s).collect()
        }
    };
    let mut out = Vec::with_capacity(groups * g);
    for k in 0..groups {
        let id = code.direction_ids[k] as usize;
        let d_vec = codebook.direction(id);
        let alpha = alphas[k];
        for j in 0..g {
            out.push(alpha * d_vec[j]);
        }
    }
    out
}

/// Pack a per-vector [`BesicovitchCode`] into a contiguous byte buffer
/// for storage alongside other codec metadata.  Layout:
///
/// 1. Direction indices: `⌈G · direction_bits / 8⌉` packed bytes
/// 2. Magnitude payload:
///    - `F16`: `G × 2` bytes
///    - `QuantizedWithScale`: `⌈G · magnitude_bits / 8⌉ + 2` bytes
pub fn serialize_code(code: &BesicovitchCode, params: &BesicovitchParams) -> Vec<u8> {
    let groups = code.direction_ids.len();
    let indices: Vec<u8> = code.direction_ids.iter().map(|&id| id as u8).collect();
    // For direction_bits > 8 we'd need u16 packing; for the ranges
    // we use (d ∈ [1,10]) u8 is fine for ≤8 bits.  Assert this.
    assert!(params.direction_bits <= 8,
            "serialize_code currently assumes direction_bits ≤ 8, got {}",
            params.direction_bits);
    let mut out = crate::quantize::pack_bits(&indices, params.direction_bits);
    match &code.magnitudes {
        MagnitudePayload::F16(vals) => {
            for v in vals {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        MagnitudePayload::QuantizedWithScale { packed_indices, scale } => {
            out.extend_from_slice(packed_indices);
            out.extend_from_slice(&scale.to_le_bytes());
        }
    }
    let _ = groups;
    out
}

/// Inverse of [`serialize_code`].  Requires `groups` to reconstruct
/// the unpacked index/magnitude count.
pub fn deserialize_code(
    bytes: &[u8],
    groups: usize,
    params: &BesicovitchParams,
) -> BesicovitchCode {
    assert!(params.direction_bits <= 8,
            "deserialize_code currently assumes direction_bits ≤ 8, got {}",
            params.direction_bits);
    let dir_bits = groups * params.direction_bits as usize;
    let dir_bytes = dir_bits.div_ceil(8);
    let (dir_slice, rest) = bytes.split_at(dir_bytes);
    let indices = crate::quantize::unpack_bits(dir_slice, params.direction_bits, groups);
    let direction_ids: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
    let magnitudes = match params.magnitude_mode {
        MagnitudeMode::F16 => {
            let mut v = Vec::with_capacity(groups);
            for k in 0..groups {
                let lo = rest[k * 2];
                let hi = rest[k * 2 + 1];
                v.push(f16::from_le_bytes([lo, hi]));
            }
            MagnitudePayload::F16(v)
        }
        MagnitudeMode::QuantizedWithPerVectorScale => {
            let mag_bits = groups * params.magnitude_bits as usize;
            let mag_bytes = mag_bits.div_ceil(8);
            let packed_indices = rest[..mag_bytes].to_vec();
            let scale_bytes = [rest[mag_bytes], rest[mag_bytes + 1]];
            let scale = f16::from_le_bytes(scale_bytes);
            MagnitudePayload::QuantizedWithScale {
                packed_indices,
                scale,
            }
        }
    };
    BesicovitchCode { direction_ids, magnitudes }
}

/// Byte size of the serialized form (same accounting as
/// [`BesicovitchCode::nbytes`]).
#[must_use]
pub fn serialized_nbytes(groups: usize, params: &BesicovitchParams) -> usize {
    let dir_bytes = (groups * params.direction_bits as usize).div_ceil(8);
    let mag_bytes = match params.magnitude_mode {
        MagnitudeMode::F16 => groups * 2,
        MagnitudeMode::QuantizedWithPerVectorScale => {
            (groups * params.magnitude_bits as usize).div_ceil(8) + 2
        }
    };
    dir_bytes + mag_bytes
}

/// Decode a block back to `n × D` reconstructed vectors.  If the
/// encoder used `subtract_mean`, pass the same skeleton here.  If not,
/// pass an empty skeleton.
#[must_use]
pub fn decode_block_full(
    codebook: &DirectionCodebook,
    skeleton: &BesicovitchSkeleton,
    codes: &[BesicovitchCode],
    d: usize,
    params: &BesicovitchParams,
) -> Vec<f32> {
    let g = codebook.g;
    assert_eq!(g, params.group_size);
    let groups = d / g;
    let mean: Vec<f32> = skeleton.mean.iter().map(|v| v.to_f32()).collect();
    let mut out = Vec::with_capacity(codes.len() * d);
    for code in codes {
        debug_assert_eq!(code.direction_ids.len(), groups);
        let alphas: Vec<f32> = match &code.magnitudes {
            MagnitudePayload::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
            MagnitudePayload::QuantizedWithScale { packed_indices, scale } => {
                let bits = params.magnitude_bits;
                let indices = crate::quantize::unpack_bits(packed_indices, bits, groups);
                let centroids = crate::quantize::centroids_gaussian(bits);
                let s = scale.to_f32();
                indices
                    .iter()
                    .map(|&i| centroids[i as usize] * s)
                    .collect()
            }
        };
        for k in 0..groups {
            let id = code.direction_ids[k] as usize;
            let d_vec = codebook.direction(id);
            let alpha = alphas[k];
            for j in 0..g {
                let base = if params.subtract_mean {
                    mean[k * g + j]
                } else {
                    0.0
                };
                out.push(base + alpha * d_vec[j]);
            }
        }
    }
    out
}

/// Backwards-compatible wrapper: assumes no skeleton (no mean subtraction).
pub fn decode_block(
    codebook: &DirectionCodebook,
    codes: &[BesicovitchCode],
    d: usize,
    params: &BesicovitchParams,
) -> Vec<f32> {
    let empty = BesicovitchSkeleton { mean: Vec::new() };
    decode_block_full(codebook, &empty, codes, d, params)
}

fn nearest_centroid_idx(centroids: &[f32], v: f32) -> u8 {
    let mut best_i = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, &c) in centroids.iter().enumerate() {
        let dd = (v - c).abs();
        if dd < best_d {
            best_d = dd;
            best_i = i;
        }
    }
    best_i as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mse(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len() as f32;
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / n
    }

    #[test]
    fn codebook_g2_is_unit_circle() {
        let cb = DirectionCodebook::build(2, 4);
        assert_eq!(cb.m, 16);
        for i in 0..cb.m {
            let d = cb.direction(i);
            let norm2 = d[0] * d[0] + d[1] * d[1];
            assert!((norm2 - 1.0).abs() < 1e-5, "direction {} not unit: {}", i, norm2);
        }
    }

    #[test]
    fn codebook_g3_directions_are_unit_vectors() {
        let cb = DirectionCodebook::build(3, 5);
        for i in 0..cb.m {
            let d = cb.direction(i);
            let norm2: f32 = d.iter().map(|x| x * x).sum();
            assert!((norm2 - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn assign_returns_signed_projection_onto_nearest_line() {
        let cb = DirectionCodebook::build(2, 6); // 64 directions
        // A vector aligned with theta=0 → should pick id=0, alpha≈‖x‖.
        let x = vec![3.0_f32, 0.0];
        let (id, alpha) = cb.assign(&x);
        assert_eq!(id, 0);
        assert!((alpha - 3.0).abs() < 1e-4);

        // Vector aligned with theta=π → alpha should be negative (d_0 = (1,0),
        // projection of (-2.5, 0) onto d_0 is -2.5, nearest line remains id=0).
        let x = vec![-2.5_f32, 0.0];
        let (id, alpha) = cb.assign(&x);
        assert_eq!(id, 0);
        assert!((alpha - (-2.5)).abs() < 1e-4);
    }

    #[test]
    fn round_trip_lossless_on_codebook_aligned_vectors() {
        // If the data lies exactly on codebook directions, reconstruction
        // is perfect up to f16 precision.
        let params = BesicovitchParams {
            group_size: 2,
            direction_bits: 4,
            magnitude_bits: 0,
            magnitude_mode: MagnitudeMode::F16,
            subtract_mean: false,
        };
        let d = 8;
        let groups = d / 2;
        let cb = DirectionCodebook::build(2, params.direction_bits);
        // Build a vector whose groups are scalar multiples of codebook entries.
        let mut block = Vec::with_capacity(d);
        for k in 0..groups {
            let idx = (k * 3) % cb.m;
            let dir = cb.direction(idx);
            block.push(2.5_f32 * dir[0]);
            block.push(2.5_f32 * dir[1]);
        }
        let (cb2, codes) = encode_block(&block, d, &params);
        assert_eq!(cb2.m, cb.m);
        let rec = decode_block(&cb2, &codes, d, &params);
        let err = mse(&block, &rec);
        assert!(err < 1e-5, "expected near-zero MSE, got {err}");
    }

    #[test]
    fn round_trip_on_random_vectors_has_bounded_mse() {
        use rand::{Rng, SeedableRng};
        use rand::rngs::SmallRng;
        let mut rng = SmallRng::seed_from_u64(42);
        let n = 64;
        let d = 128;
        let mut block = vec![0.0_f32; n * d];
        for v in &mut block {
            *v = rng.r#gen::<f32>() * 2.0 - 1.0;
        }
        // 8-bit directions on g=2 → M=256 directions on circle.
        let params = BesicovitchParams {
            group_size: 2,
            direction_bits: 8,
            magnitude_bits: 0,
            magnitude_mode: MagnitudeMode::F16,
            subtract_mean: false,
        };
        let (cb, codes) = encode_block(&block, d, &params);
        let rec = decode_block(&cb, &codes, d, &params);
        let err = mse(&block, &rec);
        // Theoretical per-group MSE for uniform direction grid with M
        // directions on S^1 is bounded by E‖x‖² · (π²/(12 · M²)).
        // For uniform [-1,1], per-group E‖x‖² = 2/3; with M=256 the
        // bound is ~ 8.4e-6.  Loosen for finite-sample noise.
        assert!(err < 1e-3, "MSE should be small for M=256 directions, got {err}");
    }

    #[test]
    fn quantized_magnitude_mode_round_trip_works() {
        use rand::{Rng, SeedableRng};
        use rand::rngs::SmallRng;
        let mut rng = SmallRng::seed_from_u64(7);
        let n = 32;
        let d = 64;
        let mut block = vec![0.0_f32; n * d];
        for v in &mut block {
            *v = rng.r#gen::<f32>() * 4.0 - 2.0;
        }
        let params = BesicovitchParams {
            group_size: 2,
            direction_bits: 6,
            magnitude_bits: 4,
            magnitude_mode: MagnitudeMode::QuantizedWithPerVectorScale,
            subtract_mean: false,
        };
        let (cb, codes) = encode_block(&block, d, &params);
        let rec = decode_block(&cb, &codes, d, &params);
        let err = mse(&block, &rec);
        // Quantized magnitude mode should still give reasonable MSE:
        // b_mag=4 bits → ~0.03 relative-error per magnitude,
        // plus direction quantization error.
        assert!(err < 0.1, "quantized-magnitude MSE too large: {err}");
    }

    #[test]
    fn encode_vector_and_serialize_round_trip() {
        // encode_vector -> decode_vector must match block-style call.
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(314);
        let d = 64;
        let mut vec_in = vec![0.0_f32; d];
        for v in &mut vec_in {
            *v = rng.r#gen::<f32>() * 2.0 - 1.0;
        }
        let params = BesicovitchParams {
            group_size: 2, direction_bits: 5, magnitude_bits: 4,
            magnitude_mode: MagnitudeMode::QuantizedWithPerVectorScale,
            subtract_mean: false,
        };
        let cb = DirectionCodebook::build(params.group_size, params.direction_bits);
        let code = encode_vector(&vec_in, &cb, &params);
        let rec = decode_vector(&code, &cb, &params);
        assert_eq!(rec.len(), d);
        let e: f32 = vec_in.iter().zip(&rec).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / d as f32;
        assert!(e < 0.1, "round-trip MSE too large: {e}");
    }

    #[test]
    fn serialize_deserialize_round_trips_exactly() {
        // Serialization must be lossless at the code level.
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(271);
        for (db, mb, mode) in [
            (3_u8, 3_u8, MagnitudeMode::QuantizedWithPerVectorScale),
            (4, 4, MagnitudeMode::QuantizedWithPerVectorScale),
            (6, 0, MagnitudeMode::F16),
            (7, 0, MagnitudeMode::F16),
        ] {
            let params = BesicovitchParams {
                group_size: 2, direction_bits: db, magnitude_bits: mb,
                magnitude_mode: mode,
                subtract_mean: false,
            };
            let d = 32;
            let groups = d / params.group_size;
            let mut vec_in = vec![0.0_f32; d];
            for v in &mut vec_in {
                *v = rng.r#gen::<f32>() * 2.0 - 1.0;
            }
            let cb = DirectionCodebook::build(params.group_size, params.direction_bits);
            let code = encode_vector(&vec_in, &cb, &params);
            let bytes = serialize_code(&code, &params);
            let expected_n = serialized_nbytes(groups, &params);
            assert_eq!(bytes.len(), expected_n,
                       "serialized bytes mismatch for db={db} mb={mb:?}: got {} want {}",
                       bytes.len(), expected_n);
            let decoded = deserialize_code(&bytes, groups, &params);
            assert_eq!(code, decoded,
                       "serialize→deserialize mismatch for db={db} mb={mb:?}");
        }
    }

    #[test]
    fn subtract_mean_recovers_non_zero_mean_data() {
        // Data with large per-block mean (simulates K-cache L=0 situation):
        // mean=(10, -5, 10, -5, ...) with small per-vector jitter.
        use rand::{Rng, SeedableRng};
        use rand::rngs::SmallRng;
        let mut rng = SmallRng::seed_from_u64(99);
        let n = 64;
        let d = 16;
        let mut block = vec![0.0_f32; n * d];
        for i in 0..n {
            for j in 0..d {
                let mean = if j % 2 == 0 { 10.0_f32 } else { -5.0 };
                block[i * d + j] = mean + (rng.r#gen::<f32>() * 0.2 - 0.1);
            }
        }
        let params_off = BesicovitchParams {
            group_size: 2, direction_bits: 6, magnitude_bits: 0,
            magnitude_mode: MagnitudeMode::F16,
            subtract_mean: false,
        };
        let params_on = BesicovitchParams {
            group_size: 2, direction_bits: 6, magnitude_bits: 0,
            magnitude_mode: MagnitudeMode::F16,
            subtract_mean: true,
        };
        let (cb, sk_off, codes_off) = encode_block_full(&block, d, &params_off);
        let (_, sk_on, codes_on) = encode_block_full(&block, d, &params_on);
        let rec_off = decode_block_full(&cb, &sk_off, &codes_off, d, &params_off);
        let rec_on = decode_block_full(&cb, &sk_on, &codes_on, d, &params_on);
        let mse_off = mse(&block, &rec_off);
        let mse_on = mse(&block, &rec_on);
        assert!(sk_off.nbytes() == 0);
        assert!(sk_on.nbytes() == d * 2, "skeleton should be {d} f16s", d = d);
        assert!(
            mse_on < mse_off * 0.01,
            "subtract_mean should dramatically cut MSE on biased data: off={mse_off:.3e}, on={mse_on:.3e}"
        );
    }

    #[test]
    fn nbytes_matches_expected_per_vector_budget() {
        let params = BesicovitchParams {
            group_size: 2,
            direction_bits: 6,
            magnitude_bits: 4,
            magnitude_mode: MagnitudeMode::F16,
            subtract_mean: false,
        };
        let d = 128;
        let groups = d / 2;
        let block = vec![0.0_f32; d];
        let (_, codes) = encode_block(&block, d, &params);
        let c = &codes[0];
        let nb = c.nbytes(params.direction_bits);
        // 64 groups × 6 bits = 384 bits = 48 bytes of directions
        // 64 groups × 2 bytes (f16) = 128 bytes of magnitudes
        assert_eq!(nb, 48 + groups * 2);
    }

    #[test]
    fn quantized_magnitude_nbytes_is_compact() {
        let params = BesicovitchParams {
            group_size: 2,
            direction_bits: 6,
            magnitude_bits: 3,
            magnitude_mode: MagnitudeMode::QuantizedWithPerVectorScale,
            subtract_mean: false,
        };
        let d = 128;
        let _groups = d / 2;
        let block = vec![0.0_f32; d];
        let (_, codes) = encode_block(&block, d, &params);
        let c = &codes[0];
        let nb = c.nbytes(params.direction_bits);
        // 64 × 6 bits = 384 bits = 48 bytes of directions
        // 64 × 3 bits = 192 bits = 24 bytes of magnitudes + 2-byte scale
        assert_eq!(nb, 48 + 24 + 2);
        // Per-vector bit cost = nb × 8 / groups = (48+24+2)×8/64 = 9.25 bits/group
        // vs PCA b=4 with d_eff=64 and per-block skeleton overhead amortized:
        // PCA b=4 = 4 bits/coord × 128 coords = 64 bytes plus skeleton.
        // Besicovitch: 74 bytes, no skeleton.  The skeleton tradeoff
        // depends on block size.  See besicovitch-bench.
        let _ = nb; // silence
    }
}
