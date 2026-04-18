//! # KakeyaTurbo
//!
//! A monomorphic, zero-dispatch implementation of the unified
//! rate-distortion compressor discussed in the repository design notes.
//!
//! The algorithm is a two-stage approximation to Shannon's weighted
//! rate-distortion optimization:
//!
//! 1. **Stage 1 (structure extraction, "Kakeya"):** per-block data-adaptive
//!    transform: weighted mean, weighted PCA truncated at `d_eff`,
//!    temporal direction extraction, weighted spherical K-means on
//!    the perpendicular component.
//!
//! 2. **Stage 2 (residual coding, "TurboQuant"):** Walsh-Hadamard
//!    rotation with random sign flips (Gaussianization of the
//!    intra-cluster residual) followed by Lloyd-Max optimal scalar
//!    quantization on the rotated coordinates.
//!
//! All "attention-awareness" (metric choice, per-vector weighting,
//! norm precision) is expressed as parameters `(rho, w)` of a single
//! `encode_block` function — not as plugins or extension points.
//!
//! ## Design contract
//!
//! - `unsafe` is forbidden (`#![forbid(unsafe_code)]`)
//! - `dyn Trait` is not used anywhere
//! - `Box<dyn ...>` is not used anywhere
//! - Every hot path is monomorphized via generics + zero-sized types
//! - All const-generic bounds are compile-time enforced
//!
//! ## Modules
//!
//! - [`distortion`] — `Distortion` trait + concrete zero-sized types
//! - [`skeleton`] — block-level metadata (mean, basis, centers, rotation)
//! - [`wht`] — Walsh-Hadamard transform with sign-flip randomization
//! - [`quantize`] — Lloyd-Max scalar quantizer + bit packing
//! - [`kmeans`] — weighted spherical K-means
//! - [`pca`] — weighted PCA truncated at `d_eff`
//! - [`codec`] — the top-level `encode_block` / `decode_block`

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod codec;
pub mod distortion;
pub mod kmeans;
pub mod pca;
pub mod quantize;
pub mod skeleton;
pub mod wht;

pub use codec::{decode_block, encode_block, Code, CodecParams};
pub use distortion::{Distortion, InnerProduct, LInf, NormMode, MSE};
pub use skeleton::Skeleton;
