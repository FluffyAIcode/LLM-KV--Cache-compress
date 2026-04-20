//! Stage-by-stage codec ablation.
//!
//! For a single input block, run the encode pipeline and emit the
//! reconstruction at four intermediate points:
//!
//!   stage_1_pca_only         = mu + U Uᵀ (x - mu)          (PCA projection only)
//!   stage_2_pca_plus_kmeans  = mu + U (t*c_seg + coeff_res) (PCA + kmeans center; NO residual quantization)
//!                              where coeff_res is the exact residual c - t*c_seg (not quantized/WHT'd)
//!   stage_3_full_minus_residual_quant = stage 2 but the exact residual r is passed through WHT then
//!                              inverse WHT without quantization (sanity check: should equal stage_2)
//!   stage_4_full             = the real codec decode (what decode_block does)
//!
//! Writes each reconstruction as KKTV so Python can compare.
//! This lets us attribute the 94% correlation loss to a specific stage.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use kakeyaturbo::codec::{CodecParams, PcaMethod};
use kakeyaturbo::distortion::{Distortion, NormMode, MSE};
use kakeyaturbo::kmeans::{assign_and_project, fit_spherical_kmeans, residual as kmeans_residual};
use kakeyaturbo::pca::{fit_weighted_pca, fit_weighted_pca_randomized, project, unproject};
use kakeyaturbo::quantize::{dequantize_vector, pack_bits, quantize_vector, unpack_bits};
use kakeyaturbo::wht::{inverse_rotate, rotate};

const MAGIC: u32 = 0x4B4B_5456;
const VERSION: u32 = 1;

struct Args {
    input: PathBuf,
    stage_out: [PathBuf; 4],
    block_size: usize,
    variance_ratio: f32,
    k: usize,
    bit_width: u8,
    rotation_seed: u32,
    pca_method: String,
    rsvd_target_rank: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().collect();
    let mut input: Option<PathBuf> = None;
    let mut stage1: Option<PathBuf> = None;
    let mut stage2: Option<PathBuf> = None;
    let mut stage3: Option<PathBuf> = None;
    let mut stage4: Option<PathBuf> = None;
    let mut block_size = 512;
    let mut variance_ratio: f32 = 0.95;
    let mut k = 16;
    let mut bit_width: u8 = 3;
    let mut rotation_seed: u32 = 0xCAFE_BABE;
    let mut pca_method = "exact".to_string();
    let mut rsvd_target_rank: Option<usize> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--input" => { i += 1; input = Some(PathBuf::from(&argv[i])); }
            "--stage1-out" => { i += 1; stage1 = Some(PathBuf::from(&argv[i])); }
            "--stage2-out" => { i += 1; stage2 = Some(PathBuf::from(&argv[i])); }
            "--stage3-out" => { i += 1; stage3 = Some(PathBuf::from(&argv[i])); }
            "--stage4-out" => { i += 1; stage4 = Some(PathBuf::from(&argv[i])); }
            "--block-size" => { i += 1; block_size = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "--variance-ratio" => { i += 1; variance_ratio = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "--k" => { i += 1; k = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "--bit-width" => { i += 1; bit_width = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "--rotation-seed" => { i += 1; rotation_seed = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "--pca-method" => { i += 1; pca_method = argv[i].clone(); }
            "--rsvd-target-rank" => { i += 1; rsvd_target_rank = Some(argv[i].parse().map_err(|e| format!("{e}"))?); }
            _ => return Err(format!("unknown flag {}", argv[i])),
        }
        i += 1;
    }
    Ok(Args {
        input: input.ok_or("--input required")?,
        stage_out: [
            stage1.ok_or("--stage1-out required")?,
            stage2.ok_or("--stage2-out required")?,
            stage3.ok_or("--stage3-out required")?,
            stage4.ok_or("--stage4-out required")?,
        ],
        block_size, variance_ratio, k, bit_width, rotation_seed,
        pca_method, rsvd_target_rank,
    })
}

fn read_tensor(path: &PathBuf) -> Result<(Vec<f32>, usize, usize), String> {
    let mut r = BufReader::new(File::open(path).map_err(|e| format!("open: {e}"))?);
    let mut buf4 = [0u8; 4]; let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf4).unwrap();
    assert_eq!(u32::from_le_bytes(buf4), MAGIC);
    r.read_exact(&mut buf4).unwrap();
    assert_eq!(u32::from_le_bytes(buf4), VERSION);
    r.read_exact(&mut buf8).unwrap();
    let n = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf4).unwrap();
    let d = u32::from_le_bytes(buf4) as usize;
    r.read_exact(&mut buf4).unwrap();
    let mut bytes = vec![0u8; n * d * 4];
    r.read_exact(&mut bytes).unwrap();
    let data: Vec<f32> = bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok((data, n, d))
}

fn write_tensor(path: &PathBuf, data: &[f32], n: usize, d: usize) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    w.write_all(&MAGIC.to_le_bytes()).unwrap();
    w.write_all(&VERSION.to_le_bytes()).unwrap();
    w.write_all(&(n as u64).to_le_bytes()).unwrap();
    w.write_all(&(d as u32).to_le_bytes()).unwrap();
    w.write_all(&0u32.to_le_bytes()).unwrap();
    for v in data {
        w.write_all(&v.to_le_bytes()).unwrap();
    }
}

fn next_pow2(n: usize) -> usize {
    if n <= 1 { 1 } else { n.next_power_of_two() }
}

fn pad_zero(v: &[f32], target: usize) -> Vec<f32> {
    let mut out = v.to_vec();
    out.resize(target, 0.0);
    out
}

fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn main() {
    let args = parse_args().unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(2) });
    let (data, n, d) = read_tensor(&args.input).unwrap();
    eprintln!("[stages] n={n} d={d} block_size={}", args.block_size);
    let bs = args.block_size;
    assert!(n % bs == 0, "n must be multiple of block_size");
    let n_blocks = n / bs;

    let params = CodecParams {
        variance_ratio: args.variance_ratio,
        k: args.k,
        bit_width: args.bit_width,
        rotation_seed: args.rotation_seed,
        kmeans_max_iter: 32,
        pca_method: match args.pca_method.as_str() {
            "exact" => PcaMethod::Exact,
            "randomized" => PcaMethod::Randomized {
                target_rank: args.rsvd_target_rank.unwrap_or((d / 2).max(8)),
                oversample: 8,
                power_iters: 2,
                seed_offset: 0x9E37_79B9_7F4A_7C15,
            },
            _ => panic!("bad --pca-method"),
        },
    };

    let weights = vec![1.0_f32; bs];
    let mut stage1_out = Vec::with_capacity(n * d);
    let mut stage2_out = Vec::with_capacity(n * d);
    let mut stage3_out = Vec::with_capacity(n * d);
    let mut stage4_out = Vec::with_capacity(n * d);

    for b in 0..n_blocks {
        let off = b * bs * d;
        let block = &data[off..off + bs * d];

        // --- Fit PCA (same path as encode_block) ---
        let pca = match params.pca_method {
            PcaMethod::Exact => fit_weighted_pca(block, &weights, d, params.variance_ratio),
            PcaMethod::Randomized { target_rank, oversample, power_iters, seed_offset } =>
                fit_weighted_pca_randomized(
                    block, &weights, d, params.variance_ratio,
                    target_rank.min(d), oversample, power_iters,
                    u64::from(params.rotation_seed) ^ seed_offset,
                ),
        };
        let d_eff = pca.d_eff;
        eprintln!("  block {b}: d_eff={d_eff}");

        // --- Project every vector, stage 1 reconstruction (PCA only) ---
        let mut coeffs = Vec::with_capacity(bs * d_eff);
        for i in 0..bs {
            let x = &block[i * d..(i + 1) * d];
            let c = project(x, &pca);
            let x_hat = unproject(&c, &pca);
            coeffs.extend_from_slice(&c);
            stage1_out.extend_from_slice(&x_hat);
        }

        // --- Fit k-means (exactly as encode_block) ---
        let valid_rows = (0..bs).filter(|&i| {
            weights[i] > 0.0
                && coeffs[i * d_eff..(i + 1) * d_eff].iter().any(|c| c.abs() > f32::EPSILON)
        }).count();
        let effective_k = params.k.min(valid_rows.max(1));
        let kmeans = fit_spherical_kmeans(
            &coeffs, &weights, d_eff, effective_k,
            params.rotation_seed, params.kmeans_max_iter,
        );

        // --- Stage 2: project + kmeans center assignment + EXACT residual (no quantization) ---
        // Recompute coeff from x -> subtract center*t -> add back center*t -> unproject.
        // This is PCA + K-means, no WHT/quantization. Should round-trip perfectly.
        for i in 0..bs {
            let coeff = &coeffs[i * d_eff..(i + 1) * d_eff];
            let (seg_id, t) = assign_and_project(coeff, &kmeans);
            let center = kmeans.center(seg_id as usize);
            // residual = coeff - t * center
            let res: Vec<f32> = (0..d_eff).map(|j| coeff[j] - t * center[j]).collect();
            // reconstruct: coeff = t*center + res
            let coeff_rec: Vec<f32> = (0..d_eff).map(|j| t * center[j] + res[j]).collect();
            let x_hat = unproject(&coeff_rec, &pca);
            stage2_out.extend_from_slice(&x_hat);
        }

        // --- Stage 3: project + kmeans center + WHT round-trip of residual (NO Lloyd-Max) ---
        // r -> pad -> WHT -> inverse-WHT -> should be r again; then reconstruct.
        // This isolates just the quantization effect.
        let wht_len = next_pow2(d_eff);
        for i in 0..bs {
            let coeff = &coeffs[i * d_eff..(i + 1) * d_eff];
            let (seg_id, t) = assign_and_project(coeff, &kmeans);
            let center = kmeans.center(seg_id as usize);
            let res: Vec<f32> = (0..d_eff).map(|j| coeff[j] - t * center[j]).collect();
            let res_padded = pad_zero(&res, wht_len);
            let rotated = rotate(&res_padded, params.rotation_seed);
            let unrotated = inverse_rotate(&rotated, params.rotation_seed);
            let res_rec = &unrotated[..d_eff];
            let coeff_rec: Vec<f32> = (0..d_eff).map(|j| t * center[j] + res_rec[j]).collect();
            let x_hat = unproject(&coeff_rec, &pca);
            stage3_out.extend_from_slice(&x_hat);
        }

        // --- Stage 4: full encode+decode (match encode_block's path exactly) ---
        for i in 0..bs {
            let coeff = &coeffs[i * d_eff..(i + 1) * d_eff];
            let (seg_id, t) = assign_and_project(coeff, &kmeans);
            let center = kmeans.center(seg_id as usize);
            let res: Vec<f32> = if coeff.iter().all(|c| c.abs() <= f32::EPSILON) {
                vec![0.0; d_eff]
            } else {
                kmeans_residual(coeff, t, &center)
            };
            let res_padded = pad_zero(&res, wht_len);
            let rotated = rotate(&res_padded, params.rotation_seed);
            let res_norm = l2_norm(&res);
            // Fixed: scale = 1/res_norm so unnormalized WHT gives
            // unit-variance scaled coords matched to Lloyd-Max N(0,1).
            let scale = if res_norm > f32::EPSILON {
                1.0 / res_norm
            } else { 1.0 };
            let scaled: Vec<f32> = rotated.iter().map(|v| v * scale).collect();
            let q = quantize_vector::<MSE>(&scaled, params.bit_width);
            let packed = pack_bits(&q, params.bit_width);

            // decode
            let indices = unpack_bits(&packed, params.bit_width, wht_len);
            let q_vals = dequantize_vector(&indices, params.bit_width);
            let inv_scale = match <MSE as Distortion>::NORM_MODE {
                NormMode::Absorbed => 1.0 / scale.max(f32::EPSILON),
                NormMode::Explicit => 1.0,
            };
            let q_scaled: Vec<f32> = q_vals.iter().map(|v| v * inv_scale).collect();
            let unrotated = inverse_rotate(&q_scaled, params.rotation_seed);
            let res_rec = &unrotated[..d_eff];
            let coeff_rec: Vec<f32> = (0..d_eff).map(|j| t * center[j] + res_rec[j]).collect();
            let x_hat = unproject(&coeff_rec, &pca);
            stage4_out.extend_from_slice(&x_hat);
        }
    }

    write_tensor(&args.stage_out[0], &stage1_out, n, d);
    write_tensor(&args.stage_out[1], &stage2_out, n, d);
    write_tensor(&args.stage_out[2], &stage3_out, n, d);
    write_tensor(&args.stage_out[3], &stage4_out, n, d);
    eprintln!("[stages] wrote 4 stage outputs");
}
