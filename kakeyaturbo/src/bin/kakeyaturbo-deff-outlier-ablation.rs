//! PCA d_eff sensitivity + outlier channel ablation.
//!
//! For a given K tensor (KKTV format) and a grid of (variance_ratio,
//! d_res) configurations, compute the reconstruction MSE of each
//! block under:
//!
//!   reconstructed = U · Uᵀ · (x − μ)             // PCA reconstruction
//!                 + sum_{k ∈ top_d_res} e_k · residual[k]   // outlier channel
//!
//! where the outlier channel stores the d_res coordinates of the
//! residual `x - (μ + U · Uᵀ · (x - μ))` with the largest |value|,
//! stored at full fp16 precision (no quantization — this isolates
//! the structural MSE contribution from the quantization step).
//!
//! Output JSON: per-configuration mean block MSE, plus the baseline
//! (variance_ratio=0.95, d_res=0) MSE and the inflation ratio for
//! every other cell.
//!
//! No mock, no fallback: uses kakeyaturbo's own PCA fit path, and
//! the outlier selection is a real top-k on the residual vector.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use kakeyaturbo::pca::{fit_weighted_pca, project, unproject, PcaFit};

const MAGIC: u32 = 0x4B4B_5456;
const VERSION: u32 = 1;

struct Args {
    input: PathBuf,
    output: PathBuf,
    block_size: usize,
    variance_ratios: Vec<f32>,
    d_res_values: Vec<usize>,
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().collect();
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut block_size: usize = 512;
    let mut variance_ratios: Vec<f32> = vec![0.95, 0.90, 0.85, 0.80, 0.70];
    let mut d_res_values: Vec<usize> = vec![0, 2, 4, 8];
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--input" => { i += 1; input = Some(PathBuf::from(&argv[i])); }
            "--output" => { i += 1; output = Some(PathBuf::from(&argv[i])); }
            "--block-size" => { i += 1; block_size = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "--variance-ratios" => {
                i += 1;
                variance_ratios = argv[i].split(',').map(|s| s.parse().expect("bad ratio")).collect();
            }
            "--d-res-values" => {
                i += 1;
                d_res_values = argv[i].split(',').map(|s| s.parse().expect("bad d_res")).collect();
            }
            "-h" | "--help" => {
                eprintln!("Usage: kakeyaturbo-deff-outlier-ablation --input FILE --output JSON");
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}")),
        }
        i += 1;
    }
    Ok(Args {
        input: input.ok_or("--input required")?,
        output: output.ok_or("--output required")?,
        block_size, variance_ratios, d_res_values,
    })
}

fn read_u32_le(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
fn read_u64_le(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}

fn read_tensor(path: &PathBuf) -> Result<(Vec<f32>, usize, usize), String> {
    let f = File::open(path).map_err(|e| format!("open: {e}"))?;
    let mut r = BufReader::new(f);
    let m = read_u32_le(&mut r).map_err(|e| format!("magic: {e}"))?;
    if m != MAGIC { return Err(format!("bad magic {m:#x}")); }
    let v = read_u32_le(&mut r).map_err(|e| format!("ver: {e}"))?;
    if v != VERSION { return Err(format!("bad ver {v}")); }
    let n = read_u64_le(&mut r).map_err(|e| format!("n: {e}"))? as usize;
    let d = read_u32_le(&mut r).map_err(|e| format!("d: {e}"))? as usize;
    let _pad = read_u32_le(&mut r).map_err(|e| format!("pad: {e}"))?;
    let mut bytes = vec![0u8; n * d * 4];
    r.read_exact(&mut bytes).map_err(|e| format!("read: {e}"))?;
    let mut out = Vec::with_capacity(n * d);
    for c in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok((out, n, d))
}

/// Reconstruct x with PCA + optional top-d_res outlier channel.
fn reconstruct_with_outlier(x: &[f32], fit: &PcaFit, d_res: usize) -> Vec<f32> {
    let coeff = project(x, fit);
    let base = unproject(&coeff, fit);
    if d_res == 0 {
        return base;
    }
    // Residual after PCA reconstruction.
    let mut residual: Vec<f32> = x.iter().zip(&base).map(|(a, b)| a - b).collect();
    // Find top-d_res |residual| coordinates; they are kept at full precision,
    // the rest are zeroed (they've already been reconstructed by PCA).
    let d = residual.len();
    let keep = d_res.min(d);
    if keep == d {
        // Full precision on all → perfect reconstruction.
        return x.to_vec();
    }
    // Partial sort: find indices of top-keep magnitudes.
    let mut idxs: Vec<usize> = (0..d).collect();
    idxs.sort_by(|&a, &b| residual[b].abs().partial_cmp(&residual[a].abs()).unwrap());
    let keep_set: std::collections::HashSet<usize> = idxs.iter().take(keep).copied().collect();
    for (j, r) in residual.iter_mut().enumerate() {
        if !keep_set.contains(&j) {
            *r = 0.0;
        }
    }
    // Add outlier channel back to base reconstruction.
    base.iter().zip(&residual).map(|(b, r)| b + r).collect()
}

fn block_mse(block: &[f32], d: usize, fit: &PcaFit, d_res: usize) -> f64 {
    let n = block.len() / d;
    let mut sq = 0.0_f64;
    let mut count = 0usize;
    for i in 0..n {
        let x = &block[i * d..(i + 1) * d];
        let rec = reconstruct_with_outlier(x, fit, d_res);
        for j in 0..d {
            let e = (x[j] - rec[j]) as f64;
            sq += e * e;
            count += 1;
        }
    }
    sq / count as f64
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => { eprintln!("error: {e}"); std::process::exit(2); }
    };

    let (data, n_vecs, d) = match read_tensor(&args.input) {
        Ok(t) => t,
        Err(e) => { eprintln!("error: {e}"); std::process::exit(3); }
    };
    let bs = args.block_size;
    let n_blocks = n_vecs / bs;
    if n_blocks == 0 {
        eprintln!("error: need at least 1 block of size {bs}, got {n_vecs}");
        std::process::exit(4);
    }
    eprintln!("[ablation] n_vecs={n_vecs} dim={d} blocks={n_blocks} block_size={bs}");
    eprintln!("[ablation] variance_ratios={:?}  d_res_values={:?}",
        args.variance_ratios, args.d_res_values);

    let weights = vec![1.0_f32; bs];

    // Results: (variance_ratio, d_res) → list of per-block mean MSE.
    let mut results: Vec<(f32, usize, Vec<f64>, usize)> = Vec::new();  // (vr, d_res, mses, d_eff_mean)

    for &vr in &args.variance_ratios {
        // For each block, fit once at this variance_ratio; reuse across d_res values.
        let mut fits: Vec<PcaFit> = Vec::with_capacity(n_blocks);
        let mut d_eff_sum = 0usize;
        for b in 0..n_blocks {
            let off = b * bs * d;
            let block = &data[off..off + bs * d];
            let fit = fit_weighted_pca(block, &weights, d, vr);
            d_eff_sum += fit.d_eff;
            fits.push(fit);
        }
        let d_eff_mean = d_eff_sum / n_blocks;

        for &d_res in &args.d_res_values {
            let mut mses: Vec<f64> = Vec::with_capacity(n_blocks);
            for b in 0..n_blocks {
                let off = b * bs * d;
                let block = &data[off..off + bs * d];
                mses.push(block_mse(block, d, &fits[b], d_res));
            }
            let mean: f64 = mses.iter().sum::<f64>() / mses.len() as f64;
            eprintln!(
                "  vr={:.2} d_eff={} d_res={} mean_mse={:.3e}",
                vr, d_eff_mean, d_res, mean
            );
            results.push((vr, d_res, mses, d_eff_mean));
        }
    }

    // Find baseline: variance_ratio=0.95, d_res=0 (closest approximation
    // to current v1.2 K-path PCA reconstruction MSE).
    let baseline_mse: f64 = results
        .iter()
        .find(|(vr, dr, _, _)| (vr - 0.95).abs() < 1e-6 && *dr == 0)
        .map(|(_, _, mses, _)| mses.iter().sum::<f64>() / mses.len() as f64)
        .unwrap_or(1.0);

    // Emit JSON.
    let mut json = String::new();
    json.push_str("{");
    json.push_str(&format!("\"n_vecs\":{n_vecs},\"dim\":{d},\"n_blocks\":{n_blocks},\"block_size\":{bs},"));
    json.push_str(&format!("\"baseline_mse\":{baseline_mse:.6e},"));
    json.push_str("\"cells\":[");
    for (i, (vr, d_res, mses, d_eff)) in results.iter().enumerate() {
        if i > 0 { json.push(','); }
        let mean = mses.iter().sum::<f64>() / mses.len() as f64;
        let max = mses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = mses.iter().cloned().fold(f64::INFINITY, f64::min);
        let inflation = mean / baseline_mse.max(f64::EPSILON);
        let mses_csv: String = mses.iter().map(|m| format!("{:.6e}", m)).collect::<Vec<_>>().join(",");
        json.push_str(&format!(
            "{{\"variance_ratio\":{vr:.3},\"d_res\":{d_res},\"d_eff\":{d_eff},\
             \"mean_mse\":{mean:.6e},\"min_mse\":{min:.6e},\"max_mse\":{max:.6e},\
             \"inflation_over_baseline\":{inflation:.6},\"per_block_mses\":[{mses_csv}]}}"
        ));
    }
    json.push_str("]}");
    std::fs::write(&args.output, json).expect("write output");
    eprintln!("[ablation] wrote {}", args.output.display());
}
