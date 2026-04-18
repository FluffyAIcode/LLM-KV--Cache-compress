//! PCA basis-sharing ablation: for a given KV tensor file (KKTV format),
//! compare three PCA strategies on reconstruction MSE:
//!
//!   1. `per_block` — fit PCA independently on each block_size-sized chunk
//!      (v1 current behavior)
//!   2. `layer_pooled` — fit PCA once on the whole tensor, reuse across blocks
//!   3. `first_block` — fit PCA on block 0 only, reuse on all blocks
//!
//! Output JSON with per-strategy mean block MSE and a summary ratio
//! (pooled / per_block and first_block / per_block).
//!
//! No actual quantization or K-means is performed — this measures the
//! PCA subspace fit quality alone. Lower is better.
//!
//! Uses the real `kakeyaturbo::pca` code paths (no numpy, no tolerance
//! fudging).

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use kakeyaturbo::pca::{fit_weighted_pca, project, unproject};

const MAGIC: u32 = 0x4B4B_5456;
const VERSION: u32 = 1;

struct Args {
    input: PathBuf,
    output: PathBuf,
    block_size: usize,
    variance_ratio: f32,
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().collect();
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut block_size: usize = 512;
    let mut variance_ratio: f32 = 0.95;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--input" => { i += 1; input = Some(PathBuf::from(&argv[i])); }
            "--output" => { i += 1; output = Some(PathBuf::from(&argv[i])); }
            "--block-size" => { i += 1; block_size = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "--variance-ratio" => { i += 1; variance_ratio = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "-h" | "--help" => {
                eprintln!("Usage: kakeyaturbo-pca-ablation --input FILE --output REPORT.json \\\n    [--block-size N] [--variance-ratio R]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}")),
        }
        i += 1;
    }
    Ok(Args {
        input: input.ok_or("--input required")?,
        output: output.ok_or("--output required")?,
        block_size,
        variance_ratio,
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

/// Compute mean squared reconstruction error of `block` under the given PCA fit.
fn block_mse(block: &[f32], d: usize, fit: &kakeyaturbo::pca::PcaFit) -> f64 {
    let n = block.len() / d;
    let mut sq = 0.0_f64;
    let mut count = 0usize;
    for i in 0..n {
        let x = &block[i * d..(i + 1) * d];
        let coeff = project(x, fit);
        let recon = unproject(&coeff, fit);
        for j in 0..d {
            let e = (x[j] - recon[j]) as f64;
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
    if n_blocks < 2 {
        eprintln!("error: need at least 2 blocks of size {bs}, got n_vecs={n_vecs}");
        std::process::exit(4);
    }

    eprintln!(
        "[ablation] n_vecs={} dim={} blocks={} block_size={} variance_ratio={}",
        n_vecs, d, n_blocks, bs, args.variance_ratio
    );

    // ---------- Strategy 1: per_block ----------
    let mut per_block_mses: Vec<f64> = Vec::with_capacity(n_blocks);
    let mut per_block_d_eff: Vec<usize> = Vec::with_capacity(n_blocks);
    let uniform_weights = vec![1.0_f32; bs];
    for b in 0..n_blocks {
        let off = b * bs * d;
        let block = &data[off..off + bs * d];
        let fit = fit_weighted_pca(block, &uniform_weights, d, args.variance_ratio);
        let mse = block_mse(block, d, &fit);
        per_block_mses.push(mse);
        per_block_d_eff.push(fit.d_eff);
    }

    // ---------- Strategy 2: layer_pooled ----------
    // Use all n_blocks * bs vectors to fit the PCA.
    let pooled_block = &data[..n_blocks * bs * d];
    let pooled_weights = vec![1.0_f32; n_blocks * bs];
    let pooled_fit = fit_weighted_pca(pooled_block, &pooled_weights, d, args.variance_ratio);
    let pooled_d_eff = pooled_fit.d_eff;

    let mut pooled_mses: Vec<f64> = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let off = b * bs * d;
        let block = &data[off..off + bs * d];
        pooled_mses.push(block_mse(block, d, &pooled_fit));
    }

    // ---------- Strategy 3: first_block ----------
    let first_fit = {
        let block0 = &data[..bs * d];
        fit_weighted_pca(block0, &uniform_weights, d, args.variance_ratio)
    };
    let first_d_eff = first_fit.d_eff;
    let mut first_mses: Vec<f64> = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let off = b * bs * d;
        let block = &data[off..off + bs * d];
        first_mses.push(block_mse(block, d, &first_fit));
    }

    // ---------- Summary statistics ----------
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let mean_per_block = mean(&per_block_mses);
    let mean_pooled = mean(&pooled_mses);
    let mean_first = mean(&first_mses);

    // Per-block ratios (inflation factor of pooled over per_block)
    let mut pooled_ratios = Vec::new();
    let mut first_ratios = Vec::new();
    for b in 0..n_blocks {
        if per_block_mses[b] > 0.0 {
            pooled_ratios.push(pooled_mses[b] / per_block_mses[b]);
            first_ratios.push(first_mses[b] / per_block_mses[b]);
        }
    }
    let max_pooled_ratio = pooled_ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_first_ratio = first_ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let median_pooled_ratio = {
        let mut v = pooled_ratios.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    };
    let median_first_ratio = {
        let mut v = first_ratios.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    };

    // Emit JSON.
    let per_block_mses_csv: String = per_block_mses.iter().map(|x| format!("{:.6e}", x)).collect::<Vec<_>>().join(",");
    let pooled_mses_csv: String = pooled_mses.iter().map(|x| format!("{:.6e}", x)).collect::<Vec<_>>().join(",");
    let first_mses_csv: String = first_mses.iter().map(|x| format!("{:.6e}", x)).collect::<Vec<_>>().join(",");
    let d_eff_csv: String = per_block_d_eff.iter().map(|x| format!("{}", x)).collect::<Vec<_>>().join(",");

    let json = format!(
        "{{\
            \"n_vecs\":{n_vecs},\
            \"dim\":{d},\
            \"block_size\":{bs},\
            \"n_blocks\":{n_blocks},\
            \"variance_ratio\":{vr},\
            \"per_block_mean_mse\":{m1:.6e},\
            \"layer_pooled_mean_mse\":{m2:.6e},\
            \"first_block_mean_mse\":{m3:.6e},\
            \"pooled_over_per_block_mean\":{r1:.4},\
            \"pooled_over_per_block_median\":{r1m:.4},\
            \"pooled_over_per_block_max\":{r1x:.4},\
            \"first_over_per_block_mean\":{r2:.4},\
            \"first_over_per_block_median\":{r2m:.4},\
            \"first_over_per_block_max\":{r2x:.4},\
            \"pooled_d_eff\":{pde},\
            \"first_d_eff\":{fde},\
            \"per_block_d_eff\":[{dc}],\
            \"per_block_mses\":[{p1}],\
            \"layer_pooled_mses\":[{p2}],\
            \"first_block_mses\":[{p3}]\
        }}",
        n_vecs = n_vecs,
        d = d,
        bs = bs,
        n_blocks = n_blocks,
        vr = args.variance_ratio,
        m1 = mean_per_block,
        m2 = mean_pooled,
        m3 = mean_first,
        r1 = mean_pooled / mean_per_block.max(f64::EPSILON),
        r1m = median_pooled_ratio,
        r1x = max_pooled_ratio,
        r2 = mean_first / mean_per_block.max(f64::EPSILON),
        r2m = median_first_ratio,
        r2x = max_first_ratio,
        pde = pooled_d_eff,
        fde = first_d_eff,
        dc = d_eff_csv,
        p1 = per_block_mses_csv,
        p2 = pooled_mses_csv,
        p3 = first_mses_csv,
    );
    if let Err(e) = std::fs::write(&args.output, &json) {
        eprintln!("error: write: {e}");
        std::process::exit(5);
    }
    eprintln!(
        "[ablation] per_block={:.3e}  pooled={:.3e} ({:.3}x)  first={:.3e} ({:.3}x)",
        mean_per_block,
        mean_pooled,
        mean_pooled / mean_per_block.max(f64::EPSILON),
        mean_first,
        mean_first / mean_per_block.max(f64::EPSILON),
    );
}
