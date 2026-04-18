//! K-stream block-size ablation.
//!
//! For a single K-stream tensor (KKTV format), run the full kakeyaturbo
//! v1.2 encode/decode pipeline (PCA + spherical K-means + WHT residual
//! + Lloyd-Max quantize, MSE distortion, share_basis=false) at several
//! candidate block_size values. Measure per-block MSE of the decoded
//! reconstruction and the total payload bytes (skeleton + codes).
//!
//! This is the "full-codec" ablation — unlike the d_eff/outlier study
//! which only measured the PCA subspace fit, here we measure the
//! end-to-end reconstruction that a real deployment would see. That
//! matters because K-means cluster count and residual bit budget both
//! interact with block_size.
//!
//! No mock, no fallback: all the numerical work goes through the same
//! `encode_block` / `decode_block` that the bench binary uses.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use kakeyaturbo::codec::{decode_block, encode_block, total_bytes, CodecParams};
use kakeyaturbo::distortion::MSE;

const MAGIC: u32 = 0x4B4B_5456;
const VERSION: u32 = 1;

struct Args {
    input: PathBuf,
    output: PathBuf,
    block_sizes: Vec<usize>,
    variance_ratio: f32,
    k: usize,
    bit_width: u8,
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().collect();
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut block_sizes: Vec<usize> = vec![512, 1024, 2048];
    let mut variance_ratio: f32 = 0.95;
    let mut k: usize = 16;
    let mut bit_width: u8 = 3;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--input" => { i += 1; input = Some(PathBuf::from(&argv[i])); }
            "--output" => { i += 1; output = Some(PathBuf::from(&argv[i])); }
            "--block-sizes" => {
                i += 1;
                block_sizes = argv[i].split(',').map(|s| s.parse().expect("bad bs")).collect();
            }
            "--variance-ratio" => { i += 1; variance_ratio = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "--k" => { i += 1; k = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "--bit-width" => { i += 1; bit_width = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: kakeyaturbo-k-blocksize-ablation --input FILE --output JSON \\\n    [--block-sizes 512,1024,2048] [--variance-ratio 0.95] [--k 16] [--bit-width 3]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}")),
        }
        i += 1;
    }
    Ok(Args {
        input: input.ok_or("--input required")?,
        output: output.ok_or("--output required")?,
        block_sizes, variance_ratio, k, bit_width,
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

fn block_mse(orig: &[f32], rec: &[f32]) -> f64 {
    debug_assert_eq!(orig.len(), rec.len());
    let mut sq = 0.0_f64;
    for (a, b) in orig.iter().zip(rec) {
        let e = (*a - *b) as f64;
        sq += e * e;
    }
    sq / orig.len() as f64
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
    eprintln!("[bs-ablation] n_vecs={n_vecs} dim={d} block_sizes={:?}", args.block_sizes);

    // Per-strategy result row.
    struct Row {
        bs: usize,
        n_blocks: usize,
        mean_mse: f64,
        max_mse: f64,
        min_mse: f64,
        total_payload_bytes: usize,
        total_skeleton_bytes: usize,
        total_code_bytes: usize,
        mean_d_eff: f64,
        ratio_vs_raw: f64,
        covered_vecs: usize,
    }

    let mut rows: Vec<Row> = Vec::new();

    for &bs in &args.block_sizes {
        if bs == 0 || bs > n_vecs {
            eprintln!("  skip bs={bs}: too big for n_vecs={n_vecs}");
            continue;
        }
        let n_blocks = n_vecs / bs;
        if n_blocks == 0 { continue; }
        let covered = n_blocks * bs;
        let params = CodecParams {
            variance_ratio: args.variance_ratio,
            k: args.k,
            bit_width: args.bit_width,
            rotation_seed: 0xCAFE_BABE,
            kmeans_max_iter: 32,
        };
        let weights = vec![1.0_f32; bs];

        let mut mses: Vec<f64> = Vec::with_capacity(n_blocks);
        let mut sk_bytes_total = 0usize;
        let mut code_bytes_total = 0usize;
        let mut d_eff_sum = 0usize;

        for b in 0..n_blocks {
            let off = b * bs * d;
            let block = &data[off..off + bs * d];
            let (sk, codes) = encode_block::<MSE>(block, &weights, d, &params);
            let rec = decode_block::<MSE>(&sk, &codes);
            mses.push(block_mse(block, &rec));
            sk_bytes_total += sk.nbytes();
            code_bytes_total += codes.iter().map(|c| c.nbytes()).sum::<usize>();
            d_eff_sum += sk.d_eff();
            // Also confirm total_bytes matches.
            let t = total_bytes(&sk, &codes);
            debug_assert_eq!(t, sk.nbytes() + codes.iter().map(|c| c.nbytes()).sum::<usize>());
        }

        let mean_mse = mses.iter().sum::<f64>() / mses.len() as f64;
        let max_mse = mses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_mse = mses.iter().cloned().fold(f64::INFINITY, f64::min);
        let payload = sk_bytes_total + code_bytes_total;
        let raw = covered * d * std::mem::size_of::<f32>();
        let ratio = raw as f64 / payload.max(1) as f64;
        let mean_d_eff = d_eff_sum as f64 / n_blocks as f64;

        eprintln!(
            "  bs={bs:4}  n_blocks={n_blocks:3}  mean_mse={mean_mse:.3e}  \
             sk={sk_bytes_total:>7}B  codes={code_bytes_total:>8}B  ratio={ratio:.3}x  d_eff≈{mean_d_eff:.1}"
        );
        rows.push(Row {
            bs, n_blocks, mean_mse, max_mse, min_mse,
            total_payload_bytes: payload,
            total_skeleton_bytes: sk_bytes_total,
            total_code_bytes: code_bytes_total,
            mean_d_eff,
            ratio_vs_raw: ratio,
            covered_vecs: covered,
        });
    }

    // Baseline = first row (typically bs=512). Compare every row against it.
    let baseline_mse = rows.first().map(|r| r.mean_mse).unwrap_or(1.0);
    let baseline_bytes = rows.first().map(|r| r.total_payload_bytes).unwrap_or(1);

    // Emit JSON.
    let mut json = String::new();
    json.push_str(&format!(
        "{{\"n_vecs\":{},\"dim\":{},\"variance_ratio\":{},\"k\":{},\"bit_width\":{},\"cells\":[",
        n_vecs, d, args.variance_ratio, args.k, args.bit_width
    ));
    for (i, r) in rows.iter().enumerate() {
        if i > 0 { json.push(','); }
        let inflation = r.mean_mse / baseline_mse.max(f64::EPSILON);
        let byte_ratio = r.total_payload_bytes as f64 / baseline_bytes.max(1) as f64;
        json.push_str(&format!(
            "{{\"block_size\":{},\"n_blocks\":{},\"covered_vecs\":{},\
              \"mean_mse\":{:.6e},\"min_mse\":{:.6e},\"max_mse\":{:.6e},\
              \"mean_d_eff\":{:.3},\
              \"total_payload_bytes\":{},\"total_skeleton_bytes\":{},\"total_code_bytes\":{},\
              \"ratio_vs_raw\":{:.4},\
              \"mse_inflation_vs_baseline\":{:.4},\"bytes_vs_baseline\":{:.4}}}",
            r.bs, r.n_blocks, r.covered_vecs,
            r.mean_mse, r.min_mse, r.max_mse,
            r.mean_d_eff,
            r.total_payload_bytes, r.total_skeleton_bytes, r.total_code_bytes,
            r.ratio_vs_raw,
            inflation, byte_ratio,
        ));
    }
    json.push_str("]}");

    std::fs::write(&args.output, &json).expect("write output");
    eprintln!("[bs-ablation] wrote {}", args.output.display());
}
