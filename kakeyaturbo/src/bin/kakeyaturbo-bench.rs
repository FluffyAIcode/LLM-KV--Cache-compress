//! CLI bench wrapper around the kakeyaturbo codec.
//!
//! Reads a raw f32 KV tensor from a file, encodes it block-by-block
//! under the given codec parameters, writes a JSON report of byte
//! counts and timings, and returns a non-zero exit code on any error.
//!
//! File format (little-endian, no compression):
//!
//! ```text
//!   magic      u32   = 0x4B4B5456  // "KKTV"
//!   version    u32   = 1
//!   num_vecs   u64
//!   dim        u32
//!   _padding   u32   = 0
//!   data       [f32; num_vecs * dim]
//! ```
//!
//! Everything downstream of the codec — block splitting, weights,
//! metric choice, reporting — is driven by CLI flags. No defaults
//! are hidden; `--help` lists every knob.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::time::Instant;

use kakeyaturbo::{
    decode_block, decode_layer, encode_block, encode_layer, CodecParams, Code, Distortion,
    InnerProduct, LInf, LayerEncoding, MSE,
};

const MAGIC: u32 = 0x4B4B_5456;
const VERSION: u32 = 1;

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output_report: PathBuf,
    metric: String,
    block_size: usize,
    variance_ratio: f32,
    k: usize,
    bit_width: u8,
    rotation_seed: u32,
    verify: bool,
    share_basis: bool,
}

fn print_help() {
    eprintln!(
        "Usage: kakeyaturbo-bench --input <FILE> --output <REPORT.json> \\\n    \
            [--metric mse|inner_product|linf] \\\n    \
            [--block-size N] [--variance-ratio R] \\\n    \
            [--k K] [--bit-width B] [--rotation-seed S] \\\n    \
            [--verify]\n\n\
            Compresses a KV tensor file block-by-block using the\n\
            kakeyaturbo codec and writes a JSON report.\n"
    );
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().collect();
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut metric = "mse".to_string();
    let mut block_size: usize = 512;
    let mut variance_ratio: f32 = 0.95;
    let mut k: usize = 16;
    let mut bit_width: u8 = 3;
    let mut rotation_seed: u32 = 0xCAFE_BABE;
    let mut verify = false;
    let mut share_basis = false;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--input" => {
                i += 1;
                input = Some(PathBuf::from(&argv[i]));
            }
            "--output" => {
                i += 1;
                output = Some(PathBuf::from(&argv[i]));
            }
            "--metric" => {
                i += 1;
                metric = argv[i].clone();
            }
            "--block-size" => {
                i += 1;
                block_size = argv[i].parse().map_err(|e| format!("bad --block-size: {e}"))?;
            }
            "--variance-ratio" => {
                i += 1;
                variance_ratio = argv[i]
                    .parse()
                    .map_err(|e| format!("bad --variance-ratio: {e}"))?;
            }
            "--k" => {
                i += 1;
                k = argv[i].parse().map_err(|e| format!("bad --k: {e}"))?;
            }
            "--bit-width" => {
                i += 1;
                bit_width = argv[i].parse().map_err(|e| format!("bad --bit-width: {e}"))?;
            }
            "--rotation-seed" => {
                i += 1;
                rotation_seed = argv[i]
                    .parse()
                    .map_err(|e| format!("bad --rotation-seed: {e}"))?;
            }
            "--verify" => verify = true,
            "--share-basis" => share_basis = true,
            other => return Err(format!("unknown flag {other}; try --help")),
        }
        i += 1;
    }

    let input = input.ok_or_else(|| "--input required".to_string())?;
    let output = output.ok_or_else(|| "--output required".to_string())?;
    Ok(Args {
        input,
        output_report: output,
        metric,
        block_size,
        variance_ratio,
        k,
        bit_width,
        rotation_seed,
        verify,
        share_basis,
    })
}

fn read_u32_le(r: &mut impl Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(r: &mut impl Read) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_tensor(path: &PathBuf) -> Result<(Vec<f32>, usize, usize), String> {
    let f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut r = BufReader::new(f);

    let magic = read_u32_le(&mut r).map_err(|e| format!("read magic: {e}"))?;
    if magic != MAGIC {
        return Err(format!(
            "bad magic {magic:#x}, expected {MAGIC:#x} (file must be a KKTV tensor)"
        ));
    }
    let version = read_u32_le(&mut r).map_err(|e| format!("read version: {e}"))?;
    if version != VERSION {
        return Err(format!("unsupported version {version}"));
    }
    let num_vecs = read_u64_le(&mut r).map_err(|e| format!("read num_vecs: {e}"))? as usize;
    let dim = read_u32_le(&mut r).map_err(|e| format!("read dim: {e}"))? as usize;
    let _pad = read_u32_le(&mut r).map_err(|e| format!("read pad: {e}"))?;

    let total = num_vecs
        .checked_mul(dim)
        .ok_or_else(|| "num_vecs * dim overflow".to_string())?;
    let mut data = vec![0u8; total * 4];
    r.read_exact(&mut data).map_err(|e| format!("read data: {e}"))?;
    let mut out = Vec::with_capacity(total);
    for chunk in data.chunks_exact(4) {
        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out.push(f32::from_le_bytes(arr));
    }
    Ok((out, num_vecs, dim))
}

fn run<R: Distortion>(args: &Args, data: &[f32], num_vecs: usize, dim: usize) -> Report {
    let params = CodecParams {
        variance_ratio: args.variance_ratio,
        k: args.k,
        bit_width: args.bit_width,
        rotation_seed: args.rotation_seed,
        kmeans_max_iter: 32,
    };

    let bs = args.block_size;
    let n_full = num_vecs / bs;
    let weights = vec![1.0_f32; bs];

    let mut total_mse_sum = 0.0_f64;
    let mut total_mse_count = 0usize;
    let mut encode_ns: u128 = 0;
    let mut decode_ns: u128 = 0;

    let (total_skeleton, total_codes, total_blocks, total_vecs_encoded, shared_pca_bytes) = if args.share_basis {
        // v1.2 B' path: fit one basis over all blocks, K-means per-block.
        let mut block_vecs: Vec<Vec<f32>> = Vec::with_capacity(n_full);
        let mut ws: Vec<Vec<f32>> = Vec::with_capacity(n_full);
        for b in 0..n_full {
            let off = b * bs * dim;
            block_vecs.push(data[off..off + bs * dim].to_vec());
            ws.push(weights.clone());
        }
        let t0 = Instant::now();
        let enc: LayerEncoding = encode_layer::<R>(&block_vecs, &ws, dim, &params, true);
        encode_ns += t0.elapsed().as_nanos();

        if args.verify {
            let t1 = Instant::now();
            let recs = decode_layer::<R>(&enc);
            decode_ns += t1.elapsed().as_nanos();
            for (i, rec) in recs.iter().enumerate() {
                let orig = &block_vecs[i];
                let mut sq = 0.0_f64;
                for j in 0..bs * dim {
                    let e = orig[j] - rec[j];
                    sq += (e * e) as f64;
                }
                total_mse_sum += sq / (bs * dim) as f64;
                total_mse_count += 1;
            }
        }

        let shared_pca_bytes = enc.shared_pca.as_ref().map(|p| p.nbytes()).unwrap_or(0);
        // Skeleton accounting: one shared PCA once, K-means per block.
        let per_block_skel: usize = enc.per_block.iter().map(|(sk, _)| sk.kmeans.nbytes()).sum();
        let codes_total: usize = enc.per_block.iter().flat_map(|(_, cs)| cs.iter()).map(|c| c.nbytes()).sum();
        let skeleton_total = shared_pca_bytes + per_block_skel;
        (skeleton_total, codes_total, enc.per_block.len(), enc.per_block.len() * bs, shared_pca_bytes)
    } else {
        // v1.0/1.1 path: per-block fit.
        let mut total_skeleton = 0usize;
        let mut total_codes = 0usize;
        let mut total_blocks = 0usize;
        let mut total_vecs_encoded = 0usize;

        for b in 0..n_full {
            let off = b * bs * dim;
            let block = &data[off..off + bs * dim];
            let t0 = Instant::now();
            let (sk, codes) = encode_block::<R>(block, &weights, dim, &params);
            encode_ns += t0.elapsed().as_nanos();
            total_skeleton += sk.nbytes();
            total_codes += codes.iter().map(|c: &Code| c.nbytes()).sum::<usize>();
            total_blocks += 1;
            total_vecs_encoded += bs;

            if args.verify {
                let t1 = Instant::now();
                let rec = decode_block::<R>(&sk, &codes);
                decode_ns += t1.elapsed().as_nanos();
                let mut sq = 0.0_f64;
                for i in 0..bs * dim {
                    let e = block[i] - rec[i];
                    sq += (e * e) as f64;
                }
                total_mse_sum += sq / (bs * dim) as f64;
                total_mse_count += 1;
            }
        }
        (total_skeleton, total_codes, total_blocks, total_vecs_encoded, 0)
    };

    let baseline_bytes = total_vecs_encoded * dim * std::mem::size_of::<f32>();
    let baseline_bytes_bf16 = total_vecs_encoded * dim * 2;
    let compressed_bytes = total_skeleton + total_codes;

    Report {
        metric: R::NAME.to_string(),
        block_size: bs,
        variance_ratio: args.variance_ratio,
        k: args.k,
        bit_width: args.bit_width,
        dim,
        num_vecs_encoded: total_vecs_encoded,
        num_blocks: total_blocks,
        skeleton_bytes: total_skeleton,
        codes_bytes: total_codes,
        compressed_bytes,
        baseline_bytes_f32: baseline_bytes,
        baseline_bytes_bf16,
        ratio_vs_f32: baseline_bytes as f64 / compressed_bytes as f64,
        ratio_vs_bf16: baseline_bytes_bf16 as f64 / compressed_bytes as f64,
        encode_seconds: encode_ns as f64 / 1e9,
        decode_seconds: decode_ns as f64 / 1e9,
        verify: args.verify,
        mean_block_mse: if total_mse_count > 0 {
            total_mse_sum / total_mse_count as f64
        } else {
            -1.0
        },
        share_basis: args.share_basis,
        shared_pca_bytes,
    }
}

struct Report {
    metric: String,
    block_size: usize,
    variance_ratio: f32,
    k: usize,
    bit_width: u8,
    dim: usize,
    num_vecs_encoded: usize,
    num_blocks: usize,
    skeleton_bytes: usize,
    codes_bytes: usize,
    compressed_bytes: usize,
    baseline_bytes_f32: usize,
    baseline_bytes_bf16: usize,
    ratio_vs_f32: f64,
    ratio_vs_bf16: f64,
    encode_seconds: f64,
    decode_seconds: f64,
    verify: bool,
    mean_block_mse: f64,
    share_basis: bool,
    shared_pca_bytes: usize,
}

impl Report {
    fn to_json(&self) -> String {
        format!(
            "{{\
                \"metric\":\"{metric}\",\
                \"block_size\":{bs},\
                \"variance_ratio\":{vr},\
                \"k\":{k},\
                \"bit_width\":{bw},\
                \"dim\":{dim},\
                \"num_vecs_encoded\":{nv},\
                \"num_blocks\":{nb},\
                \"skeleton_bytes\":{sk},\
                \"codes_bytes\":{cb},\
                \"compressed_bytes\":{cmp},\
                \"baseline_bytes_f32\":{bf32},\
                \"baseline_bytes_bf16\":{bbf16},\
                \"ratio_vs_f32\":{rf32:.6},\
                \"ratio_vs_bf16\":{rbf16:.6},\
                \"encode_seconds\":{es:.6},\
                \"decode_seconds\":{ds:.6},\
                \"verify\":{verify},\
                \"mean_block_mse\":{mse:.10},\
                \"share_basis\":{sb},\
                \"shared_pca_bytes\":{spb}\
            }}",
            metric = self.metric,
            bs = self.block_size,
            vr = self.variance_ratio,
            k = self.k,
            bw = self.bit_width,
            dim = self.dim,
            nv = self.num_vecs_encoded,
            nb = self.num_blocks,
            sk = self.skeleton_bytes,
            cb = self.codes_bytes,
            cmp = self.compressed_bytes,
            bf32 = self.baseline_bytes_f32,
            bbf16 = self.baseline_bytes_bf16,
            rf32 = self.ratio_vs_f32,
            rbf16 = self.ratio_vs_bf16,
            es = self.encode_seconds,
            ds = self.decode_seconds,
            verify = self.verify,
            mse = self.mean_block_mse,
            sb = self.share_basis,
            spb = self.shared_pca_bytes,
        )
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    let (data, num_vecs, dim) = match read_tensor(&args.input) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(3);
        }
    };

    eprintln!(
        "[bench] loaded {} vectors of dim {} ({:.2} MiB f32)",
        num_vecs,
        dim,
        (num_vecs * dim * 4) as f64 / 1024.0 / 1024.0
    );

    let report = match args.metric.as_str() {
        "mse" => run::<MSE>(&args, &data, num_vecs, dim),
        "inner_product" | "ip" => run::<InnerProduct>(&args, &data, num_vecs, dim),
        "linf" => run::<LInf>(&args, &data, num_vecs, dim),
        other => {
            eprintln!("error: unknown metric {other}; try mse|inner_product|linf");
            std::process::exit(4);
        }
    };

    let json = report.to_json();
    if let Err(e) = std::fs::write(&args.output_report, &json) {
        eprintln!("error: write {}: {e}", args.output_report.display());
        std::process::exit(5);
    }
    eprintln!(
        "[bench] {} blocks of {} vectors, compressed {} -> {} bytes (bf16 baseline ratio {:.3}x)",
        report.num_blocks,
        report.block_size,
        report.baseline_bytes_bf16,
        report.compressed_bytes,
        report.ratio_vs_bf16,
    );
}
