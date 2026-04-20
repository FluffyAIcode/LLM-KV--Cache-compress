//! CLI bench wrapper around the Besicovitch-product codec.
//!
//! Reads a raw f32 KV tensor in the `.kktv` file format (same as the
//! main `kakeyaturbo-bench`), encodes it block-by-block against a
//! fixed Besicovitch-product codebook, and writes a JSON report of
//! byte counts, MSE, and timings.
//!
//! The Besicovitch codec has no per-block skeleton: the direction
//! codebook is deterministic from `(group_size, direction_bits)` and
//! therefore does not ship with the compressed block.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::time::Instant;

use kakeyaturbo::{
    BesicovitchParams, MagnitudeMode,
    besicovitch_encode_block_full, besicovitch_decode_block_full,
};

const MAGIC: u32 = 0x4B4B_5456;
const VERSION: u32 = 1;

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output_report: PathBuf,
    block_size: usize,
    group_size: usize,
    direction_bits: u8,
    magnitude_bits: u8,
    magnitude_mode: String,
    subtract_mean: bool,
    verify: bool,
    dump_decoded: Option<PathBuf>,
}

fn usage() -> ! {
    eprintln!(
        "usage: besicovitch-bench \
         --input PATH --output PATH \
         [--block-size N] \
         [--group-size G] \
         [--direction-bits B] \
         [--magnitude-bits B] \
         [--magnitude-mode {{f16|quantized}}] \
         [--verify]"
    );
    std::process::exit(2);
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().collect();
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut block_size = 1024usize;
    let mut group_size = 2usize;
    let mut direction_bits = 6u8;
    let mut magnitude_bits = 4u8;
    let mut magnitude_mode = String::from("f16");
    let mut subtract_mean = false;
    let mut verify = false;
    let mut dump_decoded: Option<PathBuf> = None;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--help" | "-h" => usage(),
            "--input" => {
                i += 1;
                input = Some(PathBuf::from(&argv[i]));
            }
            "--output" => {
                i += 1;
                output = Some(PathBuf::from(&argv[i]));
            }
            "--block-size" => {
                i += 1;
                block_size = argv[i].parse().map_err(|e| format!("bad --block-size: {e}"))?;
            }
            "--group-size" => {
                i += 1;
                group_size = argv[i].parse().map_err(|e| format!("bad --group-size: {e}"))?;
            }
            "--direction-bits" => {
                i += 1;
                direction_bits = argv[i].parse().map_err(|e| format!("bad --direction-bits: {e}"))?;
            }
            "--magnitude-bits" => {
                i += 1;
                magnitude_bits = argv[i].parse().map_err(|e| format!("bad --magnitude-bits: {e}"))?;
            }
            "--magnitude-mode" => {
                i += 1;
                magnitude_mode = argv[i].clone();
            }
            "--subtract-mean" => subtract_mean = true,
            "--verify" => verify = true,
            "--dump-decoded" => {
                i += 1;
                dump_decoded = Some(PathBuf::from(&argv[i]));
            }
            other => return Err(format!("unknown flag {other}; try --help")),
        }
        i += 1;
    }
    Ok(Args {
        input: input.ok_or("missing --input")?,
        output_report: output.ok_or("missing --output")?,
        block_size,
        group_size,
        direction_bits,
        magnitude_bits,
        magnitude_mode,
        subtract_mean,
        verify,
        dump_decoded,
    })
}

fn read_kktv(path: &PathBuf) -> Result<(usize, usize, Vec<f32>), String> {
    let f = File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let mut r = BufReader::new(f);
    let mut u32buf = [0u8; 4];
    let mut u64buf = [0u8; 8];
    r.read_exact(&mut u32buf).map_err(|e| format!("read magic: {e}"))?;
    if u32::from_le_bytes(u32buf) != MAGIC {
        return Err("bad magic".to_string());
    }
    r.read_exact(&mut u32buf).map_err(|e| format!("read version: {e}"))?;
    if u32::from_le_bytes(u32buf) != VERSION {
        return Err("bad version".to_string());
    }
    r.read_exact(&mut u64buf).map_err(|e| format!("read n_vecs: {e}"))?;
    let n = u64::from_le_bytes(u64buf) as usize;
    r.read_exact(&mut u32buf).map_err(|e| format!("read dim: {e}"))?;
    let d = u32::from_le_bytes(u32buf) as usize;
    r.read_exact(&mut u32buf).map_err(|e| format!("read pad: {e}"))?;
    let total = n * d;
    let mut data = vec![0.0_f32; total];
    let mut bytes = vec![0u8; total * 4];
    r.read_exact(&mut bytes).map_err(|e| format!("read data: {e}"))?;
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok((n, d, data))
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[besicovitch-bench] {e}");
            std::process::exit(2);
        }
    };

    let (n, d, data) = read_kktv(&args.input).unwrap_or_else(|e| {
        eprintln!("[besicovitch-bench] {e}");
        std::process::exit(1);
    });
    if d % args.group_size != 0 {
        eprintln!(
            "[besicovitch-bench] dim {d} not divisible by group_size {}",
            args.group_size
        );
        std::process::exit(1);
    }

    let magnitude_mode = match args.magnitude_mode.as_str() {
        "f16" => MagnitudeMode::F16,
        "quantized" => MagnitudeMode::QuantizedWithPerVectorScale,
        other => {
            eprintln!("[besicovitch-bench] bad --magnitude-mode {other}");
            std::process::exit(2);
        }
    };
    let params = BesicovitchParams {
        group_size: args.group_size,
        direction_bits: args.direction_bits,
        magnitude_bits: args.magnitude_bits,
        magnitude_mode,
        subtract_mean: args.subtract_mean,
    };

    let bs = args.block_size;
    let mut compressed_bytes: usize = 0;
    let mut skeleton_bytes: usize = 0;
    let mut block_mse_sum: f64 = 0.0;
    let mut block_mse_count: usize = 0;
    let t0 = Instant::now();
    let mut t_encode_ns: u128 = 0;
    let mut t_decode_ns: u128 = 0;

    // If --dump-decoded, collect all reconstructions and write them at the end.
    let mut all_decoded: Option<Vec<f32>> = if args.dump_decoded.is_some() {
        Some(Vec::with_capacity(n * d))
    } else {
        None
    };

    let mut idx = 0;
    while idx < n {
        let end = (idx + bs).min(n);
        let n_block = end - idx;
        let block = &data[idx * d..end * d];

        let te0 = Instant::now();
        let (codebook, skeleton, codes) = besicovitch_encode_block_full(block, d, &params);
        t_encode_ns += te0.elapsed().as_nanos();

        let code_bytes: usize = codes.iter().map(|c| c.nbytes(params.direction_bits)).sum();
        let sk_bytes = skeleton.nbytes();
        compressed_bytes += code_bytes + sk_bytes;
        skeleton_bytes += sk_bytes;

        if args.verify || all_decoded.is_some() {
            let td0 = Instant::now();
            let rec = besicovitch_decode_block_full(&codebook, &skeleton, &codes, d, &params);
            t_decode_ns += td0.elapsed().as_nanos();
            if args.verify {
                for i in 0..n_block {
                    let a = &block[i * d..(i + 1) * d];
                    let b = &rec[i * d..(i + 1) * d];
                    let mse: f64 = a
                        .iter()
                        .zip(b)
                        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
                        .sum::<f64>()
                        / d as f64;
                    block_mse_sum += mse;
                    block_mse_count += 1;
                }
            }
            if let Some(buf) = all_decoded.as_mut() {
                buf.extend_from_slice(&rec);
            }
        }
        idx = end;
    }

    if let (Some(path), Some(buf)) = (args.dump_decoded.as_ref(), all_decoded.as_ref()) {
        let mut out = Vec::with_capacity(16 + 4 + buf.len() * 4);
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(n as u64).to_le_bytes());
        out.extend_from_slice(&(d as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for v in buf {
            out.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, out).unwrap_or_else(|e| {
            eprintln!("write decoded: {e}");
            std::process::exit(1);
        });
    }
    let t_total_ns = t0.elapsed().as_nanos();

    let bf16_bytes = n * d * 2;
    let ratio = bf16_bytes as f64 / compressed_bytes.max(1) as f64;
    let mean_mse = if block_mse_count > 0 {
        block_mse_sum / block_mse_count as f64
    } else {
        0.0
    };
    let groups = d / args.group_size;
    let per_vec_bits = match magnitude_mode {
        MagnitudeMode::F16 => {
            (groups * args.direction_bits as usize) + groups * 16
        }
        MagnitudeMode::QuantizedWithPerVectorScale => {
            (groups * args.direction_bits as usize)
                + (groups * args.magnitude_bits as usize)
                + 16
        }
    };

    let per_vec_measured = (compressed_bytes as f64) / (n as f64);
    let report = format!(
        "{{\n  \"n_vectors\": {n},\n  \"dim\": {d},\n  \
         \"block_size\": {bs},\n  \"group_size\": {gs},\n  \
         \"direction_bits\": {db},\n  \"magnitude_bits\": {mb},\n  \
         \"magnitude_mode\": \"{mm}\",\n  \
         \"subtract_mean\": {sm},\n  \
         \"per_vector_bits_analytical\": {pvba},\n  \
         \"per_vector_bytes_measured\": {pvbm},\n  \
         \"compressed_bytes\": {cb},\n  \"skeleton_bytes\": {skb},\n  \
         \"bf16_bytes\": {bfb},\n  \
         \"ratio_vs_bf16\": {ratio:.6},\n  \
         \"mean_block_mse\": {mse},\n  \
         \"t_total_ns\": {tt},\n  \"t_encode_ns\": {te},\n  \
         \"t_decode_ns\": {td}\n}}\n",
        gs = args.group_size,
        db = args.direction_bits,
        mb = args.magnitude_bits,
        mm = args.magnitude_mode,
        sm = args.subtract_mean,
        pvba = per_vec_bits,
        pvbm = per_vec_measured,
        cb = compressed_bytes,
        skb = skeleton_bytes,
        bfb = bf16_bytes,
        ratio = ratio,
        mse = mean_mse,
        tt = t_total_ns as u64,
        te = t_encode_ns as u64,
        td = t_decode_ns as u64,
    );
    std::fs::write(&args.output_report, report)
    .unwrap_or_else(|e| {
        eprintln!("write report: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "[besicovitch-bench] n={n} D={d} block={bs} g={} b_dir={} b_mag={} mode={} \
         ratio={ratio:.2}x MSE={mean_mse:.4e}",
        args.group_size, args.direction_bits, args.magnitude_bits, args.magnitude_mode
    );
}
