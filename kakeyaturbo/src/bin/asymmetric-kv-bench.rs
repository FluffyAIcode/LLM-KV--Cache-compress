//! CLI bench wrapper for **asymmetric** K/V codec operation.
//!
//! Reads two raw f32 tensors — one K, one V — in the `.kktv` file
//! format, encodes each stream with its own codec (Kakeya-PCA for K,
//! Besicovitch for V by default, but configurable), and writes a
//! single JSON report with combined byte accounting and per-stream
//! MSE.  This bypasses the per-stream-subprocess overhead that the
//! Python harness currently pays.
//!
//! Payload format (in-memory — not serialized to disk by this bench;
//! callers interested in on-disk serialization use `--dump-k-decoded`
//! and `--dump-v-decoded`):
//!
//! ```text
//!   header:
//!     magic        u32  = 0x4B4B5641  // "KKVA" (Kakeya K/V Asymmetric)
//!     version      u32  = 1
//!     k_codec_id   u32  = 0 (kakeya) | 1 (besicovitch)
//!     v_codec_id   u32  = 0 (kakeya) | 1 (besicovitch)
//!   k_payload: ... (codec-specific)
//!   v_payload: ... (codec-specific)
//! ```
//!
//! The per-stream payloads are exactly what the existing
//! `kakeyaturbo-bench` / `besicovitch-bench` would produce, so the
//! binary is byte-compatible with the current Python harness output.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::time::Instant;

use kakeyaturbo::{
    besicovitch_decode_block_full, besicovitch_encode_block_full,
    decode_block, decode_block_with_centroids, encode_block,
    BesicovitchParams, CodecParams, InnerProduct, LInf, MagnitudeMode,
    PcaMethod, SkeletonDtype, MSE,
};

const KKTV_MAGIC: u32 = 0x4B4B_5456;
const KKTV_VERSION: u32 = 1;

#[derive(Debug)]
struct CodecConfig {
    /// Which codec this stream uses.  "kakeya" or "besicovitch".
    kind: String,

    // Kakeya params (when kind == "kakeya")
    bit_width: u8,
    variance_ratio: f32,
    k: usize,
    rotation_seed: u32,
    metric: String,
    pca_method: String,
    skeleton_dtype: String,

    // Besicovitch params (when kind == "besicovitch")
    besi_group_size: usize,
    besi_direction_bits: u8,
    besi_magnitude_bits: u8,
    besi_magnitude_mode: String,
    besi_subtract_mean: bool,
}

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            kind: "kakeya".into(),
            bit_width: 4,
            variance_ratio: 0.95,
            k: 16,
            rotation_seed: 0xCAFE_BABE,
            metric: "mse".into(),
            pca_method: "exact".into(),
            skeleton_dtype: "fp16".into(),
            besi_group_size: 2,
            besi_direction_bits: 3,
            besi_magnitude_bits: 4,
            besi_magnitude_mode: "quantized".into(),
            besi_subtract_mean: true,
        }
    }
}

#[derive(Debug)]
struct Args {
    k_input: PathBuf,
    v_input: PathBuf,
    output_report: PathBuf,
    block_size: usize,
    k_cfg: CodecConfig,
    v_cfg: CodecConfig,
    verify: bool,
}

fn usage() -> ! {
    eprintln!(
        "usage: asymmetric-kv-bench \
         --k-input K.kktv --v-input V.kktv --output REPORT.json \
         [--block-size N]\n\
         \n\
         K-stream:\n\
         --k-codec {{kakeya|besicovitch}}  (default: kakeya)\n\
         --k-bit-width B  --k-variance-ratio R  --k-k K  --k-rotation-seed S\n\
         --k-metric {{mse|inner_product|linf}}  --k-pca-method {{exact|randomized}}\n\
         --k-skeleton-dtype {{fp16|fp32}}\n\
         --k-besi-group-size G  --k-besi-direction-bits DB  --k-besi-magnitude-bits MB\n\
         --k-besi-magnitude-mode {{f16|quantized}}  --k-besi-subtract-mean\n\
         \n\
         V-stream: same flags with --v- prefix.\n\
         \n\
         --verify : decode each stream and report per-block MSE."
    );
    std::process::exit(2);
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().collect();
    let mut k_input: Option<PathBuf> = None;
    let mut v_input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut block_size: usize = 1024;
    let mut verify = false;
    let mut k_cfg = CodecConfig::default();
    let mut v_cfg = CodecConfig {
        kind: "besicovitch".into(),
        besi_direction_bits: 3,
        besi_magnitude_bits: 4,
        besi_magnitude_mode: "quantized".into(),
        besi_subtract_mean: true,
        ..CodecConfig::default()
    };

    fn apply(cfg: &mut CodecConfig, key: &str, val: &str) -> Result<(), String> {
        match key {
            "codec" => cfg.kind = val.to_string(),
            "bit-width" => cfg.bit_width = val.parse().map_err(|e| format!("{e}"))?,
            "variance-ratio" => cfg.variance_ratio = val.parse().map_err(|e| format!("{e}"))?,
            "k" => cfg.k = val.parse().map_err(|e| format!("{e}"))?,
            "rotation-seed" => cfg.rotation_seed = val.parse().map_err(|e| format!("{e}"))?,
            "metric" => cfg.metric = val.to_string(),
            "pca-method" => cfg.pca_method = val.to_string(),
            "skeleton-dtype" => cfg.skeleton_dtype = val.to_string(),
            "besi-group-size" => cfg.besi_group_size = val.parse().map_err(|e| format!("{e}"))?,
            "besi-direction-bits" => cfg.besi_direction_bits = val.parse().map_err(|e| format!("{e}"))?,
            "besi-magnitude-bits" => cfg.besi_magnitude_bits = val.parse().map_err(|e| format!("{e}"))?,
            "besi-magnitude-mode" => cfg.besi_magnitude_mode = val.to_string(),
            other => return Err(format!("unknown config key '{other}'")),
        }
        Ok(())
    }

    let mut i = 1;
    while i < argv.len() {
        let a = &argv[i];
        match a.as_str() {
            "--help" | "-h" => usage(),
            "--k-input" => { i += 1; k_input = Some(PathBuf::from(&argv[i])); }
            "--v-input" => { i += 1; v_input = Some(PathBuf::from(&argv[i])); }
            "--output" => { i += 1; output = Some(PathBuf::from(&argv[i])); }
            "--block-size" => { i += 1; block_size = argv[i].parse().map_err(|e| format!("bad --block-size: {e}"))?; }
            "--verify" => { verify = true; }
            "--k-besi-subtract-mean" => { k_cfg.besi_subtract_mean = true; }
            "--v-besi-subtract-mean" => { v_cfg.besi_subtract_mean = true; }
            "--k-besi-no-subtract-mean" => { k_cfg.besi_subtract_mean = false; }
            "--v-besi-no-subtract-mean" => { v_cfg.besi_subtract_mean = false; }
            s if s.starts_with("--k-") => {
                let key = &s[4..];
                i += 1;
                apply(&mut k_cfg, key, &argv[i]).map_err(|e| format!("--{s}: {e}"))?;
            }
            s if s.starts_with("--v-") => {
                let key = &s[4..];
                i += 1;
                apply(&mut v_cfg, key, &argv[i]).map_err(|e| format!("--{s}: {e}"))?;
            }
            other => return Err(format!("unknown flag {other}")),
        }
        i += 1;
    }
    Ok(Args {
        k_input: k_input.ok_or("missing --k-input")?,
        v_input: v_input.ok_or("missing --v-input")?,
        output_report: output.ok_or("missing --output")?,
        block_size,
        k_cfg,
        v_cfg,
        verify,
    })
}

fn read_kktv(path: &PathBuf) -> Result<(usize, usize, Vec<f32>), String> {
    let f = File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let mut r = BufReader::new(f);
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf4).map_err(|e| format!("read magic: {e}"))?;
    if u32::from_le_bytes(buf4) != KKTV_MAGIC {
        return Err(format!("{}: bad magic", path.display()));
    }
    r.read_exact(&mut buf4).map_err(|e| format!("read version: {e}"))?;
    if u32::from_le_bytes(buf4) != KKTV_VERSION {
        return Err(format!("{}: bad version", path.display()));
    }
    r.read_exact(&mut buf8).map_err(|e| format!("read n: {e}"))?;
    let n = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf4).map_err(|e| format!("read d: {e}"))?;
    let d = u32::from_le_bytes(buf4) as usize;
    r.read_exact(&mut buf4).map_err(|e| format!("read pad: {e}"))?;
    let total = n * d;
    let mut bytes = vec![0u8; total * 4];
    r.read_exact(&mut bytes).map_err(|e| format!("read data: {e}"))?;
    let data = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((n, d, data))
}

fn build_codec_params(cfg: &CodecConfig) -> Result<CodecParams, String> {
    let pca_method = match cfg.pca_method.as_str() {
        "exact" => PcaMethod::Exact,
        "randomized" => PcaMethod::Randomized {
            target_rank: 64, oversample: 8, power_iters: 2, seed_offset: 0xABCD,
        },
        other => return Err(format!("bad pca-method '{other}'")),
    };
    let skeleton_dtype = match cfg.skeleton_dtype.as_str() {
        "fp16" => SkeletonDtype::Fp16,
        "fp32" => SkeletonDtype::Fp32,
        other => return Err(format!("bad skeleton-dtype '{other}'")),
    };
    Ok(CodecParams {
        variance_ratio: cfg.variance_ratio,
        k: cfg.k,
        bit_width: cfg.bit_width,
        rotation_seed: cfg.rotation_seed,
        kmeans_max_iter: 32,
        pca_method,
        skeleton_dtype,
        exact_rank_cap: None,
        custom_centroids: None,
        outlier_threshold: None,
        residual_besi: None,
    })
}

fn build_besi_params(cfg: &CodecConfig) -> Result<BesicovitchParams, String> {
    let mode = match cfg.besi_magnitude_mode.as_str() {
        "f16" => MagnitudeMode::F16,
        "quantized" => MagnitudeMode::QuantizedWithPerVectorScale,
        other => return Err(format!("bad besi-magnitude-mode '{other}'")),
    };
    Ok(BesicovitchParams {
        group_size: cfg.besi_group_size,
        direction_bits: cfg.besi_direction_bits,
        magnitude_bits: cfg.besi_magnitude_bits,
        magnitude_mode: mode,
        subtract_mean: cfg.besi_subtract_mean,
    })
}

fn mse_of(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum::<f64>()
        / n
}

fn encode_stream_kakeya(
    data: &[f32], n: usize, d: usize, block_size: usize,
    cfg: &CodecConfig, verify: bool,
) -> Result<(usize, f64), String> {
    let params = build_codec_params(cfg)?;
    let w = vec![1.0_f32; block_size];
    let mut total_bytes = 0usize;
    let mut mse_sum = 0.0_f64;
    let mut mse_n = 0usize;
    let mut idx = 0;
    while idx < n {
        let end = (idx + block_size).min(n);
        let n_block = end - idx;
        let block = &data[idx * d..end * d];
        let weights = &w[..n_block];

        let (skeleton, codes) = match cfg.metric.as_str() {
            "mse" => encode_block::<MSE>(block, weights, d, &params),
            "inner_product" => encode_block::<InnerProduct>(block, weights, d, &params),
            "linf" => encode_block::<LInf>(block, weights, d, &params),
            other => return Err(format!("bad metric '{other}'")),
        };

        // Byte accounting: skeleton + per-vector codes
        let sk_bytes = skeleton.nbytes();
        let code_bytes: usize = codes.iter().map(|c| c.nbytes()).sum();
        total_bytes += sk_bytes + code_bytes;

        if verify {
            let rec = match cfg.metric.as_str() {
                "mse" => decode_block::<MSE>(&skeleton, &codes),
                "inner_product" => decode_block::<InnerProduct>(&skeleton, &codes),
                "linf" => decode_block::<LInf>(&skeleton, &codes),
                _ => unreachable!(),
            };
            for i in 0..n_block {
                let a = &block[i * d..(i + 1) * d];
                let r = &rec[i * d..(i + 1) * d];
                mse_sum += mse_of(a, r);
                mse_n += 1;
            }
        }
        idx = end;
    }
    let mean_mse = if mse_n > 0 { mse_sum / mse_n as f64 } else { 0.0 };
    Ok((total_bytes, mean_mse))
}

fn encode_stream_besicovitch(
    data: &[f32], n: usize, d: usize, block_size: usize,
    cfg: &CodecConfig, verify: bool,
) -> Result<(usize, f64), String> {
    let params = build_besi_params(cfg)?;
    let mut total_bytes = 0usize;
    let mut mse_sum = 0.0_f64;
    let mut mse_n = 0usize;
    let mut idx = 0;
    while idx < n {
        let end = (idx + block_size).min(n);
        let n_block = end - idx;
        let block = &data[idx * d..end * d];

        let (codebook, skeleton, codes) = besicovitch_encode_block_full(block, d, &params);
        let sk_bytes = skeleton.nbytes();
        let code_bytes: usize = codes.iter().map(|c| c.nbytes(params.direction_bits)).sum();
        total_bytes += sk_bytes + code_bytes;

        if verify {
            let rec = besicovitch_decode_block_full(&codebook, &skeleton, &codes, d, &params);
            for i in 0..n_block {
                let a = &block[i * d..(i + 1) * d];
                let r = &rec[i * d..(i + 1) * d];
                mse_sum += mse_of(a, r);
                mse_n += 1;
            }
        }
        idx = end;
    }
    let mean_mse = if mse_n > 0 { mse_sum / mse_n as f64 } else { 0.0 };
    // Silence unused-decode-path warning when verify=false
    let _ = decode_block_with_centroids::<MSE>;
    Ok((total_bytes, mean_mse))
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[asymmetric-kv-bench] {e}");
            std::process::exit(2);
        }
    };

    let (nk, dk, k_data) = read_kktv(&args.k_input).unwrap_or_else(|e| {
        eprintln!("[asymmetric-kv-bench] K: {e}"); std::process::exit(1);
    });
    let (nv, dv, v_data) = read_kktv(&args.v_input).unwrap_or_else(|e| {
        eprintln!("[asymmetric-kv-bench] V: {e}"); std::process::exit(1);
    });
    if nk != nv || dk != dv {
        eprintln!("[asymmetric-kv-bench] K and V shapes must match: K=({nk},{dk}) V=({nv},{dv})");
        std::process::exit(1);
    }
    let n = nk; let d = dk;

    let t_k0 = Instant::now();
    let (k_bytes, k_mse) = match args.k_cfg.kind.as_str() {
        "kakeya" => encode_stream_kakeya(&k_data, n, d, args.block_size, &args.k_cfg, args.verify),
        "besicovitch" => encode_stream_besicovitch(&k_data, n, d, args.block_size, &args.k_cfg, args.verify),
        other => { eprintln!("bad --k-codec '{other}'"); std::process::exit(2); }
    }.unwrap_or_else(|e| { eprintln!("K encode failed: {e}"); std::process::exit(1); });
    let t_k_ns = t_k0.elapsed().as_nanos();

    let t_v0 = Instant::now();
    let (v_bytes, v_mse) = match args.v_cfg.kind.as_str() {
        "kakeya" => encode_stream_kakeya(&v_data, n, d, args.block_size, &args.v_cfg, args.verify),
        "besicovitch" => encode_stream_besicovitch(&v_data, n, d, args.block_size, &args.v_cfg, args.verify),
        other => { eprintln!("bad --v-codec '{other}'"); std::process::exit(2); }
    }.unwrap_or_else(|e| { eprintln!("V encode failed: {e}"); std::process::exit(1); });
    let t_v_ns = t_v0.elapsed().as_nanos();

    // 8-byte asymmetric-payload header (magic + version + 2 codec ids).
    let header_bytes: usize = 16;
    let total_bytes = header_bytes + k_bytes + v_bytes;
    let bf16_bytes = 2 * n * d * 2;
    let ratio = bf16_bytes as f64 / total_bytes.max(1) as f64;

    let report = format!(
        "{{\n  \"n_vectors\": {n},\n  \"dim\": {d},\n  \"block_size\": {bs},\n  \
         \"k_codec\": \"{kc}\",\n  \"v_codec\": \"{vc}\",\n  \
         \"k_bytes\": {kb},\n  \"v_bytes\": {vb},\n  \"header_bytes\": {hb},\n  \
         \"total_bytes\": {tb},\n  \"bf16_bytes\": {bfb},\n  \
         \"ratio_vs_bf16\": {ratio:.6},\n  \
         \"k_mean_block_mse\": {kmse},\n  \"v_mean_block_mse\": {vmse},\n  \
         \"t_k_ns\": {tkn},\n  \"t_v_ns\": {tvn}\n}}\n",
        bs = args.block_size, kc = args.k_cfg.kind, vc = args.v_cfg.kind,
        kb = k_bytes, vb = v_bytes, hb = header_bytes, tb = total_bytes,
        bfb = bf16_bytes, ratio = ratio, kmse = k_mse, vmse = v_mse,
        tkn = t_k_ns as u64, tvn = t_v_ns as u64,
    );
    std::fs::write(&args.output_report, report).unwrap_or_else(|e| {
        eprintln!("write report: {e}"); std::process::exit(1);
    });

    eprintln!(
        "[asymmetric-kv-bench] n={n} D={d} block={bs} \
         K={kc}({kb}B) V={vc}({vb}B) total={tb}B ratio={ratio:.2}x \
         K_MSE={kmse:.3e} V_MSE={vmse:.3e}",
        bs = args.block_size,
        kc = args.k_cfg.kind, kb = k_bytes,
        vc = args.v_cfg.kind, vb = v_bytes,
        tb = total_bytes, ratio = ratio,
        kmse = k_mse, vmse = v_mse,
    );
}
