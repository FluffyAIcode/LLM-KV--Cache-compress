//! K-stream cross-layer shared-basis ablation.
//!
//! For a given *model* (a set of K-stream tensors, one per full-attention
//! layer), compare three PCA-basis strategies through the full v1.2
//! codec pipeline (encode → decode → MSE):
//!
//! 1. **per_block** — fit a fresh PCA on every block (v1.2 K-path default).
//! 2. **per_layer_pooled** — one PCA per layer, reused across that layer's
//!    blocks (what V-stream already does in v1.2 via `share_basis=true`).
//! 3. **per_type_pooled** — one PCA for the entire model (one basis spanning
//!    all full-attention layers' K data), reused across every block.
//!
//! For each strategy the output reports, per-layer and aggregated:
//!
//! - mean reconstruction MSE (across all blocks of the layer / model)
//! - total skeleton bytes, total code bytes, total payload bytes
//! - MSE inflation vs. per_block baseline
//! - byte savings vs. per_block baseline
//!
//! Inputs are supplied via a manifest file (one line per layer, format
//! `layer_idx:path/to/tensor.kktv`). Every tensor must share the same
//! head dimension `d`.
//!
//! No mock, no fallback: all strategies run the exact monomorphic MSE
//! kernel used by the bench binary.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;

use kakeyaturbo::codec::{
    decode_block, decode_layer, encode_block, encode_layer, layer_nbytes, total_bytes, CodecParams,
};
use kakeyaturbo::distortion::MSE;

const MAGIC: u32 = 0x4B4B_5456;
const VERSION: u32 = 1;

struct Args {
    manifest: PathBuf,
    output: PathBuf,
    block_size: usize,
    variance_ratio: f32,
    k: usize,
    bit_width: u8,
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().collect();
    let mut manifest: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut block_size: usize = 512;
    let mut variance_ratio: f32 = 0.95;
    let mut k: usize = 16;
    let mut bit_width: u8 = 3;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--manifest" => { i += 1; manifest = Some(PathBuf::from(&argv[i])); }
            "--output" => { i += 1; output = Some(PathBuf::from(&argv[i])); }
            "--block-size" => { i += 1; block_size = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "--variance-ratio" => { i += 1; variance_ratio = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "--k" => { i += 1; k = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "--bit-width" => { i += 1; bit_width = argv[i].parse().map_err(|e| format!("{e}"))?; }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: kakeyaturbo-k-crosslayer-ablation --manifest FILE --output JSON \\\n    [--block-size 512] [--variance-ratio 0.95] [--k 16] [--bit-width 3]\n\nManifest lines: '<layer_idx>:<path>'"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}")),
        }
        i += 1;
    }
    Ok(Args {
        manifest: manifest.ok_or("--manifest required")?,
        output: output.ok_or("--output required")?,
        block_size, variance_ratio, k, bit_width,
    })
}

fn read_u32_le(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
fn read_u64_le(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}

fn read_tensor(path: &PathBuf) -> Result<(Vec<f32>, usize, usize), String> {
    let f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut r = BufReader::new(f);
    let m = read_u32_le(&mut r).map_err(|e| format!("magic: {e}"))?;
    if m != MAGIC { return Err(format!("bad magic {m:#x} in {}", path.display())); }
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

fn read_manifest(path: &PathBuf) -> Result<Vec<(u32, PathBuf)>, String> {
    let f = File::open(path).map_err(|e| format!("manifest {}: {e}", path.display()))?;
    let r = BufReader::new(f);
    let mut entries = Vec::new();
    for (line_no, line) in r.lines().enumerate() {
        let line = line.map_err(|e| format!("manifest read: {e}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') { continue; }
        let (idx_s, path_s) = trimmed.split_once(':')
            .ok_or_else(|| format!("line {}: expected 'idx:path'", line_no + 1))?;
        let idx: u32 = idx_s.trim().parse()
            .map_err(|e| format!("line {}: bad idx: {e}", line_no + 1))?;
        entries.push((idx, PathBuf::from(path_s.trim())));
    }
    Ok(entries)
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

/// Slice a row-major [n, d] flat buffer into blocks of size [block_size, d].
fn into_blocks(flat: Vec<f32>, d: usize, bs: usize) -> Vec<Vec<f32>> {
    let n = flat.len() / d;
    let n_blocks = n / bs;
    let mut out = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let off = b * bs * d;
        out.push(flat[off..off + bs * d].to_vec());
    }
    out
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => { eprintln!("error: {e}"); std::process::exit(2); }
    };

    let manifest = match read_manifest(&args.manifest) {
        Ok(m) => m,
        Err(e) => { eprintln!("error: {e}"); std::process::exit(3); }
    };
    if manifest.is_empty() {
        eprintln!("error: manifest empty");
        std::process::exit(4);
    }
    eprintln!("[xlayer] {} layers from {}", manifest.len(), args.manifest.display());

    // Load every tensor and dice into blocks.
    //
    // Structure we need:
    //   layers: Vec<(layer_idx, blocks: Vec<Vec<f32>>, ws: Vec<Vec<f32>>)>
    let params = CodecParams {
        variance_ratio: args.variance_ratio,
        k: args.k,
        bit_width: args.bit_width,
        rotation_seed: 0xCAFE_BABE,
        kmeans_max_iter: 32,
    };

    struct Layer {
        idx: u32,
        blocks: Vec<Vec<f32>>,
        ws: Vec<Vec<f32>>,
    }
    let mut layers: Vec<Layer> = Vec::with_capacity(manifest.len());
    let mut dim: Option<usize> = None;
    for (idx, p) in &manifest {
        let (flat, n, d) = match read_tensor(p) {
            Ok(t) => t,
            Err(e) => { eprintln!("error reading layer {idx}: {e}"); std::process::exit(5); }
        };
        if let Some(d0) = dim {
            if d != d0 { eprintln!("error: mixed head_dim {d} vs {d0}"); std::process::exit(6); }
        } else { dim = Some(d); }
        if n < args.block_size {
            eprintln!("  skip layer {idx}: only {n} rows, need ≥ {}", args.block_size);
            continue;
        }
        let blocks = into_blocks(flat, d, args.block_size);
        let ws: Vec<Vec<f32>> = (0..blocks.len()).map(|_| vec![1.0_f32; args.block_size]).collect();
        eprintln!("  L{idx:02}: n={n} → {} blocks", blocks.len());
        layers.push(Layer { idx: *idx, blocks, ws });
    }
    let d = dim.expect("dim set");
    if layers.is_empty() {
        eprintln!("error: no usable layers");
        std::process::exit(7);
    }

    // ------------------------------------------------------------------
    // Strategy 1: per_block (v1.2 default for K).
    // ------------------------------------------------------------------
    struct LayerStat {
        idx: u32,
        n_blocks: usize,
        mean_mse: f64,
        sk_bytes: usize,
        code_bytes: usize,
        payload_bytes: usize,
    }
    let mut per_block_stats: Vec<LayerStat> = Vec::with_capacity(layers.len());
    for lyr in &layers {
        let mut mses = Vec::with_capacity(lyr.blocks.len());
        let mut sk_total = 0usize;
        let mut code_total = 0usize;
        for (bvec, wvec) in lyr.blocks.iter().zip(&lyr.ws) {
            let (sk, codes) = encode_block::<MSE>(bvec, wvec, d, &params);
            let rec = decode_block::<MSE>(&sk, &codes);
            mses.push(block_mse(bvec, &rec));
            sk_total += sk.nbytes();
            code_total += codes.iter().map(|c| c.nbytes()).sum::<usize>();
            debug_assert_eq!(total_bytes(&sk, &codes), sk.nbytes() + codes.iter().map(|c| c.nbytes()).sum::<usize>());
        }
        per_block_stats.push(LayerStat {
            idx: lyr.idx,
            n_blocks: lyr.blocks.len(),
            mean_mse: mses.iter().sum::<f64>() / mses.len() as f64,
            sk_bytes: sk_total,
            code_bytes: code_total,
            payload_bytes: sk_total + code_total,
        });
    }

    // ------------------------------------------------------------------
    // Strategy 2: per_layer_pooled (share_basis=true per layer).
    // ------------------------------------------------------------------
    let mut per_layer_stats: Vec<LayerStat> = Vec::with_capacity(layers.len());
    for lyr in &layers {
        let enc = encode_layer::<MSE>(&lyr.blocks, &lyr.ws, d, &params, true);
        let recs = decode_layer::<MSE>(&enc);
        let mut mses = Vec::with_capacity(lyr.blocks.len());
        for (orig, rec) in lyr.blocks.iter().zip(&recs) {
            mses.push(block_mse(orig, rec));
        }
        let bytes = layer_nbytes(&enc);
        // Break down skeleton vs codes: skeleton = shared PCA + per-block K-means.
        let shared_pca_bytes = enc.shared_pca.as_ref().map(|p| p.nbytes()).unwrap_or(0);
        let kmeans_bytes: usize = enc.per_block.iter().map(|(sk, _)| sk.kmeans.nbytes()).sum();
        let code_bytes: usize = enc.per_block.iter()
            .map(|(_, codes)| codes.iter().map(|c| c.nbytes()).sum::<usize>()).sum();
        per_layer_stats.push(LayerStat {
            idx: lyr.idx,
            n_blocks: lyr.blocks.len(),
            mean_mse: mses.iter().sum::<f64>() / mses.len() as f64,
            sk_bytes: shared_pca_bytes + kmeans_bytes,
            code_bytes,
            payload_bytes: bytes,
        });
    }

    // ------------------------------------------------------------------
    // Strategy 3: per_type_pooled (one PCA across ALL layers' blocks).
    //
    // We achieve this by concatenating every layer's blocks into a single
    // encode_layer call with share_basis=true. The shared PCA is fit once
    // on the pooled data of every full-attention layer. K-means still
    // runs per block. Decoded reconstruction happens per block as usual.
    // ------------------------------------------------------------------
    // Build pooled block/weight lists, plus a per-entry (layer_idx,
    // block_idx) tag so we can split metrics back out by layer.
    let mut all_blocks: Vec<Vec<f32>> = Vec::new();
    let mut all_ws: Vec<Vec<f32>> = Vec::new();
    let mut tags: Vec<u32> = Vec::new();
    for lyr in &layers {
        for b in &lyr.blocks {
            all_blocks.push(b.clone());
        }
        for w in &lyr.ws {
            all_ws.push(w.clone());
        }
        tags.extend(std::iter::repeat(lyr.idx).take(lyr.blocks.len()));
    }
    eprintln!("[xlayer] pooled encode over {} blocks across {} layers", all_blocks.len(), layers.len());
    let pooled_enc = encode_layer::<MSE>(&all_blocks, &all_ws, d, &params, true);
    let pooled_recs = decode_layer::<MSE>(&pooled_enc);

    let shared_pca_bytes = pooled_enc.shared_pca.as_ref().map(|p| p.nbytes()).unwrap_or(0);
    // Build per-layer stats from the pooled decode.
    use std::collections::HashMap;
    let mut grouped: HashMap<u32, (Vec<f64>, usize, usize)> = HashMap::new();
    for (i, (orig, rec)) in all_blocks.iter().zip(&pooled_recs).enumerate() {
        let tag = tags[i];
        let (sk, codes) = &pooled_enc.per_block[i];
        let code_b = codes.iter().map(|c| c.nbytes()).sum::<usize>();
        let km_b = sk.kmeans.nbytes();
        let entry = grouped.entry(tag).or_insert((Vec::new(), 0, 0));
        entry.0.push(block_mse(orig, rec));
        entry.1 += km_b;          // per-block K-means bytes for this layer
        entry.2 += code_b;        // per-block code bytes for this layer
    }
    let mut per_type_stats: Vec<LayerStat> = Vec::with_capacity(layers.len());
    // Also distribute the one-shot shared_pca_bytes across layers
    // proportionally to their block counts (only matters for the
    // per-layer MSE-vs-bytes bookkeeping; totals are exact).
    let total_blocks: usize = layers.iter().map(|l| l.blocks.len()).sum();
    for lyr in &layers {
        let entry = grouped.remove(&lyr.idx).expect("layer stats");
        let mean_mse = entry.0.iter().sum::<f64>() / entry.0.len() as f64;
        let km_b = entry.1;
        let code_b = entry.2;
        let amortised_shared = if total_blocks > 0 {
            shared_pca_bytes * lyr.blocks.len() / total_blocks
        } else { 0 };
        let sk_total = km_b + amortised_shared;
        per_type_stats.push(LayerStat {
            idx: lyr.idx,
            n_blocks: lyr.blocks.len(),
            mean_mse,
            sk_bytes: sk_total,
            code_bytes: code_b,
            payload_bytes: sk_total + code_b,
        });
    }

    // Also compute strict model-level totals for strategy 3 (no amortisation).
    let per_type_total_bytes: usize = layer_nbytes(&pooled_enc);
    let per_type_total_code: usize = pooled_enc.per_block.iter()
        .map(|(_, codes)| codes.iter().map(|c| c.nbytes()).sum::<usize>()).sum();
    let per_type_total_km: usize = pooled_enc.per_block.iter().map(|(sk, _)| sk.kmeans.nbytes()).sum();

    // ------------------------------------------------------------------
    // Aggregate metrics and emit JSON.
    // ------------------------------------------------------------------
    let summary = |stats: &[LayerStat]| -> (f64, usize, usize, usize) {
        let mut mse_sum = 0.0_f64;
        let mut w_sum = 0usize;
        let mut sk = 0usize;
        let mut cd = 0usize;
        for s in stats {
            mse_sum += s.mean_mse * s.n_blocks as f64;
            w_sum += s.n_blocks;
            sk += s.sk_bytes;
            cd += s.code_bytes;
        }
        let mean = if w_sum > 0 { mse_sum / w_sum as f64 } else { 0.0 };
        (mean, sk, cd, sk + cd)
    };
    let s1 = summary(&per_block_stats);
    let s2 = summary(&per_layer_stats);
    // For strategy 3, prefer exact totals (no amortisation rounding).
    let mut mse_sum3 = 0.0_f64;
    let mut w_sum3 = 0usize;
    for s in &per_type_stats { mse_sum3 += s.mean_mse * s.n_blocks as f64; w_sum3 += s.n_blocks; }
    let s3_mean = if w_sum3 > 0 { mse_sum3 / w_sum3 as f64 } else { 0.0 };
    let s3 = (s3_mean, shared_pca_bytes + per_type_total_km, per_type_total_code, per_type_total_bytes);

    eprintln!("[xlayer] per_block         : mean_mse={:.3e}  bytes={}", s1.0, s1.3);
    eprintln!("[xlayer] per_layer_pooled  : mean_mse={:.3e}  bytes={}", s2.0, s2.3);
    eprintln!("[xlayer] per_type_pooled   : mean_mse={:.3e}  bytes={}", s3.0, s3.3);

    // JSON emitter.
    let stats_to_json = |stats: &[LayerStat]| -> String {
        let mut out = String::new();
        out.push('[');
        for (i, s) in stats.iter().enumerate() {
            if i > 0 { out.push(','); }
            out.push_str(&format!(
                "{{\"layer_idx\":{},\"n_blocks\":{},\"mean_mse\":{:.6e},\"skeleton_bytes\":{},\
                \"code_bytes\":{},\"payload_bytes\":{}}}",
                s.idx, s.n_blocks, s.mean_mse, s.sk_bytes, s.code_bytes, s.payload_bytes,
            ));
        }
        out.push(']');
        out
    };

    let mut json = String::new();
    json.push_str(&format!(
        "{{\"dim\":{},\"block_size\":{},\"variance_ratio\":{},\"k\":{},\"bit_width\":{},\"num_layers\":{},",
        d, args.block_size, args.variance_ratio, args.k, args.bit_width, layers.len()
    ));
    // Include pooled PCA d_eff info for strategy 3 so we can report it.
    let pooled_d_eff = pooled_enc.shared_pca.as_ref().map(|p| p.d_eff).unwrap_or(0);
    json.push_str(&format!("\"per_type_pooled_d_eff\":{pooled_d_eff},"));
    json.push_str(&format!("\"per_type_pooled_pca_bytes\":{shared_pca_bytes},"));

    // Per-strategy aggregate numbers + per-layer arrays.
    let inflation2 = s2.0 / s1.0.max(f64::EPSILON);
    let inflation3 = s3.0 / s1.0.max(f64::EPSILON);
    let byte_ratio2 = s2.3 as f64 / s1.3.max(1) as f64;
    let byte_ratio3 = s3.3 as f64 / s1.3.max(1) as f64;

    json.push_str(&format!(
        "\"per_block\":{{\"mean_mse\":{:.6e},\"total_skeleton_bytes\":{},\"total_code_bytes\":{},\
         \"total_payload_bytes\":{},\"per_layer\":{}}},",
        s1.0, s1.1, s1.2, s1.3, stats_to_json(&per_block_stats)
    ));
    json.push_str(&format!(
        "\"per_layer_pooled\":{{\"mean_mse\":{:.6e},\"total_skeleton_bytes\":{},\"total_code_bytes\":{},\
         \"total_payload_bytes\":{},\"mse_inflation_vs_per_block\":{:.4},\
         \"bytes_vs_per_block\":{:.4},\"per_layer\":{}}},",
        s2.0, s2.1, s2.2, s2.3, inflation2, byte_ratio2, stats_to_json(&per_layer_stats)
    ));
    json.push_str(&format!(
        "\"per_type_pooled\":{{\"mean_mse\":{:.6e},\"total_skeleton_bytes\":{},\"total_code_bytes\":{},\
         \"total_payload_bytes\":{},\"mse_inflation_vs_per_block\":{:.4},\
         \"bytes_vs_per_block\":{:.4},\"per_layer\":{}}}",
        s3.0, s3.1, s3.2, s3.3, inflation3, byte_ratio3, stats_to_json(&per_type_stats)
    ));
    json.push('}');
    std::fs::write(&args.output, &json).expect("write output");
    eprintln!("[xlayer] wrote {}", args.output.display());
}
