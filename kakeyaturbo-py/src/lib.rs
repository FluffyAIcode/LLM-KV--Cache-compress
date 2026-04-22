//! In-process pyo3 bindings for the kakeyaturbo reference codec.
//!
//! Exposes a single Python entry point `roundtrip_layer` that is
//! functionally identical to the `kakeyaturbo-bench` CLI invoked with
//! `--verify --dump-decoded`, minus the KKTV disk I/O and the
//! subprocess fork.
//!
//! All codec semantics are delegated to the existing `kakeyaturbo`
//! crate; this file is glue only. Compile-time lint `unsafe_code =
//! forbid` guards the glue against introducing unsound borrows when
//! translating numpy views into `&[f32]` slices.

#![forbid(unsafe_code)]

use kakeyaturbo::{
    decode_block_with_centroids, decode_layer_with_centroids, encode_block, encode_layer, Code,
    CodecParams, InnerProduct, LInf, LayerEncoding, PcaMethod, SkeletonDtype, MSE,
};
use kakeyaturbo::Distortion;
use numpy::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Parsed PCA method selector (drives generic dispatch).
#[derive(Debug, Clone, Copy)]
enum PcaKind {
    Exact,
    Randomized {
        target_rank: usize,
        oversample: usize,
        power_iters: u32,
    },
}

/// Parsed skeleton dtype selector.
#[derive(Debug, Clone, Copy)]
enum SkelKind {
    Fp16,
    Fp32,
}

/// Parsed metric selector.
#[derive(Debug, Clone, Copy)]
enum MetricKind {
    Mse,
    InnerProduct,
    LInf,
}

impl MetricKind {
    fn parse(s: &str) -> PyResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "mse" => Ok(MetricKind::Mse),
            "inner_product" | "ip" => Ok(MetricKind::InnerProduct),
            "linf" => Ok(MetricKind::LInf),
            other => Err(PyValueError::new_err(format!(
                "unknown metric '{other}', expected 'mse' | 'inner_product' | 'linf'"
            ))),
        }
    }
}

/// Parsed, owned form of the kwargs.  All `Option<..>` fields preserve
/// the CLI's "missing means default" semantics.  Parsing is done with
/// the GIL held; the actual codec run releases it.
struct ParsedArgs {
    block_size: usize,
    dim: usize,
    num_vecs: usize,
    variance_ratio: f32,
    k: usize,
    bit_width: u8,
    rotation_seed: u32,
    kmeans_max_iter: u32,
    pca: PcaKind,
    skeleton_dtype: SkelKind,
    exact_rank_cap: Option<usize>,
    custom_centroids: Option<Vec<f32>>,
    outlier_threshold: Option<f32>,
    metric: MetricKind,
    share_basis: bool,
    /// Flattened row-major contiguous copy of the caller's array.
    /// Decoupled from the numpy buffer so the hot path can drop the GIL.
    input: Vec<f32>,
}

fn parse_args(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f32>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<ParsedArgs> {
    let _ = py;
    let shape = array.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("array must be 2-D (num_vecs, dim)"));
    }
    let num_vecs = shape[0];
    let dim = shape[1];

    // Force a contiguous copy via `as_slice()?`.  numpy 0.28 returns an
    // error if the array is non-contiguous rather than silently aliasing,
    // so this both validates and materialises the Rust-side buffer.
    let slice = array
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("array must be C-contiguous: {e}")))?;
    if slice.len() != num_vecs * dim {
        return Err(PyRuntimeError::new_err(format!(
            "internal: slice len {} != num_vecs {} × dim {} = {}",
            slice.len(),
            num_vecs,
            dim,
            num_vecs * dim
        )));
    }
    let input = slice.to_vec();

    // ---------------- kwargs parsing ----------------
    // Single helper that works for every primitive T we need to pull
    // out of `kwargs`.  pyo3 0.28 introduced an associated `Error` type
    // on `FromPyObject`; we bound `Error: Into<PyErr>` so `?` can lift
    // conversion errors into the surrounding `PyResult`.
    fn get<'py, T>(kw: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>>
    where
        T: for<'a> FromPyObject<'a, 'py>,
        for<'a> <T as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
    {
        match kw.get_item(key)? {
            Some(v) => v.extract::<T>().map(Some).map_err(Into::into),
            None => Ok(None),
        }
    }

    let block_size: usize = get::<usize>(kwargs, "block_size")?.unwrap_or(512);
    let variance_ratio: f32 = get::<f32>(kwargs, "variance_ratio")?.unwrap_or(0.95);
    let k: usize = get::<usize>(kwargs, "k")?.unwrap_or(16);
    let bit_width: u8 = {
        let raw: i64 = get::<i64>(kwargs, "bit_width")?.unwrap_or(3);
        if !(1..=4).contains(&raw) {
            return Err(PyValueError::new_err(format!(
                "bit_width must be in 1..=4, got {raw}"
            )));
        }
        raw as u8
    };
    let rotation_seed: u32 = get::<u32>(kwargs, "rotation_seed")?.unwrap_or(0xCAFE_BABE);
    let kmeans_max_iter: u32 = get::<u32>(kwargs, "kmeans_max_iter")?.unwrap_or(32);

    let pca_method_s: String =
        get::<String>(kwargs, "pca_method")?.unwrap_or_else(|| "exact".to_string());
    let pca = match pca_method_s.as_str() {
        "exact" => PcaKind::Exact,
        "randomized" => {
            let target_rank: usize =
                get::<usize>(kwargs, "rsvd_target_rank")?.unwrap_or((dim / 2).max(8));
            let oversample: usize = get::<usize>(kwargs, "rsvd_oversample")?.unwrap_or(8);
            let power_iters: u32 = get::<u32>(kwargs, "rsvd_power_iters")?.unwrap_or(2);
            PcaKind::Randomized {
                target_rank,
                oversample,
                power_iters,
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown pca_method '{other}', expected 'exact' | 'randomized'"
            )))
        }
    };

    let skel_s: String =
        get::<String>(kwargs, "skeleton_dtype")?.unwrap_or_else(|| "fp16".to_string());
    let skeleton_dtype = match skel_s.as_str() {
        "fp16" | "f16" | "half" => SkelKind::Fp16,
        "fp32" | "f32" | "float" => SkelKind::Fp32,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown skeleton_dtype '{other}', expected 'fp16' | 'fp32'"
            )))
        }
    };

    let exact_rank_cap: Option<usize> = get::<usize>(kwargs, "exact_rank_cap")?;
    let outlier_threshold: Option<f32> = get::<f32>(kwargs, "outlier_threshold")?;

    // Centroids: accept either a Python sequence of floats, or a
    // filesystem path (matching the CLI's `--centroids-file` byte format).
    let custom_centroids: Option<Vec<f32>> = if let Some(item) = kwargs.get_item("centroids")? {
        if item.is_none() {
            None
        } else if let Ok(list) = item.cast::<PyList>() {
            let mut out = Vec::with_capacity(list.len());
            for v in list.iter() {
                out.push(v.extract::<f32>()?);
            }
            validate_centroids(&out, bit_width)?;
            Some(out)
        } else if let Ok(seq) = item.extract::<Vec<f32>>() {
            validate_centroids(&seq, bit_width)?;
            Some(seq)
        } else {
            return Err(PyTypeError::new_err(
                "centroids must be a list/sequence of floats, or None",
            ));
        }
    } else if let Some(path_item) = kwargs.get_item("centroids_file")? {
        if path_item.is_none() {
            None
        } else {
            let path: String = path_item.extract()?;
            let data = load_centroids_file(&path, 1usize << bit_width)
                .map_err(PyValueError::new_err)?;
            Some(data)
        }
    } else {
        None
    };

    let metric_s: String = get::<String>(kwargs, "metric")?.ok_or_else(|| {
        PyValueError::new_err("missing required kwarg 'metric'")
    })?;
    let metric = MetricKind::parse(&metric_s)?;
    let share_basis: bool = get::<bool>(kwargs, "share_basis")?.unwrap_or(false);

    Ok(ParsedArgs {
        block_size,
        dim,
        num_vecs,
        variance_ratio,
        k,
        bit_width,
        rotation_seed,
        kmeans_max_iter,
        pca,
        skeleton_dtype,
        exact_rank_cap,
        custom_centroids,
        outlier_threshold,
        metric,
        share_basis,
        input,
    })
}

fn validate_centroids(c: &[f32], bit_width: u8) -> PyResult<()> {
    let expected = 1usize << bit_width;
    if c.len() != expected {
        return Err(PyValueError::new_err(format!(
            "centroids has {} entries, expected {} (= 1 << bit_width={})",
            c.len(),
            expected,
            bit_width
        )));
    }
    for w in c.windows(2) {
        if !(w[0] < w[1]) {
            return Err(PyValueError::new_err(format!(
                "centroids must be strictly ascending; violation: {} >= {}",
                w[0], w[1]
            )));
        }
    }
    Ok(())
}

fn load_centroids_file(path: &str, expected_count: usize) -> Result<Vec<f32>, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("reading centroids file {path}: {e}"))?;
    if bytes.len() != expected_count * 4 {
        return Err(format!(
            "centroids file {} has {} bytes, expected {} (= {} × 4)",
            path,
            bytes.len(),
            expected_count * 4,
            expected_count
        ));
    }
    let mut out = Vec::with_capacity(expected_count);
    for chunk in bytes.chunks_exact(4) {
        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out.push(f32::from_le_bytes(arr));
    }
    for w in out.windows(2) {
        if !(w[0] < w[1]) {
            return Err(format!(
                "centroids must be strictly ascending; {} violates at {} >= {}",
                path, w[0], w[1]
            ));
        }
    }
    Ok(out)
}

fn to_pca(pca: PcaKind) -> PcaMethod {
    match pca {
        PcaKind::Exact => PcaMethod::Exact,
        PcaKind::Randomized {
            target_rank,
            oversample,
            power_iters,
        } => PcaMethod::Randomized {
            target_rank,
            oversample,
            power_iters,
            // Same salt used by kakeyaturbo-bench::run — keeping both
            // paths byte-identical when --pca-method randomized is set.
            seed_offset: 0x9E37_79B9_7F4A_7C15,
        },
    }
}

fn to_skel(skel: SkelKind) -> SkeletonDtype {
    match skel {
        SkelKind::Fp16 => SkeletonDtype::Fp16,
        SkelKind::Fp32 => SkeletonDtype::Fp32,
    }
}

/// Output of a single `roundtrip_layer` call — decoded vectors + the
/// scalar report, in exactly the fields the CLI emits.  `encode_ns`
/// and `decode_ns` are split so callers can attribute cost.
struct RoundtripOutput {
    decoded: Vec<f32>,
    /// Number of vectors encoded / decoded (full blocks only; trailing
    /// partial block is passed through untouched by the caller, matching
    /// the subprocess harness's behaviour).
    num_vecs_encoded: usize,
    /// Number of full codec blocks processed.
    num_blocks: usize,
    /// Skeleton bytes (PCA mean + basis + K-means centres + per-block
    /// metadata that the codec emits).
    skeleton_bytes: usize,
    /// Per-vector code bytes (seg_id + t + norm + residual packed +
    /// outliers).
    codes_bytes: usize,
    shared_pca_bytes: usize,
    mean_block_mse: f64,
    encode_ns: u128,
    decode_ns: u128,
}

/// Generic runner parameterised by the distortion type.  The two
/// alternative code paths (per-block vs share-basis) mirror the
/// bench-binary's `run` exactly.
fn run_roundtrip<R: Distortion>(args: &ParsedArgs) -> RoundtripOutput {
    let params = CodecParams {
        variance_ratio: args.variance_ratio,
        k: args.k,
        bit_width: args.bit_width,
        rotation_seed: args.rotation_seed,
        kmeans_max_iter: args.kmeans_max_iter,
        pca_method: to_pca(args.pca),
        skeleton_dtype: to_skel(args.skeleton_dtype),
        exact_rank_cap: args.exact_rank_cap,
        custom_centroids: args.custom_centroids.clone(),
        outlier_threshold: args.outlier_threshold,
    };

    let bs = args.block_size;
    let dim = args.dim;
    let n_full = args.num_vecs / bs;
    let weights = vec![1.0_f32; bs];

    let mut encode_ns: u128 = 0;
    let mut decode_ns: u128 = 0;
    let mut mse_sum: f64 = 0.0;
    let mut mse_count: usize = 0;
    let mut decoded_full: Vec<f32> = Vec::with_capacity(n_full * bs * dim);

    if args.share_basis {
        let mut block_vecs: Vec<Vec<f32>> = Vec::with_capacity(n_full);
        let mut ws: Vec<Vec<f32>> = Vec::with_capacity(n_full);
        for b in 0..n_full {
            let off = b * bs * dim;
            block_vecs.push(args.input[off..off + bs * dim].to_vec());
            ws.push(weights.clone());
        }
        let t0 = std::time::Instant::now();
        let enc: LayerEncoding = encode_layer::<R>(&block_vecs, &ws, dim, &params, true);
        encode_ns += t0.elapsed().as_nanos();

        let t1 = std::time::Instant::now();
        let recs = decode_layer_with_centroids::<R>(&enc, params.custom_centroids.as_deref());
        decode_ns += t1.elapsed().as_nanos();

        for (i, rec) in recs.iter().enumerate() {
            let orig = &block_vecs[i];
            let mut sq = 0.0_f64;
            for j in 0..bs * dim {
                let e = orig[j] - rec[j];
                sq += (e * e) as f64;
            }
            mse_sum += sq / (bs * dim) as f64;
            mse_count += 1;
            decoded_full.extend_from_slice(rec);
        }

        let shared_pca_bytes = enc.shared_pca.as_ref().map(|p| p.nbytes()).unwrap_or(0);
        let per_block_skel: usize = enc.per_block.iter().map(|(sk, _)| sk.kmeans.nbytes()).sum();
        let codes_total: usize = enc
            .per_block
            .iter()
            .flat_map(|(_, cs)| cs.iter())
            .map(|c| c.nbytes())
            .sum();
        let skeleton_total = shared_pca_bytes + per_block_skel;
        let num_blocks = enc.per_block.len();

        RoundtripOutput {
            decoded: decoded_full,
            num_vecs_encoded: num_blocks * bs,
            num_blocks,
            skeleton_bytes: skeleton_total,
            codes_bytes: codes_total,
            shared_pca_bytes,
            mean_block_mse: if mse_count > 0 {
                mse_sum / mse_count as f64
            } else {
                -1.0
            },
            encode_ns,
            decode_ns,
        }
    } else {
        let mut total_skeleton = 0usize;
        let mut total_codes = 0usize;
        let mut total_blocks = 0usize;
        let mut total_vecs_encoded = 0usize;

        for b in 0..n_full {
            let off = b * bs * dim;
            let block = &args.input[off..off + bs * dim];
            let t0 = std::time::Instant::now();
            let (sk, codes) = encode_block::<R>(block, &weights, dim, &params);
            encode_ns += t0.elapsed().as_nanos();
            total_skeleton += sk.nbytes();
            total_codes += codes.iter().map(|c: &Code| c.nbytes()).sum::<usize>();
            total_blocks += 1;
            total_vecs_encoded += bs;

            let t1 = std::time::Instant::now();
            let rec = decode_block_with_centroids::<R>(
                &sk,
                &codes,
                params.custom_centroids.as_deref(),
            );
            decode_ns += t1.elapsed().as_nanos();
            let mut sq = 0.0_f64;
            for i in 0..bs * dim {
                let e = block[i] - rec[i];
                sq += (e * e) as f64;
            }
            mse_sum += sq / (bs * dim) as f64;
            mse_count += 1;
            decoded_full.extend_from_slice(&rec);
        }

        RoundtripOutput {
            decoded: decoded_full,
            num_vecs_encoded: total_vecs_encoded,
            num_blocks: total_blocks,
            skeleton_bytes: total_skeleton,
            codes_bytes: total_codes,
            shared_pca_bytes: 0,
            mean_block_mse: if mse_count > 0 {
                mse_sum / mse_count as f64
            } else {
                -1.0
            },
            encode_ns,
            decode_ns,
        }
    }
}

/// The one and only public Python entry point.
///
/// Signature (keyword-only; positional is the input array):
///
/// ```text
/// decoded, report = roundtrip_layer(
///     array,                         # ndarray[float32, shape=(N, D)], C-contig
///     *,
///     metric: str,                   # "mse" | "inner_product" | "linf"
///     block_size: int = 512,
///     bit_width: int = 3,            # 1..=4
///     variance_ratio: float = 0.95,
///     k: int = 16,
///     rotation_seed: int = 0xCAFEBABE,
///     kmeans_max_iter: int = 32,
///     pca_method: str = "exact",     # "exact" | "randomized"
///     rsvd_target_rank: Optional[int] = None,  # default max(D/2, 8)
///     rsvd_oversample: int = 8,
///     rsvd_power_iters: int = 2,
///     skeleton_dtype: str = "fp16",  # "fp16" | "fp32"
///     exact_rank_cap: Optional[int] = None,
///     centroids: Optional[Sequence[float]] = None,
///     centroids_file: Optional[str] = None,
///     outlier_threshold: Optional[float] = None,
///     share_basis: bool = False,
/// )
/// ```
///
/// Returns `(decoded, report)` where:
/// - `decoded`  — a freshly-allocated `ndarray[float32, shape=(M, D)]`
///                with `M = (N // block_size) * block_size`.  If `N`
///                wasn't a multiple of `block_size`, the trailing
///                `N - M` vectors are the caller's problem (matches
///                the subprocess harness's `concatenate([dec,
///                arr[n_compressible:]])` pattern).
/// - `report`   — a dict with keys matching the JSON emitted by the
///                `kakeyaturbo-bench` CLI, minus timing-only
///                nice-to-haves.  Sufficient for the validation
///                harness's bookkeeping.
#[pyfunction]
#[pyo3(signature = (array, **kwargs))]
fn roundtrip_layer<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyDict>)> {
    let kw = kwargs.ok_or_else(|| PyValueError::new_err("missing kwargs (e.g. metric='mse')"))?;
    let parsed = parse_args(py, array, kw)?;

    // Drop the GIL while the codec runs (pyo3 0.28 renamed the API to
    // `Python::detach`; semantics identical to the old `allow_threads`).
    let (out, metric_name) = py.detach(|| {
        let m = parsed.metric;
        let r = match m {
            MetricKind::Mse => run_roundtrip::<MSE>(&parsed),
            MetricKind::InnerProduct => run_roundtrip::<InnerProduct>(&parsed),
            MetricKind::LInf => run_roundtrip::<LInf>(&parsed),
        };
        let name = match m {
            MetricKind::Mse => "mse",
            MetricKind::InnerProduct => "inner_product",
            MetricKind::LInf => "linf",
        };
        (r, name)
    });

    // Shape the decoded ndarray back to [M, D].  Build a PyArray2
    // directly from the owned Vec<f32> without going through ndarray
    // (numpy 0.28 re-exports its own ndarray and the direct version
    // feature-graph is brittle).  `PyArray2::from_vec2_bound`-style
    // constructors are gone in 0.28; the supported path is
    // `numpy::PyArray1::from_vec` + `.reshape`.
    let m_rows = out.num_vecs_encoded;
    let d = parsed.dim;
    let flat = numpy::PyArray1::<f32>::from_vec(py, out.decoded);
    let decoded_np = flat.reshape([m_rows, d]).map_err(|e| {
        PyRuntimeError::new_err(format!("reshape decoded to ({m_rows}, {d}): {e}"))
    })?;

    let report = PyDict::new(py);
    report.set_item("metric", metric_name)?;
    report.set_item("block_size", parsed.block_size)?;
    report.set_item("variance_ratio", parsed.variance_ratio)?;
    report.set_item("k", parsed.k)?;
    report.set_item("bit_width", parsed.bit_width)?;
    report.set_item("dim", parsed.dim)?;
    report.set_item("num_vecs_encoded", out.num_vecs_encoded)?;
    report.set_item("num_blocks", out.num_blocks)?;
    report.set_item("skeleton_bytes", out.skeleton_bytes)?;
    report.set_item("codes_bytes", out.codes_bytes)?;
    let compressed_bytes = out.skeleton_bytes + out.codes_bytes;
    report.set_item("compressed_bytes", compressed_bytes)?;
    let baseline_bytes_f32 = out.num_vecs_encoded * parsed.dim * 4;
    let baseline_bytes_bf16 = out.num_vecs_encoded * parsed.dim * 2;
    report.set_item("baseline_bytes_f32", baseline_bytes_f32)?;
    report.set_item("baseline_bytes_bf16", baseline_bytes_bf16)?;
    if compressed_bytes > 0 {
        report.set_item(
            "ratio_vs_f32",
            baseline_bytes_f32 as f64 / compressed_bytes as f64,
        )?;
        report.set_item(
            "ratio_vs_bf16",
            baseline_bytes_bf16 as f64 / compressed_bytes as f64,
        )?;
    } else {
        report.set_item("ratio_vs_f32", 0.0f64)?;
        report.set_item("ratio_vs_bf16", 0.0f64)?;
    }
    report.set_item("encode_seconds", out.encode_ns as f64 / 1e9)?;
    report.set_item("decode_seconds", out.decode_ns as f64 / 1e9)?;
    report.set_item("verify", true)?;
    report.set_item("mean_block_mse", out.mean_block_mse)?;
    report.set_item("share_basis", parsed.share_basis)?;
    report.set_item("shared_pca_bytes", out.shared_pca_bytes)?;
    report.set_item(
        "pca_method",
        match parsed.pca {
            PcaKind::Exact => "exact",
            PcaKind::Randomized { .. } => "randomized",
        },
    )?;

    Ok((decoded_np, report))
}

// ---------------------------------------------------------------------------
// Primitive helpers for M4 — expose the handful of deterministic-random
// artefacts and low-level kernels the Rust codec uses so the PyTorch
// reference encoder can be byte-identical on the operations that
// *matter* for downstream attention (decoded tensor equality).
//
// These functions are not required for `roundtrip_layer` to work;
// they are the minimal oracle surface for validating Triton kernels
// against the Rust numeric contract.
// ---------------------------------------------------------------------------

/// WHT sign pattern for `(seed, n)`, matching Rust's
/// `kakeyaturbo::wht::sign_pattern`.
///
/// Returns `[+1.0 | -1.0]` of length `n`.  `n` must be a power of two.
/// Uses Rust's `SmallRng::seed_from_u64(seed as u64).gen::<bool>()`
/// internally, so the Python side gets the exact pattern the codec
/// will use — no need to reimplement the xoshiro stream in NumPy.
#[pyfunction]
fn wht_sign_pattern<'py>(
    py: Python<'py>,
    seed: u32,
    n: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Dimension note: we return shape (1, n) rather than (n,) so the
    // caller can broadcast against a (batch, n) residual with zero
    // reshaping friction; flatten on the numpy side if 1-D is wanted.
    let signs = kakeyaturbo::wht::sign_pattern(seed, n);
    let arr = numpy::PyArray1::<f32>::from_vec(py, signs);
    arr.reshape([1usize, n])
        .map_err(|e| PyRuntimeError::new_err(format!("reshape sign_pattern: {e}")))
}

/// Rust's unnormalised Walsh-Hadamard transform applied to each row of
/// `x: [batch, n]`.  `n` must be a power of two.
///
/// This is the byte-exact reference for any Triton / CUDA WHT
/// implementation: `y[b] = wht_inplace_unnormalised(x[b])`.
#[pyfunction]
fn wht_rows<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = x.shape().to_vec();
    let batch = shape[0];
    let n = shape[1];
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(PyValueError::new_err(format!(
            "n must be a positive power of two, got {n}"
        )));
    }
    let slice = x
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("array must be C-contiguous: {e}")))?;
    let mut out: Vec<f32> = slice.to_vec();
    py.detach(|| {
        for row in out.chunks_mut(n) {
            kakeyaturbo::wht::wht_inplace(row);
        }
    });
    let flat = numpy::PyArray1::<f32>::from_vec(py, out);
    flat.reshape([batch, n])
        .map_err(|e| PyRuntimeError::new_err(format!("reshape wht_rows: {e}")))
}

/// Rust's `kakeyaturbo::wht::rotate` (y = H·D·x), row-wise.
#[pyfunction]
fn rotate_rows<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    seed: u32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = x.shape().to_vec();
    let batch = shape[0];
    let n = shape[1];
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(PyValueError::new_err(format!(
            "n must be a positive power of two, got {n}"
        )));
    }
    let slice = x
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("array must be C-contiguous: {e}")))?;
    let input: Vec<f32> = slice.to_vec();
    let out = py.detach(|| {
        let mut buf = Vec::with_capacity(batch * n);
        for row in input.chunks(n) {
            buf.extend_from_slice(&kakeyaturbo::wht::rotate(row, seed));
        }
        buf
    });
    let flat = numpy::PyArray1::<f32>::from_vec(py, out);
    flat.reshape([batch, n])
        .map_err(|e| PyRuntimeError::new_err(format!("reshape rotate_rows: {e}")))
}

/// Rust's `kakeyaturbo::wht::inverse_rotate` (x = D·H·y / n), row-wise.
#[pyfunction]
fn inverse_rotate_rows<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f32>,
    seed: u32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = y.shape().to_vec();
    let batch = shape[0];
    let n = shape[1];
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(PyValueError::new_err(format!(
            "n must be a positive power of two, got {n}"
        )));
    }
    let slice = y
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("array must be C-contiguous: {e}")))?;
    let input: Vec<f32> = slice.to_vec();
    let out = py.detach(|| {
        let mut buf = Vec::with_capacity(batch * n);
        for row in input.chunks(n) {
            buf.extend_from_slice(&kakeyaturbo::wht::inverse_rotate(row, seed));
        }
        buf
    });
    let flat = numpy::PyArray1::<f32>::from_vec(py, out);
    flat.reshape([batch, n])
        .map_err(|e| PyRuntimeError::new_err(format!("reshape inverse_rotate_rows: {e}")))
}

/// Rust's `kakeyaturbo::quantize::pack_bits`.
///
/// `indices`: 1-D uint8, every element in `0..(1 << bits)`.
/// `bits`: 1..=8.
/// Returns a 1-D uint8 packed byte stream of length
/// `ceil(indices.len() * bits / 8)`.
#[pyfunction]
fn pack_bits<'py>(
    py: Python<'py>,
    indices: numpy::PyReadonlyArray1<'py, u8>,
    bits: u8,
) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
    if !(1..=8).contains(&bits) {
        return Err(PyValueError::new_err(format!(
            "bits must be 1..=8, got {bits}"
        )));
    }
    let slice = indices.as_slice().map_err(|e| {
        PyValueError::new_err(format!("indices must be C-contiguous: {e}"))
    })?;
    let max_idx = (1u16 << bits) - 1;
    if let Some(bad) = slice.iter().find(|&&v| u16::from(v) > max_idx) {
        return Err(PyValueError::new_err(format!(
            "index {bad} exceeds {bits}-bit range"
        )));
    }
    let input: Vec<u8> = slice.to_vec();
    let packed = py.detach(|| kakeyaturbo::quantize::pack_bits(&input, bits));
    Ok(numpy::PyArray1::<u8>::from_vec(py, packed))
}

/// Rust's `kakeyaturbo::quantize::unpack_bits`.
///
/// `bytes`: 1-D uint8 packed stream.
/// `bits`: 1..=8.
/// `count`: number of indices to recover (must match how the stream was
/// produced).
#[pyfunction]
fn unpack_bits<'py>(
    py: Python<'py>,
    bytes: numpy::PyReadonlyArray1<'py, u8>,
    bits: u8,
    count: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
    if !(1..=8).contains(&bits) {
        return Err(PyValueError::new_err(format!(
            "bits must be 1..=8, got {bits}"
        )));
    }
    let need = (count * bits as usize + 7) / 8;
    let slice = bytes.as_slice().map_err(|e| {
        PyValueError::new_err(format!("bytes must be C-contiguous: {e}"))
    })?;
    if slice.len() < need {
        return Err(PyValueError::new_err(format!(
            "bytes too short: got {}, need at least {} for count={count} at bits={bits}",
            slice.len(),
            need
        )));
    }
    let input: Vec<u8> = slice[..need].to_vec();
    let out = py.detach(|| kakeyaturbo::quantize::unpack_bits(&input, bits, count));
    Ok(numpy::PyArray1::<u8>::from_vec(py, out))
}

/// Return the fixed Gaussian Lloyd-Max centroids for `bits` in 1..=4.
/// Mirrors `kakeyaturbo::quantize::centroids_gaussian`.
#[pyfunction]
fn centroids_gaussian<'py>(
    py: Python<'py>,
    bits: u8,
) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    if !(1..=4).contains(&bits) {
        return Err(PyValueError::new_err(format!(
            "bits must be 1..=4, got {bits}"
        )));
    }
    let slice = kakeyaturbo::quantize::centroids_gaussian(bits).to_vec();
    Ok(numpy::PyArray1::<f32>::from_vec(py, slice))
}

// ---------------------------------------------------------------------------
// encode_block_codes / decode_block_from_parts
//
// These expose the structured output of `kakeyaturbo::codec::encode_block`
// (skeleton + per-vector codes) so the Python-side M4 reference can:
//   (a) consume the same skeleton Rust fit (byte-identical PCA / K-means)
//   (b) run stages 2..=5 in Torch/Triton and compare codes / decoded
//       tensors bit-exactly against Rust.
//
// The return format is a flat dict of numpy arrays. Caller reconstructs
// per-vector views by indexing.
// ---------------------------------------------------------------------------

/// Drive `kakeyaturbo::codec::encode_block` on a single block and
/// return its skeleton + codes as numpy arrays.  Phase A.1 scope:
/// `metric ∈ {mse, inner_product, linf}`, `pca ∈ {exact, randomized}`,
/// `share_basis` **always False** here (share-basis is a
/// `encode_layer` concept — use `roundtrip_layer` for that).
#[pyfunction]
#[pyo3(signature = (array, **kwargs))]
fn encode_block_codes<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let kw = kwargs.ok_or_else(|| PyValueError::new_err(
        "missing kwargs (e.g. metric='mse')"))?;
    let parsed = parse_args(py, array, kw)?;
    if parsed.num_vecs != parsed.block_size {
        return Err(PyValueError::new_err(format!(
            "encode_block_codes expects exactly one block; got num_vecs={} \
             != block_size={}",
            parsed.num_vecs, parsed.block_size
        )));
    }
    if parsed.share_basis {
        return Err(PyValueError::new_err(
            "share_basis is a layer-level concept; use roundtrip_layer",
        ));
    }

    let params = CodecParams {
        variance_ratio: parsed.variance_ratio,
        k: parsed.k,
        bit_width: parsed.bit_width,
        rotation_seed: parsed.rotation_seed,
        kmeans_max_iter: parsed.kmeans_max_iter,
        pca_method: to_pca(parsed.pca),
        skeleton_dtype: to_skel(parsed.skeleton_dtype),
        exact_rank_cap: parsed.exact_rank_cap,
        custom_centroids: parsed.custom_centroids.clone(),
        outlier_threshold: parsed.outlier_threshold,
    };
    let weights = vec![1.0_f32; parsed.block_size];

    let (sk, codes) = py.detach(|| match parsed.metric {
        MetricKind::Mse => kakeyaturbo::codec::encode_block::<MSE>(
            &parsed.input, &weights, parsed.dim, &params,
        ),
        MetricKind::InnerProduct => kakeyaturbo::codec::encode_block::<InnerProduct>(
            &parsed.input, &weights, parsed.dim, &params,
        ),
        MetricKind::LInf => kakeyaturbo::codec::encode_block::<LInf>(
            &parsed.input, &weights, parsed.dim, &params,
        ),
    });

    // Materialise skeleton tensors.
    let mean_f32 = sk.pca.mean_f32();
    let basis_f32 = sk.pca.basis_f32();
    let d_eff = sk.pca.d_eff;
    let d = sk.pca.d();
    let k_centers = sk.kmeans.k;
    let mut centers_f32 = Vec::with_capacity(k_centers * d_eff);
    for c in 0..k_centers {
        centers_f32.extend_from_slice(&sk.kmeans.center(c));
    }

    let n = codes.len();
    let wht_len = sk.wht_len;
    let bit_width = sk.bit_width;
    let pbytes = (wht_len * (bit_width as usize) + 7) / 8;

    let mut seg_id = Vec::with_capacity(n);
    let mut t_vals = Vec::with_capacity(n);
    let mut norm_vals = Vec::with_capacity(n);
    let mut residual_packed = vec![0u8; n * pbytes];
    // Outlier layout: two parallel ragged arrays padded to max per-row.
    let max_outliers = codes.iter().map(|c| c.outliers.len()).max().unwrap_or(0);
    let mut outlier_idx = vec![0u16; n * max_outliers.max(1)];
    let mut outlier_val = vec![0.0_f32; n * max_outliers.max(1)];
    let mut outlier_count = Vec::with_capacity(n);
    for (i, c) in codes.iter().enumerate() {
        seg_id.push(c.seg_id);
        t_vals.push(c.t.to_f32());
        norm_vals.push(c.norm.to_f32());
        if c.residual_packed.len() != pbytes {
            return Err(PyRuntimeError::new_err(format!(
                "internal: pbytes mismatch {} vs {pbytes}",
                c.residual_packed.len()
            )));
        }
        residual_packed[i * pbytes..(i + 1) * pbytes]
            .copy_from_slice(&c.residual_packed);
        outlier_count.push(c.outliers.len() as u32);
        if max_outliers > 0 {
            for (j, &(idx, val)) in c.outliers.iter().enumerate() {
                outlier_idx[i * max_outliers + j] = idx;
                outlier_val[i * max_outliers + j] = val.to_f32();
            }
        }
    }

    let out = PyDict::new(py);
    // Skeleton fields
    out.set_item(
        "mean",
        numpy::PyArray1::<f32>::from_vec(py, mean_f32),
    )?;
    out.set_item(
        "basis",
        numpy::PyArray1::<f32>::from_vec(py, basis_f32)
            .reshape([d_eff, d])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape basis: {e}")))?,
    )?;
    out.set_item(
        "centers",
        numpy::PyArray1::<f32>::from_vec(py, centers_f32)
            .reshape([k_centers, d_eff])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape centers: {e}")))?,
    )?;
    out.set_item("d", d)?;
    out.set_item("d_eff", d_eff)?;
    out.set_item("k", k_centers)?;
    out.set_item("rotation_seed", sk.rotation_seed)?;
    out.set_item("wht_len", wht_len)?;
    out.set_item("bit_width", bit_width)?;

    // Codes fields
    out.set_item(
        "seg_id",
        numpy::PyArray1::<u32>::from_vec(py, seg_id),
    )?;
    out.set_item(
        "t",
        numpy::PyArray1::<f32>::from_vec(py, t_vals),
    )?;
    out.set_item(
        "norm",
        numpy::PyArray1::<f32>::from_vec(py, norm_vals),
    )?;
    out.set_item(
        "residual_packed",
        numpy::PyArray1::<u8>::from_vec(py, residual_packed)
            .reshape([n, pbytes])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape packed: {e}")))?,
    )?;
    out.set_item(
        "outlier_idx",
        numpy::PyArray1::<u16>::from_vec(py, outlier_idx)
            .reshape([n, max_outliers.max(1)])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape out_idx: {e}")))?,
    )?;
    out.set_item(
        "outlier_val",
        numpy::PyArray1::<f32>::from_vec(py, outlier_val)
            .reshape([n, max_outliers.max(1)])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape out_val: {e}")))?,
    )?;
    out.set_item(
        "outlier_count",
        numpy::PyArray1::<u32>::from_vec(py, outlier_count),
    )?;
    out.set_item("max_outliers", max_outliers)?;
    out.set_item(
        "metric",
        match parsed.metric {
            MetricKind::Mse => "mse",
            MetricKind::InnerProduct => "inner_product",
            MetricKind::LInf => "linf",
        },
    )?;

    Ok(out)
}

/// Rebuild `Skeleton` + `Vec<Code>` from the dict `encode_block_codes`
/// produces, then run `decode_block_with_centroids`.  Byte-identical to
/// what `roundtrip_layer` would have returned on the same input; useful
/// for Torch-side "encode only" parity tests that want to check their
/// codes by seeing if Rust decodes them into the same tensor.
#[pyfunction]
#[pyo3(signature = (parts, *, custom_centroids=None))]
fn decode_block_from_parts<'py>(
    py: Python<'py>,
    parts: &Bound<'py, PyDict>,
    custom_centroids: Option<Vec<f32>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    use kakeyaturbo::kmeans::KmeansFit;
    use kakeyaturbo::pca::PcaFit;
    use kakeyaturbo::skeleton::Skeleton as RsSkel;

    fn get_required<'py, T>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
    where
        T: for<'a> FromPyObject<'a, 'py>,
        for<'a> <T as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
    {
        d.get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("missing key {key:?}")))?
            .extract::<T>()
            .map_err(Into::into)
    }

    let d: usize = get_required::<usize>(parts, "d")?;
    let d_eff: usize = get_required::<usize>(parts, "d_eff")?;
    let k_centers: usize = get_required::<usize>(parts, "k")?;
    let rotation_seed: u32 = get_required::<u32>(parts, "rotation_seed")?;
    let wht_len: usize = get_required::<usize>(parts, "wht_len")?;
    let bit_width: u8 = get_required::<u8>(parts, "bit_width")?;
    let metric: String = get_required::<String>(parts, "metric")?;
    let metric_kind = MetricKind::parse(&metric)?;

    let mean: Vec<f32> = get_required::<Vec<f32>>(parts, "mean")?;
    if mean.len() != d {
        return Err(PyValueError::new_err(format!(
            "mean length {} != d {}",
            mean.len(),
            d
        )));
    }
    let basis: Vec<f32> = {
        let obj = parts.get_item("basis")?.ok_or_else(|| {
            PyValueError::new_err("missing key 'basis'")
        })?;
        let arr: PyReadonlyArray2<f32> = obj.extract()?;
        let s = arr.shape();
        if s.len() != 2 || s[0] != d_eff || s[1] != d {
            return Err(PyValueError::new_err(format!(
                "basis shape {:?} != ({d_eff}, {d})",
                s
            )));
        }
        arr.as_slice()
            .map_err(|e| PyValueError::new_err(format!("basis not C-contig: {e}")))?
            .to_vec()
    };
    let centers: Vec<f32> = {
        let obj = parts.get_item("centers")?.ok_or_else(|| {
            PyValueError::new_err("missing key 'centers'")
        })?;
        let arr: PyReadonlyArray2<f32> = obj.extract()?;
        let s = arr.shape();
        if s.len() != 2 || s[0] != k_centers || s[1] != d_eff {
            return Err(PyValueError::new_err(format!(
                "centers shape {:?} != ({k_centers}, {d_eff})",
                s
            )));
        }
        arr.as_slice()
            .map_err(|e| PyValueError::new_err(format!("centers not C-contig: {e}")))?
            .to_vec()
    };

    let seg_id_np: numpy::PyReadonlyArray1<u32> =
        parts.get_item("seg_id")?.ok_or_else(|| {
            PyValueError::new_err("missing key 'seg_id'")
        })?.extract()?;
    let t_np: numpy::PyReadonlyArray1<f32> = parts.get_item("t")?.ok_or_else(
        || PyValueError::new_err("missing key 't'")
    )?.extract()?;
    let norm_np: numpy::PyReadonlyArray1<f32> = parts.get_item("norm")?.ok_or_else(
        || PyValueError::new_err("missing key 'norm'")
    )?.extract()?;
    let packed_np: numpy::PyReadonlyArray2<u8> = parts.get_item("residual_packed")?.ok_or_else(
        || PyValueError::new_err("missing key 'residual_packed'")
    )?.extract()?;
    let outlier_idx_np: numpy::PyReadonlyArray2<u16> = parts.get_item("outlier_idx")?.ok_or_else(
        || PyValueError::new_err("missing key 'outlier_idx'")
    )?.extract()?;
    let outlier_val_np: numpy::PyReadonlyArray2<f32> = parts.get_item("outlier_val")?.ok_or_else(
        || PyValueError::new_err("missing key 'outlier_val'")
    )?.extract()?;
    let outlier_count_np: numpy::PyReadonlyArray1<u32> = parts.get_item("outlier_count")?.ok_or_else(
        || PyValueError::new_err("missing key 'outlier_count'")
    )?.extract()?;

    let n = seg_id_np.shape()[0];
    let pbytes = packed_np.shape()[1];
    if packed_np.shape()[0] != n || t_np.shape()[0] != n || norm_np.shape()[0] != n {
        return Err(PyValueError::new_err("code array sizes inconsistent"));
    }
    let max_outliers = outlier_idx_np.shape()[1];

    // Reconstruct Skeleton.
    let pca = PcaFit::from_f32(mean, basis, d_eff, 0.0);
    let kmeans = KmeansFit::from_f32(centers, k_centers, d_eff);
    let skeleton = RsSkel {
        pca,
        kmeans,
        rotation_seed,
        wht_len,
        bit_width,
    };

    // Reconstruct Vec<Code>.
    use half::f16;
    use kakeyaturbo::codec::Code;
    let seg_id_slice = seg_id_np.as_slice().map_err(|e| {
        PyValueError::new_err(format!("seg_id not contig: {e}"))
    })?;
    let t_slice = t_np.as_slice().map_err(|e| {
        PyValueError::new_err(format!("t not contig: {e}"))
    })?;
    let norm_slice = norm_np.as_slice().map_err(|e| {
        PyValueError::new_err(format!("norm not contig: {e}"))
    })?;
    let packed_slice = packed_np.as_slice().map_err(|e| {
        PyValueError::new_err(format!("packed not contig: {e}"))
    })?;
    let oi_slice = outlier_idx_np.as_slice().map_err(|e| {
        PyValueError::new_err(format!("outlier_idx not contig: {e}"))
    })?;
    let ov_slice = outlier_val_np.as_slice().map_err(|e| {
        PyValueError::new_err(format!("outlier_val not contig: {e}"))
    })?;
    let oc_slice = outlier_count_np.as_slice().map_err(|e| {
        PyValueError::new_err(format!("outlier_count not contig: {e}"))
    })?;

    let mut codes: Vec<Code> = Vec::with_capacity(n);
    for i in 0..n {
        let cnt = oc_slice[i] as usize;
        let mut outliers: Vec<(u16, f16)> = Vec::with_capacity(cnt);
        for j in 0..cnt {
            outliers.push((
                oi_slice[i * max_outliers + j],
                f16::from_f32(ov_slice[i * max_outliers + j]),
            ));
        }
        codes.push(Code {
            seg_id: seg_id_slice[i],
            alpha: f16::from_f32(0.0),
            t: f16::from_f32(t_slice[i]),
            norm: f16::from_f32(norm_slice[i]),
            residual_packed: packed_slice[i * pbytes..(i + 1) * pbytes].to_vec(),
            outliers,
        });
    }

    // Validate custom centroids if given.
    if let Some(ref c) = custom_centroids {
        validate_centroids(c, bit_width)?;
    }

    let decoded: Vec<f32> = py.detach(|| match metric_kind {
        MetricKind::Mse => kakeyaturbo::codec::decode_block_with_centroids::<MSE>(
            &skeleton,
            &codes,
            custom_centroids.as_deref(),
        ),
        MetricKind::InnerProduct => {
            kakeyaturbo::codec::decode_block_with_centroids::<InnerProduct>(
                &skeleton,
                &codes,
                custom_centroids.as_deref(),
            )
        }
        MetricKind::LInf => kakeyaturbo::codec::decode_block_with_centroids::<LInf>(
            &skeleton,
            &codes,
            custom_centroids.as_deref(),
        ),
    });

    let flat = numpy::PyArray1::<f32>::from_vec(py, decoded);
    flat.reshape([n, d]).map_err(|e| {
        PyRuntimeError::new_err(format!("reshape decoded: {e}"))
    })
}

/// Module initialiser: registers every public function under
/// `kakeyaturbo_py._core`.  The Python shim re-exports them.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(roundtrip_layer, m)?)?;
    m.add_function(wrap_pyfunction!(wht_sign_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(wht_rows, m)?)?;
    m.add_function(wrap_pyfunction!(rotate_rows, m)?)?;
    m.add_function(wrap_pyfunction!(inverse_rotate_rows, m)?)?;
    m.add_function(wrap_pyfunction!(pack_bits, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_bits, m)?)?;
    m.add_function(wrap_pyfunction!(centroids_gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(encode_block_codes, m)?)?;
    m.add_function(wrap_pyfunction!(decode_block_from_parts, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
