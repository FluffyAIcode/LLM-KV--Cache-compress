"""M5: Triton DECODE kernel parity vs Rust decode + partial-block path.

Fuzz sweep:
  * 16 seeds × 7 shapes × 3 metrics × 3 bit-widths  = 1008 exact-PCA cases
  * 8  seeds × 3 metrics × 3 bit-widths              =  72 randomized-PCA
  * 8  seeds × 3 metrics × 3 bit-widths × 3 thresholds = 216 outlier
  * 6  seeds × 3 metrics × 2 bit-widths              =  36 custom-centroid
  * 4  seeds × 3 metrics                              =  12 full PR #15
                                               TOTAL = 1344 triples

Exit criterion (PLAN.md §M5):
    Bit-exact vs Rust reference on decoded tensors; attention output
    match.  Decoded-tensor bar here is the same 1e-3 L2-rel / 1 % row-flip
    two-metric used in Phase B.  The attention-output match is the M6
    gate (we don't have a live vLLM integration here yet), so this
    test only checks the decoded-tensor contract.

Partial-block coverage: one explicit test asserts
`decode_partial_block_bf16` is bit-identical to a torch.to(fp32) cast
for arbitrary `m ∈ [1, block_size_codec)`.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

try:
    from kakeyaturbo_py import triton_is_available
    if not triton_is_available():
        pytest.skip("triton not available", allow_module_level=True)
except Exception as e:  # pragma: no cover
    pytest.skip(f"triton import failed: {e}", allow_module_level=True)


from kakeyaturbo_py import (
    centroids_gaussian,
    decode_block_from_parts,
    decode_block_torch_from_parts,
    decode_block_triton_from_parts,
    decode_partial_block_bf16,
    encode_block_codes,
)


DEVICE = "cuda"


_SEEDS = list(range(1, 37))
_SHAPES = [
    (64,  32),
    (128, 32),
    (128, 64),
    (256, 64),
    (512, 128),
    (256, 128),
    (1024, 64),
]
_METRICS = ["mse", "inner_product", "linf"]
_BIT_WIDTHS = [2, 3, 4]


def _one_case(
    seed: int, n: int, d: int, metric: str, bit_width: int,
    rotation_seed: int, variance_ratio: float, k: int,
    pca_method: str = "exact", rsvd_rank: int | None = None,
    outlier_threshold: float | None = None,
    custom_centroids: np.ndarray | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((n, d)) * 0.3).astype(np.float32)

    kwargs = dict(
        metric=metric, block_size=n, bit_width=bit_width,
        variance_ratio=variance_ratio, k=k, rotation_seed=rotation_seed,
        pca_method=pca_method, skeleton_dtype="fp16",
        share_basis=False, kmeans_max_iter=32,
    )
    if pca_method == "randomized":
        kwargs["rsvd_target_rank"] = rsvd_rank if rsvd_rank is not None else max(d // 2, 8)
        kwargs["rsvd_oversample"] = 8
        kwargs["rsvd_power_iters"] = 2
    if outlier_threshold is not None:
        kwargs["outlier_threshold"] = float(outlier_threshold)
    if custom_centroids is not None:
        kwargs["centroids"] = custom_centroids.astype(np.float32).tolist()

    # Encode via Rust (byte-exact) — this is the ground-truth parts dict.
    parts = encode_block_codes(X, **kwargs)
    parts = {key: (np.asarray(v) if hasattr(v, "shape") else v)
             for key, v in parts.items()}

    # Three decode paths.
    dec_rust = np.asarray(decode_block_from_parts(
        parts,
        custom_centroids=(custom_centroids.tolist()
                           if custom_centroids is not None else None),
    ))
    dec_torch = decode_block_torch_from_parts(
        parts, custom_centroids=custom_centroids, device="cpu",
    )
    dec_triton = decode_block_triton_from_parts(
        parts, custom_centroids=custom_centroids, device=DEVICE,
    )

    # L2-relative and row-flip bars (same as Phase B).
    def _rel(a, b):
        num = np.linalg.norm(a - b)
        den = np.linalg.norm(b) + 1e-30
        return float(num / den)

    def _row_flip(a, b, *, tol=0.1):
        per_row_max = np.max(np.abs(a - b), axis=1)
        return float((per_row_max > tol).sum()) / a.shape[0]

    L2_BAR = 1e-3
    ROW_BAR = max(2.0 / n, 0.01)

    # (a) Triton decode vs Rust decode (the primary M5 gate)
    r_l2 = _rel(dec_triton, dec_rust)
    r_rows = _row_flip(dec_triton, dec_rust)
    assert r_l2 <= L2_BAR or r_rows <= ROW_BAR, (
        f"Triton decode vs Rust: L2={r_l2:.3e} (bar={L2_BAR}), "
        f"rows_flipped={r_rows:.3e} (bar={ROW_BAR:.3e}); both exceeded on "
        f"(seed={seed}, n={n}, d={d}, metric={metric}, bit_width={bit_width})"
    )

    # (b) Triton decode vs PyTorch reference decode
    r_l2 = _rel(dec_triton, dec_torch)
    r_rows = _row_flip(dec_triton, dec_torch)
    assert r_l2 <= L2_BAR or r_rows <= ROW_BAR, (
        f"Triton decode vs Torch ref: L2={r_l2:.3e}, rows_flipped={r_rows:.3e}"
    )


# ---------------------------------------------------------------------------
# Parametrisations mirror Phase B exactly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", _SEEDS[:16])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("bit_width", _BIT_WIDTHS)
def test_triton_decode_exact_pca(seed, shape, metric, bit_width):
    n, d = shape
    _one_case(
        seed=seed, n=n, d=d, metric=metric, bit_width=bit_width,
        rotation_seed=(seed * 2654435761) & 0xFFFFFFFF,
        variance_ratio=0.95, k=min(16, n),
        pca_method="exact",
    )


@pytest.mark.parametrize("seed", _SEEDS[:8])
@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("bit_width", [2, 3, 4])
def test_triton_decode_randomized_pca(seed, metric, bit_width):
    _one_case(
        seed=seed, n=512, d=128, metric=metric, bit_width=bit_width,
        rotation_seed=(seed * 2654435761) & 0xFFFFFFFF,
        variance_ratio=0.95, k=16,
        pca_method="randomized", rsvd_rank=64,
    )


@pytest.mark.parametrize("seed", _SEEDS[:8])
@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("bit_width", [2, 3, 4])
@pytest.mark.parametrize("threshold", [1.5, 2.0, 2.5])
def test_triton_decode_outlier(seed, metric, bit_width, threshold):
    _one_case(
        seed=seed, n=512, d=128, metric=metric, bit_width=bit_width,
        rotation_seed=(seed * 2654435761) & 0xFFFFFFFF,
        variance_ratio=0.95, k=16,
        pca_method="exact",
        outlier_threshold=threshold,
    )


def _synthetic_centroids(bit_width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = 1 << bit_width
    c = np.asarray(centroids_gaussian(bit_width)).astype(np.float64)
    jitter = rng.standard_normal(k) * 0.05 * np.abs(c).max()
    c += jitter
    c = np.sort(c)
    for i in range(1, k):
        if c[i] - c[i - 1] < 1e-3:
            c[i] = c[i - 1] + 1e-3
    return c.astype(np.float32)


@pytest.mark.parametrize("seed", _SEEDS[:6])
@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("bit_width", [2, 3])
def test_triton_decode_custom_centroids(seed, metric, bit_width):
    c = _synthetic_centroids(bit_width, seed)
    _one_case(
        seed=seed, n=512, d=128, metric=metric, bit_width=bit_width,
        rotation_seed=(seed * 2654435761) & 0xFFFFFFFF,
        variance_ratio=0.95, k=16,
        pca_method="exact",
        custom_centroids=c,
    )


@pytest.mark.parametrize("seed", _SEEDS[:4])
@pytest.mark.parametrize("metric", _METRICS)
def test_triton_decode_full_pr15_recipe(seed, metric):
    """Full PR #15 cell: RSVD PCA + outlier T=2.0 + calibrated centroids."""
    c = _synthetic_centroids(bit_width=3, seed=seed)
    _one_case(
        seed=seed, n=512, d=128, metric=metric, bit_width=3,
        rotation_seed=3405691582,
        variance_ratio=0.95, k=16,
        pca_method="randomized", rsvd_rank=64,
        outlier_threshold=2.0,
        custom_centroids=c,
    )


# ---------------------------------------------------------------------------
# Partial-block bf16 passthrough path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 5, 17, 63, 128, 255, 511])
@pytest.mark.parametrize("d", [32, 64, 128])
def test_partial_block_bf16_passthrough(m, d):
    """The partial-block path is a bf16 → fp32 identity cast; there's
    nothing algorithmic to test, but we assert the contract that
    `decode_partial_block_bf16` returns the bitwise-equivalent fp32
    of the input bf16 staging buffer.  This locks the M6 backend's
    partial-block read into an explicit API."""
    rng = np.random.default_rng(20260422)
    x_fp32 = torch.from_numpy(
        (rng.standard_normal((m, d)) * 0.3).astype(np.float32)
    ).to(DEVICE)
    x_bf16 = x_fp32.to(torch.bfloat16)

    out = decode_partial_block_bf16(x_bf16)
    assert out.dtype == torch.float32
    assert out.shape == (m, d)
    assert out.device == x_bf16.device

    # Bit-identical to a torch.to(fp32) cast — no rounding artefact,
    # because bf16 → fp32 is an exact-representable upcast.
    expected = x_bf16.to(torch.float32)
    assert torch.equal(out, expected), (
        f"decode_partial_block_bf16 diverged from torch upcast at m={m}, d={d}"
    )


def test_partial_block_rejects_non_bf16():
    x_fp32 = torch.zeros((4, 32), dtype=torch.float32, device=DEVICE)
    with pytest.raises(TypeError, match="bfloat16"):
        decode_partial_block_bf16(x_fp32)


def test_partial_block_accepts_3d_staging_shape():
    """M6 backend staging buffers have shape [m, n_kv_heads, d_eff].
    decode_partial_block_bf16 must accept that shape and upcast
    element-wise — the partial-block path is a simple dtype cast,
    it has no opinion about head dims."""
    x_bf16 = torch.zeros((17, 2, 64), dtype=torch.bfloat16, device=DEVICE)
    out = decode_partial_block_bf16(x_bf16)
    assert out.dtype == torch.float32
    assert out.shape == (17, 2, 64)


def test_partial_block_rejects_rank_0():
    x_bf16 = torch.tensor(0.0, dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(ValueError, match="rank="):
        decode_partial_block_bf16(x_bf16)
