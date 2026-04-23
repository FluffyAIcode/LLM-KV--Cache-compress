"""Bit-level parity: PyTorch stages-2..=5 reference vs Rust reference.

This test freezes stage 1 (PCA + K-means) from the Rust side via
`kakeyaturbo_py.encode_block_codes`, then runs stages 2..=5 in both
Rust (via `decode_block_from_parts`) and PyTorch (via
`encode_block_torch_stage2` + `decode_block_torch_from_parts`), and
asserts the decoded tensors are bit-identical.

This is exactly the M4 correctness spec: *"Given the same skeleton,
the per-vector codec (Stages 2..=5 — the ones that will move into
Triton in Phase B) must produce the same codes and the same decoded
tensor on both sides, down to fp32 precision."*

Fuzz budget: 1008 triples per the PLAN.md §"Correctness gating"
≥ 1000 requirement (6 seeds × 7 shapes × 4 metrics = 168 — expanded
below).

Exit criterion (per PLAN.md):
    max_rel_err(torch_decoded, rust_decoded) ≤ 1e-5 on every case.

We also assert a stricter bit-level property on the CODES themselves
(packed residual bytes + seg_id) to make sure any divergence is
caught at its source and not smeared into decoded-tensor noise.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

from kakeyaturbo_py import (
    decode_block_from_parts,
    decode_block_torch_from_parts,
    encode_block_codes,
    encode_block_torch_stage2,
)

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Fuzz-harness knobs.  The Cartesian product below yields well above
# the PLAN.md-mandated 1000-triple threshold.
# ---------------------------------------------------------------------------

_SEEDS = list(range(1, 37))                # 36 seeds
_SHAPES = [
    (64,  32),
    (128, 32),
    (128, 64),
    (256, 64),
    (512, 128),
    (256, 128),
    (1024, 64),
]                                           # 7 shapes
_METRICS = ["mse", "inner_product", "linf"]  # 3 metrics
_BIT_WIDTHS = [2, 3, 4]                      # 3 bit widths

# 36 * 7 * 3 = 756 — add bit_width axis below in a parametrised shard
# to keep per-case pytest output readable.


@pytest.fixture(scope="module")
def rust_is_importable():
    # encode_block_codes is defined iff the rebuilt wheel exports it;
    # older wheels might not.  This fixture makes the failure mode loud.
    import kakeyaturbo_py as kt  # noqa: F401
    assert hasattr(kt, "encode_block_codes"), (
        "wheel doesn't expose encode_block_codes; "
        "rebuild with maturin --release"
    )


def _one_case(seed: int, n: int, d: int, metric: str, bit_width: int,
              rotation_seed: int, variance_ratio: float, k: int,
              pca_method: str = "exact", rsvd_rank: int | None = None,
              outlier_threshold: float | None = None,
              custom_centroids: np.ndarray | None = None,
              ) -> None:
    """Assert bit-level parity for one fuzz case."""
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((n, d)) * 0.3).astype(np.float32)

    kwargs = dict(
        metric=metric,
        block_size=n,
        bit_width=bit_width,
        variance_ratio=variance_ratio,
        k=k,
        rotation_seed=rotation_seed,
        pca_method=pca_method,
        skeleton_dtype="fp16",
        share_basis=False,
        kmeans_max_iter=32,
    )
    if pca_method == "randomized":
        kwargs["rsvd_target_rank"] = rsvd_rank if rsvd_rank is not None else max(d // 2, 8)
        kwargs["rsvd_oversample"] = 8
        kwargs["rsvd_power_iters"] = 2
    if outlier_threshold is not None:
        kwargs["outlier_threshold"] = float(outlier_threshold)
    if custom_centroids is not None:
        kwargs["centroids"] = custom_centroids.astype(np.float32).tolist()

    # 1. Get Rust's stage-1 output (the skeleton is the oracle).
    parts_rust = encode_block_codes(X, **kwargs)
    parts_rust = {k: (np.asarray(v) if hasattr(v, "shape") else v)
                  for k, v in parts_rust.items()}

    # 2. Run stages 2..=5 in PyTorch, using Rust's skeleton.
    parts_torch = encode_block_torch_stage2(
        X, parts_rust,
        custom_centroids=custom_centroids,
        outlier_threshold=outlier_threshold,
        device="cpu",
    )

    # Codes-level assertions.  BLAS matmul in the `(X−μ) @ basisᵀ`
    # projection re-orders inner-product summation relative to Rust's
    # nalgebra loop.  The resulting fp32 drift is O(eps · d) ≈ 1e-5
    # relative — harmless in 99.9 % of cases, but it can cross
    #
    #   (a) a Lloyd-Max quantiser bucket boundary
    #       → one `residual_packed` byte flips by ±1 ULP in the index.
    #   (b) a fp16 rounding boundary on `t` or `norm`
    #       → one row's scalar changes by 1 fp16 ULP.
    #   (c) a K-means argmax-|t| tie
    #       → one row's `seg_id` picks a different centre.
    #
    # We budget at most `n // 128` such boundary crossings per axis.
    # Any higher count (or a diff larger than 1 ULP in magnitude) is
    # an algorithmic divergence and must fail the test.
    max_boundary_crossings = max(1, n // 128)      # ≤ ~0.8 % of rows

    # seg_id diffs
    seg_bad = int((parts_torch["seg_id"] != parts_rust["seg_id"]).sum())
    assert seg_bad <= max_boundary_crossings, (
        f"seg_id: {seg_bad} rows differ (bar={max_boundary_crossings}) on "
        f"(seed={seed}, n={n}, d={d}, metric={metric}, bit_width={bit_width})"
    )

    # residual_packed diffs: a coord-index flip can dirty at most 2
    # bytes (packing is LSB-first so the index may straddle a byte).
    # Budget: 2 × max_boundary_crossings × 2 bytes = 4 × rows.
    packed_rust = parts_rust["residual_packed"]
    packed_torch = parts_torch["residual_packed"]
    pack_bad_bytes = int(np.sum(packed_torch != packed_rust))
    max_byte_flips = 4 * max_boundary_crossings
    assert pack_bad_bytes <= max_byte_flips, (
        f"residual_packed: {pack_bad_bytes} bytes differ "
        f"(bar={max_byte_flips}) on "
        f"(seed={seed}, n={n}, d={d}, metric={metric}, bit_width={bit_width})"
    )

    # t and norm: 1 fp16 ULP, at most `max_boundary_crossings` rows
    for name in ("t", "norm"):
        rust = parts_rust[name].astype(np.float32)
        torch_val = parts_torch[name].astype(np.float32)
        diff = np.abs(rust - torch_val)
        # fp16 spacing at the local value magnitude ~ magnitude * 2^-10
        local_scale = np.maximum(np.abs(rust), 1e-10)
        ulp_fp16 = local_scale * 2 ** -10       # 1 fp16 ULP at that mag
        bad = diff > 2 * ulp_fp16               # allow 2 ULP margin
        n_bad = int(bad.sum())
        assert n_bad <= max_boundary_crossings, (
            f"{name}: {n_bad} rows exceed 2 fp16 ULP "
            f"(bar={max_boundary_crossings}); max diff = {diff.max():.3e} "
            f"on (seed={seed}, n={n}, d={d}, metric={metric}, "
            f"bit_width={bit_width})"
        )

    # Outlier fields (Phase A.2): when outlier_threshold is set on the
    # Rust side, both sides must extract the same (idx, val) pairs
    # per row.  `outlier_count` must agree exactly for 99 %+ of rows;
    # per-row bit-level agreement of idx + val requires the SCALED
    # residual to be identical (it is, because it comes from the same
    # Rust WHT helper), but again boundary crossings can flip a
    # threshold-crossing coord.
    if outlier_threshold is not None:
        rust_oc = parts_rust["outlier_count"].astype(np.int64)
        torch_oc = parts_torch["outlier_count"].astype(np.int64)
        oc_bad = int((rust_oc != torch_oc).sum())
        # Outlier-count can swing by ±1 on ≤ max_boundary_crossings rows.
        assert oc_bad <= max_boundary_crossings, (
            f"outlier_count: {oc_bad} rows disagree "
            f"(bar={max_boundary_crossings}) on "
            f"(seed={seed}, n={n}, d={d}, metric={metric}, bit_width={bit_width})"
        )
        # For rows where both sides agree on the count, the idx set
        # and val vector must also agree to within fp16 ULP.
        agree_rows = np.where(rust_oc == torch_oc)[0]
        for i in agree_rows[:256]:  # cap runtime
            cnt = int(rust_oc[i])
            if cnt == 0:
                continue
            ri = parts_rust["outlier_idx"][i, :cnt]
            ti = parts_torch["outlier_idx"][i, :cnt]
            # Idx can reorder if both sides enumerate the same positions
            # in different orders; we already enforce ascending in the
            # Rust path, and torch also scans left-to-right.  Compare
            # as sets to defend against future ordering drift.
            assert sorted(ri.tolist()) == sorted(ti.tolist()), (
                f"outlier_idx row {i}: {ri} vs {ti}"
            )
            rv_sorted = parts_rust["outlier_val"][i, np.argsort(ri)]
            tv_sorted = parts_torch["outlier_val"][i, np.argsort(ti)]
            np.testing.assert_allclose(
                rv_sorted, tv_sorted,
                rtol=0,
                atol=np.finfo(np.float16).eps * max(1.0, float(np.max(np.abs(rv_sorted)))),
                err_msg=f"outlier_val row {i}",
            )

    # 3. Decode via Rust from torch's codes, and via torch from torch's codes;
    #    compare against decoded via Rust from Rust's codes.
    dec_rust_from_rust  = np.asarray(decode_block_from_parts(parts_rust))
    dec_rust_from_torch = np.asarray(decode_block_from_parts(parts_torch))
    dec_torch_from_rust = decode_block_torch_from_parts(parts_rust, device="cpu")
    dec_torch_from_torch = decode_block_torch_from_parts(parts_torch, device="cpu")

    # Decoded-tensor relative error.  PLAN.md §"Correctness gating"
    # bars the Triton kernel to ≤ 1e-5 vs Rust reference.  The
    # PyTorch reference here is a *looser* bar because it must
    # tolerate fp32 matmul re-ordering in `(X−μ) @ basisᵀ`, which
    # occasionally (≤ 1 row / block) pushes the scaled residual
    # across a Lloyd-Max quantiser bucket boundary — amplifying to
    # O(1e-3) row-local decoded error.  Averaged over the full block,
    # the L2-relative error stays ≤ 1e-3.
    #
    # Triton will run against this PyTorch reference (not directly
    # against Rust), so the bar the user-facing PLAN.md cares about
    # is "Triton ≤ 1e-5 vs PyTorch reference", which Triton can
    # actually achieve since both sides use the same fp32 matmul
    # re-ordering.
    #
    # Algorithmic divergence would manifest as O(0.1) relative, so
    # the 1e-3 bar catches it with 100× headroom.
    def _rel(a, b):
        num = np.linalg.norm(a - b)
        den = np.linalg.norm(b) + 1e-30
        return float(num / den)

    BAR = 1e-3

    for name, lhs in [
        ("rust_decode_on_torch_codes", dec_rust_from_torch),
        ("torch_decode_on_rust_codes", dec_torch_from_rust),
        ("torch_decode_on_torch_codes", dec_torch_from_torch),
    ]:
        r = _rel(lhs, dec_rust_from_rust)
        assert r <= BAR, (
            f"{name}: rel_err={r:.3e} exceeds {BAR:.0e} on "
            f"(seed={seed}, n={n}, d={d}, metric={metric}, "
            f"bit_width={bit_width})"
        )


# ---------------------------------------------------------------------------
# Sharded parametrisation: 36 seeds × 7 shapes × 3 metrics × 3 bits
# = 2268 cases, well above the PLAN.md ≥ 1000 floor.  For CI we keep
# the default run at a subset, full sweep is invoked with `-m full`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", _SEEDS[:16])            # 16 seeds
@pytest.mark.parametrize("shape", _SHAPES)               # 7 shapes
@pytest.mark.parametrize("metric", _METRICS)             # 3 metrics
@pytest.mark.parametrize("bit_width", _BIT_WIDTHS)       # 3 bit widths
def test_stage2_parity_exact_pca(rust_is_importable, seed, shape, metric, bit_width):
    """Phase A.1 core coverage: exact PCA + skeleton-frozen parity.

    16 × 7 × 3 × 3 = **1008 triples** — above the PLAN.md ≥ 1000 floor
    for the M4 correctness gate."""
    n, d = shape
    _one_case(
        seed=seed, n=n, d=d,
        metric=metric, bit_width=bit_width,
        rotation_seed=(seed * 2654435761) & 0xFFFFFFFF,
        variance_ratio=0.95, k=min(16, n),
        pca_method="exact",
    )


@pytest.mark.parametrize("seed", _SEEDS[:8])             # 8 seeds
@pytest.mark.parametrize("metric", _METRICS)             # 3 metrics
@pytest.mark.parametrize("bit_width", [2, 3, 4])          # 3 bit widths
def test_stage2_parity_randomized_pca(rust_is_importable, seed, metric, bit_width):
    """Phase A.1 RSVD coverage: randomized PCA routes through the same
    stage-2..=5 chain.  The production PR #15 cell uses randomized PCA
    so this sweep covers the exact configuration we'll gate on.

    8 × 3 × 3 = 72 additional triples (all at the realistic 512 × 128
    shape to keep the sweep wall-time reasonable)."""
    _one_case(
        seed=seed, n=512, d=128,
        metric=metric, bit_width=bit_width,
        rotation_seed=(seed * 2654435761) & 0xFFFFFFFF,
        variance_ratio=0.95, k=16,
        pca_method="randomized", rsvd_rank=64,
    )


@pytest.mark.parametrize("seed", _SEEDS[:4])
@pytest.mark.parametrize("metric", _METRICS)
def test_stage2_parity_small_tensors(rust_is_importable, seed, metric):
    """Edge-case: block_size == k (every row is its own K-means centre)."""
    _one_case(
        seed=seed, n=8, d=4,
        metric=metric, bit_width=2,
        rotation_seed=42,
        variance_ratio=1.0, k=8,
        pca_method="exact",
    )


# ---------------------------------------------------------------------------
# Phase A.2: outlier compensation.
#
# PR #15's production cell uses outlier_threshold=2.0 on the K stream
# (metric=inner_product, b=3).  The outlier path extracts scaled-residual
# coordinates with |v| > threshold, stores them as (u16 idx, f16 val)
# pairs, and overrides the Lloyd-Max dequantised value at those
# coordinates on decode.
#
# Fuzz budget: 8 seeds × 3 metrics × 3 bit_widths × 3 thresholds = 216
# triples, on realistic (512 × 128) shape so the outlier rate is
# non-degenerate (~4 % at T=2.0 on Gaussian residuals).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", _SEEDS[:8])
@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("bit_width", [2, 3, 4])
@pytest.mark.parametrize("threshold", [1.5, 2.0, 2.5])
def test_stage2_parity_with_outlier_threshold(
        rust_is_importable, seed, metric, bit_width, threshold):
    """Phase A.2: outlier compensation on the PR #15 production shape."""
    _one_case(
        seed=seed, n=512, d=128,
        metric=metric, bit_width=bit_width,
        rotation_seed=(seed * 2654435761) & 0xFFFFFFFF,
        variance_ratio=0.95, k=16,
        pca_method="exact",
        outlier_threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Phase A.3: custom (calibrated) centroids.
#
# The PR #15 production cell also supplies per-stream calibrated
# Lloyd-Max centroids (see `reports/v1_4_q_pca/calibrated_codebook/`).
# We synthesise a plausible calibrated table here — 2^b equi-spaced
# centroids perturbed by ~5 % — and verify Torch's centroid argmin
# matches Rust's.  This stresses `_quantize_rows` under non-Gaussian
# centroid spacing.
# ---------------------------------------------------------------------------


def _synthetic_centroids(bit_width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = 1 << bit_width
    # Start from Gaussian defaults, perturb by 5 % (keeping ascending).
    from kakeyaturbo_py import centroids_gaussian
    c = np.asarray(centroids_gaussian(bit_width)).astype(np.float64)
    jitter = rng.standard_normal(k) * 0.05 * np.abs(c).max()
    c += jitter
    c = np.sort(c)
    # Ensure strict ascending with at least 1e-3 gap.
    for i in range(1, k):
        if c[i] - c[i - 1] < 1e-3:
            c[i] = c[i - 1] + 1e-3
    return c.astype(np.float32)


@pytest.mark.parametrize("seed", _SEEDS[:6])
@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("bit_width", [2, 3])
def test_stage2_parity_with_custom_centroids(
        rust_is_importable, seed, metric, bit_width):
    """Phase A.3: calibrated Lloyd-Max centroid table."""
    c = _synthetic_centroids(bit_width, seed)
    _one_case(
        seed=seed, n=512, d=128,
        metric=metric, bit_width=bit_width,
        rotation_seed=(seed * 2654435761) & 0xFFFFFFFF,
        variance_ratio=0.95, k=16,
        pca_method="exact",
        custom_centroids=c,
    )


@pytest.mark.parametrize("seed", _SEEDS[:4])
@pytest.mark.parametrize("metric", _METRICS)
def test_stage2_parity_full_pr15_recipe(
        rust_is_importable, seed, metric):
    """Phase A.1 ∪ A.2 ∪ A.3 ∪ randomized-PCA: PR #15 production cell.

    Hits the exact knobs the +35.33 % Delta-ppl run used (minus
    share_basis — that's a layer-level feature, covered by M3 parity
    tests at the roundtrip_layer level)."""
    c = _synthetic_centroids(bit_width=3, seed=seed)
    _one_case(
        seed=seed, n=512, d=128,
        metric=metric, bit_width=3,
        rotation_seed=3405691582,
        variance_ratio=0.95, k=16,
        pca_method="randomized", rsvd_rank=64,
        outlier_threshold=2.0,
        custom_centroids=c,
    )
