"""Bit-level parity test: pyo3 roundtrip_layer vs kakeyaturbo-bench CLI.

For every CLI config the PR #15 harness actually uses, assert that
the decoded tensor and the mean_block_mse produced by the in-process
pyo3 call are byte-identical to the subprocess CLI call.  This is the
semantic anchor for M3: "the algorithm doesn't change, only the
plumbing".

Run:
    cd /workspace/LLM-KV--Cache-compress
    cargo build --release --manifest-path kakeyaturbo/Cargo.toml \
                --bin kakeyaturbo-bench
    pip install kakeyaturbo-py/target/wheels/kakeyaturbo_py-*.whl
    python -m pytest kakeyaturbo-py/tests/test_roundtrip_cli_parity.py -v
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
BENCH_BIN = REPO / "kakeyaturbo" / "target" / "release" / "kakeyaturbo-bench"
KKTV_MAGIC = 0x4B4B5456

pytest.importorskip("kakeyaturbo_py", reason="wheel not installed in this venv")

from kakeyaturbo_py import roundtrip_layer  # noqa: E402


def _write_kktv(path: Path, arr: np.ndarray) -> None:
    assert arr.dtype == np.float32 and arr.ndim == 2, (arr.dtype, arr.shape)
    n, d = arr.shape
    with path.open("wb") as f:
        f.write(struct.pack("<I", KKTV_MAGIC))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<I", d))
        f.write(struct.pack("<I", 0))
        f.write(arr.tobytes(order="C"))


def _read_kktv_f32(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == KKTV_MAGIC
        _version = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        d = struct.unpack("<I", f.read(4))[0]
        _pad = struct.unpack("<I", f.read(4))[0]
        raw = f.read(n * d * 4)
    return np.frombuffer(raw, dtype=np.float32).reshape(n, d).copy()


def _cli_roundtrip(
    arr: np.ndarray,
    *,
    metric: str,
    block_size: int,
    bit_width: int,
    variance_ratio: float,
    k: int,
    rotation_seed: int,
    pca_method: str,
    rsvd_target_rank: int | None,
    rsvd_oversample: int,
    rsvd_power_iters: int,
    skeleton_dtype: str,
    share_basis: bool,
    centroids_file: str | None,
    outlier_threshold: float | None,
) -> tuple[np.ndarray, dict]:
    if not BENCH_BIN.exists():
        pytest.skip(f"{BENCH_BIN} not built; run cargo build --release --bin kakeyaturbo-bench")
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        in_path = tdp / "x.kktv"
        rep_path = tdp / "report.json"
        dec_path = tdp / "decoded.kktv"
        _write_kktv(in_path, arr.astype(np.float32, copy=False))
        cmd = [
            str(BENCH_BIN),
            "--input", str(in_path),
            "--output", str(rep_path),
            "--metric", metric,
            "--block-size", str(block_size),
            "--variance-ratio", str(variance_ratio),
            "--k", str(k),
            "--bit-width", str(bit_width),
            "--rotation-seed", str(rotation_seed),
            "--pca-method", pca_method,
            "--skeleton-dtype", skeleton_dtype,
            "--verify",
            "--dump-decoded", str(dec_path),
        ]
        if pca_method == "randomized":
            cmd += [
                "--rsvd-target-rank", str(rsvd_target_rank if rsvd_target_rank is not None else max(arr.shape[1] // 2, 8)),
                "--rsvd-oversample", str(rsvd_oversample),
                "--rsvd-power-iters", str(rsvd_power_iters),
            ]
        if share_basis:
            cmd.append("--share-basis")
        if centroids_file is not None:
            cmd += ["--centroids-file", str(centroids_file)]
        if outlier_threshold is not None:
            cmd += ["--outlier-threshold", str(outlier_threshold)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            pytest.fail(f"CLI failed: {res.stderr[-2000:]}")
        report = json.loads(rep_path.read_text())
        decoded = _read_kktv_f32(dec_path)
    return decoded, report


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20260422)


@pytest.fixture(scope="module")
def data(rng):
    """A synthetic block-aligned input that exercises all paths.  Size
    is chosen so there are >1 blocks at block_size=128 (most tests)
    and the PCA / K-means can't trivially degenerate to identity."""
    n, d = 512, 128
    return rng.standard_normal((n, d)).astype(np.float32)


@pytest.mark.parametrize("metric", ["mse", "inner_product", "linf"])
@pytest.mark.parametrize("share_basis", [False, True])
def test_parity_default(data, metric, share_basis):
    """Exact PCA, no centroids, no outliers — default CLI behaviour."""
    cli_dec, cli_rep = _cli_roundtrip(
        data, metric=metric, block_size=128, bit_width=3,
        variance_ratio=0.95, k=16, rotation_seed=0xCAFEBABE,
        pca_method="exact", rsvd_target_rank=None,
        rsvd_oversample=8, rsvd_power_iters=2,
        skeleton_dtype="fp16", share_basis=share_basis,
        centroids_file=None, outlier_threshold=None,
    )
    py_dec, py_rep = roundtrip_layer(
        data, metric=metric, block_size=128, bit_width=3,
        variance_ratio=0.95, k=16, rotation_seed=0xCAFEBABE,
        pca_method="exact", skeleton_dtype="fp16",
        share_basis=share_basis,
    )
    np.testing.assert_array_equal(py_dec, cli_dec)
    # CLI JSON emits mean_block_mse at {:.10} precision (see
    # kakeyaturbo-bench.rs Report::to_json); pyo3 returns the full f64.
    # The only semantic requirement is that the *decoded tensor* is
    # bit-identical (checked above); mean_block_mse agreement within
    # 1e-9 absolute is a sanity check on the serialisation layer.
    assert py_rep["mean_block_mse"] == pytest.approx(cli_rep["mean_block_mse"], abs=1e-9)
    assert py_rep["num_blocks"] == cli_rep["num_blocks"]


def test_parity_randomized_pca(data):
    cli_dec, cli_rep = _cli_roundtrip(
        data, metric="inner_product", block_size=128, bit_width=3,
        variance_ratio=0.95, k=16, rotation_seed=3405691582,
        pca_method="randomized", rsvd_target_rank=64,
        rsvd_oversample=8, rsvd_power_iters=2,
        skeleton_dtype="fp16", share_basis=False,
        centroids_file=None, outlier_threshold=None,
    )
    py_dec, py_rep = roundtrip_layer(
        data, metric="inner_product", block_size=128, bit_width=3,
        variance_ratio=0.95, k=16, rotation_seed=3405691582,
        pca_method="randomized", rsvd_target_rank=64,
        rsvd_oversample=8, rsvd_power_iters=2,
        skeleton_dtype="fp16", share_basis=False,
    )
    np.testing.assert_array_equal(py_dec, cli_dec)
    assert py_rep["mean_block_mse"] == pytest.approx(cli_rep["mean_block_mse"], rel=0, abs=1e-9)


def test_parity_outlier_threshold(data):
    cli_dec, cli_rep = _cli_roundtrip(
        data, metric="inner_product", block_size=128, bit_width=3,
        variance_ratio=0.95, k=16, rotation_seed=3405691582,
        pca_method="randomized", rsvd_target_rank=64,
        rsvd_oversample=8, rsvd_power_iters=2,
        skeleton_dtype="fp16", share_basis=False,
        centroids_file=None, outlier_threshold=2.0,
    )
    py_dec, py_rep = roundtrip_layer(
        data, metric="inner_product", block_size=128, bit_width=3,
        variance_ratio=0.95, k=16, rotation_seed=3405691582,
        pca_method="randomized", rsvd_target_rank=64,
        rsvd_oversample=8, rsvd_power_iters=2,
        skeleton_dtype="fp16", share_basis=False,
        outlier_threshold=2.0,
    )
    np.testing.assert_array_equal(py_dec, cli_dec)
    assert py_rep["mean_block_mse"] == pytest.approx(cli_rep["mean_block_mse"], rel=0, abs=1e-9)


def test_parity_custom_centroids(data, tmp_path):
    """Pass calibrated centroids two ways (list vs file path) and
    confirm both byte-match the CLI's --centroids-file path."""
    # Sorted ascending centroid table.  8 centroids = 2^3 bits.
    centroids = np.array([
        -2.1519, -1.3438, -0.7563, -0.2449,
         0.2449,  0.7563,  1.3438,  2.1519,
    ], dtype=np.float32)
    centroids_path = tmp_path / "gaussian_b3.f32"
    centroids_path.write_bytes(centroids.tobytes(order="C"))

    cli_dec, cli_rep = _cli_roundtrip(
        data, metric="inner_product", block_size=128, bit_width=3,
        variance_ratio=0.95, k=16, rotation_seed=3405691582,
        pca_method="randomized", rsvd_target_rank=64,
        rsvd_oversample=8, rsvd_power_iters=2,
        skeleton_dtype="fp16", share_basis=False,
        centroids_file=str(centroids_path), outlier_threshold=None,
    )
    # Pass-through-list form
    py_dec_list, py_rep_list = roundtrip_layer(
        data, metric="inner_product", block_size=128, bit_width=3,
        variance_ratio=0.95, k=16, rotation_seed=3405691582,
        pca_method="randomized", rsvd_target_rank=64,
        rsvd_oversample=8, rsvd_power_iters=2,
        skeleton_dtype="fp16", share_basis=False,
        centroids=centroids.tolist(),
    )
    # Pass-through-file form
    py_dec_file, py_rep_file = roundtrip_layer(
        data, metric="inner_product", block_size=128, bit_width=3,
        variance_ratio=0.95, k=16, rotation_seed=3405691582,
        pca_method="randomized", rsvd_target_rank=64,
        rsvd_oversample=8, rsvd_power_iters=2,
        skeleton_dtype="fp16", share_basis=False,
        centroids_file=str(centroids_path),
    )
    np.testing.assert_array_equal(py_dec_list, cli_dec)
    np.testing.assert_array_equal(py_dec_file, cli_dec)
    assert py_rep_list["mean_block_mse"] == pytest.approx(cli_rep["mean_block_mse"], rel=0, abs=1e-9)
    assert py_rep_file["mean_block_mse"] == pytest.approx(cli_rep["mean_block_mse"], rel=0, abs=1e-9)


def test_rejects_non_contiguous():
    bad = np.zeros((128, 64), dtype=np.float32)[:, ::2]  # non-contig view
    assert not bad.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match=r"C-contiguous"):
        roundtrip_layer(bad, metric="mse")


def test_rejects_bad_bit_width():
    with pytest.raises(ValueError, match=r"bit_width"):
        roundtrip_layer(np.zeros((128, 32), dtype=np.float32),
                        metric="mse", bit_width=5)


def test_rejects_unsorted_centroids():
    with pytest.raises(ValueError, match=r"ascending"):
        roundtrip_layer(
            np.zeros((128, 32), dtype=np.float32), metric="mse",
            bit_width=2, centroids=[1.0, 0.0, 2.0, 3.0],
        )


def test_rejects_bad_metric():
    with pytest.raises(ValueError, match=r"unknown metric"):
        roundtrip_layer(np.zeros((128, 32), dtype=np.float32), metric="euclid")
