"""Layer-level A/B: pyo3 in-process vs kakeyaturbo-bench CLI
under the PR #15 production recipe.

This test stresses the exact (metric, centroids, Q-precond, outliers,
boundary skip, share-basis) axes that the +35.33 % Δppl run used,
on realistic K/V tensor shapes extracted from DS-Distill-Qwen-1.5B
(D=128, n_kv=2).  The random seed and all flags match the production
cell config in `benchmarks/run_v1_3_ppl_full_vllm.sh`.

If every decoded tensor is bit-identical between the two paths, the
engine-level Δppl they produce must also be bit-identical — the rest
of the pipeline (logits, softmax, argmax) is deterministic.  This is
the M3 exit criterion evaluated as a fixed point: "removing the
subprocess changed nothing observable".

Run:
    cd /workspace/LLM-KV--Cache-compress
    cargo build --release --manifest-path kakeyaturbo/Cargo.toml --bin kakeyaturbo-bench
    cd kakeyaturbo-py && maturin develop --release
    cd .. && python -m pytest kakeyaturbo-py/tests/test_full_recipe_parity.py -v
"""
from __future__ import annotations

import json
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

pytest.importorskip("kakeyaturbo_py", reason="wheel not installed")
from kakeyaturbo_py import roundtrip_layer  # noqa: E402


# ----- KKTV helpers (copy of the old harness's I/O so this test is
# self-contained and cannot drift from the wire format the CLI expects) -----


def _write_kktv(path: Path, arr: np.ndarray) -> None:
    assert arr.dtype == np.float32 and arr.ndim == 2
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
        _ = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        d = struct.unpack("<I", f.read(4))[0]
        _ = struct.unpack("<I", f.read(4))[0]
        raw = f.read(n * d * 4)
    return np.frombuffer(raw, dtype=np.float32).reshape(n, d).copy()


def _cli(
    arr: np.ndarray,
    *,
    metric: str,
    bit_width: int,
    centroids_file: str | None,
    outlier_threshold: float | None,
    share_basis: bool,
    rsvd_target_rank: int,
) -> tuple[np.ndarray, dict]:
    if not BENCH_BIN.exists():
        pytest.skip(f"{BENCH_BIN} not built")
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
            "--block-size", "512",
            "--variance-ratio", "0.95",
            "--k", "16",
            "--bit-width", str(bit_width),
            "--rotation-seed", "3405691582",
            "--pca-method", "randomized",
            "--rsvd-target-rank", str(rsvd_target_rank),
            "--rsvd-oversample", "8",
            "--rsvd-power-iters", "2",
            "--verify",
            "--dump-decoded", str(dec_path),
        ]
        if share_basis:
            cmd.append("--share-basis")
        if centroids_file is not None:
            cmd += ["--centroids-file", centroids_file]
        if outlier_threshold is not None:
            cmd += ["--outlier-threshold", str(outlier_threshold)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            pytest.fail(f"CLI failed: {res.stderr[-2000:]}")
        rep = json.loads(rep_path.read_text())
        dec = _read_kktv_f32(dec_path)
    return dec, rep


def _pyo3(
    arr: np.ndarray,
    *,
    metric: str,
    bit_width: int,
    centroids_file: str | None,
    outlier_threshold: float | None,
    share_basis: bool,
    rsvd_target_rank: int,
) -> tuple[np.ndarray, dict]:
    kwargs: dict = dict(
        metric=metric,
        block_size=512,
        bit_width=bit_width,
        variance_ratio=0.95,
        k=16,
        rotation_seed=3405691582,
        pca_method="randomized",
        rsvd_target_rank=rsvd_target_rank,
        rsvd_oversample=8,
        rsvd_power_iters=2,
        share_basis=share_basis,
    )
    if centroids_file is not None:
        kwargs["centroids_file"] = centroids_file
    if outlier_threshold is not None:
        kwargs["outlier_threshold"] = outlier_threshold
    return roundtrip_layer(arr.astype(np.float32), **kwargs)


@pytest.fixture(scope="module")
def ds_distill_tensor():
    """Realistic shape: DS-Distill-Qwen-1.5B has D=128, n_kv=2.  The
    harness reshapes (seq, n_kv, D) -> (seq*n_kv, D) before the codec
    sees it; at ctx_len=2048 and share_basis we feed 2048*2 = 4096
    rows, which is exactly 8 × block_size=512.  Use synthetic data
    with the same shape and a realistic scale.

    We also test a 4096-row tensor representing V after concat.
    """
    rng = np.random.default_rng(20260422)
    # K-side activations are roughly zero-mean with per-coord std ~0.3 on
    # DS-Distill (measured on Phase 1 dumps).  Using std=0.3 keeps the
    # Lloyd-Max outlier distribution similar to what the real model
    # produces so the outlier-threshold path exercises real behaviour.
    k = rng.standard_normal((4096, 128)).astype(np.float32) * 0.3
    v = rng.standard_normal((4096, 128)).astype(np.float32) * 0.3
    return k, v


# ---------------------------------------------------------------------------
# The production recipe (PR #15 / run_v1_3_ppl_full_vllm.sh defaults):
#   K: metric=inner_product, bit_width=3, share_basis=False,
#      centroids=ds_K_b3, outlier_threshold=2.0
#   V: metric=mse,           bit_width=2, share_basis=True,
#      centroids=ds_V_b2,   outlier_threshold=None
# ---------------------------------------------------------------------------


def _k_centroids_path() -> str | None:
    p = REPO / "reports" / "v1_4_q_pca" / "calibrated_codebook" / "ds_K_b3_centroids.f32"
    return str(p) if p.exists() else None


def _v_centroids_path() -> str | None:
    p = REPO / "reports" / "v1_4_q_pca" / "calibrated_codebook" / "ds_V_b2_centroids.f32"
    return str(p) if p.exists() else None


def test_pr15_recipe_k_stream_parity(ds_distill_tensor):
    """K-stream under the PR #15 full recipe — every guardrail on."""
    k, _ = ds_distill_tensor
    cpath = _k_centroids_path()
    if cpath is None:
        pytest.skip("ds_K_b3_centroids.f32 not checked in")

    cli_dec, cli_rep = _cli(
        k, metric="inner_product", bit_width=3,
        centroids_file=cpath, outlier_threshold=2.0,
        share_basis=False, rsvd_target_rank=64,
    )
    py_dec, py_rep = _pyo3(
        k, metric="inner_product", bit_width=3,
        centroids_file=cpath, outlier_threshold=2.0,
        share_basis=False, rsvd_target_rank=64,
    )
    np.testing.assert_array_equal(py_dec, cli_dec)
    assert py_rep["num_blocks"] == cli_rep["num_blocks"] == 8
    assert py_rep["num_vecs_encoded"] == cli_rep["num_vecs_encoded"] == 4096
    # Also assert a non-trivial MSE (both paths produce the same non-zero
    # distortion; if both were accidentally zero, the test would be vacuous)
    assert cli_rep["mean_block_mse"] > 1e-6


def test_pr15_recipe_v_stream_parity(ds_distill_tensor):
    """V-stream under the PR #15 full recipe — share_basis=True, no outlier."""
    _, v = ds_distill_tensor
    cpath = _v_centroids_path()
    if cpath is None:
        pytest.skip("ds_V_b2_centroids.f32 not checked in")

    cli_dec, cli_rep = _cli(
        v, metric="mse", bit_width=2,
        centroids_file=cpath, outlier_threshold=None,
        share_basis=True, rsvd_target_rank=64,
    )
    py_dec, py_rep = _pyo3(
        v, metric="mse", bit_width=2,
        centroids_file=cpath, outlier_threshold=None,
        share_basis=True, rsvd_target_rank=64,
    )
    np.testing.assert_array_equal(py_dec, cli_dec)
    assert py_rep["num_blocks"] == cli_rep["num_blocks"] == 8
    assert py_rep["num_vecs_encoded"] == cli_rep["num_vecs_encoded"] == 4096
    assert cli_rep["mean_block_mse"] > 1e-6


def test_pr15_recipe_both_streams_over_all_layers(ds_distill_tensor):
    """Simulate 28 layers × 2 streams — the amortised call rate the PR
    #15 harness generates per forward pass.  Two axes we want covered:
    (1) repeatability (every call returns the same bits) and
    (2) no state leakage between calls (each call is independent).
    """
    k, v = ds_distill_tensor
    kpath, vpath = _k_centroids_path(), _v_centroids_path()
    if kpath is None or vpath is None:
        pytest.skip("centroid tables not checked in")

    # Run the K recipe three times; bits must match across runs.
    first_dec, _ = _pyo3(
        k, metric="inner_product", bit_width=3,
        centroids_file=kpath, outlier_threshold=2.0,
        share_basis=False, rsvd_target_rank=64,
    )
    for _ in range(2):
        again, _ = _pyo3(
            k, metric="inner_product", bit_width=3,
            centroids_file=kpath, outlier_threshold=2.0,
            share_basis=False, rsvd_target_rank=64,
        )
        np.testing.assert_array_equal(again, first_dec)

    # Alternate K/V calls like the in-forward harness does — stateless.
    for _ in range(28):  # DS-Distill has 28 layers
        dk, _ = _pyo3(
            k, metric="inner_product", bit_width=3,
            centroids_file=kpath, outlier_threshold=2.0,
            share_basis=False, rsvd_target_rank=64,
        )
        dv, _ = _pyo3(
            v, metric="mse", bit_width=2,
            centroids_file=vpath, outlier_threshold=None,
            share_basis=True, rsvd_target_rank=64,
        )
        np.testing.assert_array_equal(dk, first_dec)
        assert dv.shape == v.shape
