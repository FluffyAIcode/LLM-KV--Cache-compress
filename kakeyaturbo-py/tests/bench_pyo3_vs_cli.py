"""Wall-clock speedup: in-process pyo3 vs kakeyaturbo-bench subprocess.

This is not a test — it's a measurement script, run manually.  Its
output is the "why M3 matters" artefact that lands in the M3 report.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BENCH_BIN = REPO / "kakeyaturbo" / "target" / "release" / "kakeyaturbo-bench"
KKTV_MAGIC = 0x4B4B5456

from kakeyaturbo_py import roundtrip_layer


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


def cli_roundtrip(arr, centroids_path: str) -> float:
    t0 = time.perf_counter()
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
            "--metric", "inner_product",
            "--block-size", "512",
            "--variance-ratio", "0.95",
            "--k", "16",
            "--bit-width", "3",
            "--rotation-seed", "3405691582",
            "--pca-method", "randomized",
            "--rsvd-target-rank", "64",
            "--rsvd-oversample", "8",
            "--rsvd-power-iters", "2",
            "--verify",
            "--dump-decoded", str(dec_path),
            "--centroids-file", centroids_path,
            "--outlier-threshold", "2.0",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        _ = json.loads(rep_path.read_text())
        _ = _read_kktv_f32(dec_path)
    return time.perf_counter() - t0


def pyo3_roundtrip(arr, centroids_path: str) -> float:
    t0 = time.perf_counter()
    _, _ = roundtrip_layer(
        arr, metric="inner_product", block_size=512, bit_width=3,
        variance_ratio=0.95, k=16, rotation_seed=3405691582,
        pca_method="randomized", rsvd_target_rank=64,
        rsvd_oversample=8, rsvd_power_iters=2,
        share_basis=False,
        centroids_file=centroids_path,
        outlier_threshold=2.0,
    )
    return time.perf_counter() - t0


def main() -> int:
    centroids_path = str(REPO / "reports" / "v1_4_q_pca" / "calibrated_codebook"
                         / "ds_K_b3_centroids.f32")
    if not Path(centroids_path).exists():
        print(f"centroids not present at {centroids_path}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(20260422)
    # 4096 × 128 float32 matches one DS-Distill layer's K tensor after
    # the (seq, n_kv, D) -> (seq*n_kv, D) reshape at ctx_len=2048.
    arr = rng.standard_normal((4096, 128)).astype(np.float32) * 0.3

    # Warm up both paths so we measure steady-state.
    _ = cli_roundtrip(arr, centroids_path)
    _ = pyo3_roundtrip(arr, centroids_path)

    n = 30
    cli_t = np.array([cli_roundtrip(arr, centroids_path) for _ in range(n)])
    py_t = np.array([pyo3_roundtrip(arr, centroids_path) for _ in range(n)])

    def stats(xs):
        return dict(
            median=float(np.median(xs) * 1000),
            mean=float(np.mean(xs) * 1000),
            min=float(np.min(xs) * 1000),
            p95=float(np.percentile(xs, 95) * 1000),
        )

    out = {
        "workload": "PR #15 K-stream, 4096×128 float32, "
                    "randomized rank64 + centroids + outlier T=2.0",
        "n_iterations": n,
        "cli_subprocess_ms": stats(cli_t),
        "pyo3_in_process_ms": stats(py_t),
        "speedup_median":   float(np.median(cli_t) / np.median(py_t)),
        "speedup_mean":     float(np.mean(cli_t) / np.mean(py_t)),
    }
    print(json.dumps(out, indent=2))

    # Projected amortized savings per forward pass: 28 layers × 2 streams
    # = 56 calls, so per-forward-pass cost ≈ 56 × median.
    cli_fwd = float(np.median(cli_t) * 56)
    py_fwd = float(np.median(py_t) * 56)
    print(f"\nProjected PR #15 per-forward-pass cost (28 layers × 2 streams):")
    print(f"  CLI  : {cli_fwd:.2f} s")
    print(f"  pyo3 : {py_fwd*1000:.1f} ms  ({cli_fwd / py_fwd:.1f}× faster)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
