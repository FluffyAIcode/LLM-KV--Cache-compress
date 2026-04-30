"""MLX-gated parity tests against the NumPy shadow and (optionally) PyTorch.

These only run on a machine with MLX installed (Apple Silicon); on
Linux CI they skip cleanly via ``pytest.importorskip``.

Two parity layers:

* **NumPy parity** (always on Mac): MLX output must match the NumPy
  shadow at ``max_abs_diff <= 1e-5`` in float32. This is the strongest
  equality check we can make without pulling torch into the test
  environment.
* **PyTorch parity** (optional, requires torch + the ``kakeyalattice``
  PyTorch package installed): MLX output must match the canonical
  ``V15KakeyaZamirE8GPU.roundtrip`` at the same tolerance. Gated on
  ``pytest.importorskip("torch")`` and ``importorskip("kakeyalattice")``.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kakeyalattice_mlx._reference_numpy import roundtrip_np   # noqa: E402
from kakeyalattice_mlx.codec import E8LatticeCodebookMLX       # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("Q", [4, 10, 38, 152])
def test_mlx_vs_numpy_parity(rng, D: int, Q: int) -> None:
    x_np = rng.normal(size=(32, D)).astype(np.float32) * 0.3

    # MLX path.
    codec = E8LatticeCodebookMLX(D=D, q_range=Q, dtype=mx.float32)
    x_mx = mx.array(x_np)
    y_mx = codec.roundtrip(x_mx)
    mx.eval(y_mx)
    y_mx_np = np.array(y_mx)

    # NumPy shadow.
    y_np = roundtrip_np(x_np, D, Q)

    max_abs_diff = np.max(np.abs(y_mx_np - y_np))
    assert max_abs_diff <= 1e-5, (
        f"MLX/NumPy parity failure at D={D} Q={Q}: max_abs_diff={max_abs_diff}"
    )


def test_mlx_roundtrip_preserves_shape_and_dtype() -> None:
    D, Q = 128, 38
    codec = E8LatticeCodebookMLX(D=D, q_range=Q, dtype=mx.float32)

    x = mx.random.normal(shape=(2, 4, 8, D))
    y = codec.roundtrip(x)
    mx.eval(y)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_mlx_zero_input_roundtrip() -> None:
    D, Q = 128, 38
    codec = E8LatticeCodebookMLX(D=D, q_range=Q, dtype=mx.float32)
    x = mx.zeros((3, D), dtype=mx.float32)
    y = codec.roundtrip(x)
    mx.eval(y)
    assert float(mx.max(mx.abs(y)).item()) < 1e-4


@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("Q", [38, 10])
def test_mlx_vs_pytorch_parity_if_available(rng, D: int, Q: int) -> None:
    """Optional PyTorch-side parity check.

    This runs only when both torch and kakeyalattice (PyTorch pkg) are
    installed; otherwise it skips. On a Mac dev box with the B1 env
    already set up, this will actually execute and give us a 3-way
    parity guarantee.
    """
    torch = pytest.importorskip("torch")
    kakeya = pytest.importorskip("kakeyalattice")

    x_np = rng.normal(size=(16, D)).astype(np.float32) * 0.3

    # MLX.
    codec_mx = E8LatticeCodebookMLX(D=D, q_range=Q, dtype=mx.float32)
    y_mx_np = np.array(mx.eval(codec_mx.roundtrip(mx.array(x_np))))

    # PyTorch ref on CPU (avoids MPS/CUDA skew for parity).
    codec_pt = kakeya.V15KakeyaZamirE8GPU(D=D, q_range=Q, device="cpu")
    x_pt = torch.from_numpy(x_np)
    y_pt = codec_pt.roundtrip(x_pt)
    y_pt_np = y_pt.detach().cpu().numpy()

    max_abs_diff = float(np.max(np.abs(y_mx_np - y_pt_np)))
    assert max_abs_diff <= 1e-4, (
        f"MLX/PyTorch parity failure at D={D} Q={Q}: "
        f"max_abs_diff={max_abs_diff}"
    )
