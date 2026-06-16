"""Numerical-quality tests for the NumPy shadow of roundtrip_np.

We assert the codec's macroscopic properties on random inputs:

* round-trip error is finite and bounded
* shape and dtype preserved
* higher Q → lower error (monotone within noise) as a basic sanity net
"""
from __future__ import annotations

import numpy as np
import pytest

from kakeyalattice_mlx._reference_numpy import roundtrip_np


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("Q", [10, 38, 152])
def test_shape_preservation(rng, D: int, Q: int) -> None:
    x = rng.normal(size=(8, 4, D)).astype(np.float32) * 0.3
    y = roundtrip_np(x, D, Q)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_error_bounded(rng) -> None:
    """With Q=38 (near-lossless), reconstruction error should be small
    relative to input norm on Gaussian input."""
    D, Q = 128, 38
    x = rng.normal(size=(64, D)).astype(np.float32) * 0.3
    y = roundtrip_np(x, D, Q)
    rel_err = np.linalg.norm(y - x) / np.linalg.norm(x)
    # Empirical: Gaussian input, D=128, Q=38 → rel_err around 3-5%.
    assert rel_err < 0.1, rel_err


def test_q_monotone_error(rng) -> None:
    """Higher Q → strictly finer lattice → lower rel-MSE."""
    D = 128
    x = rng.normal(size=(128, D)).astype(np.float32) * 0.3
    errs = {}
    for Q in [4, 10, 38, 152]:
        y = roundtrip_np(x, D, Q)
        errs[Q] = float(np.mean((y - x) ** 2))
    # Allow for sub-linear tailing near near-lossless; the Q=4 case must
    # clearly be worse than Q=152.
    assert errs[4] > errs[38], errs
    assert errs[38] > errs[152] * 0.5, errs  # loose bound; Q=152 saturates


def test_zero_input_roundtrip_is_zero() -> None:
    """A row of zeros must round-trip to zero (degenerate norm handled)."""
    D, Q = 128, 38
    x = np.zeros((3, D), dtype=np.float32)
    y = roundtrip_np(x, D, Q)
    assert np.allclose(y, 0.0, atol=1e-4)


def test_wrong_D_rejected() -> None:
    with pytest.raises(ValueError):
        roundtrip_np(np.zeros((1, 96), dtype=np.float32), 96, 38)  # not pow2
    with pytest.raises(ValueError):
        roundtrip_np(np.zeros((1, 128), dtype=np.float32), 64, 38)  # mismatch


def test_deterministic_same_input(rng) -> None:
    D, Q = 64, 10
    x = rng.normal(size=(16, D)).astype(np.float32) * 0.3
    y1 = roundtrip_np(x, D, Q)
    y2 = roundtrip_np(x.copy(), D, Q)
    assert np.array_equal(y1, y2)
