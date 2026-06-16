"""Hadamard structure tests using the NumPy reference builder.

These validate the matrix-level properties without requiring MLX,
so they run in the Linux CI path.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from kakeyalattice_mlx.hadamard import _build_hadamard_numpy


@pytest.mark.parametrize("D", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
def test_shape_and_entries(D: int) -> None:
    H = _build_hadamard_numpy(D)
    assert H.shape == (D, D)
    # After 1/√D normalisation the entries should be ±1/√D.
    expected_mag = 1.0 / math.sqrt(D)
    assert np.allclose(np.abs(H), expected_mag)


@pytest.mark.parametrize("D", [2, 4, 8, 16, 64, 128])
def test_self_inverse(D: int) -> None:
    """H·H = I when H is Sylvester–Hadamard / √D."""
    H = _build_hadamard_numpy(D)
    I = H @ H
    assert np.allclose(I, np.eye(D), atol=1e-5)


def test_non_power_of_2_rejected() -> None:
    with pytest.raises(ValueError):
        _build_hadamard_numpy(3)
    with pytest.raises(ValueError):
        _build_hadamard_numpy(0)
    with pytest.raises(ValueError):
        _build_hadamard_numpy(-4)


def test_d1_is_scalar_1() -> None:
    assert _build_hadamard_numpy(1).tolist() == [[1.0]]
