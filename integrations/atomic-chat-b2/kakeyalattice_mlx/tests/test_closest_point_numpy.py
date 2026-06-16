"""Tests for the NumPy shadow of closest-D8 / closest-E8.

These are the "platform-agnostic" logic tests for the Conway–Sloane
algorithm used by v1.5 E8. They verify algebraic properties of the
lattice points (parity of D8, E8 as union of two cosets) and
invariances that must hold regardless of backend.

The MLX-native parity tests live in ``test_codec_mlx_parity.py`` and
are gated on ``mx.metal.is_available()``.
"""
from __future__ import annotations

import numpy as np
import pytest

from kakeyalattice_mlx._reference_numpy import closest_d8_np, closest_e8_np


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_closest_d8_returns_integer_coords(rng) -> None:
    y = rng.normal(size=(100, 8)).astype(np.float32) * 5.0
    x = closest_d8_np(y)
    # D8 points are in Z^8.
    assert np.allclose(x, np.round(x))
    # Sum of coords is always even (that's the defining constraint).
    sums = x.sum(axis=-1).astype(np.int64)
    assert np.all(sums % 2 == 0), sums[np.where(sums % 2 != 0)]


def test_closest_e8_coset_structure(rng) -> None:
    """E8 = Z^8 ∪ (Z + ½)^8. Every returned point must be one or the other.

    Concretely: 2·x is always an integer, and if 2·x is even, all coords
    are integers (Z^8 coset); if 2·x is odd, all coords are half-integers
    ((Z + ½)^8 coset).
    """
    y = rng.normal(size=(200, 8)).astype(np.float32) * 5.0
    x = closest_e8_np(y)

    two_x = 2 * x
    assert np.allclose(two_x, np.round(two_x)), "E8 points must be half-integer"

    # Each row is all-integer or all-half-integer (not mixed).
    # Translate: row is "all coords have same fractional part ∈ {0, 0.5}".
    frac = x - np.floor(x)
    # Each row's fractional parts should all be 0 OR all be 0.5.
    row_ok = np.all((frac == 0) | (np.isclose(frac, 0.5)), axis=-1) & (
        np.all(frac == 0, axis=-1) | np.all(np.isclose(frac, 0.5), axis=-1)
    )
    assert row_ok.all(), (
        "rows with mixed integer/half-integer coords: "
        f"{np.where(~row_ok)[0][:5]}"
    )


def test_closest_e8_is_at_least_as_close_as_d8(rng) -> None:
    """E8 ⊃ D8 in density, so closest_e8 must be <= closest_d8 in distance."""
    y = rng.normal(size=(500, 8)).astype(np.float32) * 5.0
    x_d8 = closest_d8_np(y)
    x_e8 = closest_e8_np(y)
    dist_d8 = np.linalg.norm(y - x_d8, axis=-1)
    dist_e8 = np.linalg.norm(y - x_e8, axis=-1)
    # Allow an epsilon for float32 tie edge cases.
    assert np.all(dist_e8 <= dist_d8 + 1e-5), (
        np.where(dist_e8 > dist_d8 + 1e-5)[0][:5]
    )


def test_already_in_lattice_returns_self() -> None:
    """If y is already a D8 or E8 point, closest-point should return y."""
    # A D8 point: (1, 1, 0, 0, 0, 0, 0, 0), sum=2 (even).
    y_d8 = np.array([[1.0, 1.0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    assert np.allclose(closest_d8_np(y_d8), y_d8)

    # An E8 half-integer point: (0.5,)*8, sum=4 (even).
    y_e8 = np.full((1, 8), 0.5, dtype=np.float32)
    out = closest_e8_np(y_e8)
    assert np.allclose(out, y_e8), out


def test_shape_preservation() -> None:
    """closest_{d8,e8} must preserve arbitrary leading dims."""
    y = np.random.randn(3, 5, 7, 8).astype(np.float32)
    x_d8 = closest_d8_np(y)
    x_e8 = closest_e8_np(y)
    assert x_d8.shape == y.shape
    assert x_e8.shape == y.shape


def test_wrong_last_dim_raises() -> None:
    y4 = np.zeros((10, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        closest_d8_np(y4)
    with pytest.raises(ValueError):
        closest_e8_np(y4)
