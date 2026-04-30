"""Bit-accounting tests — platform-agnostic, no MLX needed.

These mirror the PyTorch reference's bit count formula exactly and
guard against drift between the B1 (PyTorch) and B2 (MLX) packages.
"""
from __future__ import annotations

import math

import pytest

from kakeyalattice_mlx.codec import bits_per_token_per_head


def _ref_bits(D: int, Q: int) -> int:
    """PyTorch reference formula from kakeyalattice.E8LatticeCodebook:
    (D/8) · ⌈8·log₂(2Q+1)⌉ + 32
    """
    return (D // 8) * int(math.ceil(8.0 * math.log2(2 * Q + 1))) + 32


@pytest.mark.parametrize("D", [64, 128, 256, 512])
@pytest.mark.parametrize("Q", [2, 4, 10, 38, 76, 152])
def test_bits_match_reference(D: int, Q: int) -> None:
    assert bits_per_token_per_head(D, Q) == _ref_bits(D, Q)


def test_bits_known_values_d128() -> None:
    """Canonical points published in v1.5 README / paper."""
    # 16 blocks × 51 bits + 32 = 848
    assert bits_per_token_per_head(128, 38) == 848
    # 16 × 35.14→36 + 32 = 608
    assert bits_per_token_per_head(128, 10) == 608
    # 16 × ⌈8·log2(9)⌉ + 32 = 16·26 + 32 = 448 (matches v1.5 CR=4.57x)
    assert bits_per_token_per_head(128, 4) == 448
    # Q=152 near-lossless: 16 × ⌈8·log2(305)⌉ + 32 = 16·67 + 32 = 1104
    assert bits_per_token_per_head(128, 152) == 1104


def test_invalid_D_rejected() -> None:
    with pytest.raises(ValueError):
        bits_per_token_per_head(100, 38)  # not divisible by 8
    with pytest.raises(ValueError):
        bits_per_token_per_head(0, 38)


def test_bits_formula_accepts_d_not_power_of_2_but_div_8() -> None:
    """``bits_per_token_per_head`` only needs D % 8 == 0.

    The power-of-2 constraint belongs to the Hadamard rotation in
    ``E8LatticeCodebookMLX`` construction, not to the bit formula —
    so a hypothetical D=96 (3·8·4, divisible by 8 but not a power of
    2) has a well-defined bit count even though we can't instantiate
    the codec on it. Keep the two concerns separate.
    """
    bits = bits_per_token_per_head(96, 38)
    assert bits == (96 // 8) * 51 + 32  # 12·51 + 32 = 644


def test_invalid_Q_rejected() -> None:
    with pytest.raises(ValueError):
        bits_per_token_per_head(128, 0)


def test_compression_ratio_vs_bf16_at_d128() -> None:
    """At D=128 head_dim, bf16 baseline = 128*16 = 2048 bits/vec.
    E8 Q=10 should give ~3.37× CR per v1.5 report §3.
    """
    bf16 = 128 * 16
    cr_q10 = bf16 / bits_per_token_per_head(128, 10)
    assert 3.3 < cr_q10 < 3.5, cr_q10
    cr_q38 = bf16 / bits_per_token_per_head(128, 38)
    assert 2.3 < cr_q38 < 2.5, cr_q38
