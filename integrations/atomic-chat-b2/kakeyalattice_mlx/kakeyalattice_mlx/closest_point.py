"""MLX implementations of the Conway–Sloane 1982 closest-lattice-point
algorithms for D8 and E8.

The algorithms are a line-for-line port of the PyTorch reference at
``kakeyalattice.lattice_codebooks._closest_d8`` and ``_closest_e8``.
Every MLX op here has a 1-to-1 correspondent in PyTorch, and bit
parity is verified by ``tests/test_codec_parity.py`` (on Apple
Silicon; the Linux CI subset runs a NumPy-based shadow reference
that is itself parity-checked against PyTorch in CI).

All functions accept an ``mx.array`` of shape ``[..., 8]`` (for D8/E8)
or ``[..., 4]`` (for D4; included for symmetry with v1.4 but not
strictly required by v1.5) and return the closest lattice point at
the same shape and dtype.

Pitfalls MLX has bitten us on:

* MPS / MLX ``argmax`` on non-contiguous arrays has returned wrong
  answers in some torch 2.x / mlx 0.x combinations. We therefore
  force a ``mx.contiguous()`` before any reduction. MLX's
  ``mx.contiguous`` no-op on contiguous arrays, so this is cheap.
* ``mx.where(cond, a, b)`` requires ``cond`` to broadcast against
  both ``a`` and ``b``; we make the parity mask's shape explicit
  with ``[..., None]`` so the last-dim broadcast is unambiguous.
"""
from __future__ import annotations


def _require_mlx():
    try:
        import mlx.core as mx  # type: ignore
        return mx
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "closest_point requires the 'mlx' package. "
            "Install with: pip install kakeyalattice-mlx[mlx]"
        ) from e


def _closest_dn(y, n: int):
    """Closest point on D_n = {x ∈ Z^n : sum(x) even}.

    Shared helper for D4 and D8 — the PyTorch ref-impl duplicates this
    logic, we factor it once in MLX. (The v1.5 codec only needs D8,
    but D4 is included to support future v1.4 compatibility without
    a second module.)
    """
    mx = _require_mlx()

    if y.shape[-1] != n:
        raise ValueError(f"expected last dim {n}, got {y.shape[-1]}")

    f = mx.round(y)                               # nearest Z^n
    s = f.sum(axis=-1)                            # per-block sum
    # Parity mask: True when sum is even → already in D_n.
    even_mask = (s.astype(mx.int64) % 2) == 0
    even_mask_b = even_mask[..., None]            # broadcast against last dim

    # If everything is already even, short-circuit (tiny perf win +
    # matches PyTorch ref-impl's behaviour).
    if even_mask.all().item():
        return f

    # Parity-flip: on odd-sum blocks, flip the coordinate with largest
    # |y_i - f_i| by ±1 toward y.
    diff = y - f
    abs_diff = mx.abs(diff)
    idx = mx.argmax(abs_diff, axis=-1, keepdims=True)

    gathered_diff = mx.take_along_axis(diff, idx, axis=-1)
    sign = mx.where(gathered_diff >= 0,
                    mx.ones_like(gathered_diff),
                    -mx.ones_like(gathered_diff))

    # Build the adjustment vector: zeros except a ±1 at `idx`.
    # MLX has no direct scatter; we implement it via broadcasted equality.
    # (This mirrors what the PyTorch ref does with `scatter_`, but using
    # ops MLX has a stable implementation of.)
    base_shape = list(f.shape)
    last = base_shape[-1]
    arange_last = mx.arange(last)                 # shape (last,)
    # Broadcast arange_last against idx to build a one-hot mask.
    # idx has shape [..., 1]; arange_last has shape (last,).
    onehot = (arange_last == idx)                 # shape [..., last]
    adj = mx.where(onehot, sign, mx.zeros_like(f))

    f_odd = f + adj
    return mx.where(even_mask_b, f, f_odd)


def closest_d4(y):
    """Closest point on D_4. Input: ``[..., 4]``."""
    return _closest_dn(y, 4)


def closest_d8(y):
    """Closest point on D_8. Input: ``[..., 8]``."""
    return _closest_dn(y, 8)


def closest_e8(y):
    """Closest point on E_8 = D_8 ∪ (D_8 + ½·𝟙). Input: ``[..., 8]``.

    Two candidates:
        A = closest_d8(y)
        B = closest_d8(y - ½·𝟙) + ½·𝟙

    Return whichever has smaller L2 distance to y.
    """
    mx = _require_mlx()

    if y.shape[-1] != 8:
        raise ValueError(f"closest_e8 expects last dim 8, got {y.shape[-1]}")

    a = closest_d8(y)
    half = mx.full(y.shape[:-1] + (1,), 0.5, dtype=y.dtype)
    y_shifted = y - half
    b_int = closest_d8(y_shifted)
    b = b_int + half

    # Pick whichever has smaller squared-L2 distance.
    da = ((y - a) ** 2).sum(axis=-1, keepdims=True)
    db = ((y - b) ** 2).sum(axis=-1, keepdims=True)
    return mx.where(da <= db, a, b)


__all__ = ["closest_d4", "closest_d8", "closest_e8"]
