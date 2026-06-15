r"""TurboQuant baseline codec — Hadamard + per-vector qmax + per-coordinate
uniform **scalar** quantisation.

This is exactly the scalar-quantise baseline used throughout the repo's
comparison reports (``ablation_codecs._scalar_quantise_roundtrip``): the same
unit-norm + Sylvester-Hadamard rotation + per-vector qmax preprocessing as
KakeyaLattice, but each rotated coordinate is rounded **independently** to a
uniform grid (no lattice shaping). It is parameterised here by a bit budget
``bits_b`` so it slots into the bit-packed real-byte comparison:

    q_range_tq = 2**(bits_b - 1) - 1     # signed b-bit; 2^b - 1 levels
    bits_per_token_per_head = D * bits_b + 32   # + fp16 norm + fp16 qmax

``roundtrip`` is bit-identical to
``make_ablation_codec("scalar_quantise", D, q_range=2**(b-1)-1)`` — proving this
IS the repo's TurboQuant, only re-expressed in bits-per-coordinate.
"""
from __future__ import annotations

import math

import torch


def _sylvester_hadamard_normalised(D: int, device) -> torch.Tensor:
    assert (D & (D - 1)) == 0, f"D must be power of 2, got {D}"
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], dim=0)
    return H / math.sqrt(D)


class TurboQuantCodec:
    """Scalar-quantise (TurboQuant-style) codec at a fixed bit budget.

    Args:
        D: head dimension (power of 2, for the Hadamard rotation).
        bits_b: bits per coordinate (>= 1). Levels = 2^b - 1, symmetric.
        device: device for the Hadamard matrix.
    """

    def __init__(self, D: int, bits_b: int, device: str = "cuda"):
        assert (D & (D - 1)) == 0, f"D must be power of 2, got {D}"
        assert bits_b >= 1
        self.D_shape = D
        self.bits_b = int(bits_b)
        self.q_range = (1 << (bits_b - 1)) - 1          # 2^(b-1) - 1
        self.H = _sylvester_hadamard_normalised(D, device)
        self.bits_per_token_per_head = D * bits_b + 32
        self.name = f"turboquant-b{bits_b}-bits{self.bits_per_token_per_head}"

    # ----- encode / decode primitives -----

    def encode_to_symbols(self, x: torch.Tensor):
        """x [..., D] -> (symbols in [0, 2^b), norms_fp16, qmax_fp16)."""
        assert x.shape[-1] == self.D_shape
        leading = x.shape[:-1]
        flat = x.reshape(-1, self.D_shape).to(torch.float32)
        eps = torch.finfo(flat.dtype).eps
        norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
        norms_f16 = norms.to(torch.float16)
        unit = flat / norms
        y = unit @ self.H
        qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
        qmax_f16 = qmax.to(torch.float16)
        scale = qmax_f16.to(torch.float32) / float(self.q_range)
        q = torch.round(y / scale).clamp(-self.q_range, self.q_range)
        symbols = (q + self.q_range).to(torch.int64)     # [0, 2^b - 2]
        return (
            symbols.reshape(*leading, self.D_shape),
            norms_f16.reshape(*leading, 1),
            qmax_f16.reshape(*leading, 1),
        )

    def decode_from_symbols(self, symbols, norms_f16, qmax_f16,
                            out_dtype: torch.dtype = torch.bfloat16):
        leading = symbols.shape[:-1]
        q = symbols.reshape(-1, self.D_shape).to(torch.float32) - self.q_range
        norms = norms_f16.reshape(-1, 1).to(torch.float32)
        qmax = qmax_f16.reshape(-1, 1).to(torch.float32)
        scale = qmax / float(self.q_range)
        y_hat = q * scale
        unit_hat = y_hat @ self.H
        x_hat = unit_hat * norms
        return x_hat.to(out_dtype).reshape(*leading, self.D_shape)

    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        """Encode + decode; output dtype matches input."""
        sym, n, m = self.encode_to_symbols(x)
        return self.decode_from_symbols(sym, n, m, out_dtype=x.dtype)


__all__ = ["TurboQuantCodec"]
