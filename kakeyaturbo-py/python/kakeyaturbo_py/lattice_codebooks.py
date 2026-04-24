"""Nested-lattice codebooks for LLM KV compression.

Shared architecture:
    x ∈ R^D (per-head K or V vector)
      → unit-normalise, store ‖x‖ (fp16)
      → Sylvester Hadamard rotation (energy equipartition)
      → per-vector qmax, store qmax (fp16)
      → scale y_scaled = y / (qmax / Q) into [-Q, Q]^D
      → split into (D / block_dim) blocks
      → closest-lattice-point per block  ← only step that depends on lattice
      → clamp to ±Q
      → decode, inverse Hadamard, rescale by ‖x‖

Subclasses provide only:
    * `block_dim`  (4 for D4, 8 for E8)
    * `_closest_lattice_point(y: [..., block_dim]) -> [..., block_dim]`
    * `bits_per_block(q_range)`  (for bit accounting)
    * `short_name`, `shaping_gain_db`  (metadata)

Public API:
    * `LatticeCodebook`  — base, never instantiated directly
    * `D4LatticeCodebook`  — v1.4 lattice (4-D, G(Λ)=0.0766, +0.37 dB vs Z^4)
    * `E8LatticeCodebook`  — v1.5 lattice (8-D, G(Λ)=0.0717, +0.66 dB vs Z^8)

Both preserve the `D4TQStyleCodebook.roundtrip(x)` contract, so any
harness that accepts a `SphericalCodebook` works with either.
"""
from __future__ import annotations

import math

import torch

from .spherical_codebooks import SphericalCodebook


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sylvester_hadamard_normalised(D: int, device) -> torch.Tensor:
    """Sylvester Hadamard H_D / √D (self-inverse at orthonormal normalisation).

    Same definition as the D4TQStyleCodebook implementation; kept local here
    so the E8 codebook does not import from the research-lineage module.
    """
    assert (D & (D - 1)) == 0, f"D must be power of 2, got {D}"
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
    return H / math.sqrt(D)


# ---------------------------------------------------------------------------
# Closest-lattice-point implementations (Conway-Sloane 1982).
# ---------------------------------------------------------------------------

def _closest_d4(y: torch.Tensor) -> torch.Tensor:
    """Alg 4 (Conway-Sloane 1982).  y: [..., 4].

    D4 = {x ∈ Z^4 : sum(x) even}.  Given y ∈ R^4, return closest D4 point.

    Procedure:
      1. f = round(y).  Nearest Z^4 point.
      2. If sum(f) is even: f ∈ D4, return it.
      3. Else: flip the coord with largest |y_i - f_i| by ±1 toward y.
    """
    f = torch.round(y)
    s = f.sum(dim=-1)
    even_mask = (s.to(torch.int64) % 2) == 0
    if even_mask.all():
        return f

    diff = y - f
    idx = diff.abs().argmax(dim=-1, keepdim=True)
    sign = torch.where(
        diff.gather(-1, idx) >= 0,
        torch.ones_like(diff[..., :1]),
        -torch.ones_like(diff[..., :1]),
    )
    adj = torch.zeros_like(f).scatter_(-1, idx, sign)
    f_odd = f + adj

    return torch.where(even_mask.unsqueeze(-1), f, f_odd)


def _closest_d8(y: torch.Tensor) -> torch.Tensor:
    """Same algorithm as D4 but for 8-D: D8 = {x ∈ Z^8 : sum(x) even}."""
    f = torch.round(y)
    s = f.sum(dim=-1)
    even_mask = (s.to(torch.int64) % 2) == 0
    if even_mask.all():
        return f

    diff = y - f
    idx = diff.abs().argmax(dim=-1, keepdim=True)
    sign = torch.where(
        diff.gather(-1, idx) >= 0,
        torch.ones_like(diff[..., :1]),
        -torch.ones_like(diff[..., :1]),
    )
    adj = torch.zeros_like(f).scatter_(-1, idx, sign)
    f_odd = f + adj

    return torch.where(even_mask.unsqueeze(-1), f, f_odd)


def _closest_e8(y: torch.Tensor) -> torch.Tensor:
    """Alg 5 (Conway-Sloane 1982).  y: [..., 8].

    E8 = D8 ∪ (D8 + ½·𝟙)
       = {x ∈ Z^8 ∪ (Z + ½)^8 : sum(x) ∈ 2Z}

    Procedure:
      1. Candidate A: closest D8 point to y.
      2. Candidate B: closest D8 point to (y - ½·𝟙), shifted back by +½·𝟙.
      3. Return whichever has smaller L2 distance to y.
    """
    # Candidate A: integer coset.
    a = _closest_d8(y)
    # Candidate B: half-integer coset.
    half = torch.full_like(y[..., :1], 0.5)
    y_shifted = y - half                                    # align half-int grid to int grid
    b_int = _closest_d8(y_shifted)
    b = b_int + half                                        # shift back

    # Pick the closer candidate per block.
    da = ((y - a) ** 2).sum(dim=-1, keepdim=True)
    db = ((y - b) ** 2).sum(dim=-1, keepdim=True)
    return torch.where(da <= db, a, b)


# ---------------------------------------------------------------------------
# Base class — Hadamard + per-vec qmax wrapper around any lattice.
# ---------------------------------------------------------------------------

class LatticeCodebook(SphericalCodebook):
    """Hadamard-rotated, per-vector-scaled nested-lattice codec.

    Subclasses override:
      * `block_dim: int`
      * `_closest_lattice_point(y_scaled) -> y_lattice`
      * `_bits_per_block_real(q_range) -> float`  (non-integer, before ceil)
      * `short_name: str`
      * `shaping_gain_db: float`  (over Z^{block_dim}; for documentation)

    Bit accounting:
      * `bits/block` = `ceil(_bits_per_block_real(Q))`
      * Overhead: 2 fp16 scalars (‖x‖ and qmax) = 32 bits / vector
      * Total: (D / block_dim) * bits/block + 32
    """

    # Subclass must set these.
    block_dim: int = 0
    short_name: str = "lattice"
    shaping_gain_db: float = 0.0

    def __init__(
        self,
        D: int,
        q_range: int = 38,
        device: str = "cuda",
    ):
        bd = self.block_dim
        assert bd > 0, f"subclass {type(self).__name__} must set block_dim"
        assert D % bd == 0, (
            f"D must be divisible by block_dim={bd}, got D={D}"
        )
        assert (D & (D - 1)) == 0, (
            f"D must be power of 2 (Hadamard), got D={D}"
        )
        assert q_range >= 1, f"q_range must be >= 1, got {q_range}"

        self.D_shape = D
        self.K_blocks = D // bd
        self.q_range = q_range
        self.H = _sylvester_hadamard_normalised(D, device)

        # Per-block lattice bits: ceil(real bits), summed over K_blocks.
        bits_per_block = int(math.ceil(self._bits_per_block_real(q_range)))
        total_lattice_bits = self.K_blocks * bits_per_block
        overhead_bits = 32                       # fp16 ‖x‖ + fp16 qmax
        total_bits = total_lattice_bits + overhead_bits

        dummy = torch.zeros(1, D, device=device, dtype=torch.float32)
        super().__init__(
            codewords=dummy,
            name=(
                f"{self.short_name}-Q{q_range}-bits{total_bits}"
                f"(lat{total_lattice_bits}+oh{overhead_bits})"
            ),
            bits=total_bits,
        )

    # -- Subclass hooks --
    def _closest_lattice_point(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _bits_per_block_real(self, q_range: int) -> float:
        raise NotImplementedError

    # -- Codec interface --
    def encode(self, x: torch.Tensor):
        raise NotImplementedError(
            f"{type(self).__name__}.encode(): use roundtrip(x) instead — "
            f"this codec has per-vector state (qmax, ‖x‖) not representable "
            f"as a flat (seg_id, t) index pair."
        )

    def decode(self, seg_id: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError(
            f"{type(self).__name__}.decode(): use roundtrip(x) instead."
        )

    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        """Encode + decode round-trip.

        Input:  x of shape [..., D]
        Output: same shape, reconstruction of x.
        """
        assert x.shape[-1] == self.D_shape, (
            f"expected last dim {self.D_shape}, got {x.shape[-1]}"
        )
        batch = x.shape[:-1]
        flat = x.reshape(-1, self.D_shape).to(torch.float32)
        N = flat.shape[0]
        eps = torch.finfo(flat.dtype).eps

        # 1. Unit-normalise + fp16 norm storage.
        norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
        norms_f16 = norms.to(torch.float16).to(torch.float32)
        unit = flat / norms

        # 2. Hadamard rotation.
        y = unit @ self.H

        # 3. Per-vector qmax + fp16 storage.
        qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
        qmax_f16 = qmax.to(torch.float16).to(torch.float32)
        scale = qmax_f16 / float(self.q_range)

        # 4. Scale to lattice coordinates.
        y_scaled = y / scale

        # 5. Closest-lattice-point per block.  Subclass-specific.
        y_blocks = y_scaled.reshape(N, self.K_blocks, self.block_dim)
        q_lat = self._closest_lattice_point(y_blocks)

        # 6. Clamp defensively against parity-flip going slightly out of range.
        q_lat = q_lat.clamp(-self.q_range, self.q_range)

        # 7. Rescale back to coord space.
        y_hat = (q_lat.to(torch.float32) * scale.unsqueeze(-1)).reshape(
            N, self.D_shape,
        )

        # 8. Inverse Hadamard (H is self-inverse after 1/√D normalisation).
        unit_hat = y_hat @ self.H

        # 9. Restore original scale via fp16-rounded ‖x‖.
        x_hat_flat = unit_hat * norms_f16
        return x_hat_flat.reshape(*batch, self.D_shape)


# ---------------------------------------------------------------------------
# D4 concrete subclass — 4-D, Conway-Sloane Alg 4.  Used by v1.4.
# ---------------------------------------------------------------------------

class D4LatticeCodebook(LatticeCodebook):
    """D4 nested lattice (4-D root lattice).

    D4 = {x ∈ Z^4 : sum(x) even}.  Voronoi region is the 24-cell, giving
    G(Λ) = 0.076603 vs G(Z^4) = 1/12 ≈ 0.083333.  Shaping gain over Z^4:
    +0.37 dB.  This is the lattice used by v1.4 KakeyaLattice.

    Bit accounting:
      * Point density: half of Z^4 (parity constraint) → 1/2 bit/coord saved
      * bits_per_block_real(Q) = 4·log₂(2Q+1) − 1
      * For head_dim=128 at Q=38: 32 blocks × 25 = 800 lattice bits + 32 overhead = 832 total
    """
    block_dim = 4
    short_name = "D4Lattice"
    shaping_gain_db = 0.37

    def _closest_lattice_point(self, y: torch.Tensor) -> torch.Tensor:
        return _closest_d4(y)

    def _bits_per_block_real(self, q_range: int) -> float:
        # D4 count in [-Q, Q]^4: ~(2Q+1)^4 / 2 (even-sum constraint).
        return 4.0 * math.log2(2 * q_range + 1) - 1.0


# ---------------------------------------------------------------------------
# E8 concrete subclass — 8-D, Conway-Sloane Alg 5.  Used by v1.5.
# ---------------------------------------------------------------------------

class E8LatticeCodebook(LatticeCodebook):
    """E8 nested lattice (8-D, the densest 8-D lattice).

    E8 = D8 ∪ (D8 + ½·𝟙) where D8 = {x ∈ Z^8 : sum(x) even}.  Voronoi
    region approaches the sphere more closely than D4 (G(Λ) = 0.071682
    vs 0.076603).  Shaping gain over Z^8: +0.66 dB — i.e. +0.29 dB over
    D4 at matched rate.

    Bit accounting:
      * Point density in [-Q, Q]^8: ~(2Q+1)^8 (half D8 integer points +
        half D8+½·𝟙 half-integer points, asymptotic equality for large Q).
      * bits_per_block_real(Q) = 8·log₂(2Q+1)
      * For head_dim=128 at Q=38: 16 blocks × 51 = 816 lattice bits + 32 overhead = 848 total
      * At iso-bit to v1.4 D4 Q=38 (832 bits): E8 uses Q=37 → 16 × 50 + 32 = 832

    GPU kernel complexity: ~2.5× the ops-per-block of D4 (two D8
    sub-candidates + min), but block count halves, so per-vector
    compute is about +25-30% vs D4.  Small warp divergence inside the
    D8 parity-flip branch.
    """
    block_dim = 8
    short_name = "E8Lattice"
    shaping_gain_db = 0.66

    def _closest_lattice_point(self, y: torch.Tensor) -> torch.Tensor:
        return _closest_e8(y)

    def _bits_per_block_real(self, q_range: int) -> float:
        # E8 count in [-Q, Q]^8 ≈ (2Q+1)^8 (half D8 + half D8+½ with
        # roughly matching density for Q large enough).  The parity
        # constraint (sum even within each coset) halves each coset,
        # but the union of two cosets restores the full Z^8 density, so
        # there's no −1 bit here — unlike D4's case.
        return 8.0 * math.log2(2 * q_range + 1)


__all__ = [
    "LatticeCodebook",
    "D4LatticeCodebook",
    "E8LatticeCodebook",
]
