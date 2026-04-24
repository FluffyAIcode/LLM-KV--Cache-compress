"""Ablation codec variants for v1.4 KakeyaLattice.

Six codecs, each removing exactly one engineering factor from the full
v1.4 stack.  This isolates the contribution of every design choice.

Reference table (v1.4 full stack has all six ON):

  Variant name          unit_norm  Hadamard  per_vec_qmax  joint_scale  D4_lattice  bdry_layers
  ------------------    ---------  --------  ------------  -----------  ----------  -----------
  v14_full              yes        yes       yes           yes          yes         bf16 (2+2)
  no_unit_norm          NO         yes       yes           yes          yes         yes
  no_hadamard           yes        NO        yes           yes          yes         yes
  no_per_vec_qmax       yes        yes       NO (global)   yes          yes         yes
  no_joint_scale        yes        yes       yes           NO (per-blk) yes         yes
  scalar_not_d4         yes        yes       yes           yes          NO (Z^4)    yes
  no_boundary           yes        yes       yes           yes          yes         NO (all layers)

"unit_norm":      divide by L2 norm, store norm as fp16, rescale at decode
"Hadamard":       Sylvester H_D / sqrt(D) rotation before quantisation
"per_vec_qmax":   each vector gets its own qmax (fp16); alternative is a
                  single global qmax computed over the dataset
"joint_scale":    same qmax shared across all 4-D blocks within a vector
                  (D4 lattice across the whole head); alternative is a
                  separate per-block qmax (lower overhead but worse shaping)
"D4_lattice":     Conway-Sloane closest-point in D4; alternative is
                  independent per-coord Z^4 rounding (scalar quantise)
"boundary":       keep first 2 + last 2 layers bf16; alternative is apply
                  the codec to every layer

Only the LAST one (boundary) changes the harness sweep, not the codec.
The harness handles it by passing an empty boundary set.

All codecs preserve the exact bit budget of v14_full at the same Q, so
the iso-bit / iso-PPL comparison is clean (TQ scalar quantise with bits
per coord = b gets its own bit accounting).
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


def _closest_d4_lattice_point(y: torch.Tensor) -> torch.Tensor:
    """Conway-Sloane D4 = {x ∈ Z^4 : sum(x) even}.  y: [..., 4]."""
    f = torch.round(y)
    s = f.sum(dim=-1)
    even_mask = (s.to(torch.int64) % 2) == 0
    if even_mask.all():
        return f
    diff = y - f
    abs_diff = diff.abs()
    idx = abs_diff.argmax(dim=-1, keepdim=True)
    sign = torch.where(
        diff.gather(-1, idx) >= 0,
        torch.ones_like(diff[..., :1]),
        -torch.ones_like(diff[..., :1]),
    )
    adj = torch.zeros_like(f)
    adj.scatter_(-1, idx, sign)
    f_odd = f + adj
    return torch.where(even_mask.unsqueeze(-1), f, f_odd)


def _v14_full_roundtrip(
    X: torch.Tensor, D: int, q_range: int, H: torch.Tensor,
) -> torch.Tensor:
    """The full v1.4 stack — same algorithm as V14KakeyaZamirLatticeGPU.

    All six factors ON.  Parity-checked against the canonical class.
    """
    assert X.is_cuda and X.dtype == torch.float32
    N_tok, H_heads, _ = X.shape
    flat = X.reshape(-1, D)
    eps = torch.finfo(torch.float32).eps

    norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
    norms_f16 = norms.to(torch.float16).to(torch.float32)
    unit = flat / norms
    y = unit @ H

    qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
    qmax_f16 = qmax.to(torch.float16).to(torch.float32)
    scale = qmax_f16 / float(q_range)

    y_scaled = y / scale
    y_blocks = y_scaled.reshape(-1, D // 4, 4)
    q_lat = _closest_d4_lattice_point(y_blocks).clamp(-q_range, q_range)
    y_hat = (q_lat * scale.unsqueeze(-1)).reshape(-1, D)

    unit_hat = y_hat @ H
    return (unit_hat * norms_f16).reshape(N_tok, H_heads, D)


# ---------- ablation #1: remove unit-norm ----------
def _no_unit_norm_roundtrip(
    X: torch.Tensor, D: int, q_range: int, H: torch.Tensor,
) -> torch.Tensor:
    """Drop the L2 unit-normalise step.  Hadamard applies directly to x,
    and the per-vector qmax absorbs the dynamic range that unit-norm
    would otherwise factor out.  Bit budget unchanged (still 16 bits qmax
    but 0 bits for the absent norm — so saves 16 bits per vector; we
    fold that back into overhead accounting to keep matched bits).
    """
    assert X.is_cuda and X.dtype == torch.float32
    N_tok, H_heads, _ = X.shape
    flat = X.reshape(-1, D)
    eps = torch.finfo(torch.float32).eps

    y = flat @ H
    qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
    qmax_f16 = qmax.to(torch.float16).to(torch.float32)
    scale = qmax_f16 / float(q_range)

    y_scaled = y / scale
    y_blocks = y_scaled.reshape(-1, D // 4, 4)
    q_lat = _closest_d4_lattice_point(y_blocks).clamp(-q_range, q_range)
    y_hat = (q_lat * scale.unsqueeze(-1)).reshape(-1, D)

    return (y_hat @ H).reshape(N_tok, H_heads, D)


# ---------- ablation #2: remove Hadamard rotation ----------
def _no_hadamard_roundtrip(
    X: torch.Tensor, D: int, q_range: int, H: torch.Tensor,
) -> torch.Tensor:
    """Drop the Hadamard rotation.  Per-vector qmax and D4 lattice
    operate directly on the (unit-normalised) coordinate basis, so
    outlier coordinates dominate qmax and waste bits on well-behaved
    ones — this is the outlier-sensitivity factor that Hadamard solves.
    """
    assert X.is_cuda and X.dtype == torch.float32
    N_tok, H_heads, _ = X.shape
    flat = X.reshape(-1, D)
    eps = torch.finfo(torch.float32).eps

    norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
    norms_f16 = norms.to(torch.float16).to(torch.float32)
    unit = flat / norms
    # Skip Hadamard.
    y = unit

    qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
    qmax_f16 = qmax.to(torch.float16).to(torch.float32)
    scale = qmax_f16 / float(q_range)

    y_scaled = y / scale
    y_blocks = y_scaled.reshape(-1, D // 4, 4)
    q_lat = _closest_d4_lattice_point(y_blocks).clamp(-q_range, q_range)
    y_hat = (q_lat * scale.unsqueeze(-1)).reshape(-1, D)

    return (y_hat * norms_f16).reshape(N_tok, H_heads, D)


# ---------- ablation #3: remove per-vector qmax (use global) ----------
class GlobalQmaxCodec:
    """Use a single global qmax for all vectors.  The qmax is computed
    ONCE from the first batch of K/V we see (simulating a pre-calibrated
    production deployment where qmax has been set offline), then kept
    fixed.  This isolates the value of per-vector adaptive scaling.

    Stored as state because the "global" qmax must persist across layers
    and passages.
    """

    def __init__(self, D: int, q_range: int, device: str):
        self.D = D
        self.q_range = q_range
        self.H = _sylvester_hadamard_normalised(D, device)
        self.global_qmax: torch.Tensor | None = None

    def calibrate(self, X_cal: torch.Tensor) -> None:
        """Compute global qmax from a calibration batch."""
        assert X_cal.is_cuda
        flat = X_cal.reshape(-1, self.D).to(torch.float32)
        eps = torch.finfo(torch.float32).eps
        norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
        unit = flat / norms
        y = unit @ self.H
        self.global_qmax = y.abs().max().clamp(min=eps).to(torch.float32)

    def roundtrip(self, X: torch.Tensor) -> torch.Tensor:
        assert X.is_cuda and X.dtype == torch.float32
        assert self.global_qmax is not None, "Must .calibrate() before .roundtrip()"
        N_tok, H_heads, _ = X.shape
        flat = X.reshape(-1, self.D)
        eps = torch.finfo(torch.float32).eps

        norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
        norms_f16 = norms.to(torch.float16).to(torch.float32)
        unit = flat / norms
        y = unit @ self.H

        scale = self.global_qmax / float(self.q_range)
        y_scaled = y / scale
        y_blocks = y_scaled.reshape(-1, self.D // 4, 4)
        q_lat = _closest_d4_lattice_point(y_blocks).clamp(
            -self.q_range, self.q_range,
        )
        y_hat = (q_lat * scale).reshape(-1, self.D)

        unit_hat = y_hat @ self.H
        return (unit_hat * norms_f16).reshape(N_tok, H_heads, self.D)


# ---------- ablation #4: per-block qmax (no joint scaling across blocks) ----------
def _per_block_qmax_roundtrip(
    X: torch.Tensor, D: int, q_range: int, H: torch.Tensor,
) -> torch.Tensor:
    """Each 4-D block has its own qmax.  Saves per-vector joint coupling
    but costs more overhead bits (one fp16 qmax per block instead of
    one per vector).  We still only charge 32 bits overhead per vector
    here (so this is an UPPER bound on what per-block would buy — if
    we charged honestly it would lose even more).
    """
    assert X.is_cuda and X.dtype == torch.float32
    N_tok, H_heads, _ = X.shape
    flat = X.reshape(-1, D)
    eps = torch.finfo(torch.float32).eps

    norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
    norms_f16 = norms.to(torch.float16).to(torch.float32)
    unit = flat / norms
    y = unit @ H

    # Per-block qmax instead of per-vector.
    y_blocks_raw = y.reshape(-1, D // 4, 4)
    qmax_blk = y_blocks_raw.abs().max(dim=-1, keepdim=True).values.clamp(min=eps)
    qmax_blk_f16 = qmax_blk.to(torch.float16).to(torch.float32)
    scale_blk = qmax_blk_f16 / float(q_range)

    y_scaled = y_blocks_raw / scale_blk
    q_lat = _closest_d4_lattice_point(y_scaled).clamp(-q_range, q_range)
    y_hat = (q_lat * scale_blk).reshape(-1, D)

    unit_hat = y_hat @ H
    return (unit_hat * norms_f16).reshape(N_tok, H_heads, D)


# ---------- ablation #5: scalar quantise (Z^4 per-coord, no D4 lattice) ----------
def _scalar_quantise_roundtrip(
    X: torch.Tensor, D: int, q_range: int, H: torch.Tensor,
) -> torch.Tensor:
    """Replace D4 closest-lattice-point with independent Z^4 per-coord
    rounding.  This is the TurboQuant core algorithm — identical scaling
    + Hadamard + qmax, but scalar quantise instead of lattice shaping.
    Measures the "shaping gain" of D4 (Conway-Sloane: +0.37 dB).
    """
    assert X.is_cuda and X.dtype == torch.float32
    N_tok, H_heads, _ = X.shape
    flat = X.reshape(-1, D)
    eps = torch.finfo(torch.float32).eps

    norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
    norms_f16 = norms.to(torch.float16).to(torch.float32)
    unit = flat / norms
    y = unit @ H

    qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
    qmax_f16 = qmax.to(torch.float16).to(torch.float32)
    scale = qmax_f16 / float(q_range)

    # Scalar per-coord rounding (no lattice constraint).
    y_scaled = y / scale
    q = torch.round(y_scaled).clamp(-q_range, q_range)
    y_hat = q * scale

    unit_hat = y_hat @ H
    return (unit_hat * norms_f16).reshape(N_tok, H_heads, D)


# ----- factory registry -----
def make_ablation_codec(
    variant: str, D: int, q_range: int, device: str = "cuda",
):
    """Return a codec_fn conforming to the harness interface.

    The returned fn takes [N_tok, H_heads, D] fp32 CUDA and returns the
    same shape.  Has .bits_per_token_per_head / .label / .channel_id
    attributes set.
    """
    if D % 4 != 0:
        raise ValueError(
            f"D must be divisible by 4 (D4 blocks), got {D}",
        )
    H = _sylvester_hadamard_normalised(D, device)

    # Base bit budget (same as v1.4 full stack at this Q).
    bits_per_block = 4 * math.log2(2 * q_range + 1) - 1
    total_lat_bits = (D // 4) * int(math.ceil(bits_per_block))
    overhead_bits = 32  # fp16 norm + fp16 qmax
    total_bits = total_lat_bits + overhead_bits

    if variant == "v14_full":
        fn = lambda X: _v14_full_roundtrip(X, D, q_range, H)
    elif variant == "no_unit_norm":
        fn = lambda X: _no_unit_norm_roundtrip(X, D, q_range, H)
    elif variant == "no_hadamard":
        fn = lambda X: _no_hadamard_roundtrip(X, D, q_range, H)
    elif variant == "no_per_vec_qmax":
        codec = GlobalQmaxCodec(D=D, q_range=q_range, device=device)
        # Harness must call .calibrate(X_cal) once before the first
        # .roundtrip — wrap that as a stateful closure.
        def fn_global(X, _codec=codec):
            if _codec.global_qmax is None:
                _codec.calibrate(X)
            return _codec.roundtrip(X)
        fn = fn_global
    elif variant == "per_block_qmax":
        fn = lambda X: _per_block_qmax_roundtrip(X, D, q_range, H)
    elif variant == "scalar_quantise":
        fn = lambda X: _scalar_quantise_roundtrip(X, D, q_range, H)
    else:
        raise ValueError(f"Unknown ablation variant: {variant!r}")

    fn.bits_per_token_per_head = total_bits
    fn.label = f"{variant} Q={q_range}"
    fn.channel_id = f"{variant}_Q{q_range}"
    fn.variant = variant
    fn.q_range = q_range
    return fn


ABLATION_VARIANTS = [
    "v14_full",
    "no_unit_norm",
    "no_hadamard",
    "no_per_vec_qmax",
    "per_block_qmax",
    "scalar_quantise",
]
