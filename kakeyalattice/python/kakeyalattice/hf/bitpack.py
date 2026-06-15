r"""GPU-native bit-packing for KakeyaLattice (D4 / E8) and scalar codes.

The published ``KakeyaLatticeQuantizedCache`` stores one **int8** integer per
lattice coordinate, which wastes ~2 bits/coord (a Q=38 D4 value needs ~6.3 bits
but int8 uses 8) and gives a real ~1.94x HBM ratio. This module realises the
codec's *bit-rate* ceiling as real packed bytes by:

  1. mapping each lattice **block** (4 coords for D4, 8 for E8) to a single
     compact integer **code** that exploits the lattice's structural
     redundancy (D4 even-sum; E8 same-parity + sum mod 4), and
  2. packing those fixed-width codes into a contiguous ``uint8`` buffer.

Both steps are lossless: ``decode(encode(x)) == x`` exactly, so reconstruction
(and therefore perplexity) is **identical** to the int8 path — only the byte
count changes.

Bit budgets (per K- or V-vector, head_dim D), matching the codec's
``bits_per_token_per_head``:

    D4 @ Q:  block_bits = ceil(4*log2(2Q+1) - 1);  total = (D/4)*block_bits + 32
    E8 @ Q:  block_bits = ceil(8*log2(2Q+1));      total = (D/8)*block_bits + 32

The +32 is the fp16 ``norm`` + fp16 ``qmax`` overhead (unchanged).

Block-code derivations
----------------------
**D4** = {c in Z^4 : sum(c) even}. Shift u = c + Q in [0, R), R = 2Q+1. The
even-sum constraint forces u3's parity = (u0+u1+u2) mod 2, so we store only
r3 = u3 >> 1:

    code = ((u0*R + u1)*R + u2)*(Q+1) + r3

**E8**, in the doubled-integer representation w = 2x in Z^8 (the form actually
stored as int8 by the codec): E8 <=> all w_i share one parity p AND
sum(w) ≡ 0 (mod 4). Writing w_i = 2*(a_i - Q) + p with a_i in [0, R), the
constraint becomes **sum(a_i) even** — the same parity reduction as D4 on the
last coordinate — plus a single coset bit p:

    code = ( ((a0*R + a1)*R + ... + a6) * (Q+1) + r7 ) * 2 + p      (r7 = a7 >> 1)

Both widths equal the codec's per-block bit budget exactly for the canonical
Q sweep (asserted in tests). Codes are packed with a single fixed width.

High-Q fallback: when a block code would exceed 62 bits (only E8 at Q≳152),
we fall back to per-coordinate fixed-width packing (lossless, slightly above
the ceiling at those very-low-compression points). ``block_packing_fits`` and
the helpers below pick automatically.
"""
from __future__ import annotations

import math

import torch

# int64 codes: keep a safety margin below 63 bits.
_MAX_BLOCK_BITS = 62


# ---------------------------------------------------------------------------
# Fixed-width bit packer (device-native; works on CPU or CUDA tensors).
# ---------------------------------------------------------------------------

def pack_codes(codes: torch.Tensor, width: int) -> torch.Tensor:
    """Pack non-negative integer ``codes`` (each < 2**width) into a contiguous
    ``uint8`` buffer of length ``ceil(N*width/8)``.

    MSB-first within each symbol; symbols concatenated big-endian. Pure torch,
    runs on the codes' device.
    """
    if width < 1 or width > _MAX_BLOCK_BITS:
        raise ValueError(f"width must be in [1, {_MAX_BLOCK_BITS}], got {width}")
    codes = codes.reshape(-1).to(torch.int64)
    if codes.numel() == 0:
        return torch.zeros(0, dtype=torch.uint8, device=codes.device)
    dev = codes.device
    shifts = torch.arange(width - 1, -1, -1, device=dev, dtype=torch.int64)
    bits = (codes.unsqueeze(1) >> shifts) & 1            # [N, width] in {0,1}
    flat = bits.reshape(-1)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=dev)])
    bm = flat.reshape(-1, 8)                              # [M, 8]
    w8 = (1 << torch.arange(7, -1, -1, device=dev, dtype=torch.int64))
    return (bm * w8).sum(dim=1).to(torch.uint8)


def unpack_codes(buf: torch.Tensor, width: int, n: int) -> torch.Tensor:
    """Inverse of :func:`pack_codes`: recover ``n`` integers of ``width`` bits."""
    if width < 1 or width > _MAX_BLOCK_BITS:
        raise ValueError(f"width must be in [1, {_MAX_BLOCK_BITS}], got {width}")
    dev = buf.device
    b = buf.to(torch.int64)
    sh8 = torch.arange(7, -1, -1, device=dev, dtype=torch.int64)
    bits = (b.unsqueeze(1) >> sh8) & 1                    # [M, 8]
    flat = bits.reshape(-1)[: n * width]
    bm = flat.reshape(n, width)
    shifts = torch.arange(width - 1, -1, -1, device=dev, dtype=torch.int64)
    return (bm * (1 << shifts)).sum(dim=1)


# ---------------------------------------------------------------------------
# Bit-budget helpers.
# ---------------------------------------------------------------------------

def d4_block_bits(q_range: int) -> int:
    """Exact width of the D4 block code = ceil(4*log2(2Q+1) - 1)."""
    R = 2 * q_range + 1
    max_code = (R ** 3) * (q_range + 1)           # exclusive upper bound
    return (max_code - 1).bit_length()


def e8_block_bits(q_range: int) -> int:
    """Exact width of the E8 block code = ceil(8*log2(2Q+1))."""
    R = 2 * q_range + 1
    max_code = 2 * (R ** 7) * (q_range + 1)       # exclusive upper bound
    return (max_code - 1).bit_length()


def block_dim_for(variant: str) -> int:
    return 4 if variant.lower() == "d4" else 8


def is_regular_blocks(blocks: torch.Tensor, variant: str, q_range: int) -> torch.Tensor:
    """Bool mask [...]: True where the block is a valid, structurally-encodable
    lattice point. The codec's defensive ``clamp(-Q,Q)`` occasionally pushes a
    point out of the lattice (~1% of blocks); those are flagged irregular and
    stored verbatim in an exception side-channel so packing stays lossless."""
    b = blocks.to(torch.int64)
    if variant.lower() == "d4":
        even = (b.sum(dim=-1) % 2) == 0
        rng = (b.abs() <= q_range).all(dim=-1)
        return even & rng
    # e8 (doubled): all coords share parity AND sum ≡ 0 mod 4 AND |w| <= 2Q
    par = b & 1
    same = (par == par[..., :1]).all(dim=-1)
    s4 = (b.sum(dim=-1) % 4) == 0
    rng = (b.abs() <= 2 * q_range).all(dim=-1)
    return same & s4 & rng


# ---------------------------------------------------------------------------
# D4 block encode / decode.  Input coords are the raw D4 lattice integers
# (even-sum, in [-Q, Q]).
# ---------------------------------------------------------------------------

def d4_encode_blocks(coords: torch.Tensor, q_range: int) -> torch.Tensor:
    """coords: [..., 4] int (even-sum D4 points in [-Q,Q]) -> codes [...]."""
    R = 2 * q_range + 1
    u = (coords.to(torch.int64) + q_range)
    u0, u1, u2, u3 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    r3 = u3 >> 1
    return ((u0 * R + u1) * R + u2) * (q_range + 1) + r3


def d4_decode_blocks(codes: torch.Tensor, q_range: int) -> torch.Tensor:
    """Inverse of :func:`d4_encode_blocks` -> coords [..., 4] int."""
    R = 2 * q_range + 1
    Qp = q_range + 1
    c = codes.to(torch.int64)
    r3 = c % Qp
    rest = c // Qp
    u2 = rest % R; rest = rest // R
    u1 = rest % R
    u0 = rest // R
    p = (u0 + u1 + u2) & 1
    u3 = 2 * r3 + p
    u = torch.stack([u0, u1, u2, u3], dim=-1)
    return u - q_range


# ---------------------------------------------------------------------------
# E8 block encode / decode.  Input coords are the DOUBLED integers w = 2x
# actually stored by the codec (all same parity, sum(w) ≡ 0 mod 4, in [-2Q,2Q]).
# ---------------------------------------------------------------------------

def e8_encode_blocks(coords: torch.Tensor, q_range: int) -> torch.Tensor:
    """coords: [..., 8] int (doubled E8 points w=2x in [-2Q,2Q]) -> codes [...]."""
    R = 2 * q_range + 1
    Qp = q_range + 1
    w = coords.to(torch.int64)
    p = (w[..., 0] & 1)                       # coset bit (all coords share it)
    # w_i = 2*(a_i - Q) + p  =>  a_i = (w_i - p)//2 + Q
    a = ((w - p.unsqueeze(-1)) // 2) + q_range   # [..., 8] in [0, R)
    acc = a[..., 0]
    for i in range(1, 7):
        acc = acc * R + a[..., i]
    r7 = a[..., 7] >> 1
    return (acc * Qp + r7) * 2 + p


def e8_decode_blocks(codes: torch.Tensor, q_range: int) -> torch.Tensor:
    """Inverse of :func:`e8_encode_blocks` -> coords [..., 8] int (doubled)."""
    R = 2 * q_range + 1
    Qp = q_range + 1
    c = codes.to(torch.int64)
    p = c & 1
    rest = c >> 1
    r7 = rest % Qp
    rest = rest // Qp
    a_rev = []
    for _ in range(6, 0, -1):                 # recover a6..a1
        a_rev.append(rest % R)
        rest = rest // R
    a0 = rest                                 # remaining is a0
    a_list = [a0] + list(reversed(a_rev))     # a0..a6
    par = torch.zeros_like(a0)
    for ai in a_list:
        par = par + ai
    a7 = 2 * r7 + (par & 1)
    a_all = a_list + [a7]                      # a0..a7
    a = torch.stack(a_all, dim=-1)
    w = 2 * (a - q_range) + p.unsqueeze(-1)
    return w


# ---------------------------------------------------------------------------
# Real-packed-byte accounting for a stored code tensor.
# ---------------------------------------------------------------------------

def _per_coord_width(variant: str, q_range: int) -> int:
    """Fallback per-coordinate width when a block code exceeds 62 bits."""
    if variant.lower() == "d4":
        return (2 * q_range).bit_length()          # values in [0, 2Q]
    return (4 * q_range).bit_length()              # doubled values in [0, 4Q]


def block_packing_fits(variant: str, q_range: int) -> bool:
    bits = d4_block_bits(q_range) if variant.lower() == "d4" else e8_block_bits(q_range)
    return bits <= _MAX_BLOCK_BITS


def pack_lattice_codes(codes: torch.Tensor, variant: str, q_range: int) -> dict:
    """Bit-pack a stored lattice code tensor ``[..., D]`` losslessly.

    Block coding (hits the bit-rate ceiling) when it fits in 62 bits, else
    per-coordinate fixed-width. Structurally-irregular blocks (rare clamp
    artifacts) are stored verbatim in an exception side-channel.

    Returns a dict with: ``buf`` (uint8), ``width``, ``mode`` ("block"|
    "per_coord"), ``n_blocks``, ``shape`` (original ``codes.shape``),
    ``exc_idx`` (int32 block indices), ``exc_vals`` (int8 raw coords).
    """
    variant = variant.lower()
    bd = block_dim_for(variant)
    D = codes.shape[-1]
    assert D % bd == 0
    shape = tuple(codes.shape)
    blocks = codes.reshape(-1, bd).to(torch.int64)
    n_blocks = blocks.shape[0]
    exc_idx = torch.zeros(0, dtype=torch.int32, device=codes.device)
    exc_vals = torch.zeros(0, bd, dtype=torch.int8, device=codes.device)

    if block_packing_fits(variant, q_range):
        width = d4_block_bits(q_range) if variant == "d4" else e8_block_bits(q_range)
        regular = is_regular_blocks(blocks, variant, q_range)
        enc = d4_encode_blocks if variant == "d4" else e8_encode_blocks
        sym = enc(blocks, q_range)
        # zero-out irregular codes (their values are restored from exceptions)
        sym = torch.where(regular, sym, torch.zeros_like(sym))
        sym = sym.clamp_(0, (1 << width) - 1)
        if (~regular).any():
            ei = torch.nonzero(~regular, as_tuple=False).flatten()
            exc_idx = ei.to(torch.int32)
            exc_vals = blocks[ei].to(torch.int8)
        buf = pack_codes(sym, width)
        mode = "block"
    else:
        # per-coordinate fallback (lossless for any value in range; no exceptions)
        width = _per_coord_width(variant, q_range)
        off = q_range if variant == "d4" else 2 * q_range
        sym = (codes.reshape(-1).to(torch.int64) + off).clamp_(0, (1 << width) - 1)
        buf = pack_codes(sym, width)
        mode = "per_coord"

    return {
        "buf": buf, "width": width, "mode": mode, "n_blocks": n_blocks,
        "shape": shape, "exc_idx": exc_idx, "exc_vals": exc_vals,
        "bd": bd, "q_range": q_range, "variant": variant,
    }


def unpack_lattice_codes(packed: dict) -> torch.Tensor:
    """Inverse of :func:`pack_lattice_codes` -> the original code tensor."""
    variant = packed["variant"]; bd = packed["bd"]; q_range = packed["q_range"]
    width = packed["width"]; n = packed["n_blocks"]; shape = packed["shape"]
    if packed["mode"] == "block":
        sym = unpack_codes(packed["buf"], width, n)
        dec = d4_decode_blocks if variant == "d4" else e8_decode_blocks
        blocks = dec(sym, q_range)
        ei = packed["exc_idx"]
        if ei.numel() > 0:
            blocks[ei.to(torch.int64)] = packed["exc_vals"].to(blocks.dtype)
        return blocks.reshape(shape)
    # per-coord
    n_sym = 1
    for s in shape:
        n_sym *= s
    sym = unpack_codes(packed["buf"], width, n_sym)
    off = q_range if variant == "d4" else 2 * q_range
    return (sym - off).reshape(shape)


def packed_storage_bytes(packed: dict) -> int:
    """Bytes of a packed result: lattice buffer + exception side-channel
    (int32 indices + int8 raw coords)."""
    b = int(packed["buf"].numel())
    b += int(packed["exc_idx"].element_size() * packed["exc_idx"].numel())
    b += int(packed["exc_vals"].element_size() * packed["exc_vals"].numel())
    return b


def real_packed_bytes(
    codes: torch.Tensor,
    norms: torch.Tensor,
    qmax: torch.Tensor,
    variant: str,
    q_range: int,
) -> int:
    """Total real bytes if ``codes`` were bit-packed: packed lattice buffer +
    exception side-channel + fp16 norm + fp16 qmax overhead."""
    packed = pack_lattice_codes(codes, variant, q_range)
    overhead = norms.element_size() * norms.numel() + qmax.element_size() * qmax.numel()
    return packed_storage_bytes(packed) + int(overhead)


def total_bits_per_vector(variant: str, q_range: int, head_dim: int) -> int:
    """Codec bit budget per K/V vector (lattice + 32 overhead)."""
    bd = block_dim_for(variant)
    blk = d4_block_bits(q_range) if variant.lower() == "d4" else e8_block_bits(q_range)
    return (head_dim // bd) * blk + 32


__all__ = [
    "pack_codes", "unpack_codes",
    "d4_block_bits", "e8_block_bits", "block_dim_for",
    "is_regular_blocks",
    "d4_encode_blocks", "d4_decode_blocks",
    "e8_encode_blocks", "e8_decode_blocks",
    "block_packing_fits", "pack_lattice_codes", "unpack_lattice_codes",
    "packed_storage_bytes", "real_packed_bytes", "total_bits_per_vector",
]
