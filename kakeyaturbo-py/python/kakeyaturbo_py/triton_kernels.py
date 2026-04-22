"""Triton GPU kernels for stage 4..=5 of the kakeyaturbo encoder.

Scope: a single fused kernel covering WHT rotation + per-row scale by
`1/||residual||` + Lloyd-Max argmin + outlier detection.  Pack-bits
stays in the Rust helper (byte-exact, <1 % of runtime).

Data-flow in the fused kernel:

    residual[B, wht_len]            ──┐
    sign[wht_len]                    ─┤  x = residual * sign
    H[wht_len, wht_len]              ─┼→ rotated = x @ H       (WHT via fp32 matmul)
    res_norm[B]                      ─┤  scaled = rotated / max(res_norm, EPS)
    centroids[K]                     ─┘  idx = argmin_c cost(scaled - centroids[c])
                                        outlier_mask = |scaled| > T  (per coord)
                                        outlier_scaled_f16 = fp16-round(scaled)
                                           → stored where mask is set

Why matmul for WHT?

    Rust's `wht_inplace` is a Cooley-Tukey butterfly: fp32 adds are
    done in a specific pair-order (stride 1, then stride 2, then …).
    Triton cannot straightforwardly reproduce that pair-order on
    register-resident tensors without store-to-SRAM + XOR-gather
    tricks that would cost more than they save.  A `tl.dot` against
    the Hadamard matrix is numerically ≤ `wht_len·eps ≈ 1.5e-5`
    relative off the butterfly — well under PLAN.md's 1e-5 decoded-
    tensor bar, which rolls in the 1/res_norm scale and 64-way basis
    matmul *after* the WHT.

    If the relative-error bar tightens (e.g. for a future TurboQuant-
    style fused attention kernel), a register-butterfly variant is
    sketched in Appendix A of `M4_PHASE_B_REPORT.md`.

Contract (tested in kakeyaturbo-py/tests/test_triton_phase_b_parity.py):

  * Output `idx: [B, wht_len] uint8` agrees with the PyTorch reference
    `encode_block_torch_stage2` on ≥ 99.9 % of coordinates; disagreements
    are bounded to ≤ 1 Lloyd-Max bucket per coord (single-index flip).
  * After `pack_bits` + decode via Rust, the reconstructed `[n, d]`
    tensor matches the PyTorch reference within PLAN.md's 1e-5 bar
    when aggregated as L2-relative over the full block.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
    _TRITON_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover
    _TRITON_OK = False
    _TRITON_IMPORT_ERROR = _e


# Rust's `f32::EPSILON` — Python float for host-side use.  Inside the
# JIT kernel we hard-code the literal because Triton only allows
# `constexpr`-wrapped globals.
EPS_F32 = 1.1754944e-38

# Huber δ used by LInf metric (matches `kakeyaturbo::distortion::HUBER_DELTA`).
HUBER_DELTA = 0.1

_METRIC_MSE_OR_IP = 0
_METRIC_LINF = 1
_METRIC_IDS = {
    "mse":           _METRIC_MSE_OR_IP,
    "inner_product": _METRIC_MSE_OR_IP,
    "linf":          _METRIC_LINF,
}

# Triton-side `constexpr` aliases — Triton 3.6 requires every global
# referenced inside a @jit function to be wrapped in `tl.constexpr`.
if _TRITON_OK:
    _TL_EPS_F32 = tl.constexpr(1.1754944e-38)
    _TL_HUBER_DELTA = tl.constexpr(0.1)
    _TL_METRIC_LINF = tl.constexpr(_METRIC_LINF)


def _require_triton():
    if not _TRITON_OK:
        raise RuntimeError(
            "Triton is not importable; the Phase B GPU path requires a "
            "CUDA torch + triton install.  Run on an H200 / H100 / A100 "
            f"instance.  Original import error: {_TRITON_IMPORT_ERROR!r}"
        )


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


if _TRITON_OK:

    @triton.jit
    def _wht_scale_quantize_outlier_kernel(
        residual_ptr,
        H_ptr,
        sign_ptr,
        centroids_ptr,
        res_norm_ptr,
        scaled_ptr,             # [B, wht_len] fp32, written for outlier pass
        out_idx_ptr,            # [B, wht_len] uint8, written
        stride_res_b,
        stride_H_row,
        stride_scaled_b,
        stride_out_b,
        wht_len: tl.constexpr,
        K: tl.constexpr,
        metric_id: tl.constexpr,
    ):
        """Program grid: (B,) — one program per row.

        Stage 4b: `rotated = (residual * sign) @ H` where H is the
            Sylvester Hadamard matrix of side `wht_len`.
        Stage 4c: `scaled = rotated * (1 / max(res_norm, EPS))`
        Stage 5b: `idx = argmin_c cost(scaled_j - centroids_c)`

        The kernel also writes `scaled` to memory so a follow-up
        Triton pass can extract outliers (stage 5a) without recomputing
        the rotation.
        """
        row = tl.program_id(0)

        coord = tl.arange(0, wht_len)

        # Load residual * sign.
        residual = tl.load(residual_ptr + row * stride_res_b + coord)
        sign = tl.load(sign_ptr + coord)
        x = (residual * sign).to(tl.float32)

        # WHT via matmul: rotated[j] = sum_i x[i] * H[i, j]
        # We load H one column at a time and do a reduction.  For
        # wht_len ≤ 128 the whole H fits in SRAM (128*128*4 = 64 KB).
        h_tile = tl.load(
            H_ptr + tl.arange(0, wht_len)[:, None] * stride_H_row
            + tl.arange(0, wht_len)[None, :]
        )   # [wht_len, wht_len] fp32, entries ±1.0

        # Fused matmul: rotated[j] = sum_i x[i] * H[i, j]
        # The Sylvester Hadamard matrix is symmetric, so H[i,j] == H[j,i].
        rotated = tl.sum(x[:, None] * h_tile, axis=0)          # [wht_len]

        # Stage 4c: scale.
        res_norm = tl.load(res_norm_ptr + row)
        scale = tl.where(res_norm > _TL_EPS_F32, 1.0 / res_norm, 1.0)
        scaled = rotated * scale                                # [wht_len]

        # Store scaled for the outlier pass.
        tl.store(scaled_ptr + row * stride_scaled_b + coord, scaled)

        # Stage 5b: Lloyd-Max argmin.
        ci = tl.arange(0, K)
        centroids = tl.load(centroids_ptr + ci)                # [K]
        diff = scaled[:, None] - centroids[None, :]            # [wht_len, K]

        if metric_id == _TL_METRIC_LINF:
            abs_e = tl.abs(diff)
            cost = tl.where(
                abs_e < _TL_HUBER_DELTA,
                (abs_e * abs_e) / (2.0 * _TL_HUBER_DELTA),
                abs_e - (_TL_HUBER_DELTA / 2.0),
            )
        else:
            cost = diff * diff

        idx = tl.argmin(cost, axis=1).to(tl.uint8)
        tl.store(out_idx_ptr + row * stride_out_b + coord, idx)


def fused_wht_scale_quantize(
    residual: torch.Tensor,
    sign: torch.Tensor,
    centroids: torch.Tensor,
    res_norm: torch.Tensor,
    *,
    metric: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-fused encode stage 4b + 4c + 5b.

    Inputs (all CUDA, C-contig, fp32 except where noted):
        residual:  [B, wht_len] — padded residuals, zero beyond d_eff.
        sign:      [wht_len]    — Rust's ±1 sign pattern (load via
                                  `kakeyaturbo_py.wht_sign_pattern`).
        centroids: [K]          — Lloyd-Max centroids, ascending.
        res_norm:  [B]          — ||residual|| per row (fp32).
        metric:    str          — {'mse', 'inner_product', 'linf'}.

    Returns:
        idx:    [B, wht_len] uint8  — Lloyd-Max indices.
        scaled: [B, wht_len] fp32   — the scaled residual, so the
                                      caller can run a cheap outlier
                                      extraction without redoing the
                                      WHT.
    """
    _require_triton()
    if residual.device.type != "cuda":
        raise RuntimeError(
            "fused_wht_scale_quantize requires CUDA tensors; got "
            f"residual.device = {residual.device}"
        )
    for name, t in [("residual", residual), ("sign", sign),
                    ("centroids", centroids), ("res_norm", res_norm)]:
        if t.dtype != torch.float32:
            raise TypeError(f"{name} must be float32, got {t.dtype}")
        if t.device != residual.device:
            raise RuntimeError(
                f"{name} on {t.device}, expected {residual.device}"
            )

    B, wht_len = residual.shape
    if (wht_len & (wht_len - 1)) != 0 or wht_len < 1:
        raise ValueError(f"wht_len must be a positive power of two, got {wht_len}")
    if wht_len > 128:
        raise ValueError(
            f"wht_len={wht_len} exceeds the 128 upper bound of the "
            "Phase B kernel (the Hadamard tile is materialised in SRAM, "
            "128×128×4 = 64 KB is the biggest H200 SM can fit without "
            "spill).  d_eff ≤ 64 ⇒ wht_len ≤ 128 covers every production cell."
        )
    if sign.shape != (wht_len,):
        raise ValueError(f"sign shape {tuple(sign.shape)} != ({wht_len},)")
    if res_norm.shape != (B,):
        raise ValueError(f"res_norm shape {tuple(res_norm.shape)} != ({B},)")
    K = int(centroids.numel())
    if K not in (2, 4, 8, 16):
        raise ValueError(f"K must be 2/4/8/16 (= 2^bit_width), got {K}")
    metric_id = _METRIC_IDS.get(metric)
    if metric_id is None:
        raise ValueError(f"unknown metric {metric!r}")

    residual_c = residual.contiguous()
    sign_c = sign.contiguous()
    centroids_c = centroids.contiguous()
    res_norm_c = res_norm.contiguous()

    # Build the Sylvester Hadamard matrix once per call.  Caching it
    # across calls is a Phase C optimisation (Triton's kernel cache
    # can lookup on constexpr'd wht_len, so compile cost is amortised;
    # matrix build is 128×128 fp32 = 64 KB on the host, trivial).
    H_np = _sylvester_hadamard(wht_len)
    H = torch.from_numpy(H_np).to(residual.device, torch.float32).contiguous()

    scaled = torch.empty((B, wht_len), dtype=torch.float32,
                         device=residual.device)
    out_idx = torch.empty((B, wht_len), dtype=torch.uint8,
                          device=residual.device)

    grid = (B,)
    _wht_scale_quantize_outlier_kernel[grid](
        residual_c, H, sign_c, centroids_c, res_norm_c,
        scaled, out_idx,
        residual_c.stride(0), H.stride(0),
        scaled.stride(0), out_idx.stride(0),
        wht_len=wht_len, K=K, metric_id=metric_id,
    )
    return out_idx, scaled


# ---------------------------------------------------------------------------
# DECODE kernel (M5) — inverse of fused_wht_scale_quantize, plus the
# downstream coeff = t·center + residual + basis reconstruction.
# ---------------------------------------------------------------------------


if _TRITON_OK:

    @triton.jit
    def _inv_wht_rescale_kernel(
        q_vals_ptr,       # [B, wht_len] fp32 — dequantized scaled residual
                          # (Lloyd-Max dequant + outlier override done on the
                          # torch side before launch)
        H_ptr,            # [wht_len, wht_len] fp32 ±1 (Sylvester Hadamard)
        sign_ptr,         # [wht_len] fp32 ±1 (Rust's wht sign pattern)
        norm_ptr,         # [B] fp32 — the fp16-stored norm field (inv_scale)
        out_ptr,          # [B, wht_len] fp32 — residual reconstructed
        stride_q_b,
        stride_H_row,
        stride_out_b,
        wht_len: tl.constexpr,
    ):
        """Inverse of stages 4b + 4c.  One program per row.

        Stage decode-4c^-1: `q_scaled = q_vals * inv_scale`
        Stage decode-4b^-1: `x = D · H · q_scaled / wht_len`
            where `D = diag(sign_pattern)` and `H` is the Sylvester
            Hadamard matrix.  Since `H = Hᵀ`, the forward `y = H·D·x`
            and inverse `x = D·H·y/N` share the same matrix — only
            the factor-of-1/N and sign-pattern order differ.

        Emits the reconstructed residual (before coeff assembly) so
        the caller can finish with a cuBLAS `coeff @ basis + mean`.
        """
        row = tl.program_id(0)

        coord = tl.arange(0, wht_len)

        # Load the dequantised scaled residual and the inv-scale (norm).
        q_scaled = tl.load(q_vals_ptr + row * stride_q_b + coord).to(tl.float32)
        norm = tl.load(norm_ptr + row)
        y = q_scaled * norm                                       # undo 1/res_norm

        # Inverse WHT via Hadamard matmul: x_prime[j] = sum_i y[i] * H[i, j]
        # (H is symmetric so transpose is a no-op).
        h_tile = tl.load(
            H_ptr + tl.arange(0, wht_len)[:, None] * stride_H_row
            + tl.arange(0, wht_len)[None, :]
        )
        x_prime = tl.sum(y[:, None] * h_tile, axis=0)             # [wht_len]

        # Multiply by sign pattern and 1/wht_len.  `wht_len` is a
        # constexpr Python int; cast by converting to a Triton tensor
        # before the reciprocal.
        sign = tl.load(sign_ptr + coord)
        inv_n = 1.0 / tl.full([], wht_len, tl.float32)
        x = x_prime * sign * inv_n

        tl.store(out_ptr + row * stride_out_b + coord, x)


def fused_inverse_wht_rescale(
    q_vals: torch.Tensor,
    sign: torch.Tensor,
    norm: torch.Tensor,
) -> torch.Tensor:
    """Triton-fused decode stages 4b^-1 + 4c^-1.

    Inputs (CUDA, fp32, C-contig):
        q_vals: [B, wht_len]   — dequantised + outlier-overridden scaled residual
        sign:   [wht_len]      — Rust WHT sign pattern
        norm:   [B]            — fp16-stored inv_scale field

    Returns:
        x: [B, wht_len] fp32   — reconstructed residual (pre-padding)
    """
    _require_triton()
    if q_vals.device.type != "cuda":
        raise RuntimeError(
            f"fused_inverse_wht_rescale requires CUDA; got {q_vals.device}"
        )
    for name, t in [("q_vals", q_vals), ("sign", sign), ("norm", norm)]:
        if t.dtype != torch.float32:
            raise TypeError(f"{name} must be float32, got {t.dtype}")
        if t.device != q_vals.device:
            raise RuntimeError(f"{name} on {t.device}, expected {q_vals.device}")

    B, wht_len = q_vals.shape
    if (wht_len & (wht_len - 1)) != 0 or wht_len < 1:
        raise ValueError(f"wht_len must be a positive power of two, got {wht_len}")
    if wht_len > 128:
        raise ValueError(
            f"wht_len={wht_len} exceeds the 128 upper bound (Hadamard "
            "tile must fit in SRAM)"
        )
    if sign.shape != (wht_len,):
        raise ValueError(f"sign shape {tuple(sign.shape)} != ({wht_len},)")
    if norm.shape != (B,):
        raise ValueError(f"norm shape {tuple(norm.shape)} != ({B},)")

    q_c = q_vals.contiguous()
    sign_c = sign.contiguous()
    norm_c = norm.contiguous()

    H_np = _sylvester_hadamard(wht_len)
    H = torch.from_numpy(H_np).to(q_vals.device, torch.float32).contiguous()

    out = torch.empty((B, wht_len), dtype=torch.float32, device=q_vals.device)

    grid = (B,)
    _inv_wht_rescale_kernel[grid](
        q_c, H, sign_c, norm_c, out,
        q_c.stride(0), H.stride(0), out.stride(0),
        wht_len=wht_len,
    )
    return out


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------


def _sylvester_hadamard(n: int) -> np.ndarray:
    """Sylvester-ordered Hadamard matrix, entries ±1 as fp32.

    The matrix `H_n` of side `n = 2^k` built recursively as
      H_1 = [1],  H_2n = [[H_n, H_n], [H_n, -H_n]].
    """
    assert n > 0 and (n & (n - 1)) == 0
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def is_available() -> bool:
    """True iff Triton is importable AND CUDA is available."""
    if not _TRITON_OK:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# M5 end-to-end decode (inverse of encode_block_triton_stage2)
# ---------------------------------------------------------------------------


def decode_block_triton_from_parts(
    parts: dict,
    *,
    custom_centroids: Optional[np.ndarray] = None,
    device: str = "cuda",
) -> np.ndarray:
    """Triton-accelerated dual of `decode_block_from_parts`.

    Input:
        parts: dict with keys {mean, basis, centers, d, d_eff, k,
               rotation_seed, wht_len, bit_width, metric, seg_id, t,
               norm, residual_packed, outlier_idx, outlier_val,
               outlier_count, max_outliers}.  Format is exactly what
               `encode_block_codes` / `encode_block_triton_stage2`
               emit, so callers can round-trip without re-serialising.

    Output:
        x_hat: [n, d] float32 numpy array — decoded vectors, same
               semantics as `kakeyaturbo::codec::decode_block_with_centroids`.

    Contract (tested in test_triton_decode_parity.py):
        Byte-identical to `decode_block_from_parts(parts)` (Rust
        decode) when `metric` and codes match, within the PLAN.md
        1e-3 L2-rel or 1 %-row-flip bar.  Identical in shape and
        meaning to `decode_block_torch_from_parts(parts)`.
    """
    _require_triton()
    from . import _core
    from . import reference_torch as rt

    skel = rt._skeleton_from_dict(parts, device=device)
    cb = rt._codes_from_dict(parts, device=device)
    n = cb.seg_id.shape[0]

    # --- Stage 5c^-1: unpack ---
    # Rust's pack_bits is byte-exact and cheap; keep it on the Rust side
    # so the byte stream → index stream is reference-correct.  We then
    # move to CUDA once for the rest of the pipeline.
    packed_np = cb.residual_packed.detach().cpu().numpy()
    idx_np = np.empty((n, skel.wht_len), dtype=np.uint8)
    for i in range(n):
        idx_np[i, :] = np.asarray(
            _core.unpack_bits(packed_np[i], skel.bit_width, skel.wht_len)
        )
    indices = torch.from_numpy(idx_np).to(device)

    # --- Stage 5b^-1: dequantise against the centroid table ---
    if custom_centroids is None:
        centroids_np = np.asarray(_core.centroids_gaussian(skel.bit_width))
    else:
        centroids_np = np.asarray(custom_centroids, dtype=np.float32)
        if centroids_np.size != (1 << skel.bit_width):
            raise ValueError(
                f"custom centroids size {centroids_np.size} != "
                f"2^bit_width = {1 << skel.bit_width}"
            )
    centroids = torch.from_numpy(centroids_np).to(device, torch.float32)
    q_vals = centroids[indices.long()]                            # [n, wht_len] fp32

    # --- Stage 5a^-1: outlier override ---
    # Before the inverse WHT / rescale, replace the dequantised value
    # at each outlier coord with the f16-stored outlier value.  Rust
    # does this as a sparse loop; we reproduce it with torch scatter.
    if cb.max_outliers > 0:
        # Gather-scatter in torch: for each row i, for each j < outlier_count[i],
        # set q_vals[i, outlier_idx[i, j]] = outlier_val[i, j].
        # We do this vectorised by masking inactive slots (j >= count[i]).
        oi = cb.outlier_idx.long()                                 # [n, max_outliers]
        ov = cb.outlier_val                                        # [n, max_outliers]
        oc = cb.outlier_count.long()                               # [n]
        row_idx = torch.arange(n, device=device).unsqueeze(1).expand_as(oi)
        active = (torch.arange(cb.max_outliers, device=device).unsqueeze(0)
                  < oc.unsqueeze(1))                               # [n, max_outliers]
        active_flat = active.reshape(-1)
        if bool(active_flat.any().item()):
            flat_row = row_idx.reshape(-1)[active_flat]
            flat_col = oi.reshape(-1)[active_flat]
            flat_val = ov.reshape(-1)[active_flat]
            q_vals[flat_row, flat_col] = flat_val

    # --- Stages 4c^-1 + 4b^-1: Triton kernel (scale by norm, inverse WHT) ---
    sign_np = np.asarray(_core.wht_sign_pattern(int(skel.rotation_seed),
                                                int(skel.wht_len))).reshape(-1)
    sign = torch.from_numpy(sign_np).to(device, torch.float32)

    # Determine inv_scale = norm field when metric is absorbed (MSE / LInf),
    # else 1.0 (Inner-product's residual is stored un-scaled).
    metric = cb.metric
    if metric in ("mse", "linf"):
        inv_scale = cb.norm.to(torch.float32)
    else:                                                          # inner_product
        inv_scale = torch.ones_like(cb.norm, dtype=torch.float32)

    x_full = fused_inverse_wht_rescale(q_vals, sign, inv_scale)    # [n, wht_len]

    # --- Stage 4a^-1: trim to d_eff ---
    residual_rec = x_full[:, :skel.d_eff]                          # [n, d_eff]

    # --- Stage 3^-1: coeff = t * center[seg] + residual ---
    chosen = skel.centers[cb.seg_id]                               # [n, d_eff]
    coeff = cb.t.unsqueeze(1) * chosen + residual_rec

    # --- Stage 2^-1: unproject = coeff @ basis + mean ---
    x_hat = coeff @ skel.basis + skel.mean.unsqueeze(0)            # [n, d]
    return x_hat.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# M5 partial-block bf16 passthrough path.
#
# Per PLAN.md §"Consequence: the paged cache has two slot types per layer:
#   * Sealed codec blocks   (compressed, full codec pipeline applied)
#   * Trailing partial block (bf16, < block_size_codec tokens)"
#
# This function is the partial-block "decoder": it takes a bf16 staging
# buffer holding `m < block_size_codec` tokens and returns the fp32
# view that downstream attention would have seen had the block been
# sealed.  In the actual vLLM backend (M6), attention will simply fuse
# the sealed-block decoder's output with this partial-block read.
# ---------------------------------------------------------------------------


def decode_partial_block_bf16(
    staging_bf16: torch.Tensor,
) -> torch.Tensor:
    """Read a bf16 partial-block staging buffer back to fp32.

    Input:
        staging_bf16: [m, d] torch.bfloat16, CUDA.  `m` may be any
                      value in `1..block_size_codec`.

    Output:
        x: [m, d] torch.float32, same device.

    Contract: this is an identity dtype-cast, byte-identical to what
    vLLM's FlashAttention would read in the bf16 KV-cache case.  It
    exists as a named function so M6's backend can swap sealed /
    partial dispatch without conditional logic leaking into the
    kernel layer.
    """
    if staging_bf16.dim() != 2:
        raise ValueError(f"expected 2-D tensor, got {tuple(staging_bf16.shape)}")
    if staging_bf16.dtype != torch.bfloat16:
        raise TypeError(
            f"expected bfloat16 staging buffer, got {staging_bf16.dtype}"
        )
    return staging_bf16.to(torch.float32)


# ---------------------------------------------------------------------------
# End-to-end encoder that plugs the Triton kernel into the PyTorch pipeline.
# ---------------------------------------------------------------------------


def encode_block_triton_stage2(
    X_np: np.ndarray,
    skeleton_parts: dict,
    *,
    custom_centroids: Optional[np.ndarray] = None,
    outlier_threshold: Optional[float] = None,
    device: str = "cuda",
) -> dict:
    """Phase B end-to-end: same contract as
    `kakeyaturbo_py.reference_torch.encode_block_torch_stage2`, but
    stages 4b/4c/5b run in the Triton kernel above.  Stages 2/3/4a/5a/5c
    remain in torch-on-CUDA (the first two are matmuls which cuBLAS
    handles at peak; 4a is elementwise; 5a is sparse; 5c is bit-pack
    which delegates to the Rust helper for byte-exactness).
    """
    _require_triton()
    from . import _core
    from . import reference_torch as rt

    X = torch.from_numpy(np.ascontiguousarray(X_np, dtype=np.float32)).to(device)
    skel = rt._skeleton_from_dict(skeleton_parts, device=device)
    metric = str(skeleton_parts["metric"])
    if skel.bit_width not in (1, 2, 3, 4):
        raise ValueError(f"unsupported bit_width {skel.bit_width}")
    n, d = X.shape
    if d != skel.d:
        raise ValueError(f"input dim {d} != skeleton d {skel.d}")

    # Stage 2: project.
    coeff = (X - skel.mean.unsqueeze(0)) @ skel.basis.transpose(0, 1)

    # Stage 3: K-means assign.
    seg_id, t = rt._assign_and_project(coeff, skel)

    # Stage 4a: residual.
    residual = rt._residual(coeff, seg_id, t, skel)
    # Pad to wht_len.
    residual_padded = torch.zeros((n, skel.wht_len), dtype=torch.float32,
                                  device=device)
    residual_padded[:, :skel.d_eff] = residual

    # Sign pattern from Rust (byte-exact).
    sign_np = np.asarray(_core.wht_sign_pattern(int(skel.rotation_seed),
                                                int(skel.wht_len))).reshape(-1)
    sign = torch.from_numpy(sign_np).to(device, torch.float32)

    # Centroids table.
    if custom_centroids is None:
        centroids_np = np.asarray(_core.centroids_gaussian(skel.bit_width))
    else:
        centroids_np = np.asarray(custom_centroids, dtype=np.float32)
        if centroids_np.size != (1 << skel.bit_width):
            raise ValueError(
                f"centroids size {centroids_np.size} != "
                f"2^bit_width = {1 << skel.bit_width}"
            )
    centroids = torch.from_numpy(centroids_np).to(device, torch.float32)

    # Residual L2 norms.
    res_norm = torch.linalg.norm(residual, dim=1)

    # --- Triton kernel: stages 4b + 4c + 5b ---
    q, scaled = fused_wht_scale_quantize(
        residual_padded, sign, centroids, res_norm, metric=metric,
    )                                                        # q: [n, wht_len] u8

    # Stage 5a: outliers (Python-side; sparse ragged output).
    max_outliers = 0
    outlier_idx = np.zeros((n, 1), dtype=np.uint16)
    outlier_val = np.zeros((n, 1), dtype=np.float32)
    outlier_count = np.zeros(n, dtype=np.uint32)
    if outlier_threshold is not None:
        thr = float(outlier_threshold)
        abs_scaled = scaled.abs()
        mask = (abs_scaled > thr)
        counts = mask.sum(dim=1).detach().cpu().numpy().astype(np.uint32)
        max_outliers = int(counts.max()) if counts.size > 0 else 0
        if max_outliers > 0:
            outlier_idx = np.zeros((n, max_outliers), dtype=np.uint16)
            outlier_val = np.zeros((n, max_outliers), dtype=np.float32)
            mask_np = mask.detach().cpu().numpy()
            scaled_np = scaled.detach().cpu().numpy()
            for i in range(n):
                ixs = np.where(mask_np[i])[0].astype(np.uint16)
                outlier_idx[i, :ixs.size] = ixs
                vals32 = scaled_np[i, ixs]
                outlier_val[i, :ixs.size] = vals32.astype(np.float16).astype(np.float32)
        outlier_count = counts

    # Stage 5c: pack via Rust.
    q_np = q.detach().cpu().numpy()
    pbytes = (skel.wht_len * skel.bit_width + 7) // 8
    packed = np.zeros((n, pbytes), dtype=np.uint8)
    for i in range(n):
        packed[i, :] = np.asarray(_core.pack_bits(q_np[i], skel.bit_width))

    # Stored `t` and `norm` go through fp16.  Rust computes:
    #   let scale = if res_norm > eps { 1.0 / res_norm } else { 1.0 };
    #   let inv_scale = 1.0 / scale.max(eps);
    # We mirror this exactly (it's *not* the same as
    # `1.0 / clamp(res_norm, min=eps)`).
    t_f16 = t.to(torch.float16).to(torch.float32)
    if metric in ("mse", "linf"):
        scale = torch.where(
            res_norm > EPS_F32,
            1.0 / res_norm,
            torch.ones_like(res_norm),
        )
        inv_scale = 1.0 / torch.clamp(scale, min=EPS_F32)
        norm_field = inv_scale.to(torch.float16).to(torch.float32)
    else:                                                     # inner_product
        x_norm = torch.linalg.norm(X, dim=1)
        norm_field = x_norm.to(torch.float16).to(torch.float32)

    return {
        # Skeleton fields, unchanged.
        "mean":            np.asarray(skeleton_parts["mean"]),
        "basis":           np.asarray(skeleton_parts["basis"]),
        "centers":         np.asarray(skeleton_parts["centers"]),
        "d":               skel.d,
        "d_eff":           skel.d_eff,
        "k":               skel.k,
        "rotation_seed":   skel.rotation_seed,
        "wht_len":         skel.wht_len,
        "bit_width":       skel.bit_width,
        "metric":          metric,
        # Codes.
        "seg_id":          seg_id.detach().cpu().numpy().astype(np.uint32),
        "t":               t_f16.detach().cpu().numpy().astype(np.float32),
        "norm":            norm_field.detach().cpu().numpy().astype(np.float32),
        "residual_packed": packed,
        "outlier_idx":     outlier_idx,
        "outlier_val":     outlier_val,
        "outlier_count":   outlier_count,
        "max_outliers":    max_outliers,
    }
