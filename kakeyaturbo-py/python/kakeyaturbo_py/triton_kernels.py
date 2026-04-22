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
