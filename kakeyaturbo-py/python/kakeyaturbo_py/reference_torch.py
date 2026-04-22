"""PyTorch reference for stages 2..=5 of the kakeyaturbo codec.

**Scope**: given a skeleton produced by the Rust reference (`PCA +
K-means`, stage 1), this module reproduces the per-vector encode/decode
pipeline (stages 2..=5: coefficient projection, K-means assignment,
residual rotation + scaling, Lloyd-Max quantisation, bit packing) in
PyTorch, bit-identical to Rust on its decoded output.

That's the correct oracle for M4: stages 2..=5 are the ones that move
into Triton, and they are the hot loop per-token; stage 1 is a
per-block operation that runs on every prefill block regardless of
backend (CPU torch or CUDA torch is fine — `torch.linalg.eigh`
produces decoded-equivalent results up to eigenvector-column-sign
ambiguity, which would defeat bit-exact skeleton comparison but is
irrelevant *given a fixed skeleton*).

Contract of `encode_block_torch_stage2`:
    Input:
      X           : torch.Tensor[N, D] float32  (row-major, C-contig)
      skeleton    : dict from kakeyaturbo_py.encode_block_codes output
                    (so all the per-block numerics are Rust's)
    Output:
      codes_dict  : same keyed layout as what `encode_block_codes`
                    returns for the codes fields.  If the PyTorch
                    computation is byte-exact to Rust's stage 2..=5,
                    then `encode_block_codes(X, ...)` ==
                    `{**skeleton_fields, **encode_block_torch_stage2(X,
                    skeleton_dict, ...)}` (subject to rounding through
                    fp16 at the `t` and `norm` fields).

Contract of `decode_block_torch_from_parts`:
    Same dict shape as `decode_block_from_parts` input; produces
    byte-identical decoded tensor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from . import _core


# ---------------------------------------------------------------------------
# Dataclasses mirroring the pyo3 dict layout.  We provide thin
# wrappers so callers don't need to shuffle keys around, but the dict
# form is the canonical wire format.
# ---------------------------------------------------------------------------


@dataclass
class Skeleton:
    mean: torch.Tensor              # [d]            fp32
    basis: torch.Tensor             # [d_eff, d]     fp32
    centers: torch.Tensor           # [k, d_eff]     fp32
    d: int
    d_eff: int
    k: int
    rotation_seed: int
    wht_len: int
    bit_width: int


@dataclass
class CodeBatch:
    seg_id: torch.Tensor            # [n] uint32 (stored as int64 for torch ops)
    t: torch.Tensor                 # [n] float32 (already fp16-rounded)
    norm: torch.Tensor              # [n] float32 (already fp16-rounded)
    residual_packed: torch.Tensor   # [n, pbytes] uint8
    outlier_idx: torch.Tensor       # [n, max_outliers] uint16
    outlier_val: torch.Tensor       # [n, max_outliers] float32
    outlier_count: torch.Tensor     # [n] uint32
    max_outliers: int
    metric: str


# ---------------------------------------------------------------------------
# Utility: dict <-> dataclass.
# ---------------------------------------------------------------------------


def _skeleton_from_dict(parts: dict, device: str | torch.device = "cpu"
                        ) -> Skeleton:
    return Skeleton(
        mean=torch.from_numpy(np.asarray(parts["mean"])).to(device, torch.float32).contiguous(),
        basis=torch.from_numpy(np.asarray(parts["basis"])).to(device, torch.float32).contiguous(),
        centers=torch.from_numpy(np.asarray(parts["centers"])).to(device, torch.float32).contiguous(),
        d=int(parts["d"]),
        d_eff=int(parts["d_eff"]),
        k=int(parts["k"]),
        rotation_seed=int(parts["rotation_seed"]),
        wht_len=int(parts["wht_len"]),
        bit_width=int(parts["bit_width"]),
    )


def _codes_from_dict(parts: dict, device: str | torch.device = "cpu"
                     ) -> CodeBatch:
    return CodeBatch(
        seg_id=torch.from_numpy(np.asarray(parts["seg_id"])).to(device, torch.int64),
        t=torch.from_numpy(np.asarray(parts["t"])).to(device, torch.float32),
        norm=torch.from_numpy(np.asarray(parts["norm"])).to(device, torch.float32),
        residual_packed=torch.from_numpy(np.asarray(parts["residual_packed"])).to(device, torch.uint8),
        outlier_idx=torch.from_numpy(np.asarray(parts["outlier_idx"])).to(device, torch.int32),
        outlier_val=torch.from_numpy(np.asarray(parts["outlier_val"])).to(device, torch.float32),
        outlier_count=torch.from_numpy(np.asarray(parts["outlier_count"])).to(device, torch.int32),
        max_outliers=int(parts["max_outliers"]),
        metric=str(parts["metric"]),
    )


# ---------------------------------------------------------------------------
# Low-level math (mirrors kakeyaturbo/src/codec.rs stages 2..=5).
# ---------------------------------------------------------------------------


def _fp16_through(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 through fp16 and back — the bit-pattern Rust stores
    when SkeletonDtype::Fp16 (the only skeleton dtype in Phase A.1)."""
    return x.to(torch.float16).to(torch.float32)


def _project(X: torch.Tensor, skel: Skeleton) -> torch.Tensor:
    """coeff[i] = (X[i] - μ) @ basisᵀ.  No fp16 rounding here — the
    basis already came from Rust as fp16-round-tripped fp32."""
    return (X - skel.mean.unsqueeze(0)) @ skel.basis.transpose(0, 1)


def _assign_and_project(coeff: torch.Tensor, skel: Skeleton
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirrors kakeyaturbo::kmeans::assign_and_project.

    seg_id[i] = argmax_c |<coeff[i], centers[c]>|
    t[i]      = <coeff[i], centers[seg_id[i]]>
    Zero-norm rows: (seg_id=0, t=0).

    Tie-breaking: Rust's loop uses `abs_cos > best_abs`, i.e. strict-
    greater; the first centre that achieves the max wins.  torch.argmax
    has the same semantics (returns the first index of the max).  On
    ties the two sides agree as long as we iterate centres in the same
    order, which we do (centres[c] indexed by c = 0..k).
    """
    n, d_eff = coeff.shape
    cos = coeff @ skel.centers.transpose(0, 1)                  # [n, k]
    abs_cos = cos.abs()
    seg_id = torch.argmax(abs_cos, dim=1)                       # [n]
    t = cos[torch.arange(n, device=coeff.device), seg_id]

    coeff_norm_sq = (coeff * coeff).sum(dim=1)
    zero = coeff_norm_sq <= torch.finfo(torch.float32).eps
    seg_id = torch.where(zero, torch.zeros_like(seg_id), seg_id)
    t = torch.where(zero, torch.zeros_like(t), t)
    return seg_id, t


def _residual(coeff: torch.Tensor, seg_id: torch.Tensor,
              t: torch.Tensor, skel: Skeleton) -> torch.Tensor:
    """residual[i] = coeff[i] - t[i] * centers[seg_id[i]]

    Plus Rust's "if coeff == 0 then residual = 0" guard.
    """
    chosen = skel.centers[seg_id]                               # [n, d_eff]
    residual = coeff - t.unsqueeze(1) * chosen
    coeff_zero = coeff.abs().max(dim=1).values <= torch.finfo(torch.float32).eps
    residual = torch.where(coeff_zero.unsqueeze(1),
                           torch.zeros_like(residual),
                           residual)
    return residual


def _rotate_rows_via_rust(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Apply Rust's `wht::rotate` row-wise via the pyo3 helper.

    Matches the `let rotated = rotate(&res_padded, params.rotation_seed)`
    line in Rust's encode_block, bit-exactly."""
    assert x.dim() == 2 and x.dtype == torch.float32
    x_np = np.ascontiguousarray(x.detach().cpu().numpy())
    y_np = _core.rotate_rows(x_np, int(seed))
    return torch.from_numpy(np.asarray(y_np)).to(x.device)


def _inverse_rotate_rows_via_rust(y: torch.Tensor, seed: int) -> torch.Tensor:
    assert y.dim() == 2 and y.dtype == torch.float32
    y_np = np.ascontiguousarray(y.detach().cpu().numpy())
    x_np = _core.inverse_rotate_rows(y_np, int(seed))
    return torch.from_numpy(np.asarray(x_np)).to(y.device)


def _quantize_rows(scaled: torch.Tensor, centroids: torch.Tensor,
                   metric: str) -> torch.Tensor:
    """Nearest-centroid quantiser.  Returns uint8 indices.

    Per-metric `d(x, c)`:
      mse           : (x − c)²
      inner_product : (x − c)²        (per-coord; norm tracked separately)
      linf          : Huber with δ = 0.1
    """
    n, wht_len = scaled.shape
    k = centroids.numel()
    if metric in ("mse", "inner_product"):
        # argmin over (x - c)² ⇔ argmin over |x - c|; use |x - c| for
        # numerical stability at high-magnitude coordinates.
        dist = (scaled.unsqueeze(-1) - centroids.view(1, 1, k)).abs()
    elif metric == "linf":
        delta = 0.1
        e = (scaled.unsqueeze(-1) - centroids.view(1, 1, k)).abs()
        # Huberised cost.
        dist = torch.where(
            e < delta,
            (e * e) / (2.0 * delta),
            e - (delta / 2.0),
        )
    else:
        raise ValueError(f"unknown metric {metric!r}")

    idx = torch.argmin(dist, dim=-1)                             # [n, wht_len]
    return idx.to(torch.uint8)


def _pack_bits_rows(idx: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Row-wise pack_bits via the Rust helper (byte-exact).

    `idx` has shape [n, wht_len]; returns [n, pbytes] uint8 with
    pbytes = ceil(wht_len * bit_width / 8).
    """
    assert idx.dtype == torch.uint8
    n, wht_len = idx.shape
    pbytes = (wht_len * bit_width + 7) // 8
    idx_np = idx.detach().cpu().numpy()
    out = np.empty((n, pbytes), dtype=np.uint8)
    for i in range(n):
        out[i] = np.asarray(_core.pack_bits(idx_np[i], bit_width))
    return torch.from_numpy(out).to(idx.device)


def _unpack_bits_rows(packed: torch.Tensor, bit_width: int,
                      wht_len: int) -> torch.Tensor:
    assert packed.dtype == torch.uint8
    n, _pbytes = packed.shape
    np_bytes = packed.detach().cpu().numpy()
    out = np.empty((n, wht_len), dtype=np.uint8)
    for i in range(n):
        out[i] = np.asarray(_core.unpack_bits(np_bytes[i], bit_width, wht_len))
    return torch.from_numpy(out).to(packed.device)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


# NormMode per the Rust distortion impls.
_NORM_MODE = {
    "mse":           "absorbed",
    "inner_product": "explicit",
    "linf":          "absorbed",
}


def encode_block_torch_stage2(
    X_np: np.ndarray,
    skeleton_parts: dict,
    *,
    custom_centroids: Optional[np.ndarray] = None,
    outlier_threshold: Optional[float] = None,
    device: str | torch.device = "cpu",
) -> dict:
    """Run stages 2..=5 of `encode_block` in PyTorch and return the
    codes dict (same keys as `kakeyaturbo_py.encode_block_codes` emits,
    minus the skeleton fields).

    `skeleton_parts` must be the dict from a Rust `encode_block_codes`
    call; we consume its skeleton and let Rust keep owning stage 1.

    `metric` is taken from `skeleton_parts["metric"]`.
    """
    X = torch.from_numpy(np.ascontiguousarray(X_np, dtype=np.float32)).to(device)
    skel = _skeleton_from_dict(skeleton_parts, device=device)
    metric = str(skeleton_parts["metric"])
    assert skel.bit_width in (1, 2, 3, 4)
    n, d = X.shape
    assert d == skel.d, f"input dim {d} != skeleton d {skel.d}"

    # Stage 2: project into d_eff.
    coeff = _project(X, skel)                                   # [n, d_eff]

    # Stage 3: K-means assignment.
    seg_id, t = _assign_and_project(coeff, skel)

    # Stage 4a: residual.
    residual = _residual(coeff, seg_id, t, skel)                # [n, d_eff]

    # Stage 4b: pad to wht_len, rotate via Rust bit-exact helper.
    residual_padded = torch.zeros((n, skel.wht_len), dtype=torch.float32, device=device)
    residual_padded[:, :skel.d_eff] = residual
    rotated = _rotate_rows_via_rust(residual_padded, skel.rotation_seed)

    # Stage 4c: per-vector scale = 1 / ||residual||, applied to `rotated`.
    res_norm = torch.linalg.norm(residual, dim=1)               # [n]
    eps32 = torch.finfo(torch.float32).eps
    scale = torch.where(res_norm > eps32, 1.0 / res_norm, torch.ones_like(res_norm))
    scaled = rotated * scale.unsqueeze(1)                       # [n, wht_len]

    # Outlier extraction (Phase A.3 scope — always off unless asked).
    max_outliers = 0
    outlier_idx = np.zeros((n, 1), dtype=np.uint16)
    outlier_val = np.zeros((n, 1), dtype=np.float32)
    outlier_count = np.zeros(n, dtype=np.uint32)
    if outlier_threshold is not None:
        thr = float(outlier_threshold)
        abs_scaled = scaled.abs()
        mask = (abs_scaled > thr)                                # [n, wht_len]
        counts = mask.sum(dim=1).detach().cpu().numpy().astype(np.uint32)
        max_outliers = int(counts.max()) if counts.size > 0 else 0
        if max_outliers > 0:
            outlier_idx = np.zeros((n, max_outliers), dtype=np.uint16)
            outlier_val = np.zeros((n, max_outliers), dtype=np.float32)
            # Row iteration is fine here: outliers are rare (a few %).
            mask_np = mask.detach().cpu().numpy()
            scaled_np = scaled.detach().cpu().numpy()
            for i in range(n):
                idxs = np.where(mask_np[i])[0].astype(np.uint16)
                outlier_idx[i, :idxs.size] = idxs
                # f16 round-trip to match Rust's `f16::from_f32`.
                vals32 = scaled_np[i, idxs]
                outlier_val[i, :idxs.size] = vals32.astype(np.float16).astype(np.float32)
        outlier_count = counts

    # Stage 5: Lloyd-Max quantise.
    if custom_centroids is None:
        centroids = torch.from_numpy(
            np.asarray(_core.centroids_gaussian(skel.bit_width))
        ).to(torch.float32).to(device)
    else:
        c_np = np.asarray(custom_centroids, dtype=np.float32)
        if c_np.size != (1 << skel.bit_width):
            raise ValueError(
                f"custom centroids has {c_np.size} entries, "
                f"need {1 << skel.bit_width}"
            )
        centroids = torch.from_numpy(c_np).to(torch.float32).to(device)

    q = _quantize_rows(scaled, centroids, metric)                # uint8 [n, wht_len]

    # Stage 5b: bit pack.
    packed = _pack_bits_rows(q, skel.bit_width)                   # [n, pbytes]

    # `t` and `norm` fields stored through fp16.
    t_f16 = _fp16_through(t)
    if _NORM_MODE[metric] == "absorbed":
        inv_scale = 1.0 / torch.clamp(scale, min=eps32)
        norm_field = _fp16_through(inv_scale)
    else:  # explicit
        x_norm = torch.linalg.norm(X, dim=1)
        norm_field = _fp16_through(x_norm)

    out = {
        # Skeleton fields (pass through unchanged).
        "mean":           np.asarray(skeleton_parts["mean"]),
        "basis":          np.asarray(skeleton_parts["basis"]),
        "centers":        np.asarray(skeleton_parts["centers"]),
        "d":              skel.d,
        "d_eff":          skel.d_eff,
        "k":              skel.k,
        "rotation_seed":  skel.rotation_seed,
        "wht_len":        skel.wht_len,
        "bit_width":      skel.bit_width,
        "metric":         metric,
        # Codes — numpy-native for easy comparison with Rust.
        "seg_id":         seg_id.detach().cpu().numpy().astype(np.uint32),
        "t":              t_f16.detach().cpu().numpy().astype(np.float32),
        "norm":           norm_field.detach().cpu().numpy().astype(np.float32),
        "residual_packed": packed.detach().cpu().numpy().astype(np.uint8),
        "outlier_idx":    outlier_idx,
        "outlier_val":    outlier_val,
        "outlier_count":  outlier_count,
        "max_outliers":   max_outliers,
    }
    return out


def decode_block_torch_from_parts(
    parts: dict,
    *,
    custom_centroids: Optional[np.ndarray] = None,
    device: str | torch.device = "cpu",
    disable_wht: bool = False,
) -> np.ndarray:
    """Inverse of `encode_block_torch_stage2`; accepts the same dict
    shape `encode_block_codes` / `encode_block_torch_stage2` emit.

    Byte-identical to `kakeyaturbo_py.decode_block_from_parts` on the
    same `parts` dict.
    """
    skel = _skeleton_from_dict(parts, device=device)
    cb = _codes_from_dict(parts, device=device)
    n = cb.seg_id.shape[0]

    # Unpack residual indices.
    indices = _unpack_bits_rows(cb.residual_packed, skel.bit_width, skel.wht_len)

    # Dequantise.
    if custom_centroids is None:
        centroids = torch.from_numpy(
            np.asarray(_core.centroids_gaussian(skel.bit_width))
        ).to(torch.float32).to(device)
    else:
        c_np = np.asarray(custom_centroids, dtype=np.float32)
        centroids = torch.from_numpy(c_np).to(torch.float32).to(device)

    q_vals = centroids[indices.long()]                           # [n, wht_len]

    # Outlier override: replace the dequantized value at outlier
    # coordinates with the f16-stored outlier value.
    if cb.max_outliers > 0:
        for i in range(n):
            cnt = int(cb.outlier_count[i].item())
            if cnt == 0:
                continue
            ix = cb.outlier_idx[i, :cnt].long()
            vv = cb.outlier_val[i, :cnt]
            q_vals[i, ix] = vv

    # inv_scale from `norm` field per norm-mode.
    if _NORM_MODE[cb.metric] == "absorbed":
        inv_scale = cb.norm
    else:
        inv_scale = torch.ones_like(cb.norm)
    q_scaled = q_vals * inv_scale.unsqueeze(1)

    # Inverse WHT rotation via Rust (bit-exact).
    if disable_wht:
        # Ablation: stage 4b was skipped at encode time, so the
        # inverse is also a no-op.  Take the leading `d_eff`
        # coords of `q_scaled` directly as the reconstructed
        # residual.  The extra wht_len-d_eff padded entries were
        # quantised to Lloyd-Max centroids but get discarded here.
        residual_rec = q_scaled[:, :skel.d_eff]
    else:
        unrotated = _inverse_rotate_rows_via_rust(q_scaled, skel.rotation_seed)
        residual_rec = unrotated[:, :skel.d_eff]

    # coeff = t * center + residual.
    chosen = skel.centers[cb.seg_id]                              # [n, d_eff]
    coeff = cb.t.unsqueeze(1) * chosen + residual_rec

    # Unproject back to X space.
    x_hat = coeff @ skel.basis + skel.mean.unsqueeze(0)
    return x_hat.detach().cpu().numpy().astype(np.float32)
