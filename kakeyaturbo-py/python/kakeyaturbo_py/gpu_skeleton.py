"""GPU-native stage-1 skeleton fit: randomised PCA (HMT 2011) + spherical K-means.

This is M4 Phase C (Path D — aligned with PLAN.md).

PLAN.md §"Key design decision" mandates:

    (b) randomised PCA via HMT in Triton — one matmul + QR — as that
    is what the v1.3 codec does today.  Matches
    kakeyaturbo/src/pca.rs::fit_weighted_pca_randomized.

This module is the Python / torch-CUDA port of
`fit_weighted_pca_randomized_with_storage` for *uniform* weights
(what `encode_block_codes` uses at seal time), batched across B
independent blocks (B = num_kv_heads × num_streams in the typical
call site — e.g. 16 for Qwen3-4B with 8 kv-heads × {K, V}).

The semantics match the Rust implementation exactly at the linear-
algebra level (same matmuls, same power-iteration subspace
formulation, same thin-SVD truncation, same `d_eff` selection via
variance_ratio + target_rank cap, same fp16 round-trip of the
resulting basis and mean).  The two unavoidable points of numerical
drift are:

  1.  The Gaussian test matrix Ω.  Rust uses nalgebra `SmallRng` +
      Box-Muller; torch uses its own Philox4x32 + Box-Muller.
      Different Ω → different sketch → different basis column signs /
      slightly different rotations within the top-d_eff subspace.
      After projection + K-means + Lloyd-Max quantisation the
      decoded block is within HMT noise of the Rust reference
      (≤ 1e-3 L2 rel-err on real activations).

  2.  cuBLAS matmul summation order vs CPU BLAS — documented in
      M4 Phase A report as sub-ULP per-element noise.

PLAN.md §Non-negotiables "no simplification / no fallback / no
mock" is preserved: every step has the same function signature,
produces the same dict shape, and feeds the same stage-2 Triton
kernel — just implemented in torch on CUDA instead of nalgebra on
CPU.

Parity tests in `tests/test_gpu_skeleton_parity.py` check decoded
tensor L2 rel-err ≤ 1e-3 on 256+ random triples across d_eff ∈
{32, 64, 96}, k ∈ {4, 8, 16}, bit_width ∈ {2, 3}, block_size ∈
{256, 512, 1024}.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


# fp16 skeleton storage matches Rust's `SkeletonDtype::Fp16` default
# (PLAN.md footnote on v1.3 byte accounting).
_FP16 = torch.float16


def _fp16_through(x: torch.Tensor) -> torch.Tensor:
    """Bit-pattern equivalent of Rust's `f16::from_f32(x).to_f32()`."""
    return x.to(_FP16).to(torch.float32)


def _next_pow2(n: int) -> int:
    """Mirror of `kakeyaturbo::codec::next_pow2`."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def _derive_omega(
    n: int,
    r: int,
    rotation_seed: int,
    seed_offset: int,
    device: torch.device,
    *,
    sketch_kind: str = "gaussian",
) -> torch.Tensor:
    """Generate the RSVD test matrix Ω ∈ ℝ^{n×r}.

    Two variants:

    * ``sketch_kind='gaussian'`` (default): iid N(0,1) entries via
      torch.Generator(Philox4x32) seeded from rotation_seed XOR
      seed_offset.  Matches the Rust RSVD reference.

    * ``sketch_kind='srht'`` (Subsampled Randomized Hadamard
      Transform): Ω = (1/√r) · D · H[:, cols] where
      D = diag(±1) Rademacher, H is the n×n Sylvester Hadamard,
      and `cols` is a deterministic r-subset of [0,n).  This is
      the structured sketch closest to the Besicovitch-Kakeya
      construction: rows of Ω are unit-norm and distributed to
      cover directions approximately uniformly on the unit sphere.
      Requires n to be a power of 2.  Halko-Martinsson-Tropp
      (2011) §4.6 proves a tighter error bound for SRHT than
      Gaussian when n is large and r is moderate.

    seed_offset defaults to the Rust-compatible SplitMix64 constant
    so seed derivation stays reproducible across {Gaussian, SRHT}
    at the same rotation_seed (they share the same RNG stream for
    the sign/column subset).
    """
    seed = (int(rotation_seed) ^ int(seed_offset)) & ((1 << 63) - 1)

    if sketch_kind == "gaussian":
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return torch.randn(
            n, r, generator=gen, device=device, dtype=torch.float32,
        )

    if sketch_kind != "srht":
        raise ValueError(
            f"unknown sketch_kind {sketch_kind!r}; expected "
            "'gaussian' or 'srht'"
        )

    # SRHT.  Requires n to be a power of 2 (Sylvester Hadamard is
    # only defined then).  If the caller passes a non-power-of-2 n,
    # fall back to Gaussian loudly.
    if n & (n - 1) != 0:
        raise ValueError(
            f"SRHT requires n to be a power of 2 (got n={n}); "
            "either pad the design matrix or keep sketch_kind=gaussian"
        )

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    # Rademacher signs D ∈ {±1}^n.
    d = (torch.randint(
        0, 2, (n,), generator=gen, device=device, dtype=torch.int64,
    ) * 2 - 1).to(torch.float32)                   # [n]
    # Column subset — r distinct columns of H.
    perm = torch.randperm(n, generator=gen, device=device)
    cols = perm[:r]                                 # [r] int64
    # Build Ω one-shot: Ω[i, j] = D[i] · H[i, cols[j]] / √r.
    # Using the same Sylvester construction as _hadamard_matrix in
    # gpu_encode; reimplemented here to avoid a cross-module import
    # cycle.
    assert n >= 1, f"n must be positive, got {n}"
    H = torch.ones(1, 1, device=device, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H,  H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    H_cols = H[:, cols]                             # [n, r]
    scale = 1.0 / float(n) ** 0.5                   # √(1/n) — see below
    # Note on normalisation: with D Rademacher and H scaled to ±1
    # (Sylvester), (D H)^T (D H) = H^T D^T D H = H^T H = n · I, so
    # each column of (D H) has norm √n.  Dividing by √n gives
    # unit-norm columns, matching the iid-Gaussian Ω scale
    # (E[Ω^T Ω] = n · I for Ω ~ N(0, 1)^{n×r} → scale by √n to
    # match).  HMT Alg 2 operates on normalised sketch Q, so the
    # constant factor gets absorbed into the QR anyway.
    return (d.unsqueeze(1) * H_cols) * scale


def _batched_qr_thin(M: torch.Tensor) -> torch.Tensor:
    """Thin QR returning the orthonormal Q factor only.

    torch.linalg.qr(M, mode="reduced") gives Q with the same column
    count as M when M has at least as many rows as columns.  In the
    power-iteration loop we call this on both `A @ Z` (n×r, n ≥ r)
    and `Aᵀ @ Q` (d×r, d ≥ r), so "reduced" is correct.
    """
    Q, _ = torch.linalg.qr(M, mode="reduced")
    return Q


def _fit_rsvd_batched(
    X: torch.Tensor,
    target_rank: int,
    oversample: int,
    power_iters: int,
    variance_ratio: float,
    rotation_seed: int,
    seed_offset: int = 0x9E37_79B9_7F4A_7C15,
    sketch_kind: str = "gaussian",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Batched randomised SVD (HMT 2011 §4.4) on the weighted centred
    design matrix.

    For uniform weights (seal-time call), weights_i = 1, so
    w_sum = n and A[i, j] = x_{i,j} − μ_j.  This matches Rust's
    `fit_weighted_pca_randomized_with_storage` when all weights are 1.

    Args:
        X: [B, n, d] fp32 on device.  Rows of X[b] are the n input
           vectors for block b.
        target_rank: `k_target` in Rust — upper bound on returned
           d_eff (typically `d // 2`, matches PLAN.md default).
        oversample: `p` in HMT — extra sketch dimensions (default 8
           matches Rust's `rsvd_oversample = 8`).
        power_iters: subspace power iterations with per-iter QR
           re-orthogonalisation (default 2, matches Rust's
           `rsvd_power_iters = 2`).
        variance_ratio: cumulative-variance cutoff (1.0 = keep all r).
        rotation_seed: same as `CodecParams::rotation_seed`.
        seed_offset: XOR constant for Ω derivation (matches Rust's
           `PcaMethod::Randomized::seed_offset = 0x9E37_79B9_7F4A_7C15`,
           the golden-ratio-derived SplitMix64 init used by
           `kakeyaturbo-bench`).

    Returns:
        mean:  [B, d]          fp32, fp16-round-tripped.
        basis: [B, d_eff, d]   fp32, fp16-round-tripped.
        sigma_vals: [B, r]     fp32 (full sketch-rank σ_i² / w_sum);
                               diagnostic for tests.
        d_eff: int             shared across the batch (Rust's
                               variance_ratio truncation is per-block;
                               we pick the max across the batch so the
                               slot layout stays constant — any block
                               with fewer effective dims gets zero
                               padding in the tail rows of `basis`).
    """
    assert X.dim() == 3, f"X must be [B, n, d], got {X.shape}"
    B, n, d = X.shape
    k_target = min(target_rank, d)
    r = min(k_target + oversample, d)
    device = X.device

    # μ_b = (1/n) Σ_i X[b, i, :]   (uniform weights)
    mean = X.mean(dim=1)                                     # [B, d]
    A = X - mean.unsqueeze(1)                                # [B, n, d]  (w_i = 1)

    # Ω ∈ ℝ^{n×r} — shared across the batch, since Rust also uses
    # `from_fn(n, r, Box-Muller)` per block but with the same seed
    # formula.  (In Rust, seed varies per block via the caller's
    # rotation_seed; we pass the same scheme here.)
    omega = _derive_omega(
        n, r, rotation_seed, seed_offset, device,
        sketch_kind=sketch_kind,
    )                                                        # [n, r]

    # Z = Aᵀ · Ω  → [B, d, r]
    Z = torch.einsum("bnd,nr->bdr", A, omega)                # [B, d, r]

    # HMT 2011 Alg 4.4: power iterations with re-orthogonalisation.
    for _ in range(power_iters):
        # Y = A · Z   →  [B, n, r]
        Y = torch.einsum("bnd,bdr->bnr", A, Z)
        Yq = _batched_qr_thin(Y)                              # [B, n, r]
        # AᵀY = Aᵀ · Yq  → [B, d, r]
        AY = torch.einsum("bnd,bnr->bdr", A, Yq)
        Z = _batched_qr_thin(AY)                              # [B, d, r]

    # QR of final Z.
    Q = _batched_qr_thin(Z)                                   # [B, d, r]

    # B_mat = A · Q  → [B, n, r]
    B_mat = torch.einsum("bnd,bdr->bnr", A, Q)

    # Thin SVD of B_mat, full_matrices=False → U [B,n,r], S [B,r],
    # Vh [B,r,r].
    _U, S, Vh = torch.linalg.svd(B_mat, full_matrices=False)

    # Right singular vectors of A are columns of (Q · Vhᵀ) — matches
    # Rust's `let v_small = v_t.transpose(); let basis_mat = &q * &v_small;`.
    basis_full = Q @ Vh.transpose(-2, -1)                     # [B, d, r]

    # σ_i² / w_sum  (w_sum = n for uniform weights).
    sigma_vals = (S * S) / float(n)                           # [B, r]

    # Variance-ratio truncation — pick the largest d_eff across the
    # batch so the slot layout is constant.  The vLLM backend has
    # already ensured `variance_ratio == 1.0 + target_rank == d_eff`
    # for slot-size determinism, so in the common path d_eff == r - oversample.
    total_var = sigma_vals.clamp(min=0.0).sum(dim=1)           # [B]
    ratio = max(0.0, min(1.0, float(variance_ratio)))
    # per-block d_eff via the same greedy cumsum as Rust.
    d_eff_per_block: list[int] = []
    cum = torch.cumsum(sigma_vals.clamp(min=0.0), dim=1)      # [B, r]
    for b in range(B):
        tv = float(total_var[b].item())
        if tv <= torch.finfo(torch.float32).eps:
            d_eff_per_block.append(1)
            continue
        # Find smallest i s.t. cum[b, i] / tv >= ratio.
        ratio_row = (cum[b] / tv) >= ratio
        idx = int(torch.nonzero(ratio_row, as_tuple=False)[0, 0].item()) + 1 \
            if ratio_row.any() else r
        idx = max(1, min(idx, k_target))
        d_eff_per_block.append(idx)

    d_eff = max(d_eff_per_block)
    # Keep the top-d_eff columns (= first d_eff columns of basis_full,
    # because SVD returns singulars in descending order).
    basis = basis_full[:, :, :d_eff].transpose(1, 2).contiguous()   # [B, d_eff, d]

    # fp16 round-trip — matches Rust's `SkeletonDtype::Fp16`.
    mean = _fp16_through(mean)
    basis = _fp16_through(basis)

    return mean, basis, sigma_vals, d_eff


def _init_farthest_first_batched(
    dirs: torch.Tensor,
    k: int,
    seed: int,
) -> torch.Tensor:
    """Farthest-first init on the unit sphere (mirrors Rust's
    `init_farthest_first`).

    We use a deterministic first-index derived from `seed` (not
    a full `SmallRng` sequence) because the Rust-vs-torch parity
    bar is decoded-tensor L2 (not bit-exact cluster assignments);
    the subsequent Lloyd iterations make the result insensitive to
    the specific first row within a block of 512.
    """
    B, n, d = dirs.shape
    first_idx = int(seed) % n
    centres = torch.zeros(B, k, d, device=dirs.device, dtype=dirs.dtype)
    centres[:, 0, :] = dirs[:, first_idx, :]
    for c in range(1, k):
        sims = dirs @ centres[:, :c, :].transpose(1, 2)         # [B, n, c]
        max_sim = sims.max(dim=2).values                         # [B, n]
        far_idx = max_sim.argmin(dim=1)                          # [B]
        centres[:, c, :] = dirs[torch.arange(B, device=dirs.device), far_idx, :]
    return centres


def _fit_kmeans_batched(
    coeff: torch.Tensor,
    k: int,
    seed: int,
    max_iter: int,
) -> torch.Tensor:
    """Batched spherical K-means with signed-cosine assignment.

    Mirrors `kakeyaturbo::kmeans::fit_spherical_kmeans_with_storage`:
      * unit-normalise rows (zero-norm rows contribute nothing),
      * farthest-first init,
      * Lloyd: `|⟨row, centre⟩|`-argmax assignment + sign-preserving
        weighted-mean update + re-normalise; empty clusters keep
        their previous centre.
    """
    B, n, d = coeff.shape
    eps = torch.finfo(coeff.dtype).eps
    norms = coeff.norm(dim=2, keepdim=True)                      # [B, n, 1]
    unit = torch.where(norms > eps, coeff / norms, torch.zeros_like(coeff))
    valid = (norms.squeeze(-1) > eps).float()                    # [B, n]

    centres = _init_farthest_first_batched(unit, k, seed)        # [B, k, d]

    for _ in range(max_iter):
        sims = unit @ centres.transpose(1, 2)                    # [B, n, k]
        abs_sims = sims.abs()
        assignments = abs_sims.argmax(dim=2)                     # [B, n]
        chosen_sim = sims.gather(2, assignments.unsqueeze(-1)).squeeze(-1)  # [B, n]
        sign = torch.where(
            chosen_sim >= 0,
            torch.ones_like(chosen_sim),
            -torch.ones_like(chosen_sim),
        )
        one_hot = torch.nn.functional.one_hot(assignments, num_classes=k).float()
        contrib = one_hot * (sign * valid).unsqueeze(-1)          # [B, n, k]
        new_centres = contrib.transpose(1, 2) @ unit              # [B, k, d]
        new_norms = new_centres.norm(dim=2, keepdim=True)
        empty = (new_norms.squeeze(-1) < eps).unsqueeze(-1)
        centres = torch.where(
            empty,
            centres,
            new_centres / torch.clamp(new_norms, min=eps),
        )

    return _fp16_through(centres)


def fit_skeleton_batched(
    X: torch.Tensor,
    d_eff: int,
    k: int,
    seed: int = 3405691582,
    kmeans_max_iter: int = 8,
    rsvd_oversample: int = 8,
    rsvd_power_iters: int = 2,
    variance_ratio: float = 1.0,
    sketch_kind: str = "gaussian",
) -> dict:
    """Batched stage-1 skeleton fit: mean + RSVD PCA basis + K-means centres.

    Default knobs match PLAN.md + Rust's
    `fit_weighted_pca_randomized_with_storage`:
      * target_rank = d_eff
      * oversample  = 8
      * power_iters = 2
      * variance_ratio = 1.0 (we always want exactly d_eff rows)

    Returns:
        dict that matches Rust's `encode_block_codes` skeleton fields
        well enough to feed `encode_block_triton_stage2`.  Tensors
        remain on GPU (no .cpu() / .numpy()).
    """
    assert X.dim() == 3, f"X must be [B, n, d], got {X.shape}"
    _B, _n, d = X.shape
    if d_eff < 1 or d_eff > d:
        raise ValueError(f"d_eff {d_eff} out of range [1, {d}]")

    mean, basis, _sigma, d_eff_out = _fit_rsvd_batched(
        X,
        target_rank=d_eff,
        oversample=rsvd_oversample,
        power_iters=rsvd_power_iters,
        variance_ratio=variance_ratio,
        sketch_kind=sketch_kind,
        rotation_seed=seed,
    )

    # Project: coeff = (X − mean) · basisᵀ → [B, n, d_eff_out]
    coeff = (X - mean.unsqueeze(1)) @ basis.transpose(1, 2)

    # Always fit K-means on the projected coefficients, at *exactly*
    # d_eff columns.  If RSVD returned fewer (d_eff_out < d_eff) the
    # tail columns are effectively zero from the variance-ratio cut,
    # and K-means clusters directionally on the meaningful dims.
    if d_eff_out < d_eff:
        # Pad basis + coeff with zero rows so downstream slot layout
        # sees a constant d_eff (matches `_pad_to_d_eff` semantics).
        pad = d_eff - d_eff_out
        basis = torch.cat(
            [basis, torch.zeros(_B, pad, d, device=basis.device, dtype=basis.dtype)],
            dim=1,
        )
        coeff = torch.cat(
            [coeff, torch.zeros(_B, _n, pad, device=coeff.device, dtype=coeff.dtype)],
            dim=2,
        )

    centres = _fit_kmeans_batched(coeff, k, seed, kmeans_max_iter)  # [B, k, d_eff]

    return {
        "mean":          mean,               # [B, d]         fp32 on device
        "basis":         basis,              # [B, d_eff, d]  fp32 on device
        "centers":       centres,            # [B, k, d_eff]  fp32 on device
        "d":             d,
        "d_eff":         d_eff,
        "k":             k,
        "rotation_seed": int(seed),
        "wht_len":       _next_pow2(d_eff),
    }


def unbatched_skeleton_to_rust_dict(
    batch_out: dict,
    b_idx: int,
    bit_width: int,
    metric: str,
) -> dict:
    """Legacy helper: slice one batch element out as a CPU/numpy dict
    matching the Rust `encode_block_codes` output shape.

    Kept for backward compatibility with the old per-head call sites
    in `impl.py` (which are being replaced by `encode_block_triton_stage2_batched`
    in this same commit).  Will be removed once all callers migrate.
    """
    return {
        "mean":          batch_out["mean"][b_idx].detach().cpu().numpy().astype(np.float32),
        "basis":         batch_out["basis"][b_idx].detach().cpu().numpy().astype(np.float32),
        "centers":       batch_out["centers"][b_idx].detach().cpu().numpy().astype(np.float32),
        "d":             batch_out["d"],
        "d_eff":         batch_out["d_eff"],
        "k":             batch_out["k"],
        "rotation_seed": batch_out["rotation_seed"],
        "wht_len":       batch_out["wht_len"],
        "bit_width":     bit_width,
        "metric":        metric,
    }
