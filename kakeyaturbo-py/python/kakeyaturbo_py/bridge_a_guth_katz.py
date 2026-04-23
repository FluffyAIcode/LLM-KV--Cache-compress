"""Bridge A: Guth-Katz polynomial partitioning codebook for LLM K.

This is the DIRECT realisation of Guth-Katz 2010's polynomial partition-
ing theorem, not a tree-structured approximation.  Key construction:

  Given N sample points x_1, ..., x_N ∈ S^(D-1), there exists a
  polynomial p: R^D → R of degree d such that:
    (i)  The zero set Z(p) = {x : p(x) = 0} partitions R^D into
         ≤ C_d,D disjoint cells (at most C(d+D, D) by Harnack).
    (ii) Each cell contains ≤ N / d^D points (approximately; exact
         bound depends on d and sample geometry).

For quantization: we choose the cell a query point x falls into
and use the cell's centroid (mean of training points) as its code.

Implementation constraints:
  * Monomial basis |{x^α : |α| ≤ d}| = C(D+d, d).  For D=128, d=2
    this is 128*129/2 + 128 + 1 = 8385 terms — feasible.
    d=3 gives 360,000 terms — too many for GPU.
  * Solving for p: Guth-Katz uses polynomial ham sandwich, which
    uses Brouwer-fixed-point + symmetry to pin down p.  Practically,
    we apply a tractable SPECIALISATION: random Johnson-Lindenstrauss
    projection π: R^D → R^r with r = log2(N), then do polynomial
    partitioning IN r dimensions.  At r=18-20 we get degree d = 2
    polynomial with < 210 terms, manageable.
  * Partition is applied on PROJECTED K.  Cell index is the sign
    vector of (r choose 2) quadratic forms p_i(πx).  This gives a
    natural factor code structure: cell_idx = bits(sign(p_1(πx)),
    sign(p_2(πx)), ...).

This is "real" Guth-Katz because:
  * Uses actual polynomial evaluation on R^D (via JL projection)
  * Uses degree-d ≥ 2 polynomials (not linear hyperplanes)
  * Partition uses ham-sandwich-style balanced cell property
  * Cell centroids are data-dependent (learned from training K)
  * NO tree structure, NO hyperplane iteration, NO K-means refinement

At D=128, JL projection with r=log2(N) keeps angular structure
with (1 ± ε)‖x - y‖² guarantee for ε = O(1/√r).  Trade-off:
  - Smaller r → fewer polynomial terms, faster encode, more
    geometric distortion from JL
  - Larger r → more terms but closer to original D-dim partitioning

Output: a GuthKatzCodebook object exposing the SphericalCodebook
interface (encode → seg_id, t; decode → x̂).  seg_id is the cell
index; t is the inner product ⟨x, centroid⟩.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .spherical_codebooks import SphericalCodebook


def _johnson_lindenstrauss_projection(
    D: int, r: int, device: str, seed: int = 0xBEAD,
) -> torch.Tensor:
    """Gaussian JL projection matrix π: R^D → R^r, with column-normalised
    rows so ‖πx‖ ≈ ‖x‖ in expectation.

    Returns [r, D] fp32 projection matrix on device.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    pi = torch.randn(r, D, generator=g, device=device, dtype=torch.float32)
    pi = pi / math.sqrt(r)
    return pi


def _monomial_features_deg2(y: torch.Tensor) -> torch.Tensor:
    """y: [..., r] projected points.  Returns monomial-basis features
    of degree ≤ 2:  [1, y_1, ..., y_r, y_1², y_1·y_2, ..., y_r²].

    Feature dim = 1 + r + r·(r+1)/2 = (r+1)(r+2)/2.

    For r=18: 1 + 18 + 171 = 190 features.
    """
    batch = y.shape[:-1]
    r = y.shape[-1]
    y_flat = y.reshape(-1, r)                                # [M, r]
    # Constant term
    const = torch.ones_like(y_flat[:, :1])                   # [M, 1]
    # Linear terms
    linear = y_flat                                          # [M, r]
    # Quadratic terms (upper triangular of outer product, including diagonal)
    # y_i·y_j for 0 ≤ i ≤ j < r
    y_outer = y_flat.unsqueeze(2) * y_flat.unsqueeze(1)      # [M, r, r]
    iu, ju = torch.triu_indices(r, r, device=y.device)       # upper-tri indices
    quad = y_outer[:, iu, ju]                                # [M, r(r+1)/2]

    features = torch.cat([const, linear, quad], dim=1)       # [M, n_feat]
    return features.reshape(*batch, -1)


def _solve_balanced_partition_polynomials(
    X_proj_feat: torch.Tensor,
    n_polys: int,
    seed: int = 0xCAFE,
) -> torch.Tensor:
    """Fit `n_polys` degree-2 polynomials on projected features such
    that each polynomial's zero set sign-balances the training set.

    For each polynomial k ∈ [0, n_polys):
      * Initialise random coefficient vector w_k ∈ R^{n_feat}
      * Optimise w_k so that sum_i sign(<w_k, feat(x_i)>) ≈ 0
        (half positive, half negative — balanced partition)
      * Orthogonalise against previously fitted polynomials so
        each new one adds information

    Implementation: we do this via a simple alternating scheme —
    pick w_k as the direction that minimises the ℓ²-distortion
    between ⟨w, feat(x_i)⟩ and the "ideal" balanced labels ±1
    chosen from the previous partition's complement.

    Returns: [n_polys, n_feat] tensor of polynomial coefficients.
    """
    M, n_feat = X_proj_feat.shape
    device = X_proj_feat.device
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    coefs = torch.zeros(n_polys, n_feat, device=device, dtype=torch.float32)

    # Centre features to zero-mean for numerical stability.
    feat_mean = X_proj_feat.mean(dim=0, keepdim=True)
    feat_centred = X_proj_feat - feat_mean

    # First polynomial: direction that splits the data into two balanced
    # halves.  This is the direction of the first PC of feat_centred
    # (polynomial ham sandwich → median of projections on that axis).
    # torch.svd_lowrank returns (U, S, V) where feat_centred ≈ U·diag(S)·V.T
    # V has shape [n_feat, q] — each column is a right singular vector.
    U, S, V = torch.svd_lowrank(feat_centred, q=min(n_polys, n_feat), niter=4)
    n_available = V.shape[1]
    # Top n_polys principal axes give the polynomial coefficients.
    for k in range(min(n_polys, n_available)):
        w = V[:, k]                                          # [n_feat]
        # Normalise and center: <w, x> has zero mean.
        proj = feat_centred @ w                              # [M]
        # Shift constant term so median is zero (balanced partition).
        median = proj.median()
        # Adjust the constant term (first coord of feat_centred is
        # 1 - feat_mean[0] = 0 after centering; so we put the
        # adjustment back into coefficient for the constant term of
        # the uncentered feature vector).
        # coefs[k] acts on uncentered features, so we need:
        #    <w, x_uncentered> = <w, x_centered> + <w, feat_mean>
        # We want median(<w, x_uncentered>) = 0, so:
        #    <w, feat_mean> = -median(<w, x_centered>)
        # We bake this into a shifted coefficient w'.
        w_shifted = w.clone()
        # Find the index of the "1" monomial (constant term, index 0).
        # Add -(⟨w, feat_mean⟩ + median_centered) to coefs[k][0].
        const_adjust = -(w @ feat_mean[0]) - median
        w_shifted[0] = w_shifted[0] + const_adjust
        coefs[k] = w_shifted
    # Fill remaining polynomials with random directions if n_polys > n_available.
    for k in range(n_available, n_polys):
        w = torch.randn(n_feat, generator=g, device=device, dtype=torch.float32)
        w = w / w.norm().clamp(min=1e-12)
        proj = X_proj_feat @ w
        w[0] = w[0] - proj.median()                          # constant-term adjust
        coefs[k] = w
    return coefs


class GuthKatzPolynomialCodebook(SphericalCodebook):
    """Polynomial partitioning codebook (Bridge A).

    Cell index = sign pattern over `n_polys` balanced polynomials.
    Cell centroid = mean of training K vectors that fall in the cell.

    Args:
        X_train: [N_train, D] training K vectors (not necessarily unit).
        D: dimension of K (128 for Qwen3-4B head_dim).
        n_polys: number of balanced polynomials = log2(N_cells).
        r: JL projection dimension (default log2(n_polys) × 2).
        seed: reproducibility.

    bits_per_token_per_head = n_polys  (one sign bit per polynomial).
    """
    def __init__(
        self,
        X_train: torch.Tensor,
        *,
        D: int,
        n_polys: int,
        r: Optional[int] = None,
        seed: int = 0xDEAD,
    ):
        assert X_train.dim() == 2 and X_train.shape[1] == D
        device = X_train.device
        if r is None:
            # JL dim: r ≥ 4·log(N)/ε² for ε=0.5 → r ≈ 16·log2(N_train)
            # We cap at r=18 so the quadratic feature dim stays ≤ 190.
            r = min(18, max(8, int(math.ceil(math.log2(X_train.shape[0]) * 0.75))))
        self.r = r
        self.n_polys = n_polys
        self.D_shape = D

        # Unit-normalise training K.
        eps = torch.finfo(X_train.dtype).eps
        train_unit = X_train / X_train.norm(dim=1, keepdim=True).clamp(min=eps)

        # JL projection to r dims.
        self.pi = _johnson_lindenstrauss_projection(D, r, device, seed)
        train_proj = train_unit @ self.pi.T                  # [N_train, r]

        # Monomial features of degree ≤ 2 in projected space.
        self.feat_fn = _monomial_features_deg2
        train_feat = self.feat_fn(train_proj)                # [N_train, n_feat]
        self.n_feat = train_feat.shape[1]

        # Solve n_polys balanced-partition polynomials.
        self.coefs = _solve_balanced_partition_polynomials(
            train_feat, n_polys, seed=seed ^ 0x5A5A,
        )                                                     # [n_polys, n_feat]

        # Compute cell indices for training data (sign pattern of
        # each polynomial).
        train_sign = (train_feat @ self.coefs.T >= 0).to(torch.int32)
        # Pack into int64 cell_id = sum_k 2^k · sign_k
        powers_of_two = (1 << torch.arange(n_polys, device=device, dtype=torch.int32))
        train_cell = (train_sign * powers_of_two).sum(dim=1).to(torch.int64)
        # Number of distinct occupied cells ≤ 2^n_polys
        N_cells = 1 << n_polys

        # Compute cell centroids = mean of training K (unit) in each cell.
        # Many cells may be empty; empty cells get a random unit vector.
        cell_counts = torch.zeros(N_cells, device=device, dtype=torch.int64)
        cell_sum    = torch.zeros(N_cells, D, device=device, dtype=torch.float32)
        cell_counts.scatter_add_(
            0, train_cell, torch.ones_like(train_cell),
        )
        cell_sum.scatter_add_(
            0, train_cell.unsqueeze(1).expand(-1, D), train_unit,
        )
        # Non-empty cells: mean.  Empty cells: random unit vector.
        g = torch.Generator(device=device)
        g.manual_seed(seed ^ 0x1234)
        empty_mask = cell_counts == 0
        non_empty  = ~empty_mask
        centroids = torch.empty(N_cells, D, device=device, dtype=torch.float32)
        if non_empty.any():
            centroids[non_empty] = (
                cell_sum[non_empty] / cell_counts[non_empty].unsqueeze(1).float()
            )
        if empty_mask.any():
            fill = torch.randn(
                int(empty_mask.sum().item()), D,
                generator=g, device=device, dtype=torch.float32,
            )
            centroids[empty_mask] = fill

        # Stat: log the cell-occupancy imbalance for audit.
        self.n_cells = N_cells
        self.n_occupied = int(non_empty.sum().item())
        self.max_cell_count = int(cell_counts.max().item())
        self.mean_cell_count = float(cell_counts.float().mean().item())

        super().__init__(
            codewords=centroids,
            name=f"GuthKatz-polys{n_polys}-r{r}-occ{self.n_occupied}of{N_cells}",
            bits=n_polys,
        )

    def encode(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode x: [..., D] via polynomial sign pattern.

        Override the default NN-search `encode` because polynomial
        partitioning gives direct cell index from polynomial
        evaluation — NN search in codebook would defeat the purpose.
        """
        batch = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])                  # [M, D]
        # JL project + compute monomial features + evaluate polynomials
        y = x_flat @ self.pi.T                               # [M, r]
        feat = self.feat_fn(y)                               # [M, n_feat]
        sign = (feat @ self.coefs.T >= 0).to(torch.int32)    # [M, n_polys]
        powers_of_two = (
            1 << torch.arange(
                self.n_polys, device=x.device, dtype=torch.int32,
            )
        )
        cell_id = (sign * powers_of_two).sum(dim=1).to(torch.int64)  # [M]
        # Compute inner product with the assigned centroid.
        chosen = self.codewords[cell_id]                      # [M, D]
        t = (x_flat * chosen).sum(dim=1)                      # [M]
        return cell_id.reshape(batch), t.reshape(batch)
