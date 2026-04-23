"""Bridge C: Non-Gaussian shaping via empirical density for LLM K.

Shannon 1.53 dB shaping gap is tight for strictly i.i.d. Gaussian
sources.  LLM K has been measured non-Gaussian (Phase 1) — sub-
Gaussian, with kurtosis < 3 and W_2/σ ~ 0.3.  The shaping gap is
therefore only a TIGHT bound when the source actually is Gaussian.

This bridge builds a codebook that is optimal FOR THE EMPIRICAL
K DISTRIBUTION (not for assumed-Gaussian), via:

  1. Apply Hadamard rotation to the unit-normalised K (same as TQ
     pre-processing).  This makes the marginal per-coord more
     Gaussian-like by CLT but does NOT fully Gaussianise; the
     residual non-Gaussianity is what we harvest.

  2. Fit an EMPIRICAL Lloyd-Max codebook per coord: solve
         argmin_{c ∈ R^k} sum_i min_j (y_i - c_j)²
     on the real y = H·x̂ samples.  This is the same as classical
     Lloyd-Max but on empirical density, not the Gaussian density.

  3. Learn a data-adapted shaping region Λ_c (support set): the
     empirical 99% quantile hyper-rectangle in R^D.  Clip y to
     this region before quantisation, reducing wasted bits on
     low-density tails.

  4. Optionally apply per-coord ORDERING via entropy rank (higher-
     entropy coords first), so that a fixed-bit budget per coord
     can be redistributed by rank.  Not strictly Lloyd-Max but
     aligned with the shaping-gain theory.

Contrast with snapA/snapF's Lloyd-Max:
  snapA/snapF Lloyd-Max is applied to residual AFTER PCA+K-means,
  where the distribution is near-perfectly Gaussian by construction
  (Phase 2 showed this).  This bridge applies Lloyd-Max to the
  RAW Hadamard-rotated K unit vectors, where the non-Gaussianity
  actually lives.

This is TurboQuant's per-coord structure but with data-matched
code-book instead of Gaussian-assumed codebook.  The answer to
"can Kakeya-style non-Gaussian shaping beat TurboQuant" lives here.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .spherical_codebooks import SphericalCodebook


def _sylvester_hadamard_normalised(D: int, device: str) -> torch.Tensor:
    assert (D & (D - 1)) == 0
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(D)


def _solve_lloyd_max_per_coord(
    y: torch.Tensor,
    n_levels: int,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Lloyd's algorithm per-dimension on empirical samples.

    y: [N, D] samples in R^D.
    Returns [D, n_levels] centroids per dim (sorted ascending within each dim).
    """
    N, D = y.shape
    device = y.device
    # Init: per-dim quantiles of empirical distribution.
    quantiles = (torch.arange(n_levels, device=device, dtype=torch.float32) + 0.5) / n_levels
    # [D, n_levels] init — per-dim quantile.
    # torch.quantile expects a 1D q; iterate over dims.
    centroids = torch.empty(D, n_levels, device=device, dtype=torch.float32)
    for d in range(D):
        centroids[d] = torch.quantile(y[:, d], quantiles)

    for it in range(max_iter):
        # Sort centroids ascending per dim.
        centroids, _ = centroids.sort(dim=1)
        # Boundaries between centroids per dim.
        boundaries = (centroids[:, :-1] + centroids[:, 1:]) / 2.0  # [D, n_levels-1]
        # Bucketise y per dim via searchsorted.
        # For each d: bucket[i] = searchsorted(boundaries[d], y[i, d])
        # Vectorise: reshape to [D, N] via transpose.
        yt = y.transpose(0, 1).contiguous()                  # [D, N]
        buckets = torch.zeros(D, N, dtype=torch.int64, device=device)
        for d in range(D):
            buckets[d] = torch.searchsorted(boundaries[d], yt[d])

        # Compute cluster means per (dim, level).
        new_centroids = torch.empty_like(centroids)
        for k in range(n_levels):
            mask = (buckets == k).to(yt.dtype)                # [D, N]
            counts = mask.sum(dim=1).clamp(min=1)             # [D]
            sums = (yt * mask).sum(dim=1)                     # [D]
            new_centroids[:, k] = sums / counts
            # Keep old value where cluster empty (counts == 0 replaced by 1 → sum=0 → centroid=0; fall back to old).
            empty = (mask.sum(dim=1) == 0)
            if empty.any():
                new_centroids[empty, k] = centroids[empty, k]

        delta = (new_centroids - centroids).abs().max()
        centroids = new_centroids
        if delta < tol:
            break
    return centroids.sort(dim=1).values


def _learn_shaping_rectangle(
    y: torch.Tensor,
    percentile: float = 0.995,
) -> torch.Tensor:
    """Learn a per-dim clipping range (l_d, u_d) based on empirical
    percentile.  Returns [D, 2] with columns (lower, upper).
    """
    N, D = y.shape
    lower = torch.quantile(y, 1.0 - percentile, dim=0)
    upper = torch.quantile(y, percentile, dim=0)
    return torch.stack([lower, upper], dim=1)                 # [D, 2]


class NonGaussianShapingCodebook(SphericalCodebook):
    """Non-Gaussian shaping codebook (Bridge C).

    Pipeline per vector x ∈ R^D:
      unit = x / ‖x‖
      y    = Hadamard · unit                                 (D=128)
      y_c  = clip(y, shaping_rectangle)                      (data-adapted support)
      q    = argmin over empirical Lloyd-Max centroids per dim
      idx  = (q_1, q_2, ..., q_D)                            (cell id)
      t    = ⟨y, y_hat⟩                                      (for interface)

    bits_per_token_per_head = D · log2(n_levels).
    """
    def __init__(
        self,
        X_train: torch.Tensor,
        *,
        D: int,
        bits_per_coord: int,
    ):
        assert X_train.dim() == 2 and X_train.shape[1] == D
        assert (D & (D - 1)) == 0, f"D must be power of 2, got {D}"
        device = X_train.device
        self.D_shape = D
        self.bits_per_coord = bits_per_coord
        n_levels = 1 << bits_per_coord

        # Unit-normalise + Hadamard.
        eps = torch.finfo(X_train.dtype).eps
        train_unit = X_train / X_train.norm(dim=1, keepdim=True).clamp(min=eps)
        self.H = _sylvester_hadamard_normalised(D, device)   # [D, D]
        train_y = train_unit @ self.H                         # [N, D]

        # Learn shaping rectangle.
        self.shaping_rect = _learn_shaping_rectangle(train_y, percentile=0.995)
        train_y_clip = train_y.clamp(
            self.shaping_rect[:, 0].unsqueeze(0),
            self.shaping_rect[:, 1].unsqueeze(0),
        )

        # Empirical Lloyd-Max per coord.
        self.centroids = _solve_lloyd_max_per_coord(
            train_y_clip, n_levels=n_levels, max_iter=100,
        )                                                     # [D, n_levels]

        # No flat codewords table — encode/decode act per-coord.  For
        # SphericalCodebook interface we just store the mean of the
        # training set as a dummy single-codeword "codebook".
        dummy_codewords = train_unit.mean(dim=0, keepdim=True).clamp(min=eps)
        dummy_codewords = dummy_codewords / dummy_codewords.norm(dim=1, keepdim=True)
        total_bits = D * bits_per_coord
        super().__init__(
            codewords=dummy_codewords,
            name=f"NonGaussShaping-bits{bits_per_coord}perCoord-total{total_bits}",
            bits=total_bits,
        )

    def encode(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Override: per-coord empirical quantisation.

        Returns (cell_id, t) where cell_id is a dummy tensor (too many
        cells to enumerate) and t is the reconstructed norm factor.
        Actual encode/decode is via ._encode_decode_vector.
        """
        # Not used in roundtrip; we override roundtrip instead.
        raise NotImplementedError(
            "Use .roundtrip() for NonGaussianShapingCodebook."
        )

    def decode(
        self, seg_id: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Use .roundtrip() for NonGaussianShapingCodebook."
        )

    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode + decode via non-Gaussian shaping."""
        batch = x.shape[:-1]
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)
        norms = x_flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
        unit = x_flat / norms
        y = unit @ self.H                                     # [N, D]
        y_clip = y.clamp(
            self.shaping_rect[:, 0].unsqueeze(0),
            self.shaping_rect[:, 1].unsqueeze(0),
        )
        # Per-coord quantise: for each dim d, find the centroid in
        # self.centroids[d, :] closest to y_clip[:, d].
        # Vectorise: compute pairwise distance per coord.
        #   dist[n, d, k] = (y_clip[n, d] - centroids[d, k])²
        # For D=128, n_levels up to 256, N up to ~50k this is 1.6GB —
        # do it in chunks over n.
        n_levels = self.centroids.shape[1]
        N = y_clip.shape[0]
        out_y = torch.empty_like(y_clip)
        chunk = max(1, (1 << 28) // (D * n_levels))
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            diff = y_clip[s:e].unsqueeze(-1) - self.centroids.unsqueeze(0)  # [chunk, D, n_levels]
            idx = diff.abs().argmin(dim=-1)                                 # [chunk, D]
            out_y[s:e] = self.centroids.gather(1, idx.T).T                  # [D, chunk] → [chunk, D]

        # Un-rotate via Hadamard (self-inverse for our normalised H).
        unit_hat = out_y @ self.H
        return (unit_hat * norms).reshape(*batch, D)
