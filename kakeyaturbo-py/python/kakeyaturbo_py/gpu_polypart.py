"""Phase 3: tree-structured K-means (Guth-Katz polynomial-partitioning
prototype) for K-stream quantisation.

Simplification of the full Guth-Katz polynomial-partitioning idea:
full Guth-Katz requires solving a degree-d polynomial p in R^D such
that the zero set {p = 0} balance-partitions the point cloud into
≤ d^D cells.  At D=128 this is computationally infeasible (no closed-
form solver, O(D^d) memory for coefficients).

This prototype takes the **shape** of Guth-Katz — recursive hyperplane
partitioning — and implements the simplest constructive version:

    Binary tree K-means with PCA-axis splits.

At each tree node:
  * Points at this node = some subset of coeff vectors
  * Compute top-1 PCA direction u (i.e. **degree-1 polynomial**
    p(x) = <u, x>; its zero set is a hyperplane through 0)
  * Split points by sign of <u, x>
  * Recurse until tree depth log2(k) = 6 for k=64

This gives:
  * A tree of k = 2^depth leaves, each a cell of the partition.
  * Encoding: traverse tree, store path as log2(k) bits = same as
    flat k-means seg_id.
  * Decoding: average of points in leaf = centroid.

How this maps to Guth-Katz conceptually:
  * Guth-Katz: one degree-d polynomial, d^D cells
  * This prototype: D degree-1 polynomials (one per tree level),
    2^depth cells — same total cell count asymptotically, GPU-
    friendly construction.
  * Loses the "balanced cell count" guarantee of Guth-Katz, but
    PCA-axis split is locally greedy-balancing (maximises variance
    explained at each split).

What this EXPLICITLY doesn't do:
  * Sticky tube analysis (Wang-Zahl Technique C) — snapped up this
    round's scope; would be another follow-up.
  * Multi-scale induction across tree levels — the tree IS multi-
    scale but budget allocation is uniform per level.

Snapshot-only integration, analogous to snapF's RVQ.  Bypasses the
vLLM slot layout entirely.

The construction lives in a single function:
  `fit_skeleton_polypart_tree_batched` — drops into the same call
  slot as `fit_skeleton_batched` and `fit_skeleton_rvq_batched`.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .gpu_skeleton import (
    _fit_rsvd_batched,
    _fit_kmeans_batched,
    _next_pow2,
    _fp16_through,
)


def _tree_kmeans_split_batched(
    coeff: torch.Tensor,
    depth: int,
    kmeans_max_iter: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recursively split coeff along top-1 PCA direction per node.

    For each node at tree depth d ∈ [0, depth), the points at that
    node are split by sign of their projection on the node's local
    top-1 PCA direction.

    Args:
        coeff: [B, n, d_eff] fp32 on device.  Points to partition.
        depth: tree depth; number of leaves = 2^depth.
        kmeans_max_iter: Lloyd refinement iterations per leaf after
                         hyperplane split.  Refines the centroid
                         of each leaf from "cluster mean" to "1-
                         centroid K-means" = same thing, so this
                         actually doesn't do much; kept for API
                         compatibility with _fit_kmeans_batched.
        seed: for reproducibility.

    Returns:
        seg_id:   [B, n] int64    leaf index ∈ [0, 2^depth)
        centres:  [B, k, d_eff]   leaf centroids, k = 2^depth

    Invariant: each leaf's centroid is the MEAN of points assigned to
    it, unit-normalised (to match spherical K-means convention).
    """
    B, n, d_eff = coeff.shape
    k = 1 << depth                                       # 2^depth leaves
    eps = torch.finfo(coeff.dtype).eps
    device = coeff.device

    # Per-point leaf index, initialised to 0 (root).
    seg_id = torch.zeros(B, n, dtype=torch.int64, device=device)

    # For each level 0..depth-1, for each current node, split by top-1
    # PCA axis of points in that node.  Parallelise across batch b and
    # across nodes at the same level.
    for level in range(depth):
        n_nodes = 1 << level                             # number of nodes this level
        # Collect PCA direction for each (b, node).  We can parallelise
        # across nodes by using a [B, n_nodes, d_eff] tensor of directions.
        new_seg = seg_id.clone()
        for node in range(n_nodes):
            mask = (seg_id == node)                       # [B, n]
            # For each batch, compute PCA of points in this node.
            # We do it per batch in a loop — n_nodes grows as 2^level,
            # which means deeper levels have more nodes but fewer
            # points each; the total work across all levels is O(depth·B·n·d_eff)
            # = O(B·n·d_eff·log k), same as flat K-means but with more
            # overhead.  For depth=6 this is ~6x the flat fit cost, but
            # still microseconds for block_size=512.
            batch_dirs = torch.zeros(B, d_eff, device=device, dtype=coeff.dtype)
            for b in range(B):
                bm = mask[b]                              # [n]
                if not bm.any():
                    # Degenerate node (no points); keep direction = e_0,
                    # children inherit parent (neither left nor right)
                    batch_dirs[b] = torch.zeros(d_eff, device=device)
                    batch_dirs[b, 0] = 1.0
                    continue
                pts = coeff[b, bm, :]                      # [n_b, d_eff]
                if pts.shape[0] < 2:
                    batch_dirs[b] = torch.zeros(d_eff, device=device)
                    batch_dirs[b, 0] = 1.0
                    continue
                pts_centred = pts - pts.mean(dim=0, keepdim=True)
                # Top-1 PCA: right singular vector of centred.
                # Use svd_lowrank for speed (O(n·d·r) with r=1).
                try:
                    _, _, V = torch.svd_lowrank(pts_centred, q=1)
                    batch_dirs[b] = V[:, 0]
                except Exception:
                    # Fallback to power iteration on small nodes.
                    v = torch.randn(d_eff, device=device, generator=None)
                    v = v / v.norm().clamp(min=eps)
                    for _ in range(10):
                        v = pts_centred.t() @ (pts_centred @ v)
                        v = v / v.norm().clamp(min=eps)
                    batch_dirs[b] = v

            # Split by sign of <coeff, dir>.  batch_dirs: [B, d_eff].
            # Use einsum for batched projection.
            proj = torch.einsum("bnd,bd->bn", coeff, batch_dirs)  # [B, n]
            # Left child (proj >= 0): new_seg = 2·seg_id + 0
            # Right child (proj < 0): new_seg = 2·seg_id + 1
            # We only apply this split to the mask = (seg_id == node).
            node_mask = (seg_id == node)
            left_mask = node_mask & (proj >= 0)
            right_mask = node_mask & (proj < 0)
            new_seg = torch.where(left_mask, 2 * seg_id, new_seg)
            new_seg = torch.where(right_mask, 2 * seg_id + 1, new_seg)
        seg_id = new_seg

    # Compute leaf centroids = mean of points per leaf, unit-normalised.
    # centres[b, l, :] = mean of coeff[b, seg_id == l, :]
    centres = torch.zeros(B, k, d_eff, device=device, dtype=coeff.dtype)
    for b in range(B):
        for l in range(k):
            mask = seg_id[b] == l                          # [n]
            if mask.any():
                centres[b, l] = coeff[b, mask].mean(dim=0)
    # Unit-normalise to match spherical K-means convention.
    norms = centres.norm(dim=2, keepdim=True).clamp(min=eps)
    centres = centres / norms
    return seg_id, _fp16_through(centres)


def fit_skeleton_polypart_tree_batched(
    X: torch.Tensor,
    *,
    d_eff: int,
    k: int,
    seed: int = 3405691582,
    kmeans_max_iter: int = 8,
    rsvd_oversample: int = 8,
    rsvd_power_iters: int = 2,
    variance_ratio: float = 1.0,
) -> dict:
    """Stage-1 skeleton fit with **polynomial-partitioning tree** instead
    of flat spherical K-means.

    Same interface as `fit_skeleton_batched` so it drops into the
    snapshot harness's codec_layer → gpu_roundtrip → _gpu_codec_per_head
    call chain.

    k must be a power of 2 (number of tree leaves).
    """
    assert X.dim() == 3, f"X must be [B, n, d], got {X.shape}"
    _B, _n, d = X.shape
    if d_eff < 1 or d_eff > d:
        raise ValueError(f"d_eff {d_eff} out of range [1, {d}]")
    if k < 1 or (k & (k - 1)) != 0:
        raise ValueError(f"k must be positive power of 2, got {k}")
    depth = int(math.log2(k))

    # PCA skeleton (identical to fit_skeleton_batched).
    mean, basis, _sigma, d_eff_out = _fit_rsvd_batched(
        X, target_rank=d_eff,
        oversample=rsvd_oversample, power_iters=rsvd_power_iters,
        variance_ratio=variance_ratio, rotation_seed=seed,
    )
    coeff = (X - mean.unsqueeze(1)) @ basis.transpose(1, 2)
    if d_eff_out < d_eff:
        pad = d_eff - d_eff_out
        basis = torch.cat(
            [basis, torch.zeros(_B, pad, d, device=basis.device, dtype=basis.dtype)],
            dim=1,
        )
        coeff = torch.cat(
            [coeff, torch.zeros(_B, _n, pad, device=coeff.device, dtype=coeff.dtype)],
            dim=2,
        )

    # Tree-structured K-means with PCA-axis splits.
    _seg_id, centres = _tree_kmeans_split_batched(
        coeff, depth=depth,
        kmeans_max_iter=kmeans_max_iter, seed=seed,
    )

    return {
        "mean":          mean,
        "basis":         basis,
        "centers":       centres,
        "d":             d,
        "d_eff":         d_eff,
        "k":             k,
        "rotation_seed": int(seed),
        "wht_len":       _next_pow2(d_eff),
        "partitioning":  "polypart_tree_deg1",
    }
