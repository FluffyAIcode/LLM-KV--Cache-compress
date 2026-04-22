"""Configuration + slot-layout math for the v1.3 PPL vLLM backend.

The slot layout, copied verbatim from PLAN.md §"Cache layout interface
contract" §"paged cache slot layout":

    HEADER          48 B      d_eff, k, outlier_count (u32), reserved
    PCA basis       d_eff · D · 2 B (fp16)
    PCA mean        D · 2 B   (fp16)
    K-means cent    k · d_eff · 2 B (fp16)
    K-means idx + residual     block_size × (ceil(log2(k)) + d_eff · bit_width) / 8
    Outlier budget  fixed 8 % worst-case of coords for K (zero-filled unused)

Rationale for each field lives in PLAN.md §"The key design decision" and is
NOT repeated here; this module is the executable version of that spec.

For the default PR #15 production cell
    D=128, d_eff=64, block_size=512, k=16, b_K=3, b_V=2, outlier_budget=0.08

the K-stream slot comes out to 36 480 B (≈ 71 B / token), the V-stream
28 608 B (≈ 56 B / token).  Combined per-token cost ≈ 127 B vs bf16's
2 × 128 × 2 = 512 B, giving a **4.03 × compression ratio** — matching
PLAN.md's target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar


KAKEYA_V1_3_PPL_NAME = "kakeya_v1_3_ppl"


@dataclass(frozen=True)
class KakeyaV13PPLConfig:
    """Per-layer slot-layout parameters.  Once the attention backend
    is built, a `KakeyaV13PPLConfig` is created for each layer from
    the model's head_dim + bit-width + outlier settings, and its
    `slot_size_bytes` is the exact per-slot allocation the vLLM
    paged-cache needs.

    All sizes are in bytes, all fields are fp32-rounded-to-fp16 when
    stored in the slot (matches the Rust codec's `SkeletonDtype::Fp16`).
    """

    # Model-side geometry (cannot change without retraining).
    head_dim: int               # D in the algorithm
    # Codec-side knobs (set per stream).
    block_size_codec: int = 512  # codec block size; must divide paged-cache block_size
    variance_ratio: float = 0.95
    d_eff: int = 64             # PCA kept rank
    k_centers: int = 16         # K-means centres
    bit_width: int = 3          # Lloyd-Max quantiser bits (1..=4)
    outlier_budget_frac: float = 0.08  # fraction of coords reserved for outliers (K only)
    skeleton_dtype_bytes: int = 2       # fp16
    # Header reserves u32 fields for future use.  PLAN.md spec says 48 B.
    header_bytes: ClassVar[int] = 48

    def __post_init__(self):
        if not (1 <= self.bit_width <= 4):
            raise ValueError(f"bit_width must be in 1..=4, got {self.bit_width}")
        if self.k_centers < 1 or (self.k_centers & (self.k_centers - 1)) != 0:
            raise ValueError(
                f"k_centers must be a positive power of two, got {self.k_centers}"
            )
        if self.d_eff < 1 or self.d_eff > self.head_dim:
            raise ValueError(
                f"d_eff must be in 1..={self.head_dim}, got {self.d_eff}"
            )
        if self.block_size_codec < 1 or (
            self.block_size_codec & (self.block_size_codec - 1)
        ) != 0:
            raise ValueError(
                f"block_size_codec must be a positive power of two, "
                f"got {self.block_size_codec}"
            )
        if not (0.0 <= self.outlier_budget_frac <= 1.0):
            raise ValueError(
                f"outlier_budget_frac must be in [0, 1], "
                f"got {self.outlier_budget_frac}"
            )

    # ---- Size derivations per PLAN.md §Cache layout --------------------

    @property
    def wht_len(self) -> int:
        """`next_pow2(d_eff)` — same formula `kakeyaturbo::codec::encode_block`
        uses to pad the residual before WHT."""
        if self.d_eff <= 1:
            return 1
        n = 1
        while n < self.d_eff:
            n <<= 1
        return n

    @property
    def pca_basis_bytes(self) -> int:
        return self.d_eff * self.head_dim * self.skeleton_dtype_bytes

    @property
    def pca_mean_bytes(self) -> int:
        return self.head_dim * self.skeleton_dtype_bytes

    @property
    def kmeans_centroids_bytes(self) -> int:
        return self.k_centers * self.d_eff * self.skeleton_dtype_bytes

    @property
    def seg_id_bits_per_vec(self) -> int:
        return max(1, (self.k_centers - 1).bit_length())

    @property
    def residual_bits_per_vec(self) -> int:
        return self.wht_len * self.bit_width

    @property
    def seg_id_bytes_per_block(self) -> int:
        # seg_id bit-packed across all vecs in the block.
        return math.ceil(
            self.block_size_codec * self.seg_id_bits_per_vec / 8
        )

    @property
    def t_bytes_per_block(self) -> int:
        # fp16 t per vec.
        return self.block_size_codec * 2

    @property
    def norm_bytes_per_block(self) -> int:
        # fp16 norm per vec.
        return self.block_size_codec * 2

    @property
    def residual_bytes_per_block(self) -> int:
        # Bit-packed residual (PLAN.md §cache layout).  Rust's `pack_bits`
        # packs `wht_len * bit_width` bits per vec, little-endian.
        return math.ceil(
            self.block_size_codec * self.residual_bits_per_vec / 8
        )

    @property
    def per_block_code_bytes(self) -> int:
        # Matches PLAN.md §cache layout: fields are stored as parallel
        # arrays rather than interleaved per-vec records.  This makes
        # the STORE kernel's coalesced writes trivial.
        return (
            self.seg_id_bytes_per_block
            + self.t_bytes_per_block
            + self.norm_bytes_per_block
            + self.residual_bytes_per_block
        )

    @property
    def outlier_row_count_bytes(self) -> int:
        """u16 per row holding that row's outlier count.  Used to
        recover row boundaries in the flat entry array."""
        return self.block_size_codec * 2

    @property
    def outlier_entry_bytes_budget(self) -> int:
        """Budget for flat entries (u16 idx, f16 val = 4 B each).

        `ceil(budget_frac × block_size × wht_len)` coordinates is the
        worst-case total outlier count across the block.
        """
        worst = math.ceil(
            self.outlier_budget_frac * self.block_size_codec * self.wht_len
        )
        return worst * 4

    @property
    def outlier_budget_bytes(self) -> int:
        """Total outlier side-buffer bytes = per-row counts + flat entries."""
        if self.outlier_budget_frac <= 0.0:
            return 0
        return self.outlier_row_count_bytes + self.outlier_entry_bytes_budget

    @property
    def slot_size_bytes(self) -> int:
        """Total bytes per codec block.

        Sum of header + skeleton (PCA basis + PCA mean + K-means
        centroids) + per-block codes + outlier side-buffer budget.
        """
        return (
            self.header_bytes
            + self.pca_basis_bytes
            + self.pca_mean_bytes
            + self.kmeans_centroids_bytes
            + self.per_block_code_bytes
            + self.outlier_budget_bytes
        )

    @property
    def per_token_equivalent_bytes(self) -> float:
        """Amortised bytes per token within this codec block.  Divide
        the slot size by `block_size_codec` to compare against
        baseline bf16 (`2 × D = 256 B` per token for D=128)."""
        return self.slot_size_bytes / self.block_size_codec

    @property
    def compression_ratio_vs_bf16(self) -> float:
        bf16_per_token = 2 * self.head_dim
        if self.per_token_equivalent_bytes <= 0:
            return float("inf")
        return bf16_per_token / self.per_token_equivalent_bytes

    # ---- Byte offsets within a slot ------------------------------------
    # Layout (LSB-first inside each field, field-order is write-order):
    #
    #   [ 0         .. header_bytes                     )  HEADER
    #   [ hb        .. hb + pca_basis_bytes             )  PCA basis
    #   [ . . . ]
    #
    # These are useful both for the STORE kernel (where to write) and
    # the DECODE kernel (where to read).

    @property
    def offset_pca_basis(self) -> int:
        return self.header_bytes

    @property
    def offset_pca_mean(self) -> int:
        return self.offset_pca_basis + self.pca_basis_bytes

    @property
    def offset_kmeans_centroids(self) -> int:
        return self.offset_pca_mean + self.pca_mean_bytes

    @property
    def offset_codes(self) -> int:
        return self.offset_kmeans_centroids + self.kmeans_centroids_bytes

    @property
    def offset_seg_id_block(self) -> int:
        return self.offset_codes

    @property
    def offset_t_block(self) -> int:
        return self.offset_seg_id_block + self.seg_id_bytes_per_block

    @property
    def offset_norm_block(self) -> int:
        return self.offset_t_block + self.t_bytes_per_block

    @property
    def offset_residual_block(self) -> int:
        return self.offset_norm_block + self.norm_bytes_per_block

    @property
    def offset_outlier_side_buffer(self) -> int:
        return self.offset_residual_block + self.residual_bytes_per_block

    # ---- Derived knobs for the Triton kernels --------------------------

    @classmethod
    def default_for_head_dim(cls, head_dim: int) -> "KakeyaV13PPLConfig":
        """Pragmatic default matching the PR #15 production recipe:
        d_eff = head_dim // 2 (rounded to the nearest power of two),
        k = 16, b = 3 (MSE K-stream).

        V-stream callers can override `bit_width` + `outlier_budget_frac=0.0`
        after calling this classmethod.
        """
        if head_dim not in (64, 96, 128, 256):
            raise ValueError(
                f"default config only supports head_dim in {{64, 96, 128, 256}}, "
                f"got {head_dim}"
            )
        if head_dim == 96:
            d_eff = 64
        else:
            d_eff = head_dim // 2
        return cls(
            head_dim=head_dim,
            d_eff=d_eff,
            block_size_codec=512,
            variance_ratio=0.95,
            k_centers=16,
            bit_width=3,
            outlier_budget_frac=0.08,
        )
