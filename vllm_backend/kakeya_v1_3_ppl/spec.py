"""`FullAttentionSpec` subclass declaring a raw-byte cache shape.

vLLM v1's cache is normally `[2, num_blocks, block_size, num_kv_heads,
head_size]`.  Our layout is per-codec-block (not per-token), so we
override `real_page_size_bytes` to report the exact slot budget from
`KakeyaV13PPLConfig`, and the allocator gives us a single contiguous
`uint8` buffer of `[num_blocks, slot_budget_bytes]` that we interpret
ourselves.

The 2× leading dim for K/V goes away because we pack K+V into a
single per-block slot pair in the backend — each layer actually
reserves two contiguous slots per block: one for K, one for V.  The
spec reports this as a single combined slot per block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .config import KakeyaV13PPLConfig


@dataclass
class KakeyaV13PPLAttentionSpec:
    """Minimal stand-in for `vllm.v1.kv_cache_interface.FullAttentionSpec`.

    We wrap the real vLLM spec at registration time so that tests can
    run without a full vLLM install.  The *real* subclass, constructed
    in `registration.py`, inherits from vllm's `FullAttentionSpec` and
    simply copies these fields.
    """

    block_size: int              # = block_size_codec (must equal vLLM's block_size)
    num_kv_heads: int
    head_size: int               # Actual model head_dim
    dtype: str = "kakeya_v1_3_ppl"
    use_mla: bool = False

    # Per-stream config so K and V can have independent bit widths.
    k_config: KakeyaV13PPLConfig = None   # type: ignore[assignment]
    v_config: KakeyaV13PPLConfig = None   # type: ignore[assignment]

    def __post_init__(self):
        if self.k_config is None:
            self.k_config = KakeyaV13PPLConfig.default_for_head_dim(self.head_size)
        if self.v_config is None:
            self.v_config = KakeyaV13PPLConfig(
                head_dim=self.head_size,
                d_eff=self.k_config.d_eff,
                block_size_codec=self.k_config.block_size_codec,
                variance_ratio=self.k_config.variance_ratio,
                k_centers=self.k_config.k_centers,
                bit_width=2,               # V-stream: 2 bits per coord
                outlier_budget_frac=0.0,   # no outlier side-buffer for V
            )
        if self.block_size != self.k_config.block_size_codec:
            raise ValueError(
                f"vLLM block_size {self.block_size} must equal "
                f"codec block_size_codec {self.k_config.block_size_codec} "
                "(see PLAN.md §Block-size alignment)"
            )

    @property
    def slot_budget_bytes(self) -> int:
        """Per-block-per-kv-head bytes.  K + V slots are written to
        the same cache allocation; we report the sum so vLLM's
        allocator knows the total footprint."""
        return (
            self.k_config.slot_size_bytes
            + self.v_config.slot_size_bytes
        )

    @property
    def real_page_size_bytes(self) -> int:
        """What the `FullAttentionSpec` override returns.  Multiplies
        the per-kv-head slot budget by the number of kv-heads, because
        one paged-cache block holds all heads' data."""
        return self.slot_budget_bytes * self.num_kv_heads

    def get_kv_cache_shape(
        self,
        num_blocks: int,
    ) -> tuple[int, ...]:
        """Raw-byte cache shape: `(num_blocks, slot_budget_bytes)` per
        kv-head.  The full allocation per layer is
        `(num_blocks, num_kv_heads, slot_budget_bytes)` in uint8.

        No leading 2 dimension — K and V share the allocation and the
        codec writes into layout-controlled offsets within each slot.
        """
        return (num_blocks, self.num_kv_heads, self.slot_budget_bytes)

    def summary(self) -> dict:
        """Human-readable dump of the spec for logs."""
        k = self.k_config
        v = self.v_config
        bf16_per_token = 2 * self.head_size        # bf16 K + bf16 V
        codec_per_token = (
            k.per_token_equivalent_bytes
            + v.per_token_equivalent_bytes
        )
        return {
            "block_size": self.block_size,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_size,
            "d_eff": k.d_eff,
            "k_centers": k.k_centers,
            "wht_len": k.wht_len,
            "k_bits": k.bit_width,
            "v_bits": v.bit_width,
            "outlier_budget_frac": k.outlier_budget_frac,
            "k_slot_bytes": k.slot_size_bytes,
            "v_slot_bytes": v.slot_size_bytes,
            "combined_slot_bytes_per_kv_head": self.slot_budget_bytes,
            "per_layer_page_bytes":
                self.slot_budget_bytes * self.num_kv_heads,
            "bytes_per_token_bf16": float(bf16_per_token),
            "bytes_per_token_codec": float(codec_per_token),
            "compression_ratio_vs_bf16":
                bf16_per_token / codec_per_token if codec_per_token > 0 else 0.0,
        }
