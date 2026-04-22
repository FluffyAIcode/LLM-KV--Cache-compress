"""`AttentionBackend` subclass for the v1.3 PPL codec.

This file is a *minimal* vLLM AttentionBackend that declares:

  * `get_name()`                  — "KAKEYA_V1_3_PPL"
  * `supported_kv_cache_dtypes`   — ["kakeya_v1_3_ppl"]
  * `get_kv_cache_shape()`        — raw-byte shape from `KakeyaV13PPLAttentionSpec`
  * `get_impl_cls()`              — `KakeyaV13PPLAttentionImpl`
  * `get_builder_cls()`           — metadata builder (stub in Phase A)

The real attention work (store, decode, partial-block routing) lives
in `impl.py`; this file is the glue layer the vLLM engine looks up.

Because vLLM's AttentionBackend is defined in `vllm.v1.attention.backend`
— which pulls in `vllm._C` and therefore libcudart — we guard the
import with a try/except so the module remains importable on CPU-only
dev machines for unit testing.
"""
from __future__ import annotations

from typing import ClassVar

try:
    from vllm.v1.attention.backend import AttentionBackend, AttentionType, MultipleOf
    from vllm.config.cache import CacheDType  # noqa: F401
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False
    # Minimal stand-in for type-checking / tests.
    class AttentionBackend:  # type: ignore[no-redef]
        pass
    class AttentionType:  # type: ignore[no-redef]
        DECODER = "decoder"
    class MultipleOf:  # type: ignore[no-redef]
        def __init__(self, n: int):
            self.n = n


from .config import KAKEYA_V1_3_PPL_NAME, KakeyaV13PPLConfig


class KakeyaV13PPLAttentionBackend(AttentionBackend):
    """vLLM AttentionBackend for the v1.3 PPL codec.

    Cache layout (per layer):
        uint8[num_blocks, num_kv_heads, slot_budget_bytes]
        where slot_budget_bytes is the sum of K-stream + V-stream
        slots per `KakeyaV13PPLAttentionSpec.slot_budget_bytes`.
    """

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list] = []   # populated at registration
    supported_kv_cache_dtypes: ClassVar[list] = [KAKEYA_V1_3_PPL_NAME]

    @staticmethod
    def get_name() -> str:
        return "KAKEYA_V1_3_PPL"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list:
        # PLAN.md §"Block-size alignment": codec block must equal
        # vLLM's block_size, fixed at 512 in the PR #15 recipe.
        return [512]

    @staticmethod
    def get_preferred_block_size() -> int:
        return 512

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = KAKEYA_V1_3_PPL_NAME,
    ) -> tuple[int, ...]:
        """Raw uint8 cache shape.

        (num_blocks, num_kv_heads, slot_budget_bytes)

        NOT (2, num_blocks, ...) — K/V share the per-slot allocation.
        """
        if cache_dtype_str != KAKEYA_V1_3_PPL_NAME:
            raise ValueError(
                f"KakeyaV13PPLAttentionBackend only supports "
                f"cache_dtype={KAKEYA_V1_3_PPL_NAME!r}, got {cache_dtype_str!r}"
            )
        from .spec import KakeyaV13PPLAttentionSpec
        spec = KakeyaV13PPLAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        return (num_blocks,) + spec.get_kv_cache_shape(num_blocks)[1:]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        return kv_cache_dtype == KAKEYA_V1_3_PPL_NAME

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (64, 96, 128, 256)

    @classmethod
    def supports_block_size(cls, block_size) -> bool:
        return block_size == 512

    @staticmethod
    def get_impl_cls():
        from .impl import KakeyaV13PPLAttentionImpl
        return KakeyaV13PPLAttentionImpl

    @staticmethod
    def get_builder_cls():
        from .impl import KakeyaV13PPLMetadataBuilder
        return KakeyaV13PPLMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, compute_cap: int) -> bool:
        # Triton kernel needs SM80+ (Ampere and later).
        return compute_cap >= 80

    @classmethod
    def supports_combination(cls, **kwargs) -> bool:
        return True

    @staticmethod
    def is_mla() -> bool:
        return False

    @staticmethod
    def is_sparse() -> bool:
        return False

    @staticmethod
    def is_ssm() -> bool:
        return False
