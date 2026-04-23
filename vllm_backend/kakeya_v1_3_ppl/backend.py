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
    import torch as _torch
    from vllm.v1.attention.backend import AttentionBackend, AttentionType, MultipleOf
    from vllm.config.cache import CacheDType  # noqa: F401
    _HAS_VLLM = True
    _SUPPORTED_MODEL_DTYPES = [_torch.bfloat16, _torch.float16]
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
    _SUPPORTED_MODEL_DTYPES = []


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

    supported_dtypes: ClassVar[list] = _SUPPORTED_MODEL_DTYPES
    supported_kv_cache_dtypes: ClassVar[list] = [KAKEYA_V1_3_PPL_NAME]

    @staticmethod
    def get_name() -> str:
        # Return the AttentionBackendEnum member name under which we
        # registered in `registration.py`.  vLLM round-trips
        # `AttentionBackendEnum[self.attn_backend.get_name()]` in
        # `vllm.model_executor.layers.attention.attention:351`, so
        # this MUST match one of the declared enum members — our
        # slot is `CUSTOM` (reserved for third-party backends).
        # The human-readable name "kakeya_v1_3_ppl" is carried by
        # `kv_cache_dtype` instead.
        return "CUSTOM"

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
    def supports_compute_capability(cls, capability) -> bool:
        """Triton kernels (M4/M5) need SM80+ (Ampere and later).

        vLLM passes a `DeviceCapability` namedtuple with `.major`
        / `.minor` attributes; older versions of this method
        accepted an int SM score (e.g. 80 for Ampere, 90 for
        Hopper).  We accept either for forward compatibility.
        """
        if hasattr(capability, "major") and hasattr(capability, "minor"):
            return capability.major >= 8
        # Integer fallback — treat as SM score like 80, 89, 90.
        return int(capability) >= 80

    @classmethod
    def supports_combination(
        cls,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        device_capability,
        *args,
        **kwargs,
    ) -> str | None:
        """Return None = no objection; str = human-readable reason
        why this combination isn't supported.

        Our constraints are already enforced individually via
        `supports_{head_size, block_size, kv_cache_dtype,
        compute_capability}`; supports_combination is the catch-all
        for cross-dim constraints we don't have.  `*args` / `**kwargs`
        accept any extra fields vLLM passes in future versions
        without forcing a rebuild.
        """
        if use_mla:
            return "KakeyaV13PPLAttentionBackend does not support MLA"
        if use_sparse:
            return "KakeyaV13PPLAttentionBackend does not support sparse"
        return None

    @staticmethod
    def is_mla() -> bool:
        return False

    @staticmethod
    def is_sparse() -> bool:
        return False

    @staticmethod
    def is_ssm() -> bool:
        return False
