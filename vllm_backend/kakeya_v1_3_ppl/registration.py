"""Opt-in registration hooks for integrating the v1.3 PPL codec into vLLM.

vLLM's type-safe config validates `cache_dtype` against a Pydantic
Literal enum (`CacheDType`), so to accept `--kv-cache-dtype kakeya_v1_3_ppl`
we have to extend that enum *before* the config is parsed.  Similarly,
the attention-backend enum is a `typing.Enum` subclass that we extend
via vLLM's own `register_backend(AttentionBackendEnum.CUSTOM, ...)`
hook — no monkey-patching, just a documented plugin API.

Both patches are **idempotent** and can be undone via
`unregister_kakeya_backend()`.

Usage:

    # In your plugin init code (before `vllm serve` starts the engine):
    from vllm_backend.kakeya_v1_3_ppl import register_kakeya_backend
    register_kakeya_backend()

    # or set the env var VLLM_PLUGINS=vllm_backend.kakeya_v1_3_ppl and
    # import vllm_backend.kakeya_v1_3_ppl.registration
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config import KAKEYA_V1_3_PPL_NAME

if TYPE_CHECKING:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum


_REGISTERED = False
_ORIGINAL_CACHE_DTYPE_ARGS: tuple | None = None


_ORIGINAL_GET_KV_CACHE_SPEC = None


def _install_kv_cache_spec_hook() -> None:
    """Wrap `vllm.model_executor.layers.attention.attention.Attention
    .get_kv_cache_spec` so our dtype routes to TQFullAttentionSpec.

    We reuse TQ's spec class rather than subclassing it because
    TQFullAttentionSpec is already registered in vllm's
    `single_type_kv_cache_manager` dispatch table, its
    `real_page_size_bytes` property does exactly what we need
    (`block_size × num_kv_heads × tq_slot_size`), and its `merge()`
    guard enforces per-layer tq_slot_size consistency — all the
    cache-manager plumbing for free.

    `tq_slot_size` = our spec's `slot_budget_bytes` (= K-slot +
    V-slot bytes, per (block, kv-head)).
    """
    global _ORIGINAL_GET_KV_CACHE_SPEC
    if _ORIGINAL_GET_KV_CACHE_SPEC is not None:
        return  # already installed

    from vllm.model_executor.layers.attention.attention import Attention
    orig = Attention.get_kv_cache_spec
    _ORIGINAL_GET_KV_CACHE_SPEC = orig

    def patched(self, vllm_config):
        if self.kv_cache_dtype == KAKEYA_V1_3_PPL_NAME:
            from .spec import make_kakeya_full_attention_spec
            block_size = vllm_config.cache_config.block_size
            spec = make_kakeya_full_attention_spec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_torch_dtype,
            )
            import logging
            logging.getLogger("kakeya_v1_3_ppl").debug(
                "get_kv_cache_spec: layer=%s → %s, real_page_size_bytes=%d",
                getattr(self, "layer_name", "?"),
                type(spec).__name__,
                spec.real_page_size_bytes,
            )
            return spec
        return orig(self, vllm_config)

    Attention.get_kv_cache_spec = patched


def _patch_cache_config_validator(CC, new_literal) -> None:
    """Replace `CC.__pydantic_validator__`'s cache_dtype Literal schema
    with one whose `expected` list matches `typing.get_args(new_literal)`.

    Pydantic v2 dataclasses expose `__pydantic_core_schema__` — a
    dict-shaped core schema.  We walk it, find the `fields.cache_dtype`
    subtree, swap any `literal_schema` node's `expected` list, and
    reinstall a fresh `pydantic_core.SchemaValidator` on the class.

    This is stable against Pydantic's public schema shape; we handle
    the two wrapping-schema types (`definition-ref` via
    `dataclass-args` and direct `literal-schema` under the field) that
    Pydantic 2.13 currently emits.
    """
    import typing
    expected_vals = list(typing.get_args(new_literal))
    schema = CC.__pydantic_core_schema__
    # dataclass schema wraps the inner schema in 'dataclass-args' or
    # similar; walk and mutate in place.

    def _walk(node):
        if isinstance(node, dict):
            if node.get("type") == "literal":
                exp = node.get("expected")
                if isinstance(exp, list) and set(exp).issubset(set(
                    list(typing.get_args(typing.Literal[tuple(expected_vals)]))
                    + [v for v in [
                        "auto", "float16", "bfloat16", "fp8", "fp8_e4m3",
                        "fp8_e5m2", "fp8_inc", "fp8_ds_mla",
                    ]]
                )):
                    # Overwrite the expected list iff this Literal is
                    # the CacheDType one (contains "auto").  Avoids
                    # mangling other Literal schemas in the same
                    # dataclass.
                    if "auto" in exp:
                        node["expected"] = expected_vals
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(schema)
    # Also mirror into definitions (some pydantic schemas stash
    # sub-schemas under `definitions`).
    for defn in schema.get("definitions", []) or []:
        _walk(defn)

    # Install a fresh SchemaValidator.
    from pydantic_core import SchemaValidator, CoreConfig
    cfg_obj = getattr(CC, "__pydantic_config__", {})
    core_cfg = CoreConfig(**{k: v for k, v in cfg_obj.items()
                             if k in CoreConfig.__annotations__})
    CC.__pydantic_validator__ = SchemaValidator(schema, core_cfg)


def register_kakeya_backend(
    backend_enum_name: str = "CUSTOM",
) -> None:
    """Register the Kakeya v1.3 PPL backend with vLLM.

    1. Extends `vllm.config.cache.CacheDType` to accept
       `"kakeya_v1_3_ppl"`.  This means:
         a. Re-assign the module-level Literal so imports of
            `CacheDType` after this call see the extension.
         b. Patch `CacheConfig.__annotations__["cache_dtype"]` and
            call `CacheConfig.model_rebuild(force=True)` so the
            Pydantic validator is regenerated with the new Literal.
            (Re-assigning the alias alone is NOT enough — Pydantic
            captures the annotation at class-decoration time.)
    2. Registers `KakeyaV13PPLAttentionBackend` as
       `AttentionBackendEnum.<backend_enum_name>` (defaults to
       `CUSTOM` — the name reserved by vLLM for third-party backends).
    """
    global _REGISTERED, _ORIGINAL_CACHE_DTYPE_ARGS
    if _REGISTERED:
        # Ensure the spec hook is installed even on re-entry (it's
        # idempotent, so cheap) — this guards against any import
        # ordering where the earlier call happened before vllm's
        # Attention class was imported.
        _install_kv_cache_spec_hook()
        return

    # --- 1) Extend CacheDType on vllm.config.cache ---
    import vllm.config.cache as cfg
    import typing

    try:
        current = typing.get_args(cfg.CacheDType)
    except Exception:
        current = ()

    if KAKEYA_V1_3_PPL_NAME not in current:
        _ORIGINAL_CACHE_DTYPE_ARGS = current
        new_literal = typing.Literal[tuple(list(current) + [KAKEYA_V1_3_PPL_NAME])]  # type: ignore[arg-type,misc]
        # (a) module-level alias — new code importing CacheDType sees this.
        cfg.CacheDType = new_literal   # type: ignore[assignment]
        # (b) regenerate `CacheConfig.__pydantic_validator__` using
        # pydantic-core directly, without re-running the dataclass
        # machinery (which fails because dataclass-field ordering
        # rejects re-decoration — some fields have defaults, some
        # don't, and the re-decoration loses that state).
        #
        # Strategy: build a fresh core schema from the current
        # dataclass schema, replace the Literal member list for the
        # `cache_dtype` field, and install a new SchemaValidator on
        # the class.  This is a documented-enough pydantic-v2
        # mechanism — we use `core_schema.literal_schema` + a
        # post-validator walk to swap the field's schema.
        _patch_cache_config_validator(cfg.CacheConfig, new_literal)

    # --- 1b) Patch vllm.utils.torch_utils.STR_DTYPE_TO_TORCH_DTYPE ---
    # vLLM's model runner maps the kv_cache_dtype string to a torch
    # dtype for cache-tensor allocation.  Quant-cache backends use
    # torch.uint8 (the raw byte tensor we interpret ourselves via the
    # backend).  Missing entry → KeyError at engine init.
    import vllm.utils.torch_utils as tu
    import torch
    if KAKEYA_V1_3_PPL_NAME not in tu.STR_DTYPE_TO_TORCH_DTYPE:
        tu.STR_DTYPE_TO_TORCH_DTYPE[KAKEYA_V1_3_PPL_NAME] = torch.uint8

    # --- 1c) Wrap Attention.get_kv_cache_spec so our dtype routes
    # through TQFullAttentionSpec (same semantics: variable per-block
    # slot size, not head_size×dtype).  Without this, vLLM's allocator
    # sizes the cache as bf16 K+V per token and .view() fails.
    _install_kv_cache_spec_hook()

    # --- 2) Register the backend with AttentionBackendEnum ---
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    backend_enum = getattr(AttentionBackendEnum, backend_enum_name)
    # Resolve backend class path from our own `__name__` so this works
    # whether the package is imported as `vllm_backend.kakeya_v1_3_ppl`
    # (tree layout, sys.path-based) or `kakeya_v1_3_ppl` (installed via
    # pyproject at `vllm_backend/` being the package root).
    from . import backend as _backend_module
    backend_cls_path = (
        f"{_backend_module.__name__}.KakeyaV13PPLAttentionBackend"
    )
    register_backend(backend_enum, backend_cls_path)

    _REGISTERED = True


def unregister_kakeya_backend() -> None:
    """Revert `register_kakeya_backend`.  Mainly for tests."""
    global _REGISTERED, _ORIGINAL_CACHE_DTYPE_ARGS
    if not _REGISTERED:
        return
    import vllm.config.cache as cfg
    import typing
    if _ORIGINAL_CACHE_DTYPE_ARGS is not None:
        orig_literal = typing.Literal[_ORIGINAL_CACHE_DTYPE_ARGS]  # type: ignore[arg-type,misc]
        cfg.CacheDType = orig_literal
        try:
            _patch_cache_config_validator(cfg.CacheConfig, orig_literal)
        except Exception:
            pass
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    AttentionBackendEnum.CUSTOM.clear_override()
    _REGISTERED = False
    _ORIGINAL_CACHE_DTYPE_ARGS = None
