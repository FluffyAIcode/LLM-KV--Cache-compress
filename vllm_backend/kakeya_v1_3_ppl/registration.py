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

    # --- 2) Register the backend with AttentionBackendEnum ---
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    backend_enum = getattr(AttentionBackendEnum, backend_enum_name)
    register_backend(
        backend_enum,
        "vllm_backend.kakeya_v1_3_ppl.backend.KakeyaV13PPLAttentionBackend",
    )

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
