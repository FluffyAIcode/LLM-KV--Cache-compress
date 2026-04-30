"""Inject KakeyaLatticeMLXCache into DFlash's MLX target-model forward.

DFlash's MLX driver (``dflash.model_mlx.stream_generate``) manages the
target model's KV cache internally. There is no stable kwarg for
"replace target caches with these per-layer objects" across the 2026
DFlash releases. To avoid pinning to a single dflash version we offer
**three injection strategies** ordered by intrusiveness, and pick
whichever the installed dflash exposes:

    Strategy A — kwarg passthrough
        If ``stream_generate`` signature has a ``target_cache`` or
        ``caches`` kwarg, pass our list directly. Clean, zero monkey
        patch. Discovered via ``inspect.signature``.

    Strategy B — ``model.make_cache`` monkey-patch
        mlx-lm models conventionally expose ``.make_cache()`` returning
        the list that ``dflash.stream_generate`` then consumes. We
        override that method for the duration of one generate call via
        a context manager, then restore. Also clean; requires dflash to
        delegate to ``model.make_cache()`` rather than build caches
        inline.

    Strategy C — module-level ``make_prompt_cache`` patch
        Some dflash revisions call ``mlx_lm.models.cache.
        make_prompt_cache(model)`` directly. We wrap that module-level
        function; still reversible via context manager.

The adapter is **feature-detected at engine startup**, cached on the
``_LoadedMLXModel`` instance, and reused per request. If none of the
strategies apply (e.g. dflash API drift), we log a loud warning and
fall back to single-track MLX decode + KakeyaLatticeMLXCache — i.e.
we keep the KV compression benefit but lose the speculative-decode
speedup.

Testability:

  - Strategy detection / strategy-object construction is pure Python
    and tested via mocks in ``tests/test_cache_injection.py``.
  - The live-integration smoke (Apple Silicon only) lives behind
    ``pytest.importorskip("mlx.core")`` + ``importorskip("dflash")``.
"""
from __future__ import annotations

import contextlib
import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator

log = logging.getLogger("kakeya_sidecar_mlx.cache_injection")


class InjectionStrategy(str, Enum):
    KWARG = "kwarg"                       # A
    MODEL_MAKE_CACHE = "model_make_cache" # B
    MODULE_PROMPT_CACHE = "module_prompt_cache"  # C
    FALLBACK_NATIVE_MLX = "fallback_native_mlx"


@dataclass(frozen=True)
class InjectionDecision:
    """Result of feature-detecting the best strategy."""

    strategy: InjectionStrategy
    detail: str = ""


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

_KWARG_NAMES = ("target_cache", "caches", "target_caches", "prompt_cache")


def detect_injection_strategy(
    stream_generate_fn: Callable | None,
    model: Any | None = None,
) -> InjectionDecision:
    """Pick the best injection strategy for the installed dflash.

    Args:
        stream_generate_fn: ``dflash.model_mlx.stream_generate`` or
            compatible callable; may be ``None`` when dflash is not
            installed.
        model: the loaded mlx-lm target model; used to test for
            ``make_cache``.

    Returns:
        :class:`InjectionDecision` naming the chosen strategy.
    """
    if stream_generate_fn is None:
        return InjectionDecision(
            InjectionStrategy.FALLBACK_NATIVE_MLX,
            "dflash not installed; no DFlash speculative decoding path available",
        )

    # Strategy A: kwarg passthrough.
    try:
        sig = inspect.signature(stream_generate_fn)
        for name in _KWARG_NAMES:
            if name in sig.parameters:
                return InjectionDecision(
                    InjectionStrategy.KWARG,
                    f"stream_generate accepts kwarg {name!r}",
                )
    except (TypeError, ValueError):
        pass

    # Strategy B: model.make_cache monkey-patch.
    if model is not None and callable(getattr(model, "make_cache", None)):
        return InjectionDecision(
            InjectionStrategy.MODEL_MAKE_CACHE,
            "model.make_cache is callable; patching it for one generate call",
        )

    # Strategy C: module-level make_prompt_cache patch.
    try:
        import mlx_lm.models.cache as _mlx_lm_cache  # type: ignore  # noqa: F401
        return InjectionDecision(
            InjectionStrategy.MODULE_PROMPT_CACHE,
            "mlx_lm.models.cache.make_prompt_cache present",
        )
    except ImportError:
        pass

    return InjectionDecision(
        InjectionStrategy.FALLBACK_NATIVE_MLX,
        "no compatible dflash injection surface found",
    )


# ---------------------------------------------------------------------------
# Injector — applies the chosen strategy for one generate call
# ---------------------------------------------------------------------------


class KakeyaCacheInjector:
    """Build + inject KakeyaLatticeMLXCache into one DFlash call.

    Example:

        injector = KakeyaCacheInjector(
            model=target_model,
            variant="e8", q_range=38, boundary=0,
            strategy=InjectionStrategy.MODEL_MAKE_CACHE,
            cache_factory=make_kakeya_caches,
        )
        with injector.activate():
            for tok in dflash.stream_generate(target_model, draft, tok, prompt,
                                              **injector.extra_kwargs):
                ...

    ``cache_factory`` is injected to make testing trivial — tests pass a
    stub factory that returns ``["layer0", "layer1", ...]`` strings and
    assert that the injector wires them into the right surface.
    """

    def __init__(
        self,
        model: Any,
        *,
        variant: str = "e8",
        q_range: int = 38,
        boundary: int = 0,
        strategy: InjectionStrategy = InjectionStrategy.FALLBACK_NATIVE_MLX,
        cache_factory: Callable | None = None,
    ) -> None:
        self.model = model
        self.variant = variant
        self.q_range = int(q_range)
        self.boundary = int(boundary)
        self.strategy = strategy
        self._cache_factory = cache_factory or self._default_factory
        self._built_caches: list[Any] | None = None
        self._extra_kwargs: dict[str, Any] = {}

    @staticmethod
    def _default_factory(model, **kw):
        from kakeyalattice_mlx import KakeyaLatticeMLXCache  # noqa: F401
        from kakeyalattice_mlx.kv_cache import make_kakeya_caches
        return make_kakeya_caches(model, **kw)

    def build(self) -> list[Any]:
        self._built_caches = self._cache_factory(
            self.model,
            variant=self.variant,
            q_range=self.q_range,
            boundary=self.boundary,
        )
        return self._built_caches

    @property
    def caches(self) -> list[Any] | None:
        return self._built_caches

    @property
    def extra_kwargs(self) -> dict[str, Any]:
        """kwargs to splice into ``stream_generate(...)`` when Strategy A.

        Strategy A adds ``{kwarg_name: caches}``; other strategies keep
        this empty because they mutate state instead.
        """
        return dict(self._extra_kwargs)

    @contextlib.contextmanager
    def activate(
        self,
        stream_generate_fn: Callable | None = None,
    ) -> Iterator[list[Any] | None]:
        """Context-managed injection. Yields the built cache list.

        Args:
            stream_generate_fn: needed only for Strategy A so we can
                discover which kwarg name to use.
        """
        caches = self.build()
        if self.strategy == InjectionStrategy.FALLBACK_NATIVE_MLX:
            # No DFlash path; the engine will use `caches` directly on
            # the target model.
            yield caches
            return

        if self.strategy == InjectionStrategy.KWARG:
            kwarg = self._resolve_kwarg(stream_generate_fn)
            if kwarg is None:
                log.warning(
                    "KWARG strategy selected but no matching kwarg on "
                    "stream_generate; downgrading to FALLBACK_NATIVE_MLX"
                )
                yield caches
                return
            self._extra_kwargs = {kwarg: caches}
            try:
                yield caches
            finally:
                self._extra_kwargs = {}
            return

        if self.strategy == InjectionStrategy.MODEL_MAKE_CACHE:
            original = getattr(self.model, "make_cache", None)

            def _patched_make_cache(*_a, **_kw):
                return caches

            setattr(self.model, "make_cache", _patched_make_cache)
            try:
                yield caches
            finally:
                if original is not None:
                    setattr(self.model, "make_cache", original)
                else:
                    try:
                        delattr(self.model, "make_cache")
                    except AttributeError:
                        pass
            return

        if self.strategy == InjectionStrategy.MODULE_PROMPT_CACHE:
            import mlx_lm.models.cache as _c  # type: ignore

            original = getattr(_c, "make_prompt_cache", None)

            def _patched_make_prompt_cache(model, *_a, **_kw):
                return caches

            setattr(_c, "make_prompt_cache", _patched_make_prompt_cache)
            try:
                yield caches
            finally:
                if original is not None:
                    setattr(_c, "make_prompt_cache", original)
            return

        # Unknown strategy — be loud, don't silently fall through.
        raise RuntimeError(
            f"unknown InjectionStrategy {self.strategy!r} "
            "— cache_injection.py is out of sync"
        )

    # ----- helpers -----

    @staticmethod
    def _resolve_kwarg(stream_generate_fn: Callable | None) -> str | None:
        if stream_generate_fn is None:
            return None
        try:
            sig = inspect.signature(stream_generate_fn)
            for name in _KWARG_NAMES:
                if name in sig.parameters:
                    return name
        except (TypeError, ValueError):
            return None
        return None


__all__ = [
    "InjectionStrategy",
    "InjectionDecision",
    "detect_injection_strategy",
    "KakeyaCacheInjector",
]
