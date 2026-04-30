"""Unit tests for ``cache_injection`` — no MLX / no dflash required.

We feed the strategy detector synthetic stream_generate signatures
and stub target models, and verify:

* KWARG strategy fires when the callable exposes one of the
  well-known kwarg names
* MODEL_MAKE_CACHE strategy fires when the target has ``make_cache``
* FALLBACK path when everything else is missing
* ``activate()`` context manager applies + cleans up each strategy
  correctly (no residual state leaks after exit)
"""
from __future__ import annotations

import pytest

from kakeya_sidecar_mlx.cache_injection import (
    InjectionStrategy,
    KakeyaCacheInjector,
    detect_injection_strategy,
)


# ---------------------------------------------------------------------------
# Strategy detection
# ---------------------------------------------------------------------------


def test_detect_no_fn_is_fallback() -> None:
    d = detect_injection_strategy(None, model=object())
    assert d.strategy == InjectionStrategy.FALLBACK_NATIVE_MLX


def test_detect_kwarg_target_cache() -> None:
    def fake_stream(model, draft, tok, prompt, *, target_cache=None):
        ...

    d = detect_injection_strategy(fake_stream, model=None)
    assert d.strategy == InjectionStrategy.KWARG
    assert "target_cache" in d.detail


def test_detect_kwarg_caches() -> None:
    def fake_stream(model, draft, tok, prompt, caches=None):
        ...

    d = detect_injection_strategy(fake_stream, model=None)
    assert d.strategy == InjectionStrategy.KWARG


def test_detect_kwarg_prompt_cache() -> None:
    def fake_stream(model, draft, tok, prompt, prompt_cache=None):
        ...

    d = detect_injection_strategy(fake_stream, model=None)
    assert d.strategy == InjectionStrategy.KWARG


def test_detect_model_make_cache() -> None:
    def fake_stream(model, draft, tok, prompt):     # no matching kwarg
        ...

    class _M:
        def make_cache(self):
            return []

    d = detect_injection_strategy(fake_stream, model=_M())
    assert d.strategy == InjectionStrategy.MODEL_MAKE_CACHE


# ---------------------------------------------------------------------------
# Injector.activate() — state management
# ---------------------------------------------------------------------------


def test_fallback_activate_yields_caches_and_no_patch() -> None:
    model = object()
    inj = KakeyaCacheInjector(
        model=model,
        strategy=InjectionStrategy.FALLBACK_NATIVE_MLX,
        cache_factory=lambda m, **_kw: ["layer0", "layer1"],
    )
    with inj.activate() as caches:
        assert caches == ["layer0", "layer1"]
        assert inj.extra_kwargs == {}


def test_kwarg_activate_sets_extra_kwargs_and_cleans_up() -> None:
    def fake_stream(m, d, t, p, *, target_cache=None):
        ...

    inj = KakeyaCacheInjector(
        model=object(),
        strategy=InjectionStrategy.KWARG,
        cache_factory=lambda m, **_kw: ["C0", "C1"],
    )
    with inj.activate(fake_stream) as caches:
        assert caches == ["C0", "C1"]
        assert inj.extra_kwargs == {"target_cache": ["C0", "C1"]}
    assert inj.extra_kwargs == {}       # cleaned up


def test_kwarg_strategy_with_unresolved_kwarg_downgrades_silently() -> None:
    def no_matching_kwarg(m, d, t, p):
        ...

    inj = KakeyaCacheInjector(
        model=object(),
        strategy=InjectionStrategy.KWARG,
        cache_factory=lambda m, **_kw: ["X"],
    )
    with inj.activate(no_matching_kwarg) as caches:
        assert caches == ["X"]
        assert inj.extra_kwargs == {}   # downgrade — no kwarg injected


def test_model_make_cache_patch_restores_original() -> None:
    class _M:
        def __init__(self):
            self.original_called = 0

        def make_cache(self):
            self.original_called += 1
            return ["ORIG"]

    m = _M()
    inj = KakeyaCacheInjector(
        model=m,
        strategy=InjectionStrategy.MODEL_MAKE_CACHE,
        cache_factory=lambda _m, **_kw: ["PATCHED"],
    )
    # Before activate, calling make_cache returns original.
    assert m.make_cache() == ["ORIG"]
    assert m.original_called == 1

    with inj.activate(None) as caches:
        assert caches == ["PATCHED"]
        # Inside activate, make_cache returns the injected caches.
        assert m.make_cache() == ["PATCHED"]

    # After activate, original behaviour restored.
    assert m.make_cache() == ["ORIG"]
    assert m.original_called == 2


def test_unknown_strategy_raises() -> None:
    inj = KakeyaCacheInjector(
        model=object(),
        strategy="not_a_real_strategy",           # type: ignore[arg-type]
        cache_factory=lambda _m, **_kw: [],
    )
    with pytest.raises(RuntimeError):
        with inj.activate(None):
            pass


def test_build_caches_passes_config_through_to_factory() -> None:
    captured: dict = {}

    def _factory(model, **kw):
        captured.update(kw)
        return ["ok"]

    inj = KakeyaCacheInjector(
        model=object(),
        variant="e8", q_range=10, boundary=2,
        strategy=InjectionStrategy.FALLBACK_NATIVE_MLX,
        cache_factory=_factory,
    )
    inj.build()
    assert captured == {"variant": "e8", "q_range": 10, "boundary": 2}
