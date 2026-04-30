"""Routing tests for ``MLXEngine._run_stream``.

The real dflash / mlx-lm generators only exist on Apple Silicon, but
the routing logic inside ``_run_stream`` (DFlash vs native MLX) is
pure control flow. We exercise it by:

1. Hand-constructing a ``_LoadedMLXModel``-shaped object WITHOUT
   calling ``_LoadedMLXModel.__init__`` (which would try to import
   mlx-lm). ``object.__new__`` bypasses init so we can set the
   fields directly.
2. Overriding ``MLXEngine._ensure_loaded`` so no real weights get
   fetched.
3. Stubbing ``_dflash_iter_factory`` via monkeypatch — we control
   exactly what pieces + accept-lengths the fake DFlash emits.
4. For the native-MLX fallback we monkey-patch the deferred import
   of ``mlx_lm.generate.stream_generate`` with a generator stub.
"""
from __future__ import annotations

from typing import Any

import pytest

from kakeya_sidecar_mlx.cache_injection import (
    InjectionDecision,
    InjectionStrategy,
)
from kakeya_sidecar_mlx.engine_mlx import (
    MLXEngine,
    MLXEngineConfig,
    _LoadedMLXModel,
)
from kakeya_sidecar_mlx.model_registry_mlx import resolve_mlx_model


class _FakeTokenizer:
    """Matches the ``apply_chat_template`` method mlx-lm exposes."""

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool):
        parts = []
        for m in messages:
            parts.append(f"{m['role']}: {m['content']}")
        return " | ".join(parts) + " | assistant:"


def _make_fake_lm(dflash: bool, channel) -> _LoadedMLXModel:
    """Build a ``_LoadedMLXModel`` without importing mlx-lm."""
    lm = object.__new__(_LoadedMLXModel)
    lm.model = object()
    lm.tokenizer = _FakeTokenizer()
    lm.profile = None
    lm.channel = channel
    lm.draft_model = object() if dflash else None
    lm._stream_generate = (lambda *a, **kw: iter([])) if dflash else None
    lm._injection_decision = InjectionDecision(
        InjectionStrategy.KWARG if dflash else InjectionStrategy.FALLBACK_NATIVE_MLX,
        "test fixture",
    )
    return lm


# ---------------------------------------------------------------------------
# DFlash path
# ---------------------------------------------------------------------------


def test_dflash_path_aggregates_text_and_acceptance(monkeypatch):
    cfg = MLXEngineConfig(enable_dflash=True)
    engine = MLXEngine(cfg)

    profile, channel = resolve_mlx_model("qwen3-8b@e8-q38")
    lm = _make_fake_lm(dflash=True, channel=channel)
    engine._loaded[profile.short_id] = lm
    # Freeze ensure_loaded so no import happens.
    monkeypatch.setattr(
        engine, "_ensure_loaded", lambda prof, ch: lm,
    )

    # Stub dflash iterator: emit 3 blocks with varying acceptance.
    class _Step:
        def __init__(self, text, al):
            self.text = text
            self.accepted_length = al

    steps = [_Step("Hel", 12), _Step("lo ", 10), _Step("world.", 8)]

    def _fake_factory(lm_arg, prompt, max_tokens, temperature, top_p):
        assert "user:" in prompt   # chat template applied
        def _run(extra_kwargs):
            for s in steps:
                yield s.text, {"acceptance_length": s.accepted_length}
        return _run

    monkeypatch.setattr(engine, "_dflash_iter_factory", _fake_factory)

    # Also stub injector so it doesn't try to build real caches.
    from kakeya_sidecar_mlx import engine_mlx as em

    class _StubInjector:
        def __init__(self, *a, **kw): self.extra_kwargs = {}
        def activate(self, _sg):
            class _Ctx:
                def __enter__(self_): return ["c0", "c1"]
                def __exit__(self_, *exc): return False
            return _Ctx()
        def build(self): return ["c0", "c1"]

    monkeypatch.setattr(em, "KakeyaCacheInjector", _StubInjector)

    text, stats = engine.chat(
        "qwen3-8b@e8-q38",
        [{"role": "user", "content": "hi"}],
        max_tokens=32, temperature=0.0,
    )
    assert text == "Hello world."
    assert stats["dflash_used"] is True
    assert stats["injection_strategy"] == "kwarg"
    assert stats["acceptance_length_mean"] == pytest.approx((12 + 10 + 8) / 3)
    assert stats["variant"] == "e8"
    assert stats["q_range"] == 38


def test_dflash_stream_stops_on_stop_substring(monkeypatch):
    cfg = MLXEngineConfig(enable_dflash=True)
    engine = MLXEngine(cfg)

    profile, channel = resolve_mlx_model("qwen3-8b@e8-q38")
    lm = _make_fake_lm(dflash=True, channel=channel)
    engine._loaded[profile.short_id] = lm
    monkeypatch.setattr(engine, "_ensure_loaded", lambda p, c: lm)

    def _fake_factory(lm_arg, prompt, max_tokens, temperature, top_p):
        def _run(extra_kwargs):
            yield "hello STOP more text", {"acceptance_length": 5}
            yield "should not be yielded", {"acceptance_length": 5}
        return _run

    monkeypatch.setattr(engine, "_dflash_iter_factory", _fake_factory)

    from kakeya_sidecar_mlx import engine_mlx as em

    class _StubInjector:
        def __init__(self, *a, **kw): self.extra_kwargs = {}
        def activate(self, _sg):
            class _Ctx:
                def __enter__(self_): return []
                def __exit__(self_, *exc): return False
            return _Ctx()
        def build(self): return []

    monkeypatch.setattr(em, "KakeyaCacheInjector", _StubInjector)

    pieces = list(engine.chat_stream(
        "qwen3-8b@e8-q38",
        [{"role": "user", "content": "hi"}],
        max_tokens=32,
        stop=["STOP"],
    ))
    assert pieces == ["hello STOP more text"]


# ---------------------------------------------------------------------------
# Native MLX fallback path
# ---------------------------------------------------------------------------


def test_native_mlx_fallback_used_when_dflash_unavailable(monkeypatch):
    """Mistral has no DFlash draft; engine must fall back cleanly."""
    cfg = MLXEngineConfig(enable_dflash=True)        # even with enable=True
    engine = MLXEngine(cfg)

    profile, channel = resolve_mlx_model("mistral-7b-instruct-v0.3@e8-q38")
    assert channel.dflash_available is False

    lm = _make_fake_lm(dflash=False, channel=channel)
    engine._loaded[profile.short_id] = lm
    monkeypatch.setattr(engine, "_ensure_loaded", lambda p, c: lm)

    # Stub the mlx_lm.generate.stream_generate import.
    import sys
    import types as _types

    fake_generate = _types.ModuleType("mlx_lm.generate")

    def _fake_stream(model, tokenizer, *, prompt, max_tokens, temp, top_p, prompt_cache):
        # Yield a couple of tokens then stop.
        yield "Bonjour "
        yield "monde."

    fake_generate.stream_generate = _fake_stream

    # If mlx_lm already imported (unlikely on CI), splice a fake submodule.
    sys.modules["mlx_lm.generate"] = fake_generate
    sys.modules.setdefault("mlx_lm", _types.ModuleType("mlx_lm"))

    from kakeya_sidecar_mlx import engine_mlx as em

    class _StubInjector:
        def __init__(self, *a, **kw): self.extra_kwargs = {}
        def activate(self, _sg):
            class _Ctx:
                def __enter__(self_): return []
                def __exit__(self_, *exc): return False
            return _Ctx()
        def build(self): return []

    monkeypatch.setattr(em, "KakeyaCacheInjector", _StubInjector)

    text, stats = engine.chat(
        "mistral-7b-instruct-v0.3@e8-q38",
        [{"role": "user", "content": "bonjour"}],
        max_tokens=16,
    )
    assert text == "Bonjour monde."
    assert stats["dflash_used"] is False
    assert stats["injection_strategy"] == "fallback_native_mlx"
    assert stats["acceptance_length_mean"] is None


# ---------------------------------------------------------------------------
# Override
# ---------------------------------------------------------------------------


def test_override_applies_per_request(monkeypatch):
    cfg = MLXEngineConfig(enable_dflash=False)
    engine = MLXEngine(cfg)

    profile, channel = resolve_mlx_model("qwen3-8b@e8-q38")
    lm = _make_fake_lm(dflash=False, channel=channel)
    engine._loaded[profile.short_id] = lm
    monkeypatch.setattr(engine, "_ensure_loaded", lambda p, c: lm)

    # Fake native-mlx stream_generate (minimal).
    import sys, types as _types
    fake_generate = _types.ModuleType("mlx_lm.generate")

    def _fake_stream(model, tokenizer, *, prompt, max_tokens, temp, top_p, prompt_cache):
        yield "ok"

    fake_generate.stream_generate = _fake_stream
    sys.modules["mlx_lm.generate"] = fake_generate
    sys.modules.setdefault("mlx_lm", _types.ModuleType("mlx_lm"))

    from kakeya_sidecar_mlx import engine_mlx as em

    captured_q: list[int] = []

    class _StubInjector:
        def __init__(self, *a, variant=None, q_range=None, **kw):
            self.extra_kwargs = {}
            captured_q.append(q_range)
        def activate(self, _sg):
            class _Ctx:
                def __enter__(self_): return []
                def __exit__(self_, *exc): return False
            return _Ctx()
        def build(self): return []

    monkeypatch.setattr(em, "KakeyaCacheInjector", _StubInjector)

    # Default channel q=38; override to q=10.
    _text, stats = engine.chat(
        "qwen3-8b@e8-q38",
        [{"role": "user", "content": "hi"}],
        max_tokens=8,
        override={"q_range": 10},
    )
    assert stats["q_range"] == 10
    assert captured_q and captured_q[-1] == 10
