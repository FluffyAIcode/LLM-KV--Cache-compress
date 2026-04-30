"""Tests for ``KakeyaLatticeMLXCache`` — pure routing / delegation logic,
no MLX required.

We verify the wrapper:
- calls ``codec.roundtrip`` on both K and V before delegating
- forwards non-overridden attributes to the inner cache
- skips codec on exception and falls back to inner
- exposes the fire/skip counters
"""
from __future__ import annotations

from kakeyalattice_mlx.kv_cache import KakeyaLatticeMLXCache


class _MockInner:
    def __init__(self):
        self.offset = 0
        self.state = None
        self.updates: list = []

    def update_and_fetch(self, keys, values):
        self.updates.append((keys, values))
        return keys, values

    def some_new_attr(self) -> str:
        return "forwarded"


class _IdentityCodec:
    name = "identity-test-codec"

    def __init__(self):
        self.calls = 0

    def roundtrip(self, x):
        self.calls += 1
        return x


class _BrokenCodec:
    name = "broken-test-codec"

    def roundtrip(self, x):
        raise RuntimeError("boom")


def test_happy_path_fires_codec_twice_per_update() -> None:
    inner = _MockInner()
    codec = _IdentityCodec()
    cache = KakeyaLatticeMLXCache(inner=inner, codec=codec)

    cache.update_and_fetch("K_new", "V_new")
    cache.update_and_fetch("K_new2", "V_new2")

    assert codec.calls == 4          # 2 updates × (K, V)
    assert cache.fire_count == 2
    assert cache.skip_count == 0
    assert len(inner.updates) == 2


def test_broken_codec_falls_back_to_inner() -> None:
    inner = _MockInner()
    codec = _BrokenCodec()
    cache = KakeyaLatticeMLXCache(inner=inner, codec=codec)

    cache.update_and_fetch("K", "V")
    assert cache.fire_count == 0
    assert cache.skip_count == 1
    assert inner.updates == [("K", "V")]


def test_attribute_forwarding_to_inner() -> None:
    inner = _MockInner()
    cache = KakeyaLatticeMLXCache(inner=inner, codec=_IdentityCodec())
    assert cache.some_new_attr() == "forwarded"
    assert cache.offset == 0


def test_state_property_mirrors_inner() -> None:
    inner = _MockInner()
    cache = KakeyaLatticeMLXCache(inner=inner, codec=_IdentityCodec())
    cache.state = {"foo": 1}
    assert inner.state == {"foo": 1}
    assert cache.state == {"foo": 1}


def test_repr_includes_fire_counter() -> None:
    cache = KakeyaLatticeMLXCache(inner=_MockInner(), codec=_IdentityCodec())
    cache.update_and_fetch("K", "V")
    r = repr(cache)
    assert "KakeyaLatticeMLXCache" in r
    assert "fired=1" in r
