"""Pure-logic tests for model_registry — no torch / HF downloads."""
from __future__ import annotations

import pytest

from kakeya_sidecar.model_registry import (
    MODEL_REGISTRY,
    resolve_model,
    iter_model_channel_ids,
)


def test_registry_non_empty_and_covers_v15_4models() -> None:
    for required in (
        "qwen3-4b",
        "gemma-4-e4b",
        "glm-4-9b-chat",
        "deepseek-r1-distill-qwen-1.5b",
    ):
        assert required in MODEL_REGISTRY, required


def test_every_profile_has_default_channel_in_channels() -> None:
    for short, prof in MODEL_REGISTRY.items():
        assert prof.default_channel in prof.channels, (
            f"{short}: default_channel not in channels"
        )


def test_deepseek_small_always_has_boundary() -> None:
    prof = MODEL_REGISTRY["deepseek-r1-distill-qwen-1.5b"]
    for ch in prof.channels:
        assert ch.boundary >= 2, (
            "DeepSeek-R1-Distill-Qwen-1.5B requires boundary>=2 per v1.5 report"
        )


def test_channel_ids_roundtrip() -> None:
    """For every channel the profile lists, resolve_model() must recover it."""
    for cid in iter_model_channel_ids():
        prof, ch = resolve_model(cid)
        assert prof.channel_id(ch) == cid


def test_short_id_resolves_to_default() -> None:
    prof, ch = resolve_model("qwen3-4b")
    assert ch == prof.default_channel


def test_short_id_with_boundary_suffix() -> None:
    # DeepSeek-1.5B uses b=2.
    prof, ch = resolve_model("deepseek-r1-distill-qwen-1.5b@e8-q10-b2")
    assert ch.variant == "e8" and ch.q_range == 10 and ch.boundary == 2


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        resolve_model("nonexistent-7b")


def test_unknown_channel_raises() -> None:
    with pytest.raises(KeyError):
        # Q=9999 is not in the Qwen3-4B channel set
        resolve_model("qwen3-4b@e8-q9999")


def test_missing_q_suffix_raises() -> None:
    with pytest.raises(ValueError):
        resolve_model("qwen3-4b@e8")


def test_head_dim_shapes_are_int_or_tuple() -> None:
    for prof in MODEL_REGISTRY.values():
        assert isinstance(prof.head_dim, (int, tuple))
        if isinstance(prof.head_dim, tuple):
            for hd in prof.head_dim:
                assert isinstance(hd, int) and hd > 0


def test_compression_ratios_are_sane() -> None:
    """All listed channels should claim compression in [1.0, 10.0]."""
    for prof in MODEL_REGISTRY.values():
        for ch in prof.channels:
            assert 1.0 <= ch.est_compression <= 10.0, (prof.short_id, ch)
