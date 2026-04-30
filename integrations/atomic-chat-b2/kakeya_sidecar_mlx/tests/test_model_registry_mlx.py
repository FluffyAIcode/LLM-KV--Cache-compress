"""Pure-logic tests for the MLX model registry.

No MLX or mlx-lm required — these validate the registry structure,
DFlash draft linkage, and channel-id parser.
"""
from __future__ import annotations

import pytest

from kakeya_sidecar_mlx.model_registry_mlx import (
    MODEL_REGISTRY_MLX,
    MLXChannel,
    iter_mlx_channel_ids,
    resolve_mlx_model,
)


def test_registry_covers_dflash_hero_models() -> None:
    """B2 must expose the models z-lab published DFlash drafts for."""
    for required in ("qwen3-4b", "qwen3-8b", "llama-3.1-8b-instruct", "qwen3.5-4b"):
        assert required in MODEL_REGISTRY_MLX, required


def test_default_channel_is_one_of_the_channels() -> None:
    for short, prof in MODEL_REGISTRY_MLX.items():
        assert prof.default_channel in prof.channels, f"{short} default missing"


def test_dflash_flag_and_repo_coherent() -> None:
    """dflash_available=True must imply dflash_draft_repo is set."""
    for prof in MODEL_REGISTRY_MLX.values():
        for ch in prof.channels:
            if ch.dflash_available:
                assert ch.dflash_draft_repo is not None, (
                    f"{prof.short_id}@{ch.variant}-q{ch.q_range} "
                    "marked available but no draft repo"
                )


def test_post_init_downgrades_inconsistent_available_flag() -> None:
    """Constructing a channel with available=True but no repo must
    silently clear the flag, not lie to the UI.
    """
    bad = MLXChannel("e8", 10, 0, 3.37, None, "balanced",
                     dflash_draft_repo=None, dflash_available=True)
    assert bad.dflash_available is False


def test_channel_id_roundtrip() -> None:
    for cid in iter_mlx_channel_ids():
        prof, ch = resolve_mlx_model(cid)
        assert prof.channel_id(ch) == cid


def test_bare_short_id_resolves_to_default() -> None:
    prof, ch = resolve_mlx_model("qwen3-8b")
    assert ch == prof.default_channel


def test_qwen3_8b_default_is_near_lossless_for_safety() -> None:
    """At DFlash b=16, near-lossless Q=38 keeps acceptance rate safe.
    The B2 benchmark runs Q=38 first for the same reason — default
    should match.
    """
    prof, ch = resolve_mlx_model("qwen3-8b")
    assert ch.q_range == 38
    assert ch.label == "near-lossless"


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        resolve_mlx_model("gpt-42-megamax")


def test_unknown_channel_raises() -> None:
    with pytest.raises(KeyError):
        resolve_mlx_model("qwen3-4b@e8-q9999")


def test_missing_q_suffix_raises() -> None:
    with pytest.raises(ValueError):
        resolve_mlx_model("qwen3-4b@e8")


def test_mistral_has_no_dflash_but_still_works() -> None:
    """No DFlash draft published for Mistral; registry must still
    expose MLX-only channels so users aren't locked out."""
    prof, ch = resolve_mlx_model("mistral-7b-instruct-v0.3")
    assert ch.dflash_available is False
    assert ch.dflash_draft_repo is None
    assert ch.q_range in (38, 10)


def test_all_channels_are_e8_only_for_b2() -> None:
    """B2 ships E8 only; D4 stays in B1. Guard against regression."""
    for prof in MODEL_REGISTRY_MLX.values():
        for ch in prof.channels:
            assert ch.variant == "e8", (prof.short_id, ch.variant)
