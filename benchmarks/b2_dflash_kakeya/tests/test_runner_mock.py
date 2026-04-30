"""End-to-end runner test using MockEngine + synthetic datasets.

Verifies the full loop: dataset loading → engine.generate per prompt
→ metric aggregation → JSON serialisation → schema round-trip.

Runs in <0.5s on Linux CI, no MLX / dflash / HF.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.b2_dflash_kakeya.engines import MockEngine
from benchmarks.b2_dflash_kakeya.runner import (
    build_parser,
    main,
    run_combination,
)
from benchmarks.b2_dflash_kakeya.datasets import load_dataset_for_b2
from benchmarks.b2_dflash_kakeya.schema import SCHEMA_VERSION


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.target == "Qwen/Qwen3-8B"
    assert "z-lab/Qwen3-8B-DFlash-b16" in args.draft
    assert set(args.datasets) == {"gsm8k", "humaneval"}
    assert args.channels == ["bf16", "e8-q38", "e8-q10", "e8-q4"]


def test_run_combination_accept_length_ordering() -> None:
    """MockEngine encodes the theoretical acceptance-length ordering
    bf16 > Q=38 > Q=10 > Q=4. The benchmark runner aggregation must
    preserve it."""
    engine = MockEngine(seed=0)
    prompts = load_dataset_for_b2(
        "gsm8k", n_samples=3, allow_hf=False, allow_synthetic=True,
    )

    accept_means: dict[str, float] = {}
    for channel in ("bf16", "e8-q38", "e8-q10", "e8-q4"):
        res = run_combination(
            engine=engine, dataset="gsm8k", channel=channel,
            prompts=prompts, max_tokens=128,
            target_model="Qwen/Qwen3-8B",
            draft_model="z-lab/Qwen3-8B-DFlash-b16",
        )
        accept_means[channel] = res.aggregate.acceptance_length_mean or 0.0

    assert accept_means["bf16"]   > accept_means["e8-q38"]
    assert accept_means["e8-q38"] > accept_means["e8-q10"]
    assert accept_means["e8-q10"] > accept_means["e8-q4"]


def test_main_dry_run_writes_json_per_combination(tmp_path) -> None:
    out_dir = tmp_path / "reports"
    code = main([
        "--dry-run",
        "--n-samples", "3",
        "--max-tokens", "64",
        "--out-dir", str(out_dir),
        "--datasets", "gsm8k",
        "--channels", "bf16", "e8-q10",
    ])
    assert code == 0

    files = sorted(p.name for p in out_dir.iterdir())
    assert files == [
        "b2_dflash_kakeya_gsm8k_bf16.json",
        "b2_dflash_kakeya_gsm8k_e8-q10.json",
    ]

    for name in files:
        with (out_dir / name).open() as f:
            obj = json.load(f)
        assert obj["schema_version"] == SCHEMA_VERSION
        assert obj["dataset"] == "gsm8k"
        assert obj["n_samples"] == 3
        assert obj["aggregate"]["acceptance_length_mean"] is not None
        assert obj["aggregate"]["n_samples"] == 3
        assert len(obj["samples"]) == 3


def test_main_dry_run_humaneval_correctness_populated(tmp_path) -> None:
    out_dir = tmp_path / "reports"
    main([
        "--dry-run",
        "--n-samples", "3",
        "--max-tokens", "32",
        "--out-dir", str(out_dir),
        "--datasets", "humaneval",
        "--channels", "bf16",
    ])
    with (out_dir / "b2_dflash_kakeya_humaneval_bf16.json").open() as f:
        obj = json.load(f)
    # Mock responses don't actually contain code, so correctness proxy
    # should be False for all; n_correct populated = 0.
    # The important thing is that the field exists and isn't None for
    # a dataset with ground truth.
    assert obj["aggregate"]["n_correct"] is not None
    for s in obj["samples"]:
        assert s["correctness_proxy"] is not None


def test_channel_not_in_mock_engine_gets_default_means() -> None:
    """A channel name the MockEngine doesn't know should not crash —
    it falls back to 10/150 defaults."""
    engine = MockEngine(seed=1)
    res = engine.generate(prompt="hi", channel="custom-42", max_tokens=64)
    assert res.acceptance_lengths
    assert all(x >= 1 for x in res.acceptance_lengths)
