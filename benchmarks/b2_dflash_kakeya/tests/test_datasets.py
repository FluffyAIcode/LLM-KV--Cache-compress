"""Dataset loader tests.

Two fallback paths verified without touching the network:
- synthetic fixture: always available (3 items per dataset)
- local jsonl override: supplied via temporary directory
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.b2_dflash_kakeya import datasets as ds_mod
from benchmarks.b2_dflash_kakeya.datasets import (
    PromptItem,
    load_dataset_for_b2,
)


def test_synthetic_fixture_for_gsm8k() -> None:
    items = load_dataset_for_b2(
        "gsm8k", n_samples=3, allow_hf=False, allow_synthetic=True,
    )
    assert len(items) == 3
    for it in items:
        assert isinstance(it, PromptItem)
        assert it.dataset == "synthetic"
        assert it.ground_truth is not None


def test_synthetic_forbidden_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset_for_b2(
            "gsm8k", n_samples=3, allow_hf=False, allow_synthetic=False,
        )


def test_unknown_dataset_rejected() -> None:
    with pytest.raises(ValueError):
        load_dataset_for_b2("winograd", n_samples=1)


def test_humanval_synthetic_has_code_scaffolding() -> None:
    items = load_dataset_for_b2(
        "humaneval", n_samples=3, allow_hf=False, allow_synthetic=True,
    )
    for it in items:
        assert "def " in it.prompt


def test_local_jsonl_preferred_over_synthetic(tmp_path, monkeypatch) -> None:
    """If a local jsonl exists, it's used instead of the synthetic
    fixture (and instead of HF)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    jsonl = data_dir / "gsm8k.jsonl"
    with jsonl.open("w") as f:
        f.write(json.dumps({"question": "Q1", "answer": "A1"}) + "\n")
        f.write(json.dumps({"question": "Q2", "answer": "A2"}) + "\n")

    monkeypatch.setattr(ds_mod, "_DATA_DIR", data_dir)

    items = load_dataset_for_b2("gsm8k", n_samples=10, allow_hf=False)
    assert len(items) == 2
    assert all(it.dataset == "gsm8k" for it in items)
    assert {it.ground_truth for it in items} == {"A1", "A2"}


def test_n_samples_truncation(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    jsonl = data_dir / "humaneval.jsonl"
    with jsonl.open("w") as f:
        for i in range(10):
            f.write(json.dumps({
                "task_id": f"t/{i}",
                "prompt": f"def f{i}(): return",
                "canonical_solution": f"return {i}",
            }) + "\n")
    monkeypatch.setattr(ds_mod, "_DATA_DIR", data_dir)

    items = load_dataset_for_b2("humaneval", n_samples=4, allow_hf=False)
    assert len(items) == 4


def test_seed_determinism_for_synthetic() -> None:
    a = load_dataset_for_b2("gsm8k", n_samples=3, seed=1,
                            allow_hf=False, allow_synthetic=True)
    b = load_dataset_for_b2("gsm8k", n_samples=3, seed=1,
                            allow_hf=False, allow_synthetic=True)
    assert [(it.prompt_id, it.prompt) for it in a] == \
           [(it.prompt_id, it.prompt) for it in b]
