"""Unit tests for ``metrics.py`` — pure Python, no deps beyond stdlib."""
from __future__ import annotations

import pytest

from benchmarks.b2_dflash_kakeya.metrics import (
    gsm8k_correct,
    humaneval_correct,
    mean,
    percentile,
    summarise_accept_lengths,
)


def test_mean_empty_is_none() -> None:
    assert mean([]) is None


def test_mean_basic() -> None:
    assert mean([1, 2, 3, 4]) == 2.5


def test_percentile_empty_is_none() -> None:
    assert percentile([], 50) is None


def test_percentile_single_element() -> None:
    assert percentile([7.0], 50) == 7.0
    assert percentile([7.0], 0) == 7.0
    assert percentile([7.0], 100) == 7.0


def test_percentile_matches_numpy_default() -> None:
    xs = [10, 20, 30, 40, 50]
    assert percentile(xs, 0) == 10
    assert percentile(xs, 50) == 30
    assert percentile(xs, 100) == 50
    # Linear interpolation between sorted[1]=20 and sorted[2]=30 at rank 1.5
    assert percentile(xs, 37.5) == pytest.approx(25.0)


def test_percentile_rejects_bad_p() -> None:
    with pytest.raises(ValueError):
        percentile([1, 2, 3], -1)
    with pytest.raises(ValueError):
        percentile([1, 2, 3], 101)


def test_summarise_handles_empty_records() -> None:
    out = summarise_accept_lengths([])
    assert out == {"mean": None, "p50": None, "p95": None}


def test_summarise_flattens_per_sample_lists() -> None:
    class _R:
        def __init__(self, xs):
            self.acceptance_lengths = xs

    records = [_R([10, 12]), _R([14, 16]), _R([18])]
    out = summarise_accept_lengths(records)
    assert out["mean"] == pytest.approx(14.0)


def test_summarise_accepts_dicts() -> None:
    out = summarise_accept_lengths([
        {"acceptance_lengths": [1, 2, 3]},
        {"acceptance_lengths": [4, 5]},
    ])
    assert out["mean"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# correctness proxies
# ---------------------------------------------------------------------------


def test_gsm8k_correct_extracts_after_hash() -> None:
    expected = "Jane has three apples. #### 3"
    assert gsm8k_correct("The answer is 3.", expected) is True
    assert gsm8k_correct("The answer is 42.", expected) is False


def test_gsm8k_correct_no_hash_prefix() -> None:
    assert gsm8k_correct("Answer: 7", "7") is True
    assert gsm8k_correct("Answer: 7", "") is False


def test_humaneval_correct_requires_def_and_return() -> None:
    assert humaneval_correct("def f(x):\n    return x + 1", "") is True
    assert humaneval_correct("print('refused')", "") is False
    assert humaneval_correct("def f(x):\n    print(x)", "") is False
