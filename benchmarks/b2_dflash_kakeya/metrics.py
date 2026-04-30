"""Metric aggregation helpers.

All stats are pure Python + standard library — no numpy dependency
so this module loads on any CI.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def percentile(xs: Sequence[float], p: float) -> float | None:
    """Linear-interpolation percentile matching numpy's default.

    ``p`` in [0, 100]. Returns ``None`` for empty input to propagate
    missing-data semantics to the schema.
    """
    if not xs:
        return None
    if p < 0 or p > 100:
        raise ValueError(f"percentile p must be in [0, 100], got {p}")
    sorted_xs = sorted(xs)
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    rank = (p / 100.0) * (len(sorted_xs) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_xs[lo])
    frac = rank - lo
    return float(sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * frac)


def mean(xs: Iterable[float]) -> float | None:
    lst = list(xs)
    if not lst:
        return None
    return sum(lst) / len(lst)


def summarise_accept_lengths(
    sample_records: Iterable[object],
) -> dict[str, float | None]:
    """Flatten per-step acceptance lengths across samples and summarise."""
    flat: list[float] = []
    for s in sample_records:
        # Support either a SampleRecord or a plain dict (round-tripped).
        al = getattr(s, "acceptance_lengths", None)
        if al is None and isinstance(s, dict):
            al = s.get("acceptance_lengths")
        if al:
            flat.extend(float(x) for x in al)

    return {
        "mean": mean(flat),
        "p50": percentile(flat, 50.0),
        "p95": percentile(flat, 95.0),
    }


def gsm8k_correct(response: str, expected: str) -> bool:
    """Simple gsm8k correctness proxy.

    GSM8K ground truth format ends with ``#### <answer>``. We strip
    that, then check whether the model's response contains the exact
    numeric answer as a substring. Deliberately loose — this is a
    proxy, not a full grader.
    """
    if "####" in expected:
        expected_answer = expected.rsplit("####", 1)[-1].strip()
    else:
        expected_answer = expected.strip()
    if not expected_answer:
        return False
    return expected_answer in response


def humaneval_correct(response: str, test_snippet: str) -> bool:
    """HumanEval correctness proxy via substring match on the solution body.

    Real HumanEval grading runs the generated code against the
    reference tests in a sandbox; that's deliberately out-of-scope for
    a sidecar benchmark. We approximate by checking that the response
    contains a ``def`` signature and the ``return`` keyword — a very
    loose gate that at least separates "emitted code" from "emitted
    refusal". Upgrade path: wire the official execution harness in a
    follow-up PR.
    """
    _ = test_snippet  # unused; kept for API symmetry with gsm8k_correct
    return ("def " in response) and ("return" in response)


__all__ = [
    "percentile",
    "mean",
    "summarise_accept_lengths",
    "gsm8k_correct",
    "humaneval_correct",
]
