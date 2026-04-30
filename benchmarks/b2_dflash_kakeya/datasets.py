"""Dataset loaders for the B2 acceptance-rate benchmark.

Two datasets are supported out of the box: **gsm8k** and **humaneval**.

Loading strategy (in priority order):

1. **Local JSONL file**: ``benchmarks/b2_dflash_kakeya/data/<name>.jsonl``.
   Users who can't reach HF hub (or want a frozen subset for the
   paper) check in a jsonl snapshot and we read it directly. Keeps
   the benchmark reproducible offline.
2. **HuggingFace ``datasets`` library** if available. We load
   ``openai/gsm8k`` (``main`` config, ``test`` split) and
   ``openai/humaneval`` (``test`` split). Cached under HF_HOME.
3. **Synthetic fixture** — a tiny built-in 3-prompt dataset per name.
   Used by unit tests and ``--dry-run`` mode; explicitly labelled so
   nobody publishes numbers from it by accident.

Each prompt is returned as a ``PromptItem`` dataclass carrying an
id, the prompt string the target LLM will see, and an optional
ground-truth field used by the correctness proxy in ``metrics.py``.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptItem:
    dataset: str              # "gsm8k" | "humaneval" | "synthetic"
    prompt_id: str
    prompt: str
    ground_truth: str | None = None


_SUPPORTED = ("gsm8k", "humaneval")

_DATA_DIR = Path(__file__).parent / "data"


def _load_local_jsonl(name: str) -> list[dict] | None:
    path = _DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        return None
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_hf(name: str) -> list[dict] | None:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        return None
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [dict(row) for row in ds]
    if name == "humaneval":
        ds = load_dataset("openai/humaneval", split="test")
        return [dict(row) for row in ds]
    return None


_SYNTHETIC_FIXTURES: dict[str, list[PromptItem]] = {
    "gsm8k": [
        PromptItem("synthetic", "s0",
                   "Q: Janet has 3 apples, gives 1 to Bob. How many are left?",
                   "2"),
        PromptItem("synthetic", "s1",
                   "Q: A train travels 60 miles in 1.5 hours. What is its speed?",
                   "40"),
        PromptItem("synthetic", "s2",
                   "Q: If 5 pencils cost $2.50, what is the cost of 8 pencils?",
                   "4"),
    ],
    "humaneval": [
        PromptItem("synthetic", "h0",
                   "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n",
                   "def add(a, b):\n    return a + b"),
        PromptItem("synthetic", "h1",
                   "def is_even(n):\n    \"\"\"Return True if n is even.\"\"\"\n",
                   "def is_even(n):\n    return n % 2 == 0"),
        PromptItem("synthetic", "h2",
                   "def reverse(s):\n    \"\"\"Return s reversed.\"\"\"\n",
                   "def reverse(s):\n    return s[::-1]"),
    ],
}


def load_dataset_for_b2(
    name: str,
    *,
    n_samples: int,
    seed: int = 42,
    allow_hf: bool = True,
    allow_synthetic: bool = True,
) -> list[PromptItem]:
    """Load up to ``n_samples`` prompts for the named dataset.

    The loader degrades gracefully: local jsonl → HF datasets →
    synthetic. ``allow_hf=False`` forces the local/synthetic path
    (useful for offline CI). ``allow_synthetic=False`` forbids the
    synthetic fallback (useful for real benchmark runs so nobody
    accidentally "runs gsm8k" on 3 fake prompts).
    """
    if name not in _SUPPORTED:
        raise ValueError(
            f"dataset {name!r} not supported; pick from {_SUPPORTED}"
        )

    rng = random.Random(seed)

    rows: list[dict] | None = _load_local_jsonl(name)
    if rows is None and allow_hf:
        rows = _load_hf(name)

    if rows is not None:
        rng.shuffle(rows)
        rows = rows[:n_samples]
        return [_row_to_item(name, i, r) for i, r in enumerate(rows)]

    if not allow_synthetic:
        raise FileNotFoundError(
            f"no local jsonl for {name!r} and synthetic fallback disabled. "
            f"Expected file at {_DATA_DIR / (name + '.jsonl')}, or install "
            "the `datasets` library and set allow_hf=True."
        )

    fixture = list(_SYNTHETIC_FIXTURES[name])
    rng.shuffle(fixture)
    return fixture[:n_samples] if n_samples < len(fixture) else fixture


def _row_to_item(name: str, i: int, row: dict) -> PromptItem:
    if name == "gsm8k":
        return PromptItem(
            dataset="gsm8k",
            prompt_id=f"gsm8k-{i}",
            prompt=row.get("question", ""),
            ground_truth=row.get("answer"),
        )
    if name == "humaneval":
        return PromptItem(
            dataset="humaneval",
            prompt_id=str(row.get("task_id", f"humaneval-{i}")),
            prompt=row.get("prompt", ""),
            ground_truth=row.get("canonical_solution"),
        )
    raise ValueError(name)


__all__ = [
    "PromptItem",
    "load_dataset_for_b2",
]
