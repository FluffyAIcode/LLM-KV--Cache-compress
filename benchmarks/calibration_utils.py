"""Minimal calibration utilities extracted from e2e_ppl_pre_rope.py.

`e2e_ppl_pre_rope.py` depends on turboquant_roundtrip and the Rust
kakeyaturbo-bench binary, which are *not* required for offline
calibration. This module exposes just what the calibration step needs
without dragging in those dependencies.

The implementations are byte-identical to the ones in
e2e_ppl_pre_rope.py so we can keep both sides importing from here
without semantic drift.
"""
from __future__ import annotations

import os
from typing import Iterable

import torch
from transformers import DynamicCache


def load_wikitext_passages(tok, min_tokens: int, n_passages: int,
                           split: str | None = None) -> list[str]:
    """Draw at least `n_passages` passages of >= `min_tokens` tokens.

    Mirrors benchmarks.e2e_ppl_pre_rope.load_wikitext_passages, but the
    default split is overridable via the DATASETS_WIKITEXT_SPLIT env var
    (this lets downstream callers that hardcode the old default 'test'
    be retargeted to 'train' for disjoint calibration).
    """
    if split is None:
        split = os.environ.get("DATASETS_WIKITEXT_SPLIT", "test")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    passages: list[str] = []
    current: list[str] = []
    current_tok = 0
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        current.append(text)
        current_tok += int(len(text.split()) * 1.3)
        if current_tok >= min_tokens:
            passage = "".join(current)
            if tok(passage, return_tensors="pt")["input_ids"].shape[-1] >= min_tokens:
                passages.append(passage)
                if len(passages) >= n_passages:
                    return passages
            current, current_tok = [], 0
    return passages


@torch.inference_mode()
def prefill_cache(model, input_ids, prefill_chunk: int = 0) -> DynamicCache:
    cache = DynamicCache(config=model.config)
    if prefill_chunk <= 0 or input_ids.shape[-1] <= prefill_chunk:
        _ = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    else:
        for s in range(0, input_ids.shape[-1], prefill_chunk):
            e = min(s + prefill_chunk, input_ids.shape[-1])
            _ = model(input_ids=input_ids[:, s:e], past_key_values=cache,
                      use_cache=True)
    return cache
