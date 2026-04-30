"""Output JSON schema for b2-dflash-kakeya benchmark runs.

We pin the schema version here so downstream tooling (reports site,
comparison scripts) can detect breaking changes without guessing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "b2-dflash-kakeya-v1"


@dataclass
class HardwareInfo:
    device: str = "unknown"
    chip: str = "unknown"
    total_memory_gb: float | None = None


@dataclass
class SoftwareInfo:
    mlx: str | None = None
    mlx_lm: str | None = None
    dflash: str | None = None
    kakeyalattice_mlx: str | None = None
    kakeya_sidecar_mlx: str | None = None


@dataclass
class SampleRecord:
    prompt_id: str
    prompt: str
    response: str
    acceptance_lengths: list[int]
    generation_tps: float | None
    first_token_latency_s: float | None
    total_tokens: int
    codec_fired: int | None = None
    correctness_proxy: bool | None = None


@dataclass
class AggregateMetrics:
    acceptance_length_mean: float | None
    acceptance_length_p50: float | None
    acceptance_length_p95: float | None
    generation_tps_mean: float | None
    first_token_latency_s: float | None
    total_tokens_sum: int
    codec_fired_mean: float | None
    n_correct: int | None
    n_samples: int


@dataclass
class BenchmarkResult:
    schema_version: str
    target_model: str
    draft_model: str | None
    dataset: str
    channel: str                 # e.g. "bf16" or "e8-q10"
    n_samples: int
    samples: list[SampleRecord]
    aggregate: AggregateMetrics
    hardware: HardwareInfo = field(default_factory=HardwareInfo)
    software: SoftwareInfo = field(default_factory=SoftwareInfo)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


__all__ = [
    "SCHEMA_VERSION",
    "HardwareInfo",
    "SoftwareInfo",
    "SampleRecord",
    "AggregateMetrics",
    "BenchmarkResult",
]
