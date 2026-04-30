"""Top-level benchmark runner.

Usage:

    python -m benchmarks.b2_dflash_kakeya.runner \\
        --target Qwen/Qwen3-8B \\
        --draft  z-lab/Qwen3-8B-DFlash-b16 \\
        --datasets gsm8k humaneval \\
        --n-samples 32 \\
        --channels bf16 e8-q38 e8-q10 e8-q4 \\
        --out-dir reports/b2_release

    python -m benchmarks.b2_dflash_kakeya.runner --dry-run
        # CI-friendly: uses MockEngine + synthetic dataset fallback

The runner is deliberately engine-agnostic — all MLX / dflash /
mlx-lm imports are behind the ``RealEngine`` constructor in
``engines.py``. Linux CI exercises the whole runner via
``--mock-engine``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from . import __version__ as RUNNER_VERSION
from .datasets import PromptItem, load_dataset_for_b2
from .engines import Engine, EngineResult, MockEngine
from .metrics import (
    gsm8k_correct,
    humaneval_correct,
    mean,
    percentile,
)
from .schema import (
    SCHEMA_VERSION,
    AggregateMetrics,
    BenchmarkResult,
    HardwareInfo,
    SampleRecord,
    SoftwareInfo,
)


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def _build_engine(args) -> Engine:
    if args.dry_run or args.mock_engine:
        return MockEngine(seed=args.seed)
    from .engines import RealEngine
    return RealEngine(
        target_model=args.target,
        enable_dflash=not args.no_dflash,
    )


# ---------------------------------------------------------------------------
# Run one (dataset, channel) combination
# ---------------------------------------------------------------------------


def run_combination(
    *,
    engine: Engine,
    dataset: str,
    channel: str,
    prompts: list[PromptItem],
    max_tokens: int,
    target_model: str,
    draft_model: str | None,
) -> BenchmarkResult:
    sample_records: list[SampleRecord] = []
    n_correct = 0
    any_correctness_scored = False

    for item in prompts:
        result: EngineResult = engine.generate(
            prompt=item.prompt,
            channel=channel,
            max_tokens=max_tokens,
        )

        correct: bool | None = None
        if item.ground_truth is not None:
            if dataset == "gsm8k":
                correct = gsm8k_correct(result.response, item.ground_truth)
            elif dataset == "humaneval":
                correct = humaneval_correct(result.response, item.ground_truth)
            if correct is not None:
                any_correctness_scored = True
                if correct:
                    n_correct += 1

        sample_records.append(SampleRecord(
            prompt_id=item.prompt_id,
            prompt=item.prompt,
            response=result.response,
            acceptance_lengths=list(result.acceptance_lengths),
            generation_tps=result.generation_tps,
            first_token_latency_s=result.first_token_latency_s,
            total_tokens=result.total_tokens,
            codec_fired=result.codec_fired,
            correctness_proxy=correct,
        ))

    flat_al: list[float] = []
    for s in sample_records:
        flat_al.extend(float(x) for x in s.acceptance_lengths)

    agg = AggregateMetrics(
        acceptance_length_mean=mean(flat_al),
        acceptance_length_p50=percentile(flat_al, 50.0),
        acceptance_length_p95=percentile(flat_al, 95.0),
        generation_tps_mean=mean(
            [s.generation_tps for s in sample_records if s.generation_tps]
        ),
        first_token_latency_s=mean(
            [s.first_token_latency_s for s in sample_records
             if s.first_token_latency_s is not None]
        ),
        total_tokens_sum=sum(s.total_tokens for s in sample_records),
        codec_fired_mean=mean(
            [float(s.codec_fired) for s in sample_records
             if s.codec_fired is not None]
        ),
        n_correct=n_correct if any_correctness_scored else None,
        n_samples=len(sample_records),
    )

    return BenchmarkResult(
        schema_version=SCHEMA_VERSION,
        target_model=target_model,
        draft_model=draft_model,
        dataset=dataset,
        channel=channel,
        n_samples=len(sample_records),
        samples=sample_records,
        aggregate=agg,
        hardware=detect_hardware(),
        software=detect_software(),
    )


# ---------------------------------------------------------------------------
# Env detection (safe to call on any OS; returns "unknown" fields when lib
# isn't installed).
# ---------------------------------------------------------------------------


def detect_hardware() -> HardwareInfo:
    import platform
    chip = platform.processor() or "unknown"
    device = "unknown"
    try:
        import mlx.core as mx  # type: ignore
        if mx.metal.is_available():
            device = "mlx:metal"
        else:
            device = "mlx:cpu"
    except ImportError:
        pass
    return HardwareInfo(device=device, chip=chip)


def detect_software() -> SoftwareInfo:
    def _ver(mod: str) -> str | None:
        try:
            m = __import__(mod)
            return getattr(m, "__version__", None)
        except ImportError:
            return None
    return SoftwareInfo(
        mlx=_ver("mlx"),
        mlx_lm=_ver("mlx_lm"),
        dflash=_ver("dflash"),
        kakeyalattice_mlx=_ver("kakeyalattice_mlx"),
        kakeya_sidecar_mlx=_ver("kakeya_sidecar_mlx"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="b2-dflash-kakeya-benchmark",
        description=(
            "Acceptance-rate benchmark for DFlash speculative decoding "
            "combined with KakeyaLattice E8 KV-cache compression "
            "(B2 / M5)."
        ),
    )
    p.add_argument("--target", default="Qwen/Qwen3-8B",
                   help="HuggingFace / mlx-community target model id")
    p.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16",
                   help="DFlash draft model id (non-thinking, b16).")
    p.add_argument("--datasets", nargs="+",
                   default=["gsm8k", "humaneval"],
                   choices=["gsm8k", "humaneval"])
    p.add_argument("--channels", nargs="+",
                   default=["bf16", "e8-q38", "e8-q10", "e8-q4"],
                   help="KV cache channels to evaluate.")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-dflash", action="store_true",
                   help="Disable DFlash (debug only).")
    p.add_argument("--dry-run", action="store_true",
                   help="Use MockEngine + synthetic datasets. "
                        "No MLX / dflash / HF required.")
    p.add_argument("--mock-engine", action="store_true",
                   help="Force MockEngine but still load real datasets "
                        "(via local jsonl or HF datasets).")
    p.add_argument("--allow-synthetic", action="store_true",
                   help="Permit synthetic dataset fallback even outside "
                        "--dry-run. Off by default to prevent publishing "
                        "numbers from 3-prompt fixtures.")
    p.add_argument("--out-dir", default="reports/b2_release",
                   help="Where to write per-combination JSON.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("b2-dflash-kakeya")
    log.info("runner version=%s schema=%s", RUNNER_VERSION, SCHEMA_VERSION)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = _build_engine(args)

    try:
        for dataset in args.datasets:
            prompts = load_dataset_for_b2(
                dataset,
                n_samples=args.n_samples,
                seed=args.seed,
                allow_hf=not args.dry_run,
                allow_synthetic=args.dry_run or args.allow_synthetic,
            )
            log.info("dataset=%s n_prompts=%d", dataset, len(prompts))

            for channel in args.channels:
                log.info("running channel=%s", channel)
                result = run_combination(
                    engine=engine,
                    dataset=dataset,
                    channel=channel,
                    prompts=prompts,
                    max_tokens=args.max_tokens,
                    target_model=args.target,
                    draft_model=None if args.no_dflash else args.draft,
                )
                fname = f"b2_dflash_kakeya_{dataset}_{channel}.json"
                out_path = out_dir / fname
                with out_path.open("w") as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
                log.info("wrote %s  (accept_mean=%s)",
                         out_path, result.aggregate.acceptance_length_mean)
    finally:
        engine.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
