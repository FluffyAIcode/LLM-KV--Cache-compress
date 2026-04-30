"""Engine abstractions for the M5 benchmark.

Real runs use ``RealEngine`` which delegates to the B2 MLXEngine
(i.e. DFlash path + KakeyaLatticeMLXCache). CI / dry-runs use
``MockEngine`` which returns deterministic fake acceptance-length
traces so the runner + metrics pipeline is exercised without any
MLX / dflash / Metal dependency.

Both engines expose the same ``generate(prompt, channel, max_tokens)``
method, returning ``EngineResult``.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger("benchmarks.b2_dflash_kakeya.engines")


@dataclass
class EngineResult:
    response: str
    acceptance_lengths: list[int]
    generation_tps: float | None
    first_token_latency_s: float | None
    total_tokens: int
    codec_fired: int | None = None
    extra: dict = field(default_factory=dict)


class Engine(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        channel: str,
        max_tokens: int,
    ) -> EngineResult:
        ...

    def close(self) -> None:
        ...


# ---------------------------------------------------------------------------
# MockEngine
# ---------------------------------------------------------------------------


class MockEngine:
    """Deterministic fake engine for CI + dry-run.

    Simulates the relationship we expect from the real stack:

    - bf16 baseline: acceptance length ~15 (DFlash's Qwen3-8B number)
    - Kakeya Q=38: ~14 (small hit)
    - Kakeya Q=10: ~12 (moderate hit)
    - Kakeya Q=4:  ~8  (large hit)

    Values are drawn from a small Gaussian with those means so the
    metrics pipeline sees realistic distribution shapes.
    """

    _ACCEPT_MEAN_BY_CHANNEL = {
        "bf16":   15.0,
        "e8-q38": 14.0,
        "e8-q10": 12.0,
        "e8-q4":   8.0,
    }
    _TPS_MEAN_BY_CHANNEL = {
        "bf16":   200.0,
        "e8-q38": 195.0,
        "e8-q10": 175.0,
        "e8-q4":  120.0,
    }

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def generate(
        self,
        *,
        prompt: str,
        channel: str,
        max_tokens: int,
    ) -> EngineResult:
        al_mean = self._ACCEPT_MEAN_BY_CHANNEL.get(channel, 10.0)
        tps_mean = self._TPS_MEAN_BY_CHANNEL.get(channel, 150.0)

        # Decide how many verify steps a max_tokens budget produces.
        n_steps = max(1, int(max_tokens / max(al_mean, 1.0)))
        acc = [
            max(1, int(self._rng.gauss(al_mean, 1.5)))
            for _ in range(n_steps)
        ]
        total_tokens = sum(acc)
        tps = max(10.0, self._rng.gauss(tps_mean, tps_mean * 0.05))
        ttft = max(0.02, self._rng.gauss(0.12, 0.02))

        return EngineResult(
            response=f"<mock-response-for:{prompt[:30]!r}>",
            acceptance_lengths=acc,
            generation_tps=tps,
            first_token_latency_s=ttft,
            total_tokens=total_tokens,
            codec_fired=0 if channel == "bf16" else n_steps * 30,
            extra={"channel": channel, "backend": "mock"},
        )

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# RealEngine (Apple Silicon only; thin wrapper over B2 MLXEngine)
# ---------------------------------------------------------------------------


class RealEngine:
    """Adapter over ``kakeya_sidecar_mlx.MLXEngine``.

    Lazily imports everything so ``import engines`` on Linux CI works.
    """

    def __init__(
        self,
        *,
        target_model: str,
        enable_dflash: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        from kakeya_sidecar_mlx.engine_mlx import MLXEngine, MLXEngineConfig

        cfg = MLXEngineConfig(
            enable_dflash=enable_dflash,
            trust_remote_code=trust_remote_code,
        )
        self._engine = MLXEngine(cfg)
        self._target = target_model

    def generate(
        self,
        *,
        prompt: str,
        channel: str,
        max_tokens: int,
    ) -> EngineResult:
        # channel maps ("bf16", "e8-q38", "e8-q10", ...) → B2 channel id
        channel_id, override = self._channel_to_id(channel)

        t0 = time.time()
        response, stats = self._engine.chat(
            channel_id,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            override=override,
        )
        wall = time.time() - t0

        # MLXEngine stats carry 'acceptance_length_mean' only, not the
        # full per-step list; for the benchmark we want distribution.
        # The B2 engine will be extended in a later PR to expose the
        # per-step list; for now we synthesize a single-element list
        # so the aggregation still works.
        al_mean = stats.get("acceptance_length_mean")
        acc_list = [int(al_mean)] if al_mean else []

        total_tokens = stats.get("generated_chars", 0) // 4   # rough proxy
        tps = (total_tokens / wall) if wall > 0 else None

        return EngineResult(
            response=response,
            acceptance_lengths=acc_list,
            generation_tps=tps,
            first_token_latency_s=None,
            total_tokens=total_tokens,
            codec_fired=None,
            extra={"channel": channel, "backend": "mlx+dflash+kakeya"},
        )

    @staticmethod
    def _channel_to_id(channel: str) -> tuple[str, dict | None]:
        """Map benchmark channel name to MLXEngine channel id + override.

        ``channel="bf16"`` maps to the target's Q=38 channel with a
        per-request override that disables the codec path (boundary
        covers every layer). In practice we ship a "bypass" dedicated
        channel in a follow-up; for now the override is a clear
        signal.
        """
        # The benchmark assumes Qwen3-8B as target model id.
        if channel == "bf16":
            return "qwen3-8b@e8-q38", {"q_range": 38, "boundary": 99999}
        if channel == "e8-q38":
            return "qwen3-8b@e8-q38", None
        if channel == "e8-q10":
            return "qwen3-8b@e8-q10", None
        if channel == "e8-q4":
            return "qwen3-8b@e8-q4", None
        raise ValueError(f"unknown channel {channel!r}")

    def close(self) -> None:
        pass


__all__ = ["Engine", "EngineResult", "MockEngine", "RealEngine"]
