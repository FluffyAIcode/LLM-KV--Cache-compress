"""MLX inference engine — M4 integrates DFlash speculative decoding.

Two code paths:

1. **DFlash path**: when ``cfg.enable_dflash`` and the resolved channel
   has ``dflash_available=True``. We load ``dflash.model_mlx.load_draft``,
   detect an injection strategy for the target KV cache
   (`cache_injection.detect_injection_strategy`), wrap target caches in
   ``KakeyaLatticeMLXCache``, and delegate to
   ``dflash.model_mlx.stream_generate`` under an ``activate()`` context.

2. **Native MLX path** (fallback): when DFlash is off, the draft repo
   isn't available, or the dflash API doesn't expose any injection
   surface. We use ``mlx_lm.generate`` directly with the Kakeya caches
   passed on the target model.

Both paths emit the same ``chat()`` return shape as B1 so the server
layer doesn't need to fork.

MLX / mlx-lm imports are done lazily inside method bodies so the
module is importable on Linux CI for the pure-logic tests.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from .cache_injection import (
    InjectionDecision,
    InjectionStrategy,
    KakeyaCacheInjector,
    detect_injection_strategy,
)
from .model_registry_mlx import (
    MLXChannel,
    MLXDeploymentProfile,
    resolve_mlx_model,
)

log = logging.getLogger("kakeya_sidecar_mlx.engine")


@dataclass
class MLXEngineConfig:
    device: str = "auto"
    dtype: str = "auto"
    max_resident: int = 1
    enable_dflash: bool = False
    trust_remote_code: bool = True
    hf_cache_dir: str | None = None
    dflash_block_size: int = 16
    dflash_num_speculative_tokens: int = 16
    dflash_sliding_window_size: int | None = None
    _runtime: dict[str, Any] = field(default_factory=dict)


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import mlx.core as mx  # type: ignore
        if mx.metal.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# _LoadedMLXModel
# ---------------------------------------------------------------------------


class _LoadedMLXModel:
    """Holds a loaded mlx-lm target model + tokenizer + optional DFlash draft.

    Also pre-computes the injection decision once per model load so the
    hot path doesn't re-inspect dflash's signature per request.
    """

    def __init__(
        self,
        profile: MLXDeploymentProfile,
        cfg: MLXEngineConfig,
        channel: MLXChannel,
    ) -> None:
        from mlx_lm import load  # type: ignore

        repo = profile.mlx_repo_id or profile.hf_repo_id
        log.info("mlx_lm.load(%s)", repo)
        t0 = time.time()
        self.model, self.tokenizer = load(repo)
        log.info("loaded target %s in %.1fs", repo, time.time() - t0)

        self.profile = profile
        self.channel = channel
        self.draft_model = None
        self._stream_generate: Callable | None = None
        self._injection_decision: InjectionDecision = InjectionDecision(
            InjectionStrategy.FALLBACK_NATIVE_MLX, "DFlash not enabled"
        )

        if cfg.enable_dflash and channel.dflash_available:
            self._maybe_load_dflash(channel)

    def _maybe_load_dflash(self, channel: MLXChannel) -> None:
        repo = channel.dflash_draft_repo
        if repo is None:
            return
        try:
            from dflash.model_mlx import (  # type: ignore
                load_draft as _load_draft,
                stream_generate as _stream_generate,
            )
        except ImportError:
            log.warning(
                "dflash not importable; `pip install dflash` to enable "
                "speculative decoding. Falling back to single-track MLX."
            )
            return
        log.info("dflash.load_draft(%s)", repo)
        t0 = time.time()
        self.draft_model = _load_draft(repo)
        log.info("loaded draft %s in %.1fs", repo, time.time() - t0)
        self._stream_generate = _stream_generate
        self._injection_decision = detect_injection_strategy(
            _stream_generate, self.model,
        )
        log.info(
            "DFlash injection strategy = %s (%s)",
            self._injection_decision.strategy.value,
            self._injection_decision.detail,
        )


# ---------------------------------------------------------------------------
# MLXEngine
# ---------------------------------------------------------------------------


class MLXEngine:
    """M4 MLXEngine: DFlash + Kakeya KV, or native-MLX fallback."""

    def __init__(self, cfg: MLXEngineConfig | None = None) -> None:
        self.cfg = cfg or MLXEngineConfig()
        self._device = _pick_device(self.cfg.device)
        self._loaded: "OrderedDict[str, _LoadedMLXModel]" = OrderedDict()
        self._lock = threading.Lock()
        log.info(
            "MLXEngine device=%s max_resident=%d dflash=%s",
            self._device, self.cfg.max_resident, self.cfg.enable_dflash,
        )

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _ensure_loaded(
        self, profile: MLXDeploymentProfile, channel: MLXChannel
    ) -> _LoadedMLXModel:
        with self._lock:
            if profile.short_id in self._loaded:
                self._loaded.move_to_end(profile.short_id)
                return self._loaded[profile.short_id]
            lm = _LoadedMLXModel(profile, self.cfg, channel)
            self._loaded[profile.short_id] = lm
            while len(self._loaded) > self.cfg.max_resident:
                evicted_id, evicted = self._loaded.popitem(last=False)
                log.info("evicting %s", evicted_id)
                del evicted
            return lm

    def warmup(self, channel_id: str) -> None:
        profile, channel = resolve_mlx_model(channel_id)
        self._ensure_loaded(profile, channel)

    # ------------------------------------------------------------------
    # Chat entry points
    # ------------------------------------------------------------------

    def chat(
        self,
        channel_id: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        override: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Non-streaming chat completion.

        Returns ``(text, stats)`` matching B1's shape. ``stats`` adds
        DFlash-specific fields: ``dflash_used``, ``injection_strategy``,
        ``acceptance_length_mean``.
        """
        profile, channel = resolve_mlx_model(channel_id)
        if override:
            channel = self._apply_override(channel, override)
        lm = self._ensure_loaded(profile, channel)

        pieces: list[str] = []
        stats: dict[str, Any] = {}
        for piece, partial_stats in self._run_stream(
            lm, channel, messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ):
            pieces.append(piece)
            stats = partial_stats  # last update wins

        return "".join(pieces), stats

    def chat_stream(
        self,
        channel_id: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        override: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        profile, channel = resolve_mlx_model(channel_id)
        if override:
            channel = self._apply_override(channel, override)
        lm = self._ensure_loaded(profile, channel)
        for piece, _stats in self._run_stream(
            lm, channel, messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ):
            yield piece

    # ------------------------------------------------------------------
    # Core: streaming generator, used by both chat() and chat_stream()
    # ------------------------------------------------------------------

    def _run_stream(
        self,
        lm: _LoadedMLXModel,
        channel: MLXChannel,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        prompt = self._render_prompt(lm, messages)

        injector = KakeyaCacheInjector(
            model=lm.model,
            variant=channel.variant,
            q_range=channel.q_range,
            boundary=channel.boundary,
            strategy=lm._injection_decision.strategy,
        )

        t0 = time.time()
        if (
            lm._stream_generate is not None
            and lm.draft_model is not None
            and lm._injection_decision.strategy
               != InjectionStrategy.FALLBACK_NATIVE_MLX
        ):
            iterator_factory = self._dflash_iter_factory(
                lm, prompt, max_tokens, temperature, top_p,
            )
            yielded_pieces: list[str] = []
            accept_lens: list[int] = []
            with injector.activate(lm._stream_generate):
                for piece, step_info in iterator_factory(
                    extra_kwargs=injector.extra_kwargs
                ):
                    yielded_pieces.append(piece)
                    al = step_info.get("acceptance_length")
                    if al is not None:
                        accept_lens.append(int(al))
                    yield piece, self._stats(
                        channel, t0, yielded_pieces, accept_lens,
                        dflash_used=True, lm=lm,
                    )
                    if stop and any(s in "".join(yielded_pieces) for s in stop):
                        break
        else:
            # Native MLX fallback.
            from mlx_lm.generate import stream_generate as _mlx_stream  # type: ignore

            yielded_pieces = []
            caches = injector.build()
            for piece in _mlx_stream(
                lm.model, lm.tokenizer, prompt=prompt,
                max_tokens=max_tokens,
                temp=max(temperature, 1e-4),
                top_p=top_p,
                prompt_cache=caches,
            ):
                yielded_pieces.append(piece)
                yield piece, self._stats(
                    channel, t0, yielded_pieces, [],
                    dflash_used=False, lm=lm,
                )
                if stop and any(s in "".join(yielded_pieces) for s in stop):
                    break

    def _dflash_iter_factory(
        self, lm, prompt, max_tokens, temperature, top_p,
    ) -> Callable:
        """Build a callable that produces (text, step_info) tuples.

        Abstracted so tests can substitute a mock without touching the
        real dflash import.
        """
        stream_generate = lm._stream_generate

        block_size = self.cfg.dflash_block_size
        draft = lm.draft_model
        model = lm.model
        tokenizer = lm.tokenizer

        def _factory(extra_kwargs: dict[str, Any]):
            for step in stream_generate(
                model,
                draft,
                tokenizer,
                prompt,
                block_size=block_size,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra_kwargs,
            ):
                # dflash.model_mlx.stream_generate emits objects with
                # at least `.text` and often `.accepted_length` /
                # `.generation_tps`. We normalise into (text, info).
                text = getattr(step, "text", None) or getattr(step, "delta", "") or ""
                info: dict[str, Any] = {}
                if hasattr(step, "accepted_length"):
                    info["acceptance_length"] = step.accepted_length
                elif hasattr(step, "acceptance_length"):
                    info["acceptance_length"] = step.acceptance_length
                if hasattr(step, "generation_tps"):
                    info["generation_tps"] = step.generation_tps
                yield text, info

        return _factory

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_override(channel: MLXChannel, override: dict[str, Any]) -> MLXChannel:
        return MLXChannel(
            variant=override.get("variant", channel.variant),
            q_range=int(override.get("q_range", channel.q_range)),
            boundary=int(override.get("boundary", channel.boundary)),
            est_compression=channel.est_compression,
            est_delta_ppl_pct=channel.est_delta_ppl_pct,
            label=channel.label,
            dflash_draft_repo=channel.dflash_draft_repo,
            dflash_available=channel.dflash_available,
        )

    @staticmethod
    def _render_prompt(lm: _LoadedMLXModel, messages: list[dict[str, Any]]) -> str:
        """Apply the target model's chat template to the message list."""
        tok = lm.tokenizer
        apply = getattr(tok, "apply_chat_template", None)
        if callable(apply):
            return apply(messages, tokenize=False, add_generation_prompt=True)
        # Absolute fallback — a flat prompt, used only if the tokenizer
        # has no chat template (rare for the curated B2 registry).
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"<|{role}|>\n{content}")
        lines.append("<|assistant|>\n")
        return "\n".join(lines)

    @staticmethod
    def _stats(
        channel: MLXChannel,
        t0: float,
        pieces: list[str],
        accept_lens: list[int],
        *,
        dflash_used: bool,
        lm: _LoadedMLXModel,
    ) -> dict[str, Any]:
        gen_time = time.time() - t0
        mean_accept = (
            sum(accept_lens) / len(accept_lens) if accept_lens else None
        )
        return {
            "variant": channel.variant,
            "q_range": channel.q_range,
            "boundary": channel.boundary,
            "est_compression": channel.est_compression,
            "est_delta_ppl_pct": channel.est_delta_ppl_pct,
            "dflash_used": dflash_used,
            "injection_strategy": lm._injection_decision.strategy.value,
            "dflash_draft_repo": channel.dflash_draft_repo,
            "generation_time_s": gen_time,
            "acceptance_length_mean": mean_accept,
            "generated_chars": sum(len(p) for p in pieces),
        }
