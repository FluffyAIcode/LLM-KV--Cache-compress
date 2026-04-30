"""MLX inference engine — *skeleton* for B2.

This file intentionally stops short of a working generate() loop; that
work is gated on M4 (DFlash integration) and belongs in a separate PR.
What IS finalised here:

- ``MLXEngineConfig`` dataclass with the same shape as B1's
  ``EngineConfig`` so Atomic-Chat's plugin can swap sidecars.
- ``MLXEngine`` with ``_ensure_loaded`` LRU and a clear
  ``NotImplementedError`` on ``.chat()`` / ``.chat_stream()`` that
  points downstream PRs at what still needs implementing.
- The warmup path (model load only) IS implemented, so
  ``kakeya-sidecar-mlx --prewarm <id>`` is enough to validate
  that weights load on Apple Silicon.

Why skeleton-first: the mlx-lm API is a moving target (0.20, 0.21
reshaped ``generate_step`` signatures). Locking in a dummy chat path
would either (a) pin us to a fragile version or (b) silently skew
from B1's behaviour. Better to gate the real implementation on a
dedicated PR that CI-validates on a Mac runner.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterator

from .model_registry_mlx import (
    MLXChannel,
    MLXDeploymentProfile,
    resolve_mlx_model,
)

log = logging.getLogger("kakeya_sidecar_mlx.engine")


@dataclass
class MLXEngineConfig:
    device: str = "auto"         # "auto" | "mps" | "cpu"
    dtype: str = "auto"          # "auto" | "bfloat16" | "float16" | "float32"
    max_resident: int = 1
    enable_dflash: bool = False
    trust_remote_code: bool = True
    hf_cache_dir: str | None = None

    # Runtime prefs set by --enable-dflash + channel.dflash_available.
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


class _LoadedMLXModel:
    """Holds a loaded mlx-lm model + tokenizer + optional DFlash draft."""

    def __init__(
        self,
        profile: MLXDeploymentProfile,
        cfg: MLXEngineConfig,
        channel: MLXChannel,
    ) -> None:
        from mlx_lm import load  # type: ignore

        repo = profile.mlx_repo_id or profile.hf_repo_id
        log.info("mlx_lm.load(%s) ...", repo)
        t0 = time.time()
        self.model, self.tokenizer = load(repo)
        log.info("loaded target %s in %.1fs", repo, time.time() - t0)

        self.profile = profile
        self.channel = channel
        self.draft_model = None
        self.draft_tokenizer = None

        if cfg.enable_dflash and channel.dflash_available:
            self._load_dflash_draft(channel)

    def _load_dflash_draft(self, channel: MLXChannel) -> None:
        repo = channel.dflash_draft_repo
        if repo is None:
            return
        try:
            from dflash.model_mlx import load_draft  # type: ignore
        except ImportError:
            log.warning(
                "dflash not installed; install with `pip install dflash` "
                "to enable speculative decoding. Falling back to "
                "single-track decode."
            )
            return
        log.info("dflash.load_draft(%s) ...", repo)
        t0 = time.time()
        self.draft_model = load_draft(repo)
        log.info("loaded draft %s in %.1fs", repo, time.time() - t0)


class MLXEngine:
    """Skeleton MLX engine. Implements warmup + LRU; defers generate."""

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
        raise NotImplementedError(
            "MLXEngine.chat() is a M4 deliverable. "
            "Current PR (M1-M3) ships only model loading, registry, "
            "and server routing. See integrations/atomic-chat-b2/ROADMAP.md "
            "M4 for the DFlash-integrated generate loop."
        )

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
        raise NotImplementedError(
            "MLXEngine.chat_stream() is a M4 deliverable."
        )
