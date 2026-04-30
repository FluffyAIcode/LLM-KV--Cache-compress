"""Inference engine: HuggingFace transformers + KakeyaLatticeCache.

The engine is intentionally small; there is no batching, no dynamic
multi-tenant scheduler, no speculative decoding — Atomic-Chat's
single-user desktop use case does not need any of that.

Design:

- One ``_LoadedModel`` per (short_id) with an LRU eviction policy of
  size ``max_resident`` (default 1). Switching models unloads the
  previous one, same as Atomic-Chat's llama.cpp plugin already does.
- ``KakeyaLatticeCache`` is **per request** (caches have sequence state
  tied to the generation). We rebuild it fresh for each call because
  that matches the standard HF generate pattern and makes concurrent
  requests trivially safe.
- All torch imports are lazy so the CLI can `--help` with no torch
  installed.

The engine does NOT depend on FastAPI — the server module wires it up.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterator

from .model_registry import Channel, DeploymentProfile, resolve_model

log = logging.getLogger("kakeya_sidecar.engine")


@dataclass
class EngineConfig:
    device: str = "auto"        # "auto" | "mps" | "cuda" | "cpu"
    dtype: str = "auto"         # "auto" | "bfloat16" | "float16" | "float32"
    max_resident: int = 1       # LRU size of loaded models
    trust_remote_code: bool = True
    hf_cache_dir: str | None = None


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch  # type: ignore
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(requested: str, device: str):
    import torch  # type: ignore

    if requested != "auto":
        return getattr(torch, requested)
    # MPS supports bf16 on macOS 14+; fall back to fp16 on older.
    if device == "mps":
        return torch.float16
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


class _LoadedModel:
    """HF model + tokenizer, loaded on a specific device/dtype."""

    def __init__(
        self,
        profile: DeploymentProfile,
        cfg: EngineConfig,
        device: str,
        dtype,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        log.info("loading %s on %s/%s …", profile.hf_repo_id, device, dtype)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            profile.hf_repo_id,
            trust_remote_code=cfg.trust_remote_code,
            cache_dir=cfg.hf_cache_dir,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            profile.hf_repo_id,
            torch_dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            cache_dir=cfg.hf_cache_dir,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.dtype = dtype
        self.profile = profile
        log.info(
            "loaded %s in %.1fs  (L=%d, head_dim=%r)",
            profile.hf_repo_id, time.time() - t0,
            self.model.config.num_hidden_layers,
            getattr(self.model.config, "head_dim", "?"),
        )


class KakeyaEngine:
    """Public-facing inference engine used by the FastAPI server."""

    def __init__(self, cfg: EngineConfig | None = None) -> None:
        self.cfg = cfg or EngineConfig()
        self._device = _pick_device(self.cfg.device)
        self._loaded: "OrderedDict[str, _LoadedModel]" = OrderedDict()
        self._lock = threading.Lock()
        log.info("KakeyaEngine device=%s max_resident=%d",
                 self._device, self.cfg.max_resident)

    # ------------------------------------------------------------------
    # model lifecycle
    # ------------------------------------------------------------------

    def _ensure_loaded(self, profile: DeploymentProfile) -> _LoadedModel:
        with self._lock:
            if profile.short_id in self._loaded:
                self._loaded.move_to_end(profile.short_id)
                return self._loaded[profile.short_id]

            import torch  # type: ignore
            dtype = _pick_dtype(self.cfg.dtype, self._device)
            lm = _LoadedModel(profile, self.cfg, self._device, dtype)

            self._loaded[profile.short_id] = lm
            while len(self._loaded) > self.cfg.max_resident:
                evicted_id, evicted = self._loaded.popitem(last=False)
                log.info("evicting %s", evicted_id)
                del evicted
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return lm

    def warmup(self, channel_id: str) -> None:
        """Pre-load a model (used by ``--prewarm`` CLI flag)."""
        profile, _ = resolve_model(channel_id)
        self._ensure_loaded(profile)

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------

    def _build_cache(
        self, lm: _LoadedModel, channel: Channel
    ):
        from kakeyalattice.hf import KakeyaLatticeCache  # type: ignore

        head_dim = getattr(lm.model.config, "head_dim", None)
        if head_dim is None:
            hidden = lm.model.config.hidden_size
            n_heads = lm.model.config.num_attention_heads
            head_dim = hidden // n_heads

        return KakeyaLatticeCache(
            variant=channel.variant,
            q_range=channel.q_range,
            num_hidden_layers=lm.model.config.num_hidden_layers,
            head_dim=int(head_dim),
            device=lm.device,
            boundary=channel.boundary,
            strict=False,
        )

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
        """One-shot (non-streaming) chat completion.

        Returns ``(text, stats)`` where ``stats`` contains kakeya-specific
        telemetry (codec fire counts etc.) for the ``x_kakeya`` field.
        """
        profile, channel = resolve_model(channel_id)
        if override:
            channel = Channel(
                variant=override.get("variant", channel.variant),
                q_range=int(override.get("q_range", channel.q_range)),
                boundary=int(override.get("boundary", channel.boundary)),
                est_compression=channel.est_compression,
                est_delta_ppl_pct=channel.est_delta_ppl_pct,
                label=channel.label,
            )

        lm = self._ensure_loaded(profile)
        cache = self._build_cache(lm, channel)

        import torch  # type: ignore

        prompt = lm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = lm.tokenizer(prompt, return_tensors="pt").to(lm.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            past_key_values=cache,
            pad_token_id=lm.tokenizer.pad_token_id,
        )

        t0 = time.time()
        with torch.inference_mode():
            out = lm.model.generate(**inputs, **gen_kwargs)
        gen_time = time.time() - t0

        new_tokens = out[0, prompt_len:]
        text = lm.tokenizer.decode(new_tokens, skip_special_tokens=True)

        stats = {
            "variant": channel.variant,
            "q_range": channel.q_range,
            "boundary": channel.boundary,
            "est_compression": channel.est_compression,
            "est_delta_ppl_pct": channel.est_delta_ppl_pct,
            "prompt_tokens": int(prompt_len),
            "completion_tokens": int(new_tokens.shape[0]),
            "generation_time_s": gen_time,
            "codec_fired_per_layer": dict(getattr(cache, "codec_fired_per_layer", {})),
            "skip_fired_per_layer": dict(getattr(cache, "skip_fired_per_layer", {})),
        }
        return text, stats

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
        """Streaming variant — yields delta strings (not full SSE frames).

        The server wraps each delta into an OpenAI ``chat.completion.chunk``.
        """
        profile, channel = resolve_model(channel_id)
        if override:
            channel = Channel(
                variant=override.get("variant", channel.variant),
                q_range=int(override.get("q_range", channel.q_range)),
                boundary=int(override.get("boundary", channel.boundary)),
                est_compression=channel.est_compression,
                est_delta_ppl_pct=channel.est_delta_ppl_pct,
                label=channel.label,
            )
        lm = self._ensure_loaded(profile)
        cache = self._build_cache(lm, channel)

        import torch  # type: ignore
        from transformers import TextIteratorStreamer  # type: ignore

        prompt = lm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = lm.tokenizer(prompt, return_tensors="pt").to(lm.device)
        streamer = TextIteratorStreamer(
            lm.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs: dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            past_key_values=cache,
            streamer=streamer,
            pad_token_id=lm.tokenizer.pad_token_id,
        )

        def _run():
            with torch.inference_mode():
                lm.model.generate(**gen_kwargs)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        for piece in streamer:
            if piece:
                yield piece
        thread.join()
