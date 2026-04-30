"""B2 FastAPI server — M4 opens /v1/chat/completions.

Route shape mirrors B1 (PR #57):
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions      (stream + non-stream)
  GET  /v1/kakeya/stats

The B2-specific surface additions:
  - /v1/models entries carry ``x_kakeya.dflash_draft_repo`` and
    ``x_kakeya.dflash_available``.
  - /health reports the MLX backend variant and whether DFlash is
    enabled on this engine instance.
  - /v1/chat/completions response ``x_kakeya`` carries ``dflash_used``,
    ``injection_strategy``, and ``acceptance_length_mean``.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .engine_mlx import MLXEngine, MLXEngineConfig
from .model_registry_mlx import MODEL_REGISTRY_MLX

log = logging.getLogger("kakeya_sidecar_mlx.server")


# ---------------------------------------------------------------------------
# request / response schemas (subset of OpenAI spec + x_kakeya extension)
# ---------------------------------------------------------------------------


class _ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any


class _ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[_ChatMessage]
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int | None = None
    stop: Any | None = None
    x_kakeya_override: dict[str, Any] | None = Field(
        default=None, alias="x_kakeya_override",
    )


# ---------------------------------------------------------------------------
# app factory
# ---------------------------------------------------------------------------


def create_app(
    cfg: MLXEngineConfig | None = None,
    *,
    lazy_engine: bool = True,
    engine_instance: MLXEngine | None = None,
) -> FastAPI:
    app = FastAPI(title="kakeya-sidecar-mlx", version="0.2.0")

    state: dict[str, Any] = {
        "engine": engine_instance,
        "cfg": cfg or MLXEngineConfig(),
    }

    def engine() -> MLXEngine:
        if state["engine"] is None:
            state["engine"] = MLXEngine(state["cfg"])
        return state["engine"]  # type: ignore[return-value]

    if not lazy_engine and state["engine"] is None:
        state["engine"] = MLXEngine(state["cfg"])

    # -------------------------------------------------------------- health

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "engine_loaded": state["engine"] is not None,
            "variant": "B2 (MLX + DFlash + KakeyaLattice)",
            "dflash_enabled": state["cfg"].enable_dflash,
            "milestone": "M4 — chat completions live (DFlash + KV compression)",
        }

    # ------------------------------------------------------------- /models

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        data: list[dict[str, Any]] = []
        for profile in MODEL_REGISTRY_MLX.values():
            for ch in profile.channels:
                data.append({
                    "id": profile.channel_id(ch),
                    "object": "model",
                    "owned_by": "kakeyalattice-mlx",
                    "x_kakeya": {
                        "hf_repo_id": profile.hf_repo_id,
                        "mlx_repo_id": profile.mlx_repo_id,
                        "head_dim": profile.head_dim,
                        "num_hidden_layers": profile.num_hidden_layers,
                        "variant": ch.variant,
                        "q_range": ch.q_range,
                        "boundary": ch.boundary,
                        "est_compression": ch.est_compression,
                        "est_delta_ppl_pct": ch.est_delta_ppl_pct,
                        "label": ch.label,
                        "is_default": ch == profile.default_channel,
                        "notes": profile.notes,
                        "dflash_draft_repo": ch.dflash_draft_repo,
                        "dflash_available": ch.dflash_available,
                    },
                })
        return {"object": "list", "data": data}

    # --------------------------------------------------- /chat/completions

    @app.post("/v1/chat/completions")
    def chat_completions(req: _ChatCompletionRequest):
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        try:
            eng = engine()
        except Exception as e:  # pragma: no cover
            raise HTTPException(500, f"engine init failed: {e}") from e

        max_tokens = req.max_tokens or 512
        stop = (
            [req.stop] if isinstance(req.stop, str)
            else (req.stop if isinstance(req.stop, list) else None)
        )

        if req.stream:
            return StreamingResponse(
                _sse_stream(
                    eng, req, cid, created,
                    messages, max_tokens, stop,
                ),
                media_type="text/event-stream",
            )

        try:
            text, stats = eng.chat(
                req.model,
                messages,
                max_tokens=max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=stop,
                override=req.x_kakeya_override,
            )
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except NotImplementedError as e:
            raise HTTPException(501, str(e)) from e
        except Exception as e:  # pragma: no cover
            log.exception("chat failed")
            raise HTTPException(500, str(e)) from e

        return JSONResponse({
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": stats.get("generated_chars", 0),
                "total_tokens": stats.get("generated_chars", 0),
            },
            "x_kakeya": stats,
        })

    # ------------------------------------------------- /v1/kakeya/stats

    @app.get("/v1/kakeya/stats")
    def kakeya_stats() -> dict[str, Any]:
        eng = state["engine"]
        if eng is None:
            return {"engine_loaded": False, "variant": "B2"}
        return {
            "engine_loaded": True,
            "variant": "B2",
            "device": eng._device,
            "resident_models": list(eng._loaded.keys()),
            "max_resident": eng.cfg.max_resident,
            "dflash_enabled": eng.cfg.enable_dflash,
        }

    return app


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


def _sse_stream(eng, req, cid: str, created: int,
                messages, max_tokens: int, stop):
    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant"})
    try:
        for piece in eng.chat_stream(
            req.model,
            messages,
            max_tokens=max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=stop,
            override=req.x_kakeya_override,
        ):
            if piece:
                yield chunk({"content": piece})
    except KeyError as e:
        yield chunk({"content": f"[error] {e}"}, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return
    except NotImplementedError as e:
        yield chunk({"content": f"[error] {e}"}, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return
    except Exception as e:  # pragma: no cover
        log.exception("stream failed")
        yield chunk({"content": f"[error] {e}"}, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    yield chunk({}, finish_reason="stop")
    yield "data: [DONE]\n\n"
