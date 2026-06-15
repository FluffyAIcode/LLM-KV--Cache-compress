"""FastAPI OpenAI-compatible server.

Endpoints:

    GET  /health
    GET  /v1/models
    POST /v1/chat/completions      (stream + non-stream)
    GET  /v1/kakeya/stats          (extension)

The server imports :mod:`kakeya_sidecar.engine` lazily so unit tests
that only exercise routing / schema logic can run without torch.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .engine import EngineConfig, KakeyaEngine
from .model_registry import MODEL_REGISTRY
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ModelInfo,
    ModelList,
)

log = logging.getLogger("kakeya_sidecar.server")


def create_app(
    cfg: EngineConfig | None = None,
    *,
    lazy_engine: bool = True,
    engine_instance: KakeyaEngine | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        cfg: engine configuration (device, dtype, max_resident).
        lazy_engine: if True, the engine is constructed on first use
            instead of app startup. Default True — this lets the process
            start fast and surface model-load errors on the first
            request instead of at boot.
        engine_instance: optional pre-built engine (used by ``--prewarm``).
    """
    app = FastAPI(title="kakeya-sidecar", version="0.1.0")

    state: dict[str, Any] = {"engine": engine_instance, "cfg": cfg or EngineConfig()}

    def engine() -> KakeyaEngine:
        if state["engine"] is None:
            state["engine"] = KakeyaEngine(state["cfg"])
        return state["engine"]  # type: ignore[return-value]

    if not lazy_engine and state["engine"] is None:
        state["engine"] = KakeyaEngine(state["cfg"])

    # -------------------------------------------------------------- health

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "engine_loaded": state["engine"] is not None}

    # ------------------------------------------------------------- /models

    @app.get("/v1/models", response_model=ModelList)
    def list_models() -> ModelList:
        data: list[ModelInfo] = []
        for profile in MODEL_REGISTRY.values():
            for ch in profile.channels:
                data.append(
                    ModelInfo(
                        id=profile.channel_id(ch),
                        owned_by="kakeyalattice",
                        x_kakeya={
                            "hf_repo_id": profile.hf_repo_id,
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
                        },
                    )
                )
        return ModelList(data=data)

    # --------------------------------------------------- /chat/completions

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        try:
            eng = engine()
        except Exception as e:  # pragma: no cover
            raise HTTPException(500, f"engine init failed: {e}") from e

        if req.stream:
            return StreamingResponse(
                _sse_stream(eng, req, cid, created),
                media_type="text/event-stream",
            )

        try:
            text, stats = eng.chat(
                req.model,
                messages,
                max_tokens=req.max_tokens or 512,
                temperature=req.temperature,
                top_p=req.top_p,
                override=req.x_kakeya_override,
            )
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except Exception as e:  # pragma: no cover
            log.exception("chat failed")
            raise HTTPException(500, str(e)) from e

        resp = ChatCompletionResponse(
            id=cid,
            created=created,
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(content=text),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=stats["prompt_tokens"],
                completion_tokens=stats["completion_tokens"],
                total_tokens=stats["prompt_tokens"] + stats["completion_tokens"],
            ),
            x_kakeya=stats,
        )
        return JSONResponse(resp.model_dump())

    # ------------------------------------------------- /v1/kakeya/stats

    @app.get("/v1/kakeya/stats")
    def kakeya_stats() -> dict[str, Any]:
        eng = state["engine"]
        if eng is None:
            return {"engine_loaded": False}
        return {
            "engine_loaded": True,
            "device": eng._device,
            "resident_models": list(eng._loaded.keys()),
            "max_resident": eng.cfg.max_resident,
        }

    return app


# --------------------------------------------------------------- sse helper


def _sse_stream(eng: KakeyaEngine, req: ChatCompletionRequest, cid: str, created: int):
    messages = [m.model_dump(exclude_none=True) for m in req.messages]

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant"})
    try:
        for piece in eng.chat_stream(
            req.model,
            messages,
            max_tokens=req.max_tokens or 512,
            temperature=req.temperature,
            top_p=req.top_p,
            override=req.x_kakeya_override,
        ):
            yield chunk({"content": piece})
    except KeyError as e:
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
