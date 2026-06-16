"""B2 FastAPI server — same route shape as B1, MLX-specific metadata.

Endpoints:

    GET  /health
    GET  /v1/models
    POST /v1/chat/completions           (stream + non-stream — **503 until M4**)
    GET  /v1/kakeya/stats

Until the M4 PR lands, ``/v1/chat/completions`` returns HTTP 503 with
a body pointing at ``ROADMAP.md``. This is deliberate — better a clean
503 than a half-working chat that diverges from B1.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .engine_mlx import MLXEngine, MLXEngineConfig
from .model_registry_mlx import MODEL_REGISTRY_MLX

log = logging.getLogger("kakeya_sidecar_mlx.server")


def create_app(
    cfg: MLXEngineConfig | None = None,
    *,
    lazy_engine: bool = True,
    engine_instance: MLXEngine | None = None,
) -> FastAPI:
    app = FastAPI(title="kakeya-sidecar-mlx", version="0.1.0")

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
            "milestone": "M1-M3 skeleton; /v1/chat/completions disabled until M4",
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
    def chat_completions(_body: dict[str, Any]) -> JSONResponse:
        raise HTTPException(
            status_code=503,
            detail=(
                "B2 sidecar is at M1-M3 skeleton stage. "
                "Chat completion will be enabled in the M4 PR "
                "(DFlash integration). For now please use the B1 "
                "sidecar on :1338."
            ),
        )

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
