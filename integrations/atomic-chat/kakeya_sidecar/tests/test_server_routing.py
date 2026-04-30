"""Server-level smoke tests — mock the engine so torch is not needed."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

# Must import FastAPI / TestClient lazily so that environments without
# uvicorn can still `python -m compileall`.
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kakeya_sidecar import server as server_mod  # noqa: E402


@pytest.fixture()
def app_with_mock_engine(monkeypatch):
    mock_engine = MagicMock()
    mock_engine._device = "cpu"
    mock_engine._loaded = {}
    mock_engine.cfg.max_resident = 1

    def _fake_chat(channel_id: str, messages, **kw) -> tuple[str, dict[str, Any]]:
        return ("hello from kakeya", {
            "variant": "e8", "q_range": 10, "boundary": 0,
            "est_compression": 3.37, "est_delta_ppl_pct": 3.85,
            "prompt_tokens": 5, "completion_tokens": 4, "generation_time_s": 0.01,
            "codec_fired_per_layer": {}, "skip_fired_per_layer": {},
        })

    mock_engine.chat.side_effect = _fake_chat

    app = server_mod.create_app(lazy_engine=True)

    # Force the route closure to see our mocked engine.
    class _State(dict):
        pass

    # Swap the `engine()` closure by overriding the route function's globals.
    # Easiest path: hit /health to materialise the real engine, then
    # monkeypatch the handler-local `state` by replacing `KakeyaEngine`.
    monkeypatch.setattr(server_mod, "KakeyaEngine", lambda *a, **kw: mock_engine)

    return app, mock_engine


def test_list_models_returns_v15_4models(app_with_mock_engine) -> None:
    app, _ = app_with_mock_engine
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200, r.text
    data = r.json()
    ids = {m["id"] for m in data["data"]}
    # Must include the balanced channel for each v1.5 hero model.
    assert "qwen3-4b@e8-q10" in ids
    assert "gemma-4-e4b@e8-q10" in ids
    assert "glm-4-9b-chat@e8-q10" in ids
    # And the DeepSeek small must carry boundary=2.
    assert "deepseek-r1-distill-qwen-1.5b@e8-q10-b2" in ids


def test_health_before_engine_use(app_with_mock_engine) -> None:
    app, _ = app_with_mock_engine
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_chat_completions_non_stream(app_with_mock_engine) -> None:
    app, mock_engine = app_with_mock_engine
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "model": "qwen3-4b@e8-q10",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "max_tokens": 8,
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["choices"][0]["message"]["content"] == "hello from kakeya"
    assert data["usage"]["total_tokens"] == 9
    assert data["x_kakeya"]["variant"] == "e8"
    mock_engine.chat.assert_called_once()


def test_chat_completions_unknown_model(app_with_mock_engine) -> None:
    app, mock_engine = app_with_mock_engine

    def _raise(channel_id, messages, **kw):
        raise KeyError(f"unknown model id {channel_id!r}")

    mock_engine.chat.side_effect = _raise
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "model": "nonexistent-99b@e8-q10",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 404
