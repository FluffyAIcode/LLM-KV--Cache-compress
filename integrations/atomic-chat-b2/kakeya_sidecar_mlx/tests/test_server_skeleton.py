"""Server smoke tests for the B2 M1-M3 skeleton.

These exercise the routing shape without MLX loaded:
- /v1/models returns valid MLX-flavoured entries (incl. dflash_draft_repo)
- /v1/chat/completions returns 503 with a ROADMAP pointer
- /health / /v1/kakeya/stats return the B2 variant tag
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kakeya_sidecar_mlx.server import create_app  # noqa: E402


@pytest.fixture()
def app():
    return create_app(lazy_engine=True)


def test_health_reports_b2_variant(app) -> None:
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "B2" in body["variant"]
    # M4: health carries explicit milestone status.
    assert "M4" in body["milestone"]


def test_models_exposes_dflash_metadata(app) -> None:
    c = TestClient(app)
    r = c.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    by_id = {m["id"]: m for m in data}

    assert "qwen3-8b@e8-q38" in by_id
    m = by_id["qwen3-8b@e8-q38"]
    assert m["owned_by"] == "kakeyalattice-mlx"
    xk = m["x_kakeya"]
    assert xk["dflash_available"] is True
    assert xk["dflash_draft_repo"] == "z-lab/Qwen3-8B-DFlash-b16"
    assert xk["variant"] == "e8"
    assert xk["q_range"] == 38


def test_chat_unknown_model_returns_404(app, monkeypatch) -> None:
    """M4 note: /v1/chat/completions is live; we validate its routing
    by pointing it at a model that doesn't exist in the MLX registry.
    The KeyError path is platform-agnostic — no mlx/mlx-lm required.
    """
    from kakeya_sidecar_mlx import server as srv_mod

    class _FakeEngine:
        def chat(self, *_a, **_kw):
            raise KeyError("unknown MLX model id 'nonexistent-9000b'")

    monkeypatch.setattr(srv_mod, "MLXEngine", lambda *a, **kw: _FakeEngine())

    # Force the app to build a fresh engine via the monkey-patched ctor.
    app2 = srv_mod.create_app(lazy_engine=True)
    c2 = TestClient(app2)
    r = c2.post("/v1/chat/completions", json={
        "model": "nonexistent-9000b@e8-q10",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 404


def test_stats_no_engine(app) -> None:
    c = TestClient(app)
    r = c.get("/v1/kakeya/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["engine_loaded"] is False
    assert body["variant"] == "B2"


def test_mistral_has_no_dflash_in_models_list(app) -> None:
    c = TestClient(app)
    r = c.get("/v1/models")
    data = r.json()["data"]
    mistral_entries = [m for m in data
                       if m["id"].startswith("mistral-7b-instruct")]
    assert mistral_entries, "mistral entries missing from /v1/models"
    for m in mistral_entries:
        assert m["x_kakeya"]["dflash_available"] is False
        assert m["x_kakeya"]["dflash_draft_repo"] is None
