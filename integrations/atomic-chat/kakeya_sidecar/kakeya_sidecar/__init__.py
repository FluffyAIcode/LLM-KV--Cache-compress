"""kakeya_sidecar — OpenAI-compatible local inference sidecar.

Top-level imports are lazy so that pure-logic modules (notably
:mod:`model_registry`) can be imported in test / packaging
environments where FastAPI or torch may not be installed.
"""
from __future__ import annotations

from .model_registry import MODEL_REGISTRY, DeploymentProfile, resolve_model

__all__ = [
    "MODEL_REGISTRY",
    "DeploymentProfile",
    "resolve_model",
    "KakeyaEngine",
    "create_app",
]

__version__ = "0.1.0"


def __getattr__(name):
    if name == "KakeyaEngine":
        from .engine import KakeyaEngine
        return KakeyaEngine
    if name == "create_app":
        from .server import create_app
        return create_app
    raise AttributeError(f"module 'kakeya_sidecar' has no attribute {name!r}")
