"""kakeya_sidecar_mlx — B2 OpenAI-compatible sidecar (MLX + DFlash).

Top-level imports are lazy: pure-logic modules (notably
:mod:`model_registry_mlx`) stay importable without ``mlx`` / ``mlx_lm``
/ ``fastapi`` installed.
"""
from __future__ import annotations

from .model_registry_mlx import (
    MODEL_REGISTRY_MLX,
    MLXDeploymentProfile,
    MLXChannel,
    resolve_mlx_model,
)

__all__ = [
    "MODEL_REGISTRY_MLX",
    "MLXDeploymentProfile",
    "MLXChannel",
    "resolve_mlx_model",
    "MLXEngine",
    "create_app",
]

__version__ = "0.1.0"


def __getattr__(name):
    if name == "MLXEngine":
        from .engine_mlx import MLXEngine
        return MLXEngine
    if name == "create_app":
        from .server import create_app
        return create_app
    raise AttributeError(f"module 'kakeya_sidecar_mlx' has no attribute {name!r}")
