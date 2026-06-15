"""OpenAI-compatible request / response schemas (minimal subset).

We deliberately model only the fields Atomic-Chat's client actually
sends. Anything unknown is accepted via ``model_config = {"extra": "allow"}``
so we forward-propagate.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    # Extension: override the per-model default channel per-request.
    # Example: {"variant": "e8", "q_range": 38, "boundary": 0}
    x_kakeya_override: dict[str, Any] | None = Field(default=None, alias="x_kakeya_override")


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
    x_kakeya: dict[str, Any] | None = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "kakeyalattice"
    x_kakeya: dict[str, Any] | None = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
