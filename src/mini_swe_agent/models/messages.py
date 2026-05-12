"""Message and response data models for model interactions."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool call from the model."""

    id: str
    type: Literal["function"] = "function"
    function_name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    """A single conversation message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Usage(BaseModel):
    """Token usage for a single model call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class ModelResponse(BaseModel):
    """Response from a model call."""

    messages: list[Message]
    usage: Usage = Field(default_factory=Usage)
    cost: float = 0.0
    latency: float = 0.0
    stop_reason: str = ""
