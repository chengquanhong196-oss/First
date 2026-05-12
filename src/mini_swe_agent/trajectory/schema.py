"""Trajectory data models for recording agent runs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """A single execution observation."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    exception: str | None = None
    timed_out: bool = False
    elapsed: float = 0.0
    tool_call_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Step(BaseModel):
    """A single agent step: model response, parsed action, observation."""

    step_index: int
    messages_before: list[dict[str, Any]] = Field(default_factory=list)
    assistant_message: dict[str, Any] | None = None
    action: str | None = None
    action_family: str | None = None
    observation: Observation | None = None
    format_error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Trajectory(BaseModel):
    """Complete trajectory of an agent run."""

    task: str
    model_name: str
    terminal_state: str
    steps: list[Step] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    total_cost: float = 0.0
    total_steps: int = 0
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    config: dict[str, Any] = Field(default_factory=dict)
