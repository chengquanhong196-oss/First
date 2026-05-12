"""Shared fixtures for all test modules."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_swe_agent.config.schema import Config
from mini_swe_agent.executor.result import ExecutionResult
from mini_swe_agent.models.messages import Message, ModelResponse, ToolCall, Usage


@pytest.fixture
def sample_config_dict() -> dict:
    """A minimal valid configuration dictionary."""
    return {
        "model": {
            "provider": "anthropic",
            "name": "claude-sonnet-4-6",
            "max_tokens": 4096,
        },
        "executor": {
            "backend": "local",
            "timeout": 30.0,
        },
        "limits": {
            "max_steps": 10,
            "max_cost": 1.0,
            "max_consecutive_format_errors": 3,
        },
    }


@pytest.fixture
def sample_config(sample_config_dict) -> Config:
    """A validated Config object."""
    return Config(**sample_config_dict)


@pytest.fixture
def make_message():
    """Factory fixture for creating Message objects."""

    def _make(role="user", content="", **kwargs):
        return Message(role=role, content=content, **kwargs)

    return _make


@pytest.fixture
def make_assistant_with_tool_call():
    """Factory fixture for creating an assistant message with one bash tool call."""

    def _make(command="echo hello", tool_call_id="tc_001"):
        return Message(
            role="assistant",
            content="I'll run a command.",
            tool_calls=[
                ToolCall(
                    id=tool_call_id,
                    function_name="bash",
                    arguments={"command": command},
                )
            ],
        )

    return _make


@pytest.fixture
def make_model_response_from_messages():
    """Factory fixture for creating a ModelResponse from messages."""

    def _make(messages: list[Message], usage=None, cost=0.0):
        return ModelResponse(
            messages=messages,
            usage=usage or Usage(),
            cost=cost,
        )

    return _make


@pytest.fixture
def make_success_result():
    """Factory fixture for creating a successful ExecutionResult."""

    def _make(stdout="", stderr="", returncode=0, command="echo hello"):
        return ExecutionResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            command=command,
        )

    return _make


@pytest.fixture
def temp_trajectory_path(tmp_path):
    """A temporary path for trajectory output."""
    return tmp_path / "test.traj.json"


@pytest.fixture
def temp_dir(tmp_path):
    """A temporary directory for file operations."""
    return tmp_path
