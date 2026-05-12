"""Unit tests for AgentLoop state machine with mocked model/executor."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_swe_agent.config.schema import Config, LimitsConfig
from mini_swe_agent.core.loop import AgentLoop
from mini_swe_agent.core.submission import SUBMISSION_MARKER
from mini_swe_agent.executor.result import ExecutionResult
from mini_swe_agent.models.messages import Message, ModelResponse, ToolCall, Usage
from mini_swe_agent.types import RunMode, TerminalState


class MockModelAdapter:
    """Mock model adapter that returns predefined responses."""

    def __init__(self, responses=None, cost=0.0):
        self._responses = responses or []
        self._call_count = 0
        self._cost = cost
        self._total_cost = 0.0

    async def send(self, messages):
        self._call_count += 1
        if self._call_count <= len(self._responses):
            resp = self._responses[self._call_count - 1]
            self._total_cost += getattr(resp, 'cost', self._cost)
            return resp
        return ModelResponse(
            messages=[Message(role="assistant", content="No more responses.")],
            usage=Usage(),
            cost=self._cost,
        )

    def supports_tool_calls(self):
        return True

    def get_total_cost(self):
        return self._total_cost

    def get_last_cost(self):
        return self._cost

    def reset_cost(self):
        self._total_cost = 0.0

    @property
    def model_name(self):
        return "mock-model"


class MockExecutor:
    """Mock executor that returns predefined results."""

    def __init__(self, results=None):
        self._results = results or []
        self._call_count = 0
        self.commands: list[str] = []

    async def execute(self, command, timeout):
        self.commands.append(command)
        self._call_count += 1
        if self._call_count <= len(self._results):
            return self._results[self._call_count - 1]
        return ExecutionResult(returncode=0, stdout="ok", command=command)

    @property
    def backend_name(self):
        return "mock"


def make_tool_response(command, tool_call_id="tc_001", cost=0.0):
    """Factory: create a ModelResponse with one bash tool call."""
    msg = Message(
        role="assistant",
        content="I will run a command.",
        tool_calls=[
            ToolCall(id=tool_call_id, function_name="bash", arguments={"command": command})
        ],
    )
    return ModelResponse(messages=[msg], usage=Usage(), cost=cost)


def make_submission_response(cost=0.0):
    """Factory: create a ModelResponse that triggers submission."""
    return make_tool_response(
        f"echo '{SUBMISSION_MARKER}\nAll done!'", tool_call_id="tc_sub", cost=cost
    )


def make_minimal_config(**overrides):
    """Factory: create a Config with minimal settings."""
    data = {
        "model": {"provider": "anthropic", "name": "claude-sonnet-4-6", "max_tokens": 4096},
        "executor": {"backend": "local", "timeout": 30.0},
        "limits": {"max_steps": 10, "max_cost": 1.0, "max_consecutive_format_errors": 3},
    }
    data.update(overrides)
    return Config(**data)


class TestAgentLoopTerminalStates:
    """Test that the AgentLoop reaches the correct terminal states."""

    @pytest.mark.asyncio
    async def test_submitted_on_marker(self):
        """Agent reaches SUBMITTED when model outputs the submission marker with rc=0."""
        model = MockModelAdapter(responses=[
            make_submission_response(cost=0.001),
        ])
        executor = MockExecutor(results=[
            ExecutionResult(returncode=0, stdout=f"{SUBMISSION_MARKER}\nPatch here", command="echo ..."),
        ])
        config = make_minimal_config()

        loop = AgentLoop(
            task="Fix the bug",
            model=model,
            executor=executor,
            config=config,
            mode=RunMode.YOLO,
        )

        state, trajectory = await loop.run()
        assert state == TerminalState.SUBMITTED
        assert trajectory.total_steps == 1

    @pytest.mark.asyncio
    async def test_submitted_rejected_when_rc_nonzero(self):
        """returncode != 0 with marker must NOT submit."""
        model = MockModelAdapter(responses=[
            make_submission_response(cost=0.001),
            make_tool_response("echo done", cost=0.001),
        ])
        executor = MockExecutor(results=[
            ExecutionResult(returncode=1, stdout=f"{SUBMISSION_MARKER}\nFail", command="echo ..."),
            ExecutionResult(returncode=0, stdout="done", command="echo done"),
        ])
        config = make_minimal_config()

        loop = AgentLoop(
            task="Fix the bug",
            model=model,
            executor=executor,
            config=config,
            mode=RunMode.YOLO,
        )

        state, trajectory = await loop.run()
        # Should not submit on the first response (rc=1)
        # Second response has no submission marker, so it continues
        # Eventually hits step limit or keeps going
        assert state != TerminalState.SUBMITTED

    @pytest.mark.asyncio
    async def test_limit_step(self):
        """Agent reaches LIMIT_STEP when max_steps is exceeded."""
        model = MockModelAdapter(responses=[
            make_tool_response("echo step", cost=0.0) for _ in range(5)
        ])
        executor = MockExecutor(results=[
            ExecutionResult(returncode=0, stdout="ok", command="echo step") for _ in range(5)
        ])
        config = make_minimal_config(**{"limits": {"max_steps": 2, "max_cost": 100.0, "max_consecutive_format_errors": 3}})

        loop = AgentLoop(
            task="Fix the bug",
            model=model,
            executor=executor,
            config=config,
            mode=RunMode.YOLO,
        )

        state, trajectory = await loop.run()
        assert state == TerminalState.LIMIT_STEP
        assert trajectory.total_steps == 2

    @pytest.mark.asyncio
    async def test_limit_cost(self):
        """Agent reaches LIMIT_COST when cumulative cost exceeds max_cost."""
        model = MockModelAdapter(responses=[
            make_tool_response("echo step1", cost=0.60),
            make_tool_response("echo step2", cost=0.60),
        ])
        executor = MockExecutor(results=[
            ExecutionResult(returncode=0, stdout="ok", command="echo step1"),
            ExecutionResult(returncode=0, stdout="ok", command="echo step2"),
        ])
        config = make_minimal_config(**{
            "limits": {"max_steps": 100, "max_cost": 1.0, "max_consecutive_format_errors": 3},
        })

        loop = AgentLoop(
            task="Fix the bug",
            model=model,
            executor=executor,
            config=config,
            mode=RunMode.YOLO,
        )

        state, trajectory = await loop.run()
        # First call: cost 0.60, total 0.60 (< 1.0) — continues, step counted
        # Second call: cost 0.60, total 1.20 (> 1.0) — LIMIT_COST before step is appended
        assert state == TerminalState.LIMIT_COST
        assert trajectory.total_steps == 1

    @pytest.mark.asyncio
    async def test_fatal_config_on_model_error(self):
        """Agent reaches FATAL_CONFIG when model raises an exception."""
        class FailingModel(MockModelAdapter):
            async def send(self, messages):
                raise ConnectionError("API unreachable")

        model = FailingModel()
        executor = MockExecutor()
        config = make_minimal_config()

        loop = AgentLoop(
            task="Fix the bug",
            model=model,
            executor=executor,
            config=config,
            mode=RunMode.YOLO,
        )

        state, trajectory = await loop.run()
        assert state == TerminalState.FATAL_CONFIG

    @pytest.mark.asyncio
    async def test_fatal_config_on_format_error_limit(self):
        """Agent reaches FATAL_CONFIG after max_consecutive_format_errors."""
        # Return responses with NO commands — each triggers a FormatError
        model = MockModelAdapter(responses=[
            ModelResponse(
                messages=[Message(role="assistant", content="I think about this...")],
                usage=Usage(),
                cost=0.0,
            )
            for _ in range(5)
        ])
        executor = MockExecutor()
        config = make_minimal_config(**{
            "limits": {"max_steps": 100, "max_cost": 100.0, "max_consecutive_format_errors": 2},
        })

        loop = AgentLoop(
            task="Fix the bug",
            model=model,
            executor=executor,
            config=config,
            mode=RunMode.YOLO,
        )

        state, trajectory = await loop.run()
        assert state == TerminalState.FATAL_CONFIG

    @pytest.mark.asyncio
    async def test_trajectory_excludes_api_key(self):
        """Trajectory config must NOT contain api_key."""
        model = MockModelAdapter(responses=[
            make_submission_response(cost=0.001),
        ])
        executor = MockExecutor(results=[
            ExecutionResult(returncode=0, stdout=f"{SUBMISSION_MARKER}\nDone", command="echo ..."),
        ])

        # Use an override dict to simulate api_key being present
        overrides = {"model": {"provider": "anthropic", "name": "test", "max_tokens": 4096, "api_key": "sk-secret-123"}}
        config = Config(**{
            "model": overrides["model"],
            "executor": {"backend": "local", "timeout": 30.0},
            "limits": {"max_steps": 10, "max_cost": 1.0, "max_consecutive_format_errors": 3},
        })

        loop = AgentLoop(
            task="Fix the bug",
            model=model,
            executor=executor,
            config=config,
            mode=RunMode.YOLO,
        )

        state, trajectory = await loop.run()
        assert state == TerminalState.SUBMITTED
        # The serialized config must NOT contain api_key
        assert "api_key" not in trajectory.config.get("model", {})

    @pytest.mark.asyncio
    async def test_executed_commands_recorded(self):
        """The executor should receive the correct commands."""
        model = MockModelAdapter(responses=[
            make_tool_response("ls -la", cost=0.0),
            make_submission_response(cost=0.0),
        ])
        executor = MockExecutor(results=[
            ExecutionResult(returncode=0, stdout="file1.txt", command="ls -la"),
            ExecutionResult(returncode=0, stdout=f"{SUBMISSION_MARKER}\nDone", command="echo ..."),
        ])
        config = make_minimal_config()

        loop = AgentLoop(
            task="List files",
            model=model,
            executor=executor,
            config=config,
            mode=RunMode.YOLO,
        )

        state, trajectory = await loop.run()
        assert state == TerminalState.SUBMITTED
        assert executor.commands[0] == "ls -la"
        assert len(executor.commands) == 2
        assert trajectory.steps[0].action == "ls -la"


class TestAgentLoopConfirmMode:
    """Tests for CONFIRM mode behavior."""

    @pytest.mark.asyncio
    async def test_confirm_reject_then_retry(self):
        """User rejects an action, agent gets another chance."""
        model = MockModelAdapter(responses=[
            make_tool_response("rm -rf /", cost=0.0),
            make_tool_response("ls -la", cost=0.0),
        ])
        executor = MockExecutor(results=[
            ExecutionResult(returncode=0, stdout="ok", command="ls -la"),
        ])
        config = make_minimal_config()

        # Mock input to reject first, accept second
        input_values = ["n", "y"]

        mock_loop = MagicMock()
        mock_loop.run_in_executor = AsyncMock(
            side_effect=lambda executor, func, *args: input_values.pop(0)
        )

        with patch("asyncio.get_event_loop", return_value=mock_loop):
            loop = AgentLoop(
                task="Clean up",
                model=model,
                executor=executor,
                config=config,
                mode=RunMode.CONFIRM,
            )

            state, trajectory = await loop.run()
        # First command (rm -rf /) was rejected
        # Second command (ls -la) was accepted and executed
        assert executor.commands[0] == "ls -la"

    @pytest.mark.asyncio
    async def test_confirm_accept_executes(self):
        """User accepts an action, it proceeds to execute."""
        model = MockModelAdapter(responses=[
            make_submission_response(cost=0.001),
        ])
        executor = MockExecutor(results=[
            ExecutionResult(returncode=0, stdout=f"{SUBMISSION_MARKER}\nDone", command="echo ..."),
        ])
        config = make_minimal_config()

        mock_loop = MagicMock()
        mock_loop.run_in_executor = AsyncMock(return_value="y")

        with patch("asyncio.get_event_loop", return_value=mock_loop):
            loop = AgentLoop(
                task="Fix bug",
                model=model,
                executor=executor,
                config=config,
                mode=RunMode.CONFIRM,
            )

            state, trajectory = await loop.run()
        assert state == TerminalState.SUBMITTED
        assert trajectory.total_steps == 1


class TestAgentLoopInterrupt:
    """Tests for INTERRUPT terminal state."""

    @pytest.mark.asyncio
    async def test_interrupt_on_cancelled_error(self):
        """asyncio.CancelledError leads to INTERRUPT state."""
        class CancellingModel(MockModelAdapter):
            async def send(self, messages):
                raise asyncio.CancelledError()

        model = CancellingModel(responses=[])
        executor = MockExecutor()
        config = make_minimal_config()

        loop = AgentLoop(
            task="Fix bug",
            model=model,
            executor=executor,
            config=config,
            mode=RunMode.YOLO,
        )

        state, trajectory = await loop.run()
        assert state == TerminalState.INTERRUPT
