"""Unit tests for LocalExecutor with async subprocess mocking."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_swe_agent.executor.local import LocalExecutor
from mini_swe_agent.executor.result import ExecutionResult


class TestLocalExecutor:
    def test_backend_name(self):
        executor = LocalExecutor()
        assert executor.backend_name == "local"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"hello", b""))

        with patch("asyncio.create_subprocess_shell", AsyncMock(return_value=mock_proc)):
            executor = LocalExecutor()
            result = await executor.execute("echo hello", timeout=30.0)

        assert result.returncode == 0
        assert result.stdout == "hello"
        assert result.timed_out is False
        assert result.command == "echo hello"

    @pytest.mark.asyncio
    async def test_execute_returncode_nonzero(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error occurred"))

        with patch("asyncio.create_subprocess_shell", AsyncMock(return_value=mock_proc)):
            executor = LocalExecutor()
            result = await executor.execute("bad-command", timeout=30.0)

        assert result.returncode == 1
        assert result.stderr == "error occurred"

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = -1
        mock_proc.kill = MagicMock()
        mock_proc.communicate = AsyncMock()
        mock_proc.communicate.side_effect = [asyncio.TimeoutError(), (b"partial", b"")]

        with patch("asyncio.create_subprocess_shell", AsyncMock(return_value=mock_proc)):
            executor = LocalExecutor()
            result = await executor.execute("sleep 999", timeout=0.1)

        assert result.timed_out is True
        assert "timed out" in (result.exception or "")

    @pytest.mark.asyncio
    async def test_execute_subprocess_error(self):
        with patch(
            "asyncio.create_subprocess_shell",
            side_effect=FileNotFoundError("shell not found"),
        ):
            executor = LocalExecutor()
            result = await executor.execute("some command", timeout=30.0)

        assert result.returncode == -1
        assert "FileNotFoundError" in (result.exception or "")
