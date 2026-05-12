"""Local subprocess executor."""
from __future__ import annotations

import asyncio
import logging
import time
import traceback

from mini_swe_agent.executor.base import SandboxExecutor
from mini_swe_agent.executor.result import ExecutionResult

logger = logging.getLogger(__name__)


class LocalExecutor(SandboxExecutor):
    """Execute shell commands on the local machine via subprocess."""

    @property
    def backend_name(self) -> str:
        return "local"

    async def execute(self, command: str, timeout: float) -> ExecutionResult:
        """Run a shell command locally with timeout.

        Uses asyncio.create_subprocess_shell to capture stdout/stderr
        and enforce a timeout via asyncio.wait_for.
        """
        start = time.monotonic()
        timed_out = False
        exception: str | None = None
        stdout = ""
        stderr = ""
        returncode = -1

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                returncode = proc.returncode if proc.returncode is not None else -1
            except asyncio.TimeoutError:
                timed_out = True
                exception = f"Command timed out after {timeout}s"
                try:
                    proc.kill()
                    stdout_bytes, stderr_bytes = await proc.communicate()
                    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
                    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
                except Exception:
                    pass
        except Exception as e:
            exception = "".join(
                traceback.format_exception_only(type(e), e)
            )

        elapsed = time.monotonic() - start

        return ExecutionResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            exception=exception,
            elapsed=elapsed,
            timed_out=timed_out,
            command=command,
        )
