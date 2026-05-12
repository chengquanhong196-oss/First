"""Docker-based sandbox executor (optional)."""

import logging

from mini_swe_agent.executor.base import SandboxExecutor
from mini_swe_agent.executor.result import ExecutionResult

logger = logging.getLogger(__name__)


class DockerExecutor(SandboxExecutor):
    """Execute shell commands inside a Docker container.

    This is an optional backend. If Docker is not available, initialization
    logs a warning and the backend is skipped gracefully.
    """

    def __init__(self, image: str, timeout: float = 120.0) -> None:
        self._image = image
        self._timeout = timeout
        self._available = self._check_docker()

    @staticmethod
    def _check_docker() -> bool:
        import shutil

        if shutil.which("docker") is None:
            logger.warning("Docker not found. DockerExecutor unavailable.")
            return False
        return True

    @property
    def backend_name(self) -> str:
        return "docker"

    @property
    def is_available(self) -> bool:
        return self._available

    async def execute(self, command: str, timeout: float) -> ExecutionResult:
        """Execute a command inside the configured Docker container."""
        if not self._available:
            return ExecutionResult(
                returncode=-1,
                exception="Docker is not available on this system.",
                command=command,
            )

        import asyncio
        import shlex
        import time

        start = time.monotonic()
        docker_cmd = [
            "docker", "exec", "-i",
            self._image,
            "bash", "-c", command,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
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
                return ExecutionResult(
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                    elapsed=time.monotonic() - start,
                    command=command,
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ExecutionResult(
                    returncode=-1,
                    timed_out=True,
                    exception=f"Command timed out after {timeout}s",
                    elapsed=time.monotonic() - start,
                    command=command,
                )
        except Exception as e:
            return ExecutionResult(
                returncode=-1,
                exception=str(e),
                elapsed=time.monotonic() - start,
                command=command,
            )
