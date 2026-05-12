"""Abstract base class for shell executors."""

from abc import ABC, abstractmethod

from mini_swe_agent.executor.result import ExecutionResult


class SandboxExecutor(ABC):
    """Abstract interface for executing shell commands.

    Implementations can be local subprocess, Docker container, etc.
    """

    @abstractmethod
    async def execute(self, command: str, timeout: float) -> ExecutionResult:
        """Execute a shell command and return the result."""
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier."""
        ...
