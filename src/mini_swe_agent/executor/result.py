"""Execution result data class."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExecutionResult:
    """Result of executing a shell command."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    exception: str | None = None
    elapsed: float = 0.0
    timed_out: bool = False
    command: str = ""
