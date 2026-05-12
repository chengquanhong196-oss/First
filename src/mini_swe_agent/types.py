"""Shared enums and type aliases for the mini SWE agent."""

from enum import Enum


class TerminalState(str, Enum):
    """Possible terminal states for the agent loop."""

    SUBMITTED = "submitted"
    LIMIT_STEP = "limit_step"
    LIMIT_COST = "limit_cost"
    INTERRUPT = "interrupt"
    FATAL_CONFIG = "fatal_config"


class RunMode(str, Enum):
    """Execution mode: confirm each action or auto-execute."""

    CONFIRM = "confirm"
    YOLO = "yolo"


class ActionFamily(str, Enum):
    """Detected action format in a model response."""

    TOOL_CALL = "tool_call"
    TEXT = "text"
    NONE = "none"
