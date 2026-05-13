"""Action family detection: determines whether the model used tool-call or text format."""

from __future__ import annotations

import re

from mini_swe_agent.models.messages import ModelResponse
from mini_swe_agent.parser.errors import FormatError
from mini_swe_agent.types import ActionFamily

_TEXT_FENCED_RE = re.compile(r"```mswea_bash_command\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TEXT_XML_RE = re.compile(
    r"<mswea_bash_command>\s*(.*?)</mswea_bash_command>", re.DOTALL | re.IGNORECASE
)


def _count_text_actions(text: str) -> int:
    """Count both fenced and XML mswea_bash_command blocks in text."""
    fenced = len(_TEXT_FENCED_RE.findall(text))
    xml = len(_TEXT_XML_RE.findall(text))
    return fenced + xml


def _has_tool_calls(response: ModelResponse) -> bool:
    """Check whether any message in the response contains tool calls."""
    for msg in response.messages:
        if msg.tool_calls and len(msg.tool_calls) > 0:
            return True
    return False


def _concat_text_content(response: ModelResponse) -> str:
    """Extract all text content from response messages into a single string."""
    parts = []
    for msg in response.messages:
        if msg.content and isinstance(msg.content, str):
            parts.append(msg.content)
        elif msg.content and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts)


def detect_action_family(response: ModelResponse) -> ActionFamily:
    """Determine the action family used in the model response.

    Returns ActionFamily.TOOL_CALL or ActionFamily.TEXT.
    Raises FormatError if:
    - Both families are present (tool + text conflicting)
    - No action is present
    """
    has_tc = _has_tool_calls(response)
    text = _concat_text_content(response)
    text_action_count = _count_text_actions(text)

    has_text_markers = text_action_count > 0

    if has_tc and has_text_markers:
        raise FormatError(
            reason="both_families",
            feedback=(
                "Your response contains BOTH tool calls AND text command blocks "
                "(```mswea_bash_command``` or <mswea_bash_command>). "
                "Use exactly ONE method. Provide exactly one command."
            ),
        )

    if has_tc:
        return ActionFamily.TOOL_CALL

    if has_text_markers:
        return ActionFamily.TEXT

    raise FormatError(
        reason="zero_actions",
        feedback=(
            "No executable action found in your response. "
            "Provide exactly one shell command using either a bash tool call "
            "or a ```mswea_bash_command``` / <mswea_bash_command> block."
        ),
    )
