"""Action extraction: pulls exactly one command from a model response."""

from __future__ import annotations

import re
from typing import Any

from mini_swe_agent.models.messages import ModelResponse
from mini_swe_agent.parser.errors import FormatError
from mini_swe_agent.types import ActionFamily

_TEXT_FENCED_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_TEXT_XML_RE = re.compile(
    r"<mswea_bash_command>\s*(.*?)</mswea_bash_command>", re.DOTALL | re.IGNORECASE
)


def _count_text_actions(text: str) -> int:
    """Count all text command markers in the given text."""
    fenced = len(_TEXT_FENCED_RE.findall(text))
    xml = len(_TEXT_XML_RE.findall(text))
    return fenced + xml


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


def _collect_tool_calls(response: ModelResponse) -> list[dict[str, Any]]:
    """Flatten all tool calls from all assistant messages."""
    all_calls = []
    for msg in response.messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                all_calls.append({
                    "id": tc.id,
                    "function_name": tc.function_name,
                    "arguments": tc.arguments,
                })
    return all_calls


def extract_command(response: ModelResponse, family: ActionFamily) -> tuple[str, str | None]:
    """Extract exactly one command from the model response.

    Returns (command, tool_call_id) where tool_call_id is None for text mode.
    Raises FormatError on 0 or >1 actions.
    """
    if family == ActionFamily.TOOL_CALL:
        return _extract_from_tool_call(response)

    if family == ActionFamily.TEXT:
        return _extract_from_text(response)

    raise FormatError(
        reason="unknown_family",
        feedback="Internal error: unknown action family.",
    )


def _extract_from_tool_call(response: ModelResponse) -> tuple[str, str | None]:
    """Extract command from tool calls. Expects exactly one bash call."""
    tool_calls = _collect_tool_calls(response)

    if len(tool_calls) == 0:
        raise FormatError(
            reason="zero_actions",
            feedback="No tool calls found. Use exactly one bash tool call.",
        )

    if len(tool_calls) > 1:
        names = [tc["function_name"] for tc in tool_calls]
        raise FormatError(
            reason="multiple_actions",
            feedback=(
                f"Found {len(tool_calls)} tool calls ({', '.join(names)}). "
                "Provide exactly ONE bash tool call."
            ),
        )

    tc = tool_calls[0]
    if tc["function_name"] != "bash":
        raise FormatError(
            reason="wrong_tool",
            feedback=(
                f"Only the 'bash' tool is supported. "
                f"You called '{tc['function_name']}'. "
                "Use a single bash tool call with a 'command' argument."
            ),
        )

    command = tc["arguments"].get("command", "")
    if not command:
        raise FormatError(
            reason="missing_command",
            feedback="Your bash tool call has no 'command' argument. Provide exactly one shell command.",
        )

    return command, tc["id"]


def _extract_from_text(response: ModelResponse) -> tuple[str, str | None]:
    """Extract command from text markers. Expects exactly one block."""
    text = _concat_text_content(response)
    count = _count_text_actions(text)

    if count == 0:
        raise FormatError(
            reason="zero_actions",
            feedback=(
                "No command block found in your response. "
                "Wrap your shell command in ```mswea_bash_command``` or "
                "<mswea_bash_command> tags."
            ),
        )

    if count > 1:
        raise FormatError(
            reason="multiple_actions",
            feedback=(
                f"Found {count} command blocks. Provide exactly ONE. "
                "Wrap a single shell command in ```mswea_bash_command``` "
                "or <mswea_bash_command>."
            ),
        )

    # Try fenced first, then XML
    match = _TEXT_FENCED_RE.search(text)
    if match:
        return match.group(1).strip(), None

    match = _TEXT_XML_RE.search(text)
    if match:
        return match.group(1).strip(), None

    raise FormatError(
        reason="extraction_failure",
        feedback="Internal parser error: unable to extract command from text block.",
    )
