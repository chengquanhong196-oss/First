"""Acceptance test: Parser rules (Section L).

0 actions, 2 actions, tool+text conflict -> FormatError, no execution.
"""

import pytest

from mini_swe_agent.models.messages import Message, ModelResponse, ToolCall, Usage
from mini_swe_agent.parser.detector import detect_action_family
from mini_swe_agent.parser.errors import FormatError
from mini_swe_agent.parser.extractor import extract_command
from mini_swe_agent.types import ActionFamily


def make_response_with_tool_calls(tool_calls: list[ToolCall]) -> ModelResponse:
    """Helper: create a ModelResponse with tool calls."""
    msg = Message(
        role="assistant",
        content="I'll run a command.",
        tool_calls=tool_calls,
    )
    return ModelResponse(messages=[msg], usage=Usage())


def make_response_with_text(content: str) -> ModelResponse:
    """Helper: create a ModelResponse with text content."""
    msg = Message(role="assistant", content=content)
    return ModelResponse(messages=[msg], usage=Usage())


class TestZeroActions:
    """Zero actions must raise FormatError."""

    def test_no_tool_calls_and_no_markers(self):
        """Response with just text and no command markers -> FormatError."""
        resp = make_response_with_text("I think I should run ls, but I'm not sure.")
        with pytest.raises(FormatError) as exc_info:
            detect_action_family(resp)
        assert exc_info.value.reason == "zero_actions"

    def test_empty_response(self):
        """Empty assistant message -> FormatError."""
        resp = make_response_with_text("")
        with pytest.raises(FormatError):
            detect_action_family(resp)


class TestTwoActions:
    """Two actions must raise FormatError (no silent selection)."""

    def test_two_tool_calls(self):
        """Two bash tool calls -> FormatError."""
        resp = make_response_with_tool_calls([
            ToolCall(id="tc_1", function_name="bash", arguments={"command": "ls"}),
            ToolCall(id="tc_2", function_name="bash", arguments={"command": "pwd"}),
        ])
        family = detect_action_family(resp)
        with pytest.raises(FormatError) as exc_info:
            extract_command(resp, family)
        assert exc_info.value.reason == "multiple_actions"

    def test_two_fenced_blocks(self):
        """Two fenced mswea_bash_command blocks -> FormatError."""
        resp = make_response_with_text(
            "Here are two commands:\n\n"
            "```mswea_bash_command\nls\n```\n\n"
            "```mswea_bash_command\npwd\n```"
        )
        family = detect_action_family(resp)
        with pytest.raises(FormatError) as exc_info:
            extract_command(resp, family)
        assert exc_info.value.reason == "multiple_actions"

    def test_two_xml_blocks(self):
        """Two XML blocks -> FormatError."""
        resp = make_response_with_text(
            "<mswea_bash_command>ls</mswea_bash_command>\n"
            "<mswea_bash_command>pwd</mswea_bash_command>"
        )
        family = detect_action_family(resp)
        with pytest.raises(FormatError) as exc_info:
            extract_command(resp, family)
        assert exc_info.value.reason == "multiple_actions"

    def test_one_fenced_one_xml(self):
        """One fenced + one XML = 2 actions -> FormatError."""
        resp = make_response_with_text(
            "```mswea_bash_command\nls\n```\n"
            "<mswea_bash_command>pwd</mswea_bash_command>"
        )
        family = detect_action_family(resp)
        with pytest.raises(FormatError) as exc_info:
            extract_command(resp, family)
        assert exc_info.value.reason == "multiple_actions"


class TestToolTextConflict:
    """Tool-call and text markers simultaneously must raise FormatError."""

    def test_tool_call_plus_fenced(self):
        """One tool call AND one fenced block -> FormatError at detection."""
        msg = Message(
            role="assistant",
            content="```mswea_bash_command\nls\n```",
            tool_calls=[
                ToolCall(id="tc_1", function_name="bash", arguments={"command": "ls"})
            ],
        )
        resp = ModelResponse(messages=[msg], usage=Usage())
        with pytest.raises(FormatError) as exc_info:
            detect_action_family(resp)
        assert exc_info.value.reason == "both_families"

    def test_tool_call_plus_xml(self):
        """One tool call AND one XML block -> FormatError at detection."""
        msg = Message(
            role="assistant",
            content="<mswea_bash_command>ls</mswea_bash_command>",
            tool_calls=[
                ToolCall(id="tc_1", function_name="bash", arguments={"command": "ls"})
            ],
        )
        resp = ModelResponse(messages=[msg], usage=Usage())
        with pytest.raises(FormatError) as exc_info:
            detect_action_family(resp)
        assert exc_info.value.reason == "both_families"


class TestExtractionSuccess:
    """Valid single actions must extract correctly."""

    def test_single_tool_call(self):
        """One bash tool call -> extract command."""
        resp = make_response_with_tool_calls([
            ToolCall(id="tc_1", function_name="bash", arguments={"command": "ls -la"})
        ])
        family = detect_action_family(resp)
        cmd, tc_id = extract_command(resp, family)
        assert cmd == "ls -la"
        assert tc_id == "tc_1"

    def test_single_fenced_block(self):
        """One fenced block -> extract command."""
        resp = make_response_with_text("```mswea_bash_command\nls -la\n```")
        family = detect_action_family(resp)
        cmd, tc_id = extract_command(resp, family)
        assert cmd == "ls -la"
        assert tc_id is None

    def test_single_xml_block(self):
        """One XML block -> extract command."""
        resp = make_response_with_text(
            "<mswea_bash_command>ls -la</mswea_bash_command>"
        )
        family = detect_action_family(resp)
        cmd, tc_id = extract_command(resp, family)
        assert cmd == "ls -la"
        assert tc_id is None
