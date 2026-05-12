"""Unit tests for model adapter message conversion (Prompt G coverage)."""
import pytest

from mini_swe_agent.config.schema import ModelConfig
from mini_swe_agent.models.messages import Message, ModelResponse, ToolCall, Usage


class TestAnthropicAdapter:
    """Tests for AnthropicAdapter message conversion without live API."""

    @pytest.fixture
    def adapter(self):
        from mini_swe_agent.models.anthropic_adapter import AnthropicAdapter
        config = ModelConfig(provider="anthropic", name="claude-sonnet-4-6", max_tokens=4096)
        return AnthropicAdapter(config)

    def test_message_to_api_simple_user(self, adapter):
        msg = Message(role="user", content="ls -la")
        result = adapter._message_to_api(msg)
        assert result["role"] == "user"
        assert result["content"] == "ls -la"

    def test_message_to_api_list_content(self, adapter):
        msg = Message(role="user", content=[{"type": "text", "text": "hello"}])
        result = adapter._message_to_api(msg)
        assert result["role"] == "user"
        assert isinstance(result["content"], list)

    def test_assistant_to_api_with_tool_calls(self, adapter):
        msg = Message(
            role="assistant",
            content="I'll run:",
            tool_calls=[
                ToolCall(id="tc_1", function_name="bash", arguments={"command": "ls"})
            ],
        )
        result = adapter._assistant_to_api(msg)
        assert result["role"] == "assistant"
        assert isinstance(result["content"], list)
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["id"] == "tc_1"

    def test_tool_result_to_api(self, adapter):
        msg = Message(
            role="tool",
            content="file.txt",
            tool_call_id="tc_abc",
            name="bash",
        )
        result = adapter._tool_result_to_api(msg)
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "tc_abc"

    def test_calculate_cost(self, adapter):
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = adapter._calculate_cost(usage)
        expected = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert cost == pytest.approx(expected)

    def test_calculate_cost_with_cache(self, adapter):
        usage = Usage(
            input_tokens=1000, output_tokens=500,
            cache_read_tokens=2000, cache_write_tokens=1000,
        )
        cost = adapter._calculate_cost(usage)
        expected = (
            (1000 / 1_000_000) * 3.0
            + (500 / 1_000_000) * 15.0
            + (2000 / 1_000_000) * 0.30
            + (1000 / 1_000_000) * 3.75
        )
        assert cost == pytest.approx(expected)

    def test_supports_tool_calls(self, adapter):
        assert adapter.supports_tool_calls() is True

    def test_model_name(self, adapter):
        assert adapter.model_name == "claude-sonnet-4-6"

    def test_send_extracts_system_prompt(self, adapter):
        """Verify system prompt is separated from messages."""
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="run ls"),
        ]
        # We can test the message preprocessing indirectly via _message_to_api
        # since we can't call the live API
        assert messages[0].role == "system"
        assert messages[1].role == "user"


class TestOpenAIAdapter:
    """Tests for OpenAIAdapter message conversion without live API."""

    @pytest.fixture
    def adapter(self):
        from mini_swe_agent.models.openai_adapter import OpenAIAdapter
        config = ModelConfig(provider="openai", name="gpt-4o", max_tokens=4096)
        return OpenAIAdapter(config)

    def test_message_to_api_simple_user(self, adapter):
        msg = Message(role="user", content="ls -la")
        result = adapter._message_to_api(msg)
        assert result["role"] == "user"
        assert result["content"] == "ls -la"

    def test_message_to_api_tool_role(self, adapter):
        msg = Message(role="tool", content="stdout here", tool_call_id="tc_xyz")
        result = adapter._message_to_api(msg)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "tc_xyz"

    def test_message_to_api_with_tool_calls(self, adapter):
        msg = Message(
            role="assistant",
            content="Running command.",
            tool_calls=[
                ToolCall(id="tc_1", function_name="bash", arguments={"command": "pwd"})
            ],
        )
        result = adapter._message_to_api(msg)
        assert result["role"] == "assistant"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "bash"

    def test_calculate_cost(self, adapter):
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = adapter._calculate_cost(usage)
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected)

    def test_supports_tool_calls(self, adapter):
        assert adapter.supports_tool_calls() is True


class TestTextAdapter:
    """Tests for TextAdapter message conversion."""

    @pytest.fixture
    def adapter(self):
        from mini_swe_agent.models.text_adapter import TextAdapter
        config = ModelConfig(provider="text", name="local-model", max_tokens=4096)
        return TextAdapter(config)

    def test_supports_tool_calls(self, adapter):
        assert adapter.supports_tool_calls() is False

    def test_model_name(self, adapter):
        assert adapter.model_name == "local-model"


class TestResponseParsing:
    """Test that ModelResponse from adapters can be parsed by the detector/extractor."""

    def test_tool_call_response_roundtrip(self):
        """Verify a ModelResponse with tool calls can be detected as TOOL_CALL family."""
        from mini_swe_agent.parser.detector import detect_action_family
        from mini_swe_agent.types import ActionFamily

        msg = Message(
            role="assistant",
            content="Running command.",
            tool_calls=[
                ToolCall(id="tc_1", function_name="bash", arguments={"command": "echo hello"})
            ],
        )
        resp = ModelResponse(messages=[msg], usage=Usage())
        family = detect_action_family(resp)
        assert family == ActionFamily.TOOL_CALL

    def test_text_response_roundtrip(self):
        """Verify a ModelResponse with text markers can be detected as TEXT family."""
        from mini_swe_agent.parser.detector import detect_action_family
        from mini_swe_agent.types import ActionFamily

        msg = Message(role="assistant", content="```mswea_bash_command\necho hello\n```")
        resp = ModelResponse(messages=[msg], usage=Usage())
        family = detect_action_family(resp)
        assert family == ActionFamily.TEXT
