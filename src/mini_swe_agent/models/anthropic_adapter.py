"""Anthropic API adapter with native tool-call support."""

import json
import logging
import time
from typing import Any, Optional

from mini_swe_agent.config.schema import ModelConfig
from mini_swe_agent.models.base import ModelAdapter
from mini_swe_agent.models.cost import CostTracker
from mini_swe_agent.models.messages import Message, ModelResponse, ToolCall, Usage

logger = logging.getLogger(__name__)

# Pricing per million tokens (configurable, defaults for Claude Sonnet 4.6)
DEFAULT_PRICING = {
    "input": 3.00,          # $3/MTok
    "output": 15.00,        # $15/MTok
    "cache_read": 0.30,     # $0.30/MTok
    "cache_write": 3.75,    # $3.75/MTok
}


BASH_TOOL_DEFINITION = {
    "name": "bash",
    "description": "Execute a shell command and return the output.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            }
        },
        "required": ["command"],
    },
}


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic's API (Claude models)."""

    def __init__(self, config: ModelConfig, pricing: Optional[dict[str, float]] = None) -> None:
        self._config = config
        self._cost_tracker = CostTracker()
        self._pricing = pricing or DEFAULT_PRICING
        self._client = None

    @property
    def model_name(self) -> str:
        return self._config.name

    def _get_client(self):
        if self._client is None:
            import anthropic

            api_key = self._config.api_key or None
            base_url = self._config.api_base or None
            self._client = anthropic.AsyncAnthropic(
                api_key=api_key,
                base_url=base_url,
            )
        return self._client

    def supports_tool_calls(self) -> bool:
        return True

    def get_total_cost(self) -> float:
        return self._cost_tracker.total

    def get_last_cost(self) -> float:
        return self._cost_tracker.last

    def reset_cost(self) -> None:
        self._cost_tracker.reset()

    async def send(self, messages: list[Message]) -> ModelResponse:
        """Send messages to Anthropic and return the response."""
        from mini_swe_agent.utils.retry import make_retry_decorator

        @make_retry_decorator()
        async def _call():
            return await self._send_impl(messages)

        return await _call()

    async def _send_impl(self, messages: list[Message]) -> ModelResponse:
        """Internal implementation of send (without retry wrapper)."""
        client = self._get_client()

        system_prompt = ""
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_prompt += (msg.content or "")
            elif msg.role == "tool":
                api_messages.append(self._tool_result_to_api(msg))
            elif msg.role == "assistant":
                api_messages.append(self._assistant_to_api(msg))
            else:
                api_messages.append(self._message_to_api(msg))

        kwargs: dict[str, Any] = {
            "model": self._config.name,
            "max_tokens": self._config.max_tokens,
            "messages": api_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if self._config.temperature > 0:
            kwargs["temperature"] = self._config.temperature
        if self._config.thinking_budget > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._config.thinking_budget,
            }

        kwargs["tools"] = [BASH_TOOL_DEFINITION]

        start = time.monotonic()
        response = await client.messages.create(**kwargs)
        latency = time.monotonic() - start

        # Convert response to internal messages
        response_messages = self._parse_response(response)
        usage = self._extract_usage(response)
        cost = self._calculate_cost(usage)

        self._cost_tracker.record(cost)

        return ModelResponse(
            messages=response_messages,
            usage=usage,
            cost=cost,
            latency=latency,
            stop_reason=getattr(response, "stop_reason", ""),
        )

    def _message_to_api(self, msg: Message) -> dict[str, Any]:
        content = msg.content or ""
        if isinstance(content, list):
            return {"role": "user", "content": content}
        return {"role": "user", "content": str(content)}

    def _assistant_to_api(self, msg: Message) -> dict[str, Any]:
        api_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            api_msg["content"] = [
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function_name,
                    "input": tc.arguments,
                }
                for tc in msg.tool_calls
            ]
        return api_msg

    def _tool_result_to_api(self, msg: Message) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.content or "",
                }
            ],
        }

    def _parse_response(self, response: Any) -> list[Message]:
        """Parse Anthropic response into internal Message objects."""
        messages = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                messages.append(Message(
                    role="assistant",
                    content=block.text,
                ))
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    function_name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        if tool_calls:
            if messages:
                messages[0].tool_calls = tool_calls
            else:
                messages.append(Message(
                    role="assistant",
                    content="",
                    tool_calls=tool_calls,
                ))

        if not messages:
            messages.append(Message(
                role="assistant",
                content=response.content[0].text if response.content else "",
            ))

        return messages

    def _extract_usage(self, response: Any) -> Usage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0),
        )

    def _calculate_cost(self, usage: Usage) -> float:
        """Calculate cost from token usage using configured pricing."""
        cost = 0.0
        cost += (usage.input_tokens / 1_000_000) * self._pricing["input"]
        cost += (usage.output_tokens / 1_000_000) * self._pricing["output"]
        cost += (usage.cache_read_tokens / 1_000_000) * self._pricing["cache_read"]
        cost += (usage.cache_write_tokens / 1_000_000) * self._pricing["cache_write"]
        return cost
