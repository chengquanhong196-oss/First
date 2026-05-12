"""OpenAI API adapter with native tool-call support."""

import json
import logging
import time
from typing import Any, Optional

from mini_swe_agent.config.schema import ModelConfig
from mini_swe_agent.models.base import ModelAdapter
from mini_swe_agent.models.cost import CostTracker
from mini_swe_agent.models.messages import Message, ModelResponse, ToolCall, Usage

logger = logging.getLogger(__name__)

# Pricing per million tokens (defaults for GPT-4o)
DEFAULT_PRICING = {
    "input": 2.50,
    "output": 10.00,
    "cache_read": 1.25,
}


BASH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a shell command and return the output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI's API (GPT models)."""

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
            from openai import AsyncOpenAI

            api_key = self._config.api_key or None
            base_url = self._config.api_base or None
            self._client = AsyncOpenAI(
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
        """Send messages to OpenAI and return the response."""
        from mini_swe_agent.utils.retry import make_retry_decorator

        @make_retry_decorator()
        async def _call():
            return await self._send_impl(messages)

        return await _call()

    async def _send_impl(self, messages: list[Message]) -> ModelResponse:
        """Internal implementation of send (without retry wrapper)."""
        client = self._get_client()

        api_messages = []
        for msg in messages:
            api_messages.append(self._message_to_api(msg))

        kwargs: dict[str, Any] = {
            "model": self._config.name,
            "messages": api_messages,
            "tools": [BASH_TOOL_DEFINITION],
        }
        if self._config.temperature > 0:
            kwargs["temperature"] = self._config.temperature
        if self._config.max_tokens:
            kwargs["max_tokens"] = self._config.max_tokens

        start = time.monotonic()
        response = await client.chat.completions.create(**kwargs)
        latency = time.monotonic() - start

        response_messages = self._parse_response(response)
        usage = self._extract_usage(response)
        cost = self._calculate_cost(usage)

        self._cost_tracker.record(cost)

        return ModelResponse(
            messages=response_messages,
            usage=usage,
            cost=cost,
            latency=latency,
            stop_reason=getattr(response.choices[0], "finish_reason", "") if response.choices else "",
        )

    def _message_to_api(self, msg: Message) -> dict[str, Any]:
        """Convert internal Message to OpenAI API format."""
        api_msg: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}

        if msg.role == "tool":
            api_msg["tool_call_id"] = msg.tool_call_id or ""

        if msg.tool_calls:
            api_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function_name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]

        return api_msg

    def _parse_response(self, response: Any) -> list[Message]:
        """Parse OpenAI response into internal Message objects."""
        messages = []
        choice = response.choices[0] if response.choices else None
        if choice is None:
            return [Message(role="assistant", content="")]

        msg = choice.message
        tool_calls = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    function_name=tc.function.name,
                    arguments=arguments,
                ))

        content = msg.content or ""
        messages.append(Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        ))

        return messages

    def _extract_usage(self, response: Any) -> Usage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            cache_read_tokens=getattr(
                getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0
            ),
        )

    def _calculate_cost(self, usage: Usage) -> float:
        cost = 0.0
        cost += (usage.input_tokens / 1_000_000) * self._pricing["input"]
        cost += (usage.output_tokens / 1_000_000) * self._pricing["output"]
        cost += (usage.cache_read_tokens / 1_000_000) * self._pricing.get("cache_read", 0)
        return cost
