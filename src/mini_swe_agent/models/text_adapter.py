"""Text-only model adapter (no native tool calls)."""

import logging
import time
from typing import Any, Optional

from mini_swe_agent.config.schema import ModelConfig
from mini_swe_agent.models.base import ModelAdapter
from mini_swe_agent.models.cost import CostTracker
from mini_swe_agent.models.messages import Message, ModelResponse, Usage

logger = logging.getLogger(__name__)

DEFAULT_PRICING = {
    "input": 0.0,
    "output": 0.0,
}


class TextAdapter(ModelAdapter):
    """Adapter for models without native tool-call support.

    Wraps any OpenAI-compatible chat completion API but strips tool
    definitions from the request. All responses are treated as text;
    the parser uses the TEXT path.
    """

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

            api_key = self._config.api_key or "not-needed"
            base_url = self._config.api_base or None
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        return self._client

    def supports_tool_calls(self) -> bool:
        return False

    def get_total_cost(self) -> float:
        return self._cost_tracker.total

    def get_last_cost(self) -> float:
        return self._cost_tracker.last

    def reset_cost(self) -> None:
        self._cost_tracker.reset()

    async def send(self, messages: list[Message]) -> ModelResponse:
        """Send messages and return a text-only response."""
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
            role = msg.role
            if role == "tool":
                role = "user"
            content = msg.content
            if content is None:
                content = ""
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                content = "\n".join(parts)
            api_messages.append({"role": role, "content": str(content)})

        kwargs: dict[str, Any] = {
            "model": self._config.name,
            "messages": api_messages,
        }
        if self._config.temperature > 0:
            kwargs["temperature"] = self._config.temperature
        if self._config.max_tokens:
            kwargs["max_tokens"] = self._config.max_tokens

        start = time.monotonic()
        response = await client.chat.completions.create(**kwargs)
        latency = time.monotonic() - start

        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""

        response_msg = Message(role="assistant", content=content)
        usage = self._extract_usage(response)
        cost = self._calculate_cost(usage)

        self._cost_tracker.record(cost)

        return ModelResponse(
            messages=[response_msg],
            usage=usage,
            cost=cost,
            latency=latency,
            stop_reason=getattr(response.choices[0], "finish_reason", "") if response.choices else "",
        )

    def _extract_usage(self, response: Any) -> Usage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
        )

    def _calculate_cost(self, usage: Usage) -> float:
        cost = 0.0
        cost += (usage.input_tokens / 1_000_000) * self._pricing["input"]
        cost += (usage.output_tokens / 1_000_000) * self._pricing["output"]
        return cost
