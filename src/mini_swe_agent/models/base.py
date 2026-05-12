"""Abstract base class for model adapters."""

from abc import ABC, abstractmethod

from mini_swe_agent.models.messages import Message, ModelResponse


class ModelAdapter(ABC):
    """Protocol for interacting with a language model.

    Each concrete adapter handles provider-specific API formats,
    tool calling, cost tracking, and error retry logic.
    """

    @abstractmethod
    async def send(self, messages: list[Message]) -> ModelResponse:
        """Send conversation to the model and return the response.

        Called once per agent step. The adapter must flatten any stateless
        API response (e.g., output_items) into conversation messages before
        the next call. May raise transient errors (HTTP 5xx, rate limits)
        which the caller should retry.
        """
        ...

    @abstractmethod
    def supports_tool_calls(self) -> bool:
        """Return True if this adapter can produce native tool calls."""
        ...

    @abstractmethod
    def get_total_cost(self) -> float:
        """Cumulative USD cost of all calls on this adapter instance."""
        ...

    @abstractmethod
    def get_last_cost(self) -> float:
        """Cost of the most recent call in USD."""
        ...

    @abstractmethod
    def reset_cost(self) -> None:
        """Reset the cost accumulator to zero."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...
