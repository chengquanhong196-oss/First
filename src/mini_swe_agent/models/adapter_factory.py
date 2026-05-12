"""Factory for creating model adapters from configuration."""

from mini_swe_agent.config.schema import Config, ModelConfig
from mini_swe_agent.models.base import ModelAdapter


def create_adapter(config: Config) -> ModelAdapter:
    """Return the appropriate adapter based on config.model.provider.

    Supported providers:
    - anthropic: AnthropicAdapter (native tool calls)
    - openai: OpenAIAdapter (native tool calls)
    - text: TextAdapter (text-only, no tools)
    """
    model_config = config.model
    provider = model_config.provider.lower()

    if provider == "anthropic":
        from mini_swe_agent.models.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(model_config)

    if provider == "openai":
        from mini_swe_agent.models.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(model_config)

    if provider == "text":
        from mini_swe_agent.models.text_adapter import TextAdapter
        return TextAdapter(model_config)

    raise ValueError(
        f"Unknown model provider: '{provider}'. "
        "Supported providers: anthropic, openai, text."
    )
