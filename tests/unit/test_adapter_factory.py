"""Unit tests for model adapter factory."""
import pytest

from mini_swe_agent.config.schema import Config
from mini_swe_agent.models.anthropic_adapter import AnthropicAdapter
from mini_swe_agent.models.openai_adapter import OpenAIAdapter
from mini_swe_agent.models.text_adapter import TextAdapter
from mini_swe_agent.models.adapter_factory import create_adapter


def make_config(**overrides):
    data = {
        "model": {"provider": "anthropic", "name": "test-model", "max_tokens": 4096},
        "executor": {"backend": "local", "timeout": 30.0},
        "limits": {"max_steps": 10, "max_cost": 1.0, "max_consecutive_format_errors": 3},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and key in data:
            data[key].update(value)
        else:
            data[key] = value
    return Config(**data)


class TestCreateAdapter:
    def test_anthropic_provider(self):
        config = make_config(**{"model": {"provider": "anthropic"}})
        adapter = create_adapter(config)
        assert isinstance(adapter, AnthropicAdapter)
        assert adapter.model_name == "test-model"

    def test_openai_provider(self):
        config = make_config(**{"model": {"provider": "openai"}})
        adapter = create_adapter(config)
        assert isinstance(adapter, OpenAIAdapter)

    def test_text_provider(self):
        config = make_config(**{"model": {"provider": "text"}})
        adapter = create_adapter(config)
        assert isinstance(adapter, TextAdapter)

    def test_case_insensitive_provider(self):
        config = make_config(**{"model": {"provider": "AnThRoPiC"}})
        adapter = create_adapter(config)
        assert isinstance(adapter, AnthropicAdapter)

    def test_unknown_provider_raises(self):
        config = make_config(**{"model": {"provider": "unknown-llm"}})
        with pytest.raises(ValueError, match="Unknown model provider"):
            create_adapter(config)
