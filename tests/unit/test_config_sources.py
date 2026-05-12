"""Unit tests for configuration resolution from multiple sources."""
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from mini_swe_agent.config.sources import (
    _coerce_value,
    _parse_dotted_key,
    _load_defaults,
    resolve_config,
)


class TestCoerceValue:
    def test_true(self):
        assert _coerce_value("true") is True

    def test_false(self):
        assert _coerce_value("false") is False

    def test_null(self):
        assert _coerce_value("null") is None
        assert _coerce_value("none") is None

    def test_int(self):
        assert _coerce_value("42") == 42
        assert _coerce_value("-10") == -10

    def test_float(self):
        assert _coerce_value("3.14") == 3.14

    def test_string_fallback(self):
        assert _coerce_value("hello") == "hello"
        assert _coerce_value("model-name") == "model-name"


class TestParseDottedKey:
    def test_simple_key(self):
        result = _parse_dotted_key("name", "claude")
        assert result == {"name": "claude"}

    def test_nested_key(self):
        result = _parse_dotted_key("model.name", "gpt-4o")
        assert result == {"model": {"name": "gpt-4o"}}

    def test_deeply_nested(self):
        result = _parse_dotted_key("a.b.c", "42")
        assert result == {"a": {"b": {"c": 42}}}

    def test_boolean_value(self):
        result = _parse_dotted_key("batch.redo_existing", "true")
        assert result == {"batch": {"redo_existing": True}}

    def test_float_value(self):
        result = _parse_dotted_key("limits.max_cost", "5.5")
        assert result == {"limits": {"max_cost": 5.5}}


class TestResolveConfig:
    def test_defaults_loaded(self):
        """resolve_config with no sources returns defaults."""
        merged = resolve_config()
        assert isinstance(merged, dict)
        assert "model" in merged
        assert "executor" in merged
        assert "limits" in merged

    def test_cli_overrides_highest_priority(self):
        """CLI key=value overrides should win over defaults."""
        merged = resolve_config(cli_overrides=["model.name=custom-model"])
        assert merged["model"]["name"] == "custom-model"

    def test_cli_flags_top_priority(self, tmp_path):
        """CLI flags are the highest priority."""
        # Use a temp dir to avoid picking up cwd mswea.yaml
        merged = resolve_config(
            cli_overrides=["model.name=from-override"],
            cli_flags={"model": {"name": "from-flag"}},
            cwd=tmp_path,
        )
        assert merged["model"]["name"] == "from-flag"

    def test_cli_files_merged(self, tmp_path):
        """CLI files are merged in order (later wins)."""
        file1 = tmp_path / "config1.yaml"
        file1.write_text(yaml.dump({"model": {"name": "model-from-file1"}}))
        file2 = tmp_path / "config2.yaml"
        file2.write_text(yaml.dump({"model": {"name": "model-from-file2"}}))

        merged = resolve_config(
            cli_files=[str(file1), str(file2)],
            cli_overrides=[],
            cwd=tmp_path,
        )
        assert merged["model"]["name"] == "model-from-file2"

    def test_cwd_config_loaded(self, tmp_path):
        """mswea.yaml in cwd is loaded automatically."""
        cwd_config = tmp_path / "mswea.yaml"
        cwd_config.write_text(yaml.dump({"limits": {"max_steps": 99}}))

        merged = resolve_config(cwd=tmp_path)
        assert merged["limits"]["max_steps"] == 99

    def test_missing_files_ignored(self, tmp_path):
        """Missing config files are silently ignored."""
        merged = resolve_config(
            cli_files=[str(tmp_path / "nonexistent.yaml")],
            cwd=tmp_path,
        )
        assert isinstance(merged, dict)

    def test_kv_without_equals_skipped(self):
        """KV string without '=' is skipped."""
        merged = resolve_config(cli_overrides=["invalid_kv"])
        assert isinstance(merged, dict)
