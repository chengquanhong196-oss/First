"""YAML + Jinja2 configuration loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from jinja2 import StrictUndefined

from mini_swe_agent.utils.templates import create_jinja_env


class ConfigLoadError(Exception):
    """Raised when a config file cannot be loaded or rendered."""


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML file from disk. Returns empty dict if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"Failed to parse YAML file {path}: {e}") from e


def render_config_template(template_str: str, variables: dict[str, Any]) -> str:
    """Render a Jinja2 template string with StrictUndefined."""
    env = create_jinja_env()
    try:
        tmpl = env.from_string(template_str)
        return tmpl.render(**variables)
    except StrictUndefined as e:
        raise ConfigLoadError(f"Undefined template variable: {e}") from e
    except Exception as e:
        raise ConfigLoadError(f"Template rendering failed: {e}") from e
