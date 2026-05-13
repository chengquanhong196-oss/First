"""YAML configuration loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigLoadError(Exception):
    """Raised when a config file cannot be loaded."""


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
