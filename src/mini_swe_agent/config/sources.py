"""Resolve configuration from all sources in priority order."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mini_swe_agent.config.loader import load_yaml_file
from mini_swe_agent.config.merger import deep_merge


def _parse_dotted_key(key: str, value: str) -> dict[str, Any]:
    """Parse 'a.b.c=val' into {'a': {'b': {'c': val}}}.

    Attempts to coerce value to int/float/bool where possible.
    """
    parts = key.split(".")
    coerced = _coerce_value(value)
    result: dict[str, Any] = {}
    current = result
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            current[part] = coerced
        else:
            current[part] = {}
            current = current[part]
    return result


def _coerce_value(raw: str) -> Any:
    """Coerce a string value to int, float, bool, or keep as str."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() == "null" or raw.lower() == "none":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _load_defaults() -> dict[str, Any]:
    """Load built-in defaults.yaml from the package."""
    import pkgutil

    try:
        import yaml

        data = pkgutil.get_data("mini_swe_agent", "config/defaults.yaml")
        if data:
            return yaml.safe_load(data) or {}
    except Exception:
        pass
    return {}


def resolve_config(
    cli_files: list[str] | None = None,
    cli_overrides: list[str] | None = None,
    cli_flags: dict[str, Any] | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Collect and merge configuration from all sources.

    Priority (highest to lowest):
    1. CLI key=value overrides
    2. CLI file paths (last specified overrides earlier)
    3. ./mswea.yaml (working directory)
    4. ~/.config/mswea/config.yaml
    5. Built-in defaults.yaml

    Returns the fully merged configuration dict.
    """
    cwd = Path(cwd) if cwd else Path.cwd()

    # Start with built-in defaults
    merged = _load_defaults()

    # Layer 4: user config
    user_config_path = Path.home() / ".config" / "mswea" / "config.yaml"
    user_config = load_yaml_file(user_config_path)
    if user_config:
        merged = deep_merge(merged, user_config)

    # Layer 3: working directory config
    cwd_config = load_yaml_file(cwd / "mswea.yaml")
    if cwd_config:
        merged = deep_merge(merged, cwd_config)

    # Layer 2: CLI file paths (in order)
    for fpath in (cli_files or []):
        file_config = load_yaml_file(fpath)
        if file_config:
            merged = deep_merge(merged, file_config)

    # Layer 1: CLI key=value overrides
    for kv in (cli_overrides or []):
        if "=" not in kv:
            continue
        key, _, value = kv.partition("=")
        override_dict = _parse_dotted_key(key, value)
        merged = deep_merge(merged, override_dict)

    # Top-level CLI flags
    if cli_flags:
        merged = deep_merge(merged, cli_flags)

    return merged
