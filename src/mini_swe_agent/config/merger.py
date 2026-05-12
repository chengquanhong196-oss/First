"""Deep merge utility for configuration dictionaries."""

from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base.

    - Scalars and lists: override replaces base.
    - Dictionaries: merged recursively.
    - None values in override: delete key from result.
    """
    result = {**base}

    for key, value in override.items():
        if value is None:
            result.pop(key, None)
        elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result
