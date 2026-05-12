"""Load trajectory data from JSON files."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mini_swe_agent.trajectory.schema import Trajectory

logger = logging.getLogger(__name__)


def load_trajectory(path: str | Path) -> Trajectory:
    """Read a Trajectory from a JSON file.

    Supports both bare messages arrays and full trajectory files.
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If the file is just a messages array, wrap it
    if isinstance(data, list):
        return Trajectory(
            task="",
            model_name="",
            terminal_state="unknown",
            messages=data,
        )

    return Trajectory(**data)


def load_trajectory_raw(path: str | Path) -> dict[str, Any]:
    """Load trajectory file as a raw dict without validation."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
