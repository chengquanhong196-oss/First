"""Save trajectory data to JSON files."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from mini_swe_agent.trajectory.schema import Trajectory

logger = logging.getLogger(__name__)


def save_trajectory(trajectory: Trajectory, path: str | Path) -> None:
    """Write a Trajectory to a JSON file.

    Creates parent directories if needed. Drops null/None fields
    from the JSON output for readability.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = trajectory.model_dump(mode="json", exclude_none=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Trajectory saved to %s (%d steps)", path, trajectory.total_steps)
