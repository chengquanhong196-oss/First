"""Cost tracking for model API calls."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CostTracker:
    """Accumulates per-call cost and enforces optional spending limits."""

    def __init__(self, limit: float | None = None) -> None:
        self._total: float = 0.0
        self._last: float = 0.0
        self._limit = limit

    def record(self, cost: float) -> bool:
        """Record cost from the latest call.

        Returns True if the cumulative total exceeds the configured limit.
        """
        self._last = cost
        self._total += cost
        if self._limit is not None and self._total > self._limit:
            logger.warning(
                "Cost limit exceeded: $%.6f > $%.2f", self._total, self._limit
            )
            return True
        return False

    @property
    def total(self) -> float:
        """Cumulative cost across all recorded calls."""
        return self._total

    @property
    def last(self) -> float:
        """Cost of the most recently recorded call."""
        return self._last

    @property
    def is_exhausted(self) -> bool:
        """True if the cost limit is set and has been exceeded."""
        if self._limit is None:
            return False
        return self._total > self._limit

    @property
    def limit(self) -> float | None:
        """The configured cost limit, or None if unlimited."""
        return self._limit

    def reset(self) -> None:
        """Reset accumulated costs to zero."""
        self._total = 0.0
        self._last = 0.0
