"""Step and cost limit trackers for the agent loop."""

import logging

logger = logging.getLogger(__name__)


class StepLimiter:
    """Tracks step count and enforces a maximum."""

    def __init__(self, max_steps: int) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        self._max_steps = max_steps
        self._current = 0

    @property
    def exhausted(self) -> bool:
        """True when the step count has reached or exceeded the limit."""
        return self._current >= self._max_steps

    @property
    def current(self) -> int:
        return self._current

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def increment(self) -> bool:
        """Increment step counter. Returns True if limit is now exhausted."""
        self._current += 1
        if self.exhausted:
            logger.warning(
                "Step limit reached: %d/%d", self._current, self._max_steps
            )
        return self.exhausted


class CostLimiter:
    """Tracks cumulative cost and enforces a maximum."""

    def __init__(self, max_cost: float) -> None:
        if max_cost < 0:
            raise ValueError(f"max_cost must be >= 0, got {max_cost}")
        self._max_cost = max_cost
        self._total = 0.0

    @property
    def exhausted(self) -> bool:
        """True when cumulative cost exceeds the limit."""
        if self._max_cost == 0:
            return False
        return self._total > self._max_cost

    @property
    def total(self) -> float:
        return self._total

    @property
    def max_cost(self) -> float:
        return self._max_cost

    def add(self, cost: float) -> bool:
        """Add cost to the tracker. Returns True if limit is now exceeded."""
        self._total += cost
        if self.exhausted:
            logger.warning(
                "Cost limit exceeded: $%.6f > $%.2f", self._total, self._max_cost
            )
        return self.exhausted
