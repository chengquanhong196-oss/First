"""Unit tests for step and cost limiters."""

import pytest

from mini_swe_agent.core.limits import CostLimiter, StepLimiter


class TestStepLimiter:
    def test_initial_state(self):
        limiter = StepLimiter(5)
        assert limiter.current == 0
        assert limiter.exhausted is False

    def test_increment(self):
        limiter = StepLimiter(3)
        assert limiter.increment() is False  # step 1
        assert limiter.increment() is False  # step 2
        assert limiter.increment() is True   # step 3, limit reached
        assert limiter.exhausted is True

    def test_exhausted_stays_exhausted(self):
        limiter = StepLimiter(1)
        limiter.increment()
        assert limiter.exhausted is True
        limiter.increment()
        assert limiter.exhausted is True

    def test_invalid_max_steps(self):
        with pytest.raises(ValueError):
            StepLimiter(0)
        with pytest.raises(ValueError):
            StepLimiter(-1)


class TestCostLimiter:
    def test_initial_state(self):
        limiter = CostLimiter(10.0)
        assert limiter.total == 0.0
        assert limiter.exhausted is False

    def test_add_within_limit(self):
        limiter = CostLimiter(10.0)
        assert limiter.add(5.0) is False
        assert limiter.total == 5.0
        assert limiter.exhausted is False

    def test_add_exceeds_limit(self):
        limiter = CostLimiter(10.0)
        limiter.add(5.0)
        assert limiter.add(6.0) is True  # 11.0 > 10.0
        assert limiter.exhausted is True

    def test_zero_limit_never_exhausted(self):
        limiter = CostLimiter(0.0)
        limiter.add(1000.0)
        assert limiter.exhausted is False

    def test_invalid_max_cost(self):
        with pytest.raises(ValueError):
            CostLimiter(-1.0)
