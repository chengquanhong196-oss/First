"""Unit tests for cost tracker."""

from mini_swe_agent.models.cost import CostTracker


class TestCostTracker:
    def test_initial_state(self):
        tracker = CostTracker()
        assert tracker.total == 0.0
        assert tracker.last == 0.0
        assert tracker.is_exhausted is False

    def test_record_accumulates(self):
        tracker = CostTracker()
        tracker.record(0.005)
        tracker.record(0.003)
        assert tracker.total == 0.008
        assert tracker.last == 0.003

    def test_limit_exceeded(self):
        tracker = CostTracker(limit=0.01)
        assert tracker.record(0.005) is False
        assert tracker.record(0.006) is True
        assert tracker.is_exhausted is True

    def test_no_limit(self):
        tracker = CostTracker(limit=None)
        tracker.record(100.0)
        assert tracker.is_exhausted is False

    def test_reset(self):
        tracker = CostTracker(limit=10.0)
        tracker.record(5.0)
        tracker.reset()
        assert tracker.total == 0.0
        assert tracker.last == 0.0
