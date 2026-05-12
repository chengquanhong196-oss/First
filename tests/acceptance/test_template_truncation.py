"""Acceptance test: Template truncation (Section H).

Observation output >= 10000 chars triggers truncation with warning,
keeping first 5000 + last 5000 + elided count.
"""

from mini_swe_agent.utils.templates import render_observation


TEMPLATE = "{{ stdout }}"


class TestTemplateTruncation:
    """Truncation rules: >= 10000 chars -> truncate."""

    def test_under_limit_no_truncation(self):
        """Under 10000 chars should NOT trigger truncation."""
        data = "a" * 9999
        result = render_observation(
            TEMPLATE, {"stdout": data},
            max_chars=10000, truncate_head=5000, truncate_tail=5000,
        )
        assert "elided" not in result.lower()
        assert len(result) == 9999

    def test_at_limit_triggers_truncation(self):
        """10000 chars should trigger truncation (>= 10000)."""
        data = "a" * 10000
        result = render_observation(
            TEMPLATE, {"stdout": data},
            max_chars=10000, truncate_head=5000, truncate_tail=5000,
        )
        assert "elided" in result.lower()
        # Total should be 5000 + marker text + 5000
        assert len(result) > 10000  # includes the marker text

    def test_truncation_structure(self):
        """Result should contain head prefix and tail suffix of the original."""
        head = "HEAD" * 2000  # 8000 chars
        tail = "TAIL" * 2000  # 8000 chars
        data = head + "MIDDLE" * 1000 + tail
        result = render_observation(
            TEMPLATE, {"stdout": data},
            max_chars=10000, truncate_head=5000, truncate_tail=5000,
        )
        # Result starts with first 5000 chars of original
        assert result[:10] == data[:10]
        # Result ends with last 5000 chars of original
        assert result[-10:] == data[-10:]
        assert "elided" in result.lower()

    def test_elided_count_correct(self):
        """Elided count accurately reflects how many chars were removed."""
        data = "x" * 15000
        result = render_observation(
            TEMPLATE, {"stdout": data},
            max_chars=10000, truncate_head=5000, truncate_tail=5000,
        )
        # 15000 - 10000 = 5000 chars elided
        assert "5000" in result
        assert "elided" in result.lower()

    def test_very_large_output(self):
        """100k chars should be handled correctly."""
        data = "y" * 100000
        result = render_observation(
            TEMPLATE, {"stdout": data},
            max_chars=10000, truncate_head=5000, truncate_tail=5000,
        )
        assert "elided" in result.lower()
        # Truncated output = 5000 + marker + 5000
        assert 10000 < len(result) < 12000
