"""Live/fire test gating: only run when MSWEA_LIVE_TESTS=1.

These tests call real model APIs. Documented in README.
"""

import os

import pytest


def requires_live_tests(func):
    """Decorator: skip the test unless MSWEA_LIVE_TESTS=1."""
    return pytest.mark.skipif(
        os.environ.get("MSWEA_LIVE_TESTS") != "1",
        reason="Set MSWEA_LIVE_TESTS=1 to run live/fire tests against real APIs.",
    )(func)
