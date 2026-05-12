"""Structured logging setup for the mini SWE agent."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with structured output."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
