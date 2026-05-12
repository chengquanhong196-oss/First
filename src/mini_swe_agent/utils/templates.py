"""Jinja2 environment factory and observation template rendering with truncation."""

import logging
import re

import jinja2

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def create_jinja_env() -> jinja2.Environment:
    """Return a Jinja2 Environment configured with StrictUndefined."""
    return jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )


def render_observation(
    template_str: str,
    variables: dict,
    max_chars: int = 10000,
    truncate_head: int = 5000,
    truncate_tail: int = 5000,
) -> str:
    """Render an observation template.

    If rendered output is >= max_chars: emit warning, keep head + tail, note elided count.
    """
    env = create_jinja_env()
    tmpl = env.from_string(template_str)
    rendered = tmpl.render(**variables)

    if len(rendered) < max_chars:
        return rendered

    elided = len(rendered) - max_chars
    logger.warning(
        "Observation output is %d chars (limit %d). Truncating: keeping first %d + last %d. Elided %d chars.",
        len(rendered),
        max_chars,
        truncate_head,
        truncate_tail,
        elided,
    )

    head = rendered[:truncate_head]
    tail = rendered[-truncate_tail:] if truncate_tail > 0 else ""
    return (
        f"{head}\n\n... [{elided} chars elided] ...\n\n{tail}"
    )


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_RE.sub("", text)
