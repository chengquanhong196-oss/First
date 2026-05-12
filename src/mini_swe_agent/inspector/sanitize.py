"""Sanitization utilities for display: strip ANSI, NUL, control chars."""

import re

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def sanitize(text: str) -> str:
    """Make text safe for Textual display.

    - Replace NUL bytes with Unicode replacement character
    - Strip ANSI escape sequences
    - Replace other control characters (except newline and tab) with space
    """
    if not text:
        return ""

    # Replace NUL
    text = text.replace("\x00", "�")

    # Strip ANSI escapes
    text = _ANSI_RE.sub("", text)

    # Replace other control chars (keep \n, \t)
    result = []
    for ch in text:
        code = ord(ch)
        if code < 32 and code not in (9, 10):  # tab, newline
            result.append(" ")
        else:
            result.append(ch)

    return "".join(result)
