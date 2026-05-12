"""Submission detection: checks if the model has signaled task completion."""

SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def check_submission(stdout: str, returncode: int) -> bool:
    """Return True iff the model has successfully submitted.

    Conditions (both must hold):
    1. returncode == 0
    2. The first non-blank line of stdout (lstrip) is exactly the marker.

    If returncode != 0: submission is rejected even if the marker text is present.
    """
    if returncode != 0:
        return False

    stripped = stdout.lstrip()
    if not stripped:
        return False

    first_line = stripped.split("\n")[0].strip()
    return first_line == SUBMISSION_MARKER


def extract_submission_body(stdout: str) -> str:
    """Extract the submission body (everything after the marker line)."""
    stripped = stdout.lstrip()
    lines = stripped.split("\n")
    if lines and lines[0].strip() == SUBMISSION_MARKER:
        return "\n".join(lines[1:]).strip()
    return ""
