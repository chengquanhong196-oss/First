"""FormatError for invalid model action formatting."""


class FormatError(Exception):
    """Raised when the model response has 0, >1, or conflicting action formats.

    This error provides structured feedback that gets appended to the
    conversation so the model can correct itself. It must never be
    retried as a transient error.
    """

    def __init__(self, reason: str, feedback: str) -> None:
        self.reason = reason
        self.feedback = feedback
        super().__init__(feedback)
