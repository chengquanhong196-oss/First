"""Tenacity retry configuration for transient model API errors."""

import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# Exceptions that indicate a transient failure worth retrying.
# FormatError is explicitly NOT in this list.
_TRANSIENT_EXCEPTIONS = (
    # httpx / aiohttp transport errors
    Exception,  # Base for all transport errors; tenacity + retry_if below narrows.
)


def is_transient(exception: BaseException) -> bool:
    """Return True if the exception represents a transient API failure.

    FormatError is never considered transient.
    """
    from mini_swe_agent.parser.errors import FormatError

    if isinstance(exception, FormatError):
        return False

    # ConnectionError, TimeoutError, HTTP 5xx, rate limits
    if isinstance(exception, (ConnectionError, TimeoutError)):
        return True

    # Check for HTTP status errors
    status = getattr(exception, "status_code", None)
    if status is not None and status >= 500:
        return True
    if status == 429:
        return True

    # Check for httpx/requests status errors
    if hasattr(exception, "response"):
        resp = getattr(exception, "response", None)
        if resp is not None:
            code = getattr(resp, "status_code", None)
            if code is not None and (code >= 500 or code == 429):
                return True

    return False


def make_retry_decorator(max_attempts: int = 3, min_wait: float = 1.0, max_wait: float = 30.0):
    """Build a tenacity retry decorator for transient API failures.

    Uses is_transient() to filter retryable exceptions.
    FormatError is never retried.
    """
    return retry(
        retry=retry_if_exception(is_transient),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
