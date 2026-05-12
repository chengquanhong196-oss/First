"""Unit tests for tenacity retry infrastructure."""
import pytest

from mini_swe_agent.parser.errors import FormatError
from mini_swe_agent.utils.retry import is_transient, make_retry_decorator


class TestIsTransient:
    def test_format_error_never_transient(self):
        """FormatError must never be retried."""
        assert is_transient(FormatError(reason="zero_actions", feedback="none")) is False

    def test_connection_error_is_transient(self):
        assert is_transient(ConnectionError("refused")) is True

    def test_timeout_error_is_transient(self):
        assert is_transient(TimeoutError("timed out")) is True

    def test_value_error_not_transient(self):
        """Regular ValueError is not transient."""
        assert is_transient(ValueError("bad input")) is False

    def test_http_500_status_is_transient(self):
        """Exception with status_code >= 500 is transient."""
        exc = Exception("server error")
        exc.status_code = 500
        assert is_transient(exc) is True

    def test_http_502_status_is_transient(self):
        exc = Exception("bad gateway")
        exc.status_code = 502
        assert is_transient(exc) is True

    def test_http_429_status_is_transient(self):
        """Rate limit 429 is transient."""
        exc = Exception("rate limited")
        exc.status_code = 429
        assert is_transient(exc) is True

    def test_http_400_status_not_transient(self):
        """Client error 400 is NOT transient."""
        exc = Exception("bad request")
        exc.status_code = 400
        assert is_transient(exc) is False

    def test_response_level_status(self):
        """Transient check via exc.response.status_code."""
        exc = Exception("gateway timeout")

        class FakeResponse:
            status_code = 504
        exc.response = FakeResponse()
        assert is_transient(exc) is True

    def test_response_level_429(self):
        exc = Exception("rate limited")

        class FakeResponse:
            status_code = 429
        exc.response = FakeResponse()
        assert is_transient(exc) is True

    def test_response_level_404_not_transient(self):
        exc = Exception("not found")

        class FakeResponse:
            status_code = 404
        exc.response = FakeResponse()
        assert is_transient(exc) is False


class TestMakeRetryDecorator:
    def test_returns_retry_decorator(self):
        """make_retry_decorator returns a valid tenacity retry decorator."""
        deco = make_retry_decorator(max_attempts=2, min_wait=0.01, max_wait=0.05)
        assert deco is not None

    @pytest.mark.asyncio
    async def test_retry_on_transient(self):
        """Decorated function retries on transient errors then succeeds."""
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "success"

        deco = make_retry_decorator(max_attempts=3, min_wait=0.001, max_wait=0.01)

        @deco
        async def wrapped():
            return await flaky()

        result = await wrapped()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_format_error(self):
        """FormatError is raised immediately without retry."""
        call_count = 0

        async def bad_format():
            nonlocal call_count
            call_count += 1
            raise FormatError(reason="zero_actions", feedback="none")

        deco = make_retry_decorator(max_attempts=3, min_wait=0.001, max_wait=0.01)

        @deco
        async def wrapped():
            return await bad_format()

        with pytest.raises(FormatError):
            await wrapped()
        assert call_count == 1  # no retry, immediate raise

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self):
        """After max_attempts transient errors, the final exception propagates."""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("persistent")

        deco = make_retry_decorator(max_attempts=2, min_wait=0.001, max_wait=0.01)

        @deco
        async def wrapped():
            return await always_fails()

        with pytest.raises(ConnectionError):
            await wrapped()
        assert call_count == 2
