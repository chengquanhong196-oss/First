"""Unit tests for submission detection."""

from mini_swe_agent.core.submission import check_submission, extract_submission_body, SUBMISSION_MARKER


class TestSubmission:
    def test_rc0_marker_first_line(self):
        assert check_submission(f"{SUBMISSION_MARKER}\nbody", 0) is True

    def test_rc_nonzero_rejects(self):
        assert check_submission(f"{SUBMISSION_MARKER}\nbody", 1) is False
        assert check_submission(f"{SUBMISSION_MARKER}\nbody", 127) is False
        assert check_submission(f"{SUBMISSION_MARKER}\nbody", -1) is False

    def test_no_marker_rc0(self):
        assert check_submission("random output", 0) is False

    def test_empty_stdout(self):
        assert check_submission("", 0) is False

    def test_whitespace_only(self):
        assert check_submission("   \n  \t  \n", 0) is False

    def test_marker_not_first_line(self):
        assert check_submission(f"prefix\n{SUBMISSION_MARKER}", 0) is False

    def test_marker_with_whitespace_before(self):
        assert check_submission(f"   \n{SUBMISSION_MARKER}\nbody", 0) is True

    def test_extract_body(self):
        stdout = f"{SUBMISSION_MARKER}\nActual patch content"
        assert extract_submission_body(stdout) == "Actual patch content"

    def test_extract_body_multiline(self):
        stdout = f"{SUBMISSION_MARKER}\nline1\nline2\nline3"
        assert extract_submission_body(stdout) == "line1\nline2\nline3"

    def test_extract_body_no_marker(self):
        assert extract_submission_body("no marker") == ""
