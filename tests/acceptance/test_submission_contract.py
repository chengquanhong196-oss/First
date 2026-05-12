"""Acceptance test: Submission contract (Section E and L).

- returncode==0 + marker first line -> submitted
- returncode!=0 + marker present -> NOT submitted (no silent submission)
"""

from mini_swe_agent.core.submission import check_submission, extract_submission_body, SUBMISSION_MARKER


class TestSubmissionSuccess:
    """returncode==0 AND marker first non-blank line -> success."""

    def test_marker_first_line(self):
        stdout = f"{SUBMISSION_MARKER}\nPatch content here"
        assert check_submission(stdout, 0) is True

    def test_marker_with_leading_whitespace(self):
        stdout = f"   \n  {SUBMISSION_MARKER}\nbody"
        assert check_submission(stdout, 0) is True

    def test_extract_submission_body(self):
        stdout = f"{SUBMISSION_MARKER}\nActual patch"
        body = extract_submission_body(stdout)
        assert body == "Actual patch"


class TestSubmissionRejected:
    """returncode!=0 must reject even with marker."""

    def test_nonzero_returncode_rejects(self):
        stdout = f"{SUBMISSION_MARKER}\npatch"
        assert check_submission(stdout, 1) is False
        assert check_submission(stdout, -1) is False
        assert check_submission(stdout, 127) is False

    def test_rc_zero_no_marker(self):
        stdout = "Some other output"
        assert check_submission(stdout, 0) is False

    def test_empty_stdout(self):
        assert check_submission("", 0) is False
        assert check_submission("   \n   ", 0) is False

    def test_marker_not_first_line(self):
        stdout = f"Some output before\n{SUBMISSION_MARKER}"
        assert check_submission(stdout, 0) is False
