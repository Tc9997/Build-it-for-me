"""Tests for web.py URL fetching safety."""

from unittest.mock import patch, MagicMock

import pytest

from build_loop.web import fetch_url, _is_blocked_host


class TestFetchUrlScheme:
    """fetch_url must reject non-HTTP schemes."""

    def test_rejects_file_scheme(self):
        result = fetch_url("file:///etc/hosts")
        assert "Rejected" in result.content

    def test_rejects_ftp_scheme(self):
        result = fetch_url("ftp://evil.com/data")
        assert "Rejected" in result.content

    def test_rejects_gopher_scheme(self):
        result = fetch_url("gopher://evil.com/")
        assert "Rejected" in result.content

    def test_rejects_data_scheme(self):
        result = fetch_url("data:text/html,<h1>hi</h1>")
        assert "Rejected" in result.content


class TestFetchUrlHostBlocking:
    """fetch_url must reject localhost, loopback, and private networks."""

    def test_rejects_localhost(self):
        result = fetch_url("http://localhost/secret")
        assert "Rejected" in result.content
        assert "localhost" in result.content.lower()

    def test_rejects_127_0_0_1(self):
        result = fetch_url("http://127.0.0.1:8080/admin")
        assert "Rejected" in result.content
        assert "loopback" in result.content.lower()

    def test_rejects_ipv6_loopback(self):
        result = fetch_url("http://[::1]/admin")
        assert "Rejected" in result.content

    def test_rejects_link_local(self):
        result = fetch_url("http://169.254.169.254/latest/meta-data/")
        assert "Rejected" in result.content
        assert "link-local" in result.content.lower()

    def test_rejects_10_x_private(self):
        result = fetch_url("http://10.0.0.1/internal")
        assert "Rejected" in result.content
        assert "private" in result.content.lower()

    def test_rejects_192_168_private(self):
        result = fetch_url("http://192.168.1.1/router")
        assert "Rejected" in result.content
        assert "private" in result.content.lower()

    def test_rejects_172_16_private(self):
        result = fetch_url("http://172.16.0.1/internal")
        assert "Rejected" in result.content
        assert "private" in result.content.lower()

    def test_allows_172_outside_private(self):
        """172.32.x.x is not private — should not be blocked by host check."""
        blocked = _is_blocked_host("http://172.32.0.1/ok")
        assert blocked is None


class TestFetchUrlAllowed:
    """Valid public URLs pass validation (tested without real network calls)."""

    def test_allows_https_public(self):
        """https://example.com passes scheme and host checks."""
        blocked = _is_blocked_host("https://example.com")
        assert blocked is None

    def test_allows_http_public(self):
        blocked = _is_blocked_host("http://docs.pydantic.dev/latest/")
        assert blocked is None

    @patch("build_loop.web.subprocess.run")
    def test_fetch_public_url_calls_curl(self, mock_run):
        """A valid public URL should reach curl (mocked)."""
        mock_run.return_value = MagicMock(stdout="<html>ok</html>")
        result = fetch_url("https://example.com")
        mock_run.assert_called_once()
        assert result.content == "<html>ok</html>"


class TestGitHubHelpersUnaffected:
    """GitHub API helpers should not be blocked by fetch_url restrictions."""

    def test_github_api_uses_different_path(self):
        """GitHub helpers use _github_api / _curl_json, not fetch_url."""
        from build_loop.web import _curl_json
        # _curl_json doesn't go through fetch_url's host blocking
        # Just verify it exists and is callable
        assert callable(_curl_json)
