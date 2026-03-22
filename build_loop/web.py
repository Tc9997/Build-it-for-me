"""Web search and content fetching utilities for the research phase.

Uses curl for HTTP and the GitHub REST API (unauthenticated or via gh CLI if available).
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
from dataclasses import dataclass
from shutil import which
from urllib.parse import quote_plus, unquote

from rich.console import Console

console = Console()

_HAS_GH = which("gh") is not None


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass
class FetchedPage:
    url: str
    content: str
    truncated: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _curl_json(url: str, timeout: int = 15) -> dict | list | None:
    """Fetch a URL with curl and parse as JSON."""
    try:
        proc = subprocess.run(
            ["curl", "-sL", "-H", "Accept: application/vnd.github+json",
             "-A", "build-loop/0.1", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)
    except (json.JSONDecodeError, subprocess.TimeoutExpired, Exception):
        return None


def _normalize_repo(repo: str) -> str:
    """Normalize 'owner/name' from a GitHub URL or short form."""
    if "github.com/" in repo:
        parts = repo.rstrip("/").split("github.com/")[-1].split("/")
        return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else repo
    return repo


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def search_web(query: str, max_results: int = 8) -> list[SearchResult]:
    """Search the web. Tries ddgr first, falls back to DuckDuckGo HTML scraping."""
    results = _search_via_ddgr(query, max_results)
    if results:
        return results
    return _search_via_ddg_lite(query, max_results)


def _search_via_ddgr(query: str, max_results: int) -> list[SearchResult]:
    """Use ddgr CLI if installed."""
    if not which("ddgr"):
        return []
    try:
        proc = subprocess.run(
            ["ddgr", "--json", "-n", str(max_results), query],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            return []
        data = json.loads(proc.stdout)
        return [
            SearchResult(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("abstract", ""))
            for r in data
        ]
    except (json.JSONDecodeError, subprocess.TimeoutExpired, Exception):
        return []


def _search_via_ddg_lite(query: str, max_results: int) -> list[SearchResult]:
    """Fallback: DuckDuckGo HTML lite."""
    try:
        proc = subprocess.run(
            ["curl", "-sL", "-A", "Mozilla/5.0",
             f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            return []
        results = []
        links = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', proc.stdout, re.DOTALL)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</(?:td|div|span)>', proc.stdout, re.DOTALL)
        for i, (url, title) in enumerate(links[:max_results]):
            if "uddg=" in url:
                m = re.search(r'uddg=([^&]+)', url)
                if m:
                    url = unquote(m.group(1))
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
            results.append(SearchResult(title=title, url=url, snippet=snippet))
        return results
    except Exception:
        return []


# ---------------------------------------------------------------------------
# GitHub search (gh CLI or REST API fallback)
# ---------------------------------------------------------------------------

def search_github(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search GitHub repos. Uses gh CLI if available, otherwise GitHub REST API via curl."""
    if _HAS_GH:
        return _search_github_gh(query, max_results)
    return _search_github_api(query, max_results)


def _search_github_gh(query: str, max_results: int) -> list[SearchResult]:
    try:
        proc = subprocess.run(
            ["gh", "search", "repos", query, "--limit", str(max_results),
             "--json", "fullName,description,url,stargazersCount"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            return _search_github_api(query, max_results)
        data = json.loads(proc.stdout)
        return [
            SearchResult(
                title=f"{r['fullName']} ({r.get('stargazersCount', 0)}★)",
                url=r.get("url", ""),
                snippet=r.get("description", "") or "",
            )
            for r in data
        ]
    except Exception:
        return _search_github_api(query, max_results)


def _search_github_api(query: str, max_results: int) -> list[SearchResult]:
    """Search via unauthenticated GitHub REST API (60 requests/hour)."""
    url = f"https://api.github.com/search/repositories?q={quote_plus(query)}&sort=stars&per_page={max_results}"
    data = _curl_json(url)
    if not data or "items" not in data:
        return []
    return [
        SearchResult(
            title=f"{r['full_name']} ({r.get('stargazers_count', 0)}★)",
            url=r.get("html_url", ""),
            snippet=r.get("description", "") or "",
        )
        for r in data["items"]
    ]


# ---------------------------------------------------------------------------
# GitHub content fetching (gh CLI or REST API fallback)
# ---------------------------------------------------------------------------

def _github_api(endpoint: str) -> dict | list | None:
    """Call a GitHub API endpoint via gh or curl."""
    if _HAS_GH:
        try:
            proc = subprocess.run(
                ["gh", "api", endpoint],
                capture_output=True, text=True, timeout=15,
            )
            if proc.returncode == 0:
                return json.loads(proc.stdout)
        except Exception:
            pass
    # Fallback to curl
    return _curl_json(f"https://api.github.com/{endpoint}")


def fetch_github_readme(repo: str) -> str:
    """Fetch a GitHub repo's README content."""
    repo = _normalize_repo(repo)
    data = _github_api(f"repos/{repo}/readme")
    if not data or "content" not in data:
        return ""
    try:
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except Exception:
        return ""


def fetch_github_tree(repo: str, path: str = "") -> list[str]:
    """List files in a GitHub repo directory."""
    repo = _normalize_repo(repo)
    endpoint = f"repos/{repo}/contents/{path}" if path else f"repos/{repo}/contents"
    data = _github_api(endpoint)
    if not isinstance(data, list):
        return []
    return [item.get("path", "") for item in data if item.get("path")]


def fetch_github_file(repo: str, path: str) -> str:
    """Fetch a specific file from a GitHub repo."""
    repo = _normalize_repo(repo)
    data = _github_api(f"repos/{repo}/contents/{path}")
    if not data or "content" not in data:
        return ""
    try:
        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        if len(content) > 30000:
            return content[:30000] + "\n\n... (truncated)"
        return content
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Generic URL fetching
# ---------------------------------------------------------------------------

def _is_blocked_host(url: str) -> str | None:
    """Check if a URL targets a blocked host. Returns reason or None."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
    except Exception:
        return "Malformed URL"

    if not host:
        return "No hostname in URL"

    # Reject localhost and loopback
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return f"Blocked: localhost/loopback ({host})"

    # Reject link-local
    if host.startswith("169.254."):
        return f"Blocked: link-local address ({host})"

    # Reject private RFC1918 ranges
    if host.startswith("10."):
        return f"Blocked: private network ({host})"
    if host.startswith("192.168."):
        return f"Blocked: private network ({host})"
    if host.startswith("172."):
        try:
            second = int(host.split(".")[1])
            if 16 <= second <= 31:
                return f"Blocked: private network ({host})"
        except (ValueError, IndexError):
            pass

    return None


def fetch_url(url: str, max_chars: int = 30000) -> FetchedPage:
    """Fetch a URL and return its text content.

    Only http/https allowed. Rejects file://, ftp://, localhost,
    loopback, link-local, and private RFC1918 targets.
    """
    # Reject non-HTTP schemes
    if not (url.startswith("http://") or url.startswith("https://")):
        return FetchedPage(url=url, content=f"Rejected: only http/https URLs are allowed, got: {url}")

    # Reject blocked hosts (SSRF protection)
    blocked = _is_blocked_host(url)
    if blocked:
        return FetchedPage(url=url, content=f"Rejected: {blocked}")

    try:
        proc = subprocess.run(
            ["curl", "-sL", "-A", "Mozilla/5.0", "--max-time", "15", url],
            capture_output=True, text=True, timeout=20,
        )
        content = proc.stdout
        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars]
        return FetchedPage(url=url, content=content, truncated=truncated)
    except Exception as e:
        return FetchedPage(url=url, content=f"Failed to fetch: {e}")
