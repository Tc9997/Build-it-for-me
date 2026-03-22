"""Researcher agent: multi-step research with web search and GitHub repo analysis.

Step 1: LLM generates search queries from the idea
Step 2: Execute searches (web + GitHub)
Step 3: Fetch and read top repos (READMEs, key files, examples)
Step 4: LLM synthesizes everything into a ResearchReport
"""

from __future__ import annotations

import json

from build_loop.agents.base import Agent
from build_loop.schemas import ResearchReport
from build_loop import llm, web


# ---------------------------------------------------------------------------
# Step 1: Generate search queries
# ---------------------------------------------------------------------------

QUERY_SYSTEM = """\
You are a research assistant. Given a project idea, generate search queries to find \
the best libraries, APIs, and reference implementations.

Respond with a JSON object:
{
  "web_queries": ["search query 1", "search query 2", ...],
  "github_queries": ["github search 1", "github search 2", ...],
  "specific_repos": ["owner/repo if you know exact repos that are relevant"],
  "specific_urls": ["any specific documentation URLs to check"]
}

Rules:
- Generate 3-5 web queries targeting: best libraries, API docs, tutorials, gotchas
- Generate 2-3 GitHub queries targeting: reference implementations, popular tools
- If you know specific repos that do exactly what's needed (or close), list them
- If you know specific doc pages (PyPI, readthedocs, API docs), list the URLs
- Respond with ONLY the JSON object.
"""

# ---------------------------------------------------------------------------
# Step 2: Pick which repos/pages to deep-read
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM = """\
You are a research assistant triaging search results. Given search results for a project idea, \
pick the most useful GitHub repos to read in detail and any URLs worth fetching.

Respond with a JSON object:
{
  "repos_to_read": [
    {"repo": "owner/name", "reason": "why this is relevant", "files_to_read": ["src/main.py", "examples/basic.py"]}
  ],
  "urls_to_fetch": ["url1", "url2"],
  "notes": "anything notable from the search snippets alone"
}

Rules:
- Pick at most 5 repos. Prefer: high stars, active maintenance, good examples, close to what we need.
- For each repo, list specific files to read (not just README — look for examples, core modules).
- Pick at most 3 URLs to fetch (API docs, tutorials with code).
- Respond with ONLY the JSON object.
"""

# ---------------------------------------------------------------------------
# Step 4: Synthesize
# ---------------------------------------------------------------------------

SYNTHESIZE_SYSTEM = """\
You are the Researcher agent in an automated build system. You have been given a project idea \
along with REAL search results, GitHub repo contents, and documentation fetched from the web.

Your job: synthesize all of this into a concrete research report that a Planner agent can use \
to build a working project.

You MUST respond with a single JSON object:

{
  "feasibility": "string — overall assessment: is this buildable? what's the hardest part?",
  "findings": [
    {
      "topic": "string — e.g. 'web scraping wine auctions'",
      "summary": "string — what you found from REAL sources",
      "recommended_approach": "string — the best way to do this, citing specific repos/libraries",
      "libraries": ["exact pip package names with versions if important"],
      "apis": ["API endpoints / services with auth details"],
      "risks": ["things that could go wrong"],
      "code_snippets": {"label": "REAL working code from repos/docs, adapted for our use case"}
    }
  ],
  "recommended_stack": ["python3.12", "beautifulsoup4", "fastapi", ...],
  "external_services": ["things that need to be running — docker containers, databases, etc."],
  "environment_setup": ["pip install X", "docker run Y", "export API_KEY=..."],
  "reference_repos": ["owner/repo — repos to use as implementation reference"],
  "open_questions": ["things that can't be resolved without user input"]
}

Rules:
- CITE REAL sources. If you saw it in a repo, say "from owner/repo". If from docs, say where.
- Code snippets should be REAL code from the repos/docs you read, adapted for this project — \
  NOT invented from your training data.
- If a library's README shows the exact API usage pattern, include it verbatim.
- Be honest — if the search results contradict your expectations, trust the search results.
- Name exact pip packages (not just library names). e.g. "beautifulsoup4" not "BeautifulSoup".
- Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


class ResearcherAgent(Agent):
    name = "researcher"
    system_prompt = SYNTHESIZE_SYSTEM
    model = "claude-haiku-4-5-20251001"  # Cheap for light research; full uses sonnet for synthesis

    def run(self, idea: str, light: bool = False) -> ResearchReport:
        """Run research. light=True skips web search and uses LLM knowledge only."""
        if light:
            return self._run_light(idea)
        return self._run_full(idea)

    def _run_light(self, idea: str) -> ResearchReport:
        """Light research: LLM knowledge only, no web search. Much cheaper."""
        self.log("light research (no web search)...")
        data = self.call_json(
            f"PROJECT IDEA:\n{idea}\n\n"
            "Provide your research report based on your training knowledge. "
            "Do NOT cite specific repos or URLs — just recommend libraries, "
            "patterns, and approaches you know work."
        )
        report = ResearchReport(**data)
        self.log(f"  {len(report.findings)} findings, stack={report.recommended_stack}")
        return report

    def _run_full(self, idea: str) -> ResearchReport:
        # Step 1: Generate search queries
        self.log("generating search queries...")
        queries = llm.call_json(
            system=QUERY_SYSTEM,
            messages=[{"role": "user", "content": f"PROJECT IDEA:\n{idea}"}],
        )

        # Step 2: Execute searches
        self.log("searching web and GitHub...")
        search_results = self._execute_searches(queries)

        # Step 3: Triage — pick what to read in detail
        self.log("triaging results...")
        triage = llm.call_json(
            system=TRIAGE_SYSTEM,
            messages=[{"role": "user", "content": (
                f"PROJECT IDEA:\n{idea}\n\n"
                f"SEARCH RESULTS:\n{json.dumps(search_results, indent=2)}"
            )}],
        )

        # Step 4: Deep-read repos and URLs
        self.log("reading repos and docs...")
        deep_content = self._deep_read(triage)

        # Step 5: Synthesize everything (truncate to fit context window)
        self.log("synthesizing research report...")
        search_json = json.dumps(search_results, indent=2)
        content_json = json.dumps(deep_content, indent=2)

        # Rough token estimate: ~4 chars per token. Keep under 150k tokens.
        max_chars = 500_000
        total = len(idea) + len(search_json) + len(content_json)
        if total > max_chars:
            # Truncate deep content first (largest), then search results
            budget = max_chars - len(idea) - 1000  # leave room for framing
            search_budget = min(len(search_json), budget // 3)
            content_budget = budget - search_budget
            search_json = search_json[:search_budget] + "\n... (truncated)"
            content_json = content_json[:content_budget] + "\n... (truncated)"
            self.log(f"  truncated research context from {total} to ~{max_chars} chars")

        synthesis_prompt = (
            f"PROJECT IDEA:\n{idea}\n\n"
            f"SEARCH RESULTS:\n{search_json}\n\n"
            f"REPO & DOC CONTENTS:\n{content_json}"
        )
        # Synthesis needs quality — use sonnet even though researcher defaults to haiku
        data = llm.call_json(
            system=self.system_prompt,
            messages=[{"role": "user", "content": synthesis_prompt}],
            model="claude-sonnet-4-20250514",
        )

        report = ResearchReport(**data)
        self.log(
            f"research complete: {len(report.findings)} findings, "
            f"stack={report.recommended_stack}"
        )
        return report

    def _execute_searches(self, queries: dict) -> dict:
        """Run all search queries and collect results."""
        results = {"web": [], "github": [], "specific_repos": [], "specific_urls": []}

        # Web searches
        for q in queries.get("web_queries", [])[:5]:
            self.log(f"  web: {q}")
            hits = web.search_web(q)
            results["web"].extend([
                {"query": q, "title": h.title, "url": h.url, "snippet": h.snippet}
                for h in hits
            ])

        # GitHub searches
        for q in queries.get("github_queries", [])[:3]:
            self.log(f"  github: {q}")
            hits = web.search_github(q)
            results["github"].extend([
                {"query": q, "title": h.title, "url": h.url, "snippet": h.snippet}
                for h in hits
            ])

        # Known repos — fetch their READMEs immediately
        for repo in queries.get("specific_repos", [])[:5]:
            self.log(f"  repo: {repo}")
            readme = web.fetch_github_readme(repo)
            if readme:
                results["specific_repos"].append({
                    "repo": repo,
                    "readme": readme[:8000]
                })

        # Known URLs — fetch immediately
        for url in queries.get("specific_urls", [])[:3]:
            self.log(f"  url: {url}")
            page = web.fetch_url(url)
            if page.content:
                results["specific_urls"].append({
                    "url": url,
                    "content": page.content[:8000]
                })

        return results

    def _deep_read(self, triage: dict) -> dict:
        """Read specific files from triaged repos and fetch URLs."""
        content = {"repos": [], "pages": []}

        for repo_info in triage.get("repos_to_read", [])[:5]:
            repo = repo_info.get("repo", "")
            if not repo:
                continue

            repo_data = {"repo": repo, "reason": repo_info.get("reason", ""), "files": {}}

            # Always get README
            readme = web.fetch_github_readme(repo)
            if readme:
                repo_data["files"]["README.md"] = readme[:8000]

            # Get file tree
            tree = web.fetch_github_tree(repo)
            if tree:
                repo_data["tree"] = tree[:50]  # First 50 files for context

            # Fetch specific requested files
            for fpath in repo_info.get("files_to_read", [])[:5]:
                self.log(f"  reading {repo}/{fpath}")
                file_content = web.fetch_github_file(repo, fpath)
                if file_content:
                    repo_data["files"][fpath] = file_content

            content["repos"].append(repo_data)

        # Fetch URLs
        for url in triage.get("urls_to_fetch", [])[:3]:
            self.log(f"  fetching {url}")
            page = web.fetch_url(url)
            if page.content:
                content["pages"].append({"url": url, "content": page.content[:8000]})

        return content
