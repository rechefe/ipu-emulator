#!/usr/bin/env python3
"""Fetch merged pull requests from GitHub and update docs/content/release-log.md.

Usage (one-time historical backfill or full refresh):
    python docs/update_release_log.py

Usage (add only the most recently merged PR, e.g. from CI):
    python docs/update_release_log.py --latest

The script reads GITHUB_TOKEN from the environment when available so that the
GitHub API rate limit is much higher (5 000 req/h vs 60 req/h unauthenticated).
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import json

REPO = "rechefe/ipu-emulator"
API_BASE = "https://api.github.com"
_DEFAULT_OUTPUT = Path(__file__).parent / "content" / "release-log.md"

HEADER = """\
# Release Log

This page lists every pull request that has been merged into `master`, newest
first.  It is regenerated automatically as part of the docs build on every
merge.

| # | Title | Merged at (UTC) |
|---|-------|----------------|
""".format(
    repo=REPO
)


def _get(url: str, token: str | None) -> dict | list:
    """Perform a GitHub API GET request and return decoded JSON."""
    req = Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def fetch_merged_prs(token: str | None, latest_only: bool = False) -> list[dict]:
    """Return a list of merged-PR dicts, sorted newest-first."""
    prs: list[dict] = []
    page = 1
    per_page = 100
    while True:
        url = (
            f"{API_BASE}/repos/{REPO}/pulls"
            f"?state=closed&base=master&per_page={per_page}&page={page}"
            f"&sort=updated&direction=desc"
        )
        try:
            batch = _get(url, token)
        except HTTPError as exc:
            print(
                f"Warning: GitHub API error while fetching release log: {exc}. "
                "Writing a minimal release log instead.",
                file=sys.stderr,
            )
            return []

        if not batch:
            break

        for pr in batch:
            if pr.get("merged_at"):
                prs.append(pr)
                if latest_only:
                    return prs[:1]

        if len(batch) < per_page or latest_only:
            break
        page += 1

    prs.sort(key=lambda p: p["merged_at"], reverse=True)
    return prs


def format_row(pr: dict) -> str:
    number = pr["number"]
    title = pr["title"].replace("|", "\\|")
    url = pr["html_url"]
    merged_at = pr["merged_at"]  # ISO-8601 string, e.g. "2024-03-15T10:22:33Z"
    # Parse and format for readability
    dt = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
    merged_str = dt.strftime("%Y-%m-%d %H:%M")
    return f"| [#{number}]({url}) | {title} | {merged_str} |\n"


def build_content(prs: list[dict]) -> str:
    rows = "".join(format_row(pr) for pr in prs)
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    footer = f"\n*Last updated: {updated} UTC*\n"
    return HEADER + rows + footer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Fetch only the most-recently merged PR and rebuild the file (fast, for CI use).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Path to write the generated release-log.md (default: docs/content/release-log.md).",
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "Warning: GITHUB_TOKEN not set; requests are unauthenticated "
            "(60 req/h limit).",
            file=sys.stderr,
        )

    prs = fetch_merged_prs(token, latest_only=args.latest)
    if not prs:
        print("No merged PRs found; writing a minimal release log.", file=sys.stderr)

    content = build_content(prs)
    args.output.write_text(content, encoding="utf-8")
    print(f"Written {len(prs)} PR(s) to {args.output}")


if __name__ == "__main__":
    main()
