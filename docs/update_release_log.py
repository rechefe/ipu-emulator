#!/usr/bin/env python3
"""Fetch merged pull requests from GitHub and update docs/content/release-log.md.

The log is committed, so the docs build never reaches the network. The
release-log workflow (.github/workflows/release-log.yml) refreshes it on master
after every merge with ``--recent``.

Usage (one-time historical backfill or full refresh):
    bazel run //docs:update_release_log -- --output "$PWD/docs/content/release-log.md"

Usage (merge the PRs on the newest page of results into the existing log):
    bazel run //docs:update_release_log -- --recent --output "$PWD/docs/content/release-log.md"

Usage (add only the most recently merged PR to the existing log):
    bazel run //docs:update_release_log -- --latest --output "$PWD/docs/content/release-log.md"

The script reads GITHUB_TOKEN from the environment when available so that the
GitHub API rate limit is much higher (5 000 req/h vs 60 req/h unauthenticated).

A network failure never fails the script: a full refresh writes a minimal log,
and ``--latest`` or ``--recent`` leaves an existing log unchanged.
"""

import argparse
import http.client
import json
import os
import re
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

REPO = "rechefe/ipu-emulator"
API_BASE = "https://api.github.com"
_DEFAULT_OUTPUT = Path(__file__).parent / "content" / "release-log.md"

HEADER = """\
# Release Log

This page lists every pull request that has been merged into `master`, newest
first.  A workflow refreshes it on `master` after every merge.

| # | Title | Merged at (UTC) |
|---|-------|----------------|
"""

# One table row as written by format_row: | [#N](url) | title | YYYY-MM-DD HH:MM |
# The title may contain escaped pipes (``\\|``).
_ROW_RE = re.compile(
    r"^\| \[#(?P<number>\d+)\]\((?P<url>[^)]*)\) \| (?P<title>(?:[^|\\]|\\.)*?) \| "
    r"(?P<merged>\d{4}-\d{2}-\d{2} \d{2}:\d{2}) \|\s*$"
)


class FetchError(Exception):
    """The GitHub API could not be reached or returned an error."""


def _get(url: str, token: str | None) -> dict | list:
    """Perform a GitHub API GET request and return decoded JSON."""
    req = Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _fetch_page(page: int, per_page: int, token: str | None) -> list[dict]:
    url = (
        f"{API_BASE}/repos/{REPO}/pulls"
        f"?state=closed&base=master&per_page={per_page}&page={page}"
        f"&sort=updated&direction=desc"
    )
    try:
        return _get(url, token)
    # URLError covers HTTPError; TimeoutError (socket.timeout) and other socket
    # errors are OSError subclasses, as is URLError itself.  HTTPException
    # covers truncated responses; ValueError covers a body that is not JSON.
    except (URLError, TimeoutError, OSError, http.client.HTTPException, ValueError) as exc:
        raise FetchError(str(exc)) from exc


class Fetch(Enum):
    """How much of the PR list ``fetch_merged_prs`` reads."""

    #: Every page: the whole history.
    ALL = "all"
    #: Every merged PR on the first page (the 100 most recently updated PRs).
    RECENT = "recent"
    #: The single PR with the greatest ``merged_at`` on the first page (the page
    #: is ordered by ``updated``, not by merge time).
    LATEST = "latest"


def fetch_merged_prs(token: str | None, fetch: Fetch = Fetch.ALL) -> list[dict]:
    """Return merged-PR dicts, newest merge first.

    Args:
        token: A GitHub token, or None for unauthenticated requests.
        fetch: How much of the list to read.

    Raises:
        FetchError: the API cannot be reached.
    """
    prs: list[dict] = []
    page = 1
    per_page = 100
    while True:
        batch = _fetch_page(page, per_page, token)
        if not batch:
            break
        prs.extend(pr for pr in batch if pr.get("merged_at"))
        if len(batch) < per_page or fetch is not Fetch.ALL:
            break
        page += 1

    prs.sort(key=lambda p: p["merged_at"], reverse=True)
    return prs[:1] if fetch is Fetch.LATEST else prs


def format_row(pr: dict) -> str:
    number = pr["number"]
    title = pr["title"].replace("|", "\\|")
    url = pr["html_url"]
    merged_at = pr["merged_at"]  # ISO-8601 string, e.g. "2024-03-15T10:22:33Z"
    # Parse and format for readability
    dt = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
    merged_str = dt.strftime("%Y-%m-%d %H:%M")
    return f"| [#{number}]({url}) | {title} | {merged_str} |\n"


def _assemble(rows: list[str]) -> str:
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    footer = f"\n*Last updated: {updated} UTC*\n"
    return HEADER + "".join(rows) + footer


def build_content(prs: list[dict]) -> str:
    return _assemble([format_row(pr) for pr in prs])


def parse_rows(text: str) -> dict[int, tuple[str, str]]:
    """Parse a release log's table rows into {number: (merged, row_line)}."""
    rows: dict[int, tuple[str, str]] = {}
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if m:
            rows[int(m["number"])] = (m["merged"], line.rstrip() + "\n")
    return rows


def merge_rows(existing_text: str, prs: list[dict]) -> str:
    """Insert or replace ``prs`` (by number) in the log ``existing_text``.

    Returns the full log with every row sorted by merge time, newest first.
    No existing row is removed.
    """
    rows = parse_rows(existing_text)
    for pr in prs:
        line = format_row(pr)
        rows[pr["number"]] = (_ROW_RE.match(line.rstrip("\n"))["merged"], line)
    ordered = sorted(rows.items(), key=lambda kv: (kv[1][0], kv[0]), reverse=True)
    return _assemble([line for _, (_, line) in ordered])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Fetch only the most recently merged PR and add it to the existing "
        "log (fast, for CI use).",
    )
    parser.add_argument(
        "--recent",
        action="store_true",
        help="Fetch the 100 most recently updated PRs and merge every merged one "
        "into the existing log (the release-log workflow's mode; catches merges "
        "a previous run missed).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Path to write the generated release-log.md (default: docs/content/release-log.md).",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "Warning: GITHUB_TOKEN not set; requests are unauthenticated (60 req/h limit).",
            file=sys.stderr,
        )

    incremental = args.latest or args.recent
    existing = args.output.exists() if incremental else False

    try:
        fetch = Fetch.LATEST if args.latest else Fetch.RECENT if args.recent else Fetch.ALL
        prs = fetch_merged_prs(token, fetch)
    except FetchError as exc:
        if existing:
            print(
                f"Warning: GitHub API error while fetching release log: {exc}. "
                f"Leaving {args.output} unchanged.",
                file=sys.stderr,
            )
            return 0
        print(
            f"Warning: GitHub API error while fetching release log: {exc}. "
            "Writing a minimal release log instead.",
            file=sys.stderr,
        )
        prs = []

    if not prs:
        if existing:
            print(f"No merged PRs found; leaving {args.output} unchanged.", file=sys.stderr)
            return 0
        print("No merged PRs found; writing a minimal release log.", file=sys.stderr)

    if existing:
        content = merge_rows(args.output.read_text(encoding="utf-8"), prs)
    else:
        content = build_content(prs)
    args.output.write_text(content, encoding="utf-8")
    print(f"Written {len(prs)} PR(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
