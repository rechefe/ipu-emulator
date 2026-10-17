"""Publish one built docs site as a version on the gh-pages branch (E09-21).

    bazel run //docs:publish_version -- --repo "$GITHUB_WORKSPACE" \\
        --site bazel-bin/docs/site.tar --version v1.2 --alias latest [--push]

Only CI runs it. ``--repo`` has no default: the tool never guesses which
checkout's gh-pages to write (an earlier default from ``bazel run``'s workspace
once pushed from a developer checkout). Without ``--push`` it only commits to
the local gh-pages branch of ``--repo``.

The site is always the offline Bazel build (``//docs:build_docs``); this tool
only lays it out the way mike does, so the Material version selector and the
``mike`` CLI (``mike list``, ``mike alias``, ``mike delete``) work on the
branch:

* ``<version>/`` holds the site;
* each alias is a directory of redirect pages into its version, one per page;
* ``versions.json`` lists ``{"version", "title", "aliases"}``, newest first:
  ``dev`` first, then releases ``vX.Y`` by number, highest first;
* the root ``index.html`` redirects to ``latest`` when a version carries that
  alias, otherwise to the first version.

It works in a temporary worktree of the repository, commits only when the
branch changes (publishing the same site twice changes nothing) and pushes
with ``--push``. CI runs it (.github/workflows/docs.yml); nobody else pushes
to gh-pages.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path

BRANCH = "gh-pages"
DEFAULT_ALIAS = "latest"
_RELEASE = re.compile(r"v(\d+)\.(\d+)")
_ORPHAN = "publish-version-tmp"  # the first commit's branch, deleted afterwards

#: mike's redirect page (its ``redirect.html`` template).
REDIRECT = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Redirecting</title>
  <noscript>
    <meta http-equiv="refresh" content="1; url={href}" />
  </noscript>
  <script>
    window.location.replace(
      "{href}" + window.location.search + window.location.hash
    );
  </script>
</head>
<body>
  Redirecting to <a href="{href}">{href}</a>...
</body>
</html>
"""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=check)


def _sort_key(entry: dict) -> tuple[int, int, int]:
    """``dev`` first, then releases by number, highest first, then anything else."""
    match = _RELEASE.fullmatch(entry["version"])
    if entry["version"] == "dev":
        return (0, 0, 0)
    if match:
        return (1, -int(match[1]), -int(match[2]))
    return (2, 0, 0)


def _versions(entries: list[dict], version: str, aliases: Sequence[str]) -> list[dict]:
    """``entries`` with ``version`` (re)listed and owning ``aliases``, sorted."""
    kept = []
    for entry in entries:
        if entry["version"] == version:
            continue
        kept.append({**entry, "aliases": [a for a in entry["aliases"] if a not in aliases]})
    kept.append({"version": version, "title": version, "aliases": list(aliases)})
    return sorted(kept, key=_sort_key)


def _write_alias(root: Path, alias: str, version: str) -> None:
    """``root/alias``: one redirect per page of ``root/version``, the way mike writes them."""
    alias_dir = root / alias
    shutil.rmtree(alias_dir, ignore_errors=True)
    for page in sorted((root / version).rglob("*.html")):
        relative = page.relative_to(root / version)
        target = alias_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        depth = len(relative.parts)  # alias dir and the page's subdirectories
        href = "../" * depth + version + "/" + relative.as_posix()
        if href.endswith("index.html"):
            href = href[: -len("index.html")]
        target.write_text(REDIRECT.format(href=href), encoding="utf-8")


def _default(entries: list[dict]) -> str:
    for entry in entries:
        if DEFAULT_ALIAS in entry["aliases"]:
            return DEFAULT_ALIAS
    return entries[0]["version"]


def lay_out(root: Path, site_tar: Path, version: str, aliases: Sequence[str]) -> None:
    """Write ``version`` (and its aliases) into a checkout of the gh-pages tree."""
    versions_file = root / "versions.json"
    entries = json.loads(versions_file.read_text()) if versions_file.exists() else []
    version_dir = root / version
    shutil.rmtree(version_dir, ignore_errors=True)
    version_dir.mkdir(parents=True)
    with tarfile.open(site_tar) as tar:
        tar.extractall(version_dir, filter="data")
    entries = _versions(entries, version, aliases)
    for entry in entries:
        for alias in entry["aliases"]:
            _write_alias(root, alias, entry["version"])
    versions_file.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    default = _default(entries)
    (root / "index.html").write_text(REDIRECT.format(href=f"{default}/"), encoding="utf-8")
    (root / ".nojekyll").write_text("", encoding="utf-8")


def publish(
    repo: Path,
    site_tar: Path,
    version: str,
    aliases: Sequence[str] = (),
    *,
    remote: str | None = "origin",
    push: bool = False,
) -> bool:
    """Publish ``site_tar`` as ``version`` on the gh-pages branch of ``repo``.

    Args:
        repo: A git checkout.
        site_tar: The ``//docs:build_docs`` output.
        version: ``dev`` or ``vX.Y``.
        aliases: Aliases moved to this version (``latest``).
        remote: The remote whose gh-pages is the base; ``None`` for local only.
        push: Push the new commit to ``remote``.

    Returns:
        True when the branch changed (and a commit was made).

    Raises:
        ValueError: An alias that is also a version name, or a bad version name.
    """
    if version != "dev" and not _RELEASE.fullmatch(version):
        raise ValueError(f"version must be dev or vX.Y, not {version!r}")
    if version in aliases:
        raise ValueError(f"{version!r} cannot alias itself")
    if remote is not None:
        _git(repo, "fetch", remote, f"{BRANCH}:refs/remotes/{remote}/{BRANCH}", check=False)
    base = f"refs/remotes/{remote}/{BRANCH}" if remote is not None else f"refs/heads/{BRANCH}"
    has_base = _git(repo, "rev-parse", "--verify", "--quiet", base, check=False).returncode == 0
    with tempfile.TemporaryDirectory() as scratch:
        tree = Path(scratch) / "pages"
        if has_base:
            _git(repo, "worktree", "add", "--detach", str(tree), base)
        else:
            _git(repo, "worktree", "add", "--detach", str(tree))
            _git(tree, "checkout", "--orphan", _ORPHAN)
            _git(tree, "rm", "-rf", "--quiet", ".", check=False)
        try:
            lay_out(tree, site_tar, version, aliases)
            _git(tree, "add", "--all")
            if _git(tree, "diff", "--cached", "--quiet", check=False).returncode == 0 and has_base:
                return False
            _git(
                tree,
                "commit",
                "--quiet",
                "-m",
                f"Deploy {version} docs" + (f" ({', '.join(aliases)})" if aliases else ""),
            )
            commit = _git(tree, "rev-parse", "HEAD").stdout.strip()
            _git(repo, "update-ref", f"refs/heads/{BRANCH}", commit)
            if push and remote is not None:
                _git(repo, "push", remote, f"{commit}:refs/heads/{BRANCH}")
            return True
        finally:
            _git(repo, "worktree", "remove", "--force", str(tree), check=False)
            _git(repo, "branch", "-D", _ORPHAN, check=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo", required=True, type=Path, help="the checkout whose gh-pages to write (no default)"
    )
    parser.add_argument("--site", required=True, type=Path, help="the //docs:build_docs site.tar")
    parser.add_argument("--version", required=True, help="dev or vX.Y")
    parser.add_argument("--alias", action="append", default=[], help="an alias to move (latest)")
    parser.add_argument(
        "--push",
        action="store_true",
        help="push gh-pages to the repo's origin (default: commit only)",
    )
    args = parser.parse_args(argv)
    # Relative paths are the caller's, under `bazel run` too.
    caller = Path(os.environ.get("BUILD_WORKING_DIRECTORY", "."))
    repo = args.repo if args.repo.is_absolute() else caller / args.repo
    site = args.site if args.site.is_absolute() else caller / args.site
    changed = publish(repo, site, args.version, args.alias, push=args.push)
    print(f"{args.version}: {'published' if changed else 'unchanged'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
