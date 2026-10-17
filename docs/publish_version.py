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
import uuid
from collections.abc import Sequence
from pathlib import Path

BRANCH = "gh-pages"
DEFAULT_ALIAS = "latest"
_RELEASE = re.compile(r"v(\d+)\.(\d+)")

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
    """``entries`` with ``version`` (re)listed and owning ``aliases``, sorted.

    A republished version keeps its title and properties, as mike keeps them.
    """
    kept = []
    previous: dict = {}
    for entry in entries:
        if entry["version"] == version:
            previous = entry
            continue
        kept.append({**entry, "aliases": [a for a in entry["aliases"] if a not in aliases]})
    kept.append(
        {
            **previous,
            "version": version,
            "title": previous.get("title", version),
            "aliases": list(aliases),
        }
    )
    return sorted(kept, key=_sort_key)


def _newest_release(entries: list[dict], version: str) -> bool:
    """Whether ``version`` sorts first among the releases (``vX.Y``) of ``entries``."""
    releases = [e for e in entries if _RELEASE.fullmatch(e["version"])]
    releases.append({"version": version})
    return min(releases, key=_sort_key)["version"] == version


def _check_alias(alias: str, version: str, entries: list[dict]) -> None:
    """mike's rules for an alias, plus: never ``dev`` or a version name."""
    if not alias or alias in (".", "..") or re.search(r"[\\/]", alias):
        raise ValueError(f"{alias!r} is not a valid alias")
    if alias == "dev" or _RELEASE.fullmatch(alias):
        raise ValueError(f"alias {alias!r} is a version name")
    if alias == version or any(e["version"] == alias for e in entries):
        raise ValueError(f"alias {alias!r} is already a version")


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
        if relative.name == "index.html":
            href = href[: -len("index.html")]
        target.write_text(REDIRECT.format(href=href), encoding="utf-8")


def _default(entries: list[dict]) -> str:
    for entry in entries:
        if DEFAULT_ALIAS in entry["aliases"]:
            return DEFAULT_ALIAS
    return entries[0]["version"]


def lay_out(root: Path, site_tar: Path, version: str, aliases: Sequence[str]) -> list[str]:
    """Write ``version`` and its aliases into a checkout of the gh-pages tree.

    ``latest`` moves only to the newest release: asked for on an older one, it
    stays where it is.

    Args:
        root: The checkout.
        site_tar: The ``//docs:build_docs`` output.
        version: ``dev`` or ``vX.Y``.
        aliases: Aliases to move to ``version``.

    Returns:
        The aliases actually moved.

    Raises:
        ValueError: An invalid alias.
    """
    versions_file = root / "versions.json"
    entries = json.loads(versions_file.read_text()) if versions_file.exists() else []
    for alias in aliases:
        _check_alias(alias, version, entries)
    moved = [a for a in aliases if a != DEFAULT_ALIAS or _newest_release(entries, version)]
    version_dir = root / version
    shutil.rmtree(version_dir, ignore_errors=True)
    version_dir.mkdir(parents=True)
    with tarfile.open(site_tar) as tar:
        tar.extractall(version_dir, filter="data")
    entries = _versions(entries, version, moved)
    for entry in entries:
        for alias in entry["aliases"]:
            _write_alias(root, alias, entry["version"])
    versions_file.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    default = _default(entries)
    (root / "index.html").write_text(REDIRECT.format(href=f"{default}/"), encoding="utf-8")
    (root / ".nojekyll").write_text("", encoding="utf-8")
    return moved


#: How many times a push rejected by a concurrent publish is retried.
PUSH_ATTEMPTS = 5


def _publish_once(
    repo: Path, site_tar: Path, version: str, aliases: Sequence[str], remote: str | None, push: bool
) -> bool | None:
    """One attempt: True when published, False when unchanged, None when the push lost a race."""
    if remote is not None:
        _git(repo, "fetch", remote, f"+{BRANCH}:refs/remotes/{remote}/{BRANCH}", check=False)
    base = f"refs/remotes/{remote}/{BRANCH}" if remote is not None else f"refs/heads/{BRANCH}"
    has_base = _git(repo, "rev-parse", "--verify", "--quiet", base, check=False).returncode == 0
    orphan = f"publish-version-{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory() as scratch:
        tree = Path(scratch) / "pages"
        try:
            if has_base:
                _git(repo, "worktree", "add", "--detach", str(tree), base)
            else:
                _git(repo, "worktree", "add", "--detach", str(tree))
                _git(tree, "checkout", "--orphan", orphan)
                _git(tree, "rm", "-rf", "--quiet", ".", check=False)
            moved = lay_out(tree, site_tar, version, aliases)
            _git(tree, "add", "--all")
            if _git(tree, "diff", "--cached", "--quiet", check=False).returncode == 0 and has_base:
                return False
            suffix = f" ({', '.join(moved)})" if moved else ""
            _git(tree, "commit", "--quiet", "-m", f"Deploy {version} docs{suffix}")
            commit = _git(tree, "rev-parse", "HEAD").stdout.strip()
            if push and remote is not None:
                pushed = _git(repo, "push", remote, f"{commit}:refs/heads/{BRANCH}", check=False)
                if pushed.returncode != 0:
                    if "rejected" in pushed.stderr or "fetch first" in pushed.stderr:
                        return None  # another publish won the race: redo on its result
                    raise subprocess.CalledProcessError(
                        pushed.returncode, pushed.args, pushed.stdout, pushed.stderr
                    )
            _git(repo, "update-ref", f"refs/heads/{BRANCH}", commit)
            return True
        finally:
            _git(repo, "worktree", "remove", "--force", str(tree), check=False)
            _git(repo, "worktree", "prune", check=False)
            _git(repo, "branch", "-D", orphan, check=False)


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

    A push rejected because another publish landed first (CI runs one per ref,
    so a tag and a master push can race) is redone on top of it, up to
    ``PUSH_ATTEMPTS`` times; the push is never forced.

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
        ValueError: A bad version name or alias.
        RuntimeError: The push lost every race.
    """
    if version != "dev" and not _RELEASE.fullmatch(version):
        raise ValueError(f"version must be dev or vX.Y, not {version!r}")
    if version in aliases:
        raise ValueError(f"{version!r} cannot alias itself")
    for _ in range(PUSH_ATTEMPTS):
        result = _publish_once(repo, site_tar, version, aliases, remote, push)
        if result is not None:
            return result
    raise RuntimeError(f"gh-pages moved under every one of {PUSH_ATTEMPTS} attempts")


def main(argv: Sequence[str] | None = None) -> int:
    """The command line: ``--repo``, ``--site``, ``--version``, ``--alias``, ``--push``.

    Returns:
        The exit code.
    """
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
