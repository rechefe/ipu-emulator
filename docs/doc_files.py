"""The hand-written docs that the doc tests check."""

from __future__ import annotations

from pathlib import Path

# Root files checked besides docs/content/**/*.md. Generated pages are not
# inputs; their generators' own tests check them.
ROOT_FILES = ("SKILL.md", "CLAUDE.md", "README.md")


def doc_files() -> list[Path]:
    """Every checked Markdown file, relative to the repository root.

    Bazel runs a test from the root of its runfiles tree, which mirrors the
    repository and holds only the test's declared data.
    """
    content = sorted(Path("docs/content").rglob("*.md"))
    return content + [Path(name) for name in ROOT_FILES]
