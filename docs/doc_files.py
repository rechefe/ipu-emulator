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


def generated_files() -> list[Path]:
    """The generated ISA reference pages a test was given as data, under docs/.

    Only ``test_doc_asm_blocks`` takes them (from ``:generate_mkdocs_md`` and
    ``:generate_instruction_format_md``); other tests see an empty list.
    """
    return sorted(path for path in Path("docs").glob("*.md") if path.name != "README.md")
