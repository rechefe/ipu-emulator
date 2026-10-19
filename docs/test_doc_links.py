"""Every relative link in SKILL.md and CLAUDE.md points at an existing file, and
every link from one docs page to another names a page the site builds.

`mkdocs build` never reads the two root files, and agents follow their links into
the repository. Between docs pages it only warns on a dangling link, so a moved
page could leave one behind in a green build; the docs-page test fails instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Root files whose relative links are checked.
LINKED_FILES = ("SKILL.md", "CLAUDE.md")

# The MkDocs docs_dir and its config (whose extra.generated_pages lists the pages
# Bazel writes into the site, which are not checked in under docs/content/).
DOCS_DIR = Path("docs/content")
MKDOCS_YML = Path("docs/mkdocs.yml")

_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_EXTERNAL = ("http://", "https://", "mailto:", "#")


def broken_links(path: Path) -> list[str]:
    """`path:line: target` for each relative link in `path` whose target does not exist."""
    problems = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for target in _LINK.findall(line):
            if target.startswith(_EXTERNAL):
                continue
            file_part = target.split("#", 1)[0]
            if not (path.parent / file_part).exists():
                problems.append(f"{path}:{number}: {target}")
    return problems


def generated_pages(mkdocs_yml: Path) -> set[str]:
    """The docs_dir-relative paths listed under extra.generated_pages in `mkdocs_yml`."""
    pages: set[str] = set()
    inside = False
    for line in mkdocs_yml.read_text(encoding="utf-8").splitlines():
        if line.strip() == "generated_pages:":
            inside = True
            continue
        if inside:
            match = re.match(r"^\s{4}([\w./-]+\.md):", line)
            if not match:
                break
            pages.add(match.group(1))
    return pages


def broken_page_links(docs_dir: Path, generated: set[str]) -> list[str]:
    """`path:line: target` for each link between docs pages that names no page of the site."""
    root = docs_dir.resolve()
    problems = []
    for path in sorted(docs_dir.rglob("*.md")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for target in _LINK.findall(line):
                if target.startswith(_EXTERNAL):
                    continue
                file_part = target.split("#", 1)[0]
                if not file_part.endswith(".md"):
                    continue
                resolved = (path.parent / file_part).resolve()
                if resolved.exists():
                    continue
                if (
                    resolved.is_relative_to(root)
                    and resolved.relative_to(root).as_posix() in generated
                ):
                    continue
                problems.append(f"{path}:{number}: {target}")
    return problems


@pytest.mark.parametrize("name", LINKED_FILES)
def test_relative_links_resolve(name: str) -> None:
    assert broken_links(Path(name)) == []


def test_links_between_docs_pages_resolve() -> None:
    assert broken_page_links(DOCS_DIR, generated_pages(MKDOCS_YML)) == []


def test_generated_pages_are_read() -> None:
    # Guards the parse: the ISA reference pages are generated.
    assert "reference/isa/instructions.md" in generated_pages(MKDOCS_YML)


def test_links_are_found() -> None:
    # Guards the scan itself: SKILL.md links docs pages.
    text = Path("SKILL.md").read_text(encoding="utf-8")
    assert len(_LINK.findall(text)) > 10


def test_renamed_target_is_reported(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "page.md").write_text("# Page\n")
    index = tmp_path / "INDEX.md"
    index.write_text(
        "See [page](docs/page.md#top), [site](https://example.com) and [here](#rules).\n"
        "Also [gone](docs/old-name.md).\n"
    )
    assert broken_links(index) == [f"{index}:2: docs/old-name.md"]


def test_moved_docs_page_is_reported(tmp_path: Path) -> None:
    docs = tmp_path / "content"
    (docs / "explanation").mkdir(parents=True)
    (docs / "explanation" / "other.md").write_text("# Other\n")
    page = docs / "explanation" / "page.md"
    page.write_text(
        "See [other](other.md#top), [ref](../reference/isa/instructions.md) and [code](../../x.py).\n"
        "Also [moved](../instructions.md).\n"
    )
    assert broken_page_links(docs, {"reference/isa/instructions.md"}) == [
        f"{page}:2: ../instructions.md"
    ]
