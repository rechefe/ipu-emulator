"""Every relative link in SKILL.md and CLAUDE.md points at an existing file.

`mkdocs build` checks links between docs pages, but it never reads these two
root files, and agents follow their links into the repository.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Root files whose relative links are checked.
LINKED_FILES = ("SKILL.md", "CLAUDE.md")

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


@pytest.mark.parametrize("name", LINKED_FILES)
def test_relative_links_resolve(name: str) -> None:
    assert broken_links(Path(name)) == []


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
