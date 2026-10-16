"""Tests for the history-link MkDocs hook."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from docs.hooks import history_link

REPO = "https://github.com/rechefe/ipu-emulator"
GENERATOR = "src/tools/ipu-as-py/src/ipu_as/gen_docs.py"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fake checkout with one hand-written page and one generator."""
    (tmp_path / "docs" / "content" / "specs").mkdir(parents=True)
    (tmp_path / "docs" / "content" / "specs" / "stage.md").write_text("# Stage\n")
    (tmp_path / "docs" / "content" / "setup.md").write_text("# Setup\n")
    (tmp_path / Path(GENERATOR).parent).mkdir(parents=True)
    (tmp_path / GENERATOR).write_text("")
    # A generated page lives outside docs/content (Bazel puts it in bazel-out).
    (tmp_path / "bazel-out" / "docs").mkdir(parents=True)
    (tmp_path / "bazel-out" / "docs" / "instructions.md").write_text("# Instructions\n")
    (tmp_path / "bazel-out" / "docs" / "unmapped.md").write_text("# Unmapped\n")
    return tmp_path


def _config(generated: dict[str, str] | None = None) -> dict:
    return {
        "repo_url": REPO,
        "docs_dir": "content",
        "extra": {"docs_git_branch": "master", "generated_pages": generated or {}},
    }


def _file(src_uri: str, abs_src: Path) -> SimpleNamespace:
    return SimpleNamespace(src_uri=src_uri, abs_src_path=str(abs_src))


def _history_url(file: SimpleNamespace, config: dict) -> str | None:
    context = history_link.on_page_context({}, SimpleNamespace(file=file), config)
    return context.get("history_url")


def test_hand_written_page_links_its_source(repo: Path) -> None:
    file = _file("specs/stage.md", repo / "docs" / "content" / "specs" / "stage.md")
    assert _history_url(file, _config()) == f"{REPO}/commits/master/docs/content/specs/stage.md"


def test_symlinked_hand_written_page_links_its_source(repo: Path, tmp_path_factory) -> None:
    # The Bazel docs build symlinks each hand-written page into a temp tree.
    temp = tmp_path_factory.mktemp("docs_temp") / "content"
    temp.mkdir()
    (temp / "setup.md").symlink_to(repo / "docs" / "content" / "setup.md")
    file = _file("setup.md", temp / "setup.md")
    assert _history_url(file, _config()) == f"{REPO}/commits/master/docs/content/setup.md"


def test_mapped_generated_page_links_its_generator(repo: Path) -> None:
    file = _file("instructions.md", repo / "bazel-out" / "docs" / "instructions.md")
    config = _config({"instructions.md": GENERATOR})
    assert _history_url(file, config) == f"{REPO}/commits/master/{GENERATOR}"


def test_unmapped_page_outside_docs_content_has_no_link(repo: Path) -> None:
    file = _file("unmapped.md", repo / "bazel-out" / "docs" / "unmapped.md")
    assert _history_url(file, _config({"instructions.md": GENERATOR})) is None


class _Files:
    def __init__(self, pages: list[SimpleNamespace]) -> None:
        self._pages = pages

    def documentation_pages(self) -> list[SimpleNamespace]:
        return self._pages


def test_map_consistent_with_files_logs_nothing(repo: Path, caplog) -> None:
    files = _Files([
        _file("setup.md", repo / "docs" / "content" / "setup.md"),
        _file("instructions.md", repo / "bazel-out" / "docs" / "instructions.md"),
    ])
    with caplog.at_level(logging.WARNING):
        history_link.on_files(files, _config({"instructions.md": GENERATOR}))
    assert caplog.records == []


def test_mapped_page_that_is_a_real_file_warns(repo: Path, caplog) -> None:
    files = _Files([_file("setup.md", repo / "docs" / "content" / "setup.md")])
    with caplog.at_level(logging.WARNING):
        history_link.on_files(files, _config({"setup.md": GENERATOR}))
    assert any("setup.md is listed in extra.generated_pages" in r.message for r in caplog.records)


def test_missing_generator_warns(repo: Path, caplog) -> None:
    files = _Files([_file("setup.md", repo / "docs" / "content" / "setup.md")])
    with caplog.at_level(logging.WARNING):
        history_link.on_files(files, _config({"instructions.md": "src/gone.py"}))
    assert any("generator src/gone.py for instructions.md" in r.message for r in caplog.records)
