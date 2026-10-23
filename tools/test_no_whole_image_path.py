"""The whole-image kernel path stays deleted (E06-20, Q83).

The old harness let a kernel's case hand over a complete, pre-built XMEM image
(the names in ``FORBIDDEN``). Every kernel now declares its inputs as tensors
in ``kernel.toml`` and the harness places and packs each region. This test
fails on any of those names in a checked-in file outside the history:
``planning/`` and the ADRs (``docs/adr/``), which record the decision.

It walks the checkout, not the runfiles (the data file below is a symlink into
it), so it is tagged ``external`` and not cached.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

#: Spelled in pieces so this file does not match itself.
FORBIDDEN = tuple(
    "".join(parts) for parts in (("Memory", "App"), ("memory", "_spec"), ("Memory", "Layout"))
)

#: Repo-relative directories that are history, generated or vendored.
SKIPPED_DIRS = frozenset(
    {
        ".git",
        "planning",
        "docs/adr",
        "node_modules",
        "__pycache__",
        ".venv",
        "playground",
    }
)

#: Files read as text; anything else (binaries, images) is skipped.
TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".toml",
        ".md",
        ".bzl",
        ".bazel",
        ".asm",
        ".j2",
        ".sh",
        ".js",
        ".json",
        ".yml",
        ".yaml",
        ".tex",
        ".txt",
        ".cfg",
        ".ini",
    }
)


def checkout_root() -> Path:
    """The checkout this test's data file links into."""
    return Path("ruff.toml").resolve().parent


def text_files(root: Path) -> Iterator[Path]:
    """Every checked-in-looking text file under *root*, skipped dirs left out."""
    for directory, subdirs, files in os.walk(root):
        rel = Path(directory).relative_to(root)
        subdirs[:] = [
            name
            for name in subdirs
            if not name.startswith("bazel-")
            and name not in SKIPPED_DIRS
            and (rel / name).as_posix() not in SKIPPED_DIRS
        ]
        for name in files:
            path = Path(directory) / name
            if path.suffix in TEXT_SUFFIXES or name in {"BUILD", "MODULE.bazel"}:
                yield path


def forbidden_uses(root: Path) -> list[str]:
    """``path:line: name`` for each use of a whole-image name under *root*."""
    found = []
    for path in text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for name in FORBIDDEN:
                if name in line:
                    found.append(f"{path.relative_to(root)}:{number}: {name}")
    return found


def test_no_whole_image_names_outside_the_history():
    root = checkout_root()
    assert (root / "MODULE.bazel").is_file(), root
    assert forbidden_uses(root) == []


def test_a_planted_name_is_found(tmp_path):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "app.py").write_text(f"class X({FORBIDDEN[0]}):\n    pass\n")
    (tmp_path / "planning").mkdir()
    (tmp_path / "planning" / "note.md").write_text(FORBIDDEN[1])
    (tmp_path / "docs" / "adr").mkdir(parents=True)
    (tmp_path / "docs" / "adr" / "0001.md").write_text(FORBIDDEN[2])
    assert forbidden_uses(tmp_path) == [f"kernels/app.py:1: {FORBIDDEN[0]}"]
