"""Where a key is in a TOML file, and errors that name it (for :mod:`ipu_harness.manifest`).

TOML parsers keep no line numbers, so :class:`Locator` reads them from the
source text, and :class:`Reader` raises errors that start ``<file>:<line>:``.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from ipu_emu.errors import ConfigurationError
from ipu_isa.formats import ElementFormat

_KEY_PART = r"(?:[A-Za-z0-9_\-]+|\"[^\"]*\"|'[^']*')"
_DOTTED_KEY = re.compile(rf"\s*({_KEY_PART}(?:\s*\.\s*{_KEY_PART})*)\s*=")
_HEADER = re.compile(rf"\s*\[\[?\s*({_KEY_PART}(?:\s*\.\s*{_KEY_PART})*)\s*\]\]?\s*(#.*)?$")
_TRIPLE = re.compile(r"\"\"\"|'''")


class ManifestError(ConfigurationError):
    """A ``kernel.toml`` that breaks the schema (a configuration error, Q59); the
    message starts ``<file>:<line>:``."""


def split_key(key: str) -> tuple[str, ...]:
    """``a."b.c".d`` -> ``("a", "b.c", "d")``: dots inside quotes do not split."""
    parts = re.findall(_KEY_PART, key)
    return tuple(part.strip("\"'") for part in parts)


class Locator:
    """Line numbers of tables and keys, read from the source text."""

    def __init__(self, text: str) -> None:
        self.lines: dict[tuple[str, ...], int] = {}
        table: tuple[str, ...] = ()
        in_string = False
        for number, line in enumerate(text.splitlines(), start=1):
            starts_in_string = in_string
            in_string ^= len(_TRIPLE.findall(line)) % 2 == 1
            if starts_in_string:
                continue  # the inside of a multi-line string holds no keys
            header = _HEADER.match(line)
            if header:
                table = split_key(header.group(1))
                self.lines.setdefault(table, number)
                continue
            key = _DOTTED_KEY.match(line)
            if key:
                self.lines.setdefault((*table, *split_key(key.group(1))), number)

    def line(self, path: tuple[str, ...]) -> int:
        """The line of ``path``; else of the first key inside it; else of its nearest
        enclosing key or table; else 1."""
        if path in self.lines:
            return self.lines[path]
        inside = [n for p, n in self.lines.items() if p[: len(path)] == path]
        if path and inside:
            return min(inside)
        for end in range(len(path) - 1, 0, -1):
            if path[:end] in self.lines:
                return self.lines[path[:end]]
        return 1


def where(path: tuple[str, ...]) -> str:
    """``("cases", "default", "rows")`` -> ``[cases.default] rows``."""
    if len(path) <= 1:
        return f"[{path[0]}]" if path else "the file"
    if path[0] in ("abi", "derived", "kernel", "golden"):
        return f"[{path[0]}] {'.'.join(path[1:])}"
    return f"[{'.'.join(path[:2])}]" + (f" {'.'.join(path[2:])}" if len(path) > 2 else "")


@dataclass
class Reader:
    """Raises :class:`ManifestError` at the line of a key path."""

    toml: Path
    locator: Locator

    def fail(self, path: tuple[str, ...], message: str) -> ManifestError:
        return ManifestError(f"{self.toml}:{self.locator.line(path)}: {message}")

    def table(self, value: object, path: tuple[str, ...]) -> Mapping[str, Any]:
        if not isinstance(value, dict):
            raise self.fail(path, f"{where(path)} must be a table")
        return cast(Mapping[str, Any], value)

    def known(self, table: Mapping[str, Any], valid: Iterable[str], path: tuple[str, ...]) -> None:
        valid = tuple(valid)
        for key in table:
            if key not in valid:
                close = difflib.get_close_matches(key, valid, n=1)
                hint = f"; did you mean {close[0]!r}?" if close else f"; valid: {', '.join(valid)}"
                what = where(path) if path else "kernel.toml"
                raise self.fail((*path, key), f"unknown key {key!r} in {what}{hint}")

    def require(self, table: Mapping[str, Any], key: str, kind: type, path: tuple[str, ...]) -> Any:
        if key not in table:
            raise self.fail(path, f"{where(path)} needs {key!r}")
        return self.typed(table[key], kind, (*path, key))

    def typed(self, value: object, kind: type | tuple[type, ...], path: tuple[str, ...]) -> Any:
        """``value`` if it is of ``kind`` (a bool is never an int); finite if a float."""
        kinds = kind if isinstance(kind, tuple) else (kind,)
        if not isinstance(value, kinds) or (isinstance(value, bool) and bool not in kinds):
            names = " or ".join(k.__name__ for k in kinds)
            raise self.fail(path, f"{where(path)} must be {names}, not {value!r}")
        if isinstance(value, float) and not math.isfinite(value):
            raise self.fail(path, f"{where(path)} must be finite, not {value!r}")
        return value

    def text(self, value: object, path: tuple[str, ...]) -> str:
        """A non-empty string."""
        text = self.typed(value, str, path)
        if not text.strip():
            raise self.fail(path, f"{where(path)} must not be empty")
        return str(text)

    def format(self, value: object, path: tuple[str, ...]) -> ElementFormat:
        try:
            return ElementFormat(value)
        except ValueError:
            valid = ", ".join(f.value for f in ElementFormat)
            raise self.fail(path, f"unknown element format {value!r}; valid: {valid}") from None

    def inside(self, directory: Path, value: str, path: tuple[str, ...]) -> Path:
        """``directory/value`` for a relative file path with no ``..`` (checked lexically;
        symlinks are followed, as Bazel's runfiles are symlink trees)."""
        file = PurePosixPath(value)
        if (
            not value
            or file.is_absolute()
            or ".." in file.parts
            or value.endswith("/")
            or not file.name
        ):
            raise self.fail(
                path, f"{where(path)} {value!r} must be a file path inside the kernel directory"
            )
        return directory / file

    def function(self, directory: Path, value: object, path: tuple[str, ...]) -> str:
        """A ``"file.py:function"`` reference to a file in the kernel directory."""
        file = self._file(value, path)
        self.inside(directory, file, path)
        return str(self.typed(value, str, path))

    def shared_function(self, directory: Path, value: object, path: tuple[str, ...]) -> str:
        """A ``"file.py:function"`` in the kernel directory, or
        ``"../file.py:function"``, a file directly in the family's folder (the
        folder that holds the kernel folder, or ``registry.toml`` for an
        entry's template), which the family's kernels share: a case's
        generator, a scratch region's ``bytes``."""
        file = self._file(value, path)
        if not file.startswith("../"):
            return self.function(directory, value, path)
        shared = PurePosixPath(file[3:])
        if len(shared.parts) != 1 or shared.parts[0] == "..":
            raise self.fail(
                path, f"{where(path)} {file!r} must be ../<file> directly in the family's folder"
            )
        return str(self.typed(value, str, path))

    def _file(self, value: object, path: tuple[str, ...]) -> str:
        """The file of a ``"file.py:function"`` reference, its form checked."""
        text = str(self.typed(value, str, path))
        file, sep, fn = text.partition(":")
        if not sep or not fn.isidentifier() or not file.endswith(".py"):
            raise self.fail(path, f"{where(path)} {text!r} must be 'file.py:function'")
        return file


def decode_error_line(message: str) -> int:
    """The line a TOML syntax error names (``... (at line 3, column 7)``), else 1."""
    m = re.search(r"at line (\d+)", message)
    return int(m.group(1)) if m else 1
