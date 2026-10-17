"""Extract fenced code blocks from hand-written Markdown docs.

Used by the doc tests (`test_doc_asm_blocks`, `test_doc_commands`) to check
that every example in the docs still works.

A block that is deliberately incomplete carries a fragment marker, an HTML
comment on the line just before its fence:

    <!-- asm-fragment: shows only the inner loop -->
    ```asm
    ...
    ```

The reason after the colon is required.

A kernel's program, which renders only with that kernel's values, carries a
kernel marker instead, naming the kernel folder (from the repository root):

    <!-- asm-kernel: kernels/elementwise/scale -->
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^`]*)$")
_MARKER = re.compile(
    r"^[ \t]*<!--[ \t]*(?P<lang>[\w+-]+)-(?P<kind>fragment|kernel)(?P<rest>.*?)-->[ \t]*$"
)


class DocBlockError(ValueError):
    """The Markdown is malformed: an unclosed fence or a bad fragment marker."""


@dataclass(frozen=True)
class Block:
    path: str
    #: 1-based line of the opening fence.
    line: int
    body: str
    #: The reason from the fragment marker, or None when the block is not a fragment.
    fragment_reason: str | None
    #: The kernel folder from a kernel marker, or None.
    kernel: str | None = None


def _strip_indent(line: str, indent: int) -> str:
    """Remove up to `indent` leading whitespace characters."""
    stripped = len(line) - len(line.lstrip(" \t"))
    return line[min(indent, stripped) :]


def _orphan_marker(path: str | Path, line: int, lang: str) -> DocBlockError:
    return DocBlockError(
        f"{path}:{line}: {lang}-fragment marker is not followed by a ```{lang} fence"
    )


_MARKER_EXAMPLE = {
    "fragment": "why this block is not checked",
    "kernel": "kernels/<family>/<name>",
}


def _marker_value(path: str | Path, line: int, marker: re.Match, lang: str) -> str:
    """The text after the colon of a marker; DocBlockError if it has none."""
    rest = marker.group("rest").strip()
    value = rest[1:].strip() if rest.startswith(":") else ""
    if not value:
        kind = marker.group("kind")
        what = "a reason" if kind == "fragment" else "a kernel folder"
        raise DocBlockError(
            f"{path}:{line}: {lang}-{kind} marker needs {what} "
            f"(<!-- {lang}-{kind}: {_MARKER_EXAMPLE[kind]} -->)"
        )
    return value


def _read_fence(
    path: str | Path, lines: list[str], start: int, fence: re.Match
) -> tuple[str, list[str], int]:
    """(language, body lines, index after the closing fence) for the fence at `start`."""
    indent = len(fence.group("indent"))
    char = fence.group("fence")[0]
    length = len(fence.group("fence"))
    info = fence.group("info").strip()
    closing = re.compile(rf"^[ \t]*{re.escape(char)}{{{length},}}[ \t]*$")
    body: list[str] = []
    i = start + 1
    while i < len(lines) and not closing.match(lines[i]):
        body.append(_strip_indent(lines[i], indent))
        i += 1
    if i == len(lines):
        raise DocBlockError(f"{path}:{start + 1}: fence has no closing line")
    return (info.split()[0] if info else ""), body, i + 1


def fenced_blocks(path: str | Path, lang: str) -> list[Block]:
    """Return every fenced block in `path` whose info string starts with `lang`.

    Fences inside list items are handled by stripping the fence's indentation
    from the body. Raises DocBlockError for a fence with no closing line, a
    `<lang>-fragment` marker without a reason, or a marker that is not followed
    by a `lang` fence.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    blocks: list[Block] = []
    pending_marker: tuple[int, str, str] | None = None  # (line, kind, value)
    i = 0
    while i < len(lines):
        marker = _MARKER.match(lines[i])
        fence = _FENCE.match(lines[i])
        if pending_marker is not None and not fence:
            raise _orphan_marker(path, pending_marker[0], lang)
        if marker and marker.group("lang") == lang:
            value = _marker_value(path, i + 1, marker, lang)
            pending_marker = (i + 1, marker.group("kind"), value)
            i += 1
            continue
        if not fence:
            i += 1
            continue
        block_lang, body, next_i = _read_fence(path, lines, i, fence)
        if block_lang == lang:
            kind, value = pending_marker[1:] if pending_marker else (None, None)
            reason = value if kind == "fragment" else None
            kernel = value if kind == "kernel" else None
            blocks.append(Block(str(path), i + 1, "\n".join(body) + "\n", reason, kernel))
        elif pending_marker is not None:
            raise _orphan_marker(path, pending_marker[0], lang)
        pending_marker = None
        i = next_i

    if pending_marker is not None:
        raise _orphan_marker(path, pending_marker[0], lang)
    return blocks
