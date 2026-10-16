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
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^`]*)$")
_MARKER = re.compile(r"^[ \t]*<!--[ \t]*(?P<lang>[\w+-]+)-fragment(?P<rest>.*?)-->[ \t]*$")


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


def _strip_indent(line: str, indent: int) -> str:
    """Remove up to `indent` leading whitespace characters."""
    stripped = len(line) - len(line.lstrip(" \t"))
    return line[min(indent, stripped) :]


def _orphan_marker(path: str | Path, line: int, lang: str) -> DocBlockError:
    return DocBlockError(
        f"{path}:{line}: {lang}-fragment marker is not followed by a ```{lang} fence"
    )


def fenced_blocks(path: str | Path, lang: str) -> list[Block]:
    """Return every fenced block in `path` whose info string starts with `lang`.

    Fences inside list items are handled by stripping the fence's indentation
    from the body. Raises DocBlockError for a fence with no closing line, a
    `<lang>-fragment` marker without a reason, or a marker that is not followed
    by a `lang` fence.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    blocks: list[Block] = []
    pending_marker: tuple[int, str] | None = None  # (line, reason)
    i = 0
    while i < len(lines):
        line = lines[i]
        marker = _MARKER.match(line)
        if marker and marker.group("lang") == lang:
            if pending_marker is not None:
                raise _orphan_marker(path, pending_marker[0], lang)
            rest = marker.group("rest").strip()
            reason = rest[1:].strip() if rest.startswith(":") else ""
            if not reason:
                raise DocBlockError(
                    f"{path}:{i + 1}: {lang}-fragment marker needs a reason "
                    f"(<!-- {lang}-fragment: why this block is not checked -->)"
                )
            pending_marker = (i + 1, reason)
            i += 1
            continue

        fence = _FENCE.match(line)
        if not fence:
            if pending_marker is not None:
                raise _orphan_marker(path, pending_marker[0], lang)
            i += 1
            continue

        open_line = i + 1
        indent = len(fence.group("indent"))
        char = fence.group("fence")[0]
        length = len(fence.group("fence"))
        info = fence.group("info").strip()
        block_lang = info.split()[0] if info else ""
        closing = re.compile(rf"^[ \t]*{re.escape(char)}{{{length},}}[ \t]*$")
        body: list[str] = []
        i += 1
        while i < len(lines) and not closing.match(lines[i]):
            body.append(_strip_indent(lines[i], indent))
            i += 1
        if i == len(lines):
            raise DocBlockError(f"{path}:{open_line}: fence has no closing line")
        i += 1  # past the closing fence

        if block_lang == lang:
            reason = pending_marker[1] if pending_marker else None
            blocks.append(Block(str(path), open_line, "\n".join(body) + "\n", reason))
        elif pending_marker is not None:
            raise _orphan_marker(path, pending_marker[0], lang)
        pending_marker = None

    if pending_marker is not None:
        raise _orphan_marker(path, pending_marker[0], lang)
    return blocks
