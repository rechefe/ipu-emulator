"""Map positions in rendered assembly back to the source the editor shows.

Jinja renders a template into different text, so a position in the rendered
program need not exist in the file. `instrument` tags every source line with
an origin marker before rendering (`#@ipu-src-line:<N>` in the open file,
`#@ipu-src:<template>:<N>` in an included one), and Jinja carries the markers
through whatever it does; `source_line_at` reads the line back. Positions here
are 0-based, matching LSP and the VS Code API.
"""

from __future__ import annotations

import re

import lark

from ipu_as.syntax import get_parser


def offset_to_linecol(text: str, offset: int) -> tuple[int, int]:
    """0-based (line, column) for a byte offset."""
    prefix = text[:offset]
    line = prefix.count("\n")
    column = offset - (prefix.rfind("\n") + 1)
    return line, column


#: Appended to each template line so the rendered output carries where it came
#: from. `#` starts a comment in this grammar, so a marker is invisible to the
#: parser — it changes the text Lark sees only by adding comments.
_LINE_MARKER = "#@ipu-src-line:"


_MARKER_RE = re.compile(re.escape(_LINE_MARKER) + r"(\d+)")


#: The marker in an included template: `#@ipu-src:<template name>:<line>`.
_INCLUDE_PREFIX = "#@ipu-src:"


_INCLUDE_MARKER_RE = re.compile(re.escape(_INCLUDE_PREFIX) + r"(?P<name>[^\s:]+):(?P<line>\d+)")


_ANY_MARKER_RE = re.compile(f"{_MARKER_RE.pattern}|{_INCLUDE_MARKER_RE.pattern}")


#: The open file's name in template.render, so the transform tells it from an include.
MAIN = "<asm>"


def instrument_template(name: str, text: str) -> str:
    """template.render transform: tag each line of the open file or of an included template."""
    prefix = _LINE_MARKER if name == MAIN else f"{_INCLUDE_PREFIX}{name}:"
    return instrument(text, prefix) or text


def include_note(rendered: str, offset: int) -> str:
    """ " (in <template>:<line>)" when the text at `offset` came from an included template."""
    match = _ANY_MARKER_RE.search(rendered, offset)
    if match is None or match.group("name") is None:
        return ""
    return f" (in {match.group('name')}:{match.group('line')})"


def strip_markers(text: str) -> str:
    """`text` up to its first origin marker, without the whitespace before it.

    A marker is appended to a line's comment when the line has one, so the
    rendered comment token carries it.
    """
    match = _ANY_MARKER_RE.search(text)
    return text if match is None else text[: match.start()].rstrip()


def instrument(source: str, prefix: str = _LINE_MARKER) -> str | None:
    """Tag every template line with its own line number.

    Jinja renders a template into different text — loops repeat lines,
    conditionals drop them, whitespace control joins them — so a byte offset in
    the rendered output has no general relationship to the file on screen.
    Rather than guess afterwards by matching line content, mark each line before
    rendering and let Jinja carry the marks through whatever it does.

    A marker is only safe at a point where no Jinja construct is open: appending
    to the middle of a multi-line ``{% ... %}`` would be a syntax error. Lines
    inside a construct are left unmarked, and the marker that follows the
    construct still bounds them.

    Returns None if the template cannot be scanned coherently.
    """
    openers = {"{%": "%}", "{{": "}}", "{#": "#}"}
    out: list[str] = []
    closer: str | None = None
    line = 1
    i = 0

    while i < len(source):
        pair = source[i : i + 2]
        if closer is None:
            if pair in openers:
                closer = openers[pair]
                out.append(pair)
                i += 2
                continue
            if source[i] == "\n":
                out.append(f" {prefix}{line}\n")
                line += 1
                i += 1
                continue
        else:
            if pair == closer:
                closer = None
                out.append(pair)
                i += 2
                continue
            if source[i] == "\n":
                line += 1
        out.append(source[i])
        i += 1

    if closer is not None:
        return None  # unterminated construct; let Jinja report it normally
    return "".join(out)


def linecol_to_offset(text: str, line: int, column: int) -> int | None:
    """Byte offset for a 0-based (line, column), or None if out of range."""
    lines = text.splitlines(keepends=True)
    if line >= len(lines):
        return None
    return sum(len(source_line) for source_line in lines[:line]) + column


def source_line_at(rendered: str, offset: int) -> int | None:
    """0-based source line for an offset in the instrumented render.

    The first marker at or after the offset is the line the offending text came
    from: markers sit at end of line, so anything before one belongs to it.
    """
    match = _MARKER_RE.search(rendered, offset)
    if not match:
        return None
    return max(int(match.group(1)) - 1, 0)


def token_at(text: str, offset: int) -> str:
    """The identifier-ish run of characters starting at an offset."""
    match = re.compile(r"[+-]?[A-Za-z0-9_][A-Za-z0-9_.]*").match(text, offset)
    return match.group(0) if match else ""


def column_in_source(raw: str, line: int, needle: str) -> int:
    """Column of `needle` on a source line, or 0 if it is not found there."""
    lines = raw.splitlines()
    if line >= len(lines) or not needle:
        return 0
    found = lines[line].find(needle)
    return found if found >= 0 else 0


#: Jinja's whitespace-control dashes. `-#}` deletes the whitespace *after* the
#: tag, including the newline, which can fuse the last token of one line with
#: the first of the next (`cr4` + `MULT.RC.VE` -> `cr4MULT.RC.VE`). An origin
#: marker placed between them would prevent that fusion — a different program —
#: so markers get rejected and the position is lost.
_WS_CONTROL_RE = re.compile(r"(\{[%#])-|-([%#]\})")


def disable_whitespace_control(source: str) -> str:
    """Strip the `-` from Jinja's whitespace-control tags."""
    return _WS_CONTROL_RE.sub(lambda m: m.group(1) or m.group(2), source)


def _squeeze(text: str) -> str:
    """Text with all whitespace removed; whitespace is only a separator here."""
    return re.sub(r"\s+", "", text)


def same_tokens(a: str, b: str) -> bool:
    """True when two renders lex to the same tokens.

    Comparing text would reject a valid instrumentation: whitespace control
    (`-%}`, `-#}`) strips whitespace *after* a tag, so a marker legitimately
    changes the whitespace — the plain render joins `;;` and `SET` into `;;SET`
    where the marked one does not. The lexer ignores whitespace and comments,
    so its token stream is the equivalence that actually matters.
    """
    parser = get_parser()
    try:
        return [(t.type, str(t)) for t in parser.lex(a)] == [
            (t.type, str(t)) for t in parser.lex(b)
        ]
    except lark.exceptions.LarkError:
        # The text being compared does not lex — which is the normal case here,
        # since a diagnostic is only being produced because something is wrong.
        # Fall back to comparing with whitespace removed entirely: whitespace
        # control legitimately differs between the two renders (`-#}` joins `;;`
        # and `SET` into `;;SET` without a marker in the way), and whitespace is
        # only ever a separator in this grammar. Comparing `.split()` here threw
        # the markers away for every erroring file — exactly when they matter.
        return _squeeze(_ANY_MARKER_RE.sub("", a)) == _squeeze(b)
