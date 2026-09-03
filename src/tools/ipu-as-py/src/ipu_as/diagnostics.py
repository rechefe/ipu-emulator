#!/usr/bin/env python3
"""Report assembly errors as positioned diagnostics, for editor squiggles.

A TextMate grammar assigns scopes to spans; it has no way to say "this is
wrong", and no notion of what the grammar expects next. Validity is not a
lexical property — whether a comma is legal depends on the parser's state, not
on the characters. So the only thing that can answer "is this valid" is the
parser itself, which is what this module runs.

Three stages are checked, in the order the assembler runs them, stopping at the
first failure (a template that will not render cannot be parsed, and a program
that will not parse cannot be encoded):

1. Jinja rendering   -> TemplateSyntaxError
2. Parsing           -> lark.exceptions.LarkError
3. Encoding          -> ValueError from CompoundInst

Positions are 0-based, matching LSP and the VS Code API.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace

import jinja2
import lark

import ipu_as.compound_inst as compound_inst
import ipu_as.label as ipu_label
from ipu_as.lark_tree import ASTBuilder, get_parser

#: Assembly-stage errors embed their position in the message text (see
#: AnnotatedToken.get_location_string). Line/column are 1-based there.
_LOCATION_RE = re.compile(r"Line (\d+), Column (\d+)")

#: Some encode errors name the offending token in quotes but carry no position
#: (e.g. "Opcode 'mac.ee' not found"); this recovers the name so it can be
#: located in the source.
_QUOTED_NAME_RE = re.compile(r"'([^']+)'")

#: Jinja is a preprocessing layer, so a template marker means the parser sees
#: different text than the editor shows. See _instrument.
_JINJA_MARKERS = ("{{", "{%", "{#")


@dataclass
class Diagnostic:
    line: int
    column: int
    end_line: int
    end_column: int
    message: str
    severity: str
    stage: str
    #: True when the position could not be pinned down exactly — either the
    #: error carried none, or Jinja rendering moved it. Callers should say so
    #: rather than imply byte accuracy.
    approximate: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _offset_to_linecol(text: str, offset: int) -> tuple[int, int]:
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


def _instrument(source: str) -> str | None:
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
                out.append(f" {_LINE_MARKER}{line}\n")
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


def _linecol_to_offset(text: str, line: int, column: int) -> int | None:
    """Byte offset for a 0-based (line, column), or None if out of range."""
    lines = text.splitlines(keepends=True)
    if line >= len(lines):
        return None
    return sum(len(l) for l in lines[:line]) + column


def _source_line_at(rendered: str, offset: int) -> int | None:
    """0-based source line for an offset in the instrumented render.

    The first marker at or after the offset is the line the offending text came
    from: markers sit at end of line, so anything before one belongs to it.
    """
    match = _MARKER_RE.search(rendered, offset)
    if not match:
        return None
    return max(int(match.group(1)) - 1, 0)


def _source_line(raw: str, line: int) -> str:
    lines = raw.splitlines()
    return lines[line] if 0 <= line < len(lines) else ""


def _rendered_line(rendered: str, line: int) -> str:
    lines = rendered.splitlines()
    return lines[line] if 0 <= line < len(lines) else ""


def _token_at(text: str, offset: int) -> str:
    """The identifier-ish run of characters starting at an offset."""
    match = re.compile(r"[+-]?[A-Za-z0-9_][A-Za-z0-9_.]*").match(text, offset)
    return match.group(0) if match else ""


def _column_in_source(raw: str, line: int, needle: str) -> int:
    """Column of `needle` on a source line, or 0 if it is not found there."""
    lines = raw.splitlines()
    if line >= len(lines) or not needle:
        return 0
    found = lines[line].find(needle)
    return found if found >= 0 else 0


def _with_expected(found: str, expected) -> str:
    if not expected:
        return found
    return f"{found}; expected one of: {_humanize(expected)}"


#: Jinja's whitespace-control dashes. `-#}` deletes the whitespace *after* the
#: tag, including the newline, which can fuse the last token of one line with
#: the first of the next (`cr4` + `MULT.RC.VE` -> `cr4MULT.RC.VE`). An origin
#: marker placed between them would prevent that fusion — a different program —
#: so markers get rejected and the position is lost.
_WS_CONTROL_RE = re.compile(r"(\{[%#])-|-([%#]\})")


def _disable_whitespace_control(source: str) -> str:
    """Strip the `-` from Jinja's whitespace-control tags."""
    return _WS_CONTROL_RE.sub(lambda m: m.group(1) or m.group(2), source)


def _recover_line_without_fusion(text: str) -> tuple[str, int] | None:
    """Recover an error's source line by rendering without whitespace control.

    The fused render is what the assembler really sees, so it is what gets
    diagnosed — but it is exactly the render markers cannot survive. Disabling
    whitespace control keeps the lines apart, so the markers do survive and the
    line is recoverable.

    That variant is a slightly different program, so only its *line* is used,
    and only when it fails at the same stage. Returns (stage, 0-based line).
    """
    instrumented = _instrument(_disable_whitespace_control(text))
    if instrumented is None:
        return None
    try:
        rendered = jinja2.Template(instrumented).render()
    except Exception:
        return None

    ipu_label.reset_labels()
    try:
        tree = get_parser().parse(rendered)
    except lark.exceptions.LarkError as error:
        token = getattr(error, "token", None)
        offset = (
            token.start_pos
            if token is not None and getattr(token, "start_pos", None) is not None
            else getattr(error, "pos_in_stream", None)
        )
        line = _source_line_at(rendered, offset) if offset is not None else None
        return ("parse", line) if line is not None else None

    try:
        for instruction in ASTBuilder().transform(tree):
            compound_inst.CompoundInst(instruction).encode()
    except (lark.exceptions.VisitError, ValueError) as error:
        original = getattr(error, "orig_exc", error)
        location = _LOCATION_RE.search(str(original))
        if not location:
            return None
        offset = _linecol_to_offset(
            rendered,
            max(int(location.group(1)) - 1, 0),
            max(int(location.group(2)) - 1, 0),
        )
        line = _source_line_at(rendered, offset) if offset is not None else None
        return ("assemble", line) if line is not None else None

    return None


def _squeeze(text: str) -> str:
    """Text with all whitespace removed; whitespace is only a separator here."""
    return re.sub(r"\s+", "", text)


def _same_tokens(a: str, b: str) -> bool:
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
        return _squeeze(_MARKER_RE.sub("", a)) == _squeeze(b)


def _humanize(terminal_names) -> str:
    """Render a set of expected terminal names as something a human reads.

    The parser reports internal names like `_SEMI2`. Literal terminals carry
    their text, so show that; the rest fall back to a lowercased name.
    """
    by_name = {t.name: t for t in get_parser().terminals}
    shown = []
    for name in sorted(terminal_names):
        terminal = by_name.get(name)
        value = getattr(getattr(terminal, "pattern", None), "value", None)
        if terminal is not None and type(terminal.pattern).__name__ == "PatternStr":
            shown.append(f"'{value}'")
        else:
            shown.append(name.lstrip("_").lower())
    return ", ".join(shown)


def _template_runtime_line(error: BaseException) -> int | None:
    """0-based template line for a Jinja *runtime* error, from its traceback.

    Unlike a syntax error, a runtime failure carries no `.lineno` — but jinja2
    rewrites the traceback so the frame executing the template reports the
    template's own line. Take the innermost such frame.
    """
    line = None
    tb = error.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename in ("<template>", "<unknown>"):
            line = tb.tb_lineno
        tb = tb.tb_next
    return max(line - 1, 0) if line else None


def _template_diagnostic(error: jinja2.TemplateSyntaxError) -> Diagnostic:
    # Jinja reports against the source template, so this is already exact.
    line = max((error.lineno or 1) - 1, 0)
    return Diagnostic(
        line=line,
        column=0,
        end_line=line,
        end_column=0,
        message=f"Template error: {error.message}",
        severity="error",
        stage="template",
    )


def _parse_diagnostic(error: lark.exceptions.LarkError, raw: str, rendered: str):
    token = getattr(error, "token", None)
    if token is not None and getattr(token, "start_pos", None) is not None:
        start = token.start_pos
        # The synthetic `$END` token carries a start but no end; so can a token
        # at EOF. Without this an empty program — which a Jinja template can
        # easily render to — crashes the checker instead of reporting.
        end = token.end_pos if token.end_pos is not None else start + 1
        expected = getattr(error, "expected", set())
        found = f"unexpected {str(token)!r}"
    else:
        start = getattr(error, "pos_in_stream", 0) or 0
        end = start + 1
        expected = getattr(error, "allowed", set()) or set()
        found = f"unexpected {rendered[start:end]!r}"

    line, column = _offset_to_linecol(rendered, start)
    end_line, end_column = _offset_to_linecol(rendered, end)
    needle = rendered[start:end]

    if raw == rendered:
        # No Jinja: rendered offsets are source offsets.
        mapped, exact = line, True
    else:
        marked = _source_line_at(rendered, start)
        if marked is None:
            # Without a marker there is nothing honest to report but the top of
            # the file. Guessing by matching line content used to fill this gap
            # and was wrong often enough to be worse than saying so.
            return Diagnostic(
                0, 0, 0, 0, _with_expected(found, expected),
                "error", "parse", approximate=True,
            )
        mapped, exact = marked, True
        column = _column_in_source(raw, marked, needle)
        end_line, end_column = mapped, column + max(len(needle), 1)

    message = _with_expected(found, expected)

    return Diagnostic(
        line=mapped,
        column=column if exact else 0,
        end_line=end_line,
        end_column=end_column if exact else 0,
        message=message,
        severity="error",
        stage="parse",
        approximate=not exact,
    )


def _tidy(message: str) -> str:
    """Flatten a multi-line assembler message and drop its trailing location.

    The assembler appends "At: Line 1, Column 1" for humans reading a terminal.
    A diagnostic already carries the position, and repeating it in the hover
    text is noise.
    """
    # Case-insensitive: the assembler writes both "At: Line 4, Column 5" and
    # "... , in Line 11, Column 5." mid-sentence. The trailing period goes too,
    # or removing the clause leaves a dangling " ." in the hover.
    without_location = re.sub(
        r"[\s,]*\b(?:at|in)\b:?\s*Line (?:\d+|None), Column (?:\d+|None)\.?",
        "",
        message,
        flags=re.IGNORECASE,
    )
    return " ".join(without_location.split())


def _encode_diagnostic(error: ValueError, raw: str, rendered: str) -> Diagnostic:
    message = str(error)
    # Initialized up front so every path below has a defined result. An earlier
    # version left these unbound when a Jinja file had no usable marker, which
    # raised UnboundLocalError out of the checker — the CLI then printed a
    # traceback instead of JSON and the editor showed no error at all.
    mapped, column, exact = 0, 0, False

    location = _LOCATION_RE.search(message)
    quoted = _QUOTED_NAME_RE.search(message)

    # Locate the offending text in the rendered program. Most encode errors
    # carry "Line N, Column M"; those that do not ("Opcode 'mac.ee' not found")
    # name the token in quotes instead.
    offset = None
    if location:
        offset = _linecol_to_offset(
            rendered,
            max(int(location.group(1)) - 1, 0),
            max(int(location.group(2)) - 1, 0),
        )
    if offset is None and quoted:
        found = rendered.find(quoted.group(1))
        offset = found if found >= 0 else None

    if offset is not None:
        marked = _source_line_at(rendered, offset)
        if marked is not None:
            # The line comes from the marker. The column cannot: it indexes the
            # rendered line, which Jinja may have rewritten, so find the token
            # in the source instead of carrying a number that points elsewhere.
            mapped, exact = marked, True
            needle = quoted.group(1) if quoted else _token_at(rendered, offset)
            column = _column_in_source(raw, mapped, needle)
        elif raw == rendered:
            mapped, column = _offset_to_linecol(rendered, offset)
            exact = True

    return Diagnostic(
        line=mapped,
        column=column if exact else 0,
        end_line=mapped,
        end_column=(column + 1) if exact else 0,
        message=_tidy(message),
        severity="error",
        stage="assemble",
        approximate=not exact,
    )


def _recover(diagnostic: Diagnostic, text: str) -> Diagnostic:
    """Last resort: give an approximate diagnostic a real line if one exists.

    Only whitespace-control fusion lands here (see
    :func:`_recover_line_without_fusion`); everything else already has an exact
    position from the origin markers.
    """
    if not diagnostic.approximate:
        return diagnostic

    recovered = _recover_line_without_fusion(text)
    if recovered is None or recovered[0] != diagnostic.stage:
        return diagnostic

    line = recovered[1]
    # The line is real; the column is not recoverable, since the offending text
    # spans the join between two source lines.
    return replace(
        diagnostic, line=line, column=0, end_line=line, end_column=0,
        approximate=False,
    )


def check(text: str) -> list[Diagnostic]:
    """Return diagnostics for one assembly source. Empty means it assembles."""
    rendered = text
    if any(marker in text for marker in _JINJA_MARKERS):
        try:
            rendered = jinja2.Template(text).render()
        except jinja2.TemplateSyntaxError as error:
            return [_template_diagnostic(error)]
        except Exception as error:
            # Rendering runs arbitrary template expressions, so a failure is not
            # necessarily a TemplateError: `{{ 1 + "a" }}` raises TypeError and
            # `{% for i in undefined_var %}` raises UndefinedError. Letting
            # either escape kills the CLI with a traceback and makes the editor
            # report a broken toolchain instead of a broken template.
            line = _template_runtime_line(error)
            return [
                Diagnostic(
                    line or 0, 0, line or 0, 0,
                    f"Template error: {type(error).__name__}: {error}",
                    "error", "template",
                    approximate=line is None,
                )
            ]

        # Render again with origin markers, so a diagnostic can name the source
        # line exactly instead of guessing. Only used if it renders identically
        # modulo the markers; anything else falls back to the plain render.
        instrumented = _instrument(text)
        if instrumented is not None:
            try:
                marked = jinja2.Template(instrumented).render()
            except jinja2.TemplateError:
                marked = None
            if marked is not None and _same_tokens(marked, rendered):
                rendered = marked

    # The label registry is module-global; a stale entry from a previous check
    # would surface as a bogus "label defined twice".
    ipu_label.reset_labels()

    try:
        tree = get_parser().parse(rendered)
    except lark.exceptions.LarkError as error:
        return [_recover(_parse_diagnostic(error, text, rendered), text)]

    try:
        ast = ASTBuilder().transform(tree)
    except lark.exceptions.VisitError as error:
        # lark wraps anything a transformer raises in VisitError, which is
        # itself a LarkError — so without unwrapping, an encode error carrying
        # an exact location gets misreported as a parse error at offset 0.
        original = error.orig_exc
        if isinstance(original, ValueError):
            return [_recover(_encode_diagnostic(original, text, rendered), text)]
        raise

    try:
        for instruction in ast:
            compound_inst.CompoundInst(instruction).encode()
    except ValueError as error:
        return [_recover(_encode_diagnostic(error, text, rendered), text)]

    return []
