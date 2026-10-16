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
2. Parsing           -> ParseError
3. Encoding          -> EncodeError (LabelError, OperandError, SlotError)

Positions are 0-based, matching LSP and the VS Code API.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jinja2

import ipu_as.template as template
from ipu_as.errors import EncodeError, ParseError, TemplateError
from ipu_as.session import AssemblySession
from ipu_as.source_map import (
    MAIN,
    column_in_source,
    disable_whitespace_control,
    include_note,
    instrument,
    instrument_template,
    linecol_to_offset,
    offset_to_linecol,
    same_tokens,
    source_line_at,
    token_at,
)
from ipu_as.syntax import get_parser

#: Jinja is a preprocessing layer, so a template marker means the parser sees
#: different text than the editor shows. See instrument.
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


def _with_expected(found: str, expected) -> str:
    if not expected:
        return found
    return f"{found}; expected one of: {_humanize(expected)}"


def _recover_line_without_fusion(
    text: str, inputs: template.TemplateInputs
) -> tuple[str, int] | None:
    """Recover an error's source line by rendering without whitespace control.

    The fused render is what the assembler really sees, so it is what gets
    diagnosed — but it is exactly the render markers cannot survive. Disabling
    whitespace control keeps the lines apart, so the markers do survive and the
    line is recoverable.

    That variant is a slightly different program, so only its *line* is used,
    and only when it fails at the same stage. Returns (stage, 0-based line).
    """
    variant = disable_whitespace_control(text)
    if instrument(variant) is None:
        return None
    try:
        rendered = template.render(variant, inputs, source_name=MAIN, transform=instrument_template)
    except TemplateError:
        return None
    # A fresh session: this variant's labels must not meet the real check's.
    stages = AssemblySession(variant)
    try:
        stages.encode(stages.parse_rendered(rendered))
    except ParseError as error:
        offset = _parse_offset(error)
        line = source_line_at(rendered, offset) if offset is not None else None
        return ("parse", line) if line is not None else None
    except EncodeError as error:
        offset = _location_offset(error, rendered)
        line = source_line_at(rendered, offset) if offset is not None else None
        return ("assemble", line) if line is not None else None
    return None


def _parse_offset(error: ParseError) -> int | None:
    """Offset in the rendered text where the parser stopped (from the lark exception)."""
    cause = error.__cause__
    token = getattr(cause, "token", None)
    if token is not None and getattr(token, "start_pos", None) is not None:
        return token.start_pos
    return getattr(cause, "pos_in_stream", None)


def _location_offset(error: EncodeError, rendered: str) -> int | None:
    """Offset in the rendered text of an error's location (1-based line and column)."""
    location = error.location
    # A synthetic token (an instruction the assembler made up) has line 0: no position.
    if location is None or location.line < 1 or location.column < 1:
        return None
    return linecol_to_offset(rendered, location.line - 1, location.column - 1)


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


def _parse_diagnostic(parse_error: ParseError, raw: str, rendered: str) -> Diagnostic:
    # The lark exception carries the offending token and the expected terminals.
    error = parse_error.__cause__
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

    line, column = offset_to_linecol(rendered, start)
    end_line, end_column = offset_to_linecol(rendered, end)
    needle = rendered[start:end]

    if raw == rendered:
        # No Jinja: rendered offsets are source offsets.
        mapped, exact = line, True
    else:
        marked = source_line_at(rendered, start)
        if marked is None:
            # Without a marker there is nothing honest to report but the top of
            # the file. Guessing by matching line content used to fill this gap
            # and was wrong often enough to be worse than saying so.
            return Diagnostic(
                0,
                0,
                0,
                0,
                _with_expected(found, expected),
                "error",
                "parse",
                approximate=True,
            )
        mapped, exact = marked, True
        column = column_in_source(raw, marked, needle)
        end_line, end_column = mapped, column + max(len(needle), 1)

    message = _with_expected(found, expected)
    if raw != rendered:
        message += include_note(rendered, start)

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


def _encode_diagnostic(error: EncodeError, raw: str, rendered: str) -> Diagnostic:
    """The diagnostic for an encode error, placed by its location and token."""
    mapped, column, exact = 0, 0, False
    note = ""
    offset = _location_offset(error, rendered)
    if offset is not None:
        note = include_note(rendered, offset) if raw != rendered else ""
        marked = source_line_at(rendered, offset)
        if marked is not None:
            # The line comes from the marker. The column cannot: it indexes the
            # rendered line, which Jinja may have rewritten, so find the token
            # in the source instead of carrying a number that points elsewhere.
            mapped, exact = marked, True
            column = column_in_source(raw, mapped, error.token or token_at(rendered, offset))
        elif raw == rendered:
            mapped, column = offset_to_linecol(rendered, offset)
            exact = True

    return Diagnostic(
        line=mapped,
        column=column if exact else 0,
        end_line=mapped,
        end_column=(column + 1) if exact else 0,
        message=_tidy(str(error)) + note,
        severity="error",
        stage="assemble",
        approximate=not exact,
    )


def _recover(diagnostic: Diagnostic, text: str, inputs: template.TemplateInputs) -> Diagnostic:
    """Last resort: give an approximate diagnostic a real line if one exists.

    Only whitespace-control fusion lands here (see
    :func:`_recover_line_without_fusion`); everything else already has an exact
    position from the origin markers.
    """
    if not diagnostic.approximate:
        return diagnostic

    recovered = _recover_line_without_fusion(text, inputs)
    if recovered is None or recovered[0] != diagnostic.stage:
        return diagnostic

    line = recovered[1]
    # The line is real; the column is not recoverable, since the offending text
    # spans the join between two source lines.
    return replace(
        diagnostic,
        line=line,
        column=0,
        end_line=line,
        end_column=0,
        approximate=False,
    )


def _template_failure(wrapped: TemplateError) -> Diagnostic:
    """The diagnostic for a failed render, from the jinja2 exception it wraps."""
    error = wrapped.__cause__
    location = wrapped.location
    note = ""
    if location is not None and location.source != MAIN:
        note = f" (in {location.source}:{location.line})"
    if error is None:
        # A bad define or ipu:params header: the error itself carries the line.
        line = max(location.line - 1, 0) if location is not None else 0
        return Diagnostic(
            line,
            0,
            line,
            0,
            f"Template error: {wrapped}",
            "error",
            "template",
            approximate=location is None,
        )
    if isinstance(error, jinja2.TemplateSyntaxError) and note == "":
        return _template_diagnostic(error)
    # Rendering runs arbitrary template expressions, so a failure is not
    # necessarily a jinja2 TemplateError: `{{ 1 + "a" }}` raises TypeError and
    # `{% for i in undefined_var %}` raises UndefinedError. Letting either escape
    # kills the CLI with a traceback and makes the editor report a broken
    # toolchain instead of a broken template.
    line = _template_runtime_line(error)
    message = (
        f"Template error: {error.message}"
        if isinstance(error, jinja2.TemplateSyntaxError)
        else f"Template error: {type(error).__name__}: {error}"
    )
    return Diagnostic(
        line or 0,
        0,
        line or 0,
        0,
        message + note,
        "error",
        "template",
        approximate=line is None,
    )


def _render_for_diagnosis(
    text: str, inputs: template.TemplateInputs
) -> tuple[str, Diagnostic | None]:
    """(The text to diagnose, or the template-stage diagnostic if rendering fails).

    The text is the instrumented render when it lexes like the plain one, so
    each position maps back to its source line; otherwise the plain render.
    """
    if not any(marker in text for marker in _JINJA_MARKERS):
        return text, None
    # Sandboxed like the assembler: the editor checks a file on open, so this must not run a
    # received .asm's payload (see ipu_as/template.py). An escape surfaces as a SecurityError.
    try:
        rendered = template.render(text, inputs, source_name=MAIN)
    except TemplateError as wrapped:
        return text, _template_failure(wrapped)
    if instrument(text) is None:
        return rendered, None
    try:
        marked = template.render(text, inputs, source_name=MAIN, transform=instrument_template)
    except TemplateError:
        return rendered, None
    return (marked if same_tokens(marked, rendered) else rendered), None


def check(
    text: str,
    *,
    defines: Mapping[str, object] | None = None,
    search_path: Sequence[Path] = (),
    abi: Mapping[str, Mapping[str, str]] | None = None,
) -> list[Diagnostic]:
    """Return diagnostics for one assembly source. Empty means it assembles.

    It runs the stages of an `AssemblySession` and reports the first
    `AssemblerError` from its location and token. Any other exception is a bug
    and propagates (the CLI reports it as an `internal` diagnostic).

    Args:
        text: The source, as the editor shows it.
        defines: Template values, as for the assembler.
        search_path: Roots `{% include %}` and `{% import %}` resolve in. An
            error inside an included template is reported at the open file's
            include line, and its message names the template and line.
        abi: The `cr` / `lr` names the template reads.
    """
    inputs = template.TemplateInputs(
        defines=dict(defines or {}),
        search_path=tuple(Path(p) for p in search_path),
        namespaces=abi or {},
    )
    rendered, failed = _render_for_diagnosis(text, inputs)
    if failed is not None:
        return [failed]
    stages = AssemblySession(text)
    try:
        stages.encode(stages.parse_rendered(rendered))
    except ParseError as error:
        return [_recover(_parse_diagnostic(error, text, rendered), text, inputs)]
    except EncodeError as error:
        return [_recover(_encode_diagnostic(error, text, rendered), text, inputs)]
    return []
