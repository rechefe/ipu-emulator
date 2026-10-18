"""Where each token of a render is in the source it was rendered from.

Jinja renders a source into other text, so a token of the render has no
position of its own in the source. A token's line comes from the origin
markers of `ipu_as.source_map`, and its column from finding its text on that
line. A token that is not literally in the source came from a `{{ }}`
expression (or an include), and is not placed: its Jinja region covers it.

    rendering = Rendering(Source(text), rendered)
    rendering.place(tokens)[tokens[0].start_pos]  # [(line, column, length)]

Positions are 0-based; a column counts code points (Python string indices).
`ipu_as.language_service` places semantic tokens and words with it.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterator, Sequence
from functools import cached_property

import lark

import ipu_as.template as template
from ipu_as.jinja_tags import TagSpan, block_keyword, tag_spans
from ipu_as.source_map import source_line_at, strip_markers

__all__ = ["COMMENTS", "Piece", "Rendering", "Source"]

#: The grammar's comment terminals.
COMMENTS = ("COMMENT_HASH", "COMMENT_LINE")

#: Where a token is on one source line: (line, column, length).
Piece = tuple[int, int, int]

#: Block tags that render nothing (`set x = ...`, imports) or render text in its place
#: (`include`, `raw`): a source with no other tag renders each stretch once, in order.
_IN_PLACE = frozenset({"set", "import", "from", "include", "raw", "endraw"})


class Source:
    """One source: its lines, and its Jinja regions (`ipu_as.jinja_tags.tag_spans`).

    Args:
        text: The source, as the editor shows it.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        #: Without Jinja the source is its own render, and positions carry over.
        self.literal = not template.has_markers(text)

    @cached_property
    def line_starts(self) -> list[int]:
        """The offset of each line's first character."""
        return [0] + [m.end() for m in re.finditer("\n", self.text)]

    def line_end(self, line: int) -> int:
        """The offset of the end of `line`, before its line break (`\\n` or `\\r\\n`)."""
        starts = self.line_starts
        if line + 1 == len(starts):
            return len(self.text)
        end = starts[line + 1] - 1
        return end - 1 if end > starts[line] and self.text[end - 1] == "\r" else end

    def position(self, offset: int) -> tuple[int, int]:
        """The (line, column) of an offset."""
        line = bisect.bisect_right(self.line_starts, offset) - 1
        return line, offset - self.line_starts[line]

    def pieces(self, start: int, end: int) -> Iterator[Piece]:
        """[start, end) split at line breaks, as non-empty (line, column, length)."""
        while start < end:
            line, column = self.position(start)
            stop = min(end, self.line_end(line))
            if stop > start:
                yield line, column, stop - start
            start = self.line_starts[line + 1] if line + 1 < len(self.line_starts) else end

    @cached_property
    def spans(self) -> tuple[TagSpan, ...]:
        """The Jinja regions, in order."""
        return () if self.literal else tag_spans(self.text)

    @cached_property
    def _span_starts(self) -> list[int]:
        return [span.start for span in self.spans]

    @cached_property
    def _span_ends(self) -> list[int]:
        return [span.end for span in self.spans]

    def overlaps(self, start: int, end: int) -> bool:
        """True when [start, end) touches a Jinja region."""
        i = bisect.bisect_right(self._span_ends, start)
        return i < len(self.spans) and self.spans[i].start < end

    def outside_spans(self, start: int, end: int) -> Iterator[tuple[int, int]]:
        """The parts of [start, end) that are not in a Jinja region."""
        for span in self.spans[bisect.bisect_right(self._span_ends, start) :]:
            if span.start >= end:
                break
            if span.start > start:
                yield start, span.start
            start = max(start, span.end)
        if start < end:
            yield start, end

    def next_span_start(self, offset: int) -> int:
        """The start of the first Jinja region at or after `offset`; the end if none."""
        i = bisect.bisect_left(self._span_starts, offset)
        return self.spans[i].start if i < len(self.spans) else len(self.text)

    def loop_starts(self, start: int, end: int) -> Iterator[int]:
        """Where the body of each `{% for %}` tag in [start, end) begins, the last first."""
        first = bisect.bisect_left(self._span_starts, start)
        for span in reversed(self.spans[first : bisect.bisect_left(self._span_starts, end)]):
            tag = self.text[span.start : span.end]
            if span.end <= end and span.kind == "block" and block_keyword(tag) == "for":
                yield span.end

    @cached_property
    def renders_in_order(self) -> bool:
        """True when each stretch of the source outside Jinja renders once, in order.

        So it is when no tag but those in `_IN_PLACE` is in it, and no `\\r`
        (Jinja reads each line break as `\\n`, so offsets would not carry over).
        """
        blocks = (self.text[span.start : span.end] for span in self.spans if span.kind == "block")
        return "\r" not in self.text and all(block_keyword(tag) in _IN_PLACE for tag in blocks)


class Rendering:
    """One render of a source, and where its tokens are in that source.

    Args:
        source: The source.
        rendered: Its render (`ipu_as.diagnostics.render_for_diagnosis`): the
            instrumented one, or the plain one when the markers would change
            how it lexes.
    """

    def __init__(self, source: Source, rendered: str) -> None:
        self.source = source
        self.rendered = rendered

    @cached_property
    def by_line(self) -> bool:
        """True when the render says which source line each of its tokens came from.

        The instrumented render does: by the origin marker after the token, or,
        after the last marker, the last line (which ends without a line break,
        so has none). So does a one-line source, whatever its render.
        """
        return (
            self.source.literal
            or len(self.source.line_starts) == 1
            or source_line_at(self.rendered, 0) is not None
        )

    def source_line(self, token: lark.Token) -> int | None:
        """The source line a token of the render came from.

        Args:
            token: A token the grammar's lexer read from the render.

        Returns:
            The line, or None when the render does not say (`by_line`).
        """
        if self.source.literal:
            return token.line - 1 if token.line is not None else None
        if not self.by_line:
            return None
        line = source_line_at(self.rendered, token.start_pos or 0)
        last = len(self.source.line_starts) - 1
        if line is None:
            return last
        return line if line <= last else None

    @cached_property
    def _stretches(self) -> tuple[list[int], list[tuple[int, int, int]]]:
        """For a plain render of several lines: where literal source text is in it.

        A source that `renders_in_order` renders to itself with each Jinja
        region replaced by its output, the whitespace beside a tag perhaps
        stripped. So each stretch of source outside Jinja, without its outer
        whitespace, is in the render, in order; one that is there exactly once
        after the last found is found for certain (an expression's output may
        hold the same text, so one that is there twice is not used).

        Returns:
            (rendered starts, [(rendered start, source start, length)]),
            sorted; both empty when the source does not render in order.
        """
        source, rendered = self.source, self.rendered
        if self.by_line or not source.renders_in_order:
            return [], []
        found: list[tuple[int, int, int]] = []
        cursor = 0
        for start, end in source.outside_spans(0, len(source.text)):
            core = source.text[start:end].strip()
            at = rendered.find(core, cursor) if core else -1
            if at >= 0 and rendered.find(core, at + 1) < 0:
                found.append((at, source.text.index(core, start), len(core)))
                cursor = at + len(core)
        return [at for at, _, _ in found], found

    def source_offset(self, begin: int, end: int) -> int | None:
        """Where [begin, end) of a plain render is in the source, when it is there literally.

        Args:
            begin: An offset in the render.
            end: The offset just past the text that must be literal.

        Returns:
            The source offset of `begin`; None when the text is not known to
            be literal source text, or the render says lines (`by_line`).
        """
        starts, stretches = self._stretches
        i = bisect.bisect_right(starts, begin) - 1
        if i < 0:
            return None
        at, offset, length = stretches[i]
        return offset + begin - at if end <= at + length else None

    def place(self, tokens: Sequence[lark.Token]) -> dict[int, list[Piece]]:
        """Where each token of the render is in the source.

        Args:
            tokens: The grammar's tokens of the render, in order.

        Returns:
            Rendered offset -> (line, column, length) pieces; a token not in
            the source has none, a comment around Jinja has several.
        """
        locator = _Locator(self)
        return {token.start_pos or 0: locator.place(token) for token in tokens}


#: A character that continues a TOKEN.
_WORD_CHAR = re.compile(r"[A-Za-z0-9_.]")


def _joins(left: str, right: str) -> bool:
    """True when two adjacent characters would lex as one token."""
    return bool(_WORD_CHAR.match(left) and _WORD_CHAR.match(right)) or left == right == ";"


class _Locator:
    """Places the tokens of one render in its source, in order.

    A token's line comes from the render (`Rendering.source_line`); its column
    from finding its text on that line, after the last token placed there from
    the same rendered line. A plain render of several lines says no line, and
    its tokens are placed by the literal source text around them
    (`Rendering.source_offset`).
    """

    def __init__(self, rendering: Rendering) -> None:
        self.rendering = rendering
        self.source = rendering.source
        self.cursors: dict[int, int] = {}
        self.rendered_line: int | None = None

    def place(self, token: lark.Token) -> list[Piece]:
        """Where `token` is in the source, as (line, column, length) pieces."""
        if not self.rendering.by_line:
            return self._in_order(token)
        if token.line != self.rendered_line:
            # A new rendered line, perhaps the next pass of a loop over the same source line.
            self.cursors.clear()
            self.rendered_line = token.line
        line = self.rendering.source_line(token)
        if line is None:
            return []
        if token.type in COMMENTS:
            return self._comment(line, strip_markers(str(token)).rstrip())
        return self._code(line, str(token))

    def _in_order(self, token: lark.Token) -> list[Piece]:
        """A token of a plain render, where the literal source text around it puts it."""
        begin = token.start_pos or 0
        comment = token.type in COMMENTS
        # A comment runs on into whatever its line renders; its opener must be literal.
        at = self.rendering.source_offset(begin, begin + (1 if comment else len(str(token))))
        if at is None:
            return []
        line, column = self.source.position(at)
        if comment:
            return self._comment_from(at, self.source.line_end(line))
        return [(line, column, len(str(token)))]

    def _code(self, line: int, value: str) -> list[Piece]:
        first, end = self.source.line_starts[line], self.source.line_end(line)
        cursor = self.cursors.get(line, first)
        at = self._find(value, cursor, end)
        # Not after the last token placed on this line: perhaps the next pass of a loop
        # on the line, which starts again where the loop's body does.
        for start in self.source.loop_starts(first, cursor) if at < 0 else ():
            at = self._find(value, start, cursor)
            if at >= 0:
                break
        if at < 0:
            return []
        self.cursors[line] = at + len(value)
        return [(line, at - first, len(value))]

    def _find(self, value: str, start: int, end: int) -> int:
        """The first whole `value` in [start, end) outside Jinja; -1 if there is none."""
        text = self.source.text
        at = text.find(value, start, end)
        while at >= 0 and not self._alone(at, at + len(value)):
            at = text.find(value, at + 1, end)
        return at

    def _alone(self, start: int, end: int) -> bool:
        """True when [start, end) is a whole token of the source, outside Jinja."""
        text = self.source.text
        before = text[start - 1] if start > 0 else " "
        after = text[end] if end < len(text) else " "
        if _joins(before, text[start]) or _joins(text[end - 1], after):
            return False
        return not self.source.overlaps(start, end)

    def _comment(self, line: int, value: str) -> list[Piece]:
        """A comment runs to the end of its source line, around the Jinja inside it."""
        if not value:
            return []  # only an origin marker
        source, text = self.source, self.source.text
        end = source.line_end(line)
        opener = "//" if value.startswith("//") else "#"
        at = text.find(opener, self.cursors.get(line, source.line_starts[line]), end)
        while at >= 0:
            fragment = text[at : min(source.next_span_start(at), end)].rstrip()
            if not source.overlaps(at, at + 1) and value.startswith(fragment):
                self.cursors[line] = end
                return self._comment_from(at, end)
            at = text.find(opener, at + 1, end)
        return []

    def _comment_from(self, at: int, end: int) -> list[Piece]:
        """The comment that starts at `at` and runs to `end`, its line's end, around its Jinja."""
        source = self.source
        stop = at + len(source.text[at:end].rstrip())
        return [
            piece
            for start, finish in source.outside_spans(at, stop)
            for piece in source.pieces(start, finish)
        ]
