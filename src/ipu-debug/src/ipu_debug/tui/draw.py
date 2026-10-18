"""What every pane draws with: glyphs, styles and a clipping line canvas.

The view and the pane modules share these, so a pane module renders without
the view: it is handed a :class:`Theme` once and a :class:`Canvas` per
render, and returns the canvas's :class:`~ipu_debug.tui.pane.Line` list.
"""

from __future__ import annotations

import locale
from collections.abc import Callable
from dataclasses import dataclass

from ipu_debug.tui.pane import Line, Rect
from ipu_debug.tui.screen import Screen, Style, overlay_style

# One content column on the right of a vertically scrollable pane is always
# reserved for its scroll thumb, so the layout never reflows when the thumb
# appears or disappears.
SCROLLBAR_COLUMNS = 1
# Content starts this far inside a pane's left border and stops one column
# short of the right one, so no value is ever painted against a frame.
PANE_CONTENT_INDENT = 2
PANE_CONTENT_MARGIN = 1
# Border columns kept between a pane title and a readout sharing its row.
TITLE_READOUT_GAP = 2
# A value made only of these characters is zero-padded rather than signed or
# fractional, so a leading run of zeros in it carries no information.
_PADDED_VALUE_CHARACTERS = frozenset("0123456789abcdef")
# Narrow values such as a single hex byte are left alone: splitting ``01`` into
# a dim and a bright half is more noise than help.
MIN_PADDED_VALUE_WIDTH = 4
BITS_PER_REGISTER = 32


@dataclass(frozen=True)
class Glyphs:
    """The drawing characters for one terminal capability level."""

    box: tuple[str, ...]
    thumb: str
    separator: str
    current_row: str
    normal_row: str
    more_before: str
    more_after: str
    ellipsis: str


UNICODE_GLYPHS = Glyphs(
    box=(
        " ",
        "│",
        "─",
        "└",
        "│",
        "│",
        "┌",
        "├",
        "─",
        "┘",
        "─",
        "┴",
        "┐",
        "┤",
        "┬",
        "┼",
    ),
    thumb="█",
    separator="·",
    current_row="▶",
    normal_row=" ",
    more_before="‹",
    more_after="›",
    ellipsis="…",
)
ASCII_GLYPHS = Glyphs(
    box=(
        " ",
        "|",
        "-",
        "+",
        "|",
        "|",
        "+",
        "+",
        "-",
        "+",
        "-",
        "+",
        "+",
        "+",
        "+",
        "+",
    ),
    thumb="#",
    separator="|",
    current_row=">",
    normal_row=" ",
    more_before="<",
    more_after=">",
    ellipsis="...",
)


def detect_glyphs() -> Glyphs:
    """Use box-drawing characters only when the terminal encoding accepts them."""
    encoding = locale.getpreferredencoding(False) or "ascii"
    try:
        "".join(UNICODE_GLYPHS.box).encode(encoding)
        (UNICODE_GLYPHS.thumb + UNICODE_GLYPHS.separator).encode(encoding)
        (UNICODE_GLYPHS.current_row + UNICODE_GLYPHS.ellipsis).encode(encoding)
        (UNICODE_GLYPHS.more_before + UNICODE_GLYPHS.more_after).encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return ASCII_GLYPHS
    return UNICODE_GLYPHS


class Palette:
    """The style of each kind of text, with monochrome fallbacks.

    Without a *screen* (or on one without colours) every style is plain
    emphasis, which is what the pure pane tests render with.
    """

    def __init__(self, screen: Screen | None = None) -> None:
        self._screen = screen
        self.enabled = screen is not None and screen.has_colors
        bold = Style.BOLD
        self.current = bold
        # Emphasis is layered, so no two of these may be the same attribute:
        # ``changed`` marks a value, ``selected`` the cursor under it, and
        # ``focus`` the frame of the pane they are in.
        self.changed = bold | Style.UNDERLINE
        self.selected = int(Style.REVERSE)
        # Every pane shows where its cursor is; only the focused one shows it
        # at full strength.
        self.cursor = Style.REVERSE | Style.DIM
        self.header = int(bold)
        self.paused = bold | Style.REVERSE
        self.breakpoint = int(self.changed)
        self.pc_marker = int(bold)
        self.key = int(bold)
        self.border = int(Style.DIM)
        self.plain = int(bold)
        self.title = int(bold)
        self.focus = int(bold)
        self.status = int(Style.REVERSE)
        self.error = int(bold)
        self.cell_selected = int(Style.UNDERLINE)
        self.cell_changed = int(Style.ITALIC)
        if screen is None or not self.enabled:
            return
        self.header = 0  # Respect the terminal's light or dark theme.
        self.paused = screen.color(0, 3)
        self.breakpoint = screen.color(9, -1) | bold
        self.pc_marker = screen.color(10, -1) | bold
        self.key = screen.color(14, -1) | bold
        self.border = screen.color(8, -1)
        self.title = screen.color(14, -1) | bold
        self.focus = screen.color(14, -1) | bold
        self.status = 0
        self.error = screen.color(15, 1) | bold
        self.selected = screen.color(0, 6)
        # A dark foreground would be brightened into the background by the
        # bold attribute rows carry, so the muted cursor keeps a light
        # foreground and only drops the accent colour.
        self.cursor = screen.color(15, 8)

    def cell(self, fg: int, bg: int) -> int:
        """A 16-colour text cell; bright foregrounds are bold."""
        if not self.enabled or self._screen is None:
            return 0
        style = self._screen.color(fg, bg)
        if fg >= 8:
            style |= Style.BOLD
        return style

    def cursor_for(self, focused: bool) -> int:
        """The cursor highlight, muted while its pane is not focused."""
        return self.selected if focused else self.cursor

    def value(self, is_zero: bool) -> int:
        """Zero values recede, so the bytes that carry data stand out."""
        return self.border if is_zero else 0


@dataclass
class Theme:
    """The palette and glyphs one view draws with, shared with its panes."""

    colors: Palette
    glyphs: Glyphs


def format_bits(value: int) -> str:
    """A 32-bit value as its binary digits."""
    return f"{value:0{BITS_PER_REGISTER}b}"


def fit_segments(segments: list[str], width: int, glyphs: Glyphs) -> str:
    """Join as many whole segments as fit; never cut one in half."""
    separator = f" {glyphs.separator} "
    text = segments[0] if segments else ""
    for segment in segments[1:]:
        candidate = f"{text}{separator}{segment}"
        if len(candidate) > width:
            break
        text = candidate
    if len(text) > width:
        text = text[: max(0, width - len(glyphs.ellipsis))]
        text += glyphs.ellipsis
    return text


class Canvas:
    """Text runs clipped to *clip*, recorded as lines or sent to *emit*."""

    def __init__(
        self,
        theme: Theme,
        clip: Rect,
        emit: Callable[[int, int, str, int], None] | None = None,
    ) -> None:
        self.theme = theme
        self.clip = clip
        self.lines: list[Line] = []
        self._emit = emit

    @property
    def colors(self) -> Palette:
        return self.theme.colors

    @property
    def glyphs(self) -> Glyphs:
        return self.theme.glyphs

    def add(self, y: int, x: int, text: str, attr: int = 0, max_width: int | None = None) -> None:
        clip = self.clip
        right = clip.x + clip.width
        if not (clip.y <= y < clip.y + clip.height and clip.x <= x < right) or not text:
            return
        width = right - x if max_width is None else min(max_width, right - x)
        if width <= 0:
            return
        if self._emit is not None:
            self._emit(y, x, text[:width], attr)
        else:
            self.lines.append(Line(y, x, text[:width], attr))

    def add_value(self, y: int, x: int, text: str, attr: int, max_width: int) -> None:
        """Draw a numeric value with its leading zeros dimmed.

        Only zero-padded fixed-width values qualify, so a float's ``0.5`` or a
        negative decimal keeps every character at full weight.
        """
        significant = text.lstrip("0")
        if (
            significant
            and len(text) >= MIN_PADDED_VALUE_WIDTH
            and text.startswith("0")
            and all(character in _PADDED_VALUE_CHARACTERS for character in text)
        ):
            padding = len(text) - len(significant)
            self.add(y, x, text[:padding], overlay_style(self.colors.border, attr), max_width)
            self.add(y, x + padding, significant, attr, max_width - padding)
            return
        self.add(y, x, text, attr, max_width)

    def fit_segments(self, segments: list[str], width: int) -> str:
        return fit_segments(segments, width, self.glyphs)

    def title(
        self,
        rect: Rect,
        segments: list[str],
        *,
        reserved: int = 0,
        focused: bool = False,
    ) -> None:
        """Draw a pane title on its top border, dropping segments that do not fit.

        ``reserved`` keeps the right end of the border free for a position
        readout that shares the row.
        """
        width = max(0, rect.width - 6 - reserved)
        title = self.fit_segments(segments, width)
        if not title:
            return
        attr = self.colors.title if focused else self.colors.plain
        self.add(rect.y, rect.x + 2, f" {title} ", attr, width + 2)

    def position(self, rect: Rect, text: str, *, top: bool = False) -> bool:
        """Right-align a scroll-position readout on one of a pane's borders.

        The top panes share their bottom border with the panes below them, so
        they report on their own top border, beside the title, instead.
        """
        if not text or len(text) > rect.width - 8:
            return False
        self.add(
            rect.y if top else rect.y + rect.height - 1,
            max(rect.x + 1, rect.x + rect.width - 2 - len(text)),
            text,
            self.colors.border,
            max(0, rect.width - 2),
        )
        return True

    def top_pane_header(
        self,
        rect: Rect,
        segments: list[str],
        readout: str,
        *,
        focused: bool = False,
    ) -> None:
        """Title and position readout for a pane whose bottom border is shared."""
        # The title owns the border: the readout appears only when sharing the
        # row costs the title nothing.
        width = max(0, rect.width - 6)
        shared = width - len(readout) - TITLE_READOUT_GAP
        reserved = 0
        if readout and self.fit_segments(segments, shared) == self.fit_segments(segments, width):
            if self.position(rect, readout, top=True):
                reserved = len(readout) + TITLE_READOUT_GAP
        self.title(rect, segments, reserved=reserved, focused=focused)
