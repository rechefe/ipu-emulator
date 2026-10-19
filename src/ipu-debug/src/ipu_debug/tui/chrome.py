"""What the view draws around the panes: frames, header, tab bars, footer, thumbs.

Every function here draws on a :class:`~ipu_debug.tui.draw.Canvas` and records
what it drew that a pointer can act on in the frame's :class:`Hits`, so a
click is resolved against what is on screen without re-deriving geometry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ipu_debug.tui.draw import SCROLLBAR_COLUMNS, Canvas, Palette
from ipu_debug.tui.keymap import KEY_BINDINGS, footer_actions, footer_text
from ipu_debug.tui.layout import FOOTER_ROWS, HEADER_ROWS
from ipu_debug.tui.pane import Rect
from ipu_debug.tui.tabs import TabSet

# Border cells are accumulated as a bit mask of the line segments that leave
# them, so two panes sharing an edge resolve to one junction character instead
# of overwriting each other.
BORDER_UP = 1
BORDER_RIGHT = 2
BORDER_DOWN = 4
BORDER_LEFT = 8
BORDER_VERTICAL = BORDER_UP | BORDER_DOWN
MAX_TAB_LABEL_WIDTH = 24
HELP_LABEL = "[? Help]"
PAUSED_BADGE = "PAUSED"
# Columns of the header kept clear of the summary: the help button and margins.
HEADER_RESERVED_COLUMNS = 12


@dataclass
class Hits:
    """Where the last frame put everything a pointer can act on.

    ``values`` maps a pane to each drawn value's rectangle and the payload
    its ``place_cursor`` takes; ``scroll`` is each scrollable pane's
    ``(first, visible, total)`` rows; ``actions`` and ``overlay`` map a
    rectangle to the key a click on it presses.
    """

    tabs: dict[str, list[tuple[Rect, int]]] = field(default_factory=dict)
    tab_closes: dict[str, list[tuple[Rect, int]]] = field(default_factory=dict)
    tab_scrolls: list[tuple[Rect, str, int]] = field(default_factory=list)
    adds: dict[str, Rect] = field(default_factory=dict)
    values: dict[str, list[tuple[Rect, Any]]] = field(default_factory=dict)
    scroll: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    scrollbars: dict[str, Rect] = field(default_factory=dict)
    thumbs: dict[str, Rect] = field(default_factory=dict)
    actions: list[tuple[Rect, Any]] = field(default_factory=list)
    overlay: list[tuple[Rect, Any]] = field(default_factory=list)
    editor_field: tuple[Rect, int] | None = None


class Frames:
    """Pane frames, recorded first and painted once as one grid of junctions."""

    def __init__(self) -> None:
        # ``(y, x) -> (segment mask, style, priority)``.
        self.cells: dict[tuple[int, int], tuple[int, int, int]] = {}

    def box(self, rect: Rect, colors: Palette, *, focused: bool = False) -> None:
        """Record one frame; a focused frame owns the style of the edges it shares."""
        if rect.height < 2 or rect.width < 2:
            return
        priority = 1 if focused else 0
        style = colors.focus if focused else colors.border
        right = rect.x + rect.width - 1
        bottom = rect.y + rect.height - 1
        for x in range(rect.x, right + 1):
            mask = (BORDER_LEFT if x > rect.x else 0) | (BORDER_RIGHT if x < right else 0)
            for y in (rect.y, bottom):
                self._mark(y, x, mask, style, priority)
        for y in range(rect.y, bottom + 1):
            mask = (BORDER_UP if y > rect.y else 0) | (BORDER_DOWN if y < bottom else 0)
            for x in (rect.x, right):
                self._mark(y, x, mask, style, priority)

    def _mark(self, y: int, x: int, mask: int, style: int, priority: int) -> None:
        current_mask, current_style, current_priority = self.cells.get((y, x), (0, style, -1))
        if priority >= current_priority:
            current_style = style
            current_priority = priority
        self.cells[(y, x)] = (current_mask | mask, current_style, current_priority)

    def flush(self, canvas: Canvas) -> None:
        """Paint every recorded border cell, then start a fresh frame."""
        for (y, x), (mask, style, _) in self.cells.items():
            canvas.add(y, x, canvas.glyphs.box[mask], style, 1)
        self.cells.clear()


def draw_header(
    canvas: Canvas, rect: Rect, segments: list[str], status: tuple[str, bool], hits: Hits
) -> None:
    """The header row: the run summary, the status message and the help button.

    *status* is ``(message, is_error)``; the message ``Ready`` shows nothing.
    """
    colors, glyphs = canvas.colors, canvas.glyphs
    text, error = status
    message = "" if text == "Ready" else text
    if error:
        message = "! " + message
    message_width = min(len(message), max(0, (rect.width - HEADER_RESERVED_COLUMNS) // 2))
    if len(message) > message_width:
        message = message[: max(0, message_width - 1)] + glyphs.ellipsis
    width = max(0, rect.width - HEADER_RESERVED_COLUMNS - (len(message) + 2 if message else 0))
    summary = canvas.fit_segments(segments[:4], width)
    separator = f" {glyphs.separator} "
    remaining = width - len(summary) - len(separator)
    if remaining > 0:
        summary += separator + canvas.fit_segments(segments[4:], remaining)
    line = " " + summary
    canvas.add(rect.y, rect.x, line.ljust(rect.width), colors.header, rect.width)
    badge_x = line.find(PAUSED_BADGE)
    if badge_x >= 0:
        canvas.add(rect.y, rect.x + badge_x, PAUSED_BADGE, colors.paused, len(PAUSED_BADGE))
    x = rect.x + rect.width - len(HELP_LABEL) - 1
    if message:
        style = colors.error if error else colors.status
        canvas.add(rect.y, x - len(message) - 2, message, style, len(message))
    canvas.add(rect.y, x, HELP_LABEL, colors.key, len(HELP_LABEL))
    hits.actions.append((Rect(rect.y, x, 1, len(HELP_LABEL)), "?"))


def _tab_label(canvas: Canvas, tab: Any, rect: Rect) -> str:
    """``[title x]``, the title cut so even a long tab fits the narrowest pane."""
    title = tab.title
    title_width = min(MAX_TAB_LABEL_WIDTH, rect.width - 8) - len("[ x]")
    if len(title) > title_width:
        ellipsis = canvas.glyphs.ellipsis
        title = title[: title_width - len(ellipsis)] + ellipsis
    return f"[{title} x]"


def _scroll_tab_bar(canvas: Canvas, tab_set: TabSet[Any], rect: Rect, width: int) -> int:
    """The first tab to show, so the active one fits in *width* columns."""
    offset = min(tab_set.offset, tab_set.active)
    while offset < tab_set.active:
        span = sum(
            len(_tab_label(canvas, tab_set.tabs[index], rect)) + 1
            for index in range(offset, tab_set.active + 1)
        )
        if span <= width:
            break
        offset += 1
    return offset


def draw_tab_bar(
    canvas: Canvas, pane: str, rect: Rect, tab_set: TabSet[Any], *, focused: bool, hits: Hits
) -> None:
    """A pane's tab bar: ``[+]``, the tabs that fit, and markers for the rest."""
    colors, glyphs = canvas.colors, canvas.glyphs
    tabs, active = tab_set.tabs, tab_set.active
    tab_y = rect.y + 1
    x = rect.x + 1
    add_label = "[+]"
    canvas.add(tab_y, x, add_label, colors.current if focused else colors.border)
    hits.adds[pane] = Rect(tab_y, x, 1, len(add_label))
    # The column after ``[+]`` always belongs to the "earlier tabs" marker,
    # so the run of tabs does not shift when the bar starts scrolling.
    marker_x = x + len(add_label)
    x = marker_x + 2
    # The right-most column carries the "more tabs" marker when the bar
    # overflows, so the visible run always stops one column short of it.
    available = rect.x + rect.width - 2
    offset = _scroll_tab_bar(canvas, tab_set, rect, max(0, available - x))
    tab_set.offset = offset
    if offset:
        canvas.add(tab_y, marker_x, glyphs.more_before, colors.border, 1)
        hits.tab_scrolls.append((Rect(tab_y, marker_x, 1, 1), pane, offset - 1))
    hits.tabs[pane] = []
    hits.tab_closes[pane] = []
    shown = 0
    for index in range(offset, len(tabs)):
        label = _tab_label(canvas, tabs[index], rect)
        if available - x < len(label):
            break
        style = colors.border
        if index == active:
            style = colors.selected if focused else colors.current
        canvas.add(tab_y, x, label, style)
        hits.tabs[pane].append((Rect(tab_y, x, 1, len(label)), index))
        hits.tab_closes[pane].append((Rect(tab_y, x + len(label) - 2, 1, 1), index))
        x += len(label) + 1
        shown += 1
    if offset + shown < len(tabs):
        hits.tab_scrolls.append((Rect(tab_y, available, 1, 1), pane, offset + shown))
        canvas.add(tab_y, available, glyphs.more_after, colors.border, 1)


def draw_footer(canvas: Canvas, rect: Rect, focus: str, hits: Hits) -> None:
    """The footer row: the focused pane's keys, each one clickable."""
    colors = canvas.colors
    text = footer_text(focus, rect.width)
    canvas.add(rect.y, rect.x, text.ljust(rect.width), colors.header, rect.width)
    for match in re.finditer(r"(?<!\S)([^\s:]+)(?=:| Quit)", text):
        canvas.add(rect.y, rect.x + match.start(), match[0], colors.key, len(match[0]))
    for binding in KEY_BINDINGS:
        if not binding.footer or focus not in binding.panes:
            continue
        if text.find(binding.footer) < 0:
            continue
        for label, key in footer_actions(binding):
            start = text.find(label)
            hits.actions.append((Rect(rect.y, rect.x + start, 1, len(label)), key))


def draw_scroll_thumb(
    canvas: Canvas,
    pane: str,
    rect: Rect,
    non_content_rows: int,
    *,
    focused: bool,
    hits: Hits,
) -> None:
    """The scroll track and thumb inside *pane*'s right-most content column."""
    first, visible, total = hits.scroll[pane]
    # The track covers the content rows only, leaving the tab bar alone.
    track = rect.height - non_content_rows
    if total <= visible or visible <= 0 or track < 2:
        return
    colors, glyphs = canvas.colors, canvas.glyphs
    top = rect.y + rect.height - 1 - track
    size = max(1, min(track, round(visible * track / total)))
    start = min(track - size, round(first * track / total))
    column = rect.x + rect.width - 1 - SCROLLBAR_COLUMNS
    hits.scrollbars[pane] = Rect(top, column, track, 1)
    hits.thumbs[pane] = Rect(top + start, column, size, 1)
    for row in range(track):
        canvas.add(top + row, column, glyphs.box[BORDER_VERTICAL], colors.border)
    for row in range(size):
        canvas.add(
            top + start + row, column, glyphs.thumb, colors.focus if focused else colors.border
        )


def overlay_rect(size: tuple[int, int], preferred_width: int, preferred_height: int) -> Rect:
    """Centre a modal overlay between the header and the footer of a *size* screen."""
    rows, columns = size
    available = max(1, rows - HEADER_ROWS - FOOTER_ROWS)
    width = max(1, min(columns - 4, preferred_width))
    height = max(1, min(available, preferred_height))
    return Rect(
        HEADER_ROWS + (available - height) // 2, max(0, (columns - width) // 2), height, width
    )


def draw_overlay_frame(canvas: Canvas, rect: Rect, segments: list[str]) -> None:
    """Blank, frame and title a modal overlay, painted on top of the panes."""
    for row in range(rect.height):
        canvas.add(rect.y + row, rect.x, " " * rect.width, max_width=rect.width)
    frames = Frames()
    frames.box(rect, canvas.colors, focused=True)
    frames.flush(canvas)
    canvas.title(rect, segments)
