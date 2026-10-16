"""The XMEM pane: windows onto external memory, one per tab.

A tab holds an unresolved :class:`~ipu_debug.session.XmemRequest`
(``row|byte BASE OFFSET COUNT FORMAT``), so a window whose base is a register
follows that register from stop to stop. Reads go through
``DebugSession.read_xmem`` and are cached, with their tokenised lines, for as
long as ``DebugSession.generation`` stays the same; every write (a value
edited here, a register set elsewhere) moves the generation on and the next
render reads again.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from ipu_debug import format as debug_format
from ipu_debug.format import (
    CELL16_BACKGROUND_SHIFT,
    CELL16_COLOR_MASK,
    CELL16_FOREGROUND_SHIFT,
    XmemDisplayLine,
    XmemDisplayToken,
)
from ipu_debug.session import (
    XMEM_FORMAT_ITEM_SIZE_BYTES,
    XMEM_FORMATS,
    DebugError,
    DebugSession,
    XmemRequest,
    XmemWindow,
)
from ipu_debug.tui.draw import SCROLLBAR_COLUMNS, Canvas, Palette, Theme
from ipu_debug.tui.editor import EditedValue, TabTarget, tab_editor, value_editor
from ipu_debug.tui.keymap import Command, PaneId
from ipu_debug.tui.pane import PaneDrawing, Rect, Reply
from ipu_debug.tui.screen import overlay_style
from ipu_debug.tui.tabs import TabSet
from ipu_emu.ipu import xmem_row_size_bytes
from ipu_emu.ipu_state import IpuState

# The pane's first row is a border, its second the tab bar, its last a border.
XMEM_NON_CONTENT_ROWS = 3
TITLE = "3 XMEM"
EDITOR_HINTS = (
    "Address: row|byte BASE OFFSET COUNT",
    "Formats: " + "  ".join(fmt for fmt in XMEM_FORMATS),
)

# What identifies a resolved window: its bytes, format and row size.
WindowKey = tuple[int, int, str, int]


@dataclass
class XmemTab:
    """One symbolic XMEM request and its persistent presentation state."""

    request: XmemRequest
    scroll: int = 0
    baseline_key: WindowKey | None = None
    baseline_data: bytes | None = None
    cursor_line: int = 0
    cursor_column: int = 0
    horizontal_scroll: int = 0

    @property
    def title(self) -> str:
        request = self.request
        return f"{request.mode}:{request.base_token}+{request.offset_token} {request.fmt}"


@dataclass(frozen=True)
class _Rows:
    """What every row of one render shares."""

    canvas: Canvas
    tab: XmemTab
    window: XmemWindow
    baseline: bytes | None
    rect: Rect
    cursor: int


def window_key(window: XmemWindow) -> WindowKey:
    return (window.byte_address, window.byte_count, window.request.fmt, window.row_size)


def default_tabs(state: IpuState) -> list[XmemTab]:
    """The first stop's tabs: row 0 in hex and in the mode's numeric format."""
    numeric = (
        "f32"
        if state.wide_vector_debug and state.wide_vector_arithmetic.value == "fp32"
        else "int8"
    )
    if state.wide_vector_debug and state.wide_vector_arithmetic.value == "int32":
        numeric = "u32"
    row_bytes = xmem_row_size_bytes(state)
    return [XmemTab(XmemRequest("row", "0", "0", row_bytes, fmt)) for fmt in ("hex", numeric)]


def commit_baselines(session: DebugSession, tabs: TabSet[XmemTab]) -> None:
    """Remember every tab's bytes, so the next stop marks what changed."""
    for tab in tabs:
        try:
            window = session.read_xmem(tab.request)
        except DebugError:
            tab.baseline_key = None
            tab.baseline_data = None
            continue
        tab.baseline_key = window_key(window)
        tab.baseline_data = window.data


def color_summary(data: bytes) -> str:
    """The foreground and background colours a ``cell16`` window uses."""
    foregrounds: set[int] = set()
    backgrounds: set[int] = set()
    for offset in range(0, len(data), 2):
        cell = struct.unpack_from("<H", data, offset)[0]
        foregrounds.add((cell >> CELL16_FOREGROUND_SHIFT) & CELL16_COLOR_MASK)
        backgrounds.add((cell >> CELL16_BACKGROUND_SHIFT) & CELL16_COLOR_MASK)
    return f"fg={_compact_color_set(foregrounds)} bg={_compact_color_set(backgrounds)}"


def _compact_color_set(colors: set[int]) -> str:
    if colors == set(range(16)):
        return "0-f"
    if len(colors) <= 4:
        return "/".join(f"{color:x}" for color in sorted(colors))
    return f"{len(colors)} values"


def clean_marker(prefix: str) -> str:
    """A row-marker line (``--- row 3 ---``) without its dashes."""
    if prefix.startswith("--- ") and prefix.endswith(" ---"):
        return "  " + prefix[4:-4]
    return prefix


def token_style(
    colors: Palette,
    token: XmemDisplayToken,
    raw: bytes,
    previous: bytes | None,
    cursor: int | None,
) -> int:
    """The style of one value: colour or zero-dimming, change mark, cursor.

    *previous* is the value's bytes at the last stop (None: no baseline) and
    *cursor* the cursor highlight when the value is under the cursor.
    """
    colored = token.fg is not None and token.bg is not None
    if token.fg is not None and token.bg is not None:
        style = colors.cell(token.fg, token.bg)
    else:
        style = colors.value(not any(raw))
    if previous is not None and raw != previous:
        style |= colors.cell_changed if colored else colors.changed
    if cursor is not None:
        style = overlay_style(style, colors.cell_selected if colored else cursor)
    return style


def data_lines(lines: list[XmemDisplayLine]) -> list[int]:
    """The indices of the lines holding values (not row markers)."""
    return [index for index, line in enumerate(lines) if line.tokens]


def normalize_cursor(tab: XmemTab, lines: list[XmemDisplayLine]) -> None:
    """Put the cursor back on a value after the lines changed under it."""
    values = data_lines(lines)
    if not values:
        tab.cursor_line = 0
        tab.cursor_column = 0
        tab.horizontal_scroll = 0
        return
    if tab.cursor_line not in values:
        tab.cursor_line = values[0]
    token_count = len(lines[tab.cursor_line].tokens)
    tab.cursor_column = min(tab.cursor_column, token_count - 1)
    tab.horizontal_scroll = min(tab.horizontal_scroll, tab.cursor_column)


def scroll_to_cursor(
    tab: XmemTab, lines: list[XmemDisplayLine], rect: Rect, *, focused: bool
) -> None:
    """Scroll so the cursor's line and value are on screen (rows only when focused)."""
    content_height = max(0, rect.height - XMEM_NON_CONTENT_ROWS)
    if focused and content_height > 0:
        if tab.cursor_line < tab.scroll:
            tab.scroll = tab.cursor_line
        elif tab.cursor_line >= tab.scroll + content_height:
            tab.scroll = tab.cursor_line - content_height + 1
    tab.scroll = min(tab.scroll, max(0, len(lines) - content_height))
    if not lines or not lines[tab.cursor_line].tokens:
        return
    line = lines[tab.cursor_line]
    available = max(1, rect.width - 2 - SCROLLBAR_COLUMNS - len(line.prefix))
    if tab.cursor_column < tab.horizontal_scroll:
        tab.horizontal_scroll = tab.cursor_column
    while tab.horizontal_scroll < tab.cursor_column:
        tokens = line.tokens[tab.horizontal_scroll : tab.cursor_column + 1]
        width = sum(len(token.leading) + len(token.text) for token in tokens)
        if width <= available:
            break
        tab.horizontal_scroll += 1


def move_cursor(tab: XmemTab, lines: list[XmemDisplayLine], dx: int, dy: int) -> None:
    """Move *dy* value lines, then *dx* values, wrapping across lines."""
    normalize_cursor(tab, lines)
    values = data_lines(lines)
    if not values:
        return
    position = values.index(tab.cursor_line)
    if dy:
        position = max(0, min(len(values) - 1, position + dy))
        tab.cursor_line = values[position]
        tab.cursor_column = min(tab.cursor_column, len(lines[tab.cursor_line].tokens) - 1)
    for _ in range(-dx):
        if tab.cursor_column > 0:
            tab.cursor_column -= 1
        elif position > 0:
            position -= 1
            tab.cursor_line = values[position]
            tab.cursor_column = len(lines[tab.cursor_line].tokens) - 1
    for _ in range(dx):
        if tab.cursor_column + 1 < len(lines[tab.cursor_line].tokens):
            tab.cursor_column += 1
        elif position + 1 < len(values):
            position += 1
            tab.cursor_line = values[position]
            tab.cursor_column = 0


def jump_cursor(tab: XmemTab, lines: list[XmemDisplayLine], *, end: bool) -> None:
    """Move the cursor to the first or last value."""
    normalize_cursor(tab, lines)
    values = data_lines(lines)
    if not values:
        return
    tab.cursor_line = values[-1 if end else 0]
    tab.cursor_column = len(lines[tab.cursor_line].tokens) - 1 if end else 0


class XmemPane:
    """The XMEM pane: renders and edits ``XmemTab`` windows."""

    id: PaneId = "xmem"

    def __init__(self, theme: Theme) -> None:
        self.theme = theme
        # One request can tokenise into far more lines than a pane shows, so
        # reads and their lines are kept until the session's generation moves.
        self._generation: int | None = None
        self._windows: dict[XmemRequest, tuple[XmemWindow | None, str | None]] = {}
        self._lines: dict[WindowKey, list[XmemDisplayLine]] = {}

    # -- reads ----------------------------------------------------------------

    def _fresh(self, session: DebugSession) -> None:
        if self._generation != session.generation:
            self._generation = session.generation
            self._windows = {}
            self._lines = {}

    def resolve(
        self, session: DebugSession, request: XmemRequest
    ) -> tuple[XmemWindow | None, str | None]:
        """*request* read now: ``(window, None)`` or ``(None, error text)``."""
        self._fresh(session)
        if request not in self._windows:
            try:
                self._windows[request] = (session.read_xmem(request), None)
            except DebugError as error:
                self._windows[request] = (None, str(error))
        return self._windows[request]

    def lines(self, session: DebugSession, window: XmemWindow) -> list[XmemDisplayLine]:
        """*window* tokenised into display lines."""
        self._fresh(session)
        key = window_key(window)
        lines = self._lines.get(key)
        if lines is None:
            lines = debug_format.tokenize_xmem(
                window.data, window.byte_address, window.request.fmt, window.row_size
            )
            self._lines[key] = lines
        return lines

    def _tab_lines(
        self, session: DebugSession, view: TabSet[XmemTab]
    ) -> tuple[XmemTab, XmemWindow, list[XmemDisplayLine]] | None:
        tab = view.current()
        if tab is None:
            return None
        window, _ = self.resolve(session, tab.request)
        if window is None:
            return None
        return tab, window, self.lines(session, window)

    # -- render ---------------------------------------------------------------

    def render(
        self,
        session: DebugSession,
        view: TabSet[XmemTab],
        rect: Rect,
        *,
        focused: bool = False,
    ) -> PaneDrawing:
        canvas = Canvas(self.theme, rect)
        drawing = PaneDrawing(canvas.lines)
        colors = self.theme.colors
        tab = view.current()
        width = max(0, rect.width - 3 - SCROLLBAR_COLUMNS)
        if tab is None:
            canvas.title(rect, [TITLE, "no tabs"], focused=focused)
            message = "No tabs. Press F2 or click [+]."
            canvas.add(rect.y + 2, rect.x + 2, message, colors.border, width)
            return drawing
        window, error = self.resolve(session, tab.request)
        if error is not None or window is None:
            canvas.title(rect, [TITLE, tab.request.fmt], focused=focused)
            canvas.add(rect.y + 2, rect.x + 2, f"Error: {error}", colors.error, width)
            return drawing
        lines = self.lines(session, window)
        normalize_cursor(tab, lines)
        scroll_to_cursor(tab, lines, rect, focused=focused)
        self._draw_title(canvas, tab, window, len(lines), rect, focused=focused)
        content_height = max(0, rect.height - XMEM_NON_CONTENT_ROWS)
        drawing.scroll = (tab.scroll, content_height, len(lines))
        baseline = tab.baseline_data if tab.baseline_key == window_key(window) else None
        rows = _Rows(canvas, tab, window, baseline, rect, colors.cursor_for(focused))
        last = min(len(lines), tab.scroll + content_height)
        for row, number in enumerate(range(tab.scroll, last)):
            drawing.hits.extend(self._draw_line(rows, lines[number], number, row))
        return drawing

    def _draw_title(
        self,
        canvas: Canvas,
        tab: XmemTab,
        window: XmemWindow,
        line_count: int,
        rect: Rect,
        *,
        focused: bool,
    ) -> None:
        content_height = max(0, rect.height - XMEM_NON_CONTENT_ROWS)
        segments = [
            TITLE,
            f"0x{window.byte_address:08x}",
            f"{window.byte_count} bytes",
            window.request.fmt,
        ]
        if window.request.fmt == "cell16":
            segments.append(color_summary(window.data))
        canvas.title(rect, segments, focused=focused)
        # The title fills the top border at the 80-column minimum, so the
        # position readout goes on the bottom border instead.
        first = min(line_count, tab.scroll + 1)
        last = min(line_count, tab.scroll + content_height)
        canvas.position(rect, f" lines {first}-{last}/{line_count} ")

    def _draw_line(
        self, rows: _Rows, line: XmemDisplayLine, number: int, row: int
    ) -> list[tuple[Rect, Any]]:
        """Draw display line *number* on content row *row*; return where each value landed."""
        canvas, tab, window, baseline, rect = (
            rows.canvas,
            rows.tab,
            rows.window,
            rows.baseline,
            rows.rect,
        )
        colors = self.theme.colors
        y = rect.y + 2 + row
        x = rect.x + 1
        limit = rect.x + rect.width - 1 - SCROLLBAR_COLUMNS
        prefix = line.prefix if line.tokens else clean_marker(line.prefix)
        prefix_style = colors.border if not line.tokens or prefix.strip() else 0
        canvas.add(y, x, prefix, prefix_style, max(0, limit - x))
        x += len(prefix)
        hits: list[tuple[Rect, Any]] = []
        for column in range(tab.horizontal_scroll, len(line.tokens)):
            token = line.tokens[column]
            if x + len(token.leading) + len(token.text) > limit:
                break
            hits.append((Rect(y, x, 1, len(token.leading) + len(token.text)), (number, column)))
            canvas.add(y, x, token.leading, max_width=limit - x)
            x += len(token.leading)
            span = slice(token.raw_start, token.raw_start + token.raw_size)
            style = token_style(
                colors,
                token,
                window.data[span],
                baseline[span] if baseline is not None else None,
                rows.cursor if (number, column) == (tab.cursor_line, tab.cursor_column) else None,
            )
            if token.fg is not None and token.bg is not None:
                canvas.add(y, x, token.text, style, limit - x)
            else:
                canvas.add_value(y, x, token.text, style, limit - x)
            x += len(token.text)
        return hits

    # -- commands -------------------------------------------------------------

    def handle(
        self, command: Command, session: DebugSession, view: TabSet[XmemTab], rect: Rect
    ) -> Reply | None:
        if command in (Command.ADD_TAB, Command.EDIT_TAB):
            return self._tab_editor(command, session, view)
        if command is Command.CYCLE_FORMAT:
            return self._cycle_format(session, view)
        if command is Command.EDIT_VALUE:
            return self._value_editor(session, view)
        page = max(1, rect.height - XMEM_NON_CONTENT_ROWS)
        moves = {
            Command.MOVE_UP: (0, -1),
            Command.MOVE_DOWN: (0, 1),
            Command.MOVE_LEFT: (-1, 0),
            Command.MOVE_RIGHT: (1, 0),
            Command.PAGE_UP: (0, -page),
            Command.PAGE_DOWN: (0, page),
        }
        if command in moves:
            dx, dy = moves[command]
            self.move(session, view, rect, dx=dx, dy=dy)
        elif command in (Command.JUMP_FIRST, Command.JUMP_LAST):
            found = self._tab_lines(session, view)
            if found is not None:
                jump_cursor(found[0], found[2], end=command is Command.JUMP_LAST)
        return None

    def move(
        self,
        session: DebugSession,
        view: TabSet[XmemTab],
        rect: Rect,
        *,
        dx: int = 0,
        dy: int = 0,
    ) -> None:
        del rect
        found = self._tab_lines(session, view)
        if found is not None:
            move_cursor(found[0], found[2], dx, dy)

    def _cycle_format(self, session: DebugSession, view: TabSet[XmemTab]) -> Reply | None:
        """The next format whose item size divides the window's address and length."""
        tab = view.current()
        if tab is None:
            return Reply("XMEM has no tabs")
        window, error = self.resolve(session, tab.request)
        if error is not None or window is None:
            return Reply(f"XMEM format error: {error}", error=True)
        current = XMEM_FORMATS.index(tab.request.fmt)
        for distance in range(1, len(XMEM_FORMATS) + 1):
            candidate = XMEM_FORMATS[(current + distance) % len(XMEM_FORMATS)]
            size = XMEM_FORMAT_ITEM_SIZE_BYTES[candidate]
            if window.byte_address % size or window.byte_count % size:
                continue
            request = tab.request
            tab.request = XmemRequest(
                request.mode,
                request.base_token,
                request.offset_token,
                window.byte_count // size,
                candidate,
            )
            tab.scroll = tab.cursor_line = tab.cursor_column = tab.horizontal_scroll = 0
            tab.baseline_key = None
            tab.baseline_data = None
            return Reply(f"XMEM format: {candidate}")
        return None

    def _value_editor(self, session: DebugSession, view: TabSet[XmemTab]) -> Reply | None:
        """Edit the value under the cursor, written back with ``write_xmem``."""
        tab = view.current()
        if tab is None:
            return None
        window, error = self.resolve(session, tab.request)
        if error or window is None:
            return Reply(str(error or "No memory value selected"), error=True)
        lines = self.lines(session, window)
        normalize_cursor(tab, lines)
        tokens = lines[tab.cursor_line].tokens
        if not tokens:
            return None
        token = tokens[tab.cursor_column]
        address = window.byte_address + token.raw_start
        raw = window.data[token.raw_start : token.raw_start + token.raw_size]

        def write(data: bytes) -> None:
            session.write_xmem(address, data)

        label = f"XMEM 0x{address:08x}"
        value = EditedValue(label, tab.request.fmt, raw, token.raw_size)
        return value_editor(session, self.id, value, write)

    def _tab_editor(self, command: Command, session: DebugSession, view: TabSet[XmemTab]) -> Reply:
        tab = view.current()
        if command is Command.EDIT_TAB and tab is not None:
            mode, text = "edit", tab.request.command()[len("xmem ") :]
        else:
            mode, text = "add", f"row 0 0 {xmem_row_size_bytes(session.state)} hex"

        def parse(command: str, existing: Any) -> tuple[XmemTab | None, str | None]:
            del existing  # A new request replaces the old one whole.
            try:
                request = session.parse_xmem(command)
            except DebugError as error:
                return None, str(error)
            _, error = self.resolve(session, request)
            if error is not None:
                return None, error
            return XmemTab(request), None

        return tab_editor(self.id, TabTarget("XMEM", EDITOR_HINTS, parse), mode, view, text)

    # -- pointer --------------------------------------------------------------

    def place_cursor(self, session: DebugSession, view: TabSet[XmemTab], payload: Any) -> None:
        del session
        tab = view.current()
        if tab is not None:
            tab.cursor_line, tab.cursor_column = payload

    def cursor_row(self, session: DebugSession, view: TabSet[XmemTab], rect: Rect) -> int:
        del session, rect
        tab = view.current()
        return tab.cursor_line if tab is not None else 0

    def scroll_to_row(
        self, session: DebugSession, view: TabSet[XmemTab], rect: Rect, row: int
    ) -> None:
        del session, rect
        tab = view.current()
        if tab is not None:
            tab.cursor_line = row
