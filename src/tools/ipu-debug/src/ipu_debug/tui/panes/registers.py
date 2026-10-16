"""The registers pane: the 32-bit scalar registers (LR, CR) as a grid.

Which registers it shows comes from the register schema in ``ipu_isa``: every
scalar register file entry is a group, and a tab shows ``all`` of them or one
group. The grid reflows to the height it is given; values changed since the
last stop are marked. The pane reads through ``DebugSession.registers`` and
writes only through ``DebugSession.set_register``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from ipu_debug.format import F32_DISPLAY_WIDTH, F32_SIGNIFICANT_DIGITS, U32_DECIMAL_WIDTH
from ipu_debug.session import DebugSession, is_scalar
from ipu_debug.tui.draw import (
    BITS_PER_REGISTER,
    PANE_CONTENT_INDENT,
    PANE_CONTENT_MARGIN,
    Canvas,
    Theme,
    format_bits,
)
from ipu_debug.tui.editor import TabTarget, scalar_editor, tab_editor
from ipu_debug.tui.keymap import Command, PaneId
from ipu_debug.tui.pane import PaneDrawing, Rect, Reply
from ipu_debug.tui.screen import overlay_style
from ipu_debug.tui.tabs import TabSet
from ipu_isa.constants import CR_READ_ONLY_INITIAL_VALUES, CR_REGISTER_NAME
from ipu_isa.registers import create_regfile_schema

# ``(name, count)`` of every scalar register group, in schema order.
SCALAR_GROUPS: tuple[tuple[str, int], ...] = tuple(
    (desc.name, desc.count) for desc in create_regfile_schema() if is_scalar(desc)
)
REGISTER_GROUPS = tuple(name for name, _ in SCALAR_GROUPS)
_GROUP_COUNTS = dict(SCALAR_GROUPS)
REGISTERS_PER_GROUP = max(_GROUP_COUNTS.values())
REGISTER_TAB_GROUPS = ("all", *REGISTER_GROUPS)
# ``LR / CR``: how the status line and the editors name the pane.
LABEL = " / ".join(name.upper() for name in REGISTER_GROUPS)
TITLE = "4 " + " / ".join(name.upper() for name in reversed(REGISTER_GROUPS))
SCALAR_REGISTER_FORMATS = ("hex", "u32", "int32", "f32", "bits")
VALUE_WIDTHS = {
    "hex": 8,
    "u32": 10,
    "int32": 11,
    "f32": F32_DISPLAY_WIDTH,
    "bits": BITS_PER_REGISTER,
}

# The pane spends its first row on a border, its second on the tab bar, its
# third on column headers and its last on a border.
REGISTER_NON_CONTENT_ROWS = 4
# The grid reflows to the height it is given: every shape divides a group
# exactly, so no column is ever ragged. The tallest shape that fits wins,
# because it also needs the fewest columns.
REGISTER_GRID_ROWS = (REGISTERS_PER_GROUP, REGISTERS_PER_GROUP // 2, REGISTERS_PER_GROUP // 4)
DEFAULT_REGISTER_ROWS = REGISTERS_PER_GROUP // 2
REGISTER_LABEL_WIDTH = len("L00 ")
REGISTER_MIN_GUTTER = 2
REGISTER_MAX_GUTTER = 4


def _choices(names: tuple[str, ...]) -> str:
    return ", ".join(names[:-1]) + ", or " + names[-1] if len(names) > 1 else names[0]


_READ_ONLY = " and ".join(
    f"{CR_REGISTER_NAME.upper()}{index}" for index in sorted(CR_READ_ONLY_INITIAL_VALUES)
)
SCALAR_HINT = f"Register masking applies; {_READ_ONLY} are read-only"
EDITOR_HINTS = (
    f"Group: {_choices(REGISTER_TAB_GROUPS)}",
    "Format is retained independently per tab",
)


@dataclass
class RegisterTab:
    """One filtered scalar-register view."""

    group: str = "all"
    fmt: str = "hex"
    cursor: int = 0
    column_scroll: int = 0
    baseline: dict[tuple[str, int], int] | None = None

    @property
    def title(self) -> str:
        return f"{self.group} {self.fmt}"

    @property
    def groups(self) -> tuple[str, ...]:
        return REGISTER_GROUPS if self.group == "all" else (self.group,)

    @property
    def size(self) -> int:
        """How many grid positions the tab has."""
        return len(self.groups) * REGISTERS_PER_GROUP

    def register_at(self, position: int) -> str:
        """The register at grid position *position*, as ``DebugSession`` names it."""
        return f"{self.groups[position // REGISTERS_PER_GROUP]}{position % REGISTERS_PER_GROUP}"


def default_tabs(fresh: bool) -> list[RegisterTab]:
    """One tab per group choice on the first stop; afterwards just ``all``."""
    return [RegisterTab(group=group) for group in (REGISTER_TAB_GROUPS if fresh else ("all",))]


def register_names(groups: tuple[str, ...] = REGISTER_GROUPS) -> list[str]:
    """Every register of *groups*, as ``DebugSession`` names them."""
    return [f"{name}{index}" for name in groups for index in range(_GROUP_COUNTS[name])]


def snapshot(
    session: DebugSession, groups: tuple[str, ...] = REGISTER_GROUPS
) -> dict[tuple[str, int], int]:
    """``(group, index) -> value`` for every register of *groups*."""
    values: dict[tuple[str, int], int] = {}
    for view in session.registers(register_names(groups)):
        assert isinstance(view.value, int)  # Scalar registers read as integers.
        values[(view.name, view.index)] = view.value
    return values


def commit_baselines(session: DebugSession, tabs: TabSet[RegisterTab]) -> None:
    """Remember every register, so the next stop marks what changed."""
    values = snapshot(session)
    for tab in tabs:
        tab.baseline = values.copy()


def register_grid_rows(content_rows: int) -> int:
    """The tallest register-grid shape that fits in *content_rows*."""
    for rows in REGISTER_GRID_ROWS:
        if rows <= content_rows:
            return rows
    return min(REGISTER_GRID_ROWS)


def grid_shape(tab: RegisterTab, rect: Rect) -> tuple[int, int]:
    """The ``(rows, columns)`` shape *tab*'s grid takes in *rect*."""
    rows = register_grid_rows(max(1, rect.height - REGISTER_NON_CONTENT_ROWS))
    return rows, tab.size // rows


def natural_width(tab: RegisterTab, height: int) -> int:
    """The pane width *tab* needs to show its whole grid at *height* rows."""
    _, columns = grid_shape(tab, Rect(0, 0, height, 0))
    column_width = REGISTER_LABEL_WIDTH + VALUE_WIDTHS[tab.fmt]
    return (
        columns * column_width
        + (columns - 1) * REGISTER_MAX_GUTTER
        + 2
        + PANE_CONTENT_INDENT
        + PANE_CONTENT_MARGIN
    )


def format_scalar(value: int, fmt: str, width: int) -> str:
    """A 32-bit register value in format *fmt*, right-aligned in *width*."""
    unsigned = value & 0xFFFFFFFF
    if fmt == "hex":
        text = f"{unsigned:08x}"
    elif fmt == "u32":
        text = f"{unsigned:0{U32_DECIMAL_WIDTH}d}"
    elif fmt == "int32":
        text = str(struct.unpack("<i", struct.pack("<I", unsigned))[0])
    elif fmt == "f32":
        number = struct.unpack("<f", struct.pack("<I", unsigned))[0]
        text = f"{number:>{F32_DISPLAY_WIDTH}.{F32_SIGNIFICANT_DIGITS}g}"
    else:
        text = format_bits(unsigned)
    return text.rjust(width)[:width]


@dataclass(frozen=True)
class Grid:
    """Where one render puts the grid's columns."""

    rows: int
    columns: int
    visible: int
    x: int
    width: int
    stride: int
    column_width: int


def layout_grid(tab: RegisterTab, rect: Rect) -> Grid:
    """Fit as many whole columns as *rect* holds, spreading the spare width."""
    rows, columns = grid_shape(tab, rect)
    column_width = REGISTER_LABEL_WIDTH + VALUE_WIDTHS[tab.fmt]
    # The indent and margin are padding, not a budget: a wide format such
    # as ``bits`` gives them up rather than truncating a register value.
    inner = max(1, rect.width - 2)
    padded = max(1, inner - PANE_CONTENT_INDENT - PANE_CONTENT_MARGIN)
    if padded < column_width <= inner:
        x, width = rect.x + 1, inner
    else:
        x, width = rect.x + 1 + PANE_CONTENT_INDENT, padded
    fit = (width + REGISTER_MIN_GUTTER) // (column_width + REGISTER_MIN_GUTTER)
    visible = max(1, min(columns, fit))
    gaps = visible - 1
    spare = width - visible * column_width
    gutter = (
        REGISTER_MIN_GUTTER
        if gaps <= 0
        else max(REGISTER_MIN_GUTTER, min(REGISTER_MAX_GUTTER, spare // gaps))
    )
    return Grid(rows, columns, visible, x, width, column_width + gutter, column_width)


def scroll_to_cursor(tab: RegisterTab, grid: Grid) -> None:
    """Clamp the cursor and scroll the columns so it is on screen."""
    tab.cursor = max(0, min(tab.cursor, tab.size - 1))
    column = tab.cursor // grid.rows
    if column < tab.column_scroll:
        tab.column_scroll = column
    elif column >= tab.column_scroll + grid.visible:
        tab.column_scroll = column - grid.visible + 1
    tab.column_scroll = max(0, min(tab.column_scroll, grid.columns - grid.visible))


@dataclass(frozen=True)
class _Cell:
    """One register's place on screen and in the grid."""

    y: int
    x: int
    width: int
    name: str
    index: int
    position: int


def _draw_cell(canvas: Canvas, tab: RegisterTab, cell: _Cell, value: int, cursor: int) -> None:
    colors = canvas.colors
    changed = tab.baseline is not None and tab.baseline[(cell.name, cell.index)] != value
    label = f"{cell.name[0].upper()}{cell.index:02d} "
    label_style = colors.border
    value_style = colors.changed if changed else colors.value(not value)
    if tab.cursor == cell.position:
        label_style = overlay_style(label_style, cursor)
        value_style = overlay_style(value_style, cursor)
    canvas.add(cell.y, cell.x, label, label_style, min(len(label), cell.width))
    remaining = cell.width - len(label)
    if remaining > 0:
        text = format_scalar(value, tab.fmt, VALUE_WIDTHS[tab.fmt])
        canvas.add_value(cell.y, cell.x + len(label), text, value_style, remaining)


class RegistersPane:
    """The LR / CR pane: renders and edits ``RegisterTab`` grids."""

    id: PaneId = "registers"

    def __init__(self, theme: Theme) -> None:
        self.theme = theme

    def render(
        self,
        session: DebugSession,
        view: TabSet[RegisterTab],
        rect: Rect,
        *,
        focused: bool = False,
    ) -> PaneDrawing:
        tab = view.current()
        assert tab is not None  # Closing the last tab reopens a default one.
        canvas = Canvas(self.theme, rect)
        drawing = PaneDrawing(canvas.lines)
        grid = layout_grid(tab, rect)
        scroll_to_cursor(tab, grid)
        values = snapshot(session, tab.groups)
        cursor = self.theme.colors.cursor_for(focused)
        limit = grid.x + grid.width
        last = min(grid.columns, tab.column_scroll + grid.visible)
        for shown, column in enumerate(range(tab.column_scroll, last)):
            x = grid.x + shown * grid.stride
            width = min(grid.column_width, limit - x)
            first = column * grid.rows
            name = tab.groups[first // REGISTERS_PER_GROUP]
            first_index = first % REGISTERS_PER_GROUP
            header = f"{name.upper()} {first_index:02d}-{first_index + grid.rows - 1:02d}"
            canvas.add(rect.y + 2, x, header, self.theme.colors.border, width)
            for row in range(grid.rows):
                index = first_index + row
                if index >= _GROUP_COUNTS[name] or width <= 0:
                    continue
                cell = _Cell(rect.y + 3 + row, x, width, name, index, first + row)
                drawing.hits.append((Rect(cell.y, x, 1, width), cell.position))
                _draw_cell(canvas, tab, cell, values[(name, index)], cursor)
        readout = (
            f" cols {tab.column_scroll + 1}-{tab.column_scroll + grid.visible}/{grid.columns} "
            if grid.columns > grid.visible
            else ""
        )
        canvas.top_pane_header(rect, [TITLE], readout, focused=focused)
        return drawing

    # -- commands -------------------------------------------------------------

    def handle(
        self, command: Command, session: DebugSession, view: TabSet[RegisterTab], rect: Rect
    ) -> Reply | None:
        tab = view.current()
        assert tab is not None
        if command in (Command.ADD_TAB, Command.EDIT_TAB):
            return self._tab_editor(command, view)
        if command is Command.CYCLE_FORMAT:
            formats = SCALAR_REGISTER_FORMATS
            index = formats.index(tab.fmt) if tab.fmt in formats else -1
            tab.fmt = formats[(index + 1) % len(formats)]
            tab.column_scroll = 0
            return Reply(f"{LABEL} format: {tab.fmt}")
        if command is Command.EDIT_VALUE:
            register = tab.register_at(max(0, min(tab.cursor, tab.size - 1)))
            current = session.registers([register])[0].value
            assert isinstance(current, int)
            return scalar_editor(session, self.id, register, current, SCALAR_HINT)
        page = max(1, rect.height - REGISTER_NON_CONTENT_ROWS + 1)
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
            tab.cursor = tab.size - 1 if command is Command.JUMP_LAST else 0
        return None

    def _tab_editor(self, command: Command, view: TabSet[RegisterTab]) -> Reply:
        tab = view.current()
        editing = command is Command.EDIT_TAB and tab is not None
        mode, text = ("edit", tab.group) if editing and tab is not None else ("add", "all")

        def parse(text: str, existing: Any) -> tuple[RegisterTab | None, str | None]:
            group = text.strip().lower()
            if group not in REGISTER_TAB_GROUPS:
                return None, f"expected {_choices(REGISTER_TAB_GROUPS)}"
            fmt = existing.fmt if isinstance(existing, RegisterTab) else "hex"
            return RegisterTab(group=group, fmt=fmt), None

        return tab_editor(self.id, TabTarget(LABEL, EDITOR_HINTS, parse), mode, view, text)

    def move(
        self,
        session: DebugSession,
        view: TabSet[RegisterTab],
        rect: Rect,
        *,
        dx: int = 0,
        dy: int = 0,
    ) -> None:
        """Move the cursor *dx* columns and *dy* rows, stopping at the grid's edges."""
        del session
        tab = view.current()
        assert tab is not None
        rows, columns = grid_shape(tab, rect)
        row = max(0, min(rows - 1, tab.cursor % rows + dy))
        column = max(0, min(columns - 1, tab.cursor // rows + dx))
        tab.cursor = column * rows + row

    # -- pointer --------------------------------------------------------------

    def place_cursor(self, session: DebugSession, view: TabSet[RegisterTab], payload: Any) -> None:
        del session
        tab = view.current()
        if tab is not None:
            tab.cursor = payload

    def cursor_row(self, session: DebugSession, view: TabSet[RegisterTab], rect: Rect) -> int:
        """The grid does not scroll vertically: there is no scroll row."""
        del session, view, rect
        return 0

    def scroll_to_row(
        self, session: DebugSession, view: TabSet[RegisterTab], rect: Rect, row: int
    ) -> None:
        del session, view, rect, row
