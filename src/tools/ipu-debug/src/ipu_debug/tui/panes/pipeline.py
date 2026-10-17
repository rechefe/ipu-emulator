"""The pipeline-register pane: one vector pipeline register per tab, as values.

A tab names a logical register (R0, R1, R_C, R_MASK, R_ACC, POST_AAQ,
MULT_RES, MEM_BYPASS) and a format; in wide-vector debug mode R0, R1 and R_C
read their wide debug storage. The values wrap to the pane's width; R_MASK
keeps each of its 16-byte slots on lines of its own, with a separator line
between slots. Values changed since the last stop are marked. The pane reads
through ``DebugSession.register`` and writes only through
``DebugSession.write_register_bytes``.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ipu_debug.format import (
    CELL16_BACKGROUND_SHIFT,
    CELL16_CHARACTER_MASK,
    CELL16_CHARACTERS_PER_GROUP,
    CELL16_COLOR_MASK,
    CELL16_FOREGROUND_SHIFT,
    F32_DISPLAY_WIDTH,
    F32_SIGNIFICANT_DIGITS,
    HEX_BYTES_PER_GROUP,
    INT8_DECIMAL_WIDTH,
    U8_DECIMAL_WIDTH,
    U32_DECIMAL_WIDTH,
    decode_16bit_cell,
)
from ipu_debug.session import DebugSession
from ipu_debug.tui.draw import BITS_PER_REGISTER, SCROLLBAR_COLUMNS, Canvas, Theme, format_bits
from ipu_debug.tui.editor import EditedValue, TabTarget, tab_editor, value_editor
from ipu_debug.tui.keymap import Command, PaneId
from ipu_debug.tui.pane import PaneDrawing, Rect, Reply
from ipu_debug.tui.screen import overlay_style
from ipu_debug.tui.tabs import TabSet
from ipu_emu.ipu_state import IpuState
from ipu_isa.formats import ElementFormat

# The pane's first row is a border, its second the tab bar, its last a border.
PIPELINE_NON_CONTENT_ROWS = 3
TITLE = "1 Pipeline Registers"
PIPELINE_REGISTER_FORMATS = ("hex", "int8", "u8", "cell16", "u32", "int32", "f32", "bits")
EDITOR_HINTS = (
    "Register: R0, R1, R_C, R_MASK, R_ACC, POST_AAQ,",
    "          MULT_RES, or MEM_BYPASS",
)
# Each line starts with the byte offset of its first value.
OFFSET_PREFIX_WIDTH = len("  0000: ")
# R_MASK is eight independent 128-bit slots; a separator line divides them.
MASK_SLOT_BYTES = 16
SLOT_SEPARATOR = "------------"
INT32_DISPLAY_WIDTH = 11


@dataclass(frozen=True)
class PipelineRegisterSpec:
    """One logical pipeline register and its mode-specific storage."""

    key: str
    label: str
    backing: str
    index: int = 0
    wide_backing: str | None = None


PIPELINE_REGISTERS = (
    PipelineRegisterSpec("r0", "R0", "r", 0, "r_wide_debug"),
    PipelineRegisterSpec("r1", "R1", "r", 1, "r_wide_debug"),
    PipelineRegisterSpec("r_cyclic", "R_C", "r_cyclic", 0, "r_cyclic_wide_debug"),
    PipelineRegisterSpec("r_mask", "R_MASK", "r_mask"),
    PipelineRegisterSpec("r_acc", "R_ACC", "r_acc"),
    PipelineRegisterSpec("post_aaq_reg", "POST_AAQ", "post_aaq_reg"),
    PipelineRegisterSpec("mult_res", "MULT_RES", "mult_res"),
    PipelineRegisterSpec("mem_bypass", "MEM_BYPASS", "mem_bypass"),
)
_WORD_REGISTERS = frozenset({"r_acc", "mult_res"})
_WIDE_VALUE_REGISTERS = frozenset({"r0", "r1", "r_cyclic"} | _WORD_REGISTERS)


def pipeline_spec(register_key: str) -> PipelineRegisterSpec:
    """The logical register *register_key* names (``r0``, ``r_mask``...)."""
    return next(spec for spec in PIPELINE_REGISTERS if spec.key == register_key)


@dataclass
class PipelineTab:
    """One logical pipeline-register view."""

    register_key: str = "r0"
    fmt: str | None = None
    scroll: int = 0
    cursor: int = 0
    baseline_key: str | None = None
    baseline_data: bytes | None = None

    @property
    def title(self) -> str:
        return f"{pipeline_spec(self.register_key).label} {self.fmt or 'auto'}"


def _is_wide(state: IpuState) -> bool:
    """*state* runs in a four-byte element format (FP32 or INT32)."""
    return state.codec.element_bytes != 1


def backing(state: IpuState, spec: PipelineRegisterSpec) -> str:
    """The register file entry holding *spec* in *state*'s mode."""
    if _is_wide(state) and spec.wide_backing is not None:
        return spec.wide_backing
    return spec.backing


def read_register(session: DebugSession, spec: PipelineRegisterSpec) -> bytes:
    """The bytes of *spec* now."""
    value = session.register(backing(session.state, spec), spec.index).value
    assert isinstance(value, bytes)  # Vector registers read as bytes.
    return value


def auto_format(state: IpuState, spec: PipelineRegisterSpec) -> str:
    """The format of a tab without one: the mode's arithmetic for the wide values."""
    if _is_wide(state) and spec.key in _WIDE_VALUE_REGISTERS:
        return "f32" if state.fmt is ElementFormat.FP32 else "int32"
    if spec.key in _WORD_REGISTERS:
        return "int32"
    return "hex"


def default_format(state: IpuState, spec: PipelineRegisterSpec) -> str:
    """The explicit format a fresh tab of *spec* starts in (auto tabs keep auto)."""
    if spec.key == "r_mask":
        return "bits"
    if spec.key == "post_aaq_reg":
        if _is_wide(state) and not state.aaq.quantize_output:
            return "f32" if state.fmt is ElementFormat.FP32 else "int32"
        return "int8"
    if state.fmt is ElementFormat.INT8 and spec.key in ("r0", "r1", "r_cyclic"):
        return "int8"
    return auto_format(state, spec)


def default_tabs(state: IpuState, fresh: bool) -> list[PipelineTab]:
    """One tab per register on the first stop; afterwards just R0."""
    specs = PIPELINE_REGISTERS if fresh else PIPELINE_REGISTERS[:1]
    return [PipelineTab(register_key=spec.key, fmt=default_format(state, spec)) for spec in specs]


def commit_baselines(session: DebugSession, tabs: TabSet[PipelineTab]) -> None:
    """Remember every tab's bytes, so the next stop marks what changed."""
    for tab in tabs:
        spec = pipeline_spec(tab.register_key)
        tab.baseline_key = spec.key
        tab.baseline_data = read_register(session, spec)


def value_geometry(fmt: str) -> tuple[int, int]:
    """``(bytes per value, characters per value)`` of format *fmt*."""
    geometry = {
        "hex": (1, 2),
        "u8": (1, U8_DECIMAL_WIDTH),
        "int8": (1, INT8_DECIMAL_WIDTH),
        "cell16": (2, 1),
        "u32": (4, U32_DECIMAL_WIDTH),
        "int32": (4, INT32_DISPLAY_WIDTH),
        "bits": (4, BITS_PER_REGISTER),
    }
    return geometry.get(fmt, (4, F32_DISPLAY_WIDTH))


def item_spacing(state: IpuState, fmt: str, item_in_line: int) -> int:
    """The gap before value *item_in_line* of a line: groups get a wider one."""
    if item_in_line == 0:
        return 0
    if fmt == "cell16":
        return 1 if item_in_line % CELL16_CHARACTERS_PER_GROUP == 0 else 0
    if fmt == "hex" and _is_wide(state) and item_in_line % HEX_BYTES_PER_GROUP == 0:
        return 2
    return 1


def format_value(raw: bytes, fmt: str, width: int) -> str:
    """One value's text (``cell16`` excepted, which decodes to a coloured character)."""
    if fmt == "hex":
        return f"{raw[0]:02x}"
    if fmt == "u8":
        return f"{raw[0]:0{U8_DECIMAL_WIDTH}d}"
    if fmt == "int8":
        return f"{struct.unpack('<b', raw)[0]:>{width}d}"
    if fmt == "u32":
        return f"{struct.unpack('<I', raw)[0]:0{U32_DECIMAL_WIDTH}d}"
    if fmt == "int32":
        return f"{struct.unpack('<i', raw)[0]:>{width}d}"
    if fmt == "bits":
        return format_bits(struct.unpack("<I", raw)[0])
    value = struct.unpack("<f", raw)[0]
    return f"{value:>{width}.{F32_SIGNIFICANT_DIGITS}g}"


def _fit_items(value_width: int, rect: Rect, spacing: Callable[[int], int]) -> int:
    """How many values (and their gaps) fit on one line of *rect*."""
    content_width = max(0, rect.width - 2 - SCROLLBAR_COLUMNS)
    available_width = max(1, content_width - OFFSET_PREFIX_WIDTH)
    used_width = 0
    item_count = 0
    while True:
        next_width = value_width + spacing(item_count)
        if item_count > 0 and used_width + next_width > available_width:
            break
        used_width += next_width
        item_count += 1
        if used_width >= available_width:
            break
    return item_count


@dataclass(frozen=True)
class ValueGrid:
    """How one tab's register lays out in a rectangle.

    ``rows`` holds, per display line, the index of its first value, or None
    for an R_MASK slot separator.
    """

    spec: PipelineRegisterSpec
    data: bytes
    fmt: str
    item_size: int
    value_width: int
    per_line: int
    rows: list[int | None]

    @property
    def item_count(self) -> int:
        return len(self.data) // self.item_size

    def line_of(self, item: int) -> int:
        """The display line value *item* is on."""
        return next(
            index
            for index, first in enumerate(self.rows)
            if first is not None and first <= item < first + self.per_line
        )


def value_grid(session: DebugSession, tab: PipelineTab, rect: Rect) -> ValueGrid:
    """Lay *tab*'s register out in *rect*."""
    spec = pipeline_spec(tab.register_key)
    data = read_register(session, spec)
    fmt = tab.fmt or auto_format(session.state, spec)
    size, width = value_geometry(fmt)
    per_line = _fit_items(width, rect, lambda item: item_spacing(session.state, fmt, item))
    mask = spec.key == "r_mask"
    if mask:
        # A line never straddles two mask slots.
        slot_items = MASK_SLOT_BYTES // size
        per_line = max(
            count
            for count in range(1, slot_items + 1)
            if count <= max(1, per_line) and slot_items % count == 0
        )
    per_line = max(1, per_line)
    rows: list[int | None] = []
    for first in range(0, len(data) // size, per_line):
        if mask and first and first * size % MASK_SLOT_BYTES == 0:
            rows.append(None)
        rows.append(first)
    return ValueGrid(spec, data, fmt, size, width, per_line, rows)


@dataclass(frozen=True)
class _Lines:
    """What every line of one render shares."""

    canvas: Canvas
    session: DebugSession
    tab: PipelineTab
    grid: ValueGrid
    baseline: bytes | None
    cursor: int


class PipelinePane:
    """The pipeline-register pane: renders and edits ``PipelineTab`` values."""

    id: PaneId = "pipeline"

    def __init__(self, theme: Theme) -> None:
        self.theme = theme

    # -- render ---------------------------------------------------------------

    def render(
        self,
        session: DebugSession,
        view: TabSet[PipelineTab],
        rect: Rect,
        *,
        focused: bool = False,
    ) -> PaneDrawing:
        tab = view.current()
        assert tab is not None  # Closing the last tab reopens a default one.
        canvas = Canvas(self.theme, rect)
        drawing = PaneDrawing(canvas.lines)
        grid = value_grid(session, tab, rect)
        tab.cursor = min(tab.cursor, max(0, grid.item_count - 1))
        content_height = max(0, rect.height - PIPELINE_NON_CONTENT_ROWS)
        _scroll_to_cursor(tab, grid, content_height, focused=focused)
        baseline = tab.baseline_data if tab.baseline_key == grid.spec.key else None
        if baseline is not None and len(baseline) != len(grid.data):
            baseline = None
        first_line = tab.scroll
        last_line = min(len(grid.rows), first_line + content_height)
        drawing.scroll = (first_line, content_height, len(grid.rows))
        # The title is already full at the 42-column minimum, so the position
        # readout goes on the bottom border, beside the scroll thumb.
        shown = [first for first in grid.rows[first_line:last_line] if first is not None]
        visible_first = shown[0] if shown else 0
        visible_last = min(grid.item_count, shown[-1] + grid.per_line) if shown else 0
        readout = f" items {visible_first + 1}-{visible_last}/{grid.item_count} "
        canvas.top_pane_header(rect, [TITLE, grid.fmt], readout, focused=focused)
        lines = _Lines(canvas, session, tab, grid, baseline, self.theme.colors.cursor_for(focused))
        for output_row, line_index in enumerate(range(first_line, last_line)):
            drawing.hits.extend(self._draw_line(lines, rect, rect.y + 2 + output_row, line_index))
        return drawing

    def _draw_line(
        self, lines: _Lines, rect: Rect, y: int, line_index: int
    ) -> list[tuple[Rect, Any]]:
        """Draw display line *line_index* on row *y*; return where each value landed."""
        canvas, grid = lines.canvas, lines.grid
        colors = self.theme.colors
        content_x = rect.x + 1
        limit = rect.x + rect.width - 1 - SCROLLBAR_COLUMNS
        first_item = grid.rows[line_index]
        if first_item is None:
            canvas.add(y, content_x + 2, SLOT_SEPARATOR, colors.border, limit - content_x - 2)
            return []
        prefix = f"  {first_item * grid.item_size:04x}: "
        canvas.add(y, content_x, prefix, colors.border, limit - content_x)
        x = content_x + len(prefix)
        hits: list[tuple[Rect, Any]] = []
        for item_index in range(first_item, min(grid.item_count, first_item + grid.per_line)):
            hit_x = x
            x += item_spacing(lines.session.state, grid.fmt, item_index - first_item)
            available = max(0, min(grid.value_width, limit - x))
            if not available:
                break
            hits.append((Rect(y, hit_x, 1, x - hit_x + available), item_index))
            self._draw_value(lines, y, x, item_index, available)
            x += grid.value_width
        return hits

    def _draw_value(self, lines: _Lines, y: int, x: int, item_index: int, available: int) -> None:
        """Draw value *item_index* at ``(y, x)`` in at most *available* columns."""
        canvas, grid, colors = lines.canvas, lines.grid, self.theme.colors
        span = slice(item_index * grid.item_size, (item_index + 1) * grid.item_size)
        raw = grid.data[span]
        foreground: int | None = None
        background: int | None = None
        if grid.fmt == "cell16":
            cell = struct.unpack("<H", raw)[0]
            text, foreground, background = decode_16bit_cell(
                cell & CELL16_CHARACTER_MASK,
                (cell >> CELL16_FOREGROUND_SHIFT) & CELL16_COLOR_MASK,
                (cell >> CELL16_BACKGROUND_SHIFT) & CELL16_COLOR_MASK,
            )
        else:
            text = format_value(raw, grid.fmt, grid.value_width)
        colored = foreground is not None and background is not None
        if foreground is not None and background is not None:
            attr = colors.cell(foreground, background)
        else:
            attr = colors.value(not any(raw))
        if lines.baseline is not None and raw != lines.baseline[span]:
            attr |= colors.cell_changed if colored else colors.changed
        if item_index == lines.tab.cursor:
            attr = overlay_style(attr, colors.cell_selected if colored else lines.cursor)
        if colored:
            canvas.add(y, x, text, attr, available)
        elif available < len(text):
            marker = self.theme.glyphs.ellipsis
            if len(marker) > available:
                marker = ">"
            canvas.add(y, x, text[: available - len(marker)] + marker, attr, available)
        else:
            canvas.add_value(y, x, text, attr, available)

    # -- commands -------------------------------------------------------------

    def handle(
        self, command: Command, session: DebugSession, view: TabSet[PipelineTab], rect: Rect
    ) -> Reply | None:
        tab = view.current()
        assert tab is not None
        if command in (Command.ADD_TAB, Command.EDIT_TAB):
            return self._tab_editor(command, view)
        if command is Command.CYCLE_FORMAT:
            spec = pipeline_spec(tab.register_key)
            current = tab.fmt or auto_format(session.state, spec)
            formats = PIPELINE_REGISTER_FORMATS
            index = formats.index(current) if current in formats else -1
            tab.fmt = formats[(index + 1) % len(formats)]
            tab.scroll = 0
            tab.cursor = 0
            return Reply(f"{spec.label} format: {tab.fmt}")
        if command is Command.EDIT_VALUE:
            return self._value_editor(session, tab)
        page = max(1, rect.height - PIPELINE_NON_CONTENT_ROWS)
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
            grid = value_grid(session, tab, rect)
            tab.cursor = max(0, grid.item_count - 1) if command is Command.JUMP_LAST else 0
        return None

    def move(
        self,
        session: DebugSession,
        view: TabSet[PipelineTab],
        rect: Rect,
        *,
        dx: int = 0,
        dy: int = 0,
    ) -> None:
        """Move the cursor *dx* values and *dy* lines, stopping at the ends."""
        tab = view.current()
        assert tab is not None
        grid = value_grid(session, tab, rect)
        if grid.item_count == 0:
            tab.cursor = 0
            return
        movement = dx + dy * grid.per_line
        tab.cursor = max(0, min(grid.item_count - 1, tab.cursor + movement))

    def _value_editor(self, session: DebugSession, tab: PipelineTab) -> Reply:
        """Edit the value under the cursor, written back with ``write_register_bytes``."""
        spec = pipeline_spec(tab.register_key)
        fmt = tab.fmt or auto_format(session.state, spec)
        size, _ = value_geometry(fmt)
        name = backing(session.state, spec)
        register = read_register(session, spec)
        tab.cursor = max(0, min(tab.cursor, len(register) // size - 1))
        start = tab.cursor * size
        offset = spec.index * len(register) + start

        def write(data: bytes) -> None:
            session.write_register_bytes(name, offset, data)

        value = EditedValue(
            f"{spec.label}[{tab.cursor}]", fmt, register[start : start + size], size
        )
        return value_editor(session, self.id, value, write)

    def _tab_editor(self, command: Command, view: TabSet[PipelineTab]) -> Reply:
        tab = view.current()
        mode = "edit" if command is Command.EDIT_TAB else "add"
        text = PIPELINE_REGISTERS[0].label
        if mode == "edit" and tab is not None:
            text = pipeline_spec(tab.register_key).label

        def parse(text: str, existing: Any) -> tuple[PipelineTab | None, str | None]:
            token = text.strip().lower()
            spec = next(
                (
                    candidate
                    for candidate in PIPELINE_REGISTERS
                    if token in (candidate.key.lower(), candidate.label.lower())
                ),
                None,
            )
            if spec is None:
                choices = ", ".join(item.label for item in PIPELINE_REGISTERS)
                return None, f"expected one of: {choices}"
            fmt = existing.fmt if isinstance(existing, PipelineTab) else None
            return PipelineTab(register_key=spec.key, fmt=fmt), None

        return tab_editor(self.id, TabTarget("Pipeline", EDITOR_HINTS, parse), mode, view, text)

    # -- pointer --------------------------------------------------------------

    def place_cursor(self, session: DebugSession, view: TabSet[PipelineTab], payload: Any) -> None:
        del session
        tab = view.current()
        if tab is not None:
            tab.cursor = payload

    def cursor_row(self, session: DebugSession, view: TabSet[PipelineTab], rect: Rect) -> int:
        tab = view.current()
        assert tab is not None
        return value_grid(session, tab, rect).line_of(tab.cursor)

    def scroll_to_row(
        self, session: DebugSession, view: TabSet[PipelineTab], rect: Rect, row: int
    ) -> None:
        """Put the cursor on the first value of display line *row*."""
        tab = view.current()
        assert tab is not None
        grid = value_grid(session, tab, rect)
        row = min(row, len(grid.rows) - 1)
        if grid.rows[row] is None:
            row = min(row + 1, len(grid.rows) - 1)
        first = grid.rows[row]
        assert first is not None  # A separator line is always followed by values.
        tab.cursor = max(0, min(grid.item_count - 1, first))


def _scroll_to_cursor(
    tab: PipelineTab, grid: ValueGrid, content_height: int, *, focused: bool
) -> None:
    """Scroll so the cursor's line is on screen (only when focused), then clamp."""
    if focused and content_height > 0:
        cursor_line = grid.line_of(tab.cursor)
        if cursor_line < tab.scroll:
            tab.scroll = cursor_line
        elif cursor_line >= tab.scroll + content_height:
            tab.scroll = cursor_line - content_height + 1
    tab.scroll = min(tab.scroll, max(0, len(grid.rows) - content_height))
