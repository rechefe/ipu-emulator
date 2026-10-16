"""The TUI's controller and renderer, over the ``Screen`` seam.

The TUI is the application debugger. The deprecated line debugger is a
separate compatibility API and cannot be entered from this interface. The
terminal is reached only through ``ipu_debug.tui.screen``; keys reach their
handlers through ``ipu_debug.tui.keymap``.
"""

from __future__ import annotations

import re
import struct
import textwrap
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ipu_debug.control import DebugControl
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
from ipu_debug.pipeline import (
    SLOT_LABELS,
    display_operation,
    pipeline_occupancy,
    semantic_operations,
)
from ipu_debug.session import DebugSession
from ipu_debug.tui.draw import (
    BITS_PER_REGISTER,
    PANE_CONTENT_INDENT,
    PANE_CONTENT_MARGIN,
    SCROLLBAR_COLUMNS,
    Canvas,
    Glyphs,
    Palette,
    Theme,
    detect_glyphs,
    fit_segments,
    format_bits,
)
from ipu_debug.tui.editor import (
    EditedValue,
    TabTarget,
    scalar_editor,
    tab_editor,
    value_editor,
)
from ipu_debug.tui.keymap import (
    KEY_BINDINGS,
    PANE_COMMANDS,
    PANE_FOCUS,
    PANE_FOCUS_ORDER,
    Command,
    Focus,
    PaneId,
    footer_actions,
    footer_text,
    help_lines,
    lookup,
)
from ipu_debug.tui.pane import (
    Editor,
    LegacyPane,
    Line,
    LineRecorder,
    Outcome,
    Pane,
    Rect,
    Reply,
)
from ipu_debug.tui.panes import xmem as xmem_pane
from ipu_debug.tui.panes.xmem import XMEM_NON_CONTENT_ROWS, XmemPane, XmemTab
from ipu_debug.tui.screen import (
    Input,
    Key,
    MouseEvent,
    Screen,
    overlay_style,
    run_in_terminal,
    terminal_available,
)
from ipu_debug.tui.tabs import TabSet
from ipu_emu.descriptors import REGFILE_SCHEMA
from ipu_emu.emulator import DebugAction
from ipu_emu.ipu import xmem_row_size_bytes
from ipu_emu.ipu_state import INST_MEM_SIZE, IpuState

MIN_TERMINAL_ROWS = 24
MIN_TERMINAL_COLUMNS = 80
MIN_COMPACT_ROWS = 12
MIN_COMPACT_COLUMNS = 48
DOUBLE_CLICK_SECONDS = 0.35
MAX_TAB_LABEL_WIDTH = 24
HEADER_ROWS = 1
MIN_DISASSEMBLY_COLUMNS = 30
MIN_XMEM_COLUMNS = 38
MIN_PIPELINE_COLUMNS = 42
# Every pane spends its first row on a border, its second on a tab bar and its
# last on a border; the disassembly and register panes spend one more on a
# column header.  The constants keep the drawing code and the cursor arithmetic
# reading from the same numbers.
PIPELINE_NON_CONTENT_ROWS = 3
DISASSEMBLY_NON_CONTENT_ROWS = 4
REGISTER_NON_CONTENT_ROWS = 4
_SCALAR_REGISTER_COUNTS = {
    descriptor.name: descriptor.count
    for descriptor in REGFILE_SCHEMA
    if descriptor.name in ("lr", "cr")
}
_REGISTERS_PER_GROUP = max(_SCALAR_REGISTER_COUNTS.values())
# The register grid reflows to the height it is given: every shape divides the
# 16 registers of a group exactly, so no column is ever ragged.  The tallest
# shape that fits wins, because it also needs the fewest columns.
REGISTER_GRID_ROWS = (16, 8, 4)
DEFAULT_REGISTER_ROWS = 8
TOP_PANE_ROWS = DEFAULT_REGISTER_ROWS + REGISTER_NON_CONTENT_ROWS - 1
MIN_TOP_PANE_ROWS = min(REGISTER_GRID_ROWS) + REGISTER_NON_CONTENT_ROWS
MIN_MEMORY_ROWS = 6
MIN_REGISTER_COLUMNS = 20
PANE_RESIZE_STEP = 2
# One wheel notch moves this many lines, the usual terminal convention.
MOUSE_WHEEL_LINES = 3
# An upper bound on one coalescing pass, so a stuck terminal cannot starve the
# redraw entirely.
MAX_COALESCED_KEYS = 64
FOOTER_ROWS = 1
EDITOR_HEIGHT = 7
REGISTER_LABEL_WIDTH = len("L00 ")
REGISTER_MIN_GUTTER = 2
REGISTER_MAX_GUTTER = 4
REGISTER_GROUPS = ("lr", "cr")
PIPELINE_PAGE_FRACTION_NUMERATOR = 2
PIPELINE_PAGE_FRACTION_DENOMINATOR = 5
EMPTY_DISASSEMBLY_OPERATIONS = ("<nop>", "<NOP>", "NOP")
# Parallel slots are stacked under their instruction and marked the way VLIW
# assembly listings mark them, rather than run out across the pane.
DISASSEMBLY_PARALLEL_MARKER = "||"
DISASSEMBLY_PREFIX_WIDTH = len("> 0000  ")
# Panes paint in this order, so a later one wins any shared cell.
PANE_DRAW_ORDER: tuple[PaneId, ...] = ("disassembly", "registers", "xmem", "pipeline")
PANE_TITLES = {
    "disassembly": "Disassembly",
    "registers": "LR / CR",
    "xmem": "XMEM",
    "pipeline": "Pipeline register",
}
SCALAR_REGISTER_FORMATS = ("hex", "u32", "int32", "f32", "bits")
DISASSEMBLY_FORMATS = ("compact", "full", "stages")
REGISTER_TAB_GROUPS = ("all", "lr", "cr")
PIPELINE_REGISTER_FORMATS = (
    "hex",
    "int8",
    "u8",
    "cell16",
    "u32",
    "int32",
    "f32",
    "bits",
)
_SCALAR_FORMAT_VALUE_WIDTHS = {
    "hex": 8,
    "u32": 10,
    "int32": 11,
    "f32": F32_DISPLAY_WIDTH,
    "bits": BITS_PER_REGISTER,
}


# Border cells are accumulated as a bit mask of the line segments that leave
# them, so two panes sharing an edge resolve to one junction character instead
# of overwriting each other.
BORDER_UP = 1
BORDER_RIGHT = 2
BORDER_DOWN = 4
BORDER_LEFT = 8


def register_grid_rows(content_rows: int) -> int:
    """The tallest register-grid shape that fits in *content_rows*."""
    for rows in REGISTER_GRID_ROWS:
        if rows <= content_rows:
            return rows
    return min(REGISTER_GRID_ROWS)


FOOTER_PANE_LABELS = {
    "disassembly": "Disasm",
    "registers": "Regs",
    "xmem": "XMEM",
    "pipeline": "Pipe",
}


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
_PIPELINE_WORD_REGISTERS = frozenset({"r_acc", "mult_res"})
_PIPELINE_WIDE_VALUE_REGISTERS = frozenset({"r0", "r1", "r_cyclic"} | _PIPELINE_WORD_REGISTERS)


@dataclass
class DisassemblyTab:
    """One current-PC or fixed-PC disassembly view."""

    target: int | None = None
    fmt: str = "compact"
    cursor_offset: int = 0

    @property
    def title(self) -> str:
        target = "current" if self.target is None else str(self.target)
        return f"{target} {self.fmt}"


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
        spec = next(spec for spec in PIPELINE_REGISTERS if spec.key == self.register_key)
        return f"{spec.label} {self.fmt or 'auto'}"


def _new_panes() -> dict[str, TabSet[Any]]:
    """One empty tab set per pane; closing a pane's last tab reopens a default
    one, except in XMEM, which may be empty."""
    return {
        "pipeline": TabSet(default=PipelineTab),
        "disassembly": TabSet(default=DisassemblyTab),
        "xmem": TabSet(),
        "registers": TabSet(default=RegisterTab),
    }


@dataclass
class DebugViewSession:
    """State retained between debugger callbacks for one IPU state."""

    active: bool = False
    initialized: bool = False
    panes: dict[str, TabSet[Any]] = field(default_factory=_new_panes)
    focus: str = "xmem"
    message: str = "Ready"
    message_is_error: bool = False
    # Preferred split sizes survive temporary shrinking of the terminal.
    register_pane_columns: int | None = None
    pipeline_pane_columns: int | None = None
    top_pane_rows: int | None = None
    maximized: bool = False
    stages_scroll: int = 0

    @classmethod
    def with_tabs(
        cls,
        *,
        xmem: list[XmemTab] | None = None,
        disassembly: list[DisassemblyTab] | None = None,
        registers: list[RegisterTab] | None = None,
        pipeline: list[PipelineTab] | None = None,
        **fields: Any,
    ) -> DebugViewSession:
        """A session whose panes start with the given tabs."""
        session = cls(**fields)
        for pane, tabs in (
            ("xmem", xmem),
            ("disassembly", disassembly),
            ("registers", registers),
            ("pipeline", pipeline),
        ):
            session.panes[pane].tabs.extend(tabs or [])
        return session

    # Typed views of ``panes``.
    @property
    def xmem(self) -> TabSet[XmemTab]:
        return self.panes["xmem"]

    @property
    def disassembly(self) -> TabSet[DisassemblyTab]:
        return self.panes["disassembly"]

    @property
    def registers(self) -> TabSet[RegisterTab]:
        return self.panes["registers"]

    @property
    def pipeline(self) -> TabSet[PipelineTab]:
        return self.panes["pipeline"]

    def disassembly_tab(self) -> DisassemblyTab:
        """The active disassembly tab (the pane always has one)."""
        tab = self.disassembly.current()
        assert tab is not None
        return tab

    def register_tab(self) -> RegisterTab:
        """The active LR / CR tab (the pane always has one)."""
        tab = self.registers.current()
        assert tab is not None
        return tab

    def pipeline_tab(self) -> PipelineTab:
        """The active pipeline-register tab (the pane always has one)."""
        tab = self.pipeline.current()
        assert tab is not None
        return tab

    @property
    def stages_visible(self) -> bool:
        return self.disassembly_tab().fmt == "stages"

    @stages_visible.setter
    def stages_visible(self, visible: bool) -> None:
        self.disassembly_tab().fmt = "stages" if visible else "compact"

    def set_message(self, text: str, *, error: bool = False) -> None:
        self.message = text.replace("\n", " ").strip() or "Ready"
        self.message_is_error = error

    def ensure_default_tab(self, state: IpuState) -> None:
        fresh = not self.initialized
        if fresh and not self.xmem.tabs:
            self.xmem.tabs.extend(xmem_pane.default_tabs(state))
        if not self.disassembly.tabs:
            self.disassembly.tabs.extend(
                DisassemblyTab(fmt=fmt) for fmt in (DISASSEMBLY_FORMATS if fresh else ("compact",))
            )
        if not self.registers.tabs:
            self.registers.tabs.extend(
                RegisterTab(group=group) for group in (("all", "lr", "cr") if fresh else ("all",))
            )
        if not self.pipeline.tabs:
            self.pipeline.tabs.extend(
                PipelineTab(register_key=spec.key, fmt=_default_pipeline_format(state, spec))
                for spec in (PIPELINE_REGISTERS if fresh else PIPELINE_REGISTERS[:1])
            )
        self.initialized = True

    def snapshot_registers(self, state: IpuState) -> dict[tuple[str, int], int]:
        snapshot: dict[tuple[str, int], int] = {}
        for name in ("lr", "cr"):
            for index in range(_SCALAR_REGISTER_COUNTS[name]):
                snapshot[(name, index)] = state.regfile.get_scalar(name, index)
        return snapshot

    def commit_baseline(self, state: IpuState) -> None:
        debug = DebugSession(state)
        register_snapshot = self.snapshot_registers(state)
        for tab in self.registers:
            tab.baseline = register_snapshot.copy()
        for tab in self.pipeline:
            spec = _pipeline_spec(tab.register_key)
            tab.baseline_key = spec.key
            tab.baseline_data = _read_pipeline_register(state, spec)
        xmem_pane.commit_baselines(debug, self.xmem)


def new_debug_view_session(state: IpuState) -> DebugViewSession:
    """A TUI session for *state*, with its default XMEM tab. Whoever runs the
    debugger keeps it and passes it to every stop, so the view persists."""
    session = DebugViewSession()
    session.ensure_default_tab(state)
    return session


def _read_pipeline_register(
    state: IpuState,
    spec: PipelineRegisterSpec,
) -> bytes:
    backing = (
        spec.wide_backing
        if state.wide_vector_debug and spec.wide_backing is not None
        else spec.backing
    )
    value = DebugSession(state).register(backing, spec.index).value
    assert isinstance(value, bytes)
    return value


def _pipeline_format(state: IpuState, spec: PipelineRegisterSpec) -> str:
    if state.wide_vector_debug and spec.key in _PIPELINE_WIDE_VALUE_REGISTERS:
        arithmetic = state.wide_vector_arithmetic.value
        return "f32" if arithmetic == "fp32" else arithmetic
    if spec.key in _PIPELINE_WORD_REGISTERS:
        return "int32"
    return "hex"


def _default_pipeline_format(state: IpuState, spec: PipelineRegisterSpec) -> str:
    """Choose an explicit initial view without changing existing auto tabs."""
    if spec.key == "r_mask":
        return "bits"
    if spec.key == "post_aaq_reg":
        if state.wide_vector_debug and not state.wide_vector_quantize_output:
            return "f32" if state.wide_vector_arithmetic.value == "fp32" else "int32"
        return "int8"
    if (
        not state.wide_vector_debug
        and state.dtype.value == 0
        and spec.key in ("r0", "r1", "r_cyclic")
    ):
        return "int8"
    return _pipeline_format(state, spec)


def _pipeline_spec(register_key: str) -> PipelineRegisterSpec:
    return next(spec for spec in PIPELINE_REGISTERS if spec.key == register_key)


class CursesDebugView:
    """Controller and renderer for one debugger stop, over a :class:`Screen`."""

    def __init__(
        self,
        screen: Screen,
        state: IpuState,
        session: DebugViewSession,
        cycle: int,
        control: DebugControl | None = None,
        debug: DebugSession | None = None,
    ) -> None:
        self.screen = screen
        self.state = state
        # The run's control and session, from whoever started the run; else
        # this view's own.
        if debug is not None:
            self.control = debug.control
            self.debug = debug
        else:
            self.control = control if control is not None else DebugControl()
            self.debug = DebugSession(state, self.control)
        self.session = session
        self.cycle = cycle
        # Shared with the pane modules, which draw with the same styles.
        self.theme = Theme(Palette(screen), detect_glyphs())
        # ``(y, x) -> (segment mask, attribute, priority)`` for the pane frames.
        self.borders: dict[tuple[int, int], tuple[int, int, int]] = {}
        self.command = ""
        self.cursor = 0
        # The open editor, as the pane that asked for it described it.
        self.editor: Editor | None = None
        self.editor_mode: str | None = None
        self.editor_pane: str | None = None
        # The scalar being edited, as ``DebugSession.set_register`` names it.
        self.editor_register: str | None = None
        # Overlay state is per stop, so help never survives a step.
        self.help_visible = False
        self.help_scroll = 0
        self.help_track = None
        self.help_thumb = None
        self.help_drag = None
        self.help_max_scroll = 0
        self.consume_mouse_release = False
        self.scroll_state: dict[str, tuple[int, int, int]] = {}
        self.tab_hits: dict[str, list[tuple[Rect, int]]] = {}
        self.tab_close_hits: dict[str, list[tuple[Rect, int]]] = {}
        self.add_hits: dict[str, Rect] = {}
        # Every drawn value records where it landed, so a click can put the
        # cursor on it without the handler re-deriving the pane's geometry.
        self.value_hits: dict[str, list[tuple[Rect, Any]]] = {}
        self.scrollbar_hits: dict[str, Rect] = {}
        self.action_hits: list[tuple[Rect, Any]] = []
        self.tab_scroll_hits: list[tuple[Rect, str, int]] = []
        self.overlay_hits: list[tuple[Rect, Any]] = []
        self.editor_field: tuple[Rect, int] | None = None
        self.stage_close_hit: Rect | None = None
        self.scroll_drag: str | None = None
        self.scroll_origin: tuple[int, float] | None = None
        self.scroll_drag_moved = False
        self.scroll_thumb_hits: dict[str, Rect] = {}
        self.last_value_click: tuple[int, int, float] | None = None
        # ``(split, origin_y, origin_x, moved)`` while a border is held.
        self.drag: tuple[str, int, int, bool] | None = None
        # Decoding an instruction is the single most expensive thing a render
        # does, and instruction memory cannot change while the debugger is
        # stopped, so each PC is decoded at most once per stop.
        self._disassembly_cache: dict[int, str] = {}
        self.redraw = True
        self.disassembly_rect = Rect(0, 0, 0, 0)
        self.registers_rect = Rect(0, 0, 0, 0)
        self.xmem_rect = Rect(0, 0, 0, 0)
        self.pipeline_rect = Rect(0, 0, 0, 0)
        self.editor_rect = Rect(0, 0, 0, 0)
        # Where ``_add`` sends text: the screen, or a pane's line recorder.
        self._sink: Callable[[int, int, str, int], None] = self.screen.draw
        self.panes: dict[str, Pane] = self._legacy_panes()
        self.panes["xmem"] = XmemPane(self.theme)
        self._commands = self._command_table()

    @property
    def colors(self) -> Palette:
        return self.theme.colors

    @colors.setter
    def colors(self, colors: Palette) -> None:
        self.theme.colors = colors

    @property
    def glyphs(self) -> Glyphs:
        return self.theme.glyphs

    @glyphs.setter
    def glyphs(self, glyphs: Glyphs) -> None:
        self.theme.glyphs = glyphs

    def run(self) -> DebugAction:
        while True:
            if self.screen.resized():
                self.redraw = True
                self._cancel_gestures()
            if self.redraw:
                self.render()
                self.redraw = False
            try:
                key = self.screen.read_key()
                # Pointer motion arrives far faster than a frame can be drawn,
                # so everything already queued is handled before the redraw.
                handled = 0
                while key is not None:
                    outcome = self.handle_key(key)
                    if outcome is not None:
                        return outcome
                    handled += 1
                    key = self.screen.pending_key() if handled <= MAX_COALESCED_KEYS else None
            except KeyboardInterrupt:
                self.session.active = False
                return DebugAction.QUIT

    def _cancel_gestures(self) -> None:
        """Drop pointer gestures in flight."""
        self.drag = None
        self.scroll_drag = None
        self.scroll_origin = None
        self.last_value_click = None

    def _natural_register_columns(self, top_rows: int) -> int:
        """The width the active LR/CR tab needs at the grid shape it will use."""
        tab = self.session.register_tab()
        grid_rows = register_grid_rows(max(1, top_rows - REGISTER_NON_CONTENT_ROWS))
        groups = self._register_groups(tab)
        columns = len(groups) * _REGISTERS_PER_GROUP // grid_rows
        column_width = REGISTER_LABEL_WIDTH + _SCALAR_FORMAT_VALUE_WIDTHS[tab.fmt]
        return (
            columns * column_width
            + (columns - 1) * REGISTER_MAX_GUTTER
            + 2
            + PANE_CONTENT_INDENT
            + PANE_CONTENT_MARGIN
        )

    def _clamped_register_columns(self, columns: int, top_rows: int) -> int:
        """Clamp the visible width without discarding the preferred size.

        Without an override the LR/CR grid takes the width its current shape
        needs but never more than half the terminal, so a narrow terminal keeps
        the disassembly readable and the grid scrolls its columns instead.  A
        taller pane therefore also returns width to the disassembly.
        """
        requested = self.session.register_pane_columns or min(
            max(self._natural_register_columns(top_rows), columns * 2 // 5),
            max(MIN_REGISTER_COLUMNS, columns // 2),
        )
        width = max(
            MIN_REGISTER_COLUMNS,
            min(requested, columns - MIN_XMEM_COLUMNS),
        )
        return width

    def _clamped_pipeline_columns(self, columns: int) -> int:
        """Clamp the visible pipeline width without changing its preference."""
        requested = self.session.pipeline_pane_columns or max(
            MIN_PIPELINE_COLUMNS,
            columns // 2,
        )
        width = max(
            MIN_PIPELINE_COLUMNS,
            min(requested, columns - MIN_DISASSEMBLY_COLUMNS + 1),
        )
        return width

    @staticmethod
    def _default_top_pane_rows(rows: int) -> int:
        """Split usable rows evenly, including the shared border."""
        return max(MIN_TOP_PANE_ROWS, (rows - HEADER_ROWS - FOOTER_ROWS + 1) // 2)

    def _clamped_top_pane_rows(self, rows: int) -> int:
        """Clamp the visible height without changing its preference."""
        requested = self.session.top_pane_rows or self._default_top_pane_rows(rows)
        available = rows - HEADER_ROWS - FOOTER_ROWS - MIN_MEMORY_ROWS + 1
        height = max(MIN_TOP_PANE_ROWS, min(requested, available))
        return height

    def _resize_vertical_split(self, delta: int) -> None:
        """Move the focused row divider by delta columns (positive is right)."""
        if self._single_pane():
            return
        rows, columns = self.screen.size()
        top_rows = rows - HEADER_ROWS - FOOTER_ROWS - self._clamped_top_pane_rows(rows) + 1
        if self.session.focus in ("disassembly", "pipeline"):
            current = self._clamped_pipeline_columns(columns)
            self.session.pipeline_pane_columns = current + delta
            width = self._clamped_pipeline_columns(columns)
            self.session.pipeline_pane_columns = width
            self.session.set_message(f"Pipeline pane width: {width} columns")
            return
        current = self._clamped_register_columns(columns, top_rows)
        self.session.register_pane_columns = current - delta
        width = self._clamped_register_columns(columns, top_rows)
        self.session.register_pane_columns = width
        self.session.set_message(f"LR / CR pane width: {width} columns")

    def _resize_horizontal_split(self, delta: int) -> None:
        if self._single_pane():
            return
        rows, _ = self.screen.size()
        current = self._clamped_top_pane_rows(rows)
        self.session.top_pane_rows = current + delta
        height = self._clamped_top_pane_rows(rows)
        self.session.top_pane_rows = height
        self.session.set_message(f"Top pane height: {height} rows")

    def _reset_splits(self) -> None:
        self.session.register_pane_columns = None
        self.session.pipeline_pane_columns = None
        self.session.top_pane_rows = None
        self.session.set_message("Pane sizes reset")

    def _single_pane(self) -> bool:
        rows, columns = self.screen.size()
        return self.session.maximized or rows < MIN_TERMINAL_ROWS or columns < MIN_TERMINAL_COLUMNS

    def layout(self) -> dict[str, Rect] | None:
        rows, columns = self.screen.size()
        if rows < MIN_COMPACT_ROWS or columns < MIN_COMPACT_COLUMNS:
            return None
        if self._single_pane():
            return {
                "header": Rect(0, 0, HEADER_ROWS, columns),
                self.session.focus: Rect(HEADER_ROWS, 0, rows - HEADER_ROWS - FOOTER_ROWS, columns),
                "status": Rect(rows - 1, 0, 1, columns),
                "keys": Rect(rows - 1, 0, 1, columns),
            }
        top_y = HEADER_ROWS
        top_rows = self._clamped_top_pane_rows(rows)
        xmem_y = top_y + top_rows - 1
        xmem_bottom = rows - FOOTER_ROWS
        memory_rows = xmem_bottom - xmem_y
        register_width = self._clamped_register_columns(columns, memory_rows)
        pipeline_width = self._clamped_pipeline_columns(columns)
        layout = {
            "header": Rect(0, 0, HEADER_ROWS, columns),
            "pipeline": Rect(top_y, 0, top_rows, pipeline_width),
            "disassembly": Rect(top_y, pipeline_width - 1, top_rows, columns - pipeline_width + 1),
            "xmem": Rect(xmem_y, 0, memory_rows, columns - register_width + 1),
            "registers": Rect(xmem_y, columns - register_width, memory_rows, register_width),
            "status": Rect(rows - 1, 0, 1, columns),
            "keys": Rect(rows - 1, 0, 1, columns),
        }
        return layout

    def _update_pane_rects(self, layout: dict[str, Rect] | None) -> None:
        for pane in PANE_FOCUS_ORDER:
            rect = (layout or {}).get(pane, Rect(0, 0, 0, 0))
            setattr(self, f"{pane}_rect", rect)

    def render(self) -> None:
        self.screen.clear()
        self.tab_hits.clear()
        self.tab_close_hits.clear()
        self.add_hits.clear()
        self.value_hits.clear()
        self.scrollbar_hits.clear()
        self.scroll_thumb_hits.clear()
        self.scroll_state.clear()
        self.action_hits.clear()
        self.tab_scroll_hits.clear()
        self.overlay_hits.clear()
        self.help_track = None
        self.help_thumb = None
        self.editor_field = None
        self.stage_close_hit = None
        self.borders.clear()
        layout = self.layout()
        self._update_pane_rects(layout)
        if layout is None:
            self.drag = None
            self.scroll_drag = None
            rows, columns = self.screen.size()
            message = (
                f"Terminal must be at least {MIN_COMPACT_COLUMNS}x"
                f"{MIN_COMPACT_ROWS}; current size is {columns}x{rows}."
            )
            self._add(max(0, rows // 2), max(0, (columns - len(message)) // 2), message)
            self._add(
                max(0, rows // 2 + 1),
                0,
                "Resize the terminal or press q to quit.",
            )
            self._refresh()
            self.redraw = False
            return

        # Adjacent panes share a border column or row, so every frame is
        # recorded first and painted once, as a single grid of junctions.
        for pane in PANE_FOCUS_ORDER:
            if pane in layout:
                self._box(layout[pane], focused=self.session.focus == pane)
        self._flush_borders()
        self._draw_header(layout["header"])
        for pane_id in PANE_DRAW_ORDER:
            if pane_id in layout:
                self._draw_tab_bar(pane_id, layout[pane_id])
                self._draw_pane(pane_id, layout[pane_id])
        self._draw_scrollbars()
        self._draw_keys(layout["keys"])
        if self.help_visible:
            self._draw_help()
        elif self.editor_mode:
            self._draw_editor()
        else:
            self.screen.cursor(False)
        self._refresh()
        self.redraw = False

    def _refresh(self) -> None:
        self.screen.refresh()

    def _draw_pane(self, pane_id: str, rect: Rect) -> None:
        """Paint one pane's content and keep what it reports for the pointer."""
        drawing = self.panes[pane_id].render(
            self.debug, self.session.panes[pane_id], rect, focused=self.session.focus == pane_id
        )
        for line in drawing.lines:
            self.screen.draw(line.y, line.x, line.text, line.style)
        if drawing.hits:
            self.value_hits.setdefault(pane_id, []).extend(drawing.hits)
        if drawing.scroll is not None:
            self.scroll_state[pane_id] = drawing.scroll

    def _pane_rect(self, pane_id: str) -> Rect:
        rect: Rect = getattr(self, f"{pane_id}_rect")
        return rect

    def _legacy_panes(self) -> dict[str, Pane]:
        """Each pane, wrapping the drawing code that still lives here."""
        commands: dict[Command, Callable[[], Outcome]] = {
            Command.ADD_TAB: lambda: self._begin_tab_editor("add"),
            Command.EDIT_TAB: lambda: self._begin_tab_editor("edit"),
            Command.CYCLE_FORMAT: self._cycle_focused_format,
            Command.EDIT_VALUE: self._edit_focused_value,
            Command.MOVE_UP: lambda: self._move_or_scroll_stages(dy=-1),
            Command.MOVE_DOWN: lambda: self._move_or_scroll_stages(dy=1),
            Command.MOVE_LEFT: lambda: self._move_focused_cursor(dx=-1),
            Command.MOVE_RIGHT: lambda: self._move_focused_cursor(dx=1),
            Command.PAGE_UP: lambda: self._page_focused_cursor(-1),
            Command.PAGE_DOWN: lambda: self._page_focused_cursor(1),
            Command.JUMP_FIRST: lambda: self._jump_or_scroll_stages(end=False),
            Command.JUMP_LAST: lambda: self._jump_or_scroll_stages(end=True),
        }
        draws: dict[str, Callable[[Rect], None]] = {
            "disassembly": self._draw_disassembly,
            "registers": self._draw_registers,
            "pipeline": self._draw_pipeline_register,
        }
        return {
            pane: LegacyPane(
                pane,
                draw,
                self._capture_lines,
                commands,
                place=lambda payload, pane=pane: self._legacy_place_cursor(pane, payload),
                row=lambda pane=pane: self._legacy_cursor_row(pane),
                scroll=lambda row, pane=pane: self._legacy_scroll_to_row(pane, row),
                step=lambda dx, dy: self._move_focused_cursor(dx=dx, dy=dy),
            )
            for pane, draw in draws.items()
        }

    def _capture_lines(self, draw: Callable[[Rect], None], rect: Rect) -> list[Line]:
        """Run *draw* with ``_add`` recording into a list instead of the screen."""
        recorder = LineRecorder()
        self._sink = recorder
        try:
            draw(rect)
        finally:
            self._sink = self.screen.draw
        return recorder.lines

    def _draw_header(self, rect: Rect) -> None:
        mode = "wide" if self.state.wide_vector_debug else "normal"
        segments = [
            self.debug.kernel_name or "IPU DEBUG",
            "PAUSED",
            f"PC={self.state.program_counter:04d}",
            f"cycle={self.cycle}",
            self.control.stop_reason or "paused",
            f"{mode} / {xmem_row_size_bytes(self.state)}B rows",
        ]
        message = "" if self.session.message == "Ready" else self.session.message
        if self.session.message_is_error:
            message = "! " + message
        message_width = min(len(message), max(0, (rect.width - 12) // 2))
        if len(message) > message_width:
            message = message[: max(0, message_width - 1)] + self.glyphs.ellipsis
        width = max(0, rect.width - 12 - (len(message) + 2 if message else 0))
        summary = self._fit_segments(segments[:4], width)
        separator = f" {self.glyphs.separator} "
        remaining = width - len(summary) - len(separator)
        if remaining > 0:
            summary += separator + self._fit_segments(segments[4:], remaining)
        status = " " + summary
        self._add(
            rect.y,
            rect.x,
            status.ljust(rect.width),
            self.colors.header,
            rect.width,
        )
        badge_x = status.find("PAUSED")
        if badge_x >= 0:
            self._add(rect.y, rect.x + badge_x, "PAUSED", self.colors.paused, 6)

        label = "[? Help]"
        x = rect.x + rect.width - len(label) - 1
        if message:
            self._add(
                rect.y,
                x - len(message) - 2,
                message,
                self.colors.error if self.session.message_is_error else self.colors.status,
                len(message),
            )
        self._add(rect.y, x, label, self.colors.key, len(label))
        self.action_hits.append((Rect(rect.y, x, 1, len(label)), "?"))

    def _draw_tab_bar(self, pane: str, rect: Rect) -> None:
        tab_set = self.session.panes[pane]
        tabs = tab_set.tabs
        active = tab_set.active
        offset = tab_set.offset

        def tab_label(tab: Any) -> str:
            title = tab.title
            # Keep even a long active tab visible in the narrowest pane.
            title_width = min(MAX_TAB_LABEL_WIDTH, rect.width - 8) - len("[ x]")
            if len(title) > title_width:
                ellipsis = self.glyphs.ellipsis
                title = title[: title_width - len(ellipsis)] + ellipsis
            return f"[{title} x]"

        tab_y = rect.y + 1
        x = rect.x + 1
        add_label = "[+]"
        add_attr = self.colors.current if pane == self.session.focus else self.colors.border
        self._add(tab_y, x, add_label, add_attr)
        self.add_hits[pane] = Rect(tab_y, x, 1, len(add_label))
        # The column after ``[+]`` always belongs to the "earlier tabs" marker,
        # so the run of tabs does not shift when the bar starts scrolling.
        marker_x = x + len(add_label)
        x = marker_x + 2
        # The right-most column carries the "more tabs" marker when the bar
        # overflows, so the visible run always stops one column short of it.
        available = rect.x + rect.width - 2
        available_width = max(0, available - x)
        if active < offset:
            offset = active
        while offset < active:
            active_span = sum(
                len(tab_label(tabs[index])) + 1 for index in range(offset, active + 1)
            )
            if active_span <= available_width:
                break
            offset += 1
        tab_set.offset = offset
        if offset:
            self._add(
                tab_y,
                marker_x,
                self.glyphs.more_before,
                self.colors.border,
                1,
            )
        if offset:
            self.tab_scroll_hits.append((Rect(tab_y, marker_x, 1, 1), pane, offset - 1))
        self.tab_hits[pane] = []
        self.tab_close_hits[pane] = []
        shown = 0
        for index in range(offset, len(tabs)):
            label = tab_label(tabs[index])
            remaining = available - x
            if remaining < len(label):
                break
            attr = self.colors.border
            if index == active:
                attr = self.colors.selected if pane == self.session.focus else self.colors.current
            self._add(tab_y, x, label, attr)
            hit = Rect(tab_y, x, 1, len(label))
            self.tab_hits[pane].append((hit, index))
            self.tab_close_hits[pane].append((Rect(tab_y, x + len(label) - 2, 1, 1), index))
            x += len(label) + 1
            shown += 1
        if offset + shown < len(tabs):
            self.tab_scroll_hits.append((Rect(tab_y, available, 1, 1), pane, offset + shown))
            self._add(
                tab_y,
                available,
                self.glyphs.more_after,
                self.colors.border,
                1,
            )

    def _disassembly(self, pc: int) -> str:
        text = self._disassembly_cache.get(pc)
        if text is None:
            line = self.debug.disassemble(pc, 1)
            text = f"PC {pc}: {line[0].text}" if line else "PC out of bounds"
            operations = self._all_operations(text)
            if len(operations) == len(SLOT_LABELS):
                text = f"PC {pc}: " + ";\n".join(semantic_operations(operations)) + ";;"
            self._disassembly_cache[pc] = text
        return text

    def _draw_disassembly(self, rect: Rect) -> None:
        tab = self.session.disassembly_tab()
        if tab.fmt == "stages":
            self._draw_stages(rect)
            return
        # One content column stays free for the scroll thumb.
        row_width = max(1, rect.width - 2 - SCROLLBAR_COLUMNS)
        header = f"{'PC':>6}  OPERATIONS"
        self._add(rect.y + 2, rect.x + 1, header, self.colors.border, row_width)
        if row_width >= 60:
            self._add(rect.y + 2, rect.x + row_width - 10, "      SLOT", self.colors.border, 10)
        visible = max(1, rect.height - DISASSEMBLY_NON_CONTENT_ROWS)
        base_pc = self.state.program_counter if tab.target is None else tab.target
        centered_pc = max(0, min(INST_MEM_SIZE - 1, base_pc + tab.cursor_offset))
        rows, first, last = self._disassembly_rows(tab, centered_pc, visible)
        mnemonic_width = max(
            (len(operation.split(" ", 1)[0]) for _, _, operation in rows),
            default=0,
        )
        # Small panes need room for operands more than vertical alignment.
        if row_width < 60:
            mnemonic_width = 0
        cursor_attr = self._cursor_attr("disassembly")
        slot_labels: dict[int, list[str]] = {}
        for row, (pc, slot, operation) in enumerate(rows):
            is_current = pc == self.state.program_counter
            is_empty = operation in EMPTY_DISASSEMBLY_OPERATIONS
            # The executing instruction is always marked; the reverse-video bar
            # belongs to the cursor, and covers every slot of that instruction.
            # Unwritten instruction slots recede so a program stands out.
            attr = self.colors.current if is_current else 0
            if is_empty and not is_current:
                attr = self.colors.border
            if pc == centered_pc:
                attr = overlay_style(attr, cursor_attr)
            y = rect.y + 3 + row
            x = rect.x + 1
            limit = x + row_width
            self.value_hits.setdefault("disassembly", []).append((Rect(y, x, 1, row_width), pc))
            if slot:
                # The parallel-slot marker recedes without breaking the bar
                # the cursor draws behind it.
                prefix = f"{'':{DISASSEMBLY_PREFIX_WIDTH - 3}}"
                self._add(y, x, prefix, attr, limit - x)
                self._add(
                    y,
                    x + len(prefix),
                    DISASSEMBLY_PARALLEL_MARKER,
                    overlay_style(self.colors.border, attr),
                    limit - x - len(prefix),
                )
                self._add(
                    y,
                    x + DISASSEMBLY_PREFIX_WIDTH - 1,
                    " ",
                    attr,
                    max(0, limit - x - DISASSEMBLY_PREFIX_WIDTH + 1),
                )
            else:
                marker = self.glyphs.current_row if is_current else self.glyphs.normal_row
                breakpoint = "B" if pc in self.control.breakpoints else " "
                self._add(y, x, f"{marker}{breakpoint}{pc:04d}  ", attr, limit - x)
                if is_current:
                    self._add(y, x, marker, self.colors.pc_marker, 1)
                if breakpoint == "B":
                    self._add(y, x + 1, breakpoint, self.colors.breakpoint, 1)
            x += DISASSEMBLY_PREFIX_WIDTH
            if row_width >= 60:
                limit -= 12
            mnemonic, _, operands = operation.partition(" ")
            text = f"{mnemonic.ljust(mnemonic_width)} {operands}" if operands else mnemonic
            available = max(0, limit - x)
            if len(text) > available and available >= len(self.glyphs.ellipsis):
                text = text[: available - len(self.glyphs.ellipsis)] + self.glyphs.ellipsis
            self._add(y, x, text.ljust(available), attr, available)
            if not is_empty and pc != centered_pc:
                self._add(y, x, mnemonic, self.colors.key, min(available, len(mnemonic)))
            if row_width >= 60:
                if pc not in slot_labels:
                    ops = self._all_operations(self._disassembly(pc))
                    slot_labels[pc] = (
                        [
                            name
                            for name, op in zip(SLOT_LABELS, ops, strict=False)
                            if tab.fmt == "full" or op.upper().split(" ", 1)[0] != "NOP"
                        ]
                        if self.state.inst_mem[pc] is not None
                        else []
                    )
                names = slot_labels[pc]
                label = names[slot] if slot < len(names) else ""
                self._add(y, limit, label.rjust(11) + " ", overlay_style(self.colors.key, attr), 12)
        self.scroll_state["disassembly"] = (
            first,
            last - first + 1,
            INST_MEM_SIZE,
        )
        self._draw_top_pane_header(
            rect,
            ["2 Instructions"],
            f" {first:04d}-{last:04d}/{INST_MEM_SIZE} ",
            focused=self.session.focus == "disassembly",
        )

    def _disassembly_rows(
        self,
        tab: DisassemblyTab,
        centered_pc: int,
        visible: int,
    ) -> tuple[list[tuple[int, int, str]], int, int]:
        """One row per operation, so no instruction spans the pane sideways.

        Returns the rows as ``(pc, slot, operation)`` along with the first and
        last program counter they cover.
        """

        def operations(pc: int) -> list[str]:
            disassembly = self._disassembly(pc)
            return (
                self._all_operations(disassembly)
                if tab.fmt == "full"
                else self._active_operations(disassembly)
            )

        # Walk back from the cursor's instruction while its slots still leave
        # the cursor on screen, so a wide instruction cannot push it off.
        first = centered_pc
        above = 0
        while first > 0 and above + len(operations(first - 1)) <= visible // 2:
            first -= 1
            above += len(operations(first))
        rows: list[tuple[int, int, str]] = []
        last = first
        for pc in range(first, INST_MEM_SIZE):
            if len(rows) >= visible:
                break
            last = pc
            for slot, operation in enumerate(operations(pc)):
                if len(rows) >= visible:
                    break
                rows.append((pc, slot, operation))
        return rows, first, last

    @staticmethod
    def _all_operations(disassembly: str) -> list[str]:
        lines = [line.strip().rstrip(";") for line in disassembly.splitlines()]
        if not lines:
            return ["<empty>"]
        _, separator, first_operation = lines[0].partition(":")
        operations = [first_operation.strip() if separator else lines[0]]
        operations.extend(lines[1:])
        return [operation for operation in operations if operation] or ["<empty>"]

    @classmethod
    def _active_operations(cls, disassembly: str) -> list[str]:
        operations = cls._all_operations(disassembly)
        active = [
            operation
            for operation in operations
            if operation and operation.upper().split(" ", 1)[0] != "NOP"
        ]
        return active or ["<NOP>"]

    def _draw_stages(self, rect: Rect) -> None:
        """Render current cycle operations inside a disassembly tab."""
        pc = self.state.program_counter
        instruction = self.state.inst_mem[pc] if 0 <= pc < INST_MEM_SIZE else None
        width = max(1, rect.width - 3)
        self._add(
            rect.y + 2,
            rect.x + 1,
            f"Pipeline stages / PC {pc:04d} / cycle {self.cycle} / ready",
            self.colors.border,
            width,
        )
        lines: list[tuple[str, int]] = []
        for stage, commands in pipeline_occupancy(instruction):
            if not commands:
                lines.append((f"{stage:<10} IDLE", self.colors.border))
            for index, (_slot, operation) in enumerate(commands):
                name = stage if index == 0 else ""
                text = f"{name:<10} {display_operation(operation)}"
                lines.extend(
                    (line, self.colors.key)
                    for line in textwrap.wrap(text, width, subsequent_indent="  ")
                )
        visible = max(1, rect.height - 4)
        self.session.stages_scroll = max(0, min(self.session.stages_scroll, len(lines) - visible))
        first = self.session.stages_scroll
        self._draw_top_pane_header(
            rect,
            ["2 Instructions", "Stages"],
            f" {first + 1}-{min(len(lines), first + visible)}/{len(lines)} ",
            focused=self.session.focus == "disassembly",
        )
        for index, (line, attr) in enumerate(lines[first : first + visible]):
            self._add(rect.y + 3 + index, rect.x + 1, line, attr, width)

    def _draw_registers(self, rect: Rect) -> None:
        tab = self.session.register_tab()
        groups = self._register_groups(tab)
        grid_rows = register_grid_rows(max(1, rect.height - REGISTER_NON_CONTENT_ROWS))
        column_count = len(groups) * _REGISTERS_PER_GROUP // grid_rows
        value_width = _SCALAR_FORMAT_VALUE_WIDTHS[tab.fmt]
        column_width = REGISTER_LABEL_WIDTH + value_width
        # The indent and margin are padding, not a budget: a wide format such
        # as ``bits`` gives them up rather than truncating a register value.
        inner_width = max(1, rect.width - 2)
        padded_width = max(
            1,
            inner_width - PANE_CONTENT_INDENT - PANE_CONTENT_MARGIN,
        )
        if padded_width < column_width <= inner_width:
            content_x = rect.x + 1
            content_width = inner_width
        else:
            content_x = rect.x + 1 + PANE_CONTENT_INDENT
            content_width = padded_width
        visible_columns = max(
            1,
            min(
                column_count,
                (content_width + REGISTER_MIN_GUTTER) // (column_width + REGISTER_MIN_GUTTER),
            ),
        )
        gaps = visible_columns - 1
        spare = content_width - visible_columns * column_width
        gutter = (
            REGISTER_MIN_GUTTER
            if gaps <= 0
            else max(REGISTER_MIN_GUTTER, min(REGISTER_MAX_GUTTER, spare // gaps))
        )
        stride = column_width + gutter

        tab.cursor = max(
            0,
            min(tab.cursor, len(groups) * _REGISTERS_PER_GROUP - 1),
        )
        cursor_column = tab.cursor // grid_rows
        if cursor_column < tab.column_scroll:
            tab.column_scroll = cursor_column
        elif cursor_column >= tab.column_scroll + visible_columns:
            tab.column_scroll = cursor_column - visible_columns + 1
        tab.column_scroll = max(
            0,
            min(tab.column_scroll, column_count - visible_columns),
        )

        limit = content_x + content_width
        for visible_column in range(visible_columns):
            column = tab.column_scroll + visible_column
            if column >= column_count:
                break
            x = content_x + visible_column * stride
            first = column * grid_rows
            name = groups[first // _REGISTERS_PER_GROUP]
            first_index = first % _REGISTERS_PER_GROUP
            self._add(
                rect.y + 2,
                x,
                f"{name.upper()} {first_index:02d}-{first_index + grid_rows - 1:02d}",
                self.colors.border,
                min(column_width, limit - x),
            )
            for row in range(grid_rows):
                index = first_index + row
                if index >= _SCALAR_REGISTER_COUNTS[name]:
                    continue
                self._draw_register_cell(
                    rect.y + 3 + row,
                    x,
                    min(column_width, limit - x),
                    name,
                    index,
                    tab,
                    first + row,
                    value_width,
                )
        readout = (
            f" cols {tab.column_scroll + 1}-{tab.column_scroll + visible_columns}/{column_count} "
            if column_count > visible_columns
            else ""
        )
        self._draw_top_pane_header(
            rect,
            ["4 CR / LR"],
            readout,
            focused=self.session.focus == "registers",
        )

    @staticmethod
    def _register_groups(tab: RegisterTab) -> tuple[str, ...]:
        return REGISTER_GROUPS if tab.group == "all" else (tab.group,)

    def _register_grid(self, tab: RegisterTab) -> tuple[int, int]:
        """The ``(rows, columns)`` shape the register pane is drawing now."""
        rows = register_grid_rows(max(1, self.registers_rect.height - REGISTER_NON_CONTENT_ROWS))
        groups = self._register_groups(tab)
        return rows, len(groups) * _REGISTERS_PER_GROUP // rows

    def _draw_register_cell(
        self,
        y: int,
        x: int,
        width: int,
        name: str,
        index: int,
        tab: RegisterTab,
        visual_index: int,
        value_width: int,
    ) -> None:
        if width <= 0:
            return
        value = self.state.regfile.get_scalar(name, index)
        changed = tab.baseline is not None and tab.baseline[(name, index)] != value
        label = f"{name[0].upper()}{index:02d} "
        label_attr = self.colors.border
        value_attr = self.colors.changed if changed else self._value_attr(not value)
        if tab.cursor == visual_index:
            cursor_attr = self._cursor_attr("registers")
            label_attr = overlay_style(label_attr, cursor_attr)
            value_attr = overlay_style(value_attr, cursor_attr)
        self.value_hits.setdefault("registers", []).append((Rect(y, x, 1, width), visual_index))
        self._add(y, x, label, label_attr, min(len(label), width))
        remaining = width - len(label)
        if remaining <= 0:
            return
        self._add_value(
            y,
            x + len(label),
            self._format_scalar_value(value, tab.fmt, value_width),
            value_attr,
            remaining,
        )

    def _cursor_attr(self, pane: str) -> int:
        """The cursor highlight for *pane*, muted while it is not focused."""
        return self.colors.cursor_for(self.session.focus == pane)

    def _value_attr(self, is_zero: bool) -> int:
        """Zero values recede, so the bytes that carry data stand out."""
        return self.colors.value(is_zero)

    def _add_value(self, y: int, x: int, text: str, attr: int, max_width: int) -> None:
        """Draw a numeric value with its leading zeros dimmed."""
        self._canvas().add_value(y, x, text, attr, max_width)

    @staticmethod
    def _format_scalar_value(value: int, fmt: str, width: int) -> str:
        unsigned = value & 0xFFFFFFFF
        if fmt == "hex":
            text = f"{unsigned:08x}"
        elif fmt == "u32":
            text = f"{unsigned:0{U32_DECIMAL_WIDTH}d}"
        elif fmt == "int32":
            text = str(struct.unpack("<i", struct.pack("<I", unsigned))[0])
        elif fmt == "f32":
            value = struct.unpack("<f", struct.pack("<I", unsigned))[0]
            text = f"{value:>{F32_DISPLAY_WIDTH}.{F32_SIGNIFICANT_DIGITS}g}"
        else:
            text = format_bits(unsigned)
        return text.rjust(width)[:width]

    def _draw_pipeline_register(self, rect: Rect) -> None:
        tab = self.session.pipeline_tab()
        spec = _pipeline_spec(tab.register_key)
        data = _read_pipeline_register(self.state, spec)
        fmt = tab.fmt or _pipeline_format(self.state, spec)

        content_y = rect.y + 2
        content_height = max(0, rect.height - PIPELINE_NON_CONTENT_ROWS)
        content_x = rect.x + 1
        limit = rect.x + rect.width - 1 - SCROLLBAR_COLUMNS
        item_size, value_width = self._pipeline_value_geometry(fmt)
        items_per_line = self._pipeline_items_per_line(fmt, rect)
        item_count = len(data) // item_size
        tab.cursor = min(
            tab.cursor,
            max(0, item_count - 1),
        )
        display_rows = self._pipeline_display_rows(fmt, rect, item_count)
        cursor_line = next(
            index
            for index, first in enumerate(display_rows)
            if first is not None and first <= tab.cursor < first + items_per_line
        )
        line_count = len(display_rows)
        max_scroll = max(0, line_count - content_height)
        if self.session.focus == "pipeline" and content_height > 0:
            if cursor_line < tab.scroll:
                tab.scroll = cursor_line
            elif cursor_line >= tab.scroll + content_height:
                tab.scroll = cursor_line - content_height + 1
        tab.scroll = min(
            tab.scroll,
            max_scroll,
        )
        baseline = tab.baseline_data if tab.baseline_key == spec.key else None
        if baseline is not None and len(baseline) != len(data):
            baseline = None

        first_line = tab.scroll
        last_line = min(line_count, first_line + content_height)
        self.scroll_state["pipeline"] = (first_line, content_height, line_count)
        # The title is already full at the 42-column minimum, so the position
        # readout goes on the bottom border, beside the scroll thumb.
        visible_items = [first for first in display_rows[first_line:last_line] if first is not None]
        visible_first = visible_items[0] if visible_items else 0
        visible_last = min(item_count, visible_items[-1] + items_per_line) if visible_items else 0
        self._draw_top_pane_header(
            rect,
            ["1 Pipeline Registers", fmt],
            f" items {visible_first + 1}-{visible_last}/{item_count} ",
            focused=self.session.focus == "pipeline",
        )
        for output_row, line_index in enumerate(range(first_line, last_line)):
            first_item = display_rows[line_index]
            y = content_y + output_row
            if first_item is None:
                self._add(
                    y, content_x + 2, "------------", self.colors.border, limit - content_x - 2
                )
                continue
            offset = first_item * item_size
            prefix = f"  {offset:04x}: "
            self._add(y, content_x, prefix, self.colors.border, limit - content_x)
            x = content_x + len(prefix)
            for item_index in range(
                first_item,
                min(item_count, first_item + items_per_line),
            ):
                item_in_line = item_index - first_item
                hit_x = x
                x += self._pipeline_item_spacing(fmt, item_in_line)
                available = max(0, min(value_width, limit - x))
                if not available:
                    break
                self.value_hits.setdefault("pipeline", []).append(
                    (Rect(y, hit_x, 1, x - hit_x + available), item_index)
                )
                raw_start = item_index * item_size
                raw = data[raw_start : raw_start + item_size]
                foreground: int | None = None
                background: int | None = None
                if fmt == "cell16":
                    cell = struct.unpack("<H", raw)[0]
                    text, foreground, background = decode_16bit_cell(
                        cell & CELL16_CHARACTER_MASK,
                        (cell >> CELL16_FOREGROUND_SHIFT) & CELL16_COLOR_MASK,
                        (cell >> CELL16_BACKGROUND_SHIFT) & CELL16_COLOR_MASK,
                    )
                else:
                    text = self._format_pipeline_value(raw, fmt, value_width)
                is_colored_cell = foreground is not None and background is not None
                changed = (
                    baseline is not None and raw != baseline[raw_start : raw_start + item_size]
                )
                selected = item_index == tab.cursor
                attr = (
                    self.colors.cell(foreground, background)
                    if is_colored_cell
                    else self._value_attr(not any(raw))
                )
                if changed:
                    attr |= self.colors.cell_changed if is_colored_cell else self.colors.changed
                if selected:
                    attr = overlay_style(
                        attr,
                        (
                            self.colors.cell_selected
                            if is_colored_cell
                            else self._cursor_attr("pipeline")
                        ),
                    )
                if is_colored_cell:
                    self._add(y, x, text, attr, available)
                elif available < len(text):
                    marker = self.glyphs.ellipsis
                    if len(marker) > available:
                        marker = ">"
                    clipped = text[: available - len(marker)] + marker
                    self._add(y, x, clipped, attr, available)
                else:
                    self._add_value(y, x, text, attr, available)
                x += value_width

    @staticmethod
    def _pipeline_value_geometry(fmt: str) -> tuple[int, int]:
        if fmt == "hex":
            return 1, 2
        if fmt == "u8":
            return 1, U8_DECIMAL_WIDTH
        if fmt == "int8":
            return 1, INT8_DECIMAL_WIDTH
        if fmt == "cell16":
            return 2, 1
        if fmt == "u32":
            return 4, U32_DECIMAL_WIDTH
        if fmt == "int32":
            return 4, 11
        if fmt == "bits":
            return 4, BITS_PER_REGISTER
        return 4, F32_DISPLAY_WIDTH

    def _pipeline_display_rows(self, fmt: str, rect: Rect, item_count: int) -> list[int | None]:
        per_line = self._pipeline_items_per_line(fmt, rect)
        size, _ = self._pipeline_value_geometry(fmt)
        mask = self.session.pipeline_tab().register_key == "r_mask"
        rows: list[int | None] = []
        for first in range(0, item_count, per_line):
            # R_MASK is eight independent 128-bit (16-byte) slots.
            if mask and first and first * size % 16 == 0:
                rows.append(None)
            rows.append(first)
        return rows

    def _pipeline_items_per_line(self, fmt: str, rect: Rect) -> int:
        _, value_width = self._pipeline_value_geometry(fmt)
        content_width = max(0, rect.width - 2 - SCROLLBAR_COLUMNS)
        prefix_width = len("  0000: ")
        available_width = max(1, content_width - prefix_width)
        used_width = 0
        item_count = 0
        while True:
            next_width = value_width + self._pipeline_item_spacing(
                fmt,
                item_count,
            )
            if item_count > 0 and used_width + next_width > available_width:
                break
            used_width += next_width
            item_count += 1
            if used_width >= available_width:
                break
        if self.session.pipeline_tab().register_key == "r_mask":
            size, _ = self._pipeline_value_geometry(fmt)
            slot_items = 16 // size
            item_count = max(
                count
                for count in range(1, slot_items + 1)
                if count <= max(1, item_count) and slot_items % count == 0
            )
        return max(1, item_count)

    def _pipeline_item_spacing(self, fmt: str, item_in_line: int) -> int:
        if item_in_line == 0:
            return 0
        if fmt == "cell16":
            return 1 if item_in_line % CELL16_CHARACTERS_PER_GROUP == 0 else 0
        if (
            fmt == "hex"
            and self.state.wide_vector_debug
            and item_in_line % HEX_BYTES_PER_GROUP == 0
        ):
            return 2
        return 1

    @staticmethod
    def _format_pipeline_value(raw: bytes, fmt: str, width: int) -> str:
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

    def _draw_keys(self, rect: Rect) -> None:
        text = footer_text(self.session.focus, rect.width)
        self._add(
            rect.y,
            rect.x,
            text.ljust(rect.width),
            self.colors.header,
            rect.width,
        )
        for match in re.finditer(r"(?<!\S)([^\s:]+)(?=:| Quit)", text):
            self._add(rect.y, rect.x + match.start(), match[0], self.colors.key, len(match[0]))
        for binding in KEY_BINDINGS:
            if not binding.footer or self.session.focus not in binding.panes:
                continue
            if text.find(binding.footer) < 0:
                continue
            for label, key in footer_actions(binding):
                start = text.find(label)
                self.action_hits.append((Rect(rect.y, rect.x + start, 1, len(label)), key))

    # Every vertically scrollable pane reserves its right-most content column
    # for a thumb, so the bar never lands on a border shared with a neighbour.
    _SCROLLBAR_PANES = {
        "disassembly": DISASSEMBLY_NON_CONTENT_ROWS,
        "xmem": XMEM_NON_CONTENT_ROWS,
        "pipeline": PIPELINE_NON_CONTENT_ROWS,
    }

    def _draw_scrollbars(self) -> None:
        """Paint every pane's scroll thumb once the panes have drawn."""
        rects = {
            "disassembly": self.disassembly_rect,
            "xmem": self.xmem_rect,
            "pipeline": self.pipeline_rect,
        }
        for pane, non_content_rows in self._SCROLLBAR_PANES.items():
            if pane not in self.scroll_state:
                continue
            first, visible, total = self.scroll_state[pane]
            self._draw_scroll_thumb(
                pane,
                rects[pane],
                first,
                visible,
                total,
                non_content_rows,
            )

    def _draw_scroll_thumb(
        self,
        pane: str,
        rect: Rect,
        first: int,
        visible: int,
        total: int,
        non_content_rows: int,
    ) -> None:
        # The track covers the content rows only, leaving the tab bar alone.
        track = rect.height - non_content_rows
        if total <= visible or visible <= 0 or track < 2:
            return
        top = rect.y + rect.height - 1 - track
        size = max(1, min(track, round(visible * track / total)))
        start = min(track - size, round(first * track / total))
        column = rect.x + rect.width - 1 - SCROLLBAR_COLUMNS
        self.scrollbar_hits[pane] = Rect(top, column, track, 1)
        self.scroll_thumb_hits[pane] = Rect(top + start, column, size, 1)
        for row in range(track):
            self._add(
                top + row, column, self.glyphs.box[BORDER_UP | BORDER_DOWN], self.colors.border
            )
        for row in range(size):
            self._add(
                top + start + row,
                rect.x + rect.width - 1 - SCROLLBAR_COLUMNS,
                self.glyphs.thumb,
                self.colors.focus if self.session.focus == pane else self.colors.border,
            )

    def _draw_help(self) -> None:
        width = min(self.screen.size()[1] - 4, 78)
        lines = help_lines(width - 5)
        rect = self._overlay_rect(78, len(lines) + 4)
        for row in range(rect.height):
            self._add(rect.y + row, rect.x, " " * rect.width, max_width=rect.width)
        self._draw_overlay_frame(rect, ["Debugger controls"])
        self._add(rect.y, rect.x + rect.width - 5, "[x]", self.colors.key)
        self.overlay_hits.append((Rect(rect.y, rect.x + rect.width - 6, 1, 5), Key.ESCAPE))
        visible = max(1, rect.height - 3)
        self.help_max_scroll = max(0, len(lines) - visible)
        self.help_scroll = max(0, min(self.help_scroll, self.help_max_scroll))
        if self.help_max_scroll:
            self.help_track = Rect(rect.y + 1, rect.x + rect.width - 2, visible, 1)
            size = max(1, round(visible * visible / len(lines)))
            start = round((visible - size) * self.help_scroll / self.help_max_scroll)
            self.help_thumb = Rect(self.help_track.y + start, self.help_track.x, size, 1)
            for row in range(visible):
                self._add(
                    self.help_track.y + row,
                    self.help_track.x,
                    self.glyphs.thumb
                    if start <= row < start + size
                    else self.glyphs.box[BORDER_UP | BORDER_DOWN],
                    self.colors.key if start <= row < start + size else self.colors.border,
                    1,
                )
        for row, (text, kind) in enumerate(lines[self.help_scroll : self.help_scroll + visible]):
            self._add(
                rect.y + 1 + row,
                rect.x + 2,
                text,
                self.colors.current if kind else 0,
                rect.width - 4,
            )
        footer = (
            "Esc Close / Wheel Scroll"
            if rect.width < 70
            else "Esc / ? Close   Up Down PgUp PgDn Scroll"
        )
        if len(lines) > visible:
            footer += (
                f"   {self.help_scroll + 1}-"
                f"{min(len(lines), self.help_scroll + visible)}/{len(lines)}"
            )
        self._add(
            rect.y + rect.height - 1,
            max(rect.x + 1, rect.x + rect.width - 2 - len(footer) - 2),
            f" {footer} ",
            self.colors.border,
            rect.width - 2,
        )

    def _handle_help_mouse(self, event: MouseEvent, wheel: int) -> Outcome:
        y, x = event.y, event.x
        pressed, released = event.pressed, event.release
        self.redraw = True
        if wheel:
            self.help_scroll = max(0, min(self.help_max_scroll, self.help_scroll + wheel))
            return None
        if self.help_drag is not None and self.help_track and self.help_thumb:
            origin_y, origin_scroll = self.help_drag
            travel = max(1, self.help_track.height - self.help_thumb.height)
            self.help_scroll = max(
                0,
                min(
                    self.help_max_scroll,
                    origin_scroll + round((y - origin_y) * self.help_max_scroll / travel),
                ),
            )
            if released:
                self.help_drag = None
            return None
        if pressed or released:
            for hit, key in self.overlay_hits:
                if hit.contains(y, x):
                    # Some terminals synthesize clicks; others deliver down/up.
                    # Close on down and consume its up event after dismissal.
                    self.consume_mouse_release = pressed and not released
                    return self.handle_key(key)
        if pressed and self.help_track and self.help_track.contains(y, x):
            if self.help_thumb.contains(y, x):
                self.help_drag = (y, self.help_scroll)
            else:
                delta = self.help_track.height * (-1 if y < self.help_thumb.y else 1)
                self.help_scroll = max(0, min(self.help_max_scroll, self.help_scroll + delta))
        return None

    def _close_help(self) -> None:
        self.help_visible = False
        self.help_drag = None
        self.help_scroll = 0

    def _scroll_help(self, delta: int) -> None:
        self.help_scroll = max(0, self.help_scroll + delta)

    def _help_page(self) -> int:
        return max(1, self.screen.size()[0] - 6)

    def _help_to_bottom(self) -> None:
        width = min(self.screen.size()[1] - 4, 78)
        self.help_scroll = len(help_lines(width - 4))

    def _draw_editor(self) -> None:
        rect = self._overlay_rect(76, EDITOR_HEIGHT)
        x, y, width = rect.x, rect.y, rect.width
        self.editor_rect = rect
        for row in range(rect.height):
            self._add(rect.y + row, rect.x, " " * rect.width, max_width=rect.width)
        assert self.editor is not None
        pane = self.editor.pane
        self._draw_overlay_frame(rect, [self.editor.title])
        first_description, second_description = self.editor.hints
        self._add(
            y + 1,
            x + 2,
            first_description,
            self.colors.border,
            width - 4,
        )
        self._add(
            y + 2,
            x + 2,
            second_description,
            self.colors.border,
            width - 4,
        )
        # Matches the line-oriented debugger's own prompt.
        label = f"{pane} >>> "
        self._add(y + 4, x + 2, label, self.colors.current, width - 4)
        available = max(0, width - 4 - len(label))
        visible_start = max(0, self.cursor - available + 1)
        visible = self.command[visible_start : visible_start + available]
        self._add(y + 4, x + 2 + len(label), visible, max_width=available)
        self.editor_field = (Rect(y + 4, x + 2 + len(label), 1, available), visible_start)
        if self.session.message_is_error:
            self._add(
                y + 5,
                x + 2,
                self.session.message,
                self.colors.error,
                width - 4,
            )
        hint = f"Enter Apply {self.glyphs.separator} Esc Cancel"
        self._add(
            rect.y + rect.height - 1,
            max(rect.x + 1, rect.x + rect.width - 4 - len(hint)),
            f" {hint} ",
            self.colors.border,
            rect.width - 2,
        )
        hint_x = max(rect.x + 1, rect.x + rect.width - 4 - len(hint)) + 1
        self.overlay_hits.extend(
            [
                (Rect(rect.y + rect.height - 1, hint_x, 1, len("Enter Apply")), Key.ENTER),
                (
                    Rect(
                        rect.y + rect.height - 1, hint_x + hint.index("Esc"), 1, len("Esc Cancel")
                    ),
                    Key.ESCAPE,
                ),
            ]
        )
        cursor_x = x + 2 + len(label) + self.cursor - visible_start
        self.screen.cursor(True, y + 4, min(x + width - 2, cursor_x))

    def handle_key(self, key: Input) -> Outcome:
        """Act on one input; a ``DebugAction`` ends this stop, None stays."""
        # A preceding queued event may have changed layout without a repaint.
        # Cursor arithmetic must use the new pane dimensions immediately.
        self._update_pane_rects(self.layout())
        if isinstance(key, MouseEvent):
            return self._handle_mouse(key)
        self.redraw = True
        # A drag whose release never arrives - the pointer left the terminal,
        # say - must not turn every later mouse event into a resize.
        self._cancel_gestures()
        if key is Key.RESIZE:
            return None
        command = lookup(self._key_focus(), key)
        if command is not None:
            return self._commands[command]()
        if self.editor_mode and not self.help_visible:
            self._edit_text(key)
        return None

    def _key_focus(self) -> Focus:
        """The help overlay and the editor are modal; otherwise the focused pane."""
        if self.help_visible:
            return Focus.HELP
        if self.editor_mode:
            return Focus.EDITOR
        return PANE_FOCUS[self.session.focus]

    def _command_table(self) -> dict[Command, Callable[[], Outcome]]:
        """The method that carries out each command."""

        def does(action: Callable[[], object]) -> Callable[[], Outcome]:
            def run() -> Outcome:
                action()
                return None

            return run

        table: dict[Command, Callable[[], Outcome]] = {
            Command.QUIT: self._quit,
            Command.CONTINUE: lambda: self._execution_action(DebugAction.CONTINUE),
            Command.STEP: lambda: self._execution_action(DebugAction.STEP),
            Command.SHOW_HELP: does(self._show_help),
            Command.RESET_SPLITS: does(self._reset_splits),
            Command.NEXT_PANE: does(lambda: self._select_relative_pane(1)),
            Command.SPLIT_LEFT: does(lambda: self._resize_vertical_split(-PANE_RESIZE_STEP)),
            Command.SPLIT_RIGHT: does(lambda: self._resize_vertical_split(PANE_RESIZE_STEP)),
            Command.SPLIT_UP: does(lambda: self._resize_horizontal_split(-1)),
            Command.SPLIT_DOWN: does(lambda: self._resize_horizontal_split(1)),
            Command.CLOSE_TAB: does(self._close_active_tab),
            Command.NEXT_TAB: does(lambda: self._select_tab_in_focused_pane(1)),
            Command.PREVIOUS_TAB: does(lambda: self._select_tab_in_focused_pane(-1)),
            Command.TOGGLE_BREAKPOINT: does(self._toggle_breakpoint_at_cursor),
            Command.RUN_TO_CURSOR: self._run_to_cursor,
            Command.CANCEL: does(self._cancel),
            Command.EDITOR_LEFT: does(lambda: self._move_editor_cursor(self.cursor - 1)),
            Command.EDITOR_RIGHT: does(lambda: self._move_editor_cursor(self.cursor + 1)),
            Command.EDITOR_HOME: does(lambda: self._move_editor_cursor(0)),
            Command.EDITOR_END: does(lambda: self._move_editor_cursor(len(self.command))),
            Command.EDITOR_SUBMIT: self._submit_editor,
            Command.CLOSE_HELP: does(self._close_help),
            Command.HELP_UP: does(lambda: self._scroll_help(-1)),
            Command.HELP_DOWN: does(lambda: self._scroll_help(1)),
            Command.HELP_PAGE_UP: does(lambda: self._scroll_help(-self._help_page())),
            Command.HELP_PAGE_DOWN: does(lambda: self._scroll_help(self._help_page())),
            Command.HELP_TOP: does(self._close_help_scroll_to_top),
            Command.HELP_BOTTOM: does(self._help_to_bottom),
        }
        for command, pane in (
            (Command.FOCUS_PIPELINE, "pipeline"),
            (Command.FOCUS_DISASSEMBLY, "disassembly"),
            (Command.FOCUS_XMEM, "xmem"),
            (Command.FOCUS_REGISTERS, "registers"),
        ):
            table[command] = does(lambda pane=pane: self._focus_pane(pane))
        for command in PANE_COMMANDS:
            table[command] = lambda command=command: self._pane_command(command)
        return table

    def _pane_command(self, command: Command) -> Outcome:
        pane = self.session.focus
        rect = self._pane_rect(pane)
        self._apply(self.panes[pane].handle(command, self.debug, self.session.panes[pane], rect))
        return None

    def _close_help_scroll_to_top(self) -> None:
        self.help_scroll = 0

    def _show_help(self) -> None:
        # Opens only outside an editor, so "?" stays typable in a tab request.
        self.help_visible = True
        self.help_scroll = 0

    def _quit(self) -> Outcome:
        self.debug.halt()
        return self._execution_action(DebugAction.QUIT)

    def _cancel(self) -> None:
        """Esc: leave the editor, or the disassembly pane's stage view."""
        if self.editor_mode:
            pane = self.editor_pane
            self._clear_editor()
            label = "XMEM" if pane == "xmem" else pane
            self.session.set_message(f"{label} tab edit cancelled")
        elif self.session.stages_visible and self.session.focus == "disassembly":
            self.session.stages_visible = False

    def _disassembly_cursor_pc(self) -> int:
        tab = self.session.disassembly_tab()
        base = self.state.program_counter if tab.target is None else tab.target
        return max(0, min(INST_MEM_SIZE - 1, base + tab.cursor_offset))

    def _toggle_breakpoint_at_cursor(self) -> None:
        pc = self._disassembly_cursor_pc()
        enabled = self.debug.toggle_breakpoint(pc)
        action = "set" if enabled else "removed"
        self.session.set_message(f"Breakpoint {action} at PC {pc}")

    def _run_to_cursor(self) -> Outcome:
        pc = self._disassembly_cursor_pc()
        if self.debug.run_until(pc):
            return self._execution_action(DebugAction.CONTINUE)
        self.session.set_message(f"Already at PC {pc}")
        return None

    def _stages_shown(self) -> bool:
        return self.session.stages_visible and self.session.focus == "disassembly"

    def _scroll_stages(self, delta: int) -> None:
        self.session.stages_scroll = max(0, self.session.stages_scroll + delta)

    def _move_or_scroll_stages(self, *, dy: int) -> None:
        if self._stages_shown():
            self._scroll_stages(dy)
        else:
            self._move_focused_cursor(dy=dy)

    def _page_focused_cursor(self, direction: int) -> None:
        if self._stages_shown():
            self._scroll_stages(direction * max(1, self.disassembly_rect.height - 4))
        else:
            self._move_focused_cursor(dy=direction * self._focused_page_size())

    def _jump_or_scroll_stages(self, *, end: bool) -> None:
        if self._stages_shown():
            self.session.stages_scroll = 10000 if end else 0
        else:
            self._jump_focused_cursor(end=end)

    def _edit_focused_value(self) -> None:
        if self.session.focus == "disassembly":
            self._apply(
                scalar_editor(
                    self.debug,
                    "disassembly",
                    "pc",
                    self.state.program_counter,
                    "Range: 0 through instruction memory size - 1",
                )
            )
        else:
            self._begin_value_editor()

    def _move_editor_cursor(self, position: int) -> None:
        self.cursor = max(0, min(len(self.command), position))

    def _submit_editor(self) -> Outcome:
        """Enter: the editor's pane applies the text; an error keeps it open."""
        assert self.editor is not None
        reply = self.editor.submit(self.command)
        if not reply.error:
            self._clear_editor()
        self._apply(reply)
        return None

    def _apply(self, reply: Reply | None) -> None:
        """Open the editor a pane asked for and show its message."""
        if reply is None:
            return
        if reply.editor is not None:
            editor = reply.editor
            self.editor = editor
            self.editor_mode = editor.mode
            self.editor_pane = editor.pane
            self.editor_register = editor.register
            self.command = editor.text
            self.cursor = len(editor.text)
        if reply.message is not None:
            self.session.set_message(reply.message, error=reply.error)

    def _edit_text(self, key: Input) -> None:
        """Typing in the editor: insert a character, or delete one."""
        if key is Key.DELETE and self.cursor < len(self.command):
            self.command = self.command[: self.cursor] + self.command[self.cursor + 1 :]
        elif key is Key.BACKSPACE and self.cursor > 0:
            self.command = self.command[: self.cursor - 1] + self.command[self.cursor :]
            self.cursor -= 1
        elif isinstance(key, str) and key.isprintable():
            self.command = self.command[: self.cursor] + key + self.command[self.cursor :]
            self.cursor += len(key)

    def _handle_mouse(self, event: MouseEvent) -> Outcome:
        y, x = event.y, event.x
        released, pressed = event.release, event.pressed
        wheel = event.wheel * MOUSE_WHEEL_LINES
        if released and self.consume_mouse_release:
            self.consume_mouse_release = False
            return None
        if pressed:
            self.consume_mouse_release = False
        if not (pressed or released or wheel or self.drag or self.scroll_drag or self.help_drag):
            return None
        if self.redraw:
            self.render()
        if self.help_visible:
            return self._handle_help_mouse(event, wheel)
        # The editor is modal: it consumes every pointer event, wheel included.
        if self.editor_mode:
            self.redraw = True
            if pressed:
                for rect, key in self.overlay_hits:
                    if rect.contains(y, x):
                        return self.handle_key(key)
                if self.editor_field is not None:
                    rect, start = self.editor_field
                    if rect.contains(y, x):
                        self.cursor = min(len(self.command), start + x - rect.x)
            return None
        if self.scroll_drag is not None:
            self._continue_scroll_drag(y, released)
            return None
        if self.drag is not None:
            return self._continue_drag(y, x, released)
        if wheel:
            self.last_value_click = None
            self.redraw = True
            self._wheel(y, x, wheel, horizontal=event.shift)
            return None
        if pressed:
            return self._press(event)
        return None

    def _continue_scroll_drag(self, y: int, released: bool) -> None:
        pane = self.scroll_drag
        track = self.scrollbar_hits.get(pane) if pane is not None else None
        if pane is not None and track is not None:
            origin_y, fraction = self.scroll_origin or (track.y, 0.0)
            delta = y - origin_y
            if delta or self.scroll_drag_moved:
                self.scroll_drag_moved = True
                self._scroll_to_fraction(pane, fraction + delta / max(1, track.height - 1))
                self.redraw = True
        if released:
            self.scroll_drag = None
            self.scroll_origin = None

    def _press(self, event: MouseEvent) -> Outcome:
        y, x = event.y, event.x
        self.redraw = True
        for rect, key in self.action_hits:
            if rect.contains(y, x):
                return self.handle_key(key)
        if event.press:
            for pane, track in self.scrollbar_hits.items():
                if track.contains(y, x):
                    self._focus_pane(pane)
                    self.scroll_drag = pane
                    self.scroll_drag_moved = False
                    self.last_value_click = None
                    thumb = self.scroll_thumb_hits.get(pane)
                    if thumb is None or not thumb.contains(y, x):
                        self._scroll_to_fraction(pane, (y - track.y) / max(1, track.height - 1))
                    self.scroll_origin = (y, self._scroll_cursor_fraction(pane))
                    return None
            split = self._split_at(y, x)
            if split is not None:
                self.drag = (split, y, x, False)
                return None
        self._handle_click(y, x, double=event.double)
        return None

    def _wheel(self, y: int, x: int, delta: int, *, horizontal: bool = False) -> None:
        if self._focus_pane_at(y, x):
            rect = self._pane_rect(self.session.focus)
            if y == rect.y + 1:
                self._select_tab_in_focused_pane(1 if delta > 0 else -1)
            elif (
                self.session.focus == "disassembly"
                and self.session.stages_visible
                and not horizontal
            ):
                self.session.stages_scroll = max(0, self.session.stages_scroll + delta)
            elif horizontal:
                self._move_focused(dx=delta)
            else:
                self._move_focused(dy=delta)

    def _split_at(self, y: int, x: int) -> str | None:
        """The split whose border covers ``(y, x)``, if any."""
        if self._single_pane():
            return None
        if self.xmem_rect.height and y == self.xmem_rect.y:
            return "top_pane_rows"
        for split, rect in (
            ("register_pane_columns", self.registers_rect),
            ("pipeline_pane_columns", self.disassembly_rect),
        ):
            if x == rect.x and rect.y < y < rect.y + rect.height - 1:
                return split
        return None

    def _continue_drag(self, y: int, x: int, released: bool) -> Outcome:
        split, origin_y, origin_x, moved = self.drag or ("", 0, 0, False)
        moved = moved or (y, x) != (origin_y, origin_x)
        if moved:
            self.redraw = True
            self._drag_split_to(split, y, x)
        if released:
            self.drag = None
            if not moved:
                self.redraw = True
                # A press and release without motion is an ordinary click.
                self._handle_click(origin_y, origin_x)
        else:
            self.drag = (split, origin_y, origin_x, moved)
        return None

    def _drag_split_to(self, split: str, y: int, x: int) -> None:
        rows, columns = self.screen.size()
        if split == "top_pane_rows":
            self.session.top_pane_rows = y - HEADER_ROWS + 1
            height = self._clamped_top_pane_rows(rows)
            self.session.top_pane_rows = height
            self.session.set_message(f"Top pane height: {height} rows")
            return
        setattr(self.session, split, x + 1 if split == "pipeline_pane_columns" else columns - x)
        if split == "pipeline_pane_columns":
            width = self._clamped_pipeline_columns(columns)
            self.session.pipeline_pane_columns = width
            self.session.set_message(f"Pipeline pane width: {width} columns")
            return
        width = self._clamped_register_columns(
            columns,
            rows - HEADER_ROWS - FOOTER_ROWS - self._clamped_top_pane_rows(rows) + 1,
        )
        self.session.register_pane_columns = width
        self.session.set_message(f"LR / CR pane width: {width} columns")

    def _handle_click(self, y: int, x: int, *, double: bool = False) -> None:
        previous_click = self.last_value_click
        self.last_value_click = None
        for rect, pane, index in self.tab_scroll_hits:
            if rect.contains(y, x):
                self._focus_pane(pane)
                self.session.panes[pane].select(index)
                return
        for pane, hits in self.tab_close_hits.items():
            for rect, index in hits:
                if rect.contains(y, x):
                    self.session.focus = pane
                    self._close_tab(pane, index)
                    return
        for pane, hits in self.tab_hits.items():
            for rect, index in hits:
                if rect.contains(y, x):
                    self.session.focus = pane
                    self.session.panes[pane].select(index)
                    self.session.set_message(f"Selected {self.session.panes[pane][index].title}")
                    return
        for pane, rect in self.add_hits.items():
            if rect.contains(y, x):
                self.session.focus = pane
                self._pane_command(Command.ADD_TAB)
                return
        for pane, rect in self.scrollbar_hits.items():
            if rect.contains(y, x):
                self.session.focus = pane
                thumb = self.scroll_thumb_hits.get(pane)
                if thumb is None or not thumb.contains(y, x):
                    self._scroll_to_fraction(pane, (y - rect.y) / max(1, rect.height - 1))
                return
        for rect, pc in self.value_hits.get("disassembly", []):
            if rect.contains(y, x) and x < rect.x + 2:
                self._focus_pane("disassembly")
                self._place_cursor("disassembly", pc)
                enabled = self.debug.toggle_breakpoint(pc)
                action = "set" if enabled else "removed"
                self.session.set_message(f"Breakpoint {action} at PC {pc}")
                return
        for pane, hits in self.value_hits.items():
            for rect, payload in hits:
                if rect.contains(y, x):
                    self._focus_pane(pane)
                    self._place_cursor(pane, payload)
                    if pane in ("registers", "pipeline", "xmem"):
                        now = time.monotonic()
                        tab_set = self.session.panes[pane]
                        identity = id(tab_set[tab_set.active])
                        if double or (
                            previous_click is not None
                            and previous_click[:2] == (identity, payload)
                            and now - previous_click[2] <= DOUBLE_CLICK_SECONDS
                        ):
                            self._pane_command(Command.EDIT_VALUE)
                        else:
                            self.last_value_click = (identity, payload, now)
                    return
        for pane, rect in (
            ("pipeline", self.pipeline_rect),
            ("xmem", self.xmem_rect),
            ("disassembly", self.disassembly_rect),
            ("registers", self.registers_rect),
        ):
            if rect.contains(y, x):
                self._focus_pane(pane)
                return

    def _place_cursor(self, pane: str, payload: Any) -> None:
        """Put *pane*'s cursor on the value that was clicked."""
        self.panes[pane].place_cursor(self.session.panes[pane], payload)

    def _legacy_place_cursor(self, pane: str, payload: Any) -> None:
        if pane == "disassembly":
            tab = self.session.disassembly_tab()
            base = self.state.program_counter if tab.target is None else tab.target
            tab.cursor_offset = payload - base
            return
        if pane == "registers":
            self.session.register_tab().cursor = payload
            return
        self.session.pipeline_tab().cursor = payload

    def _scroll_cursor_fraction(self, pane: str) -> float:
        """Remember the selection at grab time, so a thumb press never jumps."""
        total = self.scroll_state[pane][2]
        target = self.panes[pane].cursor_row(
            self.debug, self.session.panes[pane], self._pane_rect(pane)
        )
        return (target + 0.5) / max(1, total)

    def _legacy_cursor_row(self, pane: str) -> int:
        if pane == "disassembly":
            tab = self.session.disassembly_tab()
            base = self.state.program_counter if tab.target is None else tab.target
            return base + tab.cursor_offset
        if pane != "pipeline":
            return 0
        tab = self.session.pipeline_tab()
        spec = _pipeline_spec(tab.register_key)
        fmt = tab.fmt or _pipeline_format(self.state, spec)
        size, _ = self._pipeline_value_geometry(fmt)
        items = len(_read_pipeline_register(self.state, spec)) // size
        rows = self._pipeline_display_rows(fmt, self.pipeline_rect, items)
        per_line = self._pipeline_items_per_line(fmt, self.pipeline_rect)
        return next(
            index
            for index, first in enumerate(rows)
            if first is not None and first <= tab.cursor < first + per_line
        )

    def _scroll_to_fraction(self, pane: str, fraction: float) -> None:
        """Jump the pane's cursor to a position along its scrollbar."""
        _, _, total = self.scroll_state.get(pane, (0, 0, 0))
        if total <= 0:
            return
        target = max(0, min(total - 1, int(fraction * total)))
        self.panes[pane].scroll_to_row(
            self.debug, self.session.panes[pane], self._pane_rect(pane), target
        )

    def _legacy_scroll_to_row(self, pane: str, target: int) -> None:
        if pane == "disassembly":
            tab = self.session.disassembly_tab()
            base = self.state.program_counter if tab.target is None else tab.target
            tab.cursor_offset = target - base
            return
        if pane != "pipeline":
            return
        tab = self.session.pipeline_tab()
        spec = _pipeline_spec(tab.register_key)
        fmt = tab.fmt or _pipeline_format(self.state, spec)
        item_size, _ = self._pipeline_value_geometry(fmt)
        items = len(_read_pipeline_register(self.state, spec)) // item_size
        rows = self._pipeline_display_rows(fmt, self.pipeline_rect, items)
        target = min(target, len(rows) - 1)
        if rows[target] is None:
            target = min(target + 1, len(rows) - 1)
        tab.cursor = max(0, min(items - 1, rows[target]))

    def _focus_pane_at(self, y: int, x: int) -> bool:
        panes = (
            ("pipeline", self.pipeline_rect),
            ("xmem", self.xmem_rect),
            ("registers", self.registers_rect),
            ("disassembly", self.disassembly_rect),
        )
        for name, rect in panes:
            if rect.contains(y, x):
                self.session.focus = name
                return True
        return False

    def _parse_editor_tab(
        self,
        pane: str,
        command: str,
        existing: Any | None,
    ) -> tuple[Any | None, str | None]:
        if pane == "disassembly":
            token = command.strip().lower()
            if token == "current":
                target = None
            else:
                try:
                    target = int(token, 0)
                except ValueError:
                    return None, "expected 'current' or a numeric PC"
                if not 0 <= target < INST_MEM_SIZE:
                    return None, f"PC must be between 0 and {INST_MEM_SIZE - 1}"
            fmt = existing.fmt if isinstance(existing, DisassemblyTab) else "compact"
            return DisassemblyTab(target=target, fmt=fmt), None
        if pane == "registers":
            group = command.strip().lower()
            if group not in REGISTER_TAB_GROUPS:
                return None, "expected all, lr, or cr"
            fmt = existing.fmt if isinstance(existing, RegisterTab) else "hex"
            return RegisterTab(group=group, fmt=fmt), None
        token = command.strip().lower()
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

    def _begin_value_editor(self) -> None:
        if self.session.focus == "registers":
            self._begin_scalar_editor()
            return
        tab = self.session.pipeline_tab()
        spec = _pipeline_spec(tab.register_key)
        fmt = tab.fmt or _pipeline_format(self.state, spec)
        size, _ = self._pipeline_value_geometry(fmt)
        backing = (
            spec.wide_backing
            if self.state.wide_vector_debug and spec.wide_backing
            else spec.backing
        )
        register = self.debug.register(backing, spec.index).value
        assert isinstance(register, bytes)
        register_size = len(register)
        tab.cursor = max(0, min(tab.cursor, register_size // size - 1))
        start = tab.cursor * size
        offset = spec.index * register_size + start

        def write(data: bytes) -> None:
            self.debug.write_register_bytes(backing, offset, data)

        value = EditedValue(
            f"{spec.label}[{tab.cursor}]", fmt, register[start : start + size], size
        )
        self._apply(value_editor(self.debug, "pipeline", value, write))

    def _begin_scalar_editor(self) -> None:
        """Edit the LR / CR under the cursor."""
        tab = self.session.register_tab()
        groups = self._register_groups(tab)
        position = max(0, min(tab.cursor, len(groups) * _REGISTERS_PER_GROUP - 1))
        register = f"{groups[position // _REGISTERS_PER_GROUP]}{position % _REGISTERS_PER_GROUP}"
        desc, index = self.debug.resolve_register(register)
        current = self.state.regfile.get_scalar(desc.name, index)
        hint = "Register masking applies; CR0 and CR1 are read-only"
        self._apply(scalar_editor(self.debug, "registers", register, current, hint))

    # The tab editor of each pane still drawn here: its label, hints and
    # default target.
    _TAB_EDITORS: dict[str, tuple[str, tuple[str, str], str]] = {
        "disassembly": (
            "Disassembly",
            ("Target: current or PC", "Range: 0 through instruction memory size - 1"),
            "current",
        ),
        "registers": (
            "LR / CR",
            ("Group: all, lr, or cr", "Format is retained independently per tab"),
            "all",
        ),
        "pipeline": (
            "Pipeline",
            (
                "Register: R0, R1, R_C, R_MASK, R_ACC, POST_AAQ,",
                "          MULT_RES, or MEM_BYPASS",
            ),
            "R0",
        ),
    }

    def _begin_tab_editor(self, mode: str) -> None:
        pane = self.session.focus
        tabs = self.session.panes[pane]
        tab = tabs[tabs.active]
        label, hints, command = self._TAB_EDITORS[pane]
        if mode == "edit":
            if pane == "disassembly":
                command = "current" if tab.target is None else str(tab.target)
            elif pane == "registers":
                command = tab.group
            else:
                command = _pipeline_spec(tab.register_key).label

        def parse(text: str, existing: Any) -> tuple[Any | None, str | None]:
            return self._parse_editor_tab(pane, text, existing)

        self._apply(tab_editor(pane, TabTarget(label, hints, parse), mode, tabs, command))

    def _clear_editor(self) -> None:
        self.command = ""
        self.cursor = 0
        self.editor = None
        self.editor_mode = None
        self.editor_pane = None
        self.editor_register = None
        self.screen.cursor(False)

    def _select_relative_pane(self, delta: int) -> None:
        panes = list(PANE_FOCUS_ORDER)
        try:
            current = panes.index(self.session.focus)
        except ValueError:
            current = panes.index("xmem")
        self._focus_pane(panes[(current + delta) % len(panes)])

    def _focus_pane(self, pane: str) -> None:
        self.session.focus = pane
        self.session.set_message("Ready")

    def _select_tab_in_focused_pane(self, delta: int) -> None:
        pane = self.session.focus
        tab = self.session.panes[pane].step(delta)
        if tab is None:
            self.session.set_message(f"{pane} has no tabs")
            return
        self.session.set_message(f"Selected {tab.title}")

    def _cycle_focused_format(self, delta: int = 1) -> None:
        pane = self.session.focus
        if pane == "disassembly":
            tab = self.session.disassembly_tab()
            tab.fmt = self._next_format(tab.fmt, DISASSEMBLY_FORMATS, delta)
            self.session.set_message(f"Disassembly format: {tab.fmt}")
            return
        if pane == "registers":
            tab = self.session.register_tab()
            tab.fmt = self._next_format(
                tab.fmt,
                SCALAR_REGISTER_FORMATS,
                delta,
            )
            tab.column_scroll = 0
            self.session.set_message(f"LR / CR format: {tab.fmt}")
            return
        tab = self.session.pipeline_tab()
        spec = _pipeline_spec(tab.register_key)
        current = tab.fmt or _pipeline_format(self.state, spec)
        selected = self._next_format(
            current,
            PIPELINE_REGISTER_FORMATS,
            delta,
        )
        tab.fmt = selected
        tab.scroll = 0
        tab.cursor = 0
        self.session.set_message(f"{spec.label} format: {selected}")

    @staticmethod
    def _next_format(
        current: str,
        formats: tuple[str, ...],
        delta: int,
    ) -> str:
        try:
            index = formats.index(current)
        except ValueError:
            index = -1
        return formats[(index + delta) % len(formats)]

    def _close_active_tab(self) -> None:
        pane = self.session.focus
        tabs = self.session.panes[pane]
        if tabs:
            self._close_tab(pane, tabs.active)

    def _close_tab(self, pane: str, index: int) -> None:
        tabs = self.session.panes[pane]
        if not 0 <= index < len(tabs):
            return
        tabs.close(index)
        self.session.set_message(f"Closed {pane} tab")

    def _move_focused(self, *, dx: int = 0, dy: int = 0) -> None:
        """Move the focused pane's cursor by *dx* values and *dy* rows."""
        pane = self.session.focus
        self.panes[pane].move(
            self.debug, self.session.panes[pane], self._pane_rect(pane), dx=dx, dy=dy
        )

    def _move_focused_cursor(self, *, dx: int = 0, dy: int = 0) -> None:
        if self.session.focus == "disassembly":
            tab = self.session.disassembly_tab()
            base = self.state.program_counter if tab.target is None else tab.target
            minimum = -base
            maximum = INST_MEM_SIZE - 1 - base
            tab.cursor_offset = max(
                minimum,
                min(maximum, tab.cursor_offset + dy),
            )
            return
        if self.session.focus == "registers":
            tab = self.session.register_tab()
            grid_rows, column_count = self._register_grid(tab)
            row = min(grid_rows - 1, tab.cursor % grid_rows)
            column = tab.cursor // grid_rows
            row = max(0, min(grid_rows - 1, row + dy))
            column = max(0, min(column_count - 1, column + dx))
            tab.cursor = column * grid_rows + row
            return
        self._move_pipeline_cursor(dx, dy)

    def _jump_focused_cursor(self, *, end: bool) -> None:
        """Move the focused pane's cursor to its first or last value."""
        if self.session.focus == "disassembly":
            tab = self.session.disassembly_tab()
            base = self.state.program_counter if tab.target is None else tab.target
            tab.cursor_offset = (INST_MEM_SIZE - 1 - base) if end else -base
            return
        if self.session.focus == "registers":
            tab = self.session.register_tab()
            groups = self._register_groups(tab)
            tab.cursor = len(groups) * _REGISTERS_PER_GROUP - 1 if end else 0
            return
        tab = self.session.pipeline_tab()
        spec = _pipeline_spec(tab.register_key)
        fmt = tab.fmt or _pipeline_format(self.state, spec)
        item_size, _ = self._pipeline_value_geometry(fmt)
        item_count = len(_read_pipeline_register(self.state, spec)) // item_size
        tab.cursor = max(0, item_count - 1) if end else 0

    def _move_pipeline_cursor(self, dx: int, dy: int) -> None:
        tab = self.session.pipeline_tab()
        spec = _pipeline_spec(tab.register_key)
        data = _read_pipeline_register(self.state, spec)
        fmt = tab.fmt or _pipeline_format(self.state, spec)
        item_size, _ = self._pipeline_value_geometry(fmt)
        item_count = len(data) // item_size
        if item_count == 0:
            tab.cursor = 0
            return
        items_per_line = self._pipeline_items_per_line(fmt, self.pipeline_rect)
        movement = dx + dy * items_per_line
        tab.cursor = max(
            0,
            min(item_count - 1, tab.cursor + movement),
        )

    def _focused_page_size(self) -> int:
        if self.session.focus == "disassembly":
            _, shown, _ = self.scroll_state.get(
                "disassembly",
                (0, self.disassembly_rect.height, 0),
            )
            return max(1, shown)
        if self.session.focus == "pipeline":
            return max(1, self.pipeline_rect.height - PIPELINE_NON_CONTENT_ROWS)
        return max(1, self.xmem_rect.height - XMEM_NON_CONTENT_ROWS)

    def _execution_action(self, action: DebugAction) -> DebugAction:
        self.session.commit_baseline(self.state)
        self.session.active = action in (DebugAction.CONTINUE, DebugAction.STEP)
        return action

    def _box(self, rect: Rect, *, focused: bool = False) -> None:
        """Record one pane frame; ``_flush_borders`` resolves the junctions."""
        if rect.height < 2 or rect.width < 2:
            return
        priority = 1 if focused else 0
        attr = self.colors.focus if focused else self.colors.border
        right = rect.x + rect.width - 1
        bottom = rect.y + rect.height - 1
        for x in range(rect.x, right + 1):
            mask = (BORDER_LEFT if x > rect.x else 0) | (BORDER_RIGHT if x < right else 0)
            for y in (rect.y, bottom):
                self._mark_border(y, x, mask, attr, priority)
        for y in range(rect.y, bottom + 1):
            mask = (BORDER_UP if y > rect.y else 0) | (BORDER_DOWN if y < bottom else 0)
            for x in (rect.x, right):
                self._mark_border(y, x, mask, attr, priority)

    def _mark_border(
        self,
        y: int,
        x: int,
        mask: int,
        attr: int,
        priority: int,
    ) -> None:
        current_mask, current_attr, current_priority = self.borders.get(
            (y, x),
            (0, attr, -1),
        )
        if priority >= current_priority:
            current_attr = attr
            current_priority = priority
        self.borders[(y, x)] = (
            current_mask | mask,
            current_attr,
            current_priority,
        )

    def _flush_borders(self) -> None:
        """Paint every recorded border cell, then start a fresh frame."""
        for (y, x), (mask, attr, _) in self.borders.items():
            self._add(y, x, self.glyphs.box[mask], attr, 1)
        self.borders.clear()

    def _fit_segments(self, segments: list[str], width: int) -> str:
        """Join as many whole segments as fit; never cut one in half."""
        return fit_segments(segments, width, self.glyphs)

    def _overlay_rect(self, preferred_width: int, preferred_height: int) -> Rect:
        """Centre a modal overlay between the header and the footer."""
        rows, columns = self.screen.size()
        available = max(1, rows - HEADER_ROWS - FOOTER_ROWS)
        width = max(1, min(columns - 4, preferred_width))
        height = max(1, min(available, preferred_height))
        return Rect(
            HEADER_ROWS + (available - height) // 2,
            max(0, (columns - width) // 2),
            height,
            width,
        )

    def _draw_overlay_frame(self, rect: Rect, segments: list[str]) -> None:
        """Frame and title a modal overlay, painted on top of the panes."""
        self._box(rect, focused=True)
        self._flush_borders()
        self._draw_pane_title(rect, segments)

    def _canvas(self) -> Canvas:
        """The shared drawing helpers, writing through ``_add``'s sink."""
        rows, columns = self.screen.size()

        def emit(y: int, x: int, text: str, attr: int) -> None:
            self._sink(y, x, text, attr)

        return Canvas(self.theme, Rect(0, 0, rows, columns), emit)

    def _draw_pane_title(
        self,
        rect: Rect,
        segments: list[str],
        *,
        reserved: int = 0,
        focused: bool = False,
    ) -> None:
        """Draw a pane title on its top border (``Canvas.title``)."""
        self._canvas().title(rect, segments, reserved=reserved, focused=focused)

    def _draw_top_pane_header(
        self,
        rect: Rect,
        segments: list[str],
        readout: str,
        *,
        focused: bool = False,
    ) -> None:
        """Title and position readout for a pane whose bottom border is shared."""
        self._canvas().top_pane_header(rect, segments, readout, focused=focused)

    def _add(
        self,
        y: int,
        x: int,
        text: str,
        attr: int = 0,
        max_width: int | None = None,
    ) -> None:
        rows, columns = self.screen.size()
        if y < 0 or y >= rows or x < 0 or x >= columns or not text:
            return
        width = columns - x if max_width is None else min(max_width, columns - x)
        if width <= 0:
            return
        self._sink(y, x, text[:width], attr)


def run_debug_tui(
    state: IpuState,
    cycle: int,
    *,
    control: DebugControl | None = None,
    session: DebugViewSession | None = None,
    debug: DebugSession | None = None,
) -> DebugAction:
    """Enter the terminal and return an execution action; interrupts cancel.

    *control* is the one passed to ``run_with_debug``, and *session* the view
    kept from the previous stop; without them, this stop has its own. *debug*
    is the run's ``DebugSession`` (it carries the kernel name the header
    shows); when given, its control is the one used.
    """
    if not terminal_available():
        raise RuntimeError("curses is not available on this platform")
    if session is None:
        session = new_debug_view_session(state)
    session.ensure_default_tab(state)
    session.active = True

    def run(screen: Screen) -> DebugAction:
        return CursesDebugView(screen, state, session, cycle, control, debug).run()

    try:
        return run_in_terminal(run)
    except KeyboardInterrupt:
        session.active = False
        return DebugAction.QUIT
    except Exception as error:
        session.active = False
        raise RuntimeError(f"curses TUI failed: {error}") from error
