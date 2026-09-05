"""Persistent curses frontend for IPU application debugging.

The TUI is the application debugger. The deprecated line debugger is a
separate compatibility API and cannot be entered from this interface.
"""

from __future__ import annotations

import locale
import os
import re
import struct
import sys
import textwrap
import time
import weakref
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - import availability is platform-dependent
    import curses
except ImportError:  # pragma: no cover
    curses = None  # type: ignore[assignment]

from ipu_emu.debug_cli import (
    _CELL16_BACKGROUND_SHIFT,
    _CELL16_CHARACTER_MASK,
    _CELL16_CHARACTERS_PER_GROUP,
    _CELL16_COLOR_MASK,
    _CELL16_FOREGROUND_SHIFT,
    _F32_DISPLAY_WIDTH,
    _F32_SIGNIFICANT_DIGITS,
    _HEX_BYTES_PER_GROUP,
    _INT8_DECIMAL_WIDTH,
    _U8_DECIMAL_WIDTH,
    _U32_DECIMAL_WIDTH,
    decode_16bit_cell,
)
from ipu_emu.descriptors import REGFILE_SCHEMA
from ipu_emu.emulator import DebugAction
from ipu_emu.debug_control import get_debug_control
from ipu_emu.debug_pipeline import pipeline_occupancy, SLOT_LABELS, display_operation, semantic_operations
from ipu_emu.errors import EmulatorError
from ipu_emu.ipu import xmem_row_size_bytes
from ipu_emu.ipu_state import IpuState, INST_MEM_SIZE

if TYPE_CHECKING:
    from ipu_emu.debug_cli import ResolvedXmemRequest, XmemRequest


MIN_TERMINAL_ROWS = 24
MIN_TERMINAL_COLUMNS = 80
MIN_COMPACT_ROWS = 12
MIN_COMPACT_COLUMNS = 48
DOUBLE_CLICK_SECONDS = 0.35
INPUT_POLL_MS = 100
# Escape also prefixes arrows/function keys; bound that ambiguity wait.
ESCAPE_DELAY_MS = 25
MAX_TAB_LABEL_WIDTH = 24
HEADER_ROWS = 1
MIN_DISASSEMBLY_COLUMNS = 30
MIN_XMEM_COLUMNS = 38
MIN_PIPELINE_COLUMNS = 42
# Every pane spends its first row on a border, its second on a tab bar and its
# last on a border; the disassembly and register panes spend one more on a
# column header.  The constants keep the drawing code and the cursor arithmetic
# reading from the same numbers.
XMEM_NON_CONTENT_ROWS = 3
PIPELINE_NON_CONTENT_ROWS = 3
DISASSEMBLY_NON_CONTENT_ROWS = 4
REGISTER_NON_CONTENT_ROWS = 4
# One content column on the right of a vertically scrollable pane is always
# reserved for its scroll thumb, so the layout never reflows when the thumb
# appears or disappears.
SCROLLBAR_COLUMNS = 1
VISIBLE_TAB_LOOKBEHIND = 2
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
# Content starts this far inside a pane's left border and stops one column
# short of the right one, so no value is ever painted against a frame.
PANE_CONTENT_INDENT = 2
PANE_CONTENT_MARGIN = 1
# Border columns kept between a pane title and a readout sharing its row.
TITLE_READOUT_GAP = 2
REGISTER_GROUPS = ("lr", "cr")
PIPELINE_PAGE_FRACTION_NUMERATOR = 2
PIPELINE_PAGE_FRACTION_DENOMINATOR = 5
BITS_PER_REGISTER = 32
PANE_FOCUS_ORDER = ("pipeline", "disassembly", "xmem", "registers")
EMPTY_DISASSEMBLY_OPERATIONS = ("<nop>", "<NOP>", "NOP")
# Parallel slots are stacked under their instruction and marked the way VLIW
# assembly listings mark them, rather than run out across the pane.
DISASSEMBLY_PARALLEL_MARKER = "||"
DISASSEMBLY_PREFIX_WIDTH = len("> 0000  ")
# Number keys jump straight to a pane, in the same order as ``Shift-Tab``.
PANE_FOCUS_KEYS = {
    str(number + 1): pane for number, pane in enumerate(PANE_FOCUS_ORDER)
}
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
# A value made only of these characters is zero-padded rather than signed or
# fractional, so a leading run of zeros in it carries no information.
_PADDED_VALUE_CHARACTERS = frozenset("0123456789abcdef")
# Narrow values such as a single hex byte are left alone: splitting ``01`` into
# a dim and a bright half is more noise than help.
MIN_PADDED_VALUE_WIDTH = 4
_SCALAR_FORMAT_VALUE_WIDTHS = {
    "hex": 8,
    "u32": 10,
    "int32": 11,
    "f32": _F32_DISPLAY_WIDTH,
    "bits": BITS_PER_REGISTER,
}


# Border cells are accumulated as a bit mask of the line segments that leave
# them, so two panes sharing an edge resolve to one junction character instead
# of overwriting each other.
BORDER_UP = 1
BORDER_RIGHT = 2
BORDER_DOWN = 4
BORDER_LEFT = 8


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
        " ", "\u2502", "\u2500", "\u2514", "\u2502", "\u2502", "\u250c", "\u251c",
        "\u2500", "\u2518", "\u2500", "\u2534", "\u2510", "\u2524", "\u252c", "\u253c",
    ),
    thumb="\u2588",
    separator="\u00b7",
    current_row="\u25b6",
    normal_row=" ",
    more_before="\u2039",
    more_after="\u203a",
    ellipsis="\u2026",
)
ASCII_GLYPHS = Glyphs(
    box=(
        " ", "|", "-", "+", "|", "|", "+", "+",
        "-", "+", "-", "+", "+", "+", "+", "+",
    ),
    thumb="#",
    separator="|",
    current_row=">",
    normal_row=" ",
    more_before="<",
    more_after=">",
    ellipsis="...",
)


def register_grid_rows(content_rows: int) -> int:
    """The tallest register-grid shape that fits in *content_rows*."""
    for rows in REGISTER_GRID_ROWS:
        if rows <= content_rows:
            return rows
    return min(REGISTER_GRID_ROWS)


def _detect_glyphs() -> Glyphs:
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


FOOTER_PANE_LABELS = {
    "disassembly": "Disasm",
    "registers": "Regs",
    "xmem": "XMEM",
    "pipeline": "Pipe",
}
HELP_SCOPE_TITLES = (
    ("global", "Execution and layout"),
    ("pane", "Focused pane"),
    ("editor", "Tab editor"),
    ("help", "This overlay"),
)


@dataclass(frozen=True)
class KeyBinding:
    """One documented control, shown in the footer and the help overlay."""

    keys: tuple[str, ...]
    scope: str
    description: str
    footer: str | None = None
    footer_rank: int = 3
    panes: tuple[str, ...] = PANE_FOCUS_ORDER


# Single source of truth for the footer and the help overlay. The footer puts
# execution controls first. ``docs/content/debugging.md`` documents the same controls and is
# synchronised by hand.
KEY_BINDINGS: tuple[KeyBinding, ...] = (
    KeyBinding(("F2",), "pane", "Add a tab to the focused pane", "F2:Add", 3),
    KeyBinding(("F3",), "pane", "Edit the active tab", "F3:Edit", 3),
    KeyBinding(("F4",), "pane", "Close the active tab", "F4:Close", 3),
    KeyBinding(("f",), "pane", "Cycle the active tab's format", "f:Format", 2),
    KeyBinding(
        ("d", "a"),
        "pane",
        "Select the next / previous tab",
        "a:Prev d:Next",
        3,
    ),
    KeyBinding(
        ("Arrows",),
        "pane",
        "Move the cursor inside the focused pane",
        "Arrows:Move",
        2,
    ),
    KeyBinding(("Tab",), "pane", "Select the next tab", "Tab:Next", 2),
    KeyBinding(
        ("PgUp", "PgDn"),
        "pane",
        "Move the cursor by one page",
    ),
    KeyBinding(
        ("Home", "End"),
        "pane",
        "Jump to the first / last value",
    ),
    KeyBinding(
        ("Click",),
        "pane",
        "Select panes, tabs, values, and footer actions",
    ),
    KeyBinding(
        ("Wheel",),
        "pane",
        "Scroll panes, help, or tabs under the pointer",
    ),
    KeyBinding(("Shift-wheel",), "pane", "Move sideways within a pane"),
    KeyBinding(("Double-click",), "pane", "Edit the selected register or memory value"),
    KeyBinding(("Code gutter",), "pane", "Click to toggle an instruction breakpoint"),
    KeyBinding(
        ("Drag",),
        "pane",
        "Drag a pane divider or scrollbar",
    ),
    KeyBinding(("Shift-Tab",), "global", "Focus the next pane", "S-Tab:Pane"),
    KeyBinding(
        ("1-4",),
        "global",
        "Focus a pane directly, left to right",
        None,
        2,
    ),
    KeyBinding(("F5",), "global", "Continue to the next breakpoint", "F5:Run", 0),
    KeyBinding(("F8",), "global", "Execute one instruction", "F8:Step", 0),
    KeyBinding(
        ("F9",), "pane", "Disassembly: toggle breakpoint", "F9:Break", 1,
        panes=("disassembly",),
    ),
    KeyBinding(
        ("F10",), "pane", "Disassembly: run to cursor or next stop", "F10:Until", 1,
        panes=("disassembly",),
    ),
    KeyBinding(
        ("e",), "pane", "Edit the selected register or memory value", "e:Value", 1,
        panes=("registers", "pipeline", "xmem"),
    ),
    KeyBinding(
        ("Shift-Left", "Shift-Right"),
        "global",
        "Move the focused row's split",
        "S-Arrows:Size",
        4,
    ),
    KeyBinding(
        ("Shift-Up", "Shift-Down"),
        "global",
        "Move the top / bottom split",
        footer_rank=4,
    ),
    KeyBinding(("=",), "global", "Reset every split", "=:Reset", 4),
    KeyBinding(("?",), "global", "Show this help overlay"),
    KeyBinding(("q",), "global", "Halt execution and exit the debugger", "q Quit", 0),
    KeyBinding(("Esc",), "editor", "Cancel the editor"),
    KeyBinding(("Enter",), "editor", "Apply the edited value"),
    KeyBinding(
        ("Left", "Right", "Home", "End"),
        "editor",
        "Move within the edited text",
    ),
    KeyBinding(("Esc", "?", "Enter"), "help", "Close this overlay"),
    KeyBinding(
        ("Up", "Down", "PgUp", "PgDn"),
        "help",
        "Scroll this overlay",
    ),
)


@dataclass(frozen=True)
class Rect:
    """Zero-based terminal rectangle."""

    y: int
    x: int
    height: int
    width: int

    def contains(self, y: int, x: int) -> bool:
        return (
            self.y <= y < self.y + self.height
            and self.x <= x < self.x + self.width
        )


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
_PIPELINE_WIDE_VALUE_REGISTERS = frozenset(
    {"r0", "r1", "r_cyclic"} | _PIPELINE_WORD_REGISTERS
)


@dataclass
class XmemTab:
    """One symbolic XMEM request and its persistent presentation state."""

    request: XmemRequest
    scroll: int = 0
    baseline_key: tuple[int, int, str, int] | None = None
    baseline_data: bytes | None = None
    cursor_line: int = 0
    cursor_column: int = 0
    horizontal_scroll: int = 0

    @property
    def title(self) -> str:
        request = self.request
        return (
            f"{request.mode}:{request.base_token}+{request.offset_token} "
            f"{request.fmt}"
        )


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
        spec = next(
            spec for spec in PIPELINE_REGISTERS if spec.key == self.register_key
        )
        return f"{spec.label} {self.fmt or 'auto'}"


@dataclass
class DebugViewSession:
    """State retained between debugger callbacks for one IPU state."""

    active: bool = False
    initialized: bool = False
    tabs: list[XmemTab] = field(default_factory=list)
    active_tab: int = 0
    tab_offset: int = 0
    disassembly_tabs: list[DisassemblyTab] = field(default_factory=list)
    active_disassembly_tab: int = 0
    disassembly_tab_offset: int = 0
    register_tabs: list[RegisterTab] = field(default_factory=list)
    active_register_tab: int = 0
    register_tab_offset: int = 0
    pipeline_tabs: list[PipelineTab] = field(default_factory=list)
    active_pipeline_tab: int = 0
    pipeline_tab_offset: int = 0
    focus: str = "xmem"
    message: str = "Ready"
    message_is_error: bool = False
    # Preferred split sizes survive temporary shrinking of the terminal.
    register_pane_columns: int | None = None
    pipeline_pane_columns: int | None = None
    top_pane_rows: int | None = None
    maximized: bool = False
    stages_scroll: int = 0

    @property
    def stages_visible(self) -> bool:
        return self.selected_disassembly_tab().fmt == "stages"

    @stages_visible.setter
    def stages_visible(self, visible: bool) -> None:
        self.selected_disassembly_tab().fmt = "stages" if visible else "compact"


    def set_message(self, text: str, *, error: bool = False) -> None:
        self.message = text.replace("\n", " ").strip() or "Ready"
        self.message_is_error = error

    def ensure_default_tab(self, state: IpuState) -> None:
        from ipu_emu.debug_cli import XmemRequest

        fresh = not self.initialized
        if fresh and not self.tabs:
            numeric = "f32" if state.wide_vector_debug and state.wide_vector_arithmetic.value == "fp32" else "int8"
            if state.wide_vector_debug and state.wide_vector_arithmetic.value == "int32":
                numeric = "u32"
            self.tabs.extend(XmemTab(XmemRequest("row", "0", "0", xmem_row_size_bytes(state), fmt))
                             for fmt in ("hex", numeric))
        if not self.disassembly_tabs:
            self.disassembly_tabs.extend(DisassemblyTab(fmt=fmt) for fmt in
                                         (DISASSEMBLY_FORMATS if fresh else ("compact",)))
        if not self.register_tabs:
            self.register_tabs.extend(RegisterTab(group=group) for group in
                                      (("all", "lr", "cr") if fresh else ("all",)))
        if not self.pipeline_tabs:
            self.pipeline_tabs.extend(PipelineTab(register_key=spec.key, fmt=_default_pipeline_format(state, spec))
                                      for spec in (PIPELINE_REGISTERS if fresh else PIPELINE_REGISTERS[:1]))
        self.initialized = True

    def selected_tab(self) -> XmemTab | None:
        if not self.tabs:
            return None
        self.active_tab = min(self.active_tab, len(self.tabs) - 1)
        return self.tabs[self.active_tab]

    def selected_disassembly_tab(self) -> DisassemblyTab:
        self.active_disassembly_tab = min(
            self.active_disassembly_tab, len(self.disassembly_tabs) - 1
        )
        return self.disassembly_tabs[self.active_disassembly_tab]

    def selected_register_tab(self) -> RegisterTab:
        self.active_register_tab = min(
            self.active_register_tab, len(self.register_tabs) - 1
        )
        return self.register_tabs[self.active_register_tab]

    def selected_pipeline_tab(self) -> PipelineTab:
        self.active_pipeline_tab = min(
            self.active_pipeline_tab, len(self.pipeline_tabs) - 1
        )
        return self.pipeline_tabs[self.active_pipeline_tab]

    def snapshot_registers(self, state: IpuState) -> dict[tuple[str, int], int]:
        snapshot: dict[tuple[str, int], int] = {}
        for name in ("lr", "cr"):
            for index in range(_SCALAR_REGISTER_COUNTS[name]):
                snapshot[(name, index)] = state.regfile.get_scalar(name, index)
        return snapshot

    def commit_baseline(self, state: IpuState) -> None:
        from ipu_emu.debug_cli import _resolve_xmem_request

        register_snapshot = self.snapshot_registers(state)
        for tab in self.register_tabs:
            tab.baseline = register_snapshot.copy()
        for tab in self.pipeline_tabs:
            spec = _pipeline_spec(tab.register_key)
            tab.baseline_key = spec.key
            tab.baseline_data = _read_pipeline_register(state, spec)
        for tab in self.tabs:
            resolved, error = _resolve_xmem_request(state, tab.request)
            if error is not None or resolved is None:
                tab.baseline_key = None
                tab.baseline_data = None
                continue
            tab.baseline_key = _resolved_key(resolved)
            tab.baseline_data = resolved.data


_SESSIONS: weakref.WeakKeyDictionary[IpuState, DebugViewSession] = (
    weakref.WeakKeyDictionary()
)


def get_debug_view_session(state: IpuState) -> DebugViewSession:
    """Return the persistent TUI session associated with *state*."""
    session = _SESSIONS.get(state)
    if session is None:
        session = DebugViewSession()
        _SESSIONS[state] = session
    session.ensure_default_tab(state)
    return session


def _resolved_key(resolved: ResolvedXmemRequest) -> tuple[int, int, str, int]:
    return (
        resolved.byte_address,
        resolved.byte_count,
        resolved.request.fmt,
        resolved.row_size,
    )


def _read_pipeline_register(
    state: IpuState,
    spec: PipelineRegisterSpec,
) -> bytes:
    backing = (
        spec.wide_backing
        if state.wide_vector_debug and spec.wide_backing is not None
        else spec.backing
    )
    descriptor = state.regfile._desc(backing)
    raw = state.regfile.raw(backing)
    start = spec.index * descriptor.size_bytes
    return bytes(raw[start : start + descriptor.size_bytes])


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
    if not state.wide_vector_debug and state.dtype.value == 0 and spec.key in ("r0", "r1", "r_cyclic"):
        return "int8"
    return _pipeline_format(state, spec)


def _pipeline_spec(register_key: str) -> PipelineRegisterSpec:
    return next(spec for spec in PIPELINE_REGISTERS if spec.key == register_key)


def _compact_color_set(colors: set[int]) -> str:
    if colors == set(range(16)):
        return "0-f"
    if len(colors) <= 4:
        return "/".join(f"{color:x}" for color in sorted(colors))
    return f"{len(colors)} values"


def _cell16_color_summary(data: bytes) -> str:
    foregrounds: set[int] = set()
    backgrounds: set[int] = set()
    for offset in range(0, len(data), 2):
        cell = struct.unpack_from("<H", data, offset)[0]
        foregrounds.add(
            (cell >> _CELL16_FOREGROUND_SHIFT) & _CELL16_COLOR_MASK
        )
        backgrounds.add(
            (cell >> _CELL16_BACKGROUND_SHIFT) & _CELL16_COLOR_MASK
        )
    return (
        f"fg={_compact_color_set(foregrounds)} "
        f"bg={_compact_color_set(backgrounds)}"
    )


def _overlay_attr(base: int, overlay: int) -> int:
    """Combine emphasis, replacing a color pair instead of OR-ing pair IDs."""
    color_mask = getattr(curses, "A_COLOR", 0) if curses else 0
    if overlay & color_mask:
        base &= ~color_mask
    return base | overlay


class _ColorManager:
    """Best-effort curses colors with monochrome fallbacks."""

    def __init__(self) -> None:
        self.enabled = False
        self._pairs: dict[tuple[int, int], int] = {}
        self.current = getattr(curses, "A_BOLD", 0) if curses else 0
        # Emphasis is layered, so no two of these may be the same attribute:
        # ``changed`` marks a value, ``selected`` the cursor under it, and
        # ``focus`` the frame of the pane they are in.
        self.changed = self.current | (
            getattr(curses, "A_UNDERLINE", 0) if curses else 0
        )
        self.selected = getattr(curses, "A_REVERSE", 0) if curses else 0
        # Every pane shows where its cursor is; only the focused one shows it
        # at full strength.
        self.cursor = self.selected | (
            getattr(curses, "A_DIM", 0) if curses else 0
        )
        self.header = self.current
        self.paused = self.current | self.selected
        self.breakpoint = self.changed
        self.pc_marker = self.current
        self.key = self.current
        self.border = getattr(curses, "A_DIM", 0) if curses else 0
        self.plain = self.current
        self.title = self.current
        self.focus = self.current
        self.status = getattr(curses, "A_REVERSE", 0) if curses else 0
        self.error = self.current
        self.cell_selected = (
            getattr(curses, "A_UNDERLINE", 0) if curses else 0
        )
        self.cell_changed = (
            getattr(curses, "A_ITALIC", 0) if curses else 0
        ) or self.cell_selected
        if curses is None:
            return
        try:
            if curses.has_colors():
                curses.start_color()
                try:
                    curses.use_default_colors()
                except curses.error:
                    pass
                self.enabled = True
                self.header = 0  # Respect the terminal's light or dark theme.
                self.paused = self._pair(0, 3)
                self.breakpoint = self._pair(9, -1) | self.current
                self.pc_marker = self._pair(10, -1) | self.current
                self.key = self._pair(14, -1) | self.current
                self.border = self._pair(8, -1)
                self.plain = self.current
                self.title = self._pair(14, -1) | self.current
                self.focus = self._pair(14, -1) | self.current
                self.status = 0
                self.error = self._pair(15, 1) | self.current
                self.selected = self._pair(0, 6)
                # A dark foreground would be brightened into the background
                # by the bold attribute rows carry, so the muted cursor keeps a
                # light foreground and only drops the accent colour.
                self.cursor = self._pair(15, 8)
        except curses.error:
            self.enabled = False

    def _pair(self, fg: int, bg: int) -> int:
        if not self.enabled or curses is None:
            return 0
        color_count = max(1, min(getattr(curses, "COLORS", 8), 16))
        normalized = (fg % color_count, bg if bg < 0 else bg % color_count)
        pair = self._pairs.get(normalized)
        if pair is None:
            pair = len(self._pairs) + 1
            if pair >= getattr(curses, "COLOR_PAIRS", 1):
                return 0
            try:
                curses.init_pair(pair, *normalized)
            except curses.error:
                return 0
            self._pairs[normalized] = pair
        try:
            return curses.color_pair(pair)
        except curses.error:
            return 0

    def cell(self, fg: int, bg: int) -> int:
        if not self.enabled or curses is None:
            return 0
        attr = self._pair(fg, bg)
        if fg >= 8:
            attr |= getattr(curses, "A_BOLD", 0)
        return attr


class CursesDebugView:
    """Controller and renderer for one curses entry."""

    def __init__(
        self,
        screen: Any,
        state: IpuState,
        session: DebugViewSession,
        cycle: int,
    ) -> None:
        self.screen = screen
        self.state = state
        self.control = get_debug_control(self.state)
        self.session = session
        self.cycle = cycle
        self.colors = _ColorManager()
        self.glyphs = _detect_glyphs()
        # ``(y, x) -> (segment mask, attribute, priority)`` for the pane frames.
        self.borders: dict[tuple[int, int], tuple[int, int, int]] = {}
        self.command = ""
        self.cursor = 0
        self.editor_mode: str | None = None
        self.editor_pane: str | None = None
        self.editor_register: tuple[str, int] | None = None
        # Overlay state is per curses entry, so help never survives a step.
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
        self.editor_value = None
        self.last_value_click: tuple[int, int, float] | None = None
        self._last_terminal_size = self.screen.getmaxyx()
        # ``(split, origin_y, origin_x, moved)`` while a border is held.
        self.drag: tuple[str, int, int, bool] | None = None
        # Decoding an instruction is the single most expensive thing a render
        # does, and instruction memory cannot change while the debugger is
        # stopped, so each PC is decoded at most once per curses entry.
        self._disassembly_cache: dict[int, str] = {}
        # XMEM is frozen for the same reason, and one request can tokenise into
        # far more lines than a pane shows.
        self._xmem_cache: dict[tuple[int, int, str, int], list[Any]] = {}
        self._xmem_requests: dict[
            XmemRequest, tuple[ResolvedXmemRequest | None, str | None]
        ] = {}
        self.redraw = True
        self.disassembly_rect = Rect(0, 0, 0, 0)
        self.registers_rect = Rect(0, 0, 0, 0)
        self.xmem_rect = Rect(0, 0, 0, 0)
        self.pipeline_rect = Rect(0, 0, 0, 0)
        self.editor_rect = Rect(0, 0, 0, 0)

    def run(self) -> DebugAction:
        self._configure_terminal()
        while True:
            self._apply_terminal_resize()
            if self.redraw:
                self.render()
                self.redraw = False
            try:
                key = self.screen.get_wch()
            except KeyboardInterrupt:
                self.session.active = False
                return DebugAction.QUIT
            except _CURSES_ERRORS:
                # A terminal resize interrupts the blocking read, and ncurses
                # reports that as "no input" rather than as ``KEY_RESIZE``.
                # Re-measure and redraw instead of falling out of the view.
                self._apply_terminal_resize()
                continue
            outcome = self.handle_key(key)
            if outcome is not _NO_OUTCOME:
                return outcome
            # Pointer motion arrives far faster than a frame can be drawn, so
            # everything already queued is handled before the next redraw.
            try:
                with closing(self._drain_input()) as pending_keys:
                    for pending in pending_keys:
                        outcome = self.handle_key(pending)
                        if outcome is not _NO_OUTCOME:
                            return outcome
            except KeyboardInterrupt:
                self.session.active = False
                return DebugAction.QUIT

    def _drain_input(self) -> Iterator[Any]:
        """Read and handle each queued event before consuming the next one."""
        try:
            self.screen.nodelay(True)
        except (AttributeError, *_CURSES_ERRORS):
            return
        try:
            for _ in range(MAX_COALESCED_KEYS):
                try:
                    key = self.screen.get_wch()
                except _CURSES_ERRORS:
                    break
                if key is None or key == -1:
                    break
                yield key
        finally:
            try:
                self.screen.nodelay(False)
                self.screen.timeout(INPUT_POLL_MS)
            except (AttributeError, *_CURSES_ERRORS):
                pass

    def _apply_terminal_resize(self) -> None:
        """Use the actual PTY size, even when ncurses misses a resize event."""
        if curses is None:
            return
        try:
            size = os.get_terminal_size(sys.stdin.fileno())
            dimensions = (size.lines, size.columns)
        except (AttributeError, OSError, ValueError):
            dimensions = self.screen.getmaxyx()
        if min(dimensions) <= 0 or dimensions == self._last_terminal_size:
            return
        try:
            if dimensions != self.screen.getmaxyx():
                curses.resizeterm(*dimensions)
        except (AttributeError, ValueError, curses.error):
            return
        self._last_terminal_size = dimensions
        self.redraw = True
        self.drag = None
        self.scroll_drag = None
        self.scroll_origin = None
        self.last_value_click = None
        try:
            self.screen.clear()
        except (AttributeError, curses.error):
            pass

    def _configure_terminal(self) -> None:
        # timeout()/nodelay() do not bound ncurses escape-sequence decoding.
        # Configure this separately so cursor/mouse support cannot skip it.
        if curses is not None:
            try:
                curses.set_escdelay(ESCAPE_DELAY_MS)
            except (AttributeError, curses.error):
                pass
        try:
            self.screen.keypad(True)
            self.screen.nodelay(False)
            try:
                self.screen.timeout(INPUT_POLL_MS)
            except AttributeError:
                pass
            if curses is not None:
                curses.curs_set(0)
                try:
                    curses.mousemask(
                        curses.ALL_MOUSE_EVENTS
                        | getattr(curses, "REPORT_MOUSE_POSITION", 0)
                    )
                    curses.mouseinterval(0)
                except curses.error:
                    pass
        except (AttributeError, curses.error if curses else Exception):
            pass

    def _natural_register_columns(self, top_rows: int) -> int:
        """The width the active LR/CR tab needs at the grid shape it will use."""
        tab = self.session.selected_register_tab()
        grid_rows = register_grid_rows(
            max(1, top_rows - REGISTER_NON_CONTENT_ROWS)
        )
        groups = self._register_groups(tab)
        columns = len(groups) * _REGISTERS_PER_GROUP // grid_rows
        column_width = (
            REGISTER_LABEL_WIDTH + _SCALAR_FORMAT_VALUE_WIDTHS[tab.fmt]
        )
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
            columns
            // 2,
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
        rows, columns = self.screen.getmaxyx()
        top_rows = rows - HEADER_ROWS - FOOTER_ROWS - self._clamped_top_pane_rows(rows) + 1
        if self.session.focus in ("disassembly", "pipeline"):
            current = self._clamped_pipeline_columns(columns)
            self.session.pipeline_pane_columns = current + delta
            width = self._clamped_pipeline_columns(columns)
            self.session.pipeline_pane_columns = width
            self.session.set_message(
                f"Pipeline pane width: {width} columns"
            )
            return
        current = self._clamped_register_columns(columns, top_rows)
        self.session.register_pane_columns = current - delta
        width = self._clamped_register_columns(columns, top_rows)
        self.session.register_pane_columns = width
        self.session.set_message(f"LR / CR pane width: {width} columns")

    def _resize_horizontal_split(self, delta: int) -> None:
        if self._single_pane():
            return
        rows, _ = self.screen.getmaxyx()
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
        rows, columns = self.screen.getmaxyx()
        return (self.session.maximized or rows < MIN_TERMINAL_ROWS
                or columns < MIN_TERMINAL_COLUMNS)

    def layout(self) -> dict[str, Rect] | None:
        rows, columns = self.screen.getmaxyx()
        if rows < MIN_COMPACT_ROWS or columns < MIN_COMPACT_COLUMNS:
            return None
        if self._single_pane():
            return {
                "header": Rect(0, 0, HEADER_ROWS, columns),
                self.session.focus: Rect(
                    HEADER_ROWS, 0, rows - HEADER_ROWS - FOOTER_ROWS, columns
                ),
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
            "disassembly": Rect(top_y, pipeline_width - 1, top_rows,
                                columns - pipeline_width + 1),
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
        self.screen.erase()
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
            rows, columns = self.screen.getmaxyx()
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
        for pane, draw in (
            ("disassembly", self._draw_disassembly),
            ("registers", self._draw_registers),
            ("xmem", self._draw_xmem),
            ("pipeline", self._draw_pipeline_register),
        ):
            if pane in layout:
                draw(layout[pane])
        self._draw_scrollbars()
        self._draw_keys(layout["keys"])
        if self.help_visible:
            self._draw_help()
        elif self.editor_mode:
            self._draw_editor()
        self._refresh()
        self.redraw = False

    def _refresh(self) -> None:
        try:
            self.screen.refresh()
        except _CURSES_ERRORS:
            pass

    def _draw_header(self, rect: Rect) -> None:
        mode = "wide" if self.state.wide_vector_debug else "normal"
        segments = [
            self.control.kernel_name or "IPU DEBUG",
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
            message = message[:max(0, message_width - 1)] + self.glyphs.ellipsis
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
            self._add(rect.y, x - len(message) - 2, message,
                      self.colors.error if self.session.message_is_error else self.colors.status,
                      len(message))
        self._add(rect.y, x, label, self.colors.key, len(label))
        self.action_hits.append((Rect(rect.y, x, 1, len(label)), "?"))

    def _pane_tabs(self, pane: str) -> list[Any]:
        if pane == "disassembly":
            return self.session.disassembly_tabs
        if pane == "registers":
            return self.session.register_tabs
        if pane == "pipeline":
            return self.session.pipeline_tabs
        return self.session.tabs

    @staticmethod
    def _pane_active_attribute(pane: str) -> str:
        return {
            "disassembly": "active_disassembly_tab",
            "registers": "active_register_tab",
            "xmem": "active_tab",
            "pipeline": "active_pipeline_tab",
        }[pane]

    @staticmethod
    def _pane_offset_attribute(pane: str) -> str:
        return {
            "disassembly": "disassembly_tab_offset",
            "registers": "register_tab_offset",
            "xmem": "tab_offset",
            "pipeline": "pipeline_tab_offset",
        }[pane]

    def _active_pane_tab_index(self, pane: str) -> int:
        return getattr(self.session, self._pane_active_attribute(pane))

    def _set_active_pane_tab_index(self, pane: str, index: int) -> None:
        setattr(self.session, self._pane_active_attribute(pane), index)

    def _draw_tab_bar(self, pane: str, rect: Rect) -> None:
        tabs = self._pane_tabs(pane)
        active = self._active_pane_tab_index(pane)
        offset_attribute = self._pane_offset_attribute(pane)
        offset = getattr(self.session, offset_attribute)

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
        add_attr = (
            self.colors.current
            if pane == self.session.focus
            else self.colors.border
        )
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
                len(tab_label(tabs[index])) + 1
                for index in range(offset, active + 1)
            )
            if active_span <= available_width:
                break
            offset += 1
        setattr(self.session, offset_attribute, offset)
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
                attr = (
                    self.colors.selected
                    if pane == self.session.focus
                    else self.colors.current
                )
            self._add(tab_y, x, label, attr)
            hit = Rect(tab_y, x, 1, len(label))
            self.tab_hits[pane].append((hit, index))
            self.tab_close_hits[pane].append(
                (Rect(tab_y, x + len(label) - 2, 1, 1), index)
            )
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

    def _resolve_xmem(
        self, request: XmemRequest
    ) -> tuple[ResolvedXmemRequest | None, str | None]:
        from ipu_emu.debug_cli import _resolve_xmem_request

        if request not in self._xmem_requests:
            self._xmem_requests[request] = _resolve_xmem_request(self.state, request)
        return self._xmem_requests[request]

    def _xmem_lines(self, resolved: ResolvedXmemRequest) -> list[Any]:
        key = _resolved_key(resolved)
        lines = self._xmem_cache.get(key)
        if lines is None:
            from ipu_emu.debug_cli import _tokenize_xmem

            lines = _tokenize_xmem(
                resolved.data,
                resolved.byte_address,
                resolved.request.fmt,
                resolved.row_size,
            )
            self._xmem_cache[key] = lines
        return lines

    def _disassembly(self, pc: int) -> str:
        text = self._disassembly_cache.get(pc)
        if text is None:
            from ipu_emu.debug_cli import disassemble_at

            text = disassemble_at(self.state, pc)
            operations = self._all_operations(text)
            if len(operations) == len(SLOT_LABELS):
                text = f"PC {pc}: " + ";\n".join(semantic_operations(operations)) + ";;"
            self._disassembly_cache[pc] = text
        return text

    def _draw_disassembly(self, rect: Rect) -> None:
        tab = self.session.selected_disassembly_tab()
        self._draw_tab_bar("disassembly", rect)
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
                attr = _overlay_attr(attr, cursor_attr)
            y = rect.y + 3 + row
            x = rect.x + 1
            limit = x + row_width
            self.value_hits.setdefault("disassembly", []).append(
                (Rect(y, x, 1, row_width), pc)
            )
            if slot:
                # The parallel-slot marker recedes without breaking the bar
                # the cursor draws behind it.
                prefix = f"{'':{DISASSEMBLY_PREFIX_WIDTH - 3}}"
                self._add(y, x, prefix, attr, limit - x)
                self._add(
                    y,
                    x + len(prefix),
                    DISASSEMBLY_PARALLEL_MARKER,
                    _overlay_attr(self.colors.border, attr),
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
                marker = (
                    self.glyphs.current_row
                    if is_current
                    else self.glyphs.normal_row
                )
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
            text = (
                f"{mnemonic.ljust(mnemonic_width)} {operands}"
                if operands
                else mnemonic
            )
            available = max(0, limit - x)
            if len(text) > available and available >= len(self.glyphs.ellipsis):
                text = text[:available - len(self.glyphs.ellipsis)] + self.glyphs.ellipsis
            self._add(y, x, text.ljust(available), attr, available)
            if not is_empty and pc != centered_pc:
                self._add(y, x, mnemonic, self.colors.key, min(available, len(mnemonic)))
            if row_width >= 60:
                if pc not in slot_labels:
                    ops = self._all_operations(self._disassembly(pc))
                    slot_labels[pc] = [
                        name for name, op in zip(SLOT_LABELS, ops)
                        if tab.fmt == "full" or op.upper().split(" ", 1)[0] != "NOP"
                    ] if self.state.inst_mem[pc] is not None else []
                names = slot_labels[pc]
                label = names[slot] if slot < len(names) else ""
                self._add(y, limit, label.rjust(11) + " ",
                          _overlay_attr(self.colors.key, attr), 12)
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
        self._add(rect.y + 2, rect.x + 1, f"Pipeline stages / PC {pc:04d} / cycle {self.cycle} / ready",
                  self.colors.border, width)
        lines: list[tuple[str, int]] = []
        for stage, commands in pipeline_occupancy(instruction):
            if not commands:
                lines.append((f"{stage:<10} IDLE", self.colors.border))
            for index, (slot, operation) in enumerate(commands):
                name = stage if index == 0 else ""
                text = f"{name:<10} {display_operation(operation)}"
                lines.extend((line, self.colors.key) for line in
                             textwrap.wrap(text, width, subsequent_indent="  "))
        visible = max(1, rect.height - 4)
        self.session.stages_scroll = max(0, min(self.session.stages_scroll, len(lines) - visible))
        first = self.session.stages_scroll
        self._draw_top_pane_header(
            rect, ["2 Instructions", "Stages"],
            f" {first + 1}-{min(len(lines), first + visible)}/{len(lines)} ",
            focused=self.session.focus == "disassembly",
        )
        for index, (line, attr) in enumerate(lines[first:first + visible]):
            self._add(rect.y + 3 + index, rect.x + 1, line, attr, width)

    def _draw_registers(self, rect: Rect) -> None:
        tab = self.session.selected_register_tab()
        self._draw_tab_bar("registers", rect)
        groups = self._register_groups(tab)
        grid_rows = register_grid_rows(
            max(1, rect.height - REGISTER_NON_CONTENT_ROWS)
        )
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
                (content_width + REGISTER_MIN_GUTTER)
                // (column_width + REGISTER_MIN_GUTTER),
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
                f"{name.upper()} {first_index:02d}-"
                f"{first_index + grid_rows - 1:02d}",
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
            f" cols {tab.column_scroll + 1}-"
            f"{tab.column_scroll + visible_columns}/{column_count} "
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
        rows = register_grid_rows(
            max(1, self.registers_rect.height - REGISTER_NON_CONTENT_ROWS)
        )
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
            label_attr = _overlay_attr(label_attr, cursor_attr)
            value_attr = _overlay_attr(value_attr, cursor_attr)
        self.value_hits.setdefault("registers", []).append(
            (Rect(y, x, 1, width), visual_index)
        )
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
        return (
            self.colors.selected
            if self.session.focus == pane
            else self.colors.cursor
        )

    def _value_attr(self, is_zero: bool) -> int:
        """Zero values recede, so the bytes that carry data stand out."""
        return self.colors.border if is_zero else 0

    def _add_value(
        self,
        y: int,
        x: int,
        text: str,
        attr: int,
        max_width: int,
    ) -> None:
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
            self._add(y, x, text[:padding], _overlay_attr(self.colors.border, attr), max_width)
            self._add(
                y,
                x + padding,
                significant,
                attr,
                max_width - padding,
            )
            return
        self._add(y, x, text, attr, max_width)

    @staticmethod
    def _format_scalar_value(value: int, fmt: str, width: int) -> str:
        unsigned = value & 0xFFFFFFFF
        if fmt == "hex":
            text = f"{unsigned:08x}"
        elif fmt == "u32":
            text = f"{unsigned:0{_U32_DECIMAL_WIDTH}d}"
        elif fmt == "int32":
            text = str(struct.unpack("<i", struct.pack("<I", unsigned))[0])
        elif fmt == "f32":
            value = struct.unpack("<f", struct.pack("<I", unsigned))[0]
            text = f"{value:>{_F32_DISPLAY_WIDTH}.{_F32_SIGNIFICANT_DIGITS}g}"
        else:
            text = CursesDebugView._format_bits(unsigned)
        return text.rjust(width)[:width]

    @staticmethod
    def _format_bits(value: int) -> str:
        return f"{value:0{BITS_PER_REGISTER}b}"

    def _draw_xmem(self, rect: Rect) -> None:
        focused = self.session.focus == "xmem"
        self._draw_tab_bar("xmem", rect)

        tab = self.session.selected_tab()
        content_y = rect.y + 2
        content_height = max(0, rect.height - XMEM_NON_CONTENT_ROWS)
        limit = rect.x + rect.width - 1 - SCROLLBAR_COLUMNS
        if tab is None:
            self._draw_pane_title(rect, ["3 XMEM", "no tabs"], focused=focused)
            self._add(
                content_y,
                rect.x + 2,
                "No tabs. Press F2 or click [+].",
                self.colors.border,
                max(0, limit - rect.x - 2),
            )
            return

        resolved, error = self._resolve_xmem(tab.request)
        if error is not None or resolved is None:
            self._draw_pane_title(
                rect,
                ["3 XMEM", tab.request.fmt],
                focused=focused,
            )
            self._add(
                content_y,
                rect.x + 2,
                f"Error: {error}",
                self.colors.error,
                max(0, limit - rect.x - 2),
            )
            return
        lines = self._xmem_lines(resolved)
        self._normalize_xmem_cursor(tab, lines)
        max_scroll = max(0, len(lines) - content_height)
        if self.session.focus == "xmem" and content_height > 0:
            if tab.cursor_line < tab.scroll:
                tab.scroll = tab.cursor_line
            elif tab.cursor_line >= tab.scroll + content_height:
                tab.scroll = tab.cursor_line - content_height + 1
        tab.scroll = min(tab.scroll, max_scroll)
        self._ensure_xmem_cursor_visible_horizontally(tab, lines, rect)
        first_visible = min(len(lines), tab.scroll + 1)
        last_visible = min(len(lines), tab.scroll + content_height)
        segments = [
            "3 XMEM",
            f"0x{resolved.byte_address:08x}",
            f"{resolved.byte_count} bytes",
            resolved.request.fmt,
        ]
        if resolved.request.fmt == "cell16":
            segments.append(_cell16_color_summary(resolved.data))
        self._draw_pane_title(rect, segments, focused=focused)
        # The title fills the top border at the 80-column minimum, so the
        # position readout goes on the bottom border instead.
        self._draw_position(
            rect,
            f" lines {first_visible}-{last_visible}/{len(lines)} ",
        )
        self.scroll_state["xmem"] = (tab.scroll, content_height, len(lines))
        baseline_matches = tab.baseline_key == _resolved_key(resolved)
        baseline = tab.baseline_data if baseline_matches else None

        visible_line_numbers = range(
            tab.scroll,
            min(len(lines), tab.scroll + content_height),
        )
        for output_row, line_number in enumerate(visible_line_numbers):
            line = lines[line_number]
            y = content_y + output_row
            x = rect.x + 1
            prefix = line.prefix
            prefix_attr = 0
            if not line.tokens:
                prefix = self._clean_xmem_marker(prefix)
                prefix_attr = self.colors.border
            elif prefix.strip():
                prefix_attr = self.colors.border
            self._add(
                y,
                x,
                prefix,
                prefix_attr,
                max_width=max(0, limit - x),
            )
            x += len(prefix)
            visible_tokens = enumerate(
                line.tokens[tab.horizontal_scroll :],
                start=tab.horizontal_scroll,
            )
            for token_number, token in visible_tokens:
                if x + len(token.leading) + len(token.text) > limit:
                    break
                self.value_hits.setdefault("xmem", []).append(
                    (
                        Rect(y, x, 1, len(token.leading) + len(token.text)),
                        (line_number, token_number),
                    )
                )
                self._add(y, x, token.leading, max_width=limit - x)
                x += len(token.leading)
                is_colored_cell = token.fg is not None and token.bg is not None
                raw = resolved.data[
                    token.raw_start : token.raw_start + token.raw_size
                ]
                attr = (
                    self.colors.cell(token.fg, token.bg)
                    if is_colored_cell
                    else self._value_attr(not any(raw))
                )
                changed = False
                if baseline is not None:
                    current = raw
                    previous = baseline[
                        token.raw_start : token.raw_start + token.raw_size
                    ]
                    changed = current != previous
                    if changed:
                        attr |= (
                            self.colors.cell_changed
                            if is_colored_cell
                            else self.colors.changed
                        )
                if (line_number, token_number) == (
                    tab.cursor_line,
                    tab.cursor_column,
                ):
                    attr = _overlay_attr(attr, (
                        self.colors.cell_selected
                        if is_colored_cell
                        else self._cursor_attr("xmem")
                    ))
                if is_colored_cell:
                    self._add(y, x, token.text, attr, limit - x)
                else:
                    self._add_value(y, x, token.text, attr, limit - x)
                x += len(token.text)

    @staticmethod
    def _clean_xmem_marker(prefix: str) -> str:
        if prefix.startswith("--- ") and prefix.endswith(" ---"):
            return "  " + prefix[4:-4]
        return prefix

    @staticmethod
    def _normalize_xmem_cursor(tab: XmemTab, lines: list[Any]) -> None:
        data_lines = [index for index, line in enumerate(lines) if line.tokens]
        if not data_lines:
            tab.cursor_line = 0
            tab.cursor_column = 0
            tab.horizontal_scroll = 0
            return
        if tab.cursor_line not in data_lines:
            tab.cursor_line = data_lines[0]
        token_count = len(lines[tab.cursor_line].tokens)
        tab.cursor_column = min(tab.cursor_column, token_count - 1)
        tab.horizontal_scroll = min(
            tab.horizontal_scroll,
            tab.cursor_column,
        )

    @staticmethod
    def _ensure_xmem_cursor_visible_horizontally(
        tab: XmemTab,
        lines: list[Any],
        rect: Rect,
    ) -> None:
        if not lines or not lines[tab.cursor_line].tokens:
            return
        line = lines[tab.cursor_line]
        available = max(
            1,
            rect.width - 2 - SCROLLBAR_COLUMNS - len(line.prefix),
        )
        if tab.cursor_column < tab.horizontal_scroll:
            tab.horizontal_scroll = tab.cursor_column
        while tab.horizontal_scroll < tab.cursor_column:
            tokens = line.tokens[tab.horizontal_scroll : tab.cursor_column + 1]
            width = sum(len(token.leading) + len(token.text) for token in tokens)
            if width <= available:
                break
            tab.horizontal_scroll += 1

    def _draw_pipeline_register(self, rect: Rect) -> None:
        tab = self.session.selected_pipeline_tab()
        spec = _pipeline_spec(tab.register_key)
        data = _read_pipeline_register(self.state, spec)
        fmt = tab.fmt or _pipeline_format(self.state, spec)
        self._draw_tab_bar("pipeline", rect)

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
        cursor_line = next(index for index, first in enumerate(display_rows)
                           if first is not None and first <= tab.cursor < first + items_per_line)
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
            rect, ["1 Pipeline Registers", fmt],
            f" items {visible_first + 1}-{visible_last}/{item_count} ",
            focused=self.session.focus == "pipeline",
        )
        for output_row, line_index in enumerate(range(first_line, last_line)):
            first_item = display_rows[line_index]
            y = content_y + output_row
            if first_item is None:
                self._add(y, content_x + 2, "------------", self.colors.border, limit - content_x - 2)
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
                        cell & _CELL16_CHARACTER_MASK,
                        (cell >> _CELL16_FOREGROUND_SHIFT)
                        & _CELL16_COLOR_MASK,
                        (cell >> _CELL16_BACKGROUND_SHIFT)
                        & _CELL16_COLOR_MASK,
                    )
                else:
                    text = self._format_pipeline_value(raw, fmt, value_width)
                is_colored_cell = (
                    foreground is not None and background is not None
                )
                changed = (
                    baseline is not None
                    and raw != baseline[raw_start : raw_start + item_size]
                )
                selected = item_index == tab.cursor
                attr = (
                    self.colors.cell(foreground, background)
                    if is_colored_cell
                    else self._value_attr(not any(raw))
                )
                if changed:
                    attr |= (
                        self.colors.cell_changed
                        if is_colored_cell
                        else self.colors.changed
                    )
                if selected:
                    attr = _overlay_attr(attr, (
                        self.colors.cell_selected
                        if is_colored_cell
                        else self._cursor_attr("pipeline")
                    ))
                if is_colored_cell:
                    self._add(y, x, text, attr, available)
                elif available < len(text):
                    marker = self.glyphs.ellipsis
                    if len(marker) > available:
                        marker = ">"
                    clipped = text[:available - len(marker)] + marker
                    self._add(y, x, clipped, attr, available)
                else:
                    self._add_value(y, x, text, attr, available)
                x += value_width

    @staticmethod
    def _pipeline_value_geometry(fmt: str) -> tuple[int, int]:
        if fmt == "hex":
            return 1, 2
        if fmt == "u8":
            return 1, _U8_DECIMAL_WIDTH
        if fmt == "int8":
            return 1, _INT8_DECIMAL_WIDTH
        if fmt == "cell16":
            return 2, 1
        if fmt == "u32":
            return 4, _U32_DECIMAL_WIDTH
        if fmt == "int32":
            return 4, 11
        if fmt == "bits":
            return 4, BITS_PER_REGISTER
        return 4, _F32_DISPLAY_WIDTH

    def _pipeline_display_rows(self, fmt: str, rect: Rect, item_count: int) -> list[int | None]:
        per_line = self._pipeline_items_per_line(fmt, rect)
        size, _ = self._pipeline_value_geometry(fmt)
        mask = self.session.selected_pipeline_tab().register_key == "r_mask"
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
        if self.session.selected_pipeline_tab().register_key == "r_mask":
            size, _ = self._pipeline_value_geometry(fmt)
            slot_items = 16 // size
            item_count = max(count for count in range(1, slot_items + 1)
                             if count <= max(1, item_count) and slot_items % count == 0)
        return max(1, item_count)

    def _pipeline_item_spacing(self, fmt: str, item_in_line: int) -> int:
        if item_in_line == 0:
            return 0
        if fmt == "cell16":
            return (
                1
                if item_in_line % _CELL16_CHARACTERS_PER_GROUP == 0
                else 0
            )
        if (
            fmt == "hex"
            and self.state.wide_vector_debug
            and item_in_line % _HEX_BYTES_PER_GROUP == 0
        ):
            return 2
        return 1

    @staticmethod
    def _format_pipeline_value(raw: bytes, fmt: str, width: int) -> str:
        if fmt == "hex":
            return f"{raw[0]:02x}"
        if fmt == "u8":
            return f"{raw[0]:0{_U8_DECIMAL_WIDTH}d}"
        if fmt == "int8":
            return f"{struct.unpack('<b', raw)[0]:>{width}d}"
        if fmt == "u32":
            return f"{struct.unpack('<I', raw)[0]:0{_U32_DECIMAL_WIDTH}d}"
        if fmt == "int32":
            return f"{struct.unpack('<i', raw)[0]:>{width}d}"
        if fmt == "bits":
            return CursesDebugView._format_bits(struct.unpack("<I", raw)[0])
        value = struct.unpack("<f", raw)[0]
        return f"{value:>{width}.{_F32_SIGNIFICANT_DIGITS}g}"

    def _draw_keys(self, rect: Rect) -> None:
        text = self._footer_text(rect.width)
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
            if binding.footer and self.session.focus in binding.panes:
                start = text.find(binding.footer)
                if start >= 0:
                    if binding.keys == ("d", "a"):
                        for label, key in (("a:Prev", "a"), ("d:Next", "d")):
                            self.action_hits.append((
                                Rect(rect.y, rect.x + text.index(label), 1, len(label)), key
                            ))
                        continue
                    key = binding.keys[0]
                    if key.startswith("F") and key[1:].isdigit():
                        key = getattr(curses, "KEY_" + key, None)
                    elif key == "1-4":
                        key = "KEY_BTAB"
                    elif key == "Tab":
                        key = "\t"
                    elif key == "Shift-Tab":
                        key = "KEY_BTAB"
                    elif key == "Arrows":
                        continue
                    elif key.startswith("Shift-"):
                        continue
                    if key == "KEY_BTAB":
                        key = getattr(curses, key, None)
                    if key is not None:
                        self.action_hits.append((Rect(rect.y, rect.x + start, 1, len(binding.footer)), key))

    def _footer_text(self, width: int) -> str:
        """Fit whole shortcut labels, most important first, never clipping."""
        # Execution, navigation, view/editing, layout, exit.
        groups = (("F5", "F8", "F9", "F10"),
                  ("Arrows", "Tab", "d", "Shift-Tab"),
                  ("f", "e", "F2", "F3", "F4"),
                  ("Shift-Left", "Shift-Up", "="), ("q",))
        candidates = [binding for binding in KEY_BINDINGS
                      if binding.footer and binding.scope in ("global", "pane")
                      and self.session.focus in binding.panes]

        def format_bindings(selected: list[KeyBinding]) -> str:
            parts = []
            for group in groups:
                ordered = sorted((binding for binding in selected if binding.keys[0] in group),
                                 key=lambda binding: group.index(binding.keys[0]))
                if ordered:
                    parts.append(" ".join(binding.footer for binding in ordered))
            return " " + " | ".join(parts)

        selected: list[KeyBinding] = []
        for binding in sorted(candidates, key=lambda item: item.footer_rank):
            if len(format_bindings([*selected, binding])) <= width:
                selected.append(binding)
        return format_bindings(selected)[:width]

    def _draw_position(self, rect: Rect, text: str, *, top: bool = False) -> bool:
        """Right-align a scroll-position readout on one of a pane's borders.

        The top panes share their bottom border with the panes below them, so
        they report on their own top border, beside the title, instead.
        """
        if not text or len(text) > rect.width - 8:
            return False
        self._add(
            rect.y if top else rect.y + rect.height - 1,
            max(rect.x + 1, rect.x + rect.width - 2 - len(text)),
            text,
            self.colors.border,
            max(0, rect.width - 2),
        )
        return True

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
            self._add(top + row, column, self.glyphs.box[BORDER_UP | BORDER_DOWN], self.colors.border)
        for row in range(size):
            self._add(
                top + start + row,
                rect.x + rect.width - 1 - SCROLLBAR_COLUMNS,
                self.glyphs.thumb,
                self.colors.focus if self.session.focus == pane else self.colors.border,
            )

    @staticmethod
    def _help_lines(width: int = 72) -> list[tuple[str, int]]:
        """Every binding as ``(text, attribute_kind)`` where 1 marks a heading."""
        lines: list[tuple[str, int]] = []
        key_width = max(
            len(" / ".join(binding.keys)) for binding in KEY_BINDINGS
        )
        for scope, title in HELP_SCOPE_TITLES:
            bindings = [
                binding for binding in KEY_BINDINGS if binding.scope == scope and binding.keys != ("q",)
            ]
            if not bindings:
                continue
            if lines:
                lines.append(("", 0))
            lines.append((title, 1))
            for binding in bindings:
                keys = " / ".join(binding.keys)
                if width >= 60:
                    keys = keys.ljust(key_width)
                indent = " " * (key_width + 2)
                for line in textwrap.wrap(
                    f"{keys}  {binding.description}", width=width,
                    subsequent_indent=indent if width >= 60 else "  ",
                ):
                    lines.append((line, 0))
        return lines

    def _draw_help(self) -> None:
        width = min(self.screen.getmaxyx()[1] - 4, 78)
        lines = self._help_lines(width - 5)
        rect = self._overlay_rect(78, len(lines) + 4)
        for row in range(rect.height):
            self._add(rect.y + row, rect.x, " " * rect.width, max_width=rect.width)
        self._draw_overlay_frame(rect, ["Debugger controls"])
        self._add(rect.y, rect.x + rect.width - 5, "[x]", self.colors.key)
        self.overlay_hits.append((Rect(rect.y, rect.x + rect.width - 6, 1, 5), "\x1b"))
        visible = max(1, rect.height - 3)
        self.help_max_scroll = max(0, len(lines) - visible)
        self.help_scroll = max(0, min(self.help_scroll, self.help_max_scroll))
        if self.help_max_scroll:
            self.help_track = Rect(rect.y + 1, rect.x + rect.width - 2, visible, 1)
            size = max(1, round(visible * visible / len(lines)))
            start = round((visible - size) * self.help_scroll / self.help_max_scroll)
            self.help_thumb = Rect(self.help_track.y + start, self.help_track.x, size, 1)
            for row in range(visible):
                self._add(self.help_track.y + row, self.help_track.x,
                          self.glyphs.thumb if start <= row < start + size else self.glyphs.box[BORDER_UP | BORDER_DOWN],
                          self.colors.key if start <= row < start + size else self.colors.border, 1)
        for row, (text, kind) in enumerate(
            lines[self.help_scroll : self.help_scroll + visible]
        ):
            self._add(
                rect.y + 1 + row,
                rect.x + 2,
                text,
                self.colors.current if kind else 0,
                rect.width - 4,
            )
        footer = ("Esc Close / Wheel Scroll" if rect.width < 70
                  else "Esc / ? Close   Up Down PgUp PgDn Scroll")
        if len(lines) > visible:
            footer += f"   {self.help_scroll + 1}-" \
                f"{min(len(lines), self.help_scroll + visible)}/{len(lines)}"
        self._add(
            rect.y + rect.height - 1,
            max(rect.x + 1, rect.x + rect.width - 2 - len(footer) - 2),
            f" {footer} ",
            self.colors.border,
            rect.width - 2,
        )

    def _handle_help_mouse(self, y: int, x: int, pressed: bool,
                           released: bool, wheel: int) -> object:
        self.redraw = True
        if wheel:
            self.help_scroll = max(0, min(self.help_max_scroll, self.help_scroll + wheel))
            return _NO_OUTCOME
        if self.help_drag is not None and self.help_track and self.help_thumb:
            origin_y, origin_scroll = self.help_drag
            travel = max(1, self.help_track.height - self.help_thumb.height)
            self.help_scroll = max(0, min(self.help_max_scroll,
                origin_scroll + round((y - origin_y) * self.help_max_scroll / travel)))
            if released:
                self.help_drag = None
            return _NO_OUTCOME
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
        return _NO_OUTCOME

    def _handle_help_key(self, key: Any) -> object:
        """Consume every key while the help overlay is up."""
        if (
            key in (27, "\x1b", "\n", "\r", 10, 13)
            or key == "?"
            or self._key_is("KEY_ENTER", key)
        ):
            self.help_visible = False
            self.help_drag = None
            self.help_scroll = 0
            return _NO_OUTCOME
        page = max(1, self.screen.getmaxyx()[0] - 6)
        if self._key_is("KEY_UP", key):
            self.help_scroll = max(0, self.help_scroll - 1)
        elif self._key_is("KEY_DOWN", key):
            self.help_scroll += 1
        elif self._key_is("KEY_PPAGE", key):
            self.help_scroll = max(0, self.help_scroll - page)
        elif self._key_is("KEY_NPAGE", key):
            self.help_scroll += page
        elif self._key_is("KEY_HOME", key):
            self.help_scroll = 0
        elif self._key_is("KEY_END", key):
            width = min(self.screen.getmaxyx()[1] - 4, 78)
            self.help_scroll = len(self._help_lines(width - 4))
        return _NO_OUTCOME

    def _draw_editor(self) -> None:
        rect = self._overlay_rect(76, EDITOR_HEIGHT)
        x, y, width = rect.x, rect.y, rect.width
        self.editor_rect = rect
        for row in range(rect.height):
            self._add(rect.y + row, rect.x, " " * rect.width, max_width=rect.width)
        pane = self.editor_pane or self.session.focus
        pane_label = {
            "disassembly": "Disassembly",
            "registers": "LR / CR",
            "xmem": "XMEM",
            "pipeline": "Pipeline",
        }[pane]
        action = "Add" if self.editor_mode == "add" else "Edit"
        title = f"{action} {pane_label} tab"
        if self.editor_register is not None:
            name, index = self.editor_register
            title = f"Edit {name.upper()}{index} value"
        if self.editor_mode == "value":
            title = f"Edit {self.editor_value[4]}"
        self._draw_overlay_frame(rect, [title])
        descriptions = {
            "disassembly": (
                "Target: current or PC",
                "Range: 0 through instruction memory size - 1",
            ),
            "registers": (
                "Group: all, lr, or cr",
                "Format is retained independently per tab",
            ),
            "xmem": (
                "Address: row|byte BASE OFFSET COUNT",
                "Formats: hex  int8  u8  cell16  u32  f32",
            ),
            "pipeline": (
                "Register: R0, R1, R_C, R_MASK, R_ACC, POST_AAQ,",
                "          MULT_RES, or MEM_BYPASS",
            ),
        }
        first_description, second_description = descriptions[pane]
        if self.editor_mode == "scalar":
            first_description = "Integer value: decimal, 0x hex, or 0b binary"
            second_description = "Register masking applies; CR0 and CR1 are read-only"
        if self.editor_mode == "value":
            first_description = f"Format: {self.editor_value[3]} / {self.editor_value[2]} bytes"
            second_description = "Enter a number; hex/binary prefixes accepted for integers"
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
        self.editor_field = (
            Rect(y + 4, x + 2 + len(label), 1, available), visible_start
        )
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
        self.overlay_hits.extend([
            (Rect(rect.y + rect.height - 1, hint_x, 1, len("Enter Apply")), "\n"),
            (Rect(rect.y + rect.height - 1, hint_x + hint.index("Esc"), 1, len("Esc Cancel")), "\x1b"),
        ])
        cursor_x = x + 2 + len(label) + self.cursor - visible_start
        try:
            if curses is not None:
                curses.curs_set(1)
            self.screen.move(y + 4, min(x + width - 2, cursor_x))
        except Exception:
            pass

    def handle_key(self, key: Any) -> object:
        # A preceding queued event may have changed layout without a repaint.
        # Cursor arithmetic must use the new pane dimensions immediately.
        self._update_pane_rects(self.layout())
        if not self._key_is("KEY_MOUSE", key):
            self.redraw = True
        if curses is not None and key == getattr(curses, "KEY_RESIZE", -1):
            self._apply_terminal_resize()
            return _NO_OUTCOME
        # A drag whose release never arrives - the pointer left the terminal,
        # say - must not turn every later mouse event into a resize.
        if not self._key_is("KEY_MOUSE", key):
            self.drag = None
            self.scroll_drag = None
            self.scroll_origin = None
            self.last_value_click = None
        if self._key_is("KEY_MOUSE", key):
            return self._handle_mouse()
        # The overlay is modal: it must claim Esc before the editor sees it.
        if self.help_visible:
            return self._handle_help_key(key)
        if key in (27, "\x1b"):
            if self.session.stages_visible and self.session.focus == "disassembly" and not self.editor_mode:
                self.session.stages_visible = False
                return _NO_OUTCOME
            if self.editor_mode:
                pane = self.editor_pane
                self._clear_editor()
                label = "XMEM" if pane == "xmem" else pane
                self.session.set_message(f"{label} tab edit cancelled")
            return _NO_OUTCOME
        if self.editor_mode:
            return self._handle_editor_key(key)
        if self.session.stages_visible and self.session.focus == "disassembly":
            movements = {"KEY_DOWN": 1, "KEY_UP": -1,
                         "KEY_NPAGE": max(1, self.disassembly_rect.height - 4),
                         "KEY_PPAGE": -max(1, self.disassembly_rect.height - 4)}
            for name, delta in movements.items():
                if self._key_is(name, key):
                    self.session.stages_scroll = max(0, self.session.stages_scroll + delta)
                    return _NO_OUTCOME
            if self._key_is("KEY_HOME", key) or self._key_is("KEY_END", key):
                self.session.stages_scroll = 0 if self._key_is("KEY_HOME", key) else 10000
                return _NO_OUTCOME
        if key in ("e", "E"):
            if self.session.focus in ("registers", "pipeline", "xmem"):
                self._begin_value_editor()
            return _NO_OUTCOME
        if self._key_is("KEY_F9", key) or self._key_is("KEY_F10", key):
            if self.session.focus != "disassembly":
                return _NO_OUTCOME
            tab = self.session.selected_disassembly_tab()
            base = self.state.program_counter if tab.target is None else tab.target
            pc = max(0, min(INST_MEM_SIZE - 1, base + tab.cursor_offset))
            if self._key_is("KEY_F9", key):
                enabled = self.control.toggle_breakpoint(pc)
                action = "set" if enabled else "removed"
                self.session.set_message(f"Breakpoint {action} at PC {pc}")
            elif self.control.run_until(pc, self.state.program_counter):
                return self._execution_action(DebugAction.CONTINUE)
            else:
                self.session.set_message(f"Already at PC {pc}")
            return _NO_OUTCOME
        if isinstance(key, str) and key.lower() == "q":
            self.state.program_counter = INST_MEM_SIZE
            return self._execution_action(DebugAction.QUIT)
        if isinstance(key, str) and key.lower() == "f":
            self._cycle_focused_format()
            return _NO_OUTCOME
        if isinstance(key, str) and key in PANE_FOCUS_KEYS:
            self._focus_pane(PANE_FOCUS_KEYS[key])
            return _NO_OUTCOME
        if isinstance(key, str) and key.lower() == "a":
            self._select_tab_in_focused_pane(-1)
            return _NO_OUTCOME
        if isinstance(key, str) and key.lower() == "d":
            self._select_tab_in_focused_pane(1)
            return _NO_OUTCOME
        # Opens only outside an editor, so "?" stays typable in a tab request.
        if key == "?":
            self.help_visible = True
            self.help_scroll = 0
            return _NO_OUTCOME
        if key == "=":
            self._reset_splits()
            return _NO_OUTCOME
        if self._key_is("KEY_F2", key):
            self._begin_tab_editor("add")
            return _NO_OUTCOME
        if self._key_is("KEY_F3", key):
            self._begin_tab_editor("edit")
            return _NO_OUTCOME
        if self._key_is("KEY_F4", key):
            self._close_active_tab()
            return _NO_OUTCOME
        if self._key_is("KEY_F5", key):
            return self._execution_action(DebugAction.CONTINUE)
        if self._key_is("KEY_F8", key):
            return self._execution_action(DebugAction.STEP)
        if key in ("\t", 9):
            self._select_tab_in_focused_pane(1)
            return _NO_OUTCOME
        if self._key_is("KEY_BTAB", key):
            self._select_relative_pane(1)
            return _NO_OUTCOME
        if self._key_is("KEY_PPAGE", key):
            self._move_focused_cursor(dy=-self._focused_page_size())
            return _NO_OUTCOME
        if self._key_is("KEY_NPAGE", key):
            self._move_focused_cursor(dy=self._focused_page_size())
            return _NO_OUTCOME
        if self._key_is("KEY_HOME", key):
            self._jump_focused_cursor(end=False)
            return _NO_OUTCOME
        if self._key_is("KEY_END", key):
            self._jump_focused_cursor(end=True)
            return _NO_OUTCOME
        # Shift-Left moves the focused row's split left, so its right-hand
        # pane grows.
        if self._key_is("KEY_SLEFT", key):
            self._resize_vertical_split(-PANE_RESIZE_STEP)
            return _NO_OUTCOME
        if self._key_is("KEY_SRIGHT", key):
            self._resize_vertical_split(PANE_RESIZE_STEP)
            return _NO_OUTCOME
        if self._key_is("KEY_SR", key):
            self._resize_horizontal_split(-1)
            return _NO_OUTCOME
        if self._key_is("KEY_SF", key):
            self._resize_horizontal_split(1)
            return _NO_OUTCOME
        if self._key_is("KEY_UP", key):
            self._move_focused_cursor(dy=-1)
            return _NO_OUTCOME
        if self._key_is("KEY_DOWN", key):
            self._move_focused_cursor(dy=1)
            return _NO_OUTCOME
        if self._key_is("KEY_LEFT", key):
            self._move_focused_cursor(dx=-1)
            return _NO_OUTCOME
        if self._key_is("KEY_RIGHT", key):
            self._move_focused_cursor(dx=1)
            return _NO_OUTCOME
        return _NO_OUTCOME

    def _handle_editor_key(self, key: Any) -> object:
        if self._key_is("KEY_LEFT", key):
            self.cursor = max(0, self.cursor - 1)
        elif self._key_is("KEY_RIGHT", key):
            self.cursor = min(len(self.command), self.cursor + 1)
        elif self._key_is("KEY_HOME", key):
            self.cursor = 0
        elif self._key_is("KEY_END", key):
            self.cursor = len(self.command)
        elif self._key_is("KEY_DC", key) and self.cursor < len(self.command):
            self.command = (
                self.command[: self.cursor] + self.command[self.cursor + 1 :]
            )
        elif (
            key in ("\b", "\x7f", 8)
            or self._key_is("KEY_BACKSPACE", key)
        ) and self.cursor > 0:
            self.command = (
                self.command[: self.cursor - 1] + self.command[self.cursor :]
            )
            self.cursor -= 1
        elif key in ("\n", "\r", 10, 13) or self._key_is("KEY_ENTER", key):
            if self.editor_mode == "scalar":
                return self._submit_scalar_editor()
            if self.editor_mode == "value":
                return self._submit_value_editor()
            return self._submit_tab_editor(self.command.strip())
        elif isinstance(key, str) and key.isprintable():
            self.command = (
                self.command[: self.cursor] + key + self.command[self.cursor :]
            )
            self.cursor += len(key)
        return _NO_OUTCOME

    def _handle_mouse(self) -> object:
        if curses is None:
            return _NO_OUTCOME
        try:
            _, x, y, _, button_state = curses.getmouse()
        except curses.error:
            return _NO_OUTCOME
        released = bool(button_state & getattr(curses, "BUTTON1_RELEASED", 0))
        pressed = bool(button_state & (
            getattr(curses, "BUTTON1_PRESSED", 0)
            | getattr(curses, "BUTTON1_CLICKED", 0)
            | getattr(curses, "BUTTON1_DOUBLE_CLICKED", 0)
        ))
        wheel = 0
        if button_state & getattr(curses, "BUTTON4_PRESSED", 0):
            wheel = -MOUSE_WHEEL_LINES
        elif button_state & getattr(curses, "BUTTON5_PRESSED", 0):
            wheel = MOUSE_WHEEL_LINES
        if released and self.consume_mouse_release:
            self.consume_mouse_release = False
            return _NO_OUTCOME
        if pressed:
            self.consume_mouse_release = False
        if not (pressed or released or wheel or self.drag or self.scroll_drag or self.help_drag):
            return _NO_OUTCOME
        if self.redraw:
            self.render()
        if self.help_visible:
            return self._handle_help_mouse(y, x, pressed, released, wheel)
        # Modal controls consume every pointer event, including wheel input.
        if self.help_visible or self.editor_mode:
            self.redraw = True
            if wheel and self.help_visible:
                self.help_scroll = max(0, self.help_scroll + wheel)
            if pressed:
                for rect, key in self.overlay_hits:
                    if rect.contains(y, x):
                        return self.handle_key(key)
                if self.editor_mode and self.editor_field is not None:
                    rect, start = self.editor_field
                    if rect.contains(y, x):
                        self.cursor = min(len(self.command), start + x - rect.x)
            return _NO_OUTCOME
        if self.scroll_drag is not None:
            pane = self.scroll_drag
            track = self.scrollbar_hits.get(pane)
            if track is not None:
                origin_y, fraction = self.scroll_origin or (track.y, 0.0)
                delta = y - origin_y
                if delta or self.scroll_drag_moved:
                    self.scroll_drag_moved = True
                    self._scroll_to_fraction(
                        pane, fraction + delta / max(1, track.height - 1)
                    )
                    self.redraw = True
            if released:
                self.scroll_drag = None
                self.scroll_origin = None
            return _NO_OUTCOME
        if self.drag is not None:
            return self._continue_drag(y, x, released)
        if wheel:
            self.last_value_click = None
            self.redraw = True
            self._wheel(
                y, x, wheel,
                horizontal=bool(button_state & getattr(curses, "BUTTON_SHIFT", 0)),
            )
            return _NO_OUTCOME
        if pressed:
            self.redraw = True
            for rect, key in self.action_hits:
                if rect.contains(y, x):
                    return self.handle_key(key)
            if button_state & getattr(curses, "BUTTON1_PRESSED", 0):
                for pane, track in self.scrollbar_hits.items():
                    if track.contains(y, x):
                        self._focus_pane(pane)
                        self.scroll_drag = pane
                        self.scroll_drag_moved = False
                        self.last_value_click = None
                        thumb = self.scroll_thumb_hits.get(pane)
                        if thumb is None or not thumb.contains(y, x):
                            self._scroll_to_fraction(
                                pane, (y - track.y) / max(1, track.height - 1)
                            )
                        self.scroll_origin = (y, self._scroll_cursor_fraction(pane))
                        return _NO_OUTCOME
                split = self._split_at(y, x)
                if split is not None:
                    self.drag = (split, y, x, False)
                    return _NO_OUTCOME
            self._handle_click(
                y, x, double=bool(button_state & getattr(curses, "BUTTON1_DOUBLE_CLICKED", 0))
            )
        return _NO_OUTCOME

    def _wheel(self, y: int, x: int, delta: int, *, horizontal: bool = False) -> None:
        if self._focus_pane_at(y, x):
            rect = getattr(self, f"{self.session.focus}_rect")
            if y == rect.y + 1:
                self._select_tab_in_focused_pane(1 if delta > 0 else -1)
            elif self.session.focus == "disassembly" and self.session.stages_visible and not horizontal:
                self.session.stages_scroll = max(0, self.session.stages_scroll + delta)
            elif horizontal:
                self._move_focused_cursor(dx=delta)
            else:
                self._move_focused_cursor(dy=delta)

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

    def _continue_drag(self, y: int, x: int, released: bool) -> object:
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
        return _NO_OUTCOME

    def _drag_split_to(self, split: str, y: int, x: int) -> None:
        rows, columns = self.screen.getmaxyx()
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
                self._set_active_pane_tab_index(pane, index)
                self._ensure_active_tab_visible(pane)
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
                    self._set_active_pane_tab_index(pane, index)
                    self._ensure_active_tab_visible(pane)
                    self.session.set_message(
                        f"Selected {self._pane_tabs(pane)[index].title}"
                    )
                    return
        for pane, rect in self.add_hits.items():
            if rect.contains(y, x):
                self.session.focus = pane
                self._begin_tab_editor("add")
                return
        for pane, rect in self.scrollbar_hits.items():
            if rect.contains(y, x):
                self.session.focus = pane
                thumb = self.scroll_thumb_hits.get(pane)
                if thumb is None or not thumb.contains(y, x):
                    self._scroll_to_fraction(
                        pane, (y - rect.y) / max(1, rect.height - 1)
                    )
                return
        for rect, pc in self.value_hits.get("disassembly", []):
            if rect.contains(y, x) and x < rect.x + 2:
                self._focus_pane("disassembly")
                self._place_cursor("disassembly", pc)
                enabled = self.control.toggle_breakpoint(pc)
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
                        identity = id(self._pane_tabs(pane)[self._active_pane_tab_index(pane)])
                        if double or (previous_click is not None
                                      and previous_click[:2] == (identity, payload)
                                      and now - previous_click[2] <= DOUBLE_CLICK_SECONDS):
                            self._begin_value_editor()
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
        """Put the focused pane's cursor on the value that was clicked."""
        if pane == "disassembly":
            tab = self.session.selected_disassembly_tab()
            base = (
                self.state.program_counter if tab.target is None else tab.target
            )
            tab.cursor_offset = payload - base
            return
        if pane == "registers":
            self.session.selected_register_tab().cursor = payload
            return
        if pane == "pipeline":
            self.session.selected_pipeline_tab().cursor = payload
            return
        tab = self.session.selected_tab()
        if tab is not None:
            tab.cursor_line, tab.cursor_column = payload

    def _scroll_cursor_fraction(self, pane: str) -> float:
        """Remember the selection at grab time, so a thumb press never jumps."""
        total = self.scroll_state[pane][2]
        if pane == "disassembly":
            tab = self.session.selected_disassembly_tab()
            base = self.state.program_counter if tab.target is None else tab.target
            target = base + tab.cursor_offset
        elif pane == "pipeline":
            tab = self.session.selected_pipeline_tab()
            spec = _pipeline_spec(tab.register_key)
            fmt = tab.fmt or _pipeline_format(self.state, spec)
            size, _ = self._pipeline_value_geometry(fmt)
            items = len(_read_pipeline_register(self.state, spec)) // size
            rows = self._pipeline_display_rows(fmt, self.pipeline_rect, items)
            per_line = self._pipeline_items_per_line(fmt, self.pipeline_rect)
            target = next(index for index, first in enumerate(rows)
                          if first is not None and first <= tab.cursor < first + per_line)
        else:
            tab = self.session.selected_tab()
            target = tab.cursor_line if tab is not None else 0
        return (target + 0.5) / max(1, total)

    def _scroll_to_fraction(self, pane: str, fraction: float) -> None:
        """Jump the pane's cursor to a position along its scrollbar."""
        _, _, total = self.scroll_state.get(pane, (0, 0, 0))
        if total <= 0:
            return
        target = max(0, min(total - 1, int(fraction * total)))
        if pane == "disassembly":
            tab = self.session.selected_disassembly_tab()
            base = (
                self.state.program_counter if tab.target is None else tab.target
            )
            tab.cursor_offset = target - base
            return
        if pane == "pipeline":
            tab = self.session.selected_pipeline_tab()
            spec = _pipeline_spec(tab.register_key)
            fmt = tab.fmt or _pipeline_format(self.state, spec)
            items_per_line = self._pipeline_items_per_line(
                fmt,
                self.pipeline_rect,
            )
            item_size, _ = self._pipeline_value_geometry(fmt)
            items = len(_read_pipeline_register(self.state, spec)) // item_size
            rows = self._pipeline_display_rows(fmt, self.pipeline_rect, items)
            target = min(target, len(rows) - 1)
            if rows[target] is None:
                target = min(target + 1, len(rows) - 1)
            tab.cursor = max(0, min(items - 1, rows[target]))
            return
        xmem_tab = self.session.selected_tab()
        if xmem_tab is not None:
            xmem_tab.cursor_line = target

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

    def _submit_tab_editor(self, command: str) -> object:
        pane = self.editor_pane or self.session.focus
        mode = self.editor_mode
        tabs = self._pane_tabs(pane)
        active = self._active_pane_tab_index(pane)
        existing = tabs[active] if mode == "edit" and tabs else None
        tab, error = self._parse_editor_tab(pane, command, existing)
        if error is not None or tab is None:
            self.session.set_message(f"{pane} tab error: {error}", error=True)
            return _NO_OUTCOME
        if mode == "add" or not tabs:
            tabs.append(tab)
            self._set_active_pane_tab_index(pane, len(tabs) - 1)
        else:
            tabs[active] = tab
        self._clear_editor()
        self._ensure_active_tab_visible(pane)
        action = "added" if mode == "add" else "updated"
        self.session.set_message(f"{pane} tab {action}: {tab.title}")
        return _NO_OUTCOME

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
        if pane == "pipeline":
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

        from ipu_emu.debug_cli import _parse_xmem_request

        if command.lower().startswith("xmem "):
            command = command[5:]
        request, error = _parse_xmem_request(command)
        if error is None and request is not None:
            _, error = self._resolve_xmem(request)
        if error is not None or request is None:
            return None, error
        return XmemTab(request), None

    def _begin_value_editor(self) -> None:
        if self.session.focus == "registers":
            self._begin_scalar_editor()
            return
        pane = self.session.focus
        if pane == "pipeline":
            tab = self.session.selected_pipeline_tab()
            spec = _pipeline_spec(tab.register_key)
            fmt = tab.fmt or _pipeline_format(self.state, spec)
            size, _ = self._pipeline_value_geometry(fmt)
            backing = spec.wide_backing if self.state.wide_vector_debug and spec.wide_backing else spec.backing
            register_size = self.state.regfile._desc(backing).size_bytes
            tab.cursor = max(0, min(tab.cursor, register_size // size - 1))
            offset = spec.index * register_size + tab.cursor * size
            raw = bytes(self.state.regfile.raw(backing)[offset:offset + size])
            label = f"{spec.label}[{tab.cursor}]"
        else:
            tab = self.session.selected_tab()
            if tab is None:
                return
            resolved, error = self._resolve_xmem(tab.request)
            if error or resolved is None:
                self.session.set_message(str(error or "No memory value selected"), error=True)
                return
            lines = self._xmem_lines(resolved)
            self._normalize_xmem_cursor(tab, lines)
            tokens = lines[tab.cursor_line].tokens
            if not tokens:
                return
            token = tokens[tab.cursor_column]
            offset, size = resolved.byte_address + token.raw_start, token.raw_size
            fmt, backing = tab.request.fmt, None
            raw = self.state.xmem.read_address(offset, size)
            label = f"XMEM 0x{offset:08x}"
        if len(raw) != size:
            self.session.set_message("Selected value is out of range", error=True)
            return
        self.editor_value = (backing, offset, size, fmt, label)
        self.editor_register = None
        self.editor_mode, self.editor_pane = "value", pane
        if fmt == "f32":
            self.command = repr(struct.unpack("<f", raw)[0])
        else:
            value = int.from_bytes(raw, "little", signed=fmt in ("int8", "int32"))
            self.command = hex(value) if fmt in ("hex", "cell16") else bin(value) if fmt == "bits" else str(value)
        self.cursor = len(self.command)
        self.session.set_message(f"Editing {label}")

    def _submit_value_editor(self) -> object:
        from ipu_emu.debug_cli import _parse_int

        backing, offset, size, fmt, label = self.editor_value
        try:
            if fmt == "f32":
                raw = struct.pack("<f", float(self.command.strip()))
            else:
                value = _parse_int(self.command.strip())
                if value is None:
                    raise ValueError("Enter a valid integer")
                raw = value.to_bytes(size, "little", signed=fmt in ("int8", "int32"))
            if backing is None:
                self.state.xmem.write_address(offset, raw)
            else:
                buffer = self.state.regfile.raw(backing)
                if offset < 0 or offset + size > len(buffer):
                    raise ValueError("Selected value is out of range")
                buffer[offset:offset + size] = raw
        except (ValueError, OverflowError, struct.error, EmulatorError) as error:
            self.session.set_message(f"Invalid value: {error}", error=True)
            return _NO_OUTCOME
        self._xmem_requests.clear()
        self._xmem_cache.clear()
        self._clear_editor()
        self.editor_value = None
        self.session.set_message(f"Updated {label}")
        return _NO_OUTCOME

    def _begin_scalar_editor(self) -> None:
        tab = self.session.selected_register_tab()
        groups = self._register_groups(tab)
        position = max(0, min(tab.cursor, len(groups) * _REGISTERS_PER_GROUP - 1))
        name = groups[position // _REGISTERS_PER_GROUP]
        index = position % _REGISTERS_PER_GROUP
        self.editor_register = (name, index)
        self.editor_mode = "scalar"
        self.editor_pane = "registers"
        self.command = hex(self.state.regfile.get_scalar(name, index))
        self.cursor = len(self.command)
        self.session.set_message(f"Enter an integer for {name.upper()}{index}")

    def _submit_scalar_editor(self) -> object:
        from ipu_emu.debug_cli import _parse_int

        assert self.editor_register is not None
        name, index = self.editor_register
        value = _parse_int(self.command.strip())
        if value is None:
            self.session.set_message("Invalid integer value", error=True)
            return _NO_OUTCOME
        try:
            self.state.regfile.set_scalar(name, index, value)
        except EmulatorError as error:
            self.session.set_message(str(error), error=True)
            return _NO_OUTCOME
        self._xmem_requests.clear()
        self._xmem_cache.clear()
        self._clear_editor()
        actual = self.state.regfile.get_scalar(name, index)
        self.session.set_message(f"Set {name.upper()}{index} = {actual:#x}")
        return _NO_OUTCOME

    def _begin_tab_editor(self, mode: str) -> None:
        pane = self.session.focus
        tabs = self._pane_tabs(pane)
        active = self._active_pane_tab_index(pane)
        tab = tabs[active] if tabs else None
        if mode == "edit" and not tabs:
            mode = "add"
        if mode == "edit":
            if pane == "disassembly":
                command = "current" if tab.target is None else str(tab.target)
            elif pane == "registers":
                command = tab.group
            elif pane == "pipeline":
                command = _pipeline_spec(tab.register_key).label
            else:
                command = tab.request.command()[len("xmem ") :]
        else:
            command = {
                "disassembly": "current",
                "registers": "all",
                "pipeline": "R0",
                "xmem": f"row 0 0 {xmem_row_size_bytes(self.state)} hex",
            }[pane]
        self.editor_mode = mode
        self.editor_pane = pane
        self.command = command
        self.cursor = len(command)
        self.session.set_message(
            f"Enter a {pane} tab target; press Enter to apply or Esc to cancel"
        )

    def _clear_editor(self) -> None:
        self.command = ""
        self.cursor = 0
        self.editor_mode = None
        self.editor_pane = None
        self.editor_register = None
        self.editor_value = None
        try:
            if curses is not None:
                curses.curs_set(0)
        except Exception:
            pass

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
        tabs = self._pane_tabs(pane)
        if not tabs:
            self.session.set_message(f"{pane} has no tabs")
            return
        active = (self._active_pane_tab_index(pane) + delta) % len(tabs)
        self._set_active_pane_tab_index(pane, active)
        self._ensure_active_tab_visible(pane)
        self.session.set_message(f"Selected {tabs[active].title}")

    def _cycle_focused_format(self, delta: int = 1) -> None:
        pane = self.session.focus
        if pane == "disassembly":
            tab = self.session.selected_disassembly_tab()
            tab.fmt = self._next_format(tab.fmt, DISASSEMBLY_FORMATS, delta)
            self.session.set_message(f"Disassembly format: {tab.fmt}")
            return
        if pane == "registers":
            tab = self.session.selected_register_tab()
            tab.fmt = self._next_format(
                tab.fmt,
                SCALAR_REGISTER_FORMATS,
                delta,
            )
            tab.column_scroll = 0
            self.session.set_message(f"LR / CR format: {tab.fmt}")
            return
        if pane == "pipeline":
            tab = self.session.selected_pipeline_tab()
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
            return
        self._cycle_xmem_format(delta)

    def _cycle_xmem_format(self, delta: int) -> None:
        from ipu_emu.debug_cli import (
            XmemRequest,
            _XMEM_FORMAT_ITEM_SIZE_BYTES,
            _XMEM_FORMATS,
        )

        tab = self.session.selected_tab()
        if tab is None:
            self.session.set_message("XMEM has no tabs")
            return
        resolved, error = self._resolve_xmem(tab.request)
        if error is not None or resolved is None:
            self.session.set_message(f"XMEM format error: {error}", error=True)
            return
        current_index = _XMEM_FORMATS.index(tab.request.fmt)
        for distance in range(1, len(_XMEM_FORMATS) + 1):
            candidate = _XMEM_FORMATS[
                (current_index + distance * delta) % len(_XMEM_FORMATS)
            ]
            item_size = _XMEM_FORMAT_ITEM_SIZE_BYTES[candidate]
            if (
                resolved.byte_address % item_size == 0
                and resolved.byte_count % item_size == 0
            ):
                tab.request = XmemRequest(
                    tab.request.mode,
                    tab.request.base_token,
                    tab.request.offset_token,
                    resolved.byte_count // item_size,
                    candidate,
                )
                tab.scroll = 0
                tab.cursor_line = 0
                tab.cursor_column = 0
                tab.horizontal_scroll = 0
                tab.baseline_key = None
                tab.baseline_data = None
                self.session.set_message(f"XMEM format: {candidate}")
                return

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

    def _ensure_active_tab_visible(self, pane: str) -> None:
        active = self._active_pane_tab_index(pane)
        offset_attribute = self._pane_offset_attribute(pane)
        offset = getattr(self.session, offset_attribute)
        if active < offset:
            offset = active
        elif (
            active > offset + VISIBLE_TAB_LOOKBEHIND
        ):
            offset = max(0, active - VISIBLE_TAB_LOOKBEHIND)
        setattr(self.session, offset_attribute, offset)

    def _close_active_tab(self) -> None:
        pane = self.session.focus
        tabs = self._pane_tabs(pane)
        if tabs:
            self._close_tab(pane, self._active_pane_tab_index(pane))

    def _close_tab(self, pane: str, index: int) -> None:
        tabs = self._pane_tabs(pane)
        if not 0 <= index < len(tabs):
            return
        del tabs[index]
        if not tabs and pane != "xmem":
            defaults = {
                "disassembly": DisassemblyTab(),
                "registers": RegisterTab(),
                "pipeline": PipelineTab(),
            }
            tabs.append(defaults[pane])
        active = min(index, max(0, len(tabs) - 1))
        self._set_active_pane_tab_index(pane, active)
        offset_attribute = self._pane_offset_attribute(pane)
        setattr(
            self.session,
            offset_attribute,
            min(getattr(self.session, offset_attribute), max(0, len(tabs) - 1)),
        )
        self.session.set_message(f"Closed {pane} tab")

    def _move_focused_cursor(self, *, dx: int = 0, dy: int = 0) -> None:
        if self.session.focus == "disassembly":
            tab = self.session.selected_disassembly_tab()
            base = self.state.program_counter if tab.target is None else tab.target
            minimum = -base
            maximum = INST_MEM_SIZE - 1 - base
            tab.cursor_offset = max(
                minimum,
                min(maximum, tab.cursor_offset + dy),
            )
            return
        if self.session.focus == "registers":
            tab = self.session.selected_register_tab()
            grid_rows, column_count = self._register_grid(tab)
            row = min(grid_rows - 1, tab.cursor % grid_rows)
            column = tab.cursor // grid_rows
            row = max(0, min(grid_rows - 1, row + dy))
            column = max(0, min(column_count - 1, column + dx))
            tab.cursor = column * grid_rows + row
            return
        if self.session.focus == "pipeline":
            self._move_pipeline_cursor(dx, dy)
            return
        if self.session.focus == "xmem":
            self._move_xmem_cursor(dx, dy)

    def _jump_focused_cursor(self, *, end: bool) -> None:
        """Move the focused pane's cursor to its first or last value."""
        if self.session.focus == "disassembly":
            tab = self.session.selected_disassembly_tab()
            base = self.state.program_counter if tab.target is None else tab.target
            tab.cursor_offset = (INST_MEM_SIZE - 1 - base) if end else -base
            return
        if self.session.focus == "registers":
            tab = self.session.selected_register_tab()
            groups = self._register_groups(tab)
            tab.cursor = (
                len(groups) * _REGISTERS_PER_GROUP - 1 if end else 0
            )
            return
        if self.session.focus == "pipeline":
            tab = self.session.selected_pipeline_tab()
            spec = _pipeline_spec(tab.register_key)
            fmt = tab.fmt or _pipeline_format(self.state, spec)
            item_size, _ = self._pipeline_value_geometry(fmt)
            item_count = len(_read_pipeline_register(self.state, spec)) // item_size
            tab.cursor = max(0, item_count - 1) if end else 0
            return
        self._jump_xmem_cursor(end=end)

    def _jump_xmem_cursor(self, *, end: bool) -> None:
        tab = self.session.selected_tab()
        if tab is None:
            return
        resolved, error = self._resolve_xmem(tab.request)
        if error is not None or resolved is None:
            return
        lines = self._xmem_lines(resolved)
        self._normalize_xmem_cursor(tab, lines)
        data_lines = [index for index, line in enumerate(lines) if line.tokens]
        if not data_lines:
            return
        tab.cursor_line = data_lines[-1 if end else 0]
        tokens = lines[tab.cursor_line].tokens
        tab.cursor_column = len(tokens) - 1 if end else 0

    def _move_xmem_cursor(self, dx: int, dy: int) -> None:
        tab = self.session.selected_tab()
        if tab is None:
            return
        resolved, error = self._resolve_xmem(tab.request)
        if error is not None or resolved is None:
            return
        lines = self._xmem_lines(resolved)
        self._normalize_xmem_cursor(tab, lines)
        data_lines = [index for index, line in enumerate(lines) if line.tokens]
        if not data_lines:
            return
        line_position = data_lines.index(tab.cursor_line)
        if dy:
            line_position = max(
                0,
                min(len(data_lines) - 1, line_position + dy),
            )
            tab.cursor_line = data_lines[line_position]
            tab.cursor_column = min(
                tab.cursor_column,
                len(lines[tab.cursor_line].tokens) - 1,
            )
        while dx < 0:
            if tab.cursor_column > 0:
                tab.cursor_column -= 1
            elif line_position > 0:
                line_position -= 1
                tab.cursor_line = data_lines[line_position]
                tab.cursor_column = len(lines[tab.cursor_line].tokens) - 1
            dx += 1
        while dx > 0:
            tokens = lines[tab.cursor_line].tokens
            if tab.cursor_column + 1 < len(tokens):
                tab.cursor_column += 1
            elif line_position + 1 < len(data_lines):
                line_position += 1
                tab.cursor_line = data_lines[line_position]
                tab.cursor_column = 0
            dx -= 1

    def _move_pipeline_cursor(self, dx: int, dy: int) -> None:
        tab = self.session.selected_pipeline_tab()
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

    def _key_is(self, name: str, key: Any) -> bool:
        return curses is not None and key == getattr(curses, name, object())

    def _box(self, rect: Rect, *, focused: bool = False) -> None:
        """Record one pane frame; ``_flush_borders`` resolves the junctions."""
        if rect.height < 2 or rect.width < 2:
            return
        priority = 1 if focused else 0
        attr = self.colors.focus if focused else self.colors.border
        right = rect.x + rect.width - 1
        bottom = rect.y + rect.height - 1
        for x in range(rect.x, right + 1):
            mask = (BORDER_LEFT if x > rect.x else 0) | (
                BORDER_RIGHT if x < right else 0
            )
            for y in (rect.y, bottom):
                self._mark_border(y, x, mask, attr, priority)
        for y in range(rect.y, bottom + 1):
            mask = (BORDER_UP if y > rect.y else 0) | (
                BORDER_DOWN if y < bottom else 0
            )
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
        separator = f" {self.glyphs.separator} "
        text = segments[0] if segments else ""
        for segment in segments[1:]:
            candidate = f"{text}{separator}{segment}"
            if len(candidate) > width:
                break
            text = candidate
        if len(text) > width:
            text = text[: max(0, width - len(self.glyphs.ellipsis))]
            text += self.glyphs.ellipsis
        return text

    def _overlay_rect(self, preferred_width: int, preferred_height: int) -> Rect:
        """Centre a modal overlay between the header and the footer."""
        rows, columns = self.screen.getmaxyx()
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

    def _draw_pane_title(
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
        title = self._fit_segments(segments, width)
        if not title:
            return
        attr = self.colors.title if focused else self.colors.plain
        self._add(rect.y, rect.x + 2, f" {title} ", attr, width + 2)

    def _draw_top_pane_header(
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
        if readout and self._fit_segments(segments, shared) == (
            self._fit_segments(segments, width)
        ):
            if self._draw_position(rect, readout, top=True):
                reserved = len(readout) + TITLE_READOUT_GAP
        self._draw_pane_title(
            rect,
            segments,
            reserved=reserved,
            focused=focused,
        )

    def _add(
        self,
        y: int,
        x: int,
        text: str,
        attr: int = 0,
        max_width: int | None = None,
    ) -> None:
        rows, columns = self.screen.getmaxyx()
        if y < 0 or y >= rows or x < 0 or x >= columns or not text:
            return
        width = columns - x if max_width is None else min(max_width, columns - x)
        if width <= 0:
            return
        try:
            self.screen.addnstr(y, x, text, width, attr)
        except Exception:
            pass


_NO_OUTCOME = object()
# ``curses`` is optional, so the guarded excepts need a tuple that is valid
# even when the import failed.
_CURSES_ERRORS: tuple[type[BaseException], ...] = (
    (curses.error,) if curses is not None else ()
)


def run_debug_tui(state: IpuState, cycle: int) -> DebugAction:
    """Enter curses and return an execution action; interrupts cancel."""
    if curses is None:
        raise RuntimeError("curses is not available on this platform")
    session = get_debug_view_session(state)
    session.active = True
    result = DebugAction.QUIT

    def _run(screen: Any) -> None:
        nonlocal result
        result = CursesDebugView(screen, state, session, cycle).run()

    try:
        curses.wrapper(_run)
    except KeyboardInterrupt:
        session.active = False
        return DebugAction.QUIT
    except Exception as error:
        session.active = False
        raise RuntimeError(f"curses TUI failed: {error}") from error
    return result
