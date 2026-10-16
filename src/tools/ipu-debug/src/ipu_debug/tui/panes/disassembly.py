"""The disassembly pane: instruction memory around the cursor, or the stage view.

A tab follows the current PC or a fixed one and shows ``compact`` (NOP slots
left out), ``full`` (every slot) or ``stages`` (the commands each physical
pipeline stage holds at the paused cycle, from ``ipu_debug.pipeline``). Each
operation gets its own row, so parallel slots stack under their instruction.
The pane reads instruction memory and the PC from the session, and changes
breakpoints, the run target and the PC only through ``DebugSession``.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any

from ipu_debug.pipeline import SLOT_LABELS, display_operation, pipeline_occupancy, stage_operations
from ipu_debug.session import DebugSession
from ipu_debug.tui.draw import SCROLLBAR_COLUMNS, Canvas, Theme
from ipu_debug.tui.editor import TabTarget, scalar_editor, tab_editor
from ipu_debug.tui.keymap import Command, PaneId
from ipu_debug.tui.pane import PaneDrawing, Rect, Reply
from ipu_debug.tui.screen import overlay_style
from ipu_debug.tui.tabs import TabSet
from ipu_emu.emulator import DebugAction
from ipu_emu.ipu_state import INST_MEM_SIZE

# The pane's first row is a border, its second the tab bar, its third the
# column header and its last a border.
DISASSEMBLY_NON_CONTENT_ROWS = 4
DISASSEMBLY_FORMATS = ("compact", "full", "stages")
TITLE = "2 Instructions"
EMPTY_DISASSEMBLY_OPERATIONS = ("<nop>", "<NOP>", "NOP")
# Parallel slots are stacked under their instruction and marked the way VLIW
# assembly listings mark them, rather than run out across the pane.
DISASSEMBLY_PARALLEL_MARKER = "||"
DISASSEMBLY_PREFIX_WIDTH = len("> 0000  ")
# From this row width on, mnemonics share a column and a slot column shows
# which slot each operation is in.
SLOT_COLUMN_MIN_ROW_WIDTH = 60
SLOT_COLUMN_WIDTH = 12
# The stage view labels each stage in a column this wide.
STAGE_LABEL_WIDTH = 10
# Far past any stage list: End scrolls the stage view to its last line.
STAGES_END = 10000
PC_HINT = "Range: 0 through instruction memory size - 1"
EDITOR_HINTS = ("Target: current or PC", PC_HINT)


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
class DisassemblyTabs(TabSet[DisassemblyTab]):
    """The pane's tabs, and how far the stage view (shared by every tab) scrolled."""

    stages_scroll: int = 0


def default_tabs(fresh: bool) -> list[DisassemblyTab]:
    """One tab per format on the first stop; afterwards just ``compact``."""
    return [DisassemblyTab(fmt=fmt) for fmt in (DISASSEMBLY_FORMATS if fresh else ("compact",))]


def base_pc(session: DebugSession, tab: DisassemblyTab) -> int:
    """The PC *tab*'s cursor offset counts from: its target, or the current PC."""
    return session.state.program_counter if tab.target is None else tab.target


def cursor_pc(session: DebugSession, tab: DisassemblyTab) -> int:
    """The instruction under *tab*'s cursor, inside instruction memory."""
    return max(0, min(INST_MEM_SIZE - 1, base_pc(session, tab) + tab.cursor_offset))


def all_operations(disassembly: str) -> list[str]:
    """Every slot's operation of one ``PC <n>: op;\\nop;;`` disassembly."""
    lines = [line.strip().rstrip(";") for line in disassembly.splitlines()]
    if not lines:
        return ["<empty>"]
    _, separator, first_operation = lines[0].partition(":")
    operations = [first_operation.strip() if separator else lines[0]]
    operations.extend(lines[1:])
    return [operation for operation in operations if operation] or ["<empty>"]


def active_operations(disassembly: str) -> list[str]:
    """The operations of one disassembly that are not NOPs (``<NOP>`` if none)."""
    active = [
        operation
        for operation in all_operations(disassembly)
        if operation and operation.upper().split(" ", 1)[0] != "NOP"
    ]
    return active or ["<NOP>"]


@dataclass(frozen=True)
class _Listing:
    """What every row of one render shares."""

    canvas: Canvas
    session: DebugSession
    tab: DisassemblyTab
    centered_pc: int
    row_width: int
    mnemonic_width: int
    cursor: int


class DisassemblyPane:
    """The instructions pane: renders ``DisassemblyTab`` listings and the stage view."""

    id: PaneId = "disassembly"

    def __init__(self, theme: Theme, cycle: int = 0) -> None:
        self.theme = theme
        # The stage view's header names the cycle the debugger stopped at.
        self.cycle = cycle
        # Decoding an instruction is the single most expensive thing a render
        # does, and instruction memory cannot change while the debugger is
        # stopped, so each PC is decoded at most once per pane.
        self._texts: dict[int, str] = {}

    # -- reads ----------------------------------------------------------------

    def disassembly(self, session: DebugSession, pc: int) -> str:
        """Instruction *pc* as ``PC <pc>: op;`` lines, one per slot, ending ``;;``."""
        text = self._texts.get(pc)
        if text is None:
            instruction = session.state.inst_mem[pc] if 0 <= pc < INST_MEM_SIZE else None
            if instruction is None:
                line = session.disassemble(pc, 1)
                text = f"PC {pc}: {line[0].text}" if line else "PC out of bounds"
            else:
                operations = (operation for _, operation in stage_operations(instruction))
                text = f"PC {pc}: " + ";\n".join(operations) + ";;"
            self._texts[pc] = text
        return text

    def operations(self, session: DebugSession, tab: DisassemblyTab, pc: int) -> list[str]:
        """The rows instruction *pc* takes in *tab*'s format."""
        disassembly = self.disassembly(session, pc)
        if tab.fmt == "full":
            return all_operations(disassembly)
        return active_operations(disassembly)

    def rows(
        self, session: DebugSession, tab: DisassemblyTab, centered_pc: int, visible: int
    ) -> tuple[list[tuple[int, int, str]], int, int]:
        """One row per operation, so no instruction spans the pane sideways.

        Returns the rows as ``(pc, slot, operation)`` along with the first and
        last program counter they cover.
        """
        # Walk back from the cursor's instruction while its slots still leave
        # the cursor on screen, so a wide instruction cannot push it off.
        first = centered_pc
        above = 0
        while first > 0 and above + len(self.operations(session, tab, first - 1)) <= visible // 2:
            first -= 1
            above += len(self.operations(session, tab, first))
        rows: list[tuple[int, int, str]] = []
        last = first
        for pc in range(first, INST_MEM_SIZE):
            if len(rows) >= visible:
                break
            last = pc
            for slot, operation in enumerate(self.operations(session, tab, pc)):
                if len(rows) >= visible:
                    break
                rows.append((pc, slot, operation))
        return rows, first, last

    # -- render ---------------------------------------------------------------

    def render(
        self,
        session: DebugSession,
        view: DisassemblyTabs,
        rect: Rect,
        *,
        focused: bool = False,
    ) -> PaneDrawing:
        tab = view.current()
        assert tab is not None  # Closing the last tab reopens a default one.
        canvas = Canvas(self.theme, rect)
        drawing = PaneDrawing(canvas.lines)
        if tab.fmt == "stages":
            self._draw_stages(canvas, session, view, rect, focused=focused)
            return drawing
        colors = self.theme.colors
        # One content column stays free for the scroll thumb.
        row_width = max(1, rect.width - 2 - SCROLLBAR_COLUMNS)
        canvas.add(rect.y + 2, rect.x + 1, f"{'PC':>6}  OPERATIONS", colors.border, row_width)
        if row_width >= SLOT_COLUMN_MIN_ROW_WIDTH:
            canvas.add(rect.y + 2, rect.x + row_width - 10, "      SLOT", colors.border, 10)
        visible = max(1, rect.height - DISASSEMBLY_NON_CONTENT_ROWS)
        centered_pc = cursor_pc(session, tab)
        rows, first, last = self.rows(session, tab, centered_pc, visible)
        # Small panes need room for operands more than vertical alignment.
        mnemonic_width = 0
        if row_width >= SLOT_COLUMN_MIN_ROW_WIDTH:
            mnemonic_width = max((len(op.split(" ", 1)[0]) for _, _, op in rows), default=0)
        listing = _Listing(
            canvas, session, tab, centered_pc, row_width, mnemonic_width, colors.cursor_for(focused)
        )
        slot_labels: dict[int, list[str]] = {}
        for row, (pc, slot, operation) in enumerate(rows):
            y = rect.y + 3 + row
            drawing.hits.append((Rect(y, rect.x + 1, 1, row_width), pc))
            self._draw_row(listing, y, rect.x + 1, (pc, slot, operation), slot_labels)
        drawing.scroll = (first, last - first + 1, INST_MEM_SIZE)
        readout = f" {first:04d}-{last:04d}/{INST_MEM_SIZE} "
        canvas.top_pane_header(rect, [TITLE], readout, focused=focused)
        return drawing

    def _draw_row(
        self,
        listing: _Listing,
        y: int,
        x: int,
        row: tuple[int, int, str],
        slot_labels: dict[int, list[str]],
    ) -> None:
        """Draw one operation row: its prefix, its operation and its slot name."""
        canvas, colors, glyphs = listing.canvas, self.theme.colors, self.theme.glyphs
        pc, slot, operation = row
        is_current = pc == listing.session.state.program_counter
        is_empty = operation in EMPTY_DISASSEMBLY_OPERATIONS
        # The executing instruction is always marked; the reverse-video bar
        # belongs to the cursor, and covers every slot of that instruction.
        # Unwritten instruction slots recede so a program stands out.
        attr = colors.current if is_current else 0
        if is_empty and not is_current:
            attr = colors.border
        if pc == listing.centered_pc:
            attr = overlay_style(attr, listing.cursor)
        limit = x + listing.row_width
        self._draw_prefix(listing, y, x, (pc, slot), attr)
        x += DISASSEMBLY_PREFIX_WIDTH
        if listing.row_width >= SLOT_COLUMN_MIN_ROW_WIDTH:
            limit -= SLOT_COLUMN_WIDTH
        mnemonic, _, operands = operation.partition(" ")
        text = f"{mnemonic.ljust(listing.mnemonic_width)} {operands}" if operands else mnemonic
        available = max(0, limit - x)
        if len(text) > available and available >= len(glyphs.ellipsis):
            text = text[: available - len(glyphs.ellipsis)] + glyphs.ellipsis
        canvas.add(y, x, text.ljust(available), attr, available)
        if not is_empty and pc != listing.centered_pc:
            canvas.add(y, x, mnemonic, colors.key, min(available, len(mnemonic)))
        if listing.row_width >= SLOT_COLUMN_MIN_ROW_WIDTH:
            if pc not in slot_labels:
                slot_labels[pc] = self._slot_labels(listing, pc)
            names = slot_labels[pc]
            label = names[slot] if slot < len(names) else ""
            style = overlay_style(colors.key, attr)
            canvas.add(y, limit, label.rjust(SLOT_COLUMN_WIDTH - 1) + " ", style, SLOT_COLUMN_WIDTH)

    def _draw_prefix(
        self, listing: _Listing, y: int, x: int, position: tuple[int, int], attr: int
    ) -> None:
        """The row's PC, current-PC marker and breakpoint, or a parallel-slot marker."""
        canvas, colors, glyphs = listing.canvas, self.theme.colors, self.theme.glyphs
        pc, slot = position
        limit = x + listing.row_width
        if slot:
            # The parallel-slot marker recedes without breaking the bar the
            # cursor draws behind it.
            prefix = f"{'':{DISASSEMBLY_PREFIX_WIDTH - 3}}"
            canvas.add(y, x, prefix, attr, limit - x)
            marker_style = overlay_style(colors.border, attr)
            marker_width = limit - x - len(prefix)
            canvas.add(y, x + len(prefix), DISASSEMBLY_PARALLEL_MARKER, marker_style, marker_width)
            gap = max(0, limit - x - DISASSEMBLY_PREFIX_WIDTH + 1)
            canvas.add(y, x + DISASSEMBLY_PREFIX_WIDTH - 1, " ", attr, gap)
            return
        is_current = pc == listing.session.state.program_counter
        marker = glyphs.current_row if is_current else glyphs.normal_row
        breakpoint = "B" if pc in listing.session.control.breakpoints else " "
        canvas.add(y, x, f"{marker}{breakpoint}{pc:04d}  ", attr, limit - x)
        if is_current:
            canvas.add(y, x, marker, colors.pc_marker, 1)
        if breakpoint == "B":
            canvas.add(y, x + 1, breakpoint, colors.breakpoint, 1)

    def _slot_labels(self, listing: _Listing, pc: int) -> list[str]:
        """The slot name of each of instruction *pc*'s rows."""
        if listing.session.state.inst_mem[pc] is None:
            return []
        operations = all_operations(self.disassembly(listing.session, pc))
        return [
            name
            for name, op in zip(SLOT_LABELS, operations, strict=False)
            if listing.tab.fmt == "full" or op.upper().split(" ", 1)[0] != "NOP"
        ]

    def _draw_stages(
        self,
        canvas: Canvas,
        session: DebugSession,
        view: DisassemblyTabs,
        rect: Rect,
        *,
        focused: bool,
    ) -> None:
        """The commands each pipeline stage holds at the paused cycle."""
        colors = self.theme.colors
        pc = session.state.program_counter
        instruction = session.state.inst_mem[pc] if 0 <= pc < INST_MEM_SIZE else None
        width = max(1, rect.width - 3)
        heading = f"Pipeline stages / PC {pc:04d} / cycle {self.cycle} / ready"
        canvas.add(rect.y + 2, rect.x + 1, heading, colors.border, width)
        lines: list[tuple[str, int]] = []
        for stage, commands in pipeline_occupancy(instruction):
            if not commands:
                lines.append((f"{stage:<{STAGE_LABEL_WIDTH}} IDLE", colors.border))
            for index, (_slot, operation) in enumerate(commands):
                name = stage if index == 0 else ""
                text = f"{name:<{STAGE_LABEL_WIDTH}} {display_operation(operation)}"
                lines.extend(
                    (line, colors.key)
                    for line in textwrap.wrap(text, width, subsequent_indent="  ")
                )
        visible = max(1, rect.height - DISASSEMBLY_NON_CONTENT_ROWS)
        view.stages_scroll = max(0, min(view.stages_scroll, len(lines) - visible))
        first = view.stages_scroll
        canvas.top_pane_header(
            rect,
            [TITLE, "Stages"],
            f" {first + 1}-{min(len(lines), first + visible)}/{len(lines)} ",
            focused=focused,
        )
        for index, (line, attr) in enumerate(lines[first : first + visible]):
            canvas.add(rect.y + 3 + index, rect.x + 1, line, attr, width)

    # -- commands -------------------------------------------------------------

    def handle(
        self, command: Command, session: DebugSession, view: DisassemblyTabs, rect: Rect
    ) -> Reply | None:
        tab = view.current()
        assert tab is not None
        if command in (Command.ADD_TAB, Command.EDIT_TAB):
            return self._tab_editor(command, view)
        if command is Command.CYCLE_FORMAT:
            formats = DISASSEMBLY_FORMATS
            index = formats.index(tab.fmt) if tab.fmt in formats else -1
            tab.fmt = formats[(index + 1) % len(formats)]
            return Reply(f"Disassembly format: {tab.fmt}")
        if command is Command.EDIT_VALUE:
            pc = session.state.program_counter
            return scalar_editor(session, self.id, "pc", pc, PC_HINT)
        if command is Command.TOGGLE_BREAKPOINT:
            pc = cursor_pc(session, tab)
            action = "set" if session.toggle_breakpoint(pc) else "removed"
            return Reply(f"Breakpoint {action} at PC {pc}")
        if command is Command.RUN_TO_CURSOR:
            pc = cursor_pc(session, tab)
            if session.run_until(pc):
                return Reply(action=DebugAction.CONTINUE)
            return Reply(f"Already at PC {pc}")
        if command is Command.CANCEL:
            # Esc leaves the stage view.
            if tab.fmt == "stages":
                tab.fmt = "compact"
            return None
        self._navigate(command, session, view, rect)
        return None

    def _navigate(
        self, command: Command, session: DebugSession, view: DisassemblyTabs, rect: Rect
    ) -> None:
        """Arrows, pages and Home / End: the cursor, or the stage view's scroll."""
        tab = view.current()
        assert tab is not None
        moves = {Command.MOVE_UP: -1, Command.MOVE_DOWN: 1}
        if command in moves:
            self.move(session, view, rect, dy=moves[command])
        elif command in (Command.MOVE_LEFT, Command.MOVE_RIGHT):
            self.move(session, view, rect, dx=-1 if command is Command.MOVE_LEFT else 1)
        elif command in (Command.PAGE_UP, Command.PAGE_DOWN):
            direction = -1 if command is Command.PAGE_UP else 1
            if tab.fmt == "stages":
                page = max(1, rect.height - DISASSEMBLY_NON_CONTENT_ROWS)
            else:
                page = self._page_size(session, tab, rect)
            self.move(session, view, rect, dy=direction * page)
        elif command in (Command.JUMP_FIRST, Command.JUMP_LAST):
            end = command is Command.JUMP_LAST
            if tab.fmt == "stages":
                view.stages_scroll = STAGES_END if end else 0
            else:
                base = base_pc(session, tab)
                tab.cursor_offset = (INST_MEM_SIZE - 1 - base) if end else -base

    def _page_size(self, session: DebugSession, tab: DisassemblyTab, rect: Rect) -> int:
        """How many instructions one screen of *tab* shows."""
        visible = max(1, rect.height - DISASSEMBLY_NON_CONTENT_ROWS)
        _, first, last = self.rows(session, tab, cursor_pc(session, tab), visible)
        return max(1, last - first + 1)

    def move(
        self,
        session: DebugSession,
        view: DisassemblyTabs,
        rect: Rect,
        *,
        dx: int = 0,
        dy: int = 0,
    ) -> None:
        """Move the cursor *dy* instructions (the stage view scrolls *dy* lines).

        Instructions are one column wide, so *dx* moves nothing.
        """
        del rect, dx
        tab = view.current()
        assert tab is not None
        if tab.fmt == "stages" and dy:
            view.stages_scroll = max(0, view.stages_scroll + dy)
            return
        base = base_pc(session, tab)
        tab.cursor_offset = max(-base, min(INST_MEM_SIZE - 1 - base, tab.cursor_offset + dy))

    def _tab_editor(self, command: Command, view: DisassemblyTabs) -> Reply:
        tab = view.current()
        text = "current"
        mode = "edit" if command is Command.EDIT_TAB else "add"
        if mode == "edit" and tab is not None and tab.target is not None:
            text = str(tab.target)

        def parse(text: str, existing: Any) -> tuple[DisassemblyTab | None, str | None]:
            token = text.strip().lower()
            target: int | None = None
            if token != "current":
                try:
                    target = int(token, 0)
                except ValueError:
                    return None, "expected 'current' or a numeric PC"
                if not 0 <= target < INST_MEM_SIZE:
                    return None, f"PC must be between 0 and {INST_MEM_SIZE - 1}"
            fmt = existing.fmt if isinstance(existing, DisassemblyTab) else "compact"
            return DisassemblyTab(target=target, fmt=fmt), None

        return tab_editor(self.id, TabTarget("Disassembly", EDITOR_HINTS, parse), mode, view, text)

    # -- pointer --------------------------------------------------------------

    def place_cursor(self, session: DebugSession, view: DisassemblyTabs, payload: Any) -> None:
        """Put the cursor on instruction *payload* (a PC)."""
        tab = view.current()
        if tab is not None:
            tab.cursor_offset = payload - base_pc(session, tab)

    def cursor_row(self, session: DebugSession, view: DisassemblyTabs, rect: Rect) -> int:
        del rect
        tab = view.current()
        assert tab is not None
        return base_pc(session, tab) + tab.cursor_offset

    def scroll_to_row(
        self, session: DebugSession, view: DisassemblyTabs, rect: Rect, row: int
    ) -> None:
        del rect
        self.place_cursor(session, view, row)
