"""The disassembly pane on its own: session and tabs in, lines out.

No screen and no view: every test builds a ``DebugSession`` on an assembled
program, renders ``DisassemblyPane`` into a rectangle and reads the lines it
returns.
"""

from __future__ import annotations

import pytest
from emu_support import make_state

import ipu_debug.tui.panes.disassembly as disassembly_module
from ipu_debug.control import DebugControl
from ipu_debug.session import DebugSession
from ipu_debug.tui.draw import UNICODE_GLYPHS, Palette, Theme
from ipu_debug.tui.keymap import Command
from ipu_debug.tui.pane import Rect
from ipu_debug.tui.panes.disassembly import (
    DISASSEMBLY_FORMATS,
    DISASSEMBLY_PARALLEL_MARKER,
    DisassemblyPane,
    DisassemblyTab,
    DisassemblyTabs,
    active_operations,
)
from ipu_emu.emulator import DebugAction
from ipu_emu.ipu_state import INST_MEM_SIZE, IpuState

# Wide enough for the slot column and aligned mnemonics.
WIDE = Rect(1, 40, 14, 100)
# Narrower than the slot column needs.
NARROW = Rect(1, 40, 14, 30)
COLORS = Palette()
MARKER = UNICODE_GLYPHS.current_row
TWO_SLOTS = "SET lr2 cr1; ADD lr3 lr2 cr1;; SET lr4 cr1;;"
FOUR_SLOTS = (
    "SET lr2 cr1; ADD lr3 lr2 cr1; "
    "MULT.RC.VV lr0 r1 0 lr0 cr15.l cr14.l; AGG.MAX.FIRST lr0 cr15.l;;"
)


def make(
    source: str | None = None, *tabs: DisassemblyTab
) -> tuple[DisassemblyPane, DebugSession, DisassemblyTabs]:
    state = make_state(source) if source else IpuState()
    pane = DisassemblyPane(Theme(COLORS, UNICODE_GLYPHS), cycle=17)
    return (
        pane,
        DebugSession(state, DebugControl()),
        DisassemblyTabs(list(tabs or [DisassemblyTab()])),
    )


def rows(drawing, rect: Rect = WIDE) -> dict[int, str]:
    """Each row of *rect* as it reads once every line is laid down."""
    grid: dict[int, list[str]] = {}
    for line in drawing.lines:
        row = grid.setdefault(line.y, [" "] * (rect.x + rect.width))
        for offset, character in enumerate(line.text):
            row[line.x + offset] = character
    return {y: "".join(row)[rect.x :].rstrip() for y, row in grid.items()}


def content(drawing, rect: Rect = WIDE) -> list[str]:
    """The listing rows below the column header, each run of spaces after the
    PC prefix squeezed to one (mnemonics are padded to a shared column)."""
    shown = rows(drawing, rect)
    listing = [shown.get(y, "") for y in range(rect.y + 3, rect.y + rect.height - 1)]
    return [row[:9] + " ".join(row[9:].split()) for row in listing]


class TestListing:
    def test_parallel_operations_are_stacked_under_their_instruction(self):
        pane, session, tabs = make(TWO_SLOTS)

        listing = content(pane.render(session, tabs, WIDE))

        assert listing[0].startswith(f" {MARKER} 0000  SET lr2 cr1")
        assert listing[1].startswith(f"      {DISASSEMBLY_PARALLEL_MARKER} ADD lr3 lr2 cr1")
        # The next instruction starts a new group, not a new column.
        assert listing[2].startswith("   0001  SET lr4 cr1")

    def test_no_instruction_spans_the_pane_sideways(self):
        pane, session, tabs = make(FOUR_SLOTS)

        listing = content(pane.render(session, tabs, WIDE))[:4]

        mnemonics = ("SET", "ADD", "MULT.RC.VV", "AGG.MAX.FIRST")
        assert [sum(row.count(m) for m in mnemonics) for row in listing] == [1, 1, 1, 1]
        assert [row.count("lr3") for row in listing] == [0, 1, 0, 0]

    def test_mnemonics_share_one_column_across_slots_and_instructions(self):
        pane, session, tabs = make(FOUR_SLOTS + " SET lr0 cr1;;")

        drawing = pane.render(session, tabs, WIDE)

        operations = [
            line
            for line in drawing.lines
            if line.text.startswith(("SET ", "ADD ", "MULT.RC.VV ", "AGG.MAX.FIRST "))
        ]
        assert len(operations) == 5
        first_operand = {
            line.x + len(line.text) - len(line.text.split(" ", 1)[1].lstrip())
            for line in operations
        }
        assert first_operand == {WIDE.x + 1 + len("> 0000  ") + len("AGG.MAX.FIRST ")}

    def test_a_narrow_pane_gives_operands_the_room_instead(self):
        pane, session, tabs = make(FOUR_SLOTS)

        listing = content(pane.render(session, tabs, NARROW), NARROW)

        assert "SET lr2 cr1" in listing[0]
        assert "SLOT" not in rows(pane.render(session, tabs, NARROW), NARROW)[NARROW.y + 2]
        assert listing[2].endswith(UNICODE_GLYPHS.ellipsis)

    def test_the_parallel_marker_recedes(self):
        pane, session, tabs = make(TWO_SLOTS)

        drawing = pane.render(session, tabs, WIDE)

        markers = [line for line in drawing.lines if line.text == DISASSEMBLY_PARALLEL_MARKER]
        assert markers and all(line.style & COLORS.border for line in markers)

    def test_compact_leaves_nops_out_and_full_shows_every_slot(self):
        pane, session, tabs = make("SET lr2 cr1;;", DisassemblyTab(), DisassemblyTab(fmt="full"))

        compact = "\n".join(content(pane.render(session, tabs, WIDE)))
        tabs.select(1)
        full = "\n".join(content(pane.render(session, tabs, WIDE)))

        assert "SET lr2 cr1" in compact and "NOP" not in compact.split("0001")[0]
        assert full.split("0001")[0].count("NOP") > 1

    def test_the_slot_column_names_each_operation_slot(self):
        pane, session, tabs = make(TWO_SLOTS)

        drawing = pane.render(session, tabs, WIDE)

        listing = content(drawing)
        assert "SLOT" in rows(drawing)[WIDE.y + 2]
        assert listing[0].endswith(" LR0") and listing[1].endswith(" LR1")

    def test_empty_instruction_memory_shows_nops_that_recede(self):
        pane, session, tabs = make()

        drawing = pane.render(session, tabs, WIDE)

        assert content(drawing)[1].startswith("   0001  <nop>")
        nops = [line for line in drawing.lines if line.text.startswith("<nop>")]
        assert any(line.style == COLORS.border for line in nops)

    def test_the_operations_of_an_instruction_are_its_active_slots(self):
        pane, session, _ = make("SET lr2 cr1; ADD lr3 lr2 cr1;;")

        assert active_operations(pane.disassembly(session, 0)) == [
            "SET lr2 cr1",
            "ADD lr3 lr2 cr1",
        ]

    def test_an_instruction_is_decoded_once_per_pane(self, monkeypatch):
        decoded: list[object] = []
        real = disassembly_module.stage_operations

        def counting(instruction):
            decoded.append(instruction)
            return real(instruction)

        monkeypatch.setattr(disassembly_module, "stage_operations", counting)
        pane, session, tabs = make(TWO_SLOTS)

        pane.render(session, tabs, WIDE)
        first = len(decoded)
        for _ in range(5):
            pane.render(session, tabs, WIDE)

        assert first == 2
        assert len(decoded) == first


class TestCursorAndReadout:
    def test_every_row_of_the_cursor_instruction_is_highlighted(self):
        pane, session, tabs = make(TWO_SLOTS)

        focused = pane.render(session, tabs, WIDE, focused=True)
        idle = pane.render(session, tabs, WIDE)

        def marked(drawing, style):
            return {line.y for line in drawing.lines if line.style & style == style}

        top = WIDE.y + 3
        assert {top, top + 1} <= marked(focused, COLORS.selected)
        assert top + 2 not in marked(focused, COLORS.selected)
        assert {top, top + 1} <= marked(idle, COLORS.cursor)

    def test_each_row_is_a_hit_for_its_pc_and_the_scroll_covers_memory(self):
        pane, session, tabs = make(TWO_SLOTS)

        drawing = pane.render(session, tabs, WIDE)

        assert [pc for _, pc in drawing.hits[:3]] == [0, 0, 1]
        first, shown, total = drawing.scroll
        assert (first, total) == (0, INST_MEM_SIZE)
        assert rows(drawing)[WIDE.y].endswith(f" 0000-{shown - 1:04d}/{INST_MEM_SIZE}")
        assert "2 Instructions" in rows(drawing)[WIDE.y]

    def test_arrows_page_and_jump_move_the_cursor_within_memory(self):
        pane, session, tabs = make()
        session.state.program_counter = 10

        pane.handle(Command.MOVE_DOWN, session, tabs, WIDE)
        assert tabs[0].cursor_offset == 1
        pane.handle(Command.MOVE_RIGHT, session, tabs, WIDE)
        assert tabs[0].cursor_offset == 1
        pane.handle(Command.PAGE_DOWN, session, tabs, WIDE)
        assert tabs[0].cursor_offset == 1 + WIDE.height - 4
        pane.handle(Command.JUMP_LAST, session, tabs, WIDE)
        assert tabs[0].cursor_offset == INST_MEM_SIZE - 1 - 10
        pane.handle(Command.MOVE_DOWN, session, tabs, WIDE)
        assert tabs[0].cursor_offset == INST_MEM_SIZE - 1 - 10
        pane.handle(Command.JUMP_FIRST, session, tabs, WIDE)
        assert tabs[0].cursor_offset == -10

    def test_a_fixed_pc_tab_counts_its_cursor_from_its_target(self):
        pane, session, tabs = make(None, DisassemblyTab(target=20))

        pane.place_cursor(session, tabs, 23)
        drawing = pane.render(session, tabs, WIDE)

        assert tabs[0].cursor_offset == 3
        assert pane.cursor_row(session, tabs, WIDE) == 23
        assert any(line.text.startswith("  0023") for line in drawing.lines)
        pane.scroll_to_row(session, tabs, WIDE, 40)
        assert tabs[0].cursor_offset == 20


class TestBreakpointsAndRunToCursor:
    def test_toggle_marks_the_cursor_instruction_through_the_session(self):
        pane, session, tabs = make(TWO_SLOTS)
        pane.handle(Command.MOVE_DOWN, session, tabs, WIDE)
        generation = session.generation

        reply = pane.handle(Command.TOGGLE_BREAKPOINT, session, tabs, WIDE)

        assert reply is not None and reply.message == "Breakpoint set at PC 1"
        assert session.control.breakpoints == {1}
        assert session.generation == generation + 1
        drawing = pane.render(session, tabs, WIDE)
        assert content(drawing)[2].startswith("  B0001")
        assert [line.style for line in drawing.lines if line.text == "B"] == [COLORS.breakpoint]
        reply = pane.handle(Command.TOGGLE_BREAKPOINT, session, tabs, WIDE)
        assert reply is not None and reply.message == "Breakpoint removed at PC 1"
        assert session.control.breakpoints == set()

    def test_run_to_cursor_arms_the_target_or_reports_it_is_here(self):
        pane, session, tabs = make(TWO_SLOTS)

        reply = pane.handle(Command.RUN_TO_CURSOR, session, tabs, WIDE)
        assert reply is not None and reply.message == "Already at PC 0" and reply.action is None
        assert session.control.run_target is None

        pane.handle(Command.MOVE_DOWN, session, tabs, WIDE)
        reply = pane.handle(Command.RUN_TO_CURSOR, session, tabs, WIDE)

        assert reply is not None and reply.action is DebugAction.CONTINUE
        assert session.control.run_target == 1


class TestStageView:
    def test_stages_group_the_current_commands_by_physical_stage(self):
        pane, session, tabs = make("SET lr2 cr1;; ADD lr3 lr2 cr1;;", DisassemblyTab(fmt="stages"))
        session.state.program_counter = 1

        text = "\n".join(rows(pane.render(session, tabs, WIDE)).values())

        assert "Pipeline stages / PC 0001 / cycle 17 / ready" in text
        assert "ADD LR3, LR2, CR1" in text
        assert "MULT       IDLE" in text
        assert "Stages" in text

    def test_arrows_scroll_the_stages_and_escape_leaves_them(self):
        pane, session, tabs = make(FOUR_SLOTS, DisassemblyTab(fmt="stages"))
        short = Rect(1, 40, 6, 100)

        pane.handle(Command.MOVE_DOWN, session, tabs, short)
        assert tabs.stages_scroll == 1
        pane.move(session, tabs, short, dy=3)
        assert tabs.stages_scroll == 4
        pane.handle(Command.JUMP_LAST, session, tabs, short)
        pane.render(session, tabs, short)
        last = tabs.stages_scroll
        assert 0 < last < 10000
        pane.handle(Command.JUMP_FIRST, session, tabs, short)
        assert tabs.stages_scroll == 0
        assert tabs[0].cursor_offset == 0

        pane.handle(Command.CANCEL, session, tabs, short)
        assert tabs[0].fmt == "compact"


class TestEditors:
    def test_cycling_the_format_walks_every_format_per_tab(self):
        pane, session, tabs = make(None, DisassemblyTab(), DisassemblyTab(4))

        seen = []
        for _ in DISASSEMBLY_FORMATS:
            reply = pane.handle(Command.CYCLE_FORMAT, session, tabs, WIDE)
            assert reply is not None
            seen.append(tabs[0].fmt)

        assert seen == ["full", "stages", "compact"]
        assert reply.message == "Disassembly format: compact"
        assert tabs[1].fmt == "compact"

    def test_the_tab_editor_accepts_current_and_a_fixed_pc(self):
        pane, session, tabs = make()

        reply = pane.handle(Command.ADD_TAB, session, tabs, WIDE)
        assert reply is not None and reply.editor is not None
        assert reply.editor.text == "current"
        assert reply.editor.submit("0x10").message == "disassembly tab added: 16 compact"
        reply = pane.handle(Command.EDIT_TAB, session, tabs, WIDE)
        assert reply is not None and reply.editor is not None and reply.editor.text == "16"
        assert reply.editor.submit("current").message == "disassembly tab updated: current compact"
        assert [tab.target for tab in tabs] == [None, None]

    @pytest.mark.parametrize(
        ("text", "error"),
        [
            (str(INST_MEM_SIZE), f"PC must be between 0 and {INST_MEM_SIZE - 1}"),
            ("next", "expected 'current' or a numeric PC"),
        ],
    )
    def test_the_tab_editor_rejects_what_is_not_a_pc(self, text, error):
        pane, session, tabs = make()

        reply = pane.handle(Command.ADD_TAB, session, tabs, WIDE)
        assert reply is not None and reply.editor is not None
        refused = reply.editor.submit(text)

        assert refused.error and refused.message == f"disassembly tab error: {error}"
        assert len(tabs) == 1

    def test_the_value_editor_sets_the_pc_through_the_session(self):
        pane, session, tabs = make()

        reply = pane.handle(Command.EDIT_VALUE, session, tabs, WIDE)
        assert reply is not None and reply.editor is not None
        assert reply.editor.register == "pc" and reply.editor.title == "Edit PC value"

        refused = reply.editor.submit(str(INST_MEM_SIZE))
        assert refused.error and "PC must be between" in (refused.message or "")
        assert reply.editor.submit("0x3").message == "Set PC = 0x3"
        assert session.state.program_counter == 3
