"""The registers pane on its own: session and tabs in, lines out.

No screen and no view: every test builds a ``DebugSession`` on a real state,
renders ``RegistersPane`` into a rectangle and reads the lines it returns.
"""

from __future__ import annotations

import struct

import pytest
from emu_support import make_state

from ipu_debug.control import DebugControl
from ipu_debug.session import REGFILE_SCHEMA, DebugSession, is_scalar
from ipu_debug.tui.draw import UNICODE_GLYPHS, Palette, Theme
from ipu_debug.tui.keymap import Command
from ipu_debug.tui.pane import Rect
from ipu_debug.tui.panes.registers import (
    REGISTER_GROUPS,
    REGISTER_TAB_GROUPS,
    SCALAR_REGISTER_FORMATS,
    RegistersPane,
    RegisterTab,
    commit_baselines,
    grid_shape,
)
from ipu_debug.tui.tabs import TabSet
from ipu_emu.emulator import DebugAction, run_with_debug
from ipu_emu.ipu_state import IpuState

RECT = Rect(10, 40, 12, 100)
COLORS = Palette()


def make(tab: RegisterTab | None = None) -> tuple[RegistersPane, DebugSession, TabSet]:
    pane = RegistersPane(Theme(COLORS, UNICODE_GLYPHS))
    return pane, DebugSession(IpuState()), TabSet([tab or RegisterTab()])


def text(drawing) -> str:
    return "\n".join(line.text for line in drawing.lines)


def styles_of(drawing, value: str) -> list[int]:
    return [line.style for line in drawing.lines if line.text == value]


def test_the_groups_are_the_schemas_scalar_registers():
    scalars = tuple(desc.name for desc in REGFILE_SCHEMA if is_scalar(desc))
    assert REGISTER_GROUPS == scalars
    assert REGISTER_TAB_GROUPS == ("all", *scalars)


@pytest.mark.parametrize(
    ("fmt", "value", "shown"),
    [
        ("hex", 0x89ABCDEF, "89abcdef"),
        ("u32", 4_000_000_000, "4000000000"),
        ("int32", 0xFFFFFFFE, "-2"),
        ("f32", struct.unpack("<I", struct.pack("<f", 2.5))[0], "2.5"),
        ("bits", 0b101, "101"),
    ],
)
@pytest.mark.parametrize("group", ["lr", "cr"])
def test_every_format_renders_both_register_groups(fmt, value, shown, group):
    assert fmt in SCALAR_REGISTER_FORMATS
    pane, session, tabs = make(RegisterTab(group=group, fmt=fmt))
    session.set_register(f"{group}5", value)

    drawing = pane.render(session, tabs, RECT)

    assert f"{group[0].upper()}05 " in [line.text for line in drawing.lines]
    assert shown in text(drawing)
    assert f"{group.upper()} 00-" in text(drawing)
    # Every drawn register can be clicked, and the title names the pane.
    assert len(drawing.hits) == 16
    assert "4 CR / LR" in text(drawing)


def test_the_cursor_is_highlighted_more_strongly_when_focused():
    pane, session, tabs = make()
    session.set_register("lr0", 0x12)

    focused = pane.render(session, tabs, RECT, focused=True)
    unfocused = pane.render(session, tabs, RECT, focused=False)

    assert styles_of(focused, "12") == [COLORS.selected]
    assert styles_of(unfocused, "12") == [COLORS.cursor]
    assert COLORS.selected != COLORS.cursor


def test_values_changed_by_a_step_are_marked():
    state = make_state("SET lr2 cr1;; NOP;; HALT;;")
    control = DebugControl()
    session = DebugSession(state, control)
    pane, _, tabs = make()
    drawings = []

    def paused(state, cycle):
        if not drawings:
            commit_baselines(session, tabs)
            drawings.append(pane.render(session, tabs, RECT))
            return DebugAction.STEP
        drawings.append(pane.render(session, tabs, RECT))
        return DebugAction.QUIT

    run_with_debug(state, paused, break_on_entry=True, control=control)
    before, after = drawings

    assert state.regfile.get_lr(2) == 1
    changed = [line for line in after.lines if line.style & COLORS.changed == COLORS.changed]
    assert "".join(line.text for line in changed) == "00000001"
    assert not [line for line in before.lines if line.style == COLORS.changed]


def test_a_narrow_pane_scrolls_its_columns_to_the_cursor():
    pane, session, tabs = make()
    narrow = Rect(0, 0, 12, 40)
    rows, columns = grid_shape(tabs[0], narrow)

    drawing = pane.render(session, tabs, narrow)
    assert len(drawing.hits) == 2 * rows  # Two of the columns fit.
    assert f"cols 1-2/{columns}" in text(drawing)
    assert all(line.x + len(line.text) <= narrow.width for line in drawing.lines)

    pane.handle(Command.JUMP_LAST, session, tabs, narrow)
    drawing = pane.render(session, tabs, narrow)
    assert tabs[0].column_scroll == columns - 2
    assert "CR 08-15" in text(drawing)


def test_arrows_move_within_the_grid_and_stop_at_its_edges():
    pane, session, tabs = make()
    rows, columns = grid_shape(tabs[0], RECT)

    pane.handle(Command.MOVE_UP, session, tabs, RECT)
    pane.handle(Command.MOVE_LEFT, session, tabs, RECT)
    assert tabs[0].cursor == 0
    pane.handle(Command.MOVE_RIGHT, session, tabs, RECT)
    pane.handle(Command.MOVE_DOWN, session, tabs, RECT)
    assert tabs[0].cursor == rows + 1
    pane.handle(Command.PAGE_DOWN, session, tabs, RECT)
    assert tabs[0].cursor == 2 * rows - 1
    assert columns * rows == 32


def test_the_scalar_editor_sets_through_the_session_and_reports_errors():
    pane, session, tabs = make(RegisterTab(group="cr"))
    tabs[0].cursor = 1

    reply = pane.handle(Command.EDIT_VALUE, session, tabs, RECT)
    assert reply is not None and reply.editor is not None
    assert reply.editor.register == "cr1" and reply.editor.text == "0x1"
    assert "CR0 and CR1 are read-only" in reply.editor.hints[1]

    refused = reply.editor.submit("7")
    assert refused.error and "read-only" in (refused.message or "")
    assert reply.editor.submit("nope").message == "Invalid integer value"

    tabs[0].cursor = 4
    generation = session.generation
    reply = pane.handle(Command.EDIT_VALUE, session, tabs, RECT)
    assert reply is not None and reply.editor is not None
    assert reply.editor.submit("0x20").message == "Set CR4 = 0x20"
    assert session.generation == generation + 1
    assert session.registers(["cr4"])[0].value == 0x20


def test_the_tab_editor_accepts_only_schema_groups():
    pane, session, tabs = make()

    reply = pane.handle(Command.ADD_TAB, session, tabs, RECT)
    assert reply is not None and reply.editor is not None
    assert reply.editor.hints[0] == "Group: all, lr, or cr"
    assert reply.editor.submit("r0").message == "registers tab error: expected all, lr, or cr"
    assert reply.editor.submit(" LR ").message == "registers tab added: lr hex"
    assert [tab.group for tab in tabs] == ["all", "lr"]


def test_cycling_the_format_walks_every_format():
    pane, session, tabs = make()

    seen = []
    for _ in SCALAR_REGISTER_FORMATS:
        reply = pane.handle(Command.CYCLE_FORMAT, session, tabs, RECT)
        assert reply is not None
        seen.append(tabs[0].fmt)

    assert sorted(seen) == sorted(SCALAR_REGISTER_FORMATS)
    assert reply.message == "LR / CR format: hex"
