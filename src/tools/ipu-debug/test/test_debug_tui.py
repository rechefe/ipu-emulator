"""Deterministic tests for the persistent curses debugger view."""

from __future__ import annotations

import struct

import pytest
from emu_support import ElementFormat, make_state, new_state

import ipu_debug.format as debug_format
import ipu_debug.tui.chrome as chrome
import ipu_debug.tui.draw as draw
import ipu_debug.tui.layout as tui_layout
import ipu_debug.tui.panes.registers as registers_pane
import ipu_debug.tui.pointer as pointer
import ipu_debug.tui.view as debug_tui
from ipu_debug.control import DebugControl
from ipu_debug.session import DebugSession, XmemRequest
from ipu_debug.tui.keymap import KEY_BINDINGS, PANE_FOCUS_ORDER, footer_text, help_lines
from ipu_debug.tui.layout import MIN_TERMINAL_COLUMNS
from ipu_debug.tui.panes.disassembly import DisassemblyTab
from ipu_debug.tui.panes.pipeline import (
    PIPELINE_REGISTERS,
    PipelineTab,
    pipeline_spec,
    read_register,
    value_grid,
)
from ipu_debug.tui.panes.registers import RegisterTab
from ipu_debug.tui.panes.xmem import XmemTab
from ipu_debug.tui.screen import COLOR_MASK, FakeScreen, Key, MouseEvent
from ipu_debug.tui.view import CursesDebugView, run_debug_tui
from ipu_debug.tui.workspace import DebugViewSession, new_debug_view_session
from ipu_emu.emulator import DebugAction
from ipu_emu.ipu import xmem_row_size_bytes
from ipu_emu.ipu_state import INST_MEM_SIZE, IpuState


def cell(screen: FakeScreen, y: int, x: int) -> str:
    """The character finally left at ``(y, x)``."""
    return screen.cell(y, x)


def row_text(screen: FakeScreen, y: int) -> str:
    """The row as it finally appears, gaps between writes included."""
    return screen.row(y)


def make_view(
    state: IpuState | None = None,
    *,
    screen: FakeScreen | None = None,
    session: DebugViewSession | None = None,
) -> tuple[CursesDebugView, FakeScreen, DebugViewSession]:
    state = state or IpuState()
    screen = screen or FakeScreen()
    # Most interaction tests use one explicit tab per pane. Default-tab
    # integration tests pass a fresh session explicitly.
    session = session or DebugViewSession.with_tabs(
        initialized=True,
        xmem=[XmemTab(XmemRequest("row", "0", "0", xmem_row_size_bytes(state), "hex"))],
        disassembly=[DisassemblyTab()],
        registers=[RegisterTab()],
        pipeline=[PipelineTab()],
    )
    session.ensure_default_tab(state)
    return CursesDebugView(screen, state, session, cycle=17), screen, session


class TestLayout:
    def test_minimum_terminal_has_all_panes(self):
        view, _, _ = make_view()

        layout = view.layout()

        assert layout is not None
        assert set(layout) == {
            "header",
            "disassembly",
            "registers",
            "xmem",
            "pipeline",
            "status",
            "keys",
        }
        assert layout["pipeline"].x + layout["pipeline"].width - 1 == (layout["disassembly"].x)
        assert layout["disassembly"].y + layout["disassembly"].height - 1 == (layout["xmem"].y)
        assert layout["xmem"].x + layout["xmem"].width - 1 == (layout["registers"].x)

    def test_small_terminal_shows_resize_message(self):
        screen = FakeScreen(rows=10, columns=70)
        view, _, _ = make_view(screen=screen)

        view.render()

        assert "Terminal must be at least 48x12" in screen.text()

    def test_render_shows_status_registers_disassembly_and_xmem(self):
        state = IpuState()
        state.regfile.set_lr(0, 0x12)
        view, screen, _ = make_view(state)

        view.render()

        output = screen.text()
        assert "cycle=17" in output
        assert "Instructions" in output
        assert "LR 00-07" in output
        assert "Pipeline Registers" in output
        assert "12" in output
        assert "row:0+0 hex" in output
        assert "0x00000000" in output
        assert "128 bytes" in output
        assert "PC=0000" in output
        assert "q Quit" in output
        header = "".join(text for y, _, text, _ in screen.writes if y == 0)
        assert "[F5" not in header
        assert any(
            text == f"{view.glyphs.current_row} 0000  "
            and attr == view.colors.current | view.colors.cursor
            for _, _, text, attr in screen.writes
        )

    def test_register_columns_are_complete_and_separate(self):
        state = IpuState()
        state.regfile.set_lr(0, 0x12345678)
        state.regfile.set_lr(8, 0x89ABCDEF)
        state.regfile.set_cr(2, 0x10203040)
        state.regfile.set_cr(8, 0x50607080)
        view, screen, _ = make_view(state, screen=FakeScreen(columns=140))

        view.render()

        written = [text for _, _, text, _ in screen.writes]
        assert "L00 " in written
        assert "L08 " in written
        assert "C00 " in written
        assert "C01 " in written
        assert "C02 " in written
        assert "C08 " in written
        assert "12345678" in written
        assert "89abcdef" in written
        assert f"{state.regfile.get_cr(0):08x}" in written
        assert state.regfile.get_cr(0) == 0
        assert state.regfile.get_cr(1) == 1
        assert f"{state.regfile.get_cr(2):08x}" in written
        assert f"{state.regfile.get_cr(8):08x}" in written

    def test_register_columns_never_touch_each_other(self):
        view, screen, _ = make_view(screen=FakeScreen(columns=140))

        view.render()

        rect = view.rects["registers"]
        row = sorted(
            (x, text)
            for y, x, text, _ in screen.writes
            if y == rect.y + 3 and rect.x < x < rect.x + rect.width - 1 and text.strip()
        )
        for (x, text), (next_x, _) in zip(row, row[1:], strict=False):
            assert x + len(text) <= next_x
        labels = [text for _, text in row if text.endswith(" ")]
        assert labels == ["L00 ", "L08 ", "C00 ", "C08 "]

    def test_wide_terminal_labels_scalar_register_groups(self):
        view, screen, _ = make_view(screen=FakeScreen(columns=140))

        view.render()

        assert "LR 00-07" in screen.text()
        assert "LR 08-15" in screen.text()
        assert "CR 00-07" in screen.text()
        assert "CR 08-15" in screen.text()

    def test_narrow_register_pane_scrolls_columns_instead_of_crowding(self):
        view, screen, session = make_view()

        view.render()

        assert "cols 1-2/4" in screen.text()
        assert "CR 08-15" not in screen.text()

        session.focus = "registers"
        view.handle_key(Key.END)
        screen.clear()
        view.render()

        assert "cols 3-4/4" in screen.text()
        assert "CR 08-15" in screen.text()

    def test_tab_bar_scrolls_to_keep_active_tab_visible(self):
        view, _, session = make_view()
        session.disassembly.tabs.extend(DisassemblyTab(index) for index in range(1, 8))
        session.disassembly.active = 7

        view.render()

        assert any(index == 7 for _, index in view.hits.tabs["disassembly"])

    def test_tab_bar_marks_tabs_scrolled_out_of_view(self):
        view, screen, session = make_view()
        session.disassembly.tabs.extend(DisassemblyTab(index) for index in range(1, 8))

        view.render()

        written = [text for _, _, text, _ in screen.writes]
        assert view.glyphs.more_after in written
        assert view.glyphs.more_before not in written

        session.disassembly.active = 7
        screen.clear()
        view.render()

        assert view.glyphs.more_before in [text for _, _, text, _ in screen.writes]

    def test_only_focused_pane_uses_selected_tab_color(self):
        view, screen, session = make_view()
        session.focus = "xmem"

        view.render()

        tab_attributes = {
            text: attr
            for _, _, text, attr in screen.writes
            if text in {"[current compact x]", "[all hex x]", "[row:0+0 hex x]"}
        }
        assert tab_attributes["[row:0+0 hex x]"] == view.colors.selected
        assert tab_attributes["[current compact x]"] == view.colors.current
        assert tab_attributes["[all hex x]"] == view.colors.current

    def test_minimum_footer_uses_readable_contextual_labels(self):
        view, screen, _ = make_view()

        view.render()

        assert "F5:Run" in screen.text()
        assert "F8:Step" in screen.text()
        assert "F6" not in screen.text()
        assert "TabTabs" not in screen.text()
        assert "q Quit" in screen.text()
        assert "[? Help]" in screen.text()

    def test_wide_footer_includes_global_controls(self):
        view, screen, _ = make_view(screen=FakeScreen(columns=140))

        view.render()

        assert "F5:Run" in screen.text()
        assert "F5:Run" in screen.text()
        assert "F8:Step" in screen.text()


class TestPaneFrames:
    def test_shared_edges_resolve_into_junctions(self):
        view, screen, _ = make_view(screen=FakeScreen(columns=140))

        view.render()

        layout = view.layout()
        assert layout is not None
        box = view.glyphs.box
        split_x = layout["disassembly"].x
        shared_y = layout["xmem"].y
        assert (
            cell(screen, layout["disassembly"].y, split_x)
            == box[chrome.BORDER_LEFT | chrome.BORDER_RIGHT | chrome.BORDER_DOWN]
        )
        assert (
            cell(screen, shared_y, layout["pipeline"].x)
            == box[chrome.BORDER_UP | chrome.BORDER_RIGHT | chrome.BORDER_DOWN]
        )
        assert (
            cell(screen, shared_y, split_x)
            == box[chrome.BORDER_UP | chrome.BORDER_LEFT | chrome.BORDER_RIGHT]
        )
        assert (
            cell(screen, shared_y, layout["registers"].x)
            == box[chrome.BORDER_LEFT | chrome.BORDER_RIGHT | chrome.BORDER_DOWN]
        )

    def test_focused_pane_owns_the_attribute_of_a_shared_edge(self):
        view, screen, session = make_view(screen=FakeScreen(columns=140))
        session.focus = "disassembly"

        view.render()

        split_x = view.rects["disassembly"].x
        row = view.rects["disassembly"].y + 3
        assert any(
            (y, x) == (row, split_x) and attr == view.colors.focus
            for y, x, _, attr in screen.writes
        )

    def test_ascii_glyphs_stay_ascii(self):
        glyphs = draw.ASCII_GLYPHS
        drawn = "".join(glyphs.box) + glyphs.thumb + glyphs.separator
        drawn += glyphs.current_row + glyphs.more_before + glyphs.more_after
        drawn += glyphs.ellipsis
        assert drawn.isascii()
        assert len(glyphs.box) == len(draw.UNICODE_GLYPHS.box) == 16

    def test_glyph_set_follows_the_terminal_encoding(self, monkeypatch):
        monkeypatch.setattr(
            draw.locale,
            "getpreferredencoding",
            lambda _do_setlocale=False: "ascii",
        )
        assert draw.detect_glyphs() is draw.ASCII_GLYPHS

        monkeypatch.setattr(
            draw.locale,
            "getpreferredencoding",
            lambda _do_setlocale=False: "utf-8",
        )
        assert draw.detect_glyphs() is draw.UNICODE_GLYPHS


class TestTitleFitting:
    def test_whole_segments_are_dropped_rather_than_cut(self):
        view, _, _ = make_view()
        separator = view.glyphs.separator

        segments = ["XMEM", "0x10", "hex"]

        assert draw.fit_segments(segments, 40, view.glyphs) == (
            f"XMEM {separator} 0x10 {separator} hex"
        )
        assert draw.fit_segments(segments, 14, view.glyphs) == (f"XMEM {separator} 0x10")
        assert draw.fit_segments(segments, 4, view.glyphs) == "XMEM"
        assert len(draw.fit_segments(["Disassembly"], 6, view.glyphs)) <= 6

    def test_squeezed_pane_keeps_its_name_and_drops_the_readout(self):
        view, screen, session = make_view()
        session.focus = "pipeline"
        for _ in range(50):
            view.handle_key(Key.SHIFT_RIGHT)

        view.render()

        text = screen.text()
        assert view.rects["disassembly"].width == (tui_layout.MIN_DISASSEMBLY_COLUMNS)
        assert "Instructions" in text
        assert f"/{INST_MEM_SIZE} " not in text

    def test_pane_shows_both_the_title_and_the_readout_when_they_fit(self):
        view, screen, _ = make_view()

        view.render()

        text = screen.text()
        assert "Instructions" in text
        assert f"/{INST_MEM_SIZE} " in text


class TestSessionPersistence:
    def test_default_tab_is_one_full_active_row(self):
        normal = new_debug_view_session(IpuState())
        wide_state = new_state(fmt=ElementFormat.FP32)
        wide = new_debug_view_session(wide_state)

        assert normal.xmem.tabs[0].request.count == xmem_row_size_bytes(IpuState())
        assert wide.xmem.tabs[0].request.count == xmem_row_size_bytes(wide_state)

    def test_the_session_passed_to_each_stop_persists(self, monkeypatch):
        state = IpuState()
        session = new_debug_view_session(state)
        seen = []

        def in_terminal(body):
            return body(FakeScreen([Key.F5]))

        class View:
            def __init__(self, screen, state, session, cycle, control, debug):
                self.session, self.control = session, control

            def run(self):
                seen.append((self.session, self.control))
                if len(seen) == 1:
                    self.session.set_message("persistent message")
                    self.session.xmem.tabs[0].scroll = 3
                return DebugAction.CONTINUE

        monkeypatch.setattr(debug_tui, "run_in_terminal", in_terminal)
        monkeypatch.setattr(debug_tui, "CursesDebugView", View)
        control = DebugControl()
        run_debug_tui(state, 0, control=control, session=session)
        run_debug_tui(state, 1, control=control, session=session)

        assert seen == [(session, control)] * 2
        assert session.message == "persistent message"
        assert session.xmem.tabs[0].scroll == 3

    def test_step_keeps_view_active_and_captures_baseline(self):
        state = IpuState()
        view, _, session = make_view(state)
        state.regfile.set_lr(0, 9)

        result = view.handle_key(Key.F8)

        assert result == DebugAction.STEP
        assert session.active
        assert session.registers.tabs[0].baseline[("lr", 0)] == 9

    def test_comparison_baselines_are_stored_per_tab(self):
        state = IpuState()
        view, _, session = make_view(state)
        session.registers.tabs.append(RegisterTab("lr"))
        session.pipeline.tabs.append(PipelineTab("r1"))

        view.handle_key(Key.F8)

        assert all(tab.baseline is not None for tab in session.registers.tabs)
        assert all(tab.baseline_data is not None for tab in session.pipeline.tabs)

    def test_new_tab_has_no_stale_comparison_baseline(self):
        state = IpuState()
        view, _, session = make_view(state)
        view.handle_key(Key.F8)
        session.focus = "registers"

        view.handle_key(Key.F2)
        view.prompt.text = "lr"
        view.handle_key(Key.ENTER)

        assert session.registers.tabs[0].baseline is not None
        assert session.registers.tabs[1].baseline is None

    def test_escape_does_not_leave_tui(self):
        view, _, session = make_view()
        session.active = True

        result = view.handle_key(Key.ESCAPE)

        assert result is None
        assert session.active


class TestDedicatedControls:
    def test_printable_input_is_ignored_outside_xmem_editor(self):
        view, _, _ = make_view()

        view.handle_key("s")
        view.handle_key(Key.ENTER)

        assert view.prompt.text == ""
        assert view.prompt.mode is None

    def test_f5_f8_and_q_are_direct_execution_controls(self):
        continue_view, _, continue_session = make_view()
        step_view, _, step_session = make_view()
        quit_view, _, quit_session = make_view()
        f10_view, _, f10_session = make_view()

        assert continue_view.handle_key(Key.F5) == DebugAction.CONTINUE
        assert continue_session.active
        assert step_view.handle_key(Key.F8) == DebugAction.STEP
        assert step_session.active
        assert quit_view.handle_key("q") == DebugAction.QUIT
        assert not quit_session.active
        assert f10_view.handle_key(Key.F10) is None
        assert f10_view.state.program_counter == 0
        assert not f10_session.active

    @pytest.mark.parametrize("pane", PANE_FOCUS_ORDER)
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            (lambda: Key.F5, DebugAction.CONTINUE),
            (lambda: Key.F8, DebugAction.STEP),
        ],
    )
    def test_execution_keys_work_without_changing_focus(self, pane, key, expected):
        view, _, session = make_view()
        session.focus = pane

        assert view.handle_key(key()) == expected
        assert session.focus == pane

    def test_xmem_editor_supports_cursor_editing_and_cancel(self):
        view, screen, session = make_view()

        view.handle_key(Key.F2)
        assert view.prompt.mode == "add"
        original = view.prompt.text
        assert view.handle_key(Key.F5) is None
        assert view.handle_key(MouseEvent(0, 0)) is None
        assert not session.active
        assert view.prompt.text == original
        view.render()
        cleared_rows = {
            y
            for y, x, text, _ in screen.writes
            if x == view.prompt.rect.x and text == " " * view.prompt.rect.width
        }
        assert len(cleared_rows) == view.prompt.rect.height
        assert "Address: row|byte BASE OFFSET COUNT" in screen.text()
        assert "Formats: hex  int8  u8  cell16  u32  f32" in screen.text()
        view.handle_key(Key.HOME)
        view.handle_key("x")
        assert view.prompt.text == "x" + original
        view.handle_key(Key.ESCAPE)

        assert view.prompt.mode is None
        assert session.message == "XMEM tab edit cancelled"


class TestXmemTabs:
    def test_f2_adds_valid_symbolic_xmem_tab(self):
        state = IpuState()
        state.regfile.set_cr(4, 2)
        state.regfile.set_lr(0, 1)
        view, _, session = make_view(state)

        view.handle_key(Key.F2)
        view.prompt.text = "xmem row cr4 lr0 4 u32"
        view.prompt.cursor = len(view.prompt.text)
        view.handle_key(Key.ENTER)

        assert len(session.xmem.tabs) == 2
        tab = session.xmem.tabs[-1]
        assert tab.request.base_token == "cr4"
        resolved = DebugSession(state).read_xmem(tab.request)
        assert resolved.metadata["row"] == 3

        state.regfile.set_lr(0, 2)
        resolved = DebugSession(state).read_xmem(tab.request)
        assert resolved.metadata["row"] == 4

    def test_invalid_tab_request_stays_in_editor(self):
        view, _, session = make_view()
        original_count = len(session.xmem.tabs)
        view.handle_key(Key.F2)
        view.prompt.text = "xmem byte 1 0 1 f32"
        view.prompt.cursor = len(view.prompt.text)

        view.handle_key(Key.ENTER)

        assert len(session.xmem.tabs) == original_count
        assert view.prompt.mode == "add"
        assert session.message_is_error
        assert "4-byte aligned" in session.message

    def test_edit_replaces_active_tab_and_f4_closes_it(self):
        view, _, session = make_view()
        view.handle_key(Key.F3)
        view.prompt.text = "xmem byte 16 0 8 u8"
        view.prompt.cursor = len(view.prompt.text)
        view.handle_key(Key.ENTER)

        assert session.xmem.tabs[0].request == XmemRequest("byte", "16", "0", 8, "u8")
        view.handle_key(Key.F4)
        assert session.xmem.tabs == []

    @pytest.mark.parametrize(
        ("key", "pane"),
        [(str(number), pane) for number, pane in enumerate(PANE_FOCUS_ORDER, start=1)],
    )
    def test_number_keys_focus_a_pane_directly(self, key, pane):
        view, _, session = make_view()

        view.handle_key(key)

        assert session.focus == pane
        assert session.message == "Ready"

    def test_number_keys_inside_an_editor_are_typed_text(self):
        view, _, session = make_view()
        session.focus = "xmem"
        view.handle_key(Key.F2)

        view.handle_key("4")

        assert session.focus == "xmem"
        assert view.prompt.text.endswith("4")

    def test_shift_tab_cycles_pane_focus_and_tab_does_not(self):
        view, _, session = make_view()

        assert session.focus == "xmem"
        view.handle_key(Key.SHIFT_TAB)
        assert session.focus == "registers"
        view.handle_key(Key.SHIFT_TAB)
        assert session.focus == "pipeline"
        view.handle_key(Key.TAB)
        assert session.focus == "pipeline"
        assert session.message.startswith("Selected")

    def test_tab_cycles_xmem_tabs(self):
        _, _, session = make_view()
        session.xmem.tabs.append(XmemTab(XmemRequest("byte", "16", "0", 4, "hex")))
        view, _, _ = make_view(session=session)

        view.handle_key(Key.TAB)
        assert session.xmem.active == 1
        view.handle_key(Key.TAB)
        assert session.xmem.active == 0

    def test_arrows_move_xmem_cursor_within_active_tab(self):
        view, _, session = make_view()
        view.render()
        tab = session.xmem.tabs[0]
        starting_line = tab.cursor_line

        view.handle_key(Key.RIGHT)
        assert tab.cursor_column == 1
        view.handle_key(Key.DOWN)
        assert tab.cursor_line == starting_line + 1
        assert session.xmem.active == 0

    def test_xmem_cursor_is_highlighted(self):
        view, screen, _ = make_view()
        view.render()

        assert any(
            text == "00" and attr & view.colors.selected for _, _, text, attr in screen.writes
        )

    def test_cell16_selection_preserves_encoded_color_pair(self):
        state = IpuState()
        cell = 0 | (14 << 8) | (4 << 12)
        state.xmem.write_address(0, struct.pack("<H", cell))
        session = DebugViewSession.with_tabs(
            initialized=True,
            xmem=[XmemTab(XmemRequest("byte", "0", "0", 1, "cell16"))],
        )
        view, screen, _ = make_view(state, session=session)
        encoded_color = 1 << 20
        selection_emphasis = 1 << 5
        generic_selection = 1 << 18
        seen: list[tuple[int, int]] = []

        def cell_color(fg: int, bg: int) -> int:
            seen.append((fg, bg))
            return encoded_color

        view.colors.cell = cell_color
        view.colors.cell_selected = selection_emphasis
        view.colors.selected = generic_selection
        view.render()

        assert seen == [(14, 4)]
        character_attrs = [attr for _, _, text, attr in screen.writes if text == "!"]
        assert character_attrs == [encoded_color | selection_emphasis]

    def test_cell16_title_reports_colors_encoded_in_memory(self):
        state = IpuState()
        state.xmem.write_address(
            0,
            struct.pack("<HH", 0 | (14 << 8) | (4 << 12), 1 | (2 << 8)),
        )
        session = DebugViewSession.with_tabs(
            initialized=True,
            xmem=[XmemTab(XmemRequest("byte", "0", "0", 2, "cell16"))],
        )
        view, screen, _ = make_view(
            state,
            screen=FakeScreen(columns=140),
            session=session,
        )

        view.render()

        assert f"cell16 {view.glyphs.separator} fg=2/e bg=0/4" in screen.text()

    def test_palette_allocates_all_16_cell_colors(self):
        screen = FakeScreen(colors=16)
        colors = draw.Palette(screen)

        styles = {
            colors.cell(foreground, background)
            for foreground in range(16)
            for background in range(16)
        }

        assert all(styles)
        # Every pair is its own colour; brightness adds bold on top of it.
        assert len({style & COLOR_MASK for style in styles}) == 256

    def test_changed_register_and_xmem_value_are_highlighted(self):
        state = IpuState()
        view, screen, session = make_view(state)
        session.commit_baseline(state)
        state.regfile.set_lr(0, 2)
        state.xmem.write_address(0, b"\x01")

        view.render()

        changed = view.colors.changed
        assert any(text == "2" and attr & changed for _, _, text, attr in screen.writes)
        assert any(text == "01" and attr & changed for _, _, text, attr in screen.writes)

    @pytest.mark.parametrize(
        "xmem_request",
        [
            XmemRequest("byte", "0", "0", 4, "hex"),
            XmemRequest("byte", "0", "0", 4, "int8"),
            XmemRequest("byte", "0", "0", 4, "u8"),
            XmemRequest("byte", "0", "0", 2, "cell16"),
            XmemRequest("byte", "0", "0", 2, "u32"),
            XmemRequest("byte", "0", "0", 2, "f32"),
        ],
    )
    def test_all_xmem_formats_render(self, xmem_request):
        state = IpuState()
        session = DebugViewSession.with_tabs(initialized=True, xmem=[XmemTab(xmem_request)])
        view, screen, _ = make_view(state, session=session)

        view.render()

        assert "Error:" not in screen.text()
        assert "0x00000000" in screen.text()


class TestPipelineRegisters:
    def test_every_logical_pipeline_register_is_selectable(self):
        view, screen, session = make_view(screen=FakeScreen(columns=140))

        for spec in PIPELINE_REGISTERS:
            session.pipeline.tabs[0].register_key = spec.key
            view.render()
            assert spec.label in screen.text()

    @pytest.mark.parametrize(
        ("pane", "second_tab"),
        [
            ("disassembly", DisassemblyTab(4)),
            ("registers", RegisterTab("lr")),
            ("xmem", XmemTab(XmemRequest("byte", "16", "0", 4, "hex"))),
            ("pipeline", PipelineTab("r1")),
        ],
    )
    def test_a_d_select_tabs_without_changing_focus(self, pane, second_tab):
        view, _, session = make_view()
        session.panes[pane].tabs.append(second_tab)
        session.focus = pane

        view.handle_key("d")
        assert session.focus == pane
        assert session.panes[pane].active == 1
        view.handle_key("a")
        assert session.focus == pane
        assert session.panes[pane].active == 0

    @pytest.mark.parametrize("pane", PANE_FOCUS_ORDER)
    def test_f2_and_f3_open_editor_for_focused_pane(self, pane):
        view, _, session = make_view()
        session.focus = pane

        view.handle_key(Key.F2)
        assert view.prompt.mode == "add"
        assert view.prompt.pane == pane
        view.handle_key(Key.ESCAPE)
        view.handle_key(Key.F3)

        assert view.prompt.mode == "edit"
        assert view.prompt.pane == pane
        assert session.focus == pane

    def test_add_edit_and_close_tabs_for_each_non_xmem_pane(self):
        view, _, session = make_view()
        cases = (
            ("disassembly", "12", "9", "target"),
            ("registers", "lr", "cr", "group"),
            ("pipeline", "R_ACC", "MEM_BYPASS", "register_key"),
        )
        for pane, added, edited, value_attribute in cases:
            session.focus = pane
            view.handle_key(Key.F2)
            view.prompt.text = added
            view.handle_key(Key.ENTER)
            tabs = session.panes[pane]
            assert len(tabs) == 2
            view.handle_key(Key.F3)
            view.prompt.text = edited
            view.handle_key(Key.ENTER)
            assert getattr(tabs[-1], value_attribute) in (9, "cr", "mem_bypass")
            view.handle_key(Key.F4)
            view.handle_key(Key.F4)
            assert len(tabs) == 1

    def test_invalid_pane_editor_input_does_not_change_tabs(self):
        view, _, session = make_view()
        session.focus = "pipeline"
        view.handle_key(Key.F2)
        view.prompt.text = "NOT_A_REGISTER"
        view.handle_key(Key.ENTER)

        assert view.prompt.mode == "add"
        assert len(session.pipeline.tabs) == 1
        assert session.message_is_error

    @pytest.mark.parametrize(
        ("group", "present", "absent"),
        [("lr", "L00 ", "C00 "), ("cr", "C00 ", "L00 ")],
    )
    def test_register_tab_filters_rendered_group(self, group, present, absent):
        session = DebugViewSession.with_tabs(
            initialized=True,
            registers=[RegisterTab(group)],
        )
        view, screen, _ = make_view(session=session)

        view.render()

        assert present in [text for _, _, text, _ in screen.writes]
        assert absent not in [text for _, _, text, _ in screen.writes]

    @pytest.mark.parametrize("pane", PANE_FOCUS_ORDER)
    def test_tab_selects_next_tab_in_every_pane(self, pane):
        view, _, session = make_view()
        second_tabs = {
            "disassembly": DisassemblyTab(1),
            "registers": RegisterTab("lr"),
            "xmem": XmemTab(XmemRequest("byte", "4", "0", 4, "hex")),
            "pipeline": PipelineTab("r1"),
        }
        session.panes[pane].tabs.append(second_tabs[pane])
        session.focus = pane

        view.handle_key(Key.TAB)

        assert session.panes[pane].active == 1
        assert session.focus == pane

    def test_scalar_register_format_cycles_and_renders(self):
        state = IpuState()
        state.regfile.set_lr(0, 0xFFFFFFFF)
        view, screen, session = make_view(state)
        session.focus = "registers"

        view.handle_key("f")
        view.render()
        assert session.registers.tabs[0].fmt == "u32"
        assert "[all u32 x]" in screen.text()
        assert "4294967295" in screen.text()

        view.handle_key("f")
        view.render()
        assert session.registers.tabs[0].fmt == "int32"
        assert "-1" in screen.text()

        view.handle_key("f")
        view.handle_key("f")
        view.render()
        assert session.registers.tabs[0].fmt == "bits"
        assert "1" * 32 in screen.text()

    def test_register_format_is_per_tab(self):
        view, _, session = make_view()
        session.registers.tabs.append(RegisterTab("lr"))
        session.registers.active = 1
        session.focus = "registers"

        view.handle_key("f")

        assert session.registers.tabs[0].fmt == "hex"
        assert session.registers.tabs[1].fmt == "u32"

    def test_unsigned_register_formats_use_xmem_zero_padding(self):
        state = IpuState()
        state.regfile.set_lr(0, 1)
        state.regfile.mutable("r")[:4] = struct.pack("<I", 1)
        session = DebugViewSession(initialized=True)
        session.ensure_default_tab(state)
        session.registers.tabs[0].fmt = "u32"
        session.pipeline.tabs[0].fmt = "u8"
        view, screen, _ = make_view(
            state,
            screen=FakeScreen(columns=140),
            session=session,
        )

        view.render()

        assert "0000000001" in row_text(screen, view.rects["registers"].y + 3)
        assert any(text == "001" for _, _, text, _ in screen.writes)

        screen.clear()
        session.pipeline.tabs[0].fmt = "u32"
        view.render()

        assert "0000000001" in row_text(screen, view.rects["pipeline"].y + 2)

    def test_leading_zeros_of_a_wide_value_are_dimmed(self):
        state = IpuState()
        state.regfile.set_lr(1, 0x12)
        view, screen, _ = make_view(state, screen=FakeScreen(columns=140))

        view.render()

        assert any(
            text == "000000" and attr & view.colors.border for _, _, text, attr in screen.writes
        )
        assert any(text == "12" and not attr for _, _, text, attr in screen.writes)

    def test_pipeline_format_is_per_tab(self):
        view, _, session = make_view()
        session.pipeline.tabs.append(PipelineTab("r0"))
        session.pipeline.active = 1
        session.focus = "pipeline"

        view.handle_key("f")

        assert session.pipeline.tabs[0].fmt is None
        assert session.pipeline.tabs[1].fmt == "int8"

    def test_xmem_format_cycle_preserves_byte_range_and_skips_alignment(self):
        view, _, session = make_view()
        session.xmem.tabs[0] = XmemTab(XmemRequest("byte", "1", "0", 3, "hex"))
        session.focus = "xmem"

        view.handle_key("f")
        assert session.focus == "xmem"
        assert session.xmem.tabs[0].request.fmt == "int8"
        assert session.xmem.tabs[0].request.count == 3
        for _ in range(2):
            view.handle_key("f")
        assert session.xmem.tabs[0].request.fmt == "hex"

    def test_xmem_format_cycle_preserves_aligned_resolved_byte_range(self):
        state = IpuState()
        state.regfile.set_cr(3, 4)
        session = DebugViewSession.with_tabs(
            initialized=True,
            xmem=[XmemTab(XmemRequest("byte", "cr3", "0", 2, "u32"))],
        )
        view, _, _ = make_view(state, session=session)
        session.focus = "xmem"
        before = DebugSession(state).read_xmem(session.xmem.tabs[0].request)

        view.handle_key("f")

        after = DebugSession(state).read_xmem(session.xmem.tabs[0].request)
        assert (after.byte_address, after.byte_count) == (
            before.byte_address,
            before.byte_count,
        )

    def test_pipeline_arrows_move_value_cursor(self):
        view, _, session = make_view()
        view.render()
        session.focus = "pipeline"
        expected = value_grid(view.debug, session.pipeline_tab(), view.rects["pipeline"]).per_line

        view.handle_key(Key.DOWN)

        assert session.pipeline.tabs[0].cursor == expected
        assert session.xmem.tabs[0].scroll == 0

    def test_scalar_register_arrows_move_cursor(self):
        view, _, session = make_view()
        session.focus = "registers"

        view.render()
        grid_rows, _ = registers_pane.grid_shape(session.registers.tabs[0], view.rects["registers"])

        view.handle_key(Key.RIGHT)
        view.handle_key(Key.DOWN)

        assert session.registers.tabs[0].cursor == grid_rows + 1

    def test_disassembly_focus_owns_keyboard_scrolling(self):
        state = IpuState()
        state.program_counter = 10
        view, _, session = make_view(state)
        view.render()
        session.focus = "disassembly"

        view.handle_key(Key.DOWN)

        assert session.disassembly.tabs[0].cursor_offset == 1
        assert session.xmem.tabs[0].scroll == 0


def mouse(view, y, x, **flags):
    """Deliver one pointer event at ``(y, x)`` to *view*: a click by default,
    or the given ``MouseEvent`` flags (``motion=True`` for no button at all)."""
    if not flags:
        flags = {"click": True}
    flags.pop("motion", None)
    return view.handle_key(MouseEvent(y, x, **flags))


def cursor_payload(view, session, pane):
    """What the pane's cursor currently points at, in ``value_hits`` terms."""
    if pane == "disassembly":
        tab = session.disassembly_tab()
        base = view.state.program_counter if tab.target is None else tab.target
        return base + tab.cursor_offset
    if pane == "registers":
        return session.register_tab().cursor
    if pane == "pipeline":
        return session.pipeline_tab().cursor
    tab = session.xmem.current()
    return (tab.cursor_line, tab.cursor_column)


def cursor_attrs(view, screen, session, pane):
    """The attributes painted over the pane's cursor cell."""
    payload = cursor_payload(view, session, pane)
    rect = next(rect for rect, drawn in view.hits.values[pane] if drawn == payload)
    return [
        attr
        for y, x, text, attr in screen.writes
        if y == rect.y and rect.x <= x < rect.x + rect.width and text.strip()
    ]


class TestCursorVisibility:
    @pytest.mark.parametrize("pane", PANE_FOCUS_ORDER)
    def test_every_pane_shows_its_cursor_whether_focused_or_not(self, pane):
        view, screen, session = make_view(screen=FakeScreen(columns=140))
        session.focus = pane
        view.render()

        assert any(
            attr & view.colors.selected == view.colors.selected
            for attr in cursor_attrs(view, screen, session, pane)
        )

        session.focus = "registers" if pane != "registers" else "xmem"
        screen.clear()
        view.render()

        assert any(
            attr & view.colors.cursor == view.colors.cursor
            for attr in cursor_attrs(view, screen, session, pane)
        )

    def test_the_focused_and_idle_cursors_are_distinguishable(self):
        view, _, _ = make_view()

        assert view.colors.cursor
        assert view.colors.selected
        assert view.colors.cursor != view.colors.selected


class TestRenderCost:
    def test_an_xmem_request_is_tokenised_once_per_curses_entry(self, monkeypatch):
        tokenized: list[int] = []
        real = debug_format.tokenize_xmem

        def tokenize(data, address, fmt, row_size):
            tokenized.append(address)
            return real(data, address, fmt, row_size)

        monkeypatch.setattr(debug_format, "tokenize_xmem", tokenize)
        view, _, session = make_view()
        session.focus = "xmem"

        view.render()
        for _ in range(5):
            view.render()
            view.handle_key(Key.DOWN)

        assert len(tokenized) == 1

    def test_bare_pointer_motion_does_not_ask_for_a_redraw(self):
        view, _, _ = make_view()
        view.render()

        mouse(view, view.rects["xmem"].y + 3, view.rects["xmem"].x + 4, motion=True)
        assert view.redraw is False

        view.handle_key(Key.DOWN)
        assert view.redraw is True

    def test_queued_input_is_handled_before_the_next_frame(self):
        screen = FakeScreen(
            [
                Key.DOWN,
                Key.DOWN,
                Key.F8,
            ]
        )
        view, _, session = make_view(screen=screen)
        session.focus = "pipeline"
        renders: list[int] = []
        original = view.render

        def counted():
            renders.append(1)
            original()

        view.render = counted

        assert view.run() == DebugAction.STEP
        # Three keys arrived together, so they cost one frame, not three.
        assert len(renders) == 1
        assert not screen.keys


class TestMousePlacesTheCursor:
    def test_clicking_an_xmem_value_moves_the_cursor_to_it(self):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        session.xmem.tabs[0] = XmemTab(XmemRequest("byte", "0", "0", 512, "hex"))
        view.render()
        rect, (line, column) = next(
            (rect, payload) for rect, payload in view.hits.values["xmem"] if payload[1] == 3
        )

        mouse(view, rect.y, rect.x + rect.width - 1)

        assert session.focus == "xmem"
        assert (session.xmem.tabs[0].cursor_line, session.xmem.tabs[0].cursor_column) == (
            line,
            column,
        )

    def test_clicking_a_register_moves_the_cursor_to_it(self):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        view.render()
        rect, index = view.hits.values["registers"][-1]

        mouse(view, rect.y, rect.x + 1)

        assert session.focus == "registers"
        assert session.registers.tabs[0].cursor == index

    def test_clicking_an_instruction_moves_the_disassembly_cursor(self):
        state = IpuState()
        state.program_counter = 20
        view, _, session = make_view(state)
        view.render()
        rect, pc = view.hits.values["disassembly"][-1]

        mouse(view, rect.y, rect.x + 2)

        assert session.focus == "disassembly"
        assert session.disassembly.tabs[0].cursor_offset == pc - 20

    def test_clicking_a_pipeline_value_moves_the_cursor_to_it(self):
        view, _, session = make_view()
        view.render()
        rect, index = view.hits.values["pipeline"][5]

        mouse(view, rect.y, rect.x)

        assert session.focus == "pipeline"
        assert session.pipeline.tabs[0].cursor == index

    def test_wheel_scrolls_by_several_lines(self):
        view, _, session = make_view()
        view.render()
        rect = view.rects["pipeline"]
        per_line = value_grid(view.debug, session.pipeline_tab(), rect).per_line

        mouse(view, rect.y + 3, rect.x + 2, wheel=1)

        assert session.pipeline.tabs[0].cursor == (pointer.MOUSE_WHEEL_LINES * per_line)

    def test_clicking_the_scrollbar_jumps_through_the_content(self):
        view, _, session = make_view()
        view.render()
        track = view.hits.scrollbars["pipeline"]

        mouse(view, track.y + track.height - 1, track.x)

        assert session.focus == "pipeline"
        assert session.pipeline.tabs[0].cursor > 0


class TestMouseResizesSplits:
    @pytest.mark.parametrize(
        ("split", "pane"),
        [
            ("register_pane_columns", "registers"),
            ("pipeline_pane_columns", "disassembly"),
        ],
    )
    def test_dragging_a_vertical_border_moves_that_split(self, split, pane):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        view.render()
        rect = view.rects[pane]
        border_y = rect.y + 2

        mouse(view, border_y, rect.x, press=True)
        assert view.pointer.drag is not None
        assert getattr(session, split) is None

        mouse(view, border_y, rect.x - 6, motion=True)
        view.render()

        assert view.rects[pane].width == rect.width + 6
        assert getattr(session, split) == (
            rect.x - 5 if split == "pipeline_pane_columns" else rect.width + 6
        )

        mouse(view, border_y, rect.x - 6, release=True)
        assert view.pointer.drag is None

    def test_dragging_the_shared_row_moves_the_top_split(self):
        view, _, session = make_view(screen=FakeScreen(rows=40, columns=140))
        view.render()
        border_y = view.rects["xmem"].y

        mouse(view, border_y, 20, press=True)
        mouse(view, border_y - 3, 20, motion=True)
        view.render()

        assert view.rects["xmem"].y == border_y - 3
        assert session.top_pane_rows == view.rects["disassembly"].height

    def test_a_press_and_release_without_motion_is_a_click(self):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        session.focus = "disassembly"
        view.render()
        border_y = view.rects["xmem"].y

        mouse(view, border_y, 4, press=True)
        mouse(view, border_y, 4, release=True)

        assert view.pointer.drag is None
        assert session.top_pane_rows is None
        assert session.focus == "pipeline"

    def test_a_keystroke_abandons_an_unfinished_drag(self):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        view.render()

        mouse(view, view.rects["xmem"].y, 10, press=True)
        assert view.pointer.drag is not None

        view.handle_key(Key.DOWN)

        assert view.pointer.drag is None
        assert session.top_pane_rows is None

    def test_pointer_motion_alone_changes_nothing(self):
        view, _, session = make_view()
        view.render()
        before = session.focus

        mouse(view, view.rects["pipeline"].y + 3, view.rects["pipeline"].x + 4, motion=True)

        assert view.pointer.drag is None
        assert session.focus == before


class TestMouseAndTerminalFallbacks:
    def test_mouse_selects_pipeline_tab(self):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        session.pipeline.tabs.append(PipelineTab("r1"))
        view.render()
        rect = next(rect for rect, index in view.hits.tabs["pipeline"] if index == 1)
        view.handle_key(MouseEvent(rect.y, rect.x, click=True))

        assert session.pipeline.active == 1
        assert session.pipeline.tabs[1].register_key == "r1"
        assert session.focus == "pipeline"

    def test_mouse_click_focuses_pipeline_pane(self):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        view.render()
        x = view.rects["pipeline"].x + 2
        y = view.rects["pipeline"].y + 3
        view.handle_key(MouseEvent(y, x, click=True))

        assert session.focus == "pipeline"

    @pytest.mark.parametrize("pane", PANE_FOCUS_ORDER)
    def test_mouse_click_on_add_control_opens_focused_editor(self, pane):
        view, _, _ = make_view()
        view.render()
        rect = view.hits.adds[pane]
        view.handle_key(MouseEvent(rect.y, rect.x, click=True))

        assert view.prompt.mode == "add"
        assert view.prompt.pane == pane

    def test_mouse_wheel_moves_cursor_in_pointed_pane(self):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        view.render()
        x = view.rects["pipeline"].x + 2
        y = view.rects["pipeline"].y + 3
        view.handle_key(MouseEvent(y, x, wheel=1))

        assert session.focus == "pipeline"
        assert session.pipeline.tabs[0].cursor > 0

    def test_run_debug_tui_uses_the_terminal_and_returns_action(self, monkeypatch):
        state = IpuState()
        monkeypatch.setattr(debug_tui, "run_in_terminal", lambda body: body(FakeScreen([Key.F5])))

        session = new_debug_view_session(state)
        assert run_debug_tui(state, 3, session=session) == DebugAction.CONTINUE
        assert session.active

    def test_curses_initialization_failure_is_reported(self, monkeypatch):
        state = IpuState()

        def in_terminal(body):
            raise OSError("no terminal")

        monkeypatch.setattr(debug_tui, "run_in_terminal", in_terminal)

        session = new_debug_view_session(state)
        with pytest.raises(RuntimeError, match="curses TUI failed"):
            run_debug_tui(state, 0, session=session)
        assert not session.active

    def test_wrapper_restores_terminal_when_view_raises(self, monkeypatch):
        state = IpuState()
        restored = False

        class BrokenScreen(FakeScreen):
            def read_key(self):
                raise ValueError("render loop failed")

        def in_terminal(body):
            nonlocal restored
            try:
                return body(BrokenScreen())
            finally:
                restored = True

        monkeypatch.setattr(debug_tui, "run_in_terminal", in_terminal)

        session = new_debug_view_session(state)
        with pytest.raises(RuntimeError, match="render loop failed"):
            run_debug_tui(state, 0, session=session)
        assert restored
        assert not session.active


def footer_row(screen: FakeScreen) -> str:
    return row_text(screen, screen.rows - 1)


class TestFooterFitting:
    @pytest.mark.parametrize("pane", PANE_FOCUS_ORDER)
    def test_narrow_footer_keeps_whole_labels_quit_and_help(self, pane):
        view, screen, session = make_view()
        session.focus = pane

        view.render()

        footer = footer_row(screen)
        assert len(footer) <= MIN_TERMINAL_COLUMNS
        assert footer.rstrip().endswith("q Quit")
        assert "?:Help" not in footer
        assert "F5:Run F8:Step" in footer
        assert "F6" not in footer

    @pytest.mark.parametrize("pane", PANE_FOCUS_ORDER)
    def test_footer_labels_come_from_the_binding_table(self, pane):
        view, screen, session = make_view(screen=FakeScreen(columns=220))
        session.focus = pane

        view.render()

        footer = footer_row(screen)
        for binding in KEY_BINDINGS:
            if binding.footer is None or binding.footer_rank > 3:
                continue
            if pane not in binding.panes:
                assert binding.footer not in footer
                continue
            if binding.scope in ("global", "pane"):
                assert binding.footer in footer

    def test_footer_never_splits_the_tab_selection_labels(self):
        view, screen, session = make_view()
        session.focus = "disassembly"

        view.render()

        footer = footer_row(screen)
        assert ("a:Prev" in footer) == ("d:Next" in footer)


@pytest.mark.parametrize("pane", ["registers", "xmem", "pipeline"])
def test_zero_values_keep_the_selection_color_pair(pane):
    view, screen, session = make_view()
    session.focus = pane
    mask = COLOR_MASK
    pair_unit = mask & -mask
    view.colors.border = 2 * pair_unit
    view.colors.selected = 5 * pair_unit

    view.render()

    rect, _ = view.hits.values[pane][0]
    attributes = [
        attr & mask
        for y, x, text, attr in screen.writes
        if y == rect.y and rect.x <= x < rect.x + rect.width and text.strip()
    ]
    # OR-ing pair 2 with pair 5 would accidentally select pair 7. It can be
    # uninitialized (black on black), even though both input pairs are valid.
    assert attributes and set(attributes) == {5 * pair_unit}


class TestJumpKeys:
    def test_home_and_end_jump_to_the_first_and_last_value(self):
        view, _, session = make_view()
        session.focus = "disassembly"
        view.render()

        view.handle_key(Key.END)
        assert session.disassembly.tabs[0].cursor_offset == (INST_MEM_SIZE - 1)
        view.handle_key(Key.HOME)
        assert session.disassembly.tabs[0].cursor_offset == 0

    def test_end_selects_the_last_scalar_register(self):
        view, _, session = make_view()
        session.focus = "registers"
        view.render()

        view.handle_key(Key.END)

        assert session.registers.tabs[0].cursor == (
            len(registers_pane.REGISTER_GROUPS) * registers_pane.REGISTERS_PER_GROUP - 1
        )

    def test_xmem_home_resets_both_cursor_axes(self):
        view, _, session = make_view()
        session.xmem.tabs[0] = XmemTab(XmemRequest("byte", "0", "0", 512, "hex"))
        session.focus = "xmem"
        view.render()
        tab = session.xmem.tabs[0]
        first_line = tab.cursor_line

        view.handle_key(Key.END)
        assert (tab.cursor_line, tab.cursor_column) != (first_line, 0)

        view.handle_key(Key.HOME)
        # Home returns to the first line that actually holds values, which is
        # not row 0 when the dump opens with a marker line.
        assert tab.cursor_line == first_line
        assert tab.cursor_column == 0


class TestHelpOverlay:
    def test_question_mark_opens_a_scrollable_overlay(self):
        view, screen, _ = make_view()

        view.handle_key("?")
        assert view.help.visible
        view.render()
        assert "Debugger controls" in screen.text()
        assert "Halt execution and exit the debugger" not in screen.text()

        view.handle_key(Key.PAGE_DOWN)
        assert view.help.scroll > 0

        view.handle_key(Key.ESCAPE)
        assert not view.help.visible
        assert view.help.scroll == 0

    def test_overlay_lists_every_documented_binding(self):
        view, screen, _ = make_view(screen=FakeScreen(rows=60, columns=100))
        view.handle_key("?")

        view.render()

        text = screen.text()
        for binding in KEY_BINDINGS:
            if binding.keys != ("q",):
                assert binding.description in text

    def test_overlay_lines_fit_the_minimum_terminal(self):
        content_width = min(MIN_TERMINAL_COLUMNS - 4, 78) - 4

        lines = help_lines()

        assert lines
        assert all(len(text) <= content_width for text, _ in lines)

    def test_overlay_swallows_execution_keys(self):
        view, _, session = make_view()
        view.handle_key("?")

        assert view.handle_key(Key.F5) is None
        assert not session.active

        view.handle_key(Key.ESCAPE)
        assert view.handle_key(Key.F5) is DebugAction.CONTINUE

    def test_question_mark_inside_an_editor_is_typed_text(self):
        view, _, _ = make_view()
        view.handle_key(Key.F2)

        view.handle_key("?")

        assert not view.help.visible
        assert view.prompt.text.endswith("?")

    def test_escape_closes_the_overlay_before_the_editor(self):
        view, _, _ = make_view()
        view.handle_key(Key.F2)
        command = view.prompt.text
        view.help.visible = True

        view.handle_key(Key.ESCAPE)

        assert not view.help.visible
        assert view.prompt.mode == "add"
        assert view.prompt.text == command


class TestScrollPosition:
    def test_pipeline_reports_the_visible_item_range(self):
        view, screen, session = make_view(screen=FakeScreen(columns=180))
        session.focus = "pipeline"

        view.render()

        assert "items 1-" in screen.text()

        view.handle_key(Key.END)
        view.render()
        spec = pipeline_spec(session.pipeline.tabs[0].register_key)
        total = len(read_register(view.debug, spec))
        assert f"-{total}/{total} " in screen.text()

    @staticmethod
    def thumb_rows(view, screen, pane: str = "pipeline") -> list[int]:
        rect = view.rects[pane]
        edge = rect.x + rect.width - 1 - draw.SCROLLBAR_COLUMNS
        return [y for y, x, text, _ in screen.writes if x == edge and text == view.glyphs.thumb]

    def test_scroll_thumb_tracks_the_cursor(self):
        view, screen, session = make_view()
        session.focus = "pipeline"
        view.render()
        top = self.thumb_rows(view, screen)
        assert top

        view.handle_key(Key.END)
        view.render()
        bottom = self.thumb_rows(view, screen)
        assert bottom
        assert min(bottom) > min(top)

    def test_thumb_stays_clear_of_the_tab_bar_and_borders(self):
        view, screen, session = make_view()
        session.focus = "pipeline"

        view.render()

        rect = view.rects["pipeline"]
        rows = self.thumb_rows(view, screen)
        assert min(rows) >= rect.y + 2
        assert max(rows) <= rect.y + rect.height - 2

    def test_no_thumb_when_everything_fits(self):
        view, screen, session = make_view(screen=FakeScreen(rows=60, columns=140))
        session.focus = "pipeline"
        session.pipeline.tabs[0].fmt = "bits"
        session.top_pane_rows = 45

        view.render()

        assert not self.thumb_rows(view, screen)

    @pytest.mark.parametrize("pane", ["disassembly", "xmem", "pipeline"])
    def test_every_scrollable_pane_draws_an_inset_thumb(self, pane):
        view, screen, session = make_view()
        session.xmem.tabs[0] = XmemTab(XmemRequest("byte", "0", "0", 512, "hex"))

        view.render()

        rect = view.rects[pane]
        rows = self.thumb_rows(view, screen, pane)
        assert rows
        assert min(rows) >= rect.y + 2
        assert max(rows) <= rect.y + rect.height - 2
        # The thumb sits inside the pane, so a shared border stays unbroken.
        for row in rows:
            assert (
                cell(screen, row, rect.x + rect.width - 1)
                == (view.glyphs.box[chrome.BORDER_UP | chrome.BORDER_DOWN])
            )

    def test_xmem_reports_its_line_range(self):
        view, screen, session = make_view()
        session.xmem.tabs[0] = XmemTab(XmemRequest("byte", "0", "0", 512, "hex"))

        view.render()

        assert "lines 1-" in screen.text()


class TestResizableSplits:
    def test_shift_arrows_resize_and_persist_in_the_session(self):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        session.focus = "registers"
        view.render()
        before = view.rects["registers"].width

        view.handle_key(Key.SHIFT_LEFT)
        view.render()

        assert view.rects["registers"].width == before + tui_layout.PANE_RESIZE_STEP
        assert session.register_pane_columns == before + (tui_layout.PANE_RESIZE_STEP)

        view.handle_key(Key.SHIFT_DOWN)
        view.render()
        assert view.rects["disassembly"].height == tui_layout.TOP_PANE_ROWS + 1
        assert session.top_pane_rows == tui_layout.TOP_PANE_ROWS + 1

    @pytest.mark.parametrize(
        ("pane", "resized", "untouched"),
        [
            ("disassembly", "pipeline", "registers"),
            ("registers", "registers", "pipeline"),
            ("xmem", "registers", "pipeline"),
            ("pipeline", "pipeline", "registers"),
        ],
    )
    def test_shift_arrows_resize_the_focused_row_only(self, pane, resized, untouched):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        session.focus = pane
        view.render()
        before = view.rects[resized].width
        other = view.rects[untouched].width

        view.handle_key(Key.SHIFT_LEFT)
        view.render()

        assert view.rects[resized].width == (
            before + (1 if resized == "registers" else -1) * tui_layout.PANE_RESIZE_STEP
        )
        assert view.rects[untouched].width == other

    def test_resizing_preserves_the_shared_border_columns(self):
        view, _, _ = make_view(screen=FakeScreen(columns=140))
        for _ in range(6):
            view.handle_key(Key.SHIFT_LEFT)
        view.handle_key(Key.SHIFT_DOWN)

        layout = view.layout()

        assert layout is not None
        assert layout["pipeline"].x + layout["pipeline"].width - 1 == (layout["disassembly"].x)
        assert layout["disassembly"].y + layout["disassembly"].height - 1 == (layout["xmem"].y)
        assert layout["xmem"].x + layout["xmem"].width - 1 == (layout["registers"].x)

    def test_splits_are_clamped_to_their_minimums(self):
        view, _, session = make_view()
        session.focus = "registers"
        for _ in range(50):
            view.handle_key(Key.SHIFT_LEFT)
        layout = view.layout()

        assert layout is not None
        assert session.register_pane_columns is not None
        assert layout["disassembly"].width - 1 >= (tui_layout.MIN_DISASSEMBLY_COLUMNS)

        for _ in range(50):
            view.handle_key(Key.SHIFT_RIGHT)
        layout = view.layout()

        assert layout is not None
        assert layout["registers"].width >= tui_layout.MIN_REGISTER_COLUMNS

    def test_bottom_split_is_clamped_to_its_minimums(self):
        view, _, session = make_view()
        session.focus = "xmem"
        for _ in range(50):
            view.handle_key(Key.SHIFT_LEFT)
        layout = view.layout()

        assert layout is not None
        assert layout["xmem"].width - 1 >= tui_layout.MIN_XMEM_COLUMNS

        for _ in range(50):
            view.handle_key(Key.SHIFT_RIGHT)
        layout = view.layout()

        assert layout is not None
        assert layout["pipeline"].width >= tui_layout.MIN_PIPELINE_COLUMNS

    def test_top_pane_shrinks_and_the_register_grid_reflows(self):
        view, _, session = make_view(screen=FakeScreen(rows=40, columns=140))
        session.top_pane_rows = 30
        view.render()
        tall_rows, tall_columns = registers_pane.grid_shape(
            session.registers.tabs[0], view.rects["registers"]
        )

        for _ in range(40):
            view.handle_key(Key.SHIFT_UP)
        view.render()
        short_rows, short_columns = registers_pane.grid_shape(
            session.registers.tabs[0], view.rects["registers"]
        )

        assert view.rects["disassembly"].height == tui_layout.MIN_TOP_PANE_ROWS
        assert short_rows > tall_rows
        assert short_columns < tall_columns
        assert short_rows * short_columns == (
            len(registers_pane.REGISTER_GROUPS) * registers_pane.REGISTERS_PER_GROUP
        )

    def test_top_height_keeps_instruction_width_independent_of_register_grid(self):
        view, screen, session = make_view(screen=FakeScreen(rows=40, columns=140))
        for _ in range(40):
            view.handle_key(Key.SHIFT_UP)
        view.render()
        narrow = view.rects["disassembly"].width

        session.top_pane_rows = None
        view.render()

        assert view.rects["disassembly"].width == narrow
        assert "LR 00-15" in screen.text()
        assert "CR 00-15" in screen.text()

    def test_the_default_top_pane_fits_the_tallest_register_grid(self):
        tall, _, session = make_view(screen=FakeScreen(rows=40, columns=140))
        tall.render()

        assert registers_pane.grid_shape(session.registers.tabs[0], tall.rects["registers"]) == (
            max(registers_pane.REGISTER_GRID_ROWS),
            2,
        )

        short, _, short_session = make_view()
        short.render()

        assert short.rects["disassembly"].height == tui_layout.TOP_PANE_ROWS
        assert registers_pane.grid_shape(
            short_session.registers.tabs[0], short.rects["registers"]
        ) == (
            registers_pane.DEFAULT_REGISTER_ROWS,
            4,
        )

    def test_growing_the_top_pane_leaves_room_for_memory(self):
        view, _, _ = make_view()

        for _ in range(20):
            view.handle_key(Key.SHIFT_DOWN)
        layout = view.layout()

        assert layout is not None
        assert layout["xmem"].height >= tui_layout.MIN_MEMORY_ROWS

    def test_equals_resets_both_splits(self):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        view.handle_key(Key.SHIFT_LEFT)
        view.handle_key(Key.SHIFT_DOWN)

        view.handle_key("=")
        view.render()

        assert session.register_pane_columns is None
        assert session.top_pane_rows is None
        assert view.rects["disassembly"].height == tui_layout.TOP_PANE_ROWS


class TestDebuggerImprovements:
    def test_queued_maximize_focus_and_navigation_use_current_geometry(self):
        view, _, session = make_view()
        view.render()
        session.maximized = not session.maximized
        view.handle_key("1")
        view.handle_key(Key.DOWN)
        expected = value_grid(view.debug, session.pipeline_tab(), view.rects["pipeline"]).per_line
        assert view.rects["pipeline"].width == 80
        assert session.pipeline.tabs[0].cursor == expected

    def test_click_after_tab_close_uses_updated_targets(self):
        view, _, session = make_view()
        session.xmem.tabs.append(XmemTab(XmemRequest("byte", "128", "0", 1, "hex")))
        session.xmem.active = 1
        view.render()
        rect, _ = view.hits.tabs["xmem"][-1]
        view.handle_key(Key.F4)
        mouse(view, rect.y, rect.x)
        assert len(session.xmem.tabs) == 1
        assert session.xmem.active == 0

    def test_idle_mouse_preserves_pending_keyboard_redraw(self):
        view, screen, session = make_view()
        session.focus = "pipeline"
        view.render()
        view.handle_key(Key.DOWN)
        mouse(view, 5, 5, motion=True)
        assert view.redraw
        view.render()
        assert not view.redraw
        assert session.pipeline.tabs[0].cursor > 0

    def test_queued_step_does_not_consume_following_input(self):
        screen = FakeScreen(["4", Key.F8, "q"])
        view, _, _ = make_view(screen=screen)
        assert view.run() == DebugAction.STEP
        assert screen.keys == ["q"]

    def test_queued_mouse_event_is_handled_before_the_next_key(self):
        screen = FakeScreen(["4", MouseEvent(0, 0), Key.F8])
        view, _, _ = make_view(screen=screen)
        assert view.run() == DebugAction.STEP
        assert screen.keys == []

    def test_stationary_motion_keeps_drag_until_release(self):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        view.render()
        y, x = view.rects["disassembly"].y + 3, view.rects["disassembly"].x
        mouse(view, y, x, press=True)
        mouse(view, y, x, motion=True)
        assert view.pointer.drag is not None
        mouse(view, y, x - 4, motion=True)
        assert session.pipeline_pane_columns == x - 3
        mouse(view, y, x - 4, release=True)
        assert view.pointer.drag is None

    def test_small_terminal_discards_old_mouse_targets(self):
        view, screen, session = make_view()
        view.render()
        target, _ = view.hits.tab_closes["xmem"][0]
        screen.rows = 10
        screen.columns = 30
        view.render()
        assert not view.hits.tab_closes
        assert not view.hits.values
        assert not view.hits.scrollbars
        assert view.rects["xmem"].height == 0
        mouse(view, target.y, target.x)
        assert len(session.xmem.tabs) == 1
        screen.rows, screen.columns = 24, 80
        view.render()
        assert view.hits.values

    def test_pipeline_bits_are_clipped_and_maximize_reveals_full_value(self):
        state = IpuState()
        state.regfile.mutable("r")[:4] = b"\x01\x00\x00\x80"
        view, screen, session = make_view(state)
        session.focus = "pipeline"
        session.pipeline.tabs[0].fmt = "bits"
        view.render()
        rect = view.rects["pipeline"]
        limit = rect.x + rect.width - 1 - draw.SCROLLBAR_COLUMNS
        hits = view.hits.values["pipeline"]
        assert all(hit.x + hit.width <= limit for hit, _ in hits)
        y = hits[0][0].y
        assert view.glyphs.ellipsis in row_text(screen, y)
        assert cell(screen, y, rect.x + rect.width - 1) in view.glyphs.box
        for write_y, x, text, _ in screen.writes:
            if write_y == y and hits[0][0].x <= x < limit:
                assert x + len(text) <= limit
        session.maximized = not session.maximized
        view.render()
        assert "10000000000000000000000000000001" in screen.text()

    def test_maximize_switches_panes_and_restores_splits(self):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        session.pipeline_pane_columns = 60
        session.top_pane_rows = 12
        original = view.layout()
        session.maximized = not session.maximized
        view.render()
        assert set(view.hits.values) == {"xmem"}
        view.handle_key("4")
        view.render()
        assert set(view.hits.values) == {"registers"}
        assert view.rects["registers"].width == 140
        session.maximized = not session.maximized
        assert view.layout() == original

    def test_breakpoint_marker_and_run_to_cursor(self):
        view, screen, session = make_view()
        session.focus = "disassembly"
        view.render()
        view.handle_key(Key.F9)
        view.render()
        assert f"{view.glyphs.current_row}B0000" in screen.text()
        assert view.control.breakpoints == {0}
        assert view.handle_key(Key.F10) is None
        assert view.control.run_target is None
        view.handle_key(Key.DOWN)
        assert view.handle_key(Key.F10) == DebugAction.CONTINUE
        assert view.control.run_target == 1

    def test_scalar_edit_refreshes_symbolic_xmem_without_rereads_between_frames(self, monkeypatch):
        state = IpuState()
        state.xmem.write_address(0, b"\x11")
        state.xmem.write_address(128, b"\x22")
        view, _, session = make_view(state)
        request = XmemRequest("row", "0", "lr0", 1, "hex")
        session.xmem.tabs = [XmemTab(request)]
        session.focus = "registers"
        reads = []
        original = state.xmem.snapshot_bytes

        def read(address, count):
            reads.append((address, count))
            return original(address, count)

        monkeypatch.setattr(state.xmem, "snapshot_bytes", read)
        for _ in range(3):
            view.render()
        assert reads == [(0, 1)]
        view.handle_key("e")
        view.prompt.text = "0x1"
        view.handle_key(Key.ENTER)
        assert view.prompt.mode is None
        view.render()
        assert reads == [(0, 1), (128, 1)]
        resolved, error = view.panes["xmem"].resolve(view.debug, request)
        assert error is None
        assert resolved.data == b"\x22"

    def test_cr_edit_moves_the_session_generation_and_refreshes_xmem(self):
        state = IpuState()
        state.xmem.write_address(0, b"\x11")
        state.xmem.write_address(128, b"\x22")
        view, _, session = make_view(state)
        request = XmemRequest("row", "0", "cr2", 1, "hex")
        session.xmem.tabs = [XmemTab(request)]
        view.render()
        assert view.panes["xmem"].resolve(view.debug, request)[0].data == b"\x11"
        generation = view.debug.generation
        session.focus = "registers"
        session.registers.tabs[0].group = "cr"
        session.registers.tabs[0].cursor = 2
        view.handle_key("e")
        view.prompt.text = "1"
        view.handle_key(Key.ENTER)
        assert view.prompt.mode is None
        assert view.debug.generation == generation + 1
        assert view.panes["xmem"].resolve(view.debug, request)[0].data == b"\x22"

    @pytest.mark.parametrize("index", [0, 1])
    def test_scalar_editor_rejects_constants_and_invalid_input(self, index):
        view, _, session = make_view()
        session.focus = "registers"
        session.registers.tabs[0].group = "cr"
        session.registers.tabs[0].cursor = index
        view.handle_key("e")
        view.prompt.text = "oops"
        view.handle_key(Key.ENTER)
        assert session.message_is_error
        assert view.prompt.mode == "scalar"
        view.prompt.text = "99"
        view.handle_key(Key.ENTER)
        assert "read-only" in session.message
        assert view.prompt.mode == "scalar"
        assert view.state.regfile.get_cr(index) == index
        for key in (
            Key.F5,
            Key.F6,
            Key.F8,
            Key.F9,
            Key.F10,
            Key.F11,
        ):
            assert view.handle_key(key) is None
        view.handle_key(Key.ESCAPE)
        assert view.prompt.mode is None

    def test_scalar_editor_uses_mask_and_cancel_does_not_write(self):
        from ipu_isa.constants import LR_CR_SCALAR_VALUE_MASK

        view, _, session = make_view()
        session.focus = "registers"
        session.registers.tabs[0].group = "cr"
        session.registers.tabs[0].cursor = 2
        view.handle_key("e")
        view.prompt.text = "-1"
        view.handle_key(Key.ENTER)
        assert view.state.regfile.get_cr(2) == LR_CR_SCALAR_VALUE_MASK
        view.handle_key("e")
        view.prompt.text = "0"
        view.handle_key(Key.ESCAPE)
        assert view.state.regfile.get_cr(2) == LR_CR_SCALAR_VALUE_MASK

    def test_f6_is_unbound_and_cannot_leave_the_tui(self):
        screen = FakeScreen([Key.F6, Key.F8])
        view, _, session = make_view(screen=screen)
        assert view.run() == DebugAction.STEP
        assert view.state.program_counter == 0
        assert session.active
        assert not screen.keys

    @pytest.mark.parametrize("queued", [False, True])
    def test_keyboard_interrupt_cancels_from_blocking_or_queued_input(self, queued):
        class InterruptedScreen(FakeScreen):
            def read_key(self):
                if queued:
                    return "4"
                raise KeyboardInterrupt

            def pending_key(self):
                raise KeyboardInterrupt

        view, _, session = make_view(screen=InterruptedScreen())
        session.active = True
        assert view.run() == DebugAction.QUIT
        assert not session.active


class TestResponsiveMouseUI:
    @pytest.mark.parametrize("rows,columns", [(12, 48), (18, 60), (20, 100), (40, 70)])
    def test_compact_navigation_keeps_every_pane_accessible(self, rows, columns):
        view, screen, session = make_view(screen=FakeScreen(rows=rows, columns=columns))
        for number, pane in zip("1234", PANE_FOCUS_ORDER, strict=True):
            view.render()
            view.handle_key(number)
            view.render()
            assert session.focus == pane
            assert set(view.layout()) == {"header", pane, "status", "keys"}
            assert tui_layout.HEADER_ROWS == tui_layout.FOOTER_ROWS == 1
            assert all(
                0 <= y < rows and 0 <= x < columns and x + len(text) <= columns
                for y, x, text, _ in screen.writes
            )

    def test_shrink_and_grow_restores_user_split_sizes(self):
        view, screen, session = make_view(screen=FakeScreen(rows=50, columns=180))
        session.top_pane_rows = 27
        session.register_pane_columns = 70
        session.pipeline_pane_columns = 65
        original = view.layout()
        for screen.rows, screen.columns in [(24, 80), (12, 48), (50, 180)]:
            view.render()
        assert view.layout() == original
        assert (
            session.top_pane_rows,
            session.register_pane_columns,
            session.pipeline_pane_columns,
        ) == (27, 70, 65)

    def test_idle_poll_repaints_after_a_resize_without_key_resize(self):
        view, screen, _ = make_view(screen=FakeScreen(rows=42, columns=140))
        view.render()
        view.redraw = False
        painted = screen.refreshes
        screen.resize(18, 60)

        def read():
            assert screen.refreshes == painted + 1
            assert footer_row(screen).startswith(" F5:Run")
            assert screen.size() == (18, 60)
            return Key.F8

        screen.read_key = read
        assert view.run() == DebugAction.STEP

    def test_unchanged_terminal_does_not_repaint_on_timeout(self):
        view, screen, _ = make_view()
        view.render()
        view.redraw = False
        painted = screen.refreshes
        inputs = iter([None, Key.F8])
        screen.read_key = lambda: next(inputs)
        assert view.run() == DebugAction.STEP
        assert screen.refreshes == painted

    def test_footer_click_steps_and_tab_buttons_have_their_own_direction(self):
        view, _, session = make_view(screen=FakeScreen(columns=240))
        session.focus = "disassembly"
        session.disassembly.tabs.extend([DisassemblyTab(target=10), DisassemblyTab(target=20)])
        for key, expected in [("d", 1), ("a", 0)]:
            view.render()
            hit = next(rect for rect, action in view.hits.actions if action == key)
            mouse(view, hit.y, hit.x)
            assert session.disassembly_tab() is session.disassembly.tabs[expected]
        view.render()
        hit = next(rect for rect, action in view.hits.actions if action == Key.F8)
        assert mouse(view, hit.y, hit.x) == DebugAction.STEP

    def test_help_scrolls_and_closes_without_clicking_background(self):
        view, _, session = make_view(screen=FakeScreen(rows=18, columns=60))
        view.handle_key("?")
        view.render()
        old_focus = session.focus
        mouse(view, 1, 2)
        assert session.focus == old_focus and view.help.visible
        mouse(view, 8, 10, wheel=1)
        assert view.help.scroll == pointer.MOUSE_WHEEL_LINES
        view.render()
        hit, _ = view.hits.overlay[0]
        mouse(view, hit.y, hit.x)
        assert not view.help.visible
        assert all(len(line) <= 40 for line, _ in help_lines(40))

    @pytest.mark.parametrize("apply", [False, True])
    def test_editor_mouse_places_cursor_and_applies_or_cancels(self, apply):
        view, _, session = make_view(screen=FakeScreen(rows=18, columns=60))
        session.focus = "disassembly"
        view.handle_key(Key.F2)
        view.prompt.text, view.prompt.cursor = "123", 3
        view.render()
        field, _ = view.hits.editor_field
        mouse(view, field.y, field.x + 1)
        assert view.prompt.cursor == 1
        view.handle_key("0")
        view.render()
        hit = next(
            rect for rect, key in view.hits.overlay if key == (Key.ENTER if apply else Key.ESCAPE)
        )
        mouse(view, hit.y, hit.x)
        assert view.prompt.mode is None
        assert len(session.disassembly.tabs) == (2 if apply else 1)
        if apply:
            assert session.disassembly_tab().target == 1023

    def test_scrollbar_drag_reaches_end_and_releases(self):
        view, _, session = make_view()
        view.render()
        track = view.hits.scrollbars["disassembly"]
        mouse(view, track.y, track.x, press=True)
        assert view.pointer.scroll_drag == "disassembly"
        mouse(view, track.y + track.height - 1, track.x, motion=True)
        assert session.disassembly_tab().cursor_offset == INST_MEM_SIZE - 1
        mouse(view, track.y + track.height - 1, track.x, release=True)
        assert view.pointer.scroll_drag is None

    def test_wheel_over_tabs_selects_tabs_and_shift_wheel_moves_columns(self):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        session.registers.tabs.append(RegisterTab(group="lr"))
        view.render()
        rect = view.rects["registers"]
        mouse(view, rect.y + 1, rect.x + 4, wheel=1)
        assert session.register_tab() is session.registers.tabs[1]
        view.render()
        before = session.register_tab().cursor
        rect = view.rects["registers"]
        mouse(view, rect.y + 4, rect.x + 4, wheel=1, shift=True)
        assert session.register_tab().cursor > before

    def test_code_gutter_toggles_breakpoint_but_instruction_click_does_not(self):
        view, _, _ = make_view()
        view.render()
        rect, pc = view.hits.values["disassembly"][0]
        mouse(view, rect.y, rect.x)
        assert pc in view.control.breakpoints
        mouse(view, rect.y, rect.x + 5)
        assert pc in view.control.breakpoints
        mouse(view, rect.y, rect.x)
        assert pc not in view.control.breakpoints


class TestMouseUsability:
    def test_f11_removed_and_footer_help_clickable(self):
        view, screen, session = make_view(screen=FakeScreen(columns=140))
        view.handle_key(Key.F11)
        view.render()
        assert not session.maximized
        assert "F11" not in screen.text()
        assert all(key != Key.F11 for _, key in view.hits.actions)
        hit = next(rect for rect, key in view.hits.actions if key == "?")
        mouse(view, hit.y, hit.x)
        assert view.help.visible

    def test_long_tab_fits_smallest_pane_and_can_be_closed(self):
        view, _, session = make_view()
        session.register_pane_columns = 20
        session.registers.tabs[0].fmt = "int32"
        original_tab = session.registers.tabs[0]
        view.render()
        assert view.rects["registers"].width == 20
        assert len(view.hits.tabs["registers"]) == 1
        close, _ = view.hits.tab_closes["registers"][0]
        assert close.x < view.rects["registers"].x + view.rects["registers"].width - 1
        mouse(view, close.y, close.x)
        assert all(tab is not original_tab for tab in session.registers.tabs)
        assert session.registers.tabs[0].fmt == "hex"

    @pytest.mark.parametrize("pane", ["pipeline", "xmem", "disassembly"])
    def test_grabbing_thumb_does_not_move_selection(self, pane):
        view, _, session = make_view(screen=FakeScreen(rows=40, columns=140))
        session.xmem.tabs[0] = XmemTab(XmemRequest("byte", "0", "0", 512, "hex"))
        session.focus = pane
        view.render()
        view.handle_key(Key.DOWN)
        view.render()
        if pane == "pipeline":
            session.pipeline.tabs[0].fmt = "bits"
            view.render()
        before = cursor_payload(view, session, pane)
        thumb = view.hits.thumbs[pane]
        y = thumb.y + thumb.height - 1
        mouse(view, y, thumb.x, press=True)
        assert cursor_payload(view, session, pane) == before
        mouse(view, y, thumb.x, release=True)
        assert cursor_payload(view, session, pane) == before
        assert view.pointer.scroll_drag is None

    def test_scroll_drag_can_return_to_its_origin(self):
        view, _, session = make_view()
        session.focus = "disassembly"
        view.render()
        thumb = view.hits.thumbs["disassembly"]
        mouse(view, thumb.y, thumb.x, press=True)
        mouse(view, thumb.y + 2, thumb.x, motion=True)
        assert session.disassembly_tab().cursor_offset > 0
        mouse(view, thumb.y, thumb.x, release=True)
        assert session.disassembly_tab().cursor_offset == 0

    @pytest.mark.parametrize("interval,opens", [(0.15, True), (0.7, False)])
    def test_double_click_register_opens_editor_but_slow_clicks_select(
        self, monkeypatch, interval, opens
    ):
        view, _, _ = make_view()
        view.render()
        value, _ = view.hits.values["registers"][0]
        times = iter([100.0, 100.0 + interval])
        monkeypatch.setattr(pointer.time, "monotonic", lambda: next(times))
        mouse(view, value.y, value.x)
        assert view.prompt.mode is None
        mouse(view, value.y, value.x, release=True)
        mouse(view, value.y, value.x)
        assert (view.prompt.mode == "scalar") is opens

    def test_terminal_reported_double_click_edits_selected_register(self):
        view, _, _ = make_view()
        view.render()
        value, _ = view.hits.values["registers"][0]
        mouse(view, value.y, value.x, double=True)
        assert view.prompt.mode == "scalar"

    def test_compact_help_keeps_scroll_range_visible(self):
        view, screen, _ = make_view(screen=FakeScreen(rows=18, columns=48))
        view.handle_key("?")
        view.render()
        assert "Wheel Scroll" in screen.text()
        assert any(
            "/" in row_text(screen, y) and "Esc Close" in row_text(screen, y)
            for y in range(screen.rows)
        )

    def test_click_only_terminal_preserves_selection_on_thumb(self):
        view, _, session = make_view(screen=FakeScreen(rows=40, columns=140))
        session.xmem.tabs[0] = XmemTab(XmemRequest("byte", "0", "0", 512, "hex"))
        session.focus = "xmem"
        view.render()
        before = cursor_payload(view, session, "xmem")
        thumb = view.hits.thumbs["xmem"]
        mouse(view, thumb.y + thumb.height - 1, thumb.x)
        assert cursor_payload(view, session, "xmem") == before

    def test_compact_help_does_not_waste_space_on_column_padding(self):
        lines = [text for text, _ in help_lines(40)]
        assert "F8  Execute one instruction" in lines


class TestPipelineStageView:
    @staticmethod
    def program(source):
        return make_state(source)

    def test_stage_decoding_includes_each_metadata_slot(self):
        from ipu_debug.pipeline import SLOT_LABELS, stage_operations

        state = self.program("SET lr2 cr1;;")
        slots = dict(stage_operations(state.inst_mem[0]))
        assert tuple(slots) == SLOT_LABELS
        assert slots["LR0"].startswith("SET ")
        assert slots["MULT"].startswith("NOP")
        assert any(label.endswith("*") for label in slots)
        assert all(op == "NOP" for _, op in stage_operations(None))

    def test_trace_tracks_actual_branch_pc_and_does_not_record_stops(self):
        from ipu_emu.emulator import run_with_debug

        state = self.program("B +3;; NOP;; NOP;; SET lr2 cr1;; BKPT;;")
        control = DebugControl()
        stops = []

        def paused(state, cycle):
            stops.append((state.program_counter, cycle, control.last_completed_pc))
            return DebugAction.STEP if len(stops) < 3 else DebugAction.QUIT

        assert run_with_debug(state, paused, break_on_entry=True, control=control) == 2
        assert stops == [(0, 0, None), (3, 1, 0), (4, 2, 3)]
        assert control.last_completed_instruction == state.inst_mem[3]
        assert control.last_completed_instruction is not state.inst_mem[3]

    def test_trace_follows_pc_edits_and_resets_for_a_new_run(self):
        from ipu_emu.emulator import run_with_debug

        state = self.program("NOP;; NOP;; SET lr2 cr1;; BKPT;;")
        control = DebugControl()

        def paused(state, cycle):
            if cycle == 0:
                state.program_counter = 2
                return DebugAction.STEP
            assert control.last_completed_pc == 2
            return DebugAction.QUIT

        run_with_debug(state, paused, break_on_entry=True, control=control)
        run_with_debug(
            state, lambda state, cycle: DebugAction.QUIT, break_on_entry=True, control=control
        )
        assert control.last_completed_pc is None

    @pytest.mark.parametrize("rows,columns", [(12, 48), (24, 80), (42, 140)])
    def test_stages_is_disassembly_format_and_other_panes_remain_usable(self, rows, columns):
        view, screen, session = make_view(
            self.program("SET lr2 cr1;;"), screen=FakeScreen(rows=rows, columns=columns)
        )
        view.handle_key("2")
        view.handle_key("f")
        view.handle_key("f")
        assert session.disassembly_tab().fmt == "stages"
        view.render()
        assert "Pipeline stages" in screen.text()
        view.handle_key("4")
        view.handle_key("e")
        assert view.prompt.mode is not None
        assert session.disassembly_tab().fmt == "stages"
        view.handle_key(Key.ESCAPE)
        view.render()
        assert all(
            0 <= y < rows and 0 <= x < columns and x + len(text) <= columns
            for y, x, text, _ in screen.writes
        )

    def test_stage_view_mouse_scroll_and_execution(self):
        view, screen, session = make_view(screen=FakeScreen(rows=24, columns=120))
        view.handle_key("2")
        view.handle_key("f")
        view.handle_key("f")
        view.render()
        rect = view.rects["disassembly"]
        mouse(view, rect.y + 4, rect.x + 2, wheel=1)
        assert session.disassembly.stages_scroll > 0
        assert view.handle_key(Key.F8) == DebugAction.STEP
        assert session.stages_visible
        view.render()
        assert "CR / LR" in screen.text() and "XMEM" in screen.text()
        mouse(view, view.rects["registers"].y + 3, view.rects["registers"].x + 2)
        assert session.focus == "registers"
        assert session.stages_visible

    def test_disassembly_shows_slot_names_and_explicit_truncation(self):
        state = self.program("SET lr2 cr1;;")
        view, screen, session = make_view(state, screen=FakeScreen(columns=160))
        view.render()
        assert "LR0" in screen.text()
        assert "SLOT" in screen.text()
        session.pipeline_pane_columns = 130
        view.render()
        assert view.rects["disassembly"].width == 31

    def test_display_uses_semantic_operand_order_and_omits_unused_fields(self):
        from ipu_debug.pipeline import stage_operations

        state = self.program("SET lr2 cr1; ADD lr3 lr2 cr1;;")
        slots = dict(stage_operations(state.inst_mem[0]))
        assert slots["LR0"] == "SET lr2 cr1"
        assert slots["LR1"] == "ADD lr3 lr2 cr1"
        assert slots["MULT"] == "NOP"

    def test_concurrent_commands_occupy_separate_stages_and_nops_clear_them(self):
        from ipu_debug.pipeline import pipeline_occupancy

        state = self.program(
            "SET lr2 cr1; ADD lr3 lr2 cr1; "
            "MULT.RC.VV lr0 r1 0 lr0 cr15; AGG.MAX.FIRST lr0 cr15;; NOP;;"
        )
        occupied = dict(pipeline_occupancy(state.inst_mem[0]))
        assert [slot for slot, _ in occupied["CTRL"]] == ["LR0", "LR1"]
        assert occupied["MULT"][0][1].startswith("MULT.RC.VV ")
        assert occupied["ACC"][0][1].startswith("AGG.MAX.FIRST ")
        assert occupied["AAQ"] == []
        assert occupied["STORE"] == []
        assert all(not commands for _, commands in pipeline_occupancy(state.inst_mem[1]))

    def test_stage_occupancy_follows_branch_destination(self):
        from ipu_debug.pipeline import pipeline_occupancy
        from ipu_emu.emulator import run_with_debug

        state = self.program("B +2;; SET lr2 cr1;; ADD lr3 lr2 cr1;;")
        observed = []

        def paused(state, cycle):
            observed.append(dict(pipeline_occupancy(state.inst_mem[state.program_counter]))["CTRL"])
            return DebugAction.STEP if cycle == 0 else DebugAction.QUIT

        run_with_debug(state, paused, break_on_entry=True)
        assert observed[0] == [("COND", "BEQ cr0 cr0 2")]
        assert observed[1] == [("LR0", "ADD lr3 lr2 cr1")]


class TestConsoleBars:
    def test_removed_hints_help_button_and_group_order(self):
        view, screen, session = make_view(screen=FakeScreen(columns=200))
        session.focus = "disassembly"
        view.debug.kernel_name = "softmax_rows_partial"
        view.handle_key("p")
        assert session.disassembly_tab().fmt == "compact"
        view.render()
        assert "softmax_rows_partial" in row_text(screen, 0)
        assert not any(key in ("1", "2", "3", "4", "p") for _, key in view.hits.actions)
        hit = next(rect for rect, key in view.hits.actions if key == "?")
        assert hit.y == 0 and hit.x + hit.width == 199
        footer = footer_text(session.focus, 200)
        assert "1-4" not in footer and "Help" not in footer and "p:Stages" not in footer
        assert (
            footer.index("F5:Run")
            < footer.index("Arrows:Move")
            < footer.index("f:Format")
            < footer.index("q Quit")
        )
        assert " | " in footer
        mouse(view, hit.y, hit.x)
        assert view.help.visible


class TestPaneOrder:
    def test_positions_shortcuts_and_footer_have_no_focus_label(self):
        view, screen, session = make_view(screen=FakeScreen(columns=140))
        layout = view.layout()
        assert layout["pipeline"].x == 0
        assert layout["pipeline"].y == layout["disassembly"].y
        assert layout["disassembly"].x > 0
        assert layout["xmem"].x == 0
        assert layout["xmem"].y == layout["registers"].y > layout["pipeline"].y
        for key, pane in zip("1234", ("pipeline", "disassembly", "xmem", "registers"), strict=True):
            view.handle_key(key)
            view.render()
            assert session.focus == pane
            footer = footer_row(screen)
            assert "focused" not in footer
            assert footer.lstrip().startswith("F5:Run")


class TestMemoryValueEditing:
    def test_xmem_edit_resolves_symbolic_address_and_invalidates_cached_views(self):
        state = IpuState()
        state.regfile.set_lr(2, 16)
        state.xmem.write_address(16, b"\x55" * 16)
        view, _, session = make_view(state)
        session.focus = "xmem"
        session.xmem.tabs[0] = XmemTab(XmemRequest("byte", "lr2", "0", 16, "f32"))
        view.render()
        tab = session.xmem.current()
        tab.cursor_column = 1
        generation = view.debug.generation
        view.handle_key("e")
        assert view.prompt.mode == "value"
        view.prompt.text = "1.25"
        view.handle_key(Key.ENTER)
        assert (
            state.xmem.read_address(16, 16) == b"\x55" * 4 + struct.pack("<f", 1.25) + b"\x55" * 8
        )
        assert view.debug.generation > generation
        resolved, error = view.panes["xmem"].resolve(view.debug, tab.request)
        assert error is None
        assert resolved.cells[1].value == 1.25

    @pytest.mark.parametrize("pane", ["pipeline", "xmem"])
    def test_double_click_invalid_input_and_cancel_leave_bytes_unchanged(self, pane):
        state = IpuState()
        view, _, session = make_view(state)
        session.focus = pane
        view.render()
        rect, _ = view.hits.values[pane][0]
        mouse(view, rect.y, rect.x, double=True)
        assert view.prompt.mode == "value"
        view.render()
        view.prompt.text = "256"
        view.handle_key(Key.ENTER)
        assert view.prompt.mode == "value" and session.message_is_error
        view.handle_key(Key.ESCAPE)
        assert view.prompt.mode is None
        assert state.regfile.view("r") == bytearray(256)
        assert state.xmem.read_address(0, 16) == bytes(16)


class TestPopupEscape:
    @pytest.mark.parametrize("key", ["q", "Q"])
    def test_q_does_not_close_help_or_quit(self, key):
        view, _, _ = make_view()
        view.handle_key("?")
        assert view.handle_key(key) is None
        assert view.help.visible
        bindings = [binding for binding in KEY_BINDINGS if binding.scope == "help"]
        assert all("q" not in binding.keys for binding in bindings)
        view.handle_key(Key.ESCAPE)
        assert not view.help.visible

    @pytest.mark.parametrize("pane", ["pipeline", "registers", "xmem"])
    def test_escape_cancels_each_value_popup(self, pane):
        view, _, session = make_view()
        session.focus = pane
        view.render()
        view.handle_key("e")
        assert view.prompt.mode is not None
        view.prompt.text = "123"
        view.handle_key(Key.ESCAPE)
        assert view.prompt.mode is None


class TestStatusAndDividerDirection:
    @pytest.mark.parametrize("pane", PANE_FOCUS_ORDER)
    @pytest.mark.parametrize("key,delta", [(Key.SHIFT_LEFT, -2), (Key.SHIFT_RIGHT, 2)])
    def test_arrow_moves_divider_in_screen_direction(self, pane, key, delta):
        view, _, session = make_view(screen=FakeScreen(columns=140))
        session.focus = pane
        divider = "disassembly" if pane in ("pipeline", "disassembly") else "registers"
        before = view.layout()[divider].x
        view.handle_key(key)
        assert view.layout()[divider].x == before + delta

    @pytest.mark.parametrize("error", [False, True])
    @pytest.mark.parametrize("columns", [48, 80, 140])
    def test_status_is_left_of_help_and_footer_stays_shortcuts(self, error, columns):
        view, screen, session = make_view(screen=FakeScreen(columns=columns))
        message = "Selected all hex"
        session.set_message(message, error=error)
        view.render()
        header = row_text(screen, 0)
        assert message in header
        assert header.index(message) < header.index("[? Help]")
        footer = footer_row(screen)
        assert message not in footer and "F5:Run" in footer
        help_hit = next(rect for rect, key in view.hits.actions if key == "?")
        assert help_hit.y == 0


class TestHelpScrollbar:
    @pytest.mark.parametrize(
        "event",
        ["press", "click", "release"],
    )
    def test_close_button_accepts_terminal_mouse_events(self, event):
        view, _, session = make_view()
        view.handle_key("?")
        view.render()
        hit, _ = view.hits.overlay[0]
        focus = session.focus
        mouse(view, hit.y, hit.x + 2, **{event: True})
        assert not view.help.visible
        mouse(view, hit.y, hit.x + 2, release=True)
        assert session.focus == focus and view.prompt.mode is None

    def test_help_scrollbar_pages_drags_and_releases_without_touching_panes(self):
        view, _, session = make_view(screen=FakeScreen(rows=24, columns=100))
        view.handle_key("?")
        view.render()
        track, thumb = view.help.track, view.help.thumb
        assert track and thumb and view.help.max_scroll > 0
        focus = session.focus
        mouse(view, thumb.y, thumb.x, press=True)
        assert view.help.scroll == 0 and view.help.drag is not None
        mouse(view, track.y + track.height - 1, track.x, motion=True)
        assert view.help.scroll == view.help.max_scroll
        mouse(view, track.y + track.height - 1, track.x, release=True)
        assert view.help.drag is None
        view.render()
        mouse(view, track.y, track.x, click=True)
        assert view.help.scroll < view.help.max_scroll
        assert session.focus == focus and view.pointer.scroll_drag is None


class TestDefaultWorkspace:
    @pytest.mark.parametrize(
        "wide,arithmetic,quantize",
        [
            (False, "fp32", False),
            (True, "fp32", False),
            (True, "int32", False),
            (True, "fp32", True),
        ],
    )
    def test_fresh_tabs_cover_all_registers_and_use_storage_formats(
        self, wide, arithmetic, quantize
    ):
        fmt = ElementFormat(arithmetic) if wide else ElementFormat.INT8
        state = new_state(fmt=fmt, wide_vector_quantize_output=quantize)
        session = DebugViewSession()
        view, _, _ = make_view(state, session=session)
        assert [tab.register_key for tab in session.pipeline.tabs] == [
            spec.key for spec in PIPELINE_REGISTERS
        ]
        formats = {tab.register_key: tab.fmt for tab in session.pipeline.tabs}
        assert formats["r_mask"] == "bits"
        assert (
            formats["r0"] == ("f32" if arithmetic == "fp32" else "int32")
            if wide
            else formats["r0"] == "int8"
        )
        assert formats["post_aaq_reg"] == (
            ("f32" if arithmetic == "fp32" else "int32") if wide and not quantize else "int8"
        )
        assert formats["mem_bypass"] == "hex"
        assert [tab.fmt for tab in session.disassembly.tabs] == ["compact", "full", "stages"]
        assert [tab.group for tab in session.registers.tabs] == ["all", "lr", "cr"]
        assert len(session.xmem.tabs) == 2 and session.xmem.tabs[0].request.fmt == "hex"
        session.pipeline.tabs.pop()
        session.pipeline.tabs[0].fmt = "u32"
        session.ensure_default_tab(state)
        assert len(session.pipeline.tabs) == len(PIPELINE_REGISTERS) - 1
        assert session.pipeline.tabs[0].fmt == "u32"

    @pytest.mark.parametrize("rows,columns", [(24, 80), (42, 140), (60, 200)])
    def test_default_sizes_balance_rows_and_manual_sizes_survive_resize(self, rows, columns):
        view, screen, session = make_view(
            screen=FakeScreen(rows=rows, columns=columns), session=DebugViewSession()
        )
        layout = view.layout()
        assert abs(layout["pipeline"].height - layout["xmem"].height) <= 1
        assert abs(layout["pipeline"].width - columns // 2) <= 2
        view.render()
        assert all(
            0 <= y < rows and 0 <= x < columns and x + len(text) <= columns
            for y, x, text, _ in screen.writes
        )
        session.pipeline_pane_columns = columns // 2 + 3
        expected = view.layout()
        screen.rows, screen.columns = 12, 48
        view.render()
        screen.rows, screen.columns = rows, columns
        assert view.layout() == expected
