"""The TUI shell: golden screens, ``TabSet``, the keymap and the curses seam.

The golden screens are the proof that the per-pane moves (E08-09 to E08-11)
change nothing a user sees. Each file under ``golden/tui/`` holds the 24x80
character grid and the emphasis grid one key sequence leaves on the screen.
They were recorded from the TUI as it stood before E08-08 (commit 427c18bc),
through a recording screen patched into the old ``CursesDebugView`` once, and
the new code must reproduce them through ``FakeScreen``.
"""

from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest
from emu_support import make_state

import ipu_debug.tui.screen as screen_module
from ipu_debug.control import DebugControl
from ipu_debug.session import DebugSession
from ipu_debug.tui import new_debug_view_session
from ipu_debug.tui.draw import UNICODE_GLYPHS
from ipu_debug.tui.keymap import (
    KEY_BINDINGS,
    KEYMAP,
    PANE_FOCUS_ORDER,
    Command,
    Focus,
    footer_text,
    help_lines,
    key_label,
    lookup,
)
from ipu_debug.tui.screen import (
    CursesScreen,
    FakeScreen,
    Key,
    MouseEvent,
    Style,
    overlay_style,
)
from ipu_debug.tui.tabs import TabSet
from ipu_debug.tui.view import CursesDebugView
from ipu_emu.emulator import DebugAction, run_with_debug

_PACKAGE = Path(__file__).parent.parent
GOLDEN = Path(__file__).parent / "golden" / "tui"

PROGRAM = "SET lr2 cr1;; SET lr3 cr1;; NOP;; SET lr4 cr1;; NOP;; NOP;; BKPT;;"
# Enough Backspaces to empty any editor's default text.
CLEAR = [Key.BACKSPACE] * 40

# One list of keys per debugger stop; the screen is captured at the last stop.
SEQUENCES: dict[str, list[list[object]]] = {
    "open_tab": [[Key.F2, *CLEAR, *"row 0 0 256 u8", Key.ENTER]],
    "close_tab": [[Key.F4]],
    "switch_focus": [["4", Key.SHIFT_TAB, "d"]],
    "edit_register": [["4", Key.DOWN, Key.DOWN, "e", *CLEAR, *"0x1234", Key.ENTER]],
    "scroll_xmem": [
        [
            Key.F2,
            *CLEAR,
            *"row 0 0 2048 hex",
            Key.ENTER,
            Key.DOWN,
            Key.DOWN,
            Key.PAGE_DOWN,
            Key.RIGHT,
            Key.RIGHT,
            Key.RIGHT,
        ]
    ],
    "step": [[Key.F8], [Key.F8], []],
    "continue": [["2", Key.DOWN, Key.DOWN, Key.DOWN, Key.F9, Key.F5], []],
    "toggle_breakpoint": [["2", Key.DOWN, Key.DOWN, Key.F9, Key.DOWN, Key.F9, Key.F9]],
    "resize": [[("resize", 30, 100), Key.SHIFT_LEFT]],
    "help": [["?", Key.DOWN, Key.DOWN]],
}


def xmem_pattern() -> bytes:
    return bytes((index * 7 + 3) % 256 for index in range(4096))


def play(stops: list[list[object]]) -> str:
    """Run the program, pressing each stop's keys; the last stop's screen."""
    state = make_state(PROGRAM)
    DebugSession(state).write_xmem(0, xmem_pattern())
    control = DebugControl()
    debug = DebugSession(state, control, kernel_name="golden")
    session = new_debug_view_session(state)
    screen = FakeScreen()
    pending = list(stops)
    captured: list[str] = []

    def paused(state, cycle):
        view = CursesDebugView(screen, state, session, cycle, control, debug)
        view.glyphs = UNICODE_GLYPHS
        for key in pending.pop(0):
            view.render()
            if isinstance(key, tuple):
                _, rows, columns = key
                screen.resize(rows, columns)
                key = Key.RESIZE
            outcome = view.handle_key(key)
            if outcome is not None:
                return outcome
        view.render()
        captured.append(screen.grid())
        return DebugAction.QUIT

    run_with_debug(state, paused, break_on_entry=True, control=control)
    assert len(captured) == 1 and not pending
    return captured[0]


class TestGoldenScreens:
    @pytest.mark.parametrize("name", sorted(SEQUENCES))
    def test_screen_matches_the_recorded_golden(self, name):
        expected = (GOLDEN / f"{name}.txt").read_text(encoding="utf-8")
        assert play(SEQUENCES[name]) == expected

    def test_every_golden_has_a_sequence(self):
        assert {path.stem for path in GOLDEN.glob("*.txt")} == set(SEQUENCES)


class TestTabSet:
    def test_add_selects_the_new_tab(self):
        tabs = TabSet(["a", "b"])
        tabs.add("c")
        assert tabs.current() == "c"
        assert tabs.active == 2

    def test_current_of_an_empty_set_is_none(self):
        assert TabSet().current() is None

    def test_current_clamps_a_stale_index(self):
        tabs = TabSet(["a", "b"], active=5)
        assert tabs.current() == "b"
        assert tabs.active == 1

    def test_close_selects_the_tab_that_takes_its_place(self):
        tabs = TabSet(["a", "b", "c"], active=1)
        tabs.close(1)
        assert tabs.tabs == ["a", "c"]
        assert tabs.current() == "c"

    def test_close_of_the_last_tab_selects_the_new_last(self):
        tabs = TabSet(["a", "b", "c"], active=2, offset=2)
        tabs.close(2)
        assert tabs.current() == "b"
        assert tabs.offset == 1

    def test_close_out_of_range_does_nothing(self):
        tabs = TabSet(["a"])
        tabs.close(3)
        tabs.close(-1)
        assert tabs.tabs == ["a"]

    def test_closing_every_tab_reopens_the_default(self):
        tabs = TabSet(["a"], default=lambda: "fresh")
        tabs.close(0)
        assert tabs.tabs == ["fresh"]
        assert tabs.current() == "fresh"

    def test_closing_every_tab_without_a_default_leaves_it_empty(self):
        tabs = TabSet(["a"])
        tabs.close(0)
        assert tabs.current() is None
        assert (tabs.active, tabs.offset) == (0, 0)

    def test_select_scrolls_the_bar_to_keep_the_tab_in_view(self):
        tabs = TabSet(list("abcdefg"))
        tabs.select(5)
        assert tabs.offset == 3
        tabs.select(1)
        assert tabs.offset == 1
        tabs.select(2)
        assert tabs.offset == 1

    def test_step_wraps_around(self):
        tabs = TabSet(["a", "b", "c"])
        assert tabs.step(-1) == "c"
        assert tabs.step(1) == "a"
        assert TabSet().step(1) is None

    def test_replace_swaps_the_current_tab(self):
        tabs = TabSet(["a", "b"], active=1)
        tabs.replace("z")
        assert tabs.tabs == ["a", "z"]
        empty: TabSet[str] = TabSet()
        empty.replace("z")
        assert empty.tabs == ["z"]


class TestKeymap:
    def test_every_keymap_entry_appears_in_the_help_overlay(self):
        # ``q`` lives in the footer, which stays visible under the overlay.
        shown = "\n".join(text for text, _ in help_lines(72))
        shown += "\n" + "\n".join(footer_text(pane, 200) for pane in PANE_FOCUS_ORDER)
        missing = [
            f"{focus.value}: {key_label(key)}"
            for focus, key in KEYMAP
            if key_label(key) not in shown and not _covered_by_range(key_label(key), shown)
        ]
        assert not missing

    def test_every_bound_row_is_documented(self):
        for binding in KEY_BINDINGS:
            if binding.binds:
                assert binding.description
                assert binding.keys

    def test_modal_focuses_do_not_fall_back_to_global_keys(self):
        assert lookup(Focus.XMEM, "q") is Command.QUIT
        assert lookup(Focus.EDITOR, "q") is None
        assert lookup(Focus.HELP, "q") is None

    def test_letters_match_either_case(self):
        assert lookup(Focus.REGISTERS, "F") is Command.CYCLE_FORMAT
        assert lookup(Focus.REGISTERS, "Q") is Command.QUIT

    def test_disassembly_only_keys_do_nothing_elsewhere(self):
        assert lookup(Focus.DISASSEMBLY, Key.F9) is Command.TOGGLE_BREAKPOINT
        assert lookup(Focus.XMEM, Key.F9) is None

    def test_every_command_has_a_handler(self):
        view = CursesDebugView(FakeScreen(), make_state("NOP;;"), _session(), cycle=0)
        assert set(view.commands) >= set(KEYMAP.values())


def _session():
    state = make_state("NOP;;")
    return new_debug_view_session(state)


def _covered_by_range(label: str, shown: str) -> bool:
    """``1``-``4`` are documented together as ``1-4``."""
    return label.isdigit() and "1-4" in shown


class TestScreens:
    def test_only_the_screen_module_imports_curses(self):
        importers = []
        for path in sorted((_PACKAGE / "src").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    names = [node.module or ""]
                if any(name == "curses" or name.startswith("curses.") for name in names):
                    importers.append(path.relative_to(_PACKAGE).as_posix())
        assert importers == ["src/ipu_debug/tui/screen.py"]

    def test_fake_screen_records_draws_into_a_grid(self):
        screen = FakeScreen(rows=3, columns=10)
        screen.draw(1, 2, "hello world", Style.BOLD)
        assert screen.row(1) == "  hello wo"
        assert screen.style_at(1, 2) == Style.BOLD
        screen.clear()
        assert screen.row(1) == ""

    def test_fake_screen_hands_out_keys_then_refuses_to_block(self):
        event = MouseEvent(1, 2, click=True)
        screen = FakeScreen(["a", event])
        assert screen.read_key() == "a"
        assert screen.pending_key() == event
        assert screen.pending_key() is None
        with pytest.raises(AssertionError):
            screen.read_key()

    def test_overlay_replaces_a_colour_instead_of_mixing_pairs(self):
        screen = FakeScreen(colors=16)
        red, blue = screen.color(1, -1), screen.color(4, -1)
        assert overlay_style(red | Style.BOLD, blue) == blue | Style.BOLD
        assert overlay_style(red, Style.UNDERLINE) == red | Style.UNDERLINE


class _Window:
    """A curses window stub: records draws, hands out raw key codes."""

    def __init__(self, raw_keys=()):
        self.raw_keys = list(raw_keys)
        self.drawn = []

    def getmaxyx(self):
        return 24, 80

    def keypad(self, enabled):
        pass

    def nodelay(self, enabled):
        pass

    def timeout(self, delay):
        pass

    def addnstr(self, y, x, text, count, attribute):
        self.drawn.append((y, x, text[:count], attribute))

    def get_wch(self):
        return self.raw_keys.pop(0)


@pytest.fixture
def fake_curses(monkeypatch):
    """The real ``curses`` constants, with the calls that need a terminal stubbed."""
    real = screen_module.curses
    calls: dict[str, list[object]] = {"escdelay": [], "cursor": []}
    fake = types.SimpleNamespace(**{name: getattr(real, name) for name in dir(real)})
    fake.set_escdelay = calls["escdelay"].append
    fake.curs_set = calls["cursor"].append
    fake.mousemask = lambda mask: (mask, 0)
    fake.mouseinterval = lambda interval: None
    fake.has_colors = lambda: False
    fake.mouse_state = (0, 0, 0, 0, 0)
    fake.getmouse = lambda: fake.mouse_state
    monkeypatch.setattr(screen_module, "curses", fake)
    fake.calls = calls
    return fake


class TestCursesScreen:
    def test_terminal_sets_short_escape_sequence_timeout(self, fake_curses):
        CursesScreen(_Window())
        assert fake_curses.calls["escdelay"] == [screen_module.ESCAPE_DELAY_MS]
        assert fake_curses.calls["cursor"] == [0]

    @pytest.mark.parametrize(
        ("raw", "key"),
        [
            ("KEY_F5", Key.F5),
            ("KEY_BTAB", Key.SHIFT_TAB),
            ("KEY_NPAGE", Key.PAGE_DOWN),
            ("KEY_DC", Key.DELETE),
            ("KEY_BACKSPACE", Key.BACKSPACE),
            ("KEY_ENTER", Key.ENTER),
            ("\n", Key.ENTER),
            ("\r", Key.ENTER),
            ("\t", Key.TAB),
            ("\x1b", Key.ESCAPE),
            ("\x7f", Key.BACKSPACE),
            ("x", "x"),
            ("?", "?"),
        ],
    )
    def test_raw_keys_decode_to_keys_and_characters(self, fake_curses, raw, key):
        code = getattr(fake_curses, raw) if raw.startswith("KEY_") else raw
        assert CursesScreen(_Window([code])).read_key() == key

    def test_integer_control_codes_decode_too(self, fake_curses):
        screen = CursesScreen(_Window([27, 10, 9]))
        assert [screen.read_key() for _ in range(3)] == [Key.ESCAPE, Key.ENTER, Key.TAB]

    def test_mouse_reports_decode_to_events(self, fake_curses):
        fake_curses.mouse_state = (
            0,
            7,
            3,
            0,
            fake_curses.BUTTON1_PRESSED | fake_curses.BUTTON_SHIFT,
        )
        event = CursesScreen(_Window([fake_curses.KEY_MOUSE])).read_key()
        assert event == MouseEvent(3, 7, press=True, shift=True)

    def test_styles_become_curses_attributes(self, fake_curses):
        window = _Window()
        CursesScreen(window).draw(1, 2, "hi", Style.BOLD | Style.REVERSE)
        assert window.drawn == [(1, 2, "hi", fake_curses.A_BOLD | fake_curses.A_REVERSE)]
