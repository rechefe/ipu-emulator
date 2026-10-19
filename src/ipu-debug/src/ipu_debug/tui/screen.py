"""The terminal seam of the TUI: the only module that imports ``curses``.

The view draws, reads input and sizes itself through the :class:`Screen`
protocol. Two adapters implement it:

- :class:`CursesScreen` drives a real terminal through ``curses``;
- :class:`FakeScreen` records every draw into a character grid, for tests.

Input arrives already decoded: a character as a one-character ``str``, a
special key as a :class:`Key`, and a pointer event as a :class:`MouseEvent`.
Styles are :class:`Style` flags plus an optional colour pair (see
:func:`color_style`); only :class:`CursesScreen` knows the ``curses``
attributes they stand for.
"""

from __future__ import annotations

import enum
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from ipu_debug.errors import TerminalUnavailableError

try:  # pragma: no cover - import availability is platform-dependent
    import curses
except ImportError:  # pragma: no cover
    curses = None  # type: ignore[assignment]

INPUT_POLL_MS = 100
# Escape also prefixes arrows/function keys; bound that ambiguity wait.
ESCAPE_DELAY_MS = 25
# An upper bound on the colour pairs one screen allocates.
MAX_COLORS = 16


class Key(enum.Enum):
    """A non-character key, decoded from whatever the terminal sent."""

    UP = "Up"
    DOWN = "Down"
    LEFT = "Left"
    RIGHT = "Right"
    HOME = "Home"
    END = "End"
    PAGE_UP = "PgUp"
    PAGE_DOWN = "PgDn"
    SHIFT_LEFT = "Shift-Left"
    SHIFT_RIGHT = "Shift-Right"
    SHIFT_UP = "Shift-Up"
    SHIFT_DOWN = "Shift-Down"
    SHIFT_TAB = "Shift-Tab"
    TAB = "Tab"
    ENTER = "Enter"
    ESCAPE = "Esc"
    BACKSPACE = "Backspace"
    DELETE = "Delete"
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"
    F4 = "F4"
    F5 = "F5"
    F6 = "F6"
    F7 = "F7"
    F8 = "F8"
    F9 = "F9"
    F10 = "F10"
    F11 = "F11"
    F12 = "F12"
    RESIZE = "Resize"

    @property
    def label(self) -> str:
        """How the help overlay and footer name this key."""
        return self.value


@dataclass(frozen=True)
class MouseEvent:
    """One decoded pointer event at ``(y, x)``.

    ``press`` is a button going down (it may start a drag); ``click`` and
    ``double`` are presses the terminal already paired with their release;
    ``wheel`` is the number of notches, negative for up.
    """

    y: int
    x: int
    press: bool = False
    click: bool = False
    double: bool = False
    release: bool = False
    wheel: int = 0
    shift: bool = False

    @property
    def pressed(self) -> bool:
        return self.press or self.click or self.double


# What ``Screen.read_key`` returns: a character, a key or a pointer event.
Input = str | Key | MouseEvent


class Style(enum.IntFlag):
    """Text emphasis. A colour pair rides above these bits (``color_style``)."""

    NONE = 0
    BOLD = 1
    UNDERLINE = 2
    REVERSE = 4
    DIM = 8
    ITALIC = 16


COLOR_SHIFT = 8
COLOR_MASK = ~((1 << COLOR_SHIFT) - 1)


def color_style(pair: int) -> int:
    """The style bits that select colour pair *pair* (0 is no colour)."""
    return pair << COLOR_SHIFT


def overlay_style(base: int, overlay: int) -> int:
    """Combine emphasis, replacing a colour pair instead of OR-ing pair IDs."""
    if overlay & COLOR_MASK:
        base &= ~COLOR_MASK
    return base | overlay


class Screen(Protocol):
    """Everything the TUI needs from a terminal."""

    def size(self) -> tuple[int, int]:
        """``(rows, columns)``."""
        ...

    def draw(self, y: int, x: int, text: str, style: int = 0) -> None:
        """Write *text* at ``(y, x)``, clipped at the right edge."""
        ...

    def clear(self) -> None:
        """Blank the next frame."""
        ...

    def refresh(self) -> None:
        """Show what was drawn since the last refresh."""
        ...

    def read_key(self) -> Input | None:
        """The next input, waiting at most one poll interval (None: nothing)."""
        ...

    def pending_key(self) -> Input | None:
        """The next already-queued input, without waiting (None: nothing)."""
        ...

    def cursor(self, visible: bool, y: int = 0, x: int = 0) -> None:
        """Show the text cursor at ``(y, x)``, or hide it."""
        ...

    def resized(self) -> bool:
        """Whether the terminal changed size since the last call."""
        ...

    def color(self, fg: int, bg: int) -> int:
        """The style for colour pair *fg* on *bg* (-1: default); 0 without colour."""
        ...

    @property
    def has_colors(self) -> bool: ...


class FakeScreen:
    """A :class:`Screen` that records draws into a character grid.

    *keys* are returned by :meth:`read_key` in order; :meth:`read_key` raises
    once they run out, so a test can never hang waiting for input.
    """

    def __init__(
        self,
        keys: list[Input] | None = None,
        *,
        rows: int = 24,
        columns: int = 80,
        colors: int = 0,
    ) -> None:
        self.keys: list[Input] = list(keys or [])
        self.rows = rows
        self.columns = columns
        self.colors = colors
        self.cursor_visible = False
        self.cursor_position = (0, 0)
        self.refreshes = 0
        self._resized = False
        self._pairs: dict[tuple[int, int], int] = {}
        self.writes: list[tuple[int, int, str, int]] = []
        self._grid: list[list[str]] = []
        self._styles: list[list[int]] = []
        self.clear()

    def size(self) -> tuple[int, int]:
        return self.rows, self.columns

    def draw(self, y: int, x: int, text: str, style: int = 0) -> None:
        if not 0 <= y < self.rows:
            return
        text = text[: max(0, self.columns - x)]
        self.writes.append((y, x, text, style))
        for offset, character in enumerate(text):
            if 0 <= x + offset < self.columns:
                self._grid[y][x + offset] = character
                self._styles[y][x + offset] = style

    def clear(self) -> None:
        self.writes = []
        self._grid = [[" "] * self.columns for _ in range(self.rows)]
        self._styles = [[0] * self.columns for _ in range(self.rows)]

    def refresh(self) -> None:
        self.refreshes += 1

    def read_key(self) -> Input | None:
        if not self.keys:
            raise AssertionError("FakeScreen has no more input")
        return self.keys.pop(0)

    def pending_key(self) -> Input | None:
        return self.keys.pop(0) if self.keys else None

    def cursor(self, visible: bool, y: int = 0, x: int = 0) -> None:
        self.cursor_visible = visible
        if visible:
            self.cursor_position = (y, x)

    def resized(self) -> bool:
        resized, self._resized = self._resized, False
        return resized

    def resize(self, rows: int, columns: int) -> None:
        """Change the terminal size, as a user dragging the window would."""
        self.rows, self.columns = rows, columns
        self._resized = True
        self.clear()

    def color(self, fg: int, bg: int) -> int:
        if not self.colors:
            return 0
        key = (fg % self.colors, bg if bg < 0 else bg % self.colors)
        return color_style(self._pairs.setdefault(key, len(self._pairs) + 1))

    @property
    def has_colors(self) -> bool:
        return bool(self.colors)

    # Inspection helpers for tests.

    def row(self, y: int) -> str:
        """Row *y* as it now appears, trailing blanks removed."""
        return "".join(self._grid[y]).rstrip()

    def cell(self, y: int, x: int) -> str:
        return self._grid[y][x]

    def style_at(self, y: int, x: int) -> int:
        return self._styles[y][x]

    def text(self) -> str:
        """Every write since the last clear, one per line."""
        return "\n".join(text for _, _, text, _ in self.writes)

    def grid(self) -> str:
        """The character grid then the style grid, as golden files store them."""
        rows = [self.row(y) for y in range(self.rows)]
        styles = [
            "".join(_style_code(style) for style in self._styles[y]).rstrip(".")
            for y in range(self.rows)
        ]
        return "\n".join([*rows, "--- styles", *styles]) + "\n"


_STYLE_CODES = "0123456789abcdefghijklmnopqrstuv"


def _style_code(style: int) -> str:
    """One character per cell: ``.`` for plain, else the emphasis bits in base 32."""
    flags = style & ~COLOR_MASK
    return "." if not flags else _STYLE_CODES[flags]


class CursesScreen:
    """A :class:`Screen` over a ``curses`` window."""

    def __init__(self, window: object) -> None:
        if curses is None:  # pragma: no cover
            raise TerminalUnavailableError("curses is not available on this platform")
        self._window = window
        self._pairs: dict[tuple[int, int], int] = {}
        self._colors = False
        self._last_size = self.size()
        self._configure()

    def _configure(self) -> None:
        # timeout()/nodelay() do not bound ncurses escape-sequence decoding;
        # set_escdelay does, where the curses build has it (Python 3.9+ on ncurses).
        if hasattr(curses, "set_escdelay"):
            try:
                curses.set_escdelay(ESCAPE_DELAY_MS)
            except curses.error:
                pass
        self._window.keypad(True)
        self._window.nodelay(False)
        self._window.timeout(INPUT_POLL_MS)
        try:
            curses.curs_set(0)
        except curses.error:
            pass  # Not every terminal can hide its cursor.
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS | getattr(curses, "REPORT_MOUSE_POSITION", 0))
            curses.mouseinterval(0)
        except curses.error:
            pass
        try:
            if curses.has_colors():
                curses.start_color()
                try:
                    curses.use_default_colors()
                except curses.error:
                    pass
                self._colors = True
        except curses.error:
            self._colors = False

    def size(self) -> tuple[int, int]:
        return self._window.getmaxyx()

    def draw(self, y: int, x: int, text: str, style: int = 0) -> None:
        rows, columns = self.size()
        if not 0 <= y < rows or not 0 <= x < columns or not text:
            return
        try:
            self._window.addnstr(y, x, text, columns - x, self._attribute(style))
        except curses.error:
            # Writing the bottom-right cell moves the cursor off the screen,
            # which curses reports after the character has been drawn.
            pass

    def clear(self) -> None:
        self._window.erase()

    def refresh(self) -> None:
        try:
            self._window.refresh()
        except curses.error:
            pass

    def read_key(self) -> Input | None:
        try:
            return self._decode(self._window.get_wch())
        except curses.error:
            # A resize interrupts the blocking read, and ncurses reports that
            # as "no input" rather than as KEY_RESIZE.
            return None

    def pending_key(self) -> Input | None:
        self._window.nodelay(True)
        try:
            return self._decode(self._window.get_wch())
        except curses.error:
            return None
        finally:
            self._window.nodelay(False)
            self._window.timeout(INPUT_POLL_MS)

    def cursor(self, visible: bool, y: int = 0, x: int = 0) -> None:
        try:
            curses.curs_set(1 if visible else 0)
            if visible:
                self._window.move(y, x)
        except curses.error:
            # Not every terminal can show or hide its cursor.
            pass

    def resized(self) -> bool:
        """Use the actual PTY size, even when ncurses misses a resize event."""
        try:
            terminal = os.get_terminal_size(sys.stdin.fileno())
            dimensions = (terminal.lines, terminal.columns)
        except (OSError, ValueError):
            # stdin is not a terminal (or is closed): trust curses.
            dimensions = self.size()
        if min(dimensions) <= 0 or dimensions == self._last_size:
            return False
        try:
            if dimensions != self.size():
                curses.resizeterm(*dimensions)
        except (ValueError, curses.error):
            return False
        self._last_size = dimensions
        self._window.clear()
        return True

    def color(self, fg: int, bg: int) -> int:
        if not self._colors:
            return 0
        color_count = max(1, min(getattr(curses, "COLORS", 8), MAX_COLORS))
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
        return color_style(pair)

    @property
    def has_colors(self) -> bool:
        return self._colors

    def _attribute(self, style: int) -> int:
        attribute = 0
        for flag, name in _CURSES_ATTRIBUTES:
            if style & flag:
                attribute |= getattr(curses, name, 0) or (
                    curses.A_UNDERLINE if flag is Style.ITALIC else 0
                )
        pair = (style & COLOR_MASK) >> COLOR_SHIFT
        if pair:
            try:
                attribute |= curses.color_pair(pair)
            except curses.error:
                pass
        return attribute

    def _decode(self, raw: object) -> Input | None:
        if raw == getattr(curses, "KEY_MOUSE", None):
            return self._mouse()
        key: Input | None = None
        if isinstance(raw, int):
            key = _KEY_CODES.get(raw) or _CHARACTER_KEYS.get(
                chr(raw) if 0 <= raw < 0x110000 else ""
            )
        elif isinstance(raw, str):
            key = _CHARACTER_KEYS.get(raw, raw)
        if key is Key.RESIZE:
            self.resized()
        return key

    def _mouse(self) -> MouseEvent | None:
        try:
            _, x, y, _, state = curses.getmouse()
        except curses.error:
            return None

        def has(name: str) -> bool:
            return bool(state & getattr(curses, name, 0))

        wheel = -1 if has("BUTTON4_PRESSED") else 1 if has("BUTTON5_PRESSED") else 0
        return MouseEvent(
            y,
            x,
            press=has("BUTTON1_PRESSED"),
            click=has("BUTTON1_CLICKED"),
            double=has("BUTTON1_DOUBLE_CLICKED"),
            release=has("BUTTON1_RELEASED"),
            wheel=wheel,
            shift=has("BUTTON_SHIFT"),
        )


_CURSES_ATTRIBUTES = (
    (Style.BOLD, "A_BOLD"),
    (Style.UNDERLINE, "A_UNDERLINE"),
    (Style.REVERSE, "A_REVERSE"),
    (Style.DIM, "A_DIM"),
    (Style.ITALIC, "A_ITALIC"),
)
_CHARACTER_KEYS: dict[str, Key] = {
    "\x1b": Key.ESCAPE,
    "\n": Key.ENTER,
    "\r": Key.ENTER,
    "\t": Key.TAB,
    "\b": Key.BACKSPACE,
    "\x7f": Key.BACKSPACE,
}
_CURSES_KEY_NAMES = {
    "KEY_UP": Key.UP,
    "KEY_DOWN": Key.DOWN,
    "KEY_LEFT": Key.LEFT,
    "KEY_RIGHT": Key.RIGHT,
    "KEY_HOME": Key.HOME,
    "KEY_END": Key.END,
    "KEY_PPAGE": Key.PAGE_UP,
    "KEY_NPAGE": Key.PAGE_DOWN,
    "KEY_SLEFT": Key.SHIFT_LEFT,
    "KEY_SRIGHT": Key.SHIFT_RIGHT,
    "KEY_SR": Key.SHIFT_UP,
    "KEY_SF": Key.SHIFT_DOWN,
    "KEY_BTAB": Key.SHIFT_TAB,
    "KEY_ENTER": Key.ENTER,
    "KEY_BACKSPACE": Key.BACKSPACE,
    "KEY_DC": Key.DELETE,
    "KEY_RESIZE": Key.RESIZE,
    **{f"KEY_F{number}": Key[f"F{number}"] for number in range(1, 13)},
}
_KEY_CODES: dict[int, Key] = (
    {getattr(curses, name): key for name, key in _CURSES_KEY_NAMES.items() if hasattr(curses, name)}
    if curses is not None
    else {}
)

T = TypeVar("T")


def run_in_terminal(body: Callable[[Screen], T]) -> T:
    """Enter curses, run *body* on a :class:`CursesScreen` and restore the terminal."""
    if curses is None:
        raise TerminalUnavailableError("curses is not available on this platform")
    return curses.wrapper(lambda window: body(CursesScreen(window)))


def terminal_available() -> bool:
    """Whether this platform has ``curses`` at all."""
    return curses is not None
