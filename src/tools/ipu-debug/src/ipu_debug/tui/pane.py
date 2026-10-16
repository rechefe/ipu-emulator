"""The interface one TUI pane implements, and the shapes it trades in.

A pane renders its tabs into a rectangle and handles the commands aimed at
it. The view owns layout, the frame, each pane's tab bar, the footer and the
overlays; the panes own what is inside their frame.

A pane module (``ipu_debug.tui.panes.*``) renders purely: session and tabs
in, a :class:`PaneDrawing` out, with no screen and no view. It never touches
the view either: :meth:`Pane.handle` answers with a :class:`Reply`, which can
carry a status message and an :class:`Editor` for the view to open.

The panes not yet moved into a module of their own (disassembly and pipeline,
E08-11) are each a :class:`LegacyPane`, which wraps the drawing and command
code still living in ``CursesDebugView``. The golden screen tests in
``test_tui_shell`` prove those moves change nothing on screen.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from ipu_debug.session import DebugSession
from ipu_debug.tui.keymap import Command, PaneId
from ipu_debug.tui.tabs import TabSet
from ipu_emu.emulator import DebugAction

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class Rect:
    """Zero-based terminal rectangle."""

    y: int
    x: int
    height: int
    width: int

    def contains(self, y: int, x: int) -> bool:
        return self.y <= y < self.y + self.height and self.x <= x < self.x + self.width


@dataclass(frozen=True)
class Line:
    """One run of text a pane draws: *text* at ``(y, x)`` in *style*."""

    y: int
    x: int
    text: str
    style: int = 0


@dataclass
class PaneDrawing:
    """What one render of a pane produced.

    ``hits`` maps the rectangle each value landed in to the payload
    :meth:`Pane.place_cursor` takes, so a click can select it. ``scroll`` is
    ``(first, visible, total)`` rows for a vertically scrollable pane, from
    which the view draws the scroll thumb.
    """

    lines: list[Line]
    hits: list[tuple[Rect, Any]] = field(default_factory=list)
    scroll: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class Reply:
    """What handling a command asks of the view: a status message, an editor."""

    message: str | None = None
    error: bool = False
    editor: Editor | None = None


@dataclass(frozen=True)
class Editor:
    """A one-line editor a pane asks the view to open.

    ``mode`` is ``add`` or ``edit`` (a tab target), ``value`` (bytes under
    the cursor) or ``scalar`` (the 32-bit register named by ``register``).
    The view draws ``title``, the two ``hints`` and a ``pane >>>`` prompt
    holding ``text``; Enter hands the edited text to ``submit``. An error
    reply keeps the editor open with its message; any other reply closes it.
    """

    mode: str
    pane: PaneId
    title: str
    hints: tuple[str, str]
    text: str
    submit: Callable[[str], Reply]
    register: str | None = None


# A pane's persistent view state: its tabs.
PaneState = TabSet[Any]
# What handling a key leaves for the run: an execution action, or None to
# stay stopped.
Outcome = DebugAction | None


class Pane(Protocol):
    """One pane of the debugger."""

    id: PaneId

    def render(
        self, session: DebugSession, view: PaneState, rect: Rect, *, focused: bool = False
    ) -> PaneDrawing:
        """What draws *view* inside *rect* (the tab bar and frame excluded)."""
        ...

    def handle(
        self, command: Command, session: DebugSession, view: PaneState, rect: Rect
    ) -> Reply | None:
        """Carry out *command* on *view*, laid out in *rect*."""
        ...

    def place_cursor(self, view: PaneState, payload: Any) -> None:
        """Put the cursor on the value a :attr:`PaneDrawing.hits` entry names."""
        ...

    def cursor_row(self, session: DebugSession, view: PaneState, rect: Rect) -> int:
        """The scroll row the cursor is on, for grabbing the scroll thumb."""
        ...

    def scroll_to_row(self, session: DebugSession, view: PaneState, rect: Rect, row: int) -> None:
        """Move the cursor to scroll row *row* (a scrollbar click)."""
        ...

    def move(
        self, session: DebugSession, view: PaneState, rect: Rect, *, dx: int = 0, dy: int = 0
    ) -> None:
        """Move the cursor *dx* values sideways and *dy* rows down (the wheel)."""
        ...


class LineRecorder:
    """Collects the lines a drawing function emits, for :meth:`Pane.render`."""

    def __init__(self) -> None:
        self.lines: list[Line] = []

    def __call__(self, y: int, x: int, text: str, style: int) -> None:
        self.lines.append(Line(y, x, text, style))

    def __iter__(self) -> Iterator[Line]:
        return iter(self.lines)


@dataclass
class LegacyPane:
    """A pane whose drawing and commands are still methods of the view.

    *draw* paints the pane into a rectangle through the view's line sink, and
    *commands* maps each pane command to the view method that carries it out.
    The wrapped code records its own hits and scroll state on the view.
    """

    id: PaneId
    draw: Callable[[Rect], None]
    capture: Callable[[Callable[[Rect], None], Rect], list[Line]]
    commands: dict[Command, Callable[[], Outcome]]
    place: Callable[[Any], None]
    row: Callable[[], int]
    scroll: Callable[[int], None]
    step: Callable[[int, int], None]

    def render(
        self, session: DebugSession, view: PaneState, rect: Rect, *, focused: bool = False
    ) -> PaneDrawing:
        del session, view, focused  # The wrapped drawing code reads them from the view.
        return PaneDrawing(self.capture(self.draw, rect))

    def handle(
        self, command: Command, session: DebugSession, view: PaneState, rect: Rect
    ) -> Reply | None:
        del session, view, rect
        action = self.commands.get(command)
        if action is not None:
            action()
        return None

    def place_cursor(self, view: PaneState, payload: Any) -> None:
        del view
        self.place(payload)

    def cursor_row(self, session: DebugSession, view: PaneState, rect: Rect) -> int:
        del session, view, rect
        return self.row()

    def scroll_to_row(self, session: DebugSession, view: PaneState, rect: Rect, row: int) -> None:
        del session, view, rect
        self.scroll(row)

    def move(
        self, session: DebugSession, view: PaneState, rect: Rect, *, dx: int = 0, dy: int = 0
    ) -> None:
        del session, view, rect
        self.step(dx, dy)
