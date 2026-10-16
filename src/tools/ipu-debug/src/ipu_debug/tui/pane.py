"""The interface one TUI pane implements, and the shapes it trades in.

A pane renders its tabs into a rectangle and handles the commands aimed at
it. The view owns layout, the frame, the footer and the overlays; the panes
own what is inside their frame.

Today each pane is a :class:`LegacyPane`, which wraps the drawing and
command code still living in ``CursesDebugView``. E08-09 (XMEM), E08-10
(registers) and E08-11 (disassembly and pipeline) replace each wrapper with a
pane module of its own; the golden screen tests in ``test_debug_tui`` prove
those moves change nothing on screen.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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


# A pane's persistent view state: its tabs.
PaneState = TabSet[Any]
# What handling a command leaves for the run: an execution action, or None to
# stay stopped.
Outcome = DebugAction | None


class Pane(Protocol):
    """One pane of the debugger."""

    id: PaneId

    def render(self, session: DebugSession, view: PaneState, rect: Rect) -> list[Line]:
        """The lines that draw *view* inside *rect* (frame included)."""
        ...

    def handle(self, command: Command, session: DebugSession, view: PaneState) -> Outcome:
        """Carry out *command* on *view*."""
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
    """

    id: PaneId
    draw: Callable[[Rect], None]
    capture: Callable[[Callable[[Rect], None], Rect], list[Line]]
    commands: dict[Command, Callable[[], Outcome]]

    def render(self, session: DebugSession, view: PaneState, rect: Rect) -> list[Line]:
        del session, view  # The wrapped drawing code reads them from the view.
        return self.capture(self.draw, rect)

    def handle(self, command: Command, session: DebugSession, view: PaneState) -> Outcome:
        del session, view
        action = self.commands.get(command)
        return action() if action is not None else None
