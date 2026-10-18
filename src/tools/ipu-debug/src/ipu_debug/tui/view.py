"""The TUI's shell over the ``Screen`` seam: layout, focus, status, help, main loop.

The TUI is the application debugger. The deprecated line debugger is a
separate compatibility API and cannot be entered from this interface. The
terminal is reached only through ``ipu_debug.tui.screen``; keys reach their
handlers through ``ipu_debug.tui.keymap``. What each pane shows and does is
its own module's (``ipu_debug.tui.panes``); the view lays the panes out,
frames them, routes each key to the focused one and draws the header, the
footer and the overlays around them.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path

from ipu_debug.control import DebugControl
from ipu_debug.errors import TerminalUnavailableError
from ipu_debug.session import DebugSession
from ipu_debug.tui.chrome import (
    Frames,
    Hits,
    draw_footer,
    draw_header,
    draw_scroll_thumb,
    draw_tab_bar,
)
from ipu_debug.tui.draw import Canvas, Glyphs, Palette, Theme, detect_glyphs
from ipu_debug.tui.help import HelpOverlay
from ipu_debug.tui.keymap import (
    PANE_COMMANDS,
    PANE_FOCUS,
    PANE_FOCUS_ORDER,
    Command,
    Focus,
    PaneId,
    lookup,
)
from ipu_debug.tui.layout import (
    MIN_COMPACT_COLUMNS,
    MIN_COMPACT_ROWS,
    PANE_RESIZE_STEP,
    Layout,
    compute_layout,
    pane_rects,
    reset_splits,
    resize_horizontal_split,
    resize_vertical_split,
)
from ipu_debug.tui.pane import Outcome, Pane, Rect, Reply
from ipu_debug.tui.panes.disassembly import DISASSEMBLY_NON_CONTENT_ROWS, DisassemblyPane
from ipu_debug.tui.panes.pipeline import PIPELINE_NON_CONTENT_ROWS, PipelinePane
from ipu_debug.tui.panes.registers import RegistersPane
from ipu_debug.tui.panes.xmem import XMEM_NON_CONTENT_ROWS, XmemPane
from ipu_debug.tui.pointer import Pointer
from ipu_debug.tui.prompt import Prompt
from ipu_debug.tui.screen import Input, Key, MouseEvent, Screen, run_in_terminal, terminal_available
from ipu_debug.tui.workspace import DebugViewSession, new_debug_view_session
from ipu_emu.emulator import DebugAction
from ipu_emu.ipu import xmem_row_size_bytes
from ipu_emu.ipu_state import IpuState

# Panes paint in this order, so a later one wins any shared cell.
PANE_DRAW_ORDER: tuple[PaneId, ...] = ("disassembly", "registers", "xmem", "pipeline")
# Every vertically scrollable pane reserves its right-most content column for
# a thumb, so the bar never lands on a border shared with a neighbour. The
# track leaves out the pane's non-content rows.
SCROLLBAR_PANES = {
    "disassembly": DISASSEMBLY_NON_CONTENT_ROWS,
    "xmem": XMEM_NON_CONTENT_ROWS,
    "pipeline": PIPELINE_NON_CONTENT_ROWS,
}
# An upper bound on one coalescing pass, so a stuck terminal cannot starve the
# redraw entirely.
MAX_COALESCED_KEYS = 64
# Help pages by the screen's height less the overlay's frame and margins.
HELP_PAGE_MARGIN_ROWS = 6


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
        # Overlay state is per stop, so help never survives a step.
        self.help = HelpOverlay()
        self.prompt = Prompt()
        self.pointer = Pointer(self)
        # What the last frame drew that a pointer can act on.
        self.hits = Hits()
        self.rects: dict[str, Rect] = pane_rects(None)
        self.redraw = True
        self.panes: dict[str, Pane] = {
            "pipeline": PipelinePane(self.theme),
            "disassembly": DisassemblyPane(self.theme, cycle),
            "xmem": XmemPane(self.theme),
            "registers": RegistersPane(self.theme),
        }
        self.commands = self._command_table()

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
                self.pointer.cancel()
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

    # -- layout and drawing ---------------------------------------------------

    def screen_size(self) -> tuple[int, int]:
        return self.screen.size()

    def layout(self) -> Layout | None:
        """Every region's rectangle on the current screen, or None if it is too small."""
        return compute_layout(self.session, *self.screen.size())

    def render(self) -> None:
        """Draw one frame and record what a pointer can act on."""
        self.screen.clear()
        self.hits = Hits()
        self.help.track = None
        self.help.thumb = None
        rows, columns = self.screen.size()
        canvas = Canvas(self.theme, Rect(0, 0, rows, columns), self.screen.draw)
        layout = self.layout()
        self.rects = pane_rects(layout)
        if layout is None:
            self.pointer.drag = None
            self.pointer.scroll_drag = None
            message = (
                f"Terminal must be at least {MIN_COMPACT_COLUMNS}x"
                f"{MIN_COMPACT_ROWS}; current size is {columns}x{rows}."
            )
            canvas.add(max(0, rows // 2), max(0, (columns - len(message)) // 2), message)
            canvas.add(max(0, rows // 2 + 1), 0, "Resize the terminal or press q to quit.")
        else:
            self._draw_frame(canvas, layout)
        self.screen.refresh()
        self.redraw = False

    def _draw_frame(self, canvas: Canvas, layout: Layout) -> None:
        focus = self.session.focus
        # Adjacent panes share a border column or row, so every frame is
        # recorded first and painted once, as a single grid of junctions.
        frames = Frames()
        for pane in PANE_FOCUS_ORDER:
            if pane in layout:
                frames.box(layout[pane], self.colors, focused=focus == pane)
        frames.flush(canvas)
        self._draw_header(canvas, layout["header"])
        for pane_id in PANE_DRAW_ORDER:
            if pane_id in layout:
                tabs = self.session.panes[pane_id]
                rect = layout[pane_id]
                draw_tab_bar(canvas, pane_id, rect, tabs, focused=focus == pane_id, hits=self.hits)
                self._draw_pane(pane_id, rect)
        for pane, non_content_rows in SCROLLBAR_PANES.items():
            if pane in self.hits.scroll:
                rect = self.rects[pane]
                draw_scroll_thumb(
                    canvas, pane, rect, non_content_rows, focused=focus == pane, hits=self.hits
                )
        draw_footer(canvas, layout["keys"], focus, self.hits)
        if self.help.visible:
            self.help.draw(canvas, self.screen.size(), self.hits)
        elif self.prompt.mode:
            error = self.session.message if self.session.message_is_error else None
            y, x = self.prompt.draw(canvas, self.screen.size(), error, self.hits)
            self.screen.cursor(True, y, x)
        else:
            self.screen.cursor(False)

    def _draw_pane(self, pane_id: str, rect: Rect) -> None:
        """Paint one pane's content and keep what it reports for the pointer."""
        drawing = self.panes[pane_id].render(
            self.debug, self.session.panes[pane_id], rect, focused=self.session.focus == pane_id
        )
        for line in drawing.lines:
            self.screen.draw(line.y, line.x, line.text, line.style)
        if drawing.hits:
            self.hits.values.setdefault(pane_id, []).extend(drawing.hits)
        if drawing.scroll is not None:
            self.hits.scroll[pane_id] = drawing.scroll

    def _draw_header(self, canvas: Canvas, rect: Rect) -> None:
        mode = "wide" if self.state.codec.element_bytes != 1 else "normal"
        segments = [
            self.debug.kernel_name or "IPU DEBUG",
            "PAUSED",
            f"PC={self.state.program_counter:04d}",
            f"cycle={self.cycle}",
            self.control.stop_reason or "paused",
            f"{mode} / {xmem_row_size_bytes(self.state)}B rows",
        ]
        status = (self.session.message, self.session.message_is_error)
        draw_header(canvas, rect, segments, status, self.hits)

    # -- keys -----------------------------------------------------------------

    def handle_key(self, key: Input) -> Outcome:
        """Act on one input; a ``DebugAction`` ends this stop, None stays."""
        # A preceding queued event may have changed layout without a repaint.
        # Cursor arithmetic must use the new pane dimensions immediately.
        self.rects = pane_rects(self.layout())
        if isinstance(key, MouseEvent):
            return self.pointer.handle(key)
        self.redraw = True
        # A drag whose release never arrives - the pointer left the terminal,
        # say - must not turn every later mouse event into a resize.
        self.pointer.cancel()
        if key is Key.RESIZE:
            return None
        command = lookup(self._key_focus(), key)
        if command is not None:
            return self.commands[command]()
        if self.prompt.mode and not self.help.visible:
            self.prompt.type(key)
        return None

    def _key_focus(self) -> Focus:
        """The help overlay and the editor are modal; otherwise the focused pane."""
        if self.help.visible:
            return Focus.HELP
        if self.prompt.mode:
            return Focus.EDITOR
        return PANE_FOCUS[self.session.focus]

    def _command_table(self) -> dict[Command, Callable[[], Outcome]]:
        """The method that carries out each command."""

        def does(action: Callable[[], object]) -> Callable[[], Outcome]:
            def run() -> Outcome:
                action()
                return None

            return run

        def split(resize: Callable[..., None], delta: int) -> Callable[[], Outcome]:
            return does(lambda: resize(self.session, *self.screen.size(), delta))

        prompt, help_overlay = self.prompt, self.help
        table: dict[Command, Callable[[], Outcome]] = {
            Command.QUIT: self._quit,
            Command.CONTINUE: lambda: self._execution_action(DebugAction.CONTINUE),
            Command.STEP: lambda: self._execution_action(DebugAction.STEP),
            Command.SHOW_HELP: does(help_overlay.open),
            Command.RESET_SPLITS: does(lambda: reset_splits(self.session)),
            Command.NEXT_PANE: does(lambda: self._select_relative_pane(1)),
            Command.SPLIT_LEFT: split(resize_vertical_split, -PANE_RESIZE_STEP),
            Command.SPLIT_RIGHT: split(resize_vertical_split, PANE_RESIZE_STEP),
            Command.SPLIT_UP: split(resize_horizontal_split, -1),
            Command.SPLIT_DOWN: split(resize_horizontal_split, 1),
            Command.CLOSE_TAB: does(self._close_active_tab),
            Command.NEXT_TAB: does(lambda: self.select_tab(1)),
            Command.PREVIOUS_TAB: does(lambda: self.select_tab(-1)),
            Command.CANCEL: does(self._cancel),
            Command.EDITOR_LEFT: does(lambda: prompt.move(prompt.cursor - 1)),
            Command.EDITOR_RIGHT: does(lambda: prompt.move(prompt.cursor + 1)),
            Command.EDITOR_HOME: does(lambda: prompt.move(0)),
            Command.EDITOR_END: does(lambda: prompt.move(len(prompt.text))),
            Command.EDITOR_SUBMIT: self._submit_editor,
            Command.CLOSE_HELP: does(help_overlay.close),
            Command.HELP_UP: does(lambda: help_overlay.scroll_by(-1)),
            Command.HELP_DOWN: does(lambda: help_overlay.scroll_by(1)),
            Command.HELP_PAGE_UP: does(lambda: help_overlay.scroll_by(-self._help_page())),
            Command.HELP_PAGE_DOWN: does(lambda: help_overlay.scroll_by(self._help_page())),
            Command.HELP_TOP: does(help_overlay.scroll_to_top),
            Command.HELP_BOTTOM: does(lambda: help_overlay.scroll_to_end(self.screen.size()[1])),
        }
        for command, pane in (
            (Command.FOCUS_PIPELINE, "pipeline"),
            (Command.FOCUS_DISASSEMBLY, "disassembly"),
            (Command.FOCUS_XMEM, "xmem"),
            (Command.FOCUS_REGISTERS, "registers"),
        ):
            table[command] = does(lambda pane=pane: self.focus_pane(pane))
        for command in PANE_COMMANDS:
            table[command] = lambda command=command: self.pane_command(command)
        return table

    def _help_page(self) -> int:
        return max(1, self.screen.size()[0] - HELP_PAGE_MARGIN_ROWS)

    def pane_command(self, command: Command) -> Outcome:
        """Hand *command* to the focused pane and carry out its reply."""
        pane = self.session.focus
        reply = self.panes[pane].handle(
            command, self.debug, self.session.panes[pane], self.rects[pane]
        )
        return self._apply(reply)

    def _apply(self, reply: Reply | None) -> Outcome:
        """Open the editor a pane asked for, show its message, run its action."""
        if reply is None:
            return None
        if reply.editor is not None:
            self.prompt.open(reply.editor)
        if reply.message is not None:
            self.session.set_message(reply.message, error=reply.error)
        if reply.action is not None:
            return self._execution_action(reply.action)
        return None

    def _execution_action(self, action: DebugAction) -> DebugAction:
        self.session.commit_baseline(self.state)
        self.session.active = action in (DebugAction.CONTINUE, DebugAction.STEP)
        return action

    def _quit(self) -> Outcome:
        self.debug.halt()
        return self._execution_action(DebugAction.QUIT)

    def _cancel(self) -> None:
        """Esc: leave the editor, or let the focused pane cancel (the stage view)."""
        if self.prompt.mode:
            pane = self.prompt.pane
            self._clear_editor()
            label = "XMEM" if pane == "xmem" else pane
            self.session.set_message(f"{label} tab edit cancelled")
        else:
            self.pane_command(Command.CANCEL)

    def _submit_editor(self) -> Outcome:
        """Enter: the editor's pane applies the text; an error keeps it open."""
        reply = self.prompt.submit()
        if not reply.error:
            self.screen.cursor(False)
        return self._apply(reply)

    def _clear_editor(self) -> None:
        self.prompt.close()
        self.screen.cursor(False)

    # -- focus and tabs -------------------------------------------------------

    def focus_pane(self, pane: str) -> None:
        self.session.focus = pane
        self.session.set_message("Ready")

    def _select_relative_pane(self, delta: int) -> None:
        panes = list(PANE_FOCUS_ORDER)
        focus = self.session.focus
        current = panes.index(focus) if focus in panes else panes.index("xmem")
        self.focus_pane(panes[(current + delta) % len(panes)])

    def select_tab(self, delta: int) -> None:
        """Select the focused pane's tab *delta* places on."""
        pane = self.session.focus
        tab = self.session.panes[pane].step(delta)
        if tab is None:
            self.session.set_message(f"{pane} has no tabs")
            return
        self.session.set_message(f"Selected {tab.title}")

    def _close_active_tab(self) -> None:
        pane = self.session.focus
        tabs = self.session.panes[pane]
        if tabs:
            self.close_tab(pane, tabs.active)

    def close_tab(self, pane: str, index: int) -> None:
        tabs = self.session.panes[pane]
        if not 0 <= index < len(tabs):
            return
        tabs.close(index)
        self.session.set_message(f"Closed {pane} tab")

    # -- cursors --------------------------------------------------------------

    def place_cursor(self, pane: str, payload: object) -> None:
        """Put *pane*'s cursor on the value that was clicked."""
        self.panes[pane].place_cursor(self.debug, self.session.panes[pane], payload)

    def move_focused(self, *, dx: int = 0, dy: int = 0) -> None:
        """Move the focused pane's cursor by *dx* values and *dy* rows."""
        pane = self.session.focus
        self.panes[pane].move(self.debug, self.session.panes[pane], self.rects[pane], dx=dx, dy=dy)

    def scroll_cursor_fraction(self, pane: str) -> float:
        """Remember the selection at grab time, so a thumb press never jumps."""
        total = self.hits.scroll[pane][2]
        target = self.panes[pane].cursor_row(self.debug, self.session.panes[pane], self.rects[pane])
        return (target + 0.5) / max(1, total)

    def scroll_to_fraction(self, pane: str, fraction: float) -> None:
        """Jump the pane's cursor to a position along its scrollbar."""
        _, _, total = self.hits.scroll.get(pane, (0, 0, 0))
        if total <= 0:
            return
        target = max(0, min(total - 1, int(fraction * total)))
        self.panes[pane].scroll_to_row(
            self.debug, self.session.panes[pane], self.rects[pane], target
        )


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
        raise TerminalUnavailableError("curses is not available on this platform")
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
    except Exception:
        # A bug, not a user mistake (those are DebugErrors, shown on the
        # status line). curses has restored the terminal by now, so the
        # traceback can be kept and pointed at before the error goes on.
        session.active = False
        text = traceback.format_exc()
        try:
            path = _log_crash(text)
        except OSError:
            # No log file, but the error itself must still be the one raised.
            print(f"ipu-debug: the debugger failed:\n{text}", file=sys.stderr)
        else:
            print(f"ipu-debug: the debugger failed; traceback in {path}", file=sys.stderr)
        raise


def _log_crash(text: str) -> Path:
    """Write a crash traceback to ``$TMPDIR/ipu-debug-<pid>.log`` and return the path."""
    path = Path(tempfile.gettempdir()) / f"ipu-debug-{os.getpid()}.log"
    path.write_text(text, encoding="utf-8")
    return path
