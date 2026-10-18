"""Pointer input: clicks, double clicks, the wheel and drags, over the last frame.

Every event is resolved against the :class:`~ipu_debug.tui.chrome.Hits` the
last frame recorded, so a click lands on what is on screen. A press on a
border starts a split drag, a press on a scroll track a scroll drag; both
last until the release. The help overlay and the editor are modal and take
every event while open. The pointer acts through the view's public methods,
named by :class:`Shell`.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from ipu_debug.tui import layout
from ipu_debug.tui.chrome import Hits
from ipu_debug.tui.help import HelpOverlay
from ipu_debug.tui.keymap import Command
from ipu_debug.tui.pane import Outcome, Rect
from ipu_debug.tui.prompt import Prompt
from ipu_debug.tui.screen import Input, MouseEvent
from ipu_debug.tui.workspace import DebugViewSession

DOUBLE_CLICK_SECONDS = 0.35
# One wheel notch moves this many lines, the usual terminal convention.
MOUSE_WHEEL_LINES = 3
# Which pane a click in a shared border cell focuses, first match wins.
CLICK_ORDER = ("pipeline", "xmem", "disassembly", "registers")
WHEEL_ORDER = ("pipeline", "xmem", "registers", "disassembly")
# Panes whose values a double click edits.
EDITABLE_PANES = ("registers", "pipeline", "xmem")


class Shell(Protocol):
    """What the pointer needs of the view."""

    session: DebugViewSession
    help: HelpOverlay
    prompt: Prompt
    hits: Hits
    rects: dict[str, Rect]
    redraw: bool

    def render(self) -> None: ...

    def screen_size(self) -> tuple[int, int]: ...

    def handle_key(self, key: Input) -> Outcome: ...

    def focus_pane(self, pane: str) -> None: ...

    def pane_command(self, command: Command) -> Outcome: ...

    def place_cursor(self, pane: str, payload: Any) -> None: ...

    def close_tab(self, pane: str, index: int) -> None: ...

    def select_tab(self, delta: int) -> None: ...

    def move_focused(self, *, dx: int = 0, dy: int = 0) -> None: ...

    def scroll_to_fraction(self, pane: str, fraction: float) -> None: ...

    def scroll_cursor_fraction(self, pane: str) -> float: ...


class Pointer:
    """The gestures in flight, and the handling of each pointer event."""

    def __init__(self, view: Shell) -> None:
        self.view = view
        # A press some terminals follow with a release the press already acted on.
        self.consume_release = False
        # ``(split, origin_y, origin_x, moved)`` while a border is held.
        self.drag: tuple[str, int, int, bool] | None = None
        # The pane whose scroll thumb is held, where it was grabbed and whether
        # it moved since.
        self.scroll_drag: str | None = None
        self.scroll_origin: tuple[int, float] | None = None
        self.scroll_drag_moved = False
        # ``(tab identity, payload, time)`` of the last value click, for double clicks.
        self.last_value_click: tuple[int, Any, float] | None = None

    def cancel(self) -> None:
        """Drop the gestures in flight."""
        self.drag = None
        self.scroll_drag = None
        self.scroll_origin = None
        self.last_value_click = None

    def handle(self, event: MouseEvent) -> Outcome:
        """Act on one pointer event; a ``DebugAction`` ends this stop."""
        view = self.view
        wheel = event.wheel * MOUSE_WHEEL_LINES
        if self._ignored(event, wheel):
            return None
        if view.redraw:
            view.render()
        if view.help.visible:
            return self._help(event, wheel)
        # The editor is modal: it consumes every pointer event, wheel included.
        if view.prompt.mode:
            return self._editor(event)
        if self.scroll_drag is not None:
            self._continue_scroll_drag(event.y, event.release)
            return None
        if self.drag is not None:
            return self._continue_drag(event.y, event.x, event.release)
        if wheel:
            self.last_value_click = None
            view.redraw = True
            self._wheel(event.y, event.x, wheel, horizontal=event.shift)
            return None
        if event.pressed:
            return self._press(event)
        return None

    def _ignored(self, event: MouseEvent, wheel: int) -> bool:
        """Bare motion with nothing held, or the release a press already acted on."""
        if event.release and self.consume_release:
            self.consume_release = False
            return True
        if event.pressed:
            self.consume_release = False
        held = self.drag or self.scroll_drag or self.view.help.drag
        return not (event.pressed or event.release or wheel or held)

    def _help(self, event: MouseEvent, wheel: int) -> Outcome:
        view = self.view
        view.redraw = True
        key = view.help.mouse(event, wheel, view.hits)
        if key is None:
            return None
        # Some terminals synthesize clicks; others deliver down/up. Close on
        # down and consume its up event after dismissal.
        self.consume_release = event.pressed and not event.release
        return view.handle_key(key)

    def _editor(self, event: MouseEvent) -> Outcome:
        view = self.view
        view.redraw = True
        if event.pressed:
            for rect, key in view.hits.overlay:
                if rect.contains(event.y, event.x):
                    return view.handle_key(key)
            if view.hits.editor_field is not None:
                rect, start = view.hits.editor_field
                if rect.contains(event.y, event.x):
                    view.prompt.cursor = min(len(view.prompt.text), start + event.x - rect.x)
        return None

    def _continue_scroll_drag(self, y: int, released: bool) -> None:
        pane = self.scroll_drag
        track = self.view.hits.scrollbars.get(pane) if pane is not None else None
        if pane is not None and track is not None:
            origin_y, fraction = self.scroll_origin or (track.y, 0.0)
            delta = y - origin_y
            if delta or self.scroll_drag_moved:
                self.scroll_drag_moved = True
                self.view.scroll_to_fraction(pane, fraction + delta / max(1, track.height - 1))
                self.view.redraw = True
        if released:
            self.scroll_drag = None
            self.scroll_origin = None

    def _press(self, event: MouseEvent) -> Outcome:
        view = self.view
        y, x = event.y, event.x
        view.redraw = True
        for rect, key in view.hits.actions:
            if rect.contains(y, x):
                return view.handle_key(key)
        if event.press:
            for pane, track in view.hits.scrollbars.items():
                if track.contains(y, x):
                    self._grab_scrollbar(pane, track, y, x)
                    return None
            split = layout.split_at(view.session, view.rects, view.screen_size(), y, x)
            if split is not None:
                self.drag = (split, y, x, False)
                return None
        self._click(y, x, double=event.double)
        return None

    def _grab_scrollbar(self, pane: str, track: Rect, y: int, x: int) -> None:
        """Start a scroll drag; a press off the thumb first jumps there."""
        view = self.view
        view.focus_pane(pane)
        self.scroll_drag = pane
        self.scroll_drag_moved = False
        self.last_value_click = None
        thumb = view.hits.thumbs.get(pane)
        if thumb is None or not thumb.contains(y, x):
            view.scroll_to_fraction(pane, (y - track.y) / max(1, track.height - 1))
        self.scroll_origin = (y, view.scroll_cursor_fraction(pane))

    def _wheel(self, y: int, x: int, delta: int, *, horizontal: bool = False) -> None:
        view = self.view
        pane = next((name for name in WHEEL_ORDER if view.rects[name].contains(y, x)), None)
        if pane is None:
            return
        view.session.focus = pane
        if y == view.rects[pane].y + 1:
            view.select_tab(1 if delta > 0 else -1)
        elif horizontal:
            view.move_focused(dx=delta)
        else:
            view.move_focused(dy=delta)

    def _continue_drag(self, y: int, x: int, released: bool) -> Outcome:
        split, origin_y, origin_x, moved = self.drag or ("", 0, 0, False)
        moved = moved or (y, x) != (origin_y, origin_x)
        if moved:
            self.view.redraw = True
            layout.drag_split_to(self.view.session, split, self.view.screen_size(), y, x)
        if released:
            self.drag = None
            if not moved:
                self.view.redraw = True
                # A press and release without motion is an ordinary click.
                self._click(origin_y, origin_x)
        else:
            self.drag = (split, origin_y, origin_x, moved)
        return None

    def _click(self, y: int, x: int, *, double: bool = False) -> None:
        """A click: on a tab control, a scroll track, a value, or a pane."""
        previous_click = self.last_value_click
        self.last_value_click = None
        if self._click_tab_bar(y, x) or self._click_scrollbar(y, x):
            return
        view = self.view
        for rect, pc in view.hits.values.get("disassembly", []):
            if rect.contains(y, x) and x < rect.x + 2:
                # The code gutter toggles the breakpoint.
                view.focus_pane("disassembly")
                view.place_cursor("disassembly", pc)
                view.pane_command(Command.TOGGLE_BREAKPOINT)
                return
        for pane, hits in view.hits.values.items():
            for rect, payload in hits:
                if rect.contains(y, x):
                    view.focus_pane(pane)
                    view.place_cursor(pane, payload)
                    if pane in EDITABLE_PANES:
                        self._maybe_edit(pane, payload, previous_click, double=double)
                    return
        for pane in CLICK_ORDER:
            if view.rects[pane].contains(y, x):
                view.focus_pane(pane)
                return

    def _click_tab_bar(self, y: int, x: int) -> bool:
        """A click on a scroll marker, a tab's close box, a tab or ``[+]``."""
        view, session = self.view, self.view.session
        for rect, pane, index in view.hits.tab_scrolls:
            if rect.contains(y, x):
                view.focus_pane(pane)
                session.panes[pane].select(index)
                return True
        for pane, hits in view.hits.tab_closes.items():
            for rect, index in hits:
                if rect.contains(y, x):
                    session.focus = pane
                    view.close_tab(pane, index)
                    return True
        return self._click_tab(y, x)

    def _click_tab(self, y: int, x: int) -> bool:
        """A click on a tab selects it; one on ``[+]`` opens the add-tab editor."""
        view, session = self.view, self.view.session
        for pane, hits in view.hits.tabs.items():
            for rect, index in hits:
                if rect.contains(y, x):
                    session.focus = pane
                    session.panes[pane].select(index)
                    session.set_message(f"Selected {session.panes[pane][index].title}")
                    return True
        for pane, rect in view.hits.adds.items():
            if rect.contains(y, x):
                session.focus = pane
                view.pane_command(Command.ADD_TAB)
                return True
        return False

    def _click_scrollbar(self, y: int, x: int) -> bool:
        """A click on a scroll track jumps there; on the thumb it only focuses."""
        view = self.view
        for pane, rect in view.hits.scrollbars.items():
            if rect.contains(y, x):
                view.session.focus = pane
                thumb = view.hits.thumbs.get(pane)
                if thumb is None or not thumb.contains(y, x):
                    view.scroll_to_fraction(pane, (y - rect.y) / max(1, rect.height - 1))
                return True
        return False

    def _maybe_edit(
        self,
        pane: str,
        payload: Any,
        previous: tuple[int, Any, float] | None,
        *,
        double: bool,
    ) -> None:
        """A second click on the same value soon enough edits it."""
        now = time.monotonic()
        tab_set = self.view.session.panes[pane]
        identity = id(tab_set[tab_set.active])
        if double or (
            previous is not None
            and previous[:2] == (identity, payload)
            and now - previous[2] <= DOUBLE_CLICK_SECONDS
        ):
            self.view.pane_command(Command.EDIT_VALUE)
        else:
            self.last_value_click = (identity, payload, now)
