"""The help overlay: every documented control, scrollable, drawn over the panes.

Its lines come from ``keymap.help_lines``, so the overlay cannot drift from
the keys. It is modal: while it is open it takes every key and pointer event.
Its state is per stop, so help never survives a step.
"""

from __future__ import annotations

from ipu_debug.tui.chrome import BORDER_VERTICAL, Hits, draw_overlay_frame, overlay_rect
from ipu_debug.tui.draw import Canvas
from ipu_debug.tui.keymap import help_lines
from ipu_debug.tui.pane import Rect
from ipu_debug.tui.screen import Input, Key, MouseEvent

HELP_WIDTH = 78
# Below this overlay width the footer names fewer keys.
COMPACT_HELP_WIDTH = 70
TITLE = "Debugger controls"


class HelpOverlay:
    """The help overlay's visibility, scroll position and scrollbar drag."""

    def __init__(self) -> None:
        self.visible = False
        self.scroll = 0
        self.max_scroll = 0
        self.track: Rect | None = None
        self.thumb: Rect | None = None
        # ``(pointer y, scroll)`` when the thumb was grabbed.
        self.drag: tuple[int, int] | None = None

    def open(self) -> None:
        self.visible = True
        self.scroll = 0

    def close(self) -> None:
        self.visible = False
        self.drag = None
        self.scroll = 0

    def scroll_to_top(self) -> None:
        self.scroll = 0

    def scroll_by(self, delta: int) -> None:
        """Scroll *delta* lines; the next draw clamps to the end."""
        self.scroll = max(0, self.scroll + delta)

    def scroll_to_end(self, columns: int) -> None:
        width = min(columns - 4, HELP_WIDTH)
        self.scroll = len(help_lines(width - 4))

    def draw(self, canvas: Canvas, size: tuple[int, int], hits: Hits) -> None:
        """Draw the overlay on a *size* ``(rows, columns)`` screen."""
        colors = canvas.colors
        width = min(size[1] - 4, HELP_WIDTH)
        lines = help_lines(width - 5)
        rect = overlay_rect(size, HELP_WIDTH, len(lines) + 4)
        draw_overlay_frame(canvas, rect, [TITLE])
        canvas.add(rect.y, rect.x + rect.width - 5, "[x]", colors.key)
        hits.overlay.append((Rect(rect.y, rect.x + rect.width - 6, 1, 5), Key.ESCAPE))
        visible = max(1, rect.height - 3)
        self.max_scroll = max(0, len(lines) - visible)
        self.scroll = max(0, min(self.scroll, self.max_scroll))
        self.track = None
        self.thumb = None
        if self.max_scroll:
            self._draw_scrollbar(canvas, rect, visible, len(lines))
        for row, (text, kind) in enumerate(lines[self.scroll : self.scroll + visible]):
            style = colors.current if kind else 0
            canvas.add(rect.y + 1 + row, rect.x + 2, text, style, rect.width - 4)
        footer = (
            "Esc Close / Wheel Scroll"
            if rect.width < COMPACT_HELP_WIDTH
            else "Esc / ? Close   Up Down PgUp PgDn Scroll"
        )
        if len(lines) > visible:
            last = min(len(lines), self.scroll + visible)
            footer += f"   {self.scroll + 1}-{last}/{len(lines)}"
        canvas.add(
            rect.y + rect.height - 1,
            max(rect.x + 1, rect.x + rect.width - 2 - len(footer) - 2),
            f" {footer} ",
            colors.border,
            rect.width - 2,
        )

    def _draw_scrollbar(self, canvas: Canvas, rect: Rect, visible: int, total: int) -> None:
        colors, glyphs = canvas.colors, canvas.glyphs
        track = Rect(rect.y + 1, rect.x + rect.width - 2, visible, 1)
        size = max(1, round(visible * visible / total))
        start = round((visible - size) * self.scroll / self.max_scroll)
        self.track = track
        self.thumb = Rect(track.y + start, track.x, size, 1)
        for row in range(visible):
            on_thumb = start <= row < start + size
            glyph = glyphs.thumb if on_thumb else glyphs.box[BORDER_VERTICAL]
            canvas.add(track.y + row, track.x, glyph, colors.key if on_thumb else colors.border, 1)

    def mouse(self, event: MouseEvent, wheel: int, hits: Hits) -> Input | None:
        """Scroll, drag or page the overlay; the key of an overlay button pressed."""
        y, x = event.y, event.x
        pressed, released = event.pressed, event.release
        if wheel:
            self.scroll = max(0, min(self.max_scroll, self.scroll + wheel))
            return None
        if self.drag is not None and self.track and self.thumb:
            origin_y, origin_scroll = self.drag
            travel = max(1, self.track.height - self.thumb.height)
            moved = origin_scroll + round((y - origin_y) * self.max_scroll / travel)
            self.scroll = max(0, min(self.max_scroll, moved))
            if released:
                self.drag = None
            return None
        if pressed or released:
            for hit, key in hits.overlay:
                if hit.contains(y, x):
                    return key
        if pressed and self.track and self.thumb and self.track.contains(y, x):
            if self.thumb.contains(y, x):
                self.drag = (y, self.scroll)
            else:
                delta = self.track.height * (-1 if y < self.thumb.y else 1)
                self.scroll = max(0, min(self.max_scroll, self.scroll + delta))
        return None
