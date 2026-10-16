"""The open editor: the one-line prompt the view shows for a pane's ``Editor``.

A pane asks for an editor by replying with one (``ipu_debug.tui.editor``);
the view opens it here. The prompt holds the text being typed and its cursor,
draws itself as a modal overlay and hands the text to the editor's
``submit`` on Enter. While it is open it takes every key and pointer event.
"""

from __future__ import annotations

from ipu_debug.tui.chrome import Hits, draw_overlay_frame, overlay_rect
from ipu_debug.tui.draw import Canvas
from ipu_debug.tui.pane import Editor, Rect, Reply
from ipu_debug.tui.screen import Input, Key

EDITOR_WIDTH = 76
EDITOR_HEIGHT = 7


class Prompt:
    """The editor a pane opened (if any), the text typed into it and its cursor."""

    def __init__(self) -> None:
        self.editor: Editor | None = None
        self.text = ""
        self.cursor = 0
        # Where the last draw put the overlay.
        self.rect = Rect(0, 0, 0, 0)

    @property
    def mode(self) -> str | None:
        """The open editor's mode (``add``, ``edit``, ``value``, ``scalar``), or None."""
        return self.editor.mode if self.editor is not None else None

    @property
    def pane(self) -> str | None:
        return self.editor.pane if self.editor is not None else None

    @property
    def register(self) -> str | None:
        """The scalar being edited, as ``DebugSession.set_register`` names it."""
        return self.editor.register if self.editor is not None else None

    def open(self, editor: Editor) -> None:
        self.editor = editor
        self.text = editor.text
        self.cursor = len(editor.text)

    def close(self) -> None:
        self.editor = None
        self.text = ""
        self.cursor = 0

    def move(self, position: int) -> None:
        """Put the text cursor at *position*, inside the text."""
        self.cursor = max(0, min(len(self.text), position))

    def type(self, key: Input) -> None:
        """Typing: insert a printable character, or delete one."""
        if key is Key.DELETE and self.cursor < len(self.text):
            self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]
        elif key is Key.BACKSPACE and self.cursor > 0:
            self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
            self.cursor -= 1
        elif isinstance(key, str) and key.isprintable():
            self.text = self.text[: self.cursor] + key + self.text[self.cursor :]
            self.cursor += len(key)

    def submit(self) -> Reply:
        """Hand the text to the editor; an error reply keeps it open."""
        assert self.editor is not None
        reply = self.editor.submit(self.text)
        if not reply.error:
            self.close()
        return reply

    def draw(
        self, canvas: Canvas, size: tuple[int, int], error: str | None, hits: Hits
    ) -> tuple[int, int]:
        """Draw the overlay with *error* under the prompt; where the text cursor is."""
        assert self.editor is not None
        colors = canvas.colors
        rect = overlay_rect(size, EDITOR_WIDTH, EDITOR_HEIGHT)
        x, y, width = rect.x, rect.y, rect.width
        self.rect = rect
        draw_overlay_frame(canvas, rect, [self.editor.title])
        first_hint, second_hint = self.editor.hints
        canvas.add(y + 1, x + 2, first_hint, colors.border, width - 4)
        canvas.add(y + 2, x + 2, second_hint, colors.border, width - 4)
        # Matches the line-oriented debugger's own prompt.
        label = f"{self.editor.pane} >>> "
        canvas.add(y + 4, x + 2, label, colors.current, width - 4)
        available = max(0, width - 4 - len(label))
        visible_start = max(0, self.cursor - available + 1)
        visible = self.text[visible_start : visible_start + available]
        canvas.add(y + 4, x + 2 + len(label), visible, max_width=available)
        hits.editor_field = (Rect(y + 4, x + 2 + len(label), 1, available), visible_start)
        if error is not None:
            canvas.add(y + 5, x + 2, error, colors.error, width - 4)
        hint = f"Enter Apply {canvas.glyphs.separator} Esc Cancel"
        bottom = rect.y + rect.height - 1
        hint_x = max(rect.x + 1, rect.x + rect.width - 4 - len(hint))
        canvas.add(bottom, hint_x, f" {hint} ", colors.border, rect.width - 2)
        hits.overlay.extend(
            [
                (Rect(bottom, hint_x + 1, 1, len("Enter Apply")), Key.ENTER),
                (Rect(bottom, hint_x + 1 + hint.index("Esc"), 1, len("Esc Cancel")), Key.ESCAPE),
            ]
        )
        cursor_x = x + 2 + len(label) + self.cursor - visible_start
        return y + 4, min(x + width - 2, cursor_x)
