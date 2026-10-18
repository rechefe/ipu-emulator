"""Where each pane goes: the splits, their limits and the rectangles they give.

The screen is a header row, two rows of panes (pipeline and disassembly on
top, XMEM and the registers below) and a footer row. Adjacent panes share a
border column or row. The user's split sizes live on the
:class:`~ipu_debug.tui.workspace.DebugViewSession` as preferences: each is
clamped to the terminal on every layout but kept as chosen, so a terminal
that shrinks and grows again gets them back. A terminal below the minimum
shows one pane, the focused one; below the compact minimum, none.
"""

from __future__ import annotations

from ipu_debug.tui.keymap import PANE_FOCUS_ORDER
from ipu_debug.tui.pane import Rect
from ipu_debug.tui.panes import registers as registers_pane
from ipu_debug.tui.panes.registers import (
    DEFAULT_REGISTER_ROWS,
    REGISTER_GRID_ROWS,
    REGISTER_NON_CONTENT_ROWS,
)
from ipu_debug.tui.workspace import DebugViewSession

MIN_TERMINAL_ROWS = 24
MIN_TERMINAL_COLUMNS = 80
MIN_COMPACT_ROWS = 12
MIN_COMPACT_COLUMNS = 48
HEADER_ROWS = 1
FOOTER_ROWS = 1
MIN_DISASSEMBLY_COLUMNS = 30
MIN_XMEM_COLUMNS = 38
MIN_PIPELINE_COLUMNS = 42
# The top panes default to the height of the register grid's middle shape and
# never shrink below its smallest.
TOP_PANE_ROWS = DEFAULT_REGISTER_ROWS + REGISTER_NON_CONTENT_ROWS - 1
MIN_TOP_PANE_ROWS = min(REGISTER_GRID_ROWS) + REGISTER_NON_CONTENT_ROWS
MIN_MEMORY_ROWS = 6
MIN_REGISTER_COLUMNS = 20
PANE_RESIZE_STEP = 2

Layout = dict[str, Rect]


def single_pane(session: DebugViewSession, rows: int, columns: int) -> bool:
    """Whether only the focused pane is shown (maximized, or a small terminal)."""
    return session.maximized or rows < MIN_TERMINAL_ROWS or columns < MIN_TERMINAL_COLUMNS


def register_columns(session: DebugViewSession, columns: int, memory_rows: int) -> int:
    """The register pane's width, clamped without discarding the preferred size.

    Without an override the LR/CR grid takes the width its current shape
    needs but never more than half the terminal, so a narrow terminal keeps
    the disassembly readable and the grid scrolls its columns instead. A
    taller pane therefore also returns width to the disassembly.
    """
    requested = session.register_pane_columns or min(
        max(registers_pane.natural_width(session.register_tab(), memory_rows), columns * 2 // 5),
        max(MIN_REGISTER_COLUMNS, columns // 2),
    )
    return max(MIN_REGISTER_COLUMNS, min(requested, columns - MIN_XMEM_COLUMNS))


def pipeline_columns(session: DebugViewSession, columns: int) -> int:
    """The pipeline pane's width, clamped without changing its preference."""
    requested = session.pipeline_pane_columns or max(MIN_PIPELINE_COLUMNS, columns // 2)
    return max(MIN_PIPELINE_COLUMNS, min(requested, columns - MIN_DISASSEMBLY_COLUMNS + 1))


def default_top_pane_rows(rows: int) -> int:
    """Split usable rows evenly, including the shared border."""
    return max(MIN_TOP_PANE_ROWS, (rows - HEADER_ROWS - FOOTER_ROWS + 1) // 2)


def top_pane_rows(session: DebugViewSession, rows: int) -> int:
    """The top panes' height, clamped without changing its preference."""
    requested = session.top_pane_rows or default_top_pane_rows(rows)
    available = rows - HEADER_ROWS - FOOTER_ROWS - MIN_MEMORY_ROWS + 1
    return max(MIN_TOP_PANE_ROWS, min(requested, available))


def memory_rows(session: DebugViewSession, rows: int) -> int:
    """The bottom panes' height."""
    return rows - HEADER_ROWS - FOOTER_ROWS - top_pane_rows(session, rows) + 1


def compute_layout(session: DebugViewSession, rows: int, columns: int) -> Layout | None:
    """Every region's rectangle, or None when the terminal is too small for any."""
    if rows < MIN_COMPACT_ROWS or columns < MIN_COMPACT_COLUMNS:
        return None
    bars = {
        "header": Rect(0, 0, HEADER_ROWS, columns),
        "status": Rect(rows - 1, 0, 1, columns),
        "keys": Rect(rows - 1, 0, 1, columns),
    }
    if single_pane(session, rows, columns):
        return {
            "header": bars["header"],
            session.focus: Rect(HEADER_ROWS, 0, rows - HEADER_ROWS - FOOTER_ROWS, columns),
            "status": bars["status"],
            "keys": bars["keys"],
        }
    top_y = HEADER_ROWS
    top_rows = top_pane_rows(session, rows)
    xmem_y = top_y + top_rows - 1
    bottom_rows = rows - FOOTER_ROWS - xmem_y
    register_width = register_columns(session, columns, bottom_rows)
    pipeline_width = pipeline_columns(session, columns)
    return {
        "header": bars["header"],
        "pipeline": Rect(top_y, 0, top_rows, pipeline_width),
        "disassembly": Rect(top_y, pipeline_width - 1, top_rows, columns - pipeline_width + 1),
        "xmem": Rect(xmem_y, 0, bottom_rows, columns - register_width + 1),
        "registers": Rect(xmem_y, columns - register_width, bottom_rows, register_width),
        "status": bars["status"],
        "keys": bars["keys"],
    }


def pane_rects(layout: Layout | None) -> dict[str, Rect]:
    """Each pane's rectangle; an empty one for a pane not shown."""
    return {pane: (layout or {}).get(pane, Rect(0, 0, 0, 0)) for pane in PANE_FOCUS_ORDER}


def resize_vertical_split(session: DebugViewSession, rows: int, columns: int, delta: int) -> None:
    """Move the focused row's divider by *delta* columns (positive is right)."""
    if single_pane(session, rows, columns):
        return
    if session.focus in ("disassembly", "pipeline"):
        session.pipeline_pane_columns = pipeline_columns(session, columns) + delta
        width = pipeline_columns(session, columns)
        session.pipeline_pane_columns = width
        session.set_message(f"Pipeline pane width: {width} columns")
        return
    bottom_rows = memory_rows(session, rows)
    session.register_pane_columns = register_columns(session, columns, bottom_rows) - delta
    width = register_columns(session, columns, bottom_rows)
    session.register_pane_columns = width
    session.set_message(f"{registers_pane.LABEL} pane width: {width} columns")


def resize_horizontal_split(session: DebugViewSession, rows: int, columns: int, delta: int) -> None:
    """Move the shared row between the top and bottom panes by *delta* rows."""
    if single_pane(session, rows, columns):
        return
    session.top_pane_rows = top_pane_rows(session, rows) + delta
    height = top_pane_rows(session, rows)
    session.top_pane_rows = height
    session.set_message(f"Top pane height: {height} rows")


def reset_splits(session: DebugViewSession) -> None:
    """Forget every preferred split size."""
    session.register_pane_columns = None
    session.pipeline_pane_columns = None
    session.top_pane_rows = None
    session.set_message("Pane sizes reset")


def split_at(
    session: DebugViewSession, rects: dict[str, Rect], size: tuple[int, int], y: int, x: int
) -> str | None:
    """The split whose border covers ``(y, x)``, if any; *size* is ``(rows, columns)``."""
    if single_pane(session, *size):
        return None
    if rects["xmem"].height and y == rects["xmem"].y:
        return "top_pane_rows"
    for split, rect in (
        ("register_pane_columns", rects["registers"]),
        ("pipeline_pane_columns", rects["disassembly"]),
    ):
        if x == rect.x and rect.y < y < rect.y + rect.height - 1:
            return split
    return None


def drag_split_to(
    session: DebugViewSession, split: str, size: tuple[int, int], y: int, x: int
) -> None:
    """Move *split* so its border is under the pointer at ``(y, x)``."""
    rows, columns = size
    if split == "top_pane_rows":
        session.top_pane_rows = y - HEADER_ROWS + 1
        height = top_pane_rows(session, rows)
        session.top_pane_rows = height
        session.set_message(f"Top pane height: {height} rows")
        return
    if split == "pipeline_pane_columns":
        session.pipeline_pane_columns = x + 1
        width = pipeline_columns(session, columns)
        session.pipeline_pane_columns = width
        session.set_message(f"Pipeline pane width: {width} columns")
        return
    session.register_pane_columns = columns - x
    width = register_columns(session, columns, memory_rows(session, rows))
    session.register_pane_columns = width
    session.set_message(f"{registers_pane.LABEL} pane width: {width} columns")
