"""The XMEM pane on its own: session and tabs in, lines out.

No screen and no view: every test builds a ``DebugSession`` on a real state,
renders ``XmemPane`` into a rectangle and reads the lines it returns.
"""

from __future__ import annotations

import struct

import pytest

from ipu_debug.format import VISIBLE_ALPHABET
from ipu_debug.session import XMEM_FORMATS, DebugSession, XmemRequest
from ipu_debug.tui.draw import UNICODE_GLYPHS, Palette, Theme
from ipu_debug.tui.keymap import Command
from ipu_debug.tui.pane import Rect
from ipu_debug.tui.panes.xmem import XMEM_NON_CONTENT_ROWS, XmemPane, XmemTab
from ipu_debug.tui.tabs import TabSet
from ipu_emu.ipu import xmem_row_size_bytes
from ipu_emu.ipu_state import IpuState

RECT = Rect(10, 0, 12, 80)


def make(
    request: XmemRequest, data: bytes = b"", address: int = 0
) -> tuple[XmemPane, DebugSession, TabSet[XmemTab]]:
    state = IpuState()
    session = DebugSession(state)
    if data:
        state.xmem.write_address(address, data)
    pane = XmemPane(Theme(Palette(), UNICODE_GLYPHS))
    return pane, session, TabSet([XmemTab(request)])


def rows(lines) -> dict[int, str]:
    """Each screen row as it reads once every line is laid down."""
    grid: dict[int, list[str]] = {}
    for line in lines:
        row = grid.setdefault(line.y, [" "] * (RECT.x + RECT.width))
        for offset, character in enumerate(line.text):
            row[line.x + offset] = character
    return {y: "".join(row).rstrip() for y, row in grid.items()}


def text(drawing) -> str:
    return "\n".join(rows(drawing.lines).values())


@pytest.mark.parametrize(
    ("fmt", "data", "shown"),
    [
        ("hex", bytes([0x5A, 0x01]), "5a 01"),
        ("int8", bytes([0xFF, 0x7F]), "-1"),
        ("u8", bytes([0xFF, 0x07]), "255"),
        ("cell16", bytes([ord("A"), 0x0F]), VISIBLE_ALPHABET[ord("A")]),
        ("u32", struct.pack("<I", 4_000_000_000), "4000000000"),
        ("f32", struct.pack("<f", 1.5), "1.5"),
    ],
)
def test_every_format_renders_its_values(fmt, data, shown):
    pane, session, tabs = make(XmemRequest("byte", "0", "0", 4, fmt), data)

    drawing = pane.render(session, tabs, RECT, focused=True)

    assert shown in text(drawing)
    assert f"0x00000000 {UNICODE_GLYPHS.separator} " in text(drawing)
    assert fmt in rows(drawing.lines)[RECT.y]
    assert drawing.hits and drawing.hits[0][1] == (1, 0)


def test_the_formats_tested_are_every_format():
    tested = {"hex", "int8", "u8", "cell16", "u32", "f32"}
    assert tested == set(XMEM_FORMATS)


def test_a_cell16_title_summarises_its_colours():
    cell = struct.pack("<H", ord("A") | 0x2 << 8 | 0x5 << 12)
    pane, session, tabs = make(XmemRequest("byte", "0", "0", 1, "cell16"), cell)

    drawing = pane.render(session, tabs, RECT)

    assert "fg=2 bg=5" in rows(drawing.lines)[RECT.y]


def test_a_window_crossing_a_row_boundary_marks_the_new_row():
    row_bytes = xmem_row_size_bytes(IpuState())
    request = XmemRequest("byte", str(row_bytes - 8), "0", 16, "hex")
    pane, session, tabs = make(request, bytes(range(1, 17)), row_bytes - 8)

    drawing = pane.render(session, tabs, RECT, focused=True)
    shown = text(drawing)

    assert "XMEM row 0 [scale 2^-127] starts at byte address 0x00000000" in shown
    assert "XMEM row 1 [scale 2^-127] starts at byte address 0x00000080" in shown
    assert "---" not in shown
    # The first row's eight values and the second row's eight, on two lines.
    value_lines = {hit[1][0] for hit in drawing.hits}
    assert len(value_lines) == 2
    assert len(drawing.hits) == 16


def test_a_narrow_pane_drops_values_rather_than_clipping_them():
    pane, session, tabs = make(XmemRequest("row", "0", "0", 16, "hex"), bytes(range(16)))
    narrow = Rect(0, 0, 8, 30)

    drawing = pane.render(session, tabs, narrow, focused=True)

    assert all(line.x + len(line.text) <= narrow.x + narrow.width for line in drawing.lines)
    assert 0 < len(drawing.hits) < 16
    # Moving the cursor past the edge scrolls the line sideways.
    for _ in range(15):
        pane.handle(Command.MOVE_RIGHT, session, tabs, narrow)
    drawing = pane.render(session, tabs, narrow, focused=True)
    assert tabs[0].horizontal_scroll > 0
    assert (1, 15) in {payload for _, payload in drawing.hits}


def test_a_stale_generation_reads_memory_again():
    pane, session, tabs = make(XmemRequest("row", "0", "0", 4, "hex"), b"\x11")
    assert "11 00" in text(pane.render(session, tabs, RECT))

    session.state.xmem.write_address(0, b"\x22")  # Behind the session's back.
    assert "11 00" in text(pane.render(session, tabs, RECT))

    session.write_xmem(1, b"\x33")  # Through the session: the generation moves.
    assert "22 33" in text(pane.render(session, tabs, RECT))


def test_an_unresolvable_request_shows_its_error():
    pane, session, tabs = make(XmemRequest("byte", "1", "0", 1, "u32"))

    drawing = pane.render(session, tabs, RECT)

    assert "Error:" in text(drawing) and "aligned" in text(drawing)
    assert drawing.scroll is None


def test_an_empty_pane_says_how_to_add_a_tab():
    pane, session, _ = make(XmemRequest("row", "0", "0", 1, "hex"))

    drawing = pane.render(session, TabSet(), RECT)

    assert "No tabs. Press F2" in text(drawing)


def test_the_focused_pane_scrolls_to_its_cursor():
    pane, session, tabs = make(XmemRequest("row", "0", "0", 1024, "hex"))
    content = RECT.height - XMEM_NON_CONTENT_ROWS

    pane.handle(Command.JUMP_LAST, session, tabs, RECT)
    drawing = pane.render(session, tabs, RECT, focused=True)

    first, visible, total = drawing.scroll
    assert visible == content
    assert first + visible == total
    assert f"lines {first + 1}-{total}/{total}" in rows(drawing.lines)[RECT.y + RECT.height - 1]


def test_cycling_the_format_keeps_the_byte_range():
    pane, session, tabs = make(XmemRequest("row", "0", "0", 8, "hex"))

    reply = pane.handle(Command.CYCLE_FORMAT, session, tabs, RECT)

    assert reply is not None and reply.message == "XMEM format: int8"
    assert tabs[0].request == XmemRequest("row", "0", "0", 8, "int8")


def test_the_tab_editor_validates_before_adding():
    pane, session, tabs = make(XmemRequest("row", "0", "0", 8, "hex"))

    reply = pane.handle(Command.ADD_TAB, session, tabs, RECT)
    assert reply is not None and reply.editor is not None
    assert reply.editor.text == f"row 0 0 {xmem_row_size_bytes(session.state)} hex"

    rejected = reply.editor.submit("byte 1 0 1 f32")
    assert rejected.error and "aligned" in (rejected.message or "")
    assert len(tabs) == 1

    accepted = reply.editor.submit("byte 4 0 2 u32")
    assert not accepted.error
    assert len(tabs) == 2 and tabs.current().title == "byte:4+0 u32"


def test_the_value_editor_writes_through_the_session():
    pane, session, tabs = make(XmemRequest("row", "0", "0", 4, "hex"))
    pane.render(session, tabs, RECT, focused=True)
    pane.handle(Command.MOVE_RIGHT, session, tabs, RECT)
    generation = session.generation

    reply = pane.handle(Command.EDIT_VALUE, session, tabs, RECT)
    assert reply is not None and reply.editor is not None
    assert reply.editor.title == "Edit XMEM 0x00000001"
    assert reply.editor.submit("0x7f").message == "Updated XMEM 0x00000001"

    assert session.generation > generation
    assert "00 7f" in text(pane.render(session, tabs, RECT))
