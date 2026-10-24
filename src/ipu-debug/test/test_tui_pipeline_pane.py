"""The pipeline-register pane on its own: session and tabs in, lines out.

No screen and no view: every test builds a ``DebugSession`` on a real state,
renders ``PipelinePane`` into a rectangle and reads the lines it returns.
"""

from __future__ import annotations

import struct

import pytest
from emu_support import ElementFormat, make_state, new_state

from ipu_debug.format import VISIBLE_ALPHABET
from ipu_debug.session import DebugSession
from ipu_debug.tui.draw import UNICODE_GLYPHS, Palette, Theme
from ipu_debug.tui.keymap import Command
from ipu_debug.tui.pane import Rect
from ipu_debug.tui.panes.pipeline import (
    AAQ_ROW,
    PIPELINE_REGISTER_FORMATS,
    PIPELINE_REGISTERS,
    PipelinePane,
    PipelineTab,
    commit_baselines,
    default_tabs,
    read_register,
    value_grid,
)
from ipu_debug.tui.tabs import TabSet
from ipu_emu.descriptors import REGFILE_SCHEMA
from ipu_emu.emulator import run_until_complete
from ipu_emu.ipu_state import IpuState

RECT = Rect(1, 0, 12, 60)
WIDE = Rect(1, 0, 12, 140)
COLORS = Palette()


def make(
    tab: PipelineTab | None = None, state: IpuState | None = None
) -> tuple[PipelinePane, DebugSession, TabSet[PipelineTab]]:
    pane = PipelinePane(Theme(COLORS, UNICODE_GLYPHS))
    return pane, DebugSession(state or IpuState()), TabSet([tab or PipelineTab()])


def rows(drawing, rect: Rect = RECT) -> dict[int, str]:
    """Each row of *rect* as it reads once every line is laid down."""
    grid: dict[int, list[str]] = {}
    for line in drawing.lines:
        row = grid.setdefault(line.y, [" "] * (rect.x + rect.width))
        for offset, character in enumerate(line.text):
            row[line.x + offset] = character
    return {y: "".join(row).rstrip() for y, row in grid.items()}


def text(drawing, rect: Rect = RECT) -> str:
    return "\n".join(rows(drawing, rect).values())


# The row-metadata and LUT registers (ADR 0007) are no tabs of their own: the
# AAQ_ROW title shows the stored row's metadata.
_NOT_YET_SHOWN = frozenset({"r_meta", "r_cyclic_meta", "aaq_lut"})


def test_the_registers_are_every_logical_vector_register():
    expected = {
        descriptor.name
        for descriptor in REGFILE_SCHEMA
        if descriptor.is_vector
        and not descriptor.name.endswith("_wide_debug")
        and descriptor.name not in _NOT_YET_SHOWN
    }
    registers = {spec.backing for spec in PIPELINE_REGISTERS if spec.key != AAQ_ROW}
    assert registers == expected


@pytest.mark.parametrize("spec", PIPELINE_REGISTERS, ids=lambda spec: spec.key)
def test_every_register_renders_its_bytes_and_a_readout(spec):
    pane, session, tabs = make(PipelineTab(spec.key, "hex"))
    # AAQ_ROW's title names the row too: the readout needs a wider pane.
    rect = RECT if spec.key != AAQ_ROW else Rect(1, 0, 12, 80)

    drawing = pane.render(session, tabs, rect)

    total = len(read_register(session, spec))
    assert rows(drawing, rect)[rect.y].endswith(f"/{total}")
    assert "1 Pipeline Registers" in rows(drawing, rect)[rect.y]
    assert "\n   0000: " in text(drawing, rect)
    assert drawing.scroll is not None and drawing.hits[0][1] == 0


@pytest.mark.parametrize(
    ("fmt", "data", "shown"),
    [
        ("hex", b"\x5a\x01\x00\x00", "5a 01"),
        ("int8", b"\xff\x7f\x00\x00", "-1"),
        ("u8", b"\xff\x07\x00\x00", "255 007"),
        ("cell16", bytes([ord("A"), 0x0F, 0, 0]), VISIBLE_ALPHABET[ord("A")]),
        ("u32", struct.pack("<I", 1), "0000000001"),
        ("int32", struct.pack("<i", -5), "-5"),
        ("f32", struct.pack("<f", 1.5), "1.5"),
        ("bits", b"\x01\x00\x00\x80", "10000000000000000000000000000001"),
    ],
)
def test_every_format_renders_its_values(fmt, data, shown):
    state = IpuState()
    state.regfile.mutable("r")[:4] = data
    pane, session, tabs = make(PipelineTab("r0", fmt), state)

    drawing = pane.render(session, tabs, WIDE)

    assert shown in text(drawing, WIDE)
    assert fmt in rows(drawing, WIDE)[WIDE.y]


def test_the_formats_tested_are_every_format():
    assert set(PIPELINE_REGISTER_FORMATS) == {
        "hex",
        "int8",
        "u8",
        "cell16",
        "u32",
        "int32",
        "f32",
        "bits",
    }


def test_cycling_the_format_walks_every_format_and_resets_the_cursor():
    pane, session, tabs = make()
    tabs[0].cursor = 5

    seen = []
    for _ in PIPELINE_REGISTER_FORMATS:
        reply = pane.handle(Command.CYCLE_FORMAT, session, tabs, RECT)
        assert reply is not None
        seen.append(tabs[0].fmt)

    # An auto tab of R0 starts from its storage format, hex.
    assert seen == [*PIPELINE_REGISTER_FORMATS[1:], "hex"]
    assert reply.message == "R0 format: hex"
    assert tabs[0].cursor == 0


def test_cell16_keeps_each_cells_colours_and_groups_characters():
    state = IpuState()
    cells = (0 | (14 << 8) | (4 << 12), 1 | (2 << 8) | (3 << 12), 2 | (5 << 8) | (6 << 12))
    state.regfile.mutable("r")[:6] = struct.pack("<HHH", *cells)
    pane, session, tabs = make(PipelineTab("r0", "cell16"), state)
    seen: list[tuple[int, int]] = []

    def cell(foreground: int, background: int) -> int:
        seen.append((foreground, background))
        return 1 << 20

    pane.theme.colors = Palette()
    pane.theme.colors.cell = cell
    drawing = pane.render(session, tabs, WIDE)

    assert seen[:3] == [(14, 4), (2, 3), (5, 6)]
    characters = [line for line in drawing.lines if line.text in {"!", '"', "#"}]
    assert [line.x - characters[0].x for line in characters[:3]] == [0, 1, 3]
    # The cursor is on the first; the others keep their own colour pair alone.
    assert [line.style for line in characters[1:3]] == [1 << 20, 1 << 20]


def test_fp32_reads_the_wide_storage_in_the_arithmetic_format():
    state = new_state(fmt=ElementFormat.FP32)
    state.regfile.mutable("r")[0] = 0xFF
    state.regfile.mutable("r_wide_debug")[:4] = struct.pack("<f", 1.5)
    pane, session, tabs = make(state=state)

    drawing = pane.render(session, tabs, WIDE)

    assert "f32" in rows(drawing, WIDE)[WIDE.y]
    assert "1.5" in text(drawing, WIDE)
    assert value_grid(session, tabs[0], WIDE).item_count == 512 // 4


def test_wide_hex_groups_every_four_bytes():
    state = new_state(fmt=ElementFormat.FP32)
    state.regfile.mutable("r_wide_debug")[:8] = bytes(range(8))
    pane, session, tabs = make(PipelineTab("r0", "hex"), state)

    drawing = pane.render(session, tabs, WIDE)

    # The first line on the row is the offset prefix.
    xs = [line.x for line in drawing.lines if line.y == WIDE.y + 2][1:6]
    assert [b - a for a, b in zip(xs, xs[1:], strict=False)] == [3, 3, 3, 4]


def test_values_changed_since_the_baseline_are_marked():
    state = IpuState()
    pane, session, tabs = make(state=state)
    commit_baselines(session, tabs)
    state.regfile.mutable("r")[1] = 7

    drawing = pane.render(session, tabs, RECT)

    changed = [line for line in drawing.lines if line.style & COLORS.changed == COLORS.changed]
    assert [line.text for line in changed] == ["07"]


def test_the_cursor_is_highlighted_more_strongly_when_focused():
    state = IpuState()
    state.regfile.mutable("r")[0] = 0x12
    pane, session, tabs = make(state=state)

    focused = pane.render(session, tabs, RECT, focused=True)
    idle = pane.render(session, tabs, RECT)

    assert [line.style for line in focused.lines if line.x == 9 and line.y == RECT.y + 2] == [
        COLORS.selected
    ]
    assert [line.style for line in idle.lines if line.x == 9 and line.y == RECT.y + 2] == [
        COLORS.cursor
    ]


def test_a_value_too_wide_for_the_pane_is_clipped_with_a_marker():
    pane, session, tabs = make(PipelineTab("r0", "bits"))
    narrow = Rect(1, 0, 12, 40)

    drawing = pane.render(session, tabs, narrow)

    limit = narrow.x + narrow.width - 2
    assert all(hit.x + hit.width <= limit for hit, _ in drawing.hits)
    assert any(line.text.endswith(UNICODE_GLYPHS.ellipsis) for line in drawing.lines)


def test_arrows_move_by_values_and_lines_and_stop_at_the_ends():
    pane, session, tabs = make()
    short = Rect(1, 0, 6, 60)
    # 57 content columns hold the offset prefix and 16 two-digit values.
    assert value_grid(session, tabs[0], short).per_line == 16

    pane.handle(Command.MOVE_DOWN, session, tabs, short)
    pane.handle(Command.MOVE_RIGHT, session, tabs, short)
    assert tabs[0].cursor == 17
    pane.handle(Command.MOVE_UP, session, tabs, short)
    assert tabs[0].cursor == 1
    pane.handle(Command.MOVE_LEFT, session, tabs, short)
    pane.handle(Command.MOVE_LEFT, session, tabs, short)
    assert tabs[0].cursor == 0
    # A page is the three content rows of a six-row pane.
    pane.handle(Command.PAGE_DOWN, session, tabs, short)
    assert tabs[0].cursor == 48
    pane.handle(Command.JUMP_LAST, session, tabs, short)
    assert tabs[0].cursor == len(session.register("r").value) - 1
    pane.handle(Command.MOVE_DOWN, session, tabs, short)
    assert tabs[0].cursor == len(session.register("r").value) - 1
    pane.handle(Command.JUMP_FIRST, session, tabs, short)
    assert tabs[0].cursor == 0


def test_the_focused_pane_scrolls_to_its_cursor():
    pane, session, tabs = make()
    short = Rect(1, 0, 6, 60)
    pane.handle(Command.JUMP_LAST, session, tabs, short)

    idle = pane.render(session, tabs, short)
    drawing = pane.render(session, tabs, short, focused=True)

    assert idle.scroll is not None and idle.scroll[0] == 0
    first, visible, total = drawing.scroll
    assert first > 0 and first + visible == total
    assert pane.cursor_row(session, tabs, short) == total - 1
    pane.scroll_to_row(session, tabs, short, 0)
    assert tabs[0].cursor == 0


@pytest.mark.parametrize("fmt", ["bits", "hex", "u32", "cell16"])
@pytest.mark.parametrize("width", [42, 80])
def test_mask_slots_have_separators_and_keep_values_inside_one_slot(fmt, width):
    pane, session, tabs = make(PipelineTab("r_mask", fmt))
    rect = Rect(1, 0, 100, width)

    drawing = pane.render(session, tabs, rect)
    grid = value_grid(session, tabs[0], rect)

    assert grid.rows.count(None) == 7
    for index, first in enumerate(grid.rows):
        if first is None:
            assert grid.rows[index + 1] * grid.item_size % 16 == 0
        else:
            last_byte = (first + grid.per_line) * grid.item_size - 1
            assert first * grid.item_size // 16 == last_byte // 16
    separators = {line.y for line in drawing.lines if line.text == "------------"}
    assert len(separators) == 7
    assert all(hit.y not in separators for hit, _ in drawing.hits)
    pane.scroll_to_row(session, tabs, rect, len(grid.rows) - 1)
    assert tabs[0].cursor >= (7 * 16) // grid.item_size


@pytest.mark.parametrize("wide", [False, True])
def test_the_value_editor_writes_the_right_storage_and_keeps_its_neighbours(wide):
    state = new_state(fmt=ElementFormat.FP32 if wide else ElementFormat.NX8)
    pane, session, tabs = make(PipelineTab("r1", "int32", cursor=1), state)
    storage = "r_wide_debug" if wide else "r"
    buffer = state.regfile.mutable(storage)
    buffer[:] = b"\x55" * len(buffer)
    before = bytes(buffer)
    offset = state.regfile.descriptor(storage).size_bytes + 4

    reply = pane.handle(Command.EDIT_VALUE, session, tabs, RECT)
    assert reply is not None and reply.editor is not None
    assert reply.editor.title == "Edit R1[1]"
    assert reply.editor.submit("-123").message == "Updated R1[1]"

    assert bytes(buffer) == before[:offset] + struct.pack("<i", -123) + before[offset + 4 :]


def test_the_value_editor_refuses_a_value_out_of_range():
    pane, session, tabs = make(PipelineTab("r0", "hex"))

    reply = pane.handle(Command.EDIT_VALUE, session, tabs, RECT)
    assert reply is not None and reply.editor is not None
    refused = reply.editor.submit("256")

    assert refused.error
    assert session.state.regfile.view("r") == bytearray(256)


def test_the_tab_editor_takes_a_register_label_or_key():
    pane, session, tabs = make()

    reply = pane.handle(Command.ADD_TAB, session, tabs, RECT)
    assert reply is not None and reply.editor is not None and reply.editor.text == "R0"
    refused = reply.editor.submit("NOT_A_REGISTER")
    assert refused.error and "expected one of: R0, R1, R_C" in (refused.message or "")
    assert reply.editor.submit("r_acc").message == "pipeline tab added: R_ACC auto"
    reply = pane.handle(Command.EDIT_TAB, session, tabs, RECT)
    assert reply is not None and reply.editor is not None and reply.editor.text == "R_ACC"
    assert reply.editor.submit("MEM_BYPASS").message == "pipeline tab updated: MEM_BYPASS auto"
    assert [tab.register_key for tab in tabs] == ["r0", "mem_bypass"]


@pytest.mark.parametrize(
    ("mode", "r0", "aaq_row", "r_acc"),
    [
        (ElementFormat.NX8, "int8", "int8", "f32"),
        (ElementFormat.FP32, "f32", "f32", "f32"),
    ],
)
def test_fresh_tabs_cover_every_register_in_its_storage_format(mode, r0, aaq_row, r_acc):
    state = new_state(fmt=mode)

    formats = {tab.register_key: tab.fmt for tab in default_tabs(state, True)}

    assert list(formats) == [spec.key for spec in PIPELINE_REGISTERS]
    assert (formats["r0"], formats[AAQ_ROW], formats["r_acc"]) == (r0, aaq_row, r_acc)
    assert formats["r_mask"] == "bits"
    assert formats["mem_bypass"] == "hex"
    assert [tab.register_key for tab in default_tabs(state, False)] == ["r0"]


def test_aaq_row_shows_the_last_stored_row_and_its_metadata():
    # ACTIVATE.QUANTIZE identity stores R_ACC as one NX8 e4 row (CR14.L) at row CR2 = 3.
    state = make_state("ACTIVATE.QUANTIZE identity cr15.l cr14.l cr2 lr0;; HALT;;", cr={2: 3})
    pane, session, tabs = make(PipelineTab(AAQ_ROW, "hex"), state)
    wide = Rect(1, 0, 12, 100)
    assert "no store yet" in rows(pane.render(session, tabs, wide), wide)[wide.y]
    state.regfile.mutable("r_acc")[:] = struct.pack("<128f", 2.0, *[1.0] * 127)

    run_until_complete(state)

    # By hand (ARCH-stage-aaq-str §4.3.2): the largest magnitude 2.0 sets scale
    # 2^1; 2.0 is exponent 0 below it (code 0x00) and 1.0 exponent 1 (0x08).
    assert session.last_aaq_row().data == b"\x00" + b"\x08" * 127
    header = rows(pane.render(session, tabs, wide), wide)[wide.y]
    assert "row 3, scale 2^1" in header
