"""Tests for ``ipu_debug.session.DebugSession`` on real states: no curses, no patching."""

from __future__ import annotations

import struct

import pytest
from emu_support import make_state, new_state

from ipu_debug.control import DebugControl
from ipu_debug.session import (
    REGFILE_SCHEMA,
    DebugError,
    DebugSession,
    XmemRequest,
)
from ipu_emu.ipu import xmem_row_size_bytes
from ipu_emu.ipu_state import INST_MEM_SIZE
from ipu_isa.errors import IpuError
from ipu_isa.types import RegDtype

PROGRAM = "SET lr0 cr8;;\nBKPT;;"


@pytest.fixture
def session() -> DebugSession:
    return DebugSession(make_state(PROGRAM, cr={8: 100}), DebugControl())


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParseInt:
    @pytest.mark.parametrize(("text", "value"), [("12", 12), ("0x1f", 31), ("-3", -3)])
    def test_accepts_python_integer_syntax(self, text, value):
        assert DebugSession.parse_int(text) == value

    def test_rejects_text_with_a_debug_error(self):
        with pytest.raises(DebugError, match="Invalid integer: zz"):
            DebugSession.parse_int("zz")

    def test_debug_error_is_an_ipu_error_and_a_value_error(self):
        assert issubclass(DebugError, IpuError)
        assert issubclass(DebugError, ValueError)


class TestParseXmem:
    def test_parses_and_normalises_case(self, session):
        request = session.parse_xmem("ROW lr0 2 4 HEX")
        assert request == XmemRequest("row", "lr0", "2", 4, "hex")

    def test_accepts_the_command_word(self, session):
        assert session.parse_xmem("xmem byte 0 0 8 u8") == XmemRequest("byte", "0", "0", 8, "u8")

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("row 0 0 4", "Usage: xmem row|byte BASE OFFSET COUNT"),
            ("col 0 0 4 hex", "XMEM address mode must be 'row' or 'byte'"),
            ("row 0 0 4 f64", "XMEM format must be one of: hex|int8|u8|cell16|u32|f32"),
            ("row 0 0 0 hex", "XMEM count must be a positive integer; got '0'"),
            ("row 0 0 x hex", "XMEM count must be a positive integer; got 'x'"),
        ],
    )
    def test_rejects_malformed_requests(self, session, text, message):
        with pytest.raises(DebugError, match=message.replace("|", r"\|")):
            session.parse_xmem(text)


class TestResolveRegister:
    @pytest.mark.parametrize(
        ("name", "canonical", "index"),
        [
            ("lr5", "lr", 5),
            ("cr15", "cr", 15),
            ("acc", "r_acc", 0),
            ("rcyclic", "r_cyclic", 0),
            ("rmask", "r_mask", 0),
            ("r0", "r", 0),
            ("r1", "r", 1),
        ],
    )
    def test_names_and_aliases(self, session, name, canonical, index):
        desc, idx = session.resolve_register(name)
        assert (desc.name, idx) == (canonical, index)

    @pytest.mark.parametrize("name", ["xyz", "lr99", "r7"])
    def test_unknown_names_raise(self, session, name):
        with pytest.raises(DebugError, match=f"Unknown register: {name}"):
            session.resolve_register(name)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestRegisters:
    def test_scalar_view_is_an_integer(self, session):
        session.state.regfile.set_lr(3, 0xDEAD)
        (view,) = session.registers(["lr3"])
        assert view.name == "lr"
        assert view.index == 3
        assert view.dtype in (RegDtype.UINT32, RegDtype.INT32)
        assert view.value == 0xDEAD
        assert view.width == 4

    def test_vector_view_is_the_element_bytes(self, session):
        desc = session.state.regfile.descriptor("r")
        data = bytes(range(desc.size_bytes))
        session.state.regfile.mutable("r", desc.size_bytes, desc.size_bytes)[:] = data
        (view,) = session.registers(["r1"])
        assert isinstance(view.value, bytes)
        assert view.value == data
        assert view.width == desc.size_bytes

    def test_without_names_every_element_is_read(self, session):
        views = session.registers()
        assert len(views) == sum(desc.count for desc in REGFILE_SCHEMA)
        assert {(view.name, view.index) for view in views} >= {("lr", 0), ("cr", 15), ("r", 1)}

    def test_reflects_the_program_state(self):
        state = make_state(PROGRAM, cr={8: 100})
        assert DebugSession(state).registers(["cr8"])[0].value == 100

    def test_unknown_name_raises(self, session):
        with pytest.raises(DebugError):
            session.registers(["nope"])


class TestReadXmem:
    def test_byte_window_decodes_cells(self):
        state = new_state()
        state.xmem.write_address(16, struct.pack("<II", 7, 0xFFFFFFFF))
        window = DebugSession(state).read_xmem(XmemRequest("byte", "16", "0", 2, "u32"))
        assert window.byte_address == 16
        assert window.byte_count == 8
        assert [cell.value for cell in window.cells] == [7, 0xFFFFFFFF]
        assert [cell.address for cell in window.cells] == [16, 20]
        assert window.cells[0].raw == struct.pack("<I", 7)

    def test_signed_and_float_cells(self):
        state = new_state()
        state.xmem.write_address(0, struct.pack("<bb", -1, 2))
        state.xmem.write_address(4, struct.pack("<f", 1.5))
        debug = DebugSession(state)
        int8 = debug.read_xmem(XmemRequest("byte", "0", "0", 2, "int8"))
        assert [cell.value for cell in int8.cells] == [-1, 2]
        f32 = debug.read_xmem(XmemRequest("byte", "4", "0", 1, "f32"))
        assert [cell.value for cell in f32.cells] == [1.5]

    def test_row_window_follows_the_registers(self):
        state = new_state()
        debug = DebugSession(state)
        request = XmemRequest("row", "lr0", "1", 1, "hex")
        assert debug.read_xmem(request).metadata["row"] == 1
        state.regfile.set_lr(0, 2)
        window = debug.read_xmem(request)
        assert window.metadata["row"] == 3
        assert window.byte_address == 3 * xmem_row_size_bytes(state)
        assert window.byte_count == 1

    @pytest.mark.parametrize(
        ("request_", "message"),
        [
            (XmemRequest("byte", "2", "0", 1, "u32"), "XMEM u32 address must be 4-byte aligned"),
            (XmemRequest("byte", "-1", "0", 1, "hex"), "must be non-negative"),
            (XmemRequest("byte", "zz", "0", 1, "hex"), "Invalid address term 'zz'"),
            (XmemRequest("byte", "lrx", "0", 1, "hex"), "Invalid address register 'lrx'"),
            (XmemRequest("byte", "lr99", "0", 1, "hex"), "Unknown address register: lr99"),
            (XmemRequest("row", "100000000", "0", 1, "hex"), "out of range"),
        ],
    )
    def test_invalid_requests_raise(self, request_, message):
        with pytest.raises(DebugError, match=message):
            DebugSession(new_state()).read_xmem(request_)


class TestDisassemble:
    def test_lines_carry_text_current_pc_and_breakpoints(self):
        state = make_state(PROGRAM, cr={8: 100})
        control = DebugControl()
        control.add_breakpoint(1)
        lines = DebugSession(state, control).disassemble(0, 3)
        assert [line.pc for line in lines] == [0, 1, 2]
        assert "set" in lines[0].text.lower()
        assert "cr8" in lines[0].text.lower()
        assert not lines[0].text.startswith("PC")
        assert [line.is_current for line in lines] == [True, False, False]
        assert [line.has_breakpoint for line in lines] == [False, True, False]

    def test_empty_words_read_as_nop(self, session):
        (line,) = session.disassemble(INST_MEM_SIZE - 1, 1)
        assert line.text == "<nop>"

    def test_the_range_is_clipped_to_instruction_memory(self, session):
        assert [line.pc for line in session.disassemble(-2, 3)] == [0]
        assert [line.pc for line in session.disassemble(INST_MEM_SIZE - 1, 5)] == [
            INST_MEM_SIZE - 1
        ]
        assert session.disassemble(INST_MEM_SIZE, 4) == []


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


class TestSetRegister:
    def test_scalar_write_returns_the_masked_value_and_bumps_generation(self, session):
        from ipu_isa.constants import LR_CR_SCALAR_VALUE_MASK

        view = session.set_register("cr2", -1)
        assert view.name == "cr" and view.index == 2
        assert view.value == LR_CR_SCALAR_VALUE_MASK
        assert session.state.regfile.get_cr(2) == LR_CR_SCALAR_VALUE_MASK
        assert session.generation == 1

    def test_pc_is_a_settable_register(self, session):
        view = session.set_register("pc", 1)
        assert session.state.program_counter == 1
        assert view.name == "pc" and view.value == 1

    @pytest.mark.parametrize(
        ("name", "value", "message"),
        [
            ("pc", INST_MEM_SIZE, f"PC must be between 0 and {INST_MEM_SIZE - 1}"),
            ("pc", -1, f"PC must be between 0 and {INST_MEM_SIZE - 1}"),
            ("bogus", 1, "Unknown register: bogus"),
            ("r0", 1, "Cannot set byte-array register 'r0' with a scalar value"),
            ("cr0", 1, "CR0 is read-only"),
        ],
    )
    def test_rejects_with_the_frontend_text_and_changes_nothing(
        self, session, name, value, message
    ):
        pc = session.state.program_counter
        with pytest.raises(DebugError, match=message.replace("(", r"\(")):
            session.set_register(name, value)
        assert session.state.program_counter == pc
        assert session.state.regfile.get_cr(0) == 0
        assert session.generation == 0


class TestBreakpointsAndRunUntil:
    def test_add_toggle_remove_and_clear(self, session):
        session.add_breakpoint(1)
        assert session.control.breakpoints == {1}
        assert session.toggle_breakpoint(2) is True
        assert session.toggle_breakpoint(1) is False
        assert session.control.breakpoints == {2}
        session.remove_breakpoint(2)
        session.remove_breakpoint(3)  # absent: nothing happens
        assert session.control.breakpoints == frozenset()
        session.add_breakpoint(4)
        session.clear_breakpoints()
        assert session.control.breakpoints == frozenset()
        assert session.generation == 7

    @pytest.mark.parametrize(
        "method", ["add_breakpoint", "remove_breakpoint", "toggle_breakpoint", "run_until"]
    )
    def test_out_of_range_pc_is_a_debug_error(self, session, method):
        with pytest.raises(DebugError, match="PC must be between"):
            getattr(session, method)(INST_MEM_SIZE)
        assert session.generation == 0

    def test_run_until_resumes_only_away_from_the_current_pc(self, session):
        assert session.run_until(0) is False
        assert session.control.run_target is None
        assert session.run_until(1) is True
        assert session.control.run_target == 1

    @pytest.mark.parametrize(
        ("text", "message"), [("x", "PC must be an integer"), ("-1", "PC must be between")]
    )
    def test_parse_pc_reports_the_frontend_text(self, session, text, message):
        with pytest.raises(DebugError, match=message):
            session.parse_pc(text)


class TestMemoryWrites:
    def test_xmem_and_register_byte_writes_bump_generation(self, session):
        session.write_xmem(8, b"\x01\x02")
        assert session.state.xmem.read_address(8, 2) == b"\x01\x02"
        session.write_register_bytes("r", 3, b"\x07")
        assert session.state.regfile.view("r")[3] == 7
        assert session.generation == 2

    def test_out_of_range_writes_raise_debug_error(self, session):
        size = len(session.state.regfile.view("r"))
        with pytest.raises(DebugError, match="out of range"):
            session.write_register_bytes("r", size, b"\x00")
        with pytest.raises(DebugError):
            session.write_xmem(-1, b"\x00")
        assert session.generation == 0

    def test_halt_moves_the_pc_past_instruction_memory(self, session):
        session.halt()
        assert session.state.program_counter == INST_MEM_SIZE
        assert session.generation == 1
