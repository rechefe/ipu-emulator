"""One run mode configures a state (E04-17, Q47, ADR 0007).

``IpuState(fmt=...)`` takes an ``ipu_isa.formats.ElementFormat``: ``NX8`` (the
hardware, the default) or ``FP32`` (emulator-only). Everything mode-dependent
(element and row bytes, the vector registers, the XMEM rows a program may
address) follows from it as ``ipu_isa`` declares. The mode is configuration:
an unknown one is a ``ConfigurationError`` at setup, and it never changes
mid-run. A program that uses the FP32-only ``ACTIVATE.EXACT`` is refused when
it is loaded into an NX8 state.
"""

from __future__ import annotations

import pytest
from emu_support import assemble_program

from ipu_emu.emulator import load_program
from ipu_emu.errors import ConfigurationError
from ipu_emu.geometry import (
    lane_format,
    vector_registers,
    xmem_element_width_bytes,
    xmem_row_size_bytes,
)
from ipu_emu.ipu import EIGHT_BIT_ROWS, XMEM_ADDRESSABLE_ROWS, addressable_rows
from ipu_emu.ipu_state import IpuState
from ipu_isa.formats import FORMATS, AccumulatorType, ElementFormat, register_layout
from ipu_isa.formats import addressable_rows as isa_addressable_rows

ALL_FORMATS = list(ElementFormat)


@pytest.mark.parametrize("fmt", ALL_FORMATS, ids=str)
def test_format_sets_element_and_row_bytes(fmt):
    state = IpuState(fmt=fmt)
    info = FORMATS[fmt]
    assert state.fmt is fmt
    assert state.format_info is info
    assert state.codec.fmt is fmt
    assert state.codec.element_bytes == xmem_element_width_bytes(state) == info.element_bytes
    assert xmem_row_size_bytes(state) == info.row_bytes


@pytest.mark.parametrize("fmt", ALL_FORMATS, ids=str)
def test_format_sets_accumulator_type(fmt):
    state = IpuState(fmt=fmt)
    assert FORMATS[fmt].accumulator is AccumulatorType.FP32
    assert lane_format(state) == "<f"
    assert state.codec.acc_dtype.itemsize == FORMATS[fmt].accumulator_bytes
    assert state.codec.acc_dtype.kind == "f"


@pytest.mark.parametrize("fmt", ALL_FORMATS, ids=str)
def test_format_sets_vector_registers(fmt):
    state = IpuState(fmt=fmt)
    registers = vector_registers(state)
    for declared, held in (("r", registers.ra), ("r_cyclic", registers.r_cyclic)):
        elements, element_bytes = register_layout(declared, fmt)
        assert state.regfile.descriptor(held).size_bytes == elements * element_bytes


@pytest.mark.parametrize("fmt", ALL_FORMATS, ids=str)
def test_format_sets_addressable_rows(fmt):
    # FP32 addresses the whole of XMEM; NX8 keeps the emulator's 16384-row
    # limit (``ipu_isa`` declares the full space).
    rows = addressable_rows(IpuState(fmt=fmt))
    if FORMATS[fmt].element_bytes == 1:
        assert rows == EIGHT_BIT_ROWS <= isa_addressable_rows(fmt)
    else:
        assert rows == XMEM_ADDRESSABLE_ROWS == isa_addressable_rows(fmt)


def test_default_format_is_nx8():
    assert IpuState().fmt is ElementFormat.NX8


def test_format_value_string_is_accepted():
    assert IpuState(fmt="fp32").fmt is ElementFormat.FP32


def test_format_is_read_only():
    state = IpuState()
    with pytest.raises(AttributeError):
        state.fmt = ElementFormat.FP32  # type: ignore[misc]
    assert state.fmt is ElementFormat.NX8


@pytest.mark.parametrize("bogus", ["bogus", 4, None, "int8", "e4", "int32"])
def test_unknown_format_is_a_configuration_error(bogus):
    with pytest.raises(ConfigurationError, match="unknown element format"):
        IpuState(fmt=bogus)


def test_the_old_keywords_are_gone():
    for keyword in ("dtype", "elu_alpha"):
        with pytest.raises(TypeError):
            IpuState(**{keyword: None})


def test_reset_changes_format_and_clears_state():
    state = IpuState()
    state.regfile.set_cr(3, 7)
    state.regfile.set_lr(2, 9)
    state.xmem.write_address(0x100, b"\x01\x02")
    state.program_counter = 5
    state.stats.total_cycles = 11
    fresh = IpuState(fmt=ElementFormat.FP32)

    state.reset(ElementFormat.FP32)

    assert state.fmt is ElementFormat.FP32
    assert state.codec is fresh.codec
    assert state.regfile.to_dict() == fresh.regfile.to_dict()
    assert state.xmem == fresh.xmem
    assert state.program_counter == 0
    assert state.stats.total_cycles == 0


def test_reset_keeps_format_when_none_given():
    state = IpuState(fmt=ElementFormat.FP32)
    state.reset()
    assert state.fmt is ElementFormat.FP32


def test_reset_to_unknown_format_changes_nothing():
    state = IpuState(fmt=ElementFormat.FP32)
    state.regfile.set_cr(3, 7)
    with pytest.raises(ConfigurationError):
        state.reset("bogus")  # type: ignore[arg-type]
    assert state.fmt is ElementFormat.FP32
    assert state.regfile.get_cr(3) == 7


def test_activate_exact_is_refused_at_load_in_nx8_mode():
    program = assemble_program("NOP;;\nACTIVATE.EXACT relu CR15.L CR2 LR0;;\nHALT;;\n")
    with pytest.raises(ConfigurationError, match="word 1 uses ACTIVATE.EXACT"):
        load_program(IpuState(fmt=ElementFormat.NX8), program)
    load_program(IpuState(fmt=ElementFormat.FP32), program)  # FP32 mode takes it
