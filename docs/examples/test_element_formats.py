"""The CR scalar example of Explanation: element formats (docs/content/explanation/element-formats.md).

The page includes the sections between the ``--8<--`` markers below. The test runs the
program in each format and checks the decoded scalar, so the page shows what the emulator does.
"""

from __future__ import annotations

import struct

import pytest
from emu_support import make_state

from ipu_emu.emulator import run_until_complete
from ipu_emu.geometry import vector_registers
from ipu_emu.numerics.codecs import codec_for
from ipu_isa.formats import FORMATS, AccumulatorType, ElementFormat
from ipu_isa.geometry import LANES

SECTION = "docs/content/explanation/element-formats.md, section CR scalars"

# --8<-- [start:program]
# Ra (R0 ++ R1) holds 1 in every element, so each MULT_RES element is the decoded scalar.
PROGRAM = """
MULT.VE LR0 CR3 0 LR2 CR15;;
HALT;;
"""
CR3 = 0x3F80_0038  # the float32 1.0 (0x3F800000) plus 0x38
# --8<-- [end:program]

# --8<-- [start:expected]
EXPECTED = {
    ElementFormat.INT8: 56,  # the low byte, 0x38, as a signed byte
    ElementFormat.E4: 1.0,  # 0x38 as an E4 element: exponent 7, bias 7, mantissa 0
    ElementFormat.FP32: 1 + 56 * 2**-23,  # all 32 bits as a float32: 1.0 plus 56 ulps
    ElementFormat.INT32: 1_065_353_272,  # all 32 bits as a signed integer
}
# --8<-- [end:expected]


@pytest.mark.parametrize("fmt", list(EXPECTED), ids=str)
def test_a_cr_scalar_is_decoded_in_the_active_format(fmt):
    state = make_state(PROGRAM, fmt=fmt, cr={3: CR3})
    ra = vector_registers(state).ra  # R0 ++ R1 in this format's element width
    for half in (0, 1):
        state.regfile.set_register_bytes(ra, half, codec_for(fmt).one() * LANES)
    run_until_complete(state)
    lane = "<i" if FORMATS[fmt].accumulator is AccumulatorType.INT32 else "<f"
    got = struct.unpack_from(lane, bytes(state.regfile.view("mult_res")))[0]  # element 0
    assert got == EXPECTED[fmt], (
        f"{fmt}: CR3 = {CR3:#x} now decodes to {got}, not {EXPECTED[fmt]}. "
        f"The CR scalar rule changed: update EXPECTED and the {SECTION}, "
        "and its generated table (ipu_gen.docs_pages._cr_scalar_table)."
    )


@pytest.mark.parametrize("fmt", list(EXPECTED), ids=str)
def test_cr1_reads_as_one_in_every_format(fmt):
    """CR1, the hardware constant 1, is the scalar 1 in every format (the page says so)."""
    state = make_state(PROGRAM.replace("CR3", "CR1"), fmt=fmt)
    ra = vector_registers(state).ra
    for half in (0, 1):
        state.regfile.set_register_bytes(ra, half, codec_for(fmt).one() * LANES)
    run_until_complete(state)
    lane = "<i" if FORMATS[fmt].accumulator is AccumulatorType.INT32 else "<f"
    assert struct.unpack_from(lane, bytes(state.regfile.view("mult_res")))[0] == 1, SECTION
