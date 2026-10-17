"""The CR scalar example of Explanation: element formats (docs/content/explanation/element-formats.md).

The page includes the sections between the ``--8<--`` markers below. The test runs the
program in each format and checks the decoded scalar, so the page shows what the emulator does.
"""

from __future__ import annotations

import struct

import pytest
from emu_support import make_state

from ipu_emu.emulator import run_until_complete
from ipu_isa.formats import FORMATS, AccumulatorType, ElementFormat, encode_element
from ipu_isa.geometry import LANES

SECTION = "docs/content/explanation/element-formats.md, section CR scalars"

# --8<-- [start:program]
# Ra (R0 ++ R1) holds 1 in every element, so each MULT_RES element is the decoded scalar.
PROGRAM = """
MULT.VE LR0 CR3 0 LR2 CR15;;
HALT;;
"""
CR3 = 0x0000_0138  # low byte 0x38
# --8<-- [end:program]

# --8<-- [start:expected]
EXPECTED = {
    ElementFormat.INT8: 56,  # 0x38 as a signed byte; bit 8 is not read
    ElementFormat.E4: 1.0,  # 0x38 as an E4 element: exponent 7, bias 7, mantissa 0
    ElementFormat.FP32: 56.0,  # 0x38 as a signed byte, converted to a float
    ElementFormat.INT32: 56,  # 0x38 as a signed byte
}
# --8<-- [end:expected]


def _one(fmt: ElementFormat) -> bytes:
    """The bytes of one element of value 1 in ``fmt``."""
    if fmt is ElementFormat.FP32:
        return struct.pack("<f", 1.0)
    if fmt is ElementFormat.INT32:
        return struct.pack("<i", 1)
    return bytes([encode_element(1.0, fmt)])


@pytest.mark.parametrize("fmt", list(EXPECTED), ids=str)
def test_a_cr_scalar_is_decoded_in_the_active_format(fmt):
    state = make_state(PROGRAM, fmt=fmt, cr={3: CR3})
    wide = FORMATS[fmt].element_bits == 32
    ra = "r_wide_debug" if wide else "r"
    for half in (0, 1):
        state.regfile.set_register_bytes(ra, half, _one(fmt) * LANES)
    run_until_complete(state)
    lane = "<i" if FORMATS[fmt].accumulator is AccumulatorType.INT32 else "<f"
    got = struct.unpack_from(lane, bytes(state.regfile.view("mult_res")))[0]  # element 0
    assert got == EXPECTED[fmt], (
        f"{fmt}: CR3 = {CR3:#x} now decodes to {got}, not {EXPECTED[fmt]}. "
        f"The CR scalar rule changed (E04-28?): update EXPECTED and the {SECTION}, "
        "and its generated table (ipu_gen.docs_pages._cr_scalar_table)."
    )
