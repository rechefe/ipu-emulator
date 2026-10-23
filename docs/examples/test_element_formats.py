"""The CR scalar example of Explanation: element formats (docs/content/explanation/element-formats.md).

The page includes the sections between the ``--8<--`` markers below. The test runs the
program in each mode and checks the decoded scalar, so the page shows what the emulator does.
"""

from __future__ import annotations

import struct

import pytest
from emu_support import make_state

from ipu_emu.emulator import run_until_complete
from ipu_emu.ipu_state import IpuState
from ipu_isa.cr_halves import constant
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES
from ipu_isa.nx8 import SCALE_BIAS, metadata_word

SECTION = "docs/content/explanation/element-formats.md, section CR scalars"

# --8<-- [start:program]
# Ra (R0 ++ R1) holds 1.0 in every element, so each MULT_RES element is the decoded scalar.
# CR15.L is the DS (every element active), CR14.L the QF of the vectors (NX8 e4).
PROGRAM = """
MULT.VE LR0 CR3 0 LR2 CR15.L CR14.L;;
HALT;;
"""
CR3 = {
    # The MULT constant layout: the e4 element 0x08 (0.5) under scale 2^3.
    ElementFormat.NX8: constant(value=0x08, scale=SCALE_BIAS + 3, exp_format=4),
    # The float32 bit pattern of 1.0 (0x3F800000) plus 0x38.
    ElementFormat.FP32: 0x3F80_0038,
}
# --8<-- [end:program]

# --8<-- [start:expected]
EXPECTED = {
    ElementFormat.NX8: 4.0,  # exponent 1 below the scale: 2^-1, times 2^3
    ElementFormat.FP32: 1 + 56 * 2**-23,  # all 32 bits as a float32: 1.0 plus 56 ulps
}
# --8<-- [end:expected]


def _ones(state: IpuState, fmt: ElementFormat) -> None:
    """R0 and R1 hold 1.0 in every element: in NX8 the code 0x00 (the scale itself)
    under scale 2^0 in each row's metadata; in FP32 the float32 1.0."""
    if fmt is ElementFormat.NX8:
        state.regfile.mutable("r")[:] = bytes(2 * LANES)
        state.regfile.mutable("r_meta")[:] = metadata_word(SCALE_BIAS).to_bytes(2, "little") * 2
    else:
        state.regfile.mutable("r_wide_debug")[:] = struct.pack(f"<{2 * LANES}f", *[1.0] * 2 * LANES)


def _scalar(fmt: ElementFormat, cr: int) -> float:
    program = PROGRAM if cr == 3 else PROGRAM.replace("CR3", f"CR{cr}")
    state = make_state(program, fmt=fmt, cr={3: CR3[fmt]})
    _ones(state, fmt)
    run_until_complete(state)
    return struct.unpack_from("<f", bytes(state.regfile.view("mult_res")))[0]  # element 0


@pytest.mark.parametrize("fmt", list(EXPECTED), ids=str)
def test_a_cr_scalar_is_decoded_in_the_active_mode(fmt):
    got = _scalar(fmt, 3)
    assert got == EXPECTED[fmt], (
        f"{fmt}: CR3 = {CR3[fmt]:#x} now decodes to {got}, not {EXPECTED[fmt]}. "
        f"The CR scalar rule changed: update EXPECTED and the {SECTION}, "
        "and its generated table (ipu_gen.docs_pages._cr_scalar_table)."
    )


@pytest.mark.parametrize("fmt", list(EXPECTED), ids=str)
@pytest.mark.parametrize(("cr", "value"), [(0, 0.0), (1, 1.0)])
def test_cr0_and_cr1_read_as_zero_and_one_in_both_modes(fmt, cr, value):
    """CR0 and CR1 are the MULT constants 0.0 and 1.0 in both modes (the page says so)."""
    assert _scalar(fmt, cr) == value, SECTION
