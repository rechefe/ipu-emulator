"""A bound table set: a CR names a table by ID, and the access lands in its storage.

Everything goes through ``ReferenceModel``: ``bind_tables``, the program, and
the registers and XMEM the program leaves. With no table set bound, a CR holds
a flat row (the reset behaviour, which the conformance suite and the RTL
testbenches rely on).
"""

from __future__ import annotations

import pytest
from emu_support import QF_E4, ElementFormat, assemble_program

from ipu_emu.errors import ConfigurationError
from ipu_emu.geometry import table_row_check
from ipu_emu.reference import API_VERSION, ReferenceModel, RunState
from ipu_emu.xmem_access import XmemAccess
from ipu_isa.codec.word import DecodedWord

# Rows 0..39 are filled; row r of the NX8 image holds the byte r + 1 throughout.
FILLED_ROWS = 40
NX8_ROW, FP32_ROW = 128, 512


def _words(asm: str) -> list[int]:
    return [DecodedWord(w).to_word() for w in assemble_program(asm)]


def _model(asm: str, fmt: ElementFormat = ElementFormat.NX8) -> ReferenceModel:
    model = ReferenceModel(fmt)
    model.load_program(_words(asm))
    model.write_register("cr", QF_E4, 14)  # CR14.L: the programs' QF half
    row_bytes = NX8_ROW if fmt is ElementFormat.NX8 else FP32_ROW
    for row in range(FILLED_ROWS):
        model.write_xmem(row * row_bytes, bytes([row + 1]) * row_bytes)
    return model


def _image(model: ReferenceModel, fmt: ElementFormat = ElementFormat.NX8) -> bytes:
    row_bytes = NX8_ROW if fmt is ElementFormat.NX8 else FP32_ROW
    return model.read_xmem(0, FILLED_ROWS * row_bytes)


# -- the six XMEM instructions resolve through the table set ----------------

# (instruction, register it leaves) in a program that offsets LR0 by 3 and
# names its table with CR{cr}. Hand-written.
LOADS = {
    "LD.MULT": ("LD.MULT R0 LR0 CR{cr}", "r", 0),
    "LD.MULT.CYCLIC": ("LD.MULT.CYCLIC LR0 CR{cr} LR1", "r_cyclic", 0),
    "LD.MASK": ("LD.MASK LR0 CR{cr}", "r_mask", 0),
    "LD.LUT": ("LD.LUT 0 CR{cr} LR0", "aaq_lut", 0),
}


@pytest.mark.parametrize("name", sorted(LOADS))
def test_a_load_reads_its_offset_into_the_table_the_cr_names(name):
    instruction, register, index = LOADS[name]
    program = f"INC LR0 3 ;;\n{instruction.format(cr=5)} ;;\nHALT ;;"

    flat = _model(program)  # no table set: CR5 is row 20, so the load reads row 23
    flat.write_register("cr", 20, 5)
    flat.run()

    bound = _model(program)
    bound.bind_tables({9: (20, 8), 2: (0, 4)})  # IDs are scattered, not row numbers
    bound.write_register("cr", 9, 5)  # CR5 names table 9
    bound.run()

    assert bound.read_register(register, index) == flat.read_register(register, index)
    if name == "LD.MULT":  # row 23 holds the byte 24: written out, not derived
        assert bound.read_register("r", 0) == bytes([24]) * NX8_ROW


def test_the_same_offset_reads_a_different_row_through_a_different_id():
    program = "INC LR0 1 ;;\nLD.MULT R0 LR0 CR5 ;;\nLD.MULT R1 LR0 CR6 ;;\nHALT ;;"
    model = _model(program)
    model.bind_tables({3: (10, 4), 14: (30, 4)})
    model.write_register("cr", 3, 5)
    model.write_register("cr", 14, 6)
    model.run()
    assert model.read_register("r", 0) == bytes([12]) * NX8_ROW  # row 10 + 1
    assert model.read_register("r", 1) == bytes([32]) * NX8_ROW  # row 30 + 1


def test_activate_quantize_stores_into_its_table():
    program = "INC LR0 2 ;;\nACTIVATE.QUANTIZE identity CR15.L CR14.L CR{cr} LR0 ;;\nHALT ;;"
    flat = _model(program.format(cr=7))
    flat.write_register("cr", 25, 7)  # row 25 + 2 = 27
    flat.run()
    bound = _model(program.format(cr=7))
    bound.bind_tables({11: (25, 3)})
    bound.write_register("cr", 11, 7)
    bound.run()
    assert _image(bound) == _image(flat)
    start = 27 * NX8_ROW
    assert bound.read_xmem(start, NX8_ROW) != bytes([28]) * NX8_ROW  # row 27 was written
    assert bound.read_xmem(start - NX8_ROW, NX8_ROW) == bytes([27]) * NX8_ROW  # row 26 was not


def test_activate_exact_stores_into_its_table():
    program = "INC LR0 2 ;;\nACTIVATE.EXACT identity CR15.L CR7 LR0 ;;\nHALT ;;"
    flat = _model(program, ElementFormat.FP32)
    flat.write_register("cr", 25, 7)
    flat.run()
    bound = _model(program, ElementFormat.FP32)
    bound.bind_tables({11: (25, 3)})
    bound.write_register("cr", 11, 7)
    bound.run()
    assert _image(bound, ElementFormat.FP32) == _image(flat, ElementFormat.FP32)
    assert bound.read_xmem(27 * FP32_ROW, FP32_ROW) == bytes(FP32_ROW)  # identity of R_ACC = 0
    assert bound.read_xmem(26 * FP32_ROW, 4) == bytes([27]) * 4


# -- any CR can name a table -------------------------------------------------


def test_cr15_can_name_a_table():
    # CR15 conventionally holds the DS and QF halves; any CR is a valid operand.
    model = _model("LD.MULT R0 LR0 CR15 ;;\nHALT ;;")
    model.bind_tables({7: (33, 2)})
    model.write_register("cr", 7, 15)
    model.run()
    assert model.read_register("r", 0) == bytes([34]) * NX8_ROW


def test_cr0_and_cr1_name_tables_zero_and_one():
    program = "LD.MULT R0 LR0 CR0 ;;\nLD.MULT R1 LR0 CR1 ;;\nHALT ;;"
    model = _model(program)
    model.bind_tables({0: (6, 2), 1: (17, 2)})
    model.run()
    assert model.read_register("r", 0) == bytes([7]) * NX8_ROW
    assert model.read_register("r", 1) == bytes([18]) * NX8_ROW


# -- accesses that name no row of a bound table fault F-6 --------------------


def _faults_f6(model: ReferenceModel) -> None:
    status = model.run().status
    assert status.state is RunState.FAULTED
    assert status.fault is not None
    assert status.fault.label == "F-6"


@pytest.mark.parametrize("offset", [4, 100])
def test_an_offset_past_the_tables_rows_faults_with_no_effect(offset):
    model = _model(f"INC LR0 {offset} ; LD.MULT R0 LR0 CR5 ;;\nHALT ;;")
    model.bind_tables({2: (10, 4)})
    model.write_register("cr", 2, 5)
    _faults_f6(model)
    assert model.read_register("lr", 0) == 0  # the word retired nothing
    assert model.read_register("r", 0) == bytes(NX8_ROW)


def test_a_negative_offset_faults():
    model = _model("SET LR0 CR6 ; LD.MULT R0 LR0 CR5 ;;\nHALT ;;")
    model.bind_tables({2: (10, 4)})
    model.write_register("cr", 2, 5)
    model.write_register("cr", 0xFFFF_FFFF, 6)  # -1 as a 32-bit register
    _faults_f6(model)


def test_an_unbound_id_faults_and_says_which():
    model = _model("LD.MULT R0 LR0 CR5 ;;\nHALT ;;")
    model.bind_tables({2: (10, 4)})
    model.write_register("cr", 3, 5)  # table 3 is not bound
    _faults_f6(model)


@pytest.mark.parametrize(
    "store",
    [
        "ACTIVATE.QUANTIZE identity CR15.L CR14.L CR5 LR0",
        "ACTIVATE.EXACT identity CR15.L CR5 LR0",
    ],
)
def test_a_store_past_the_table_faults_and_writes_nothing(store):
    fmt = ElementFormat.FP32 if "EXACT" in store else ElementFormat.NX8
    model = _model(f"INC LR0 3 ; {store} ;;\nHALT ;;", fmt)
    model.bind_tables({2: (10, 3)})
    model.write_register("cr", 2, 5)
    before = _image(model, fmt)
    _faults_f6(model)
    assert _image(model, fmt) == before


def test_the_fault_detail_names_the_table_or_the_offset():
    state = _model("HALT ;;")._state  # the fault predicate takes the state, as the check phase does
    state.table_map = {2: (10, 4)}
    unbound, outside = table_row_check(state, 3, 0), table_row_check(state, 2, 4)
    assert unbound is not None and "table ID 3 is not bound" in unbound[1]
    assert outside is not None and "offset 4 is outside table 2" in outside[1]
    assert table_row_check(state, 2, 3) is None


# -- no table set: flat rows -------------------------------------------------


def test_with_no_table_set_a_cr_holds_a_flat_row():
    model = _model("INC LR0 3 ;;\nLD.MULT R0 LR0 CR5 ;;\nHALT ;;")
    assert model.tables() is None
    model.write_register("cr", 20, 5)
    model.run()
    assert model.read_register("r", 0) == bytes([24]) * NX8_ROW  # row 23


def test_unbinding_returns_to_flat_rows():
    model = _model("LD.MULT R0 LR0 CR5 ;;\nHALT ;;")
    model.bind_tables({9: (20, 8)})
    model.bind_tables(None)
    model.write_register("cr", 12, 5)
    model.run()
    assert model.read_register("r", 0) == bytes([13]) * NX8_ROW


# -- binding ----------------------------------------------------------------


def test_the_api_version_is_2_2_0():
    assert API_VERSION == "2.2.0"


def test_tables_returns_a_copy_of_what_was_bound():
    model = ReferenceModel()
    tables = {4: (0, 2), 15: (8, 1)}
    model.bind_tables(tables)
    tables[0] = (1, 1)  # the model kept a copy
    assert model.tables() == {4: (0, 2), 15: (8, 1)}
    got = model.tables()
    assert got is not None
    got.clear()  # and hands one out
    assert model.tables() == {4: (0, 2), 15: (8, 1)}


def test_reset_unbinds_and_load_program_keeps_the_set():
    model = _model("HALT ;;")
    model.bind_tables({1: (0, 1)})
    model.load_program(_words("HALT ;;"))
    assert model.tables() == {1: (0, 1)}
    model.reset(ElementFormat.NX8)
    assert model.tables() is None


@pytest.mark.parametrize(
    ("tables", "match"),
    [
        ({16: (0, 1)}, "table ID 16"),
        ({-1: (0, 1)}, "table ID -1"),
        ({0: (0, 0)}, "table 0"),
        ({0: (-1, 2)}, "table 0"),
        ({0: (16383, 2)}, "inside XMEM"),  # NX8 addresses 16384 rows
    ],
)
def test_a_table_set_that_does_not_fit_is_refused(tables, match):
    model = ReferenceModel()
    with pytest.raises(ConfigurationError, match=match):
        model.bind_tables(tables)
    assert model.tables() is None


def test_binding_is_refused_while_the_model_is_running():
    model = _model("INC LR0 1 ;;\nINC LR0 1 ;;\nHALT ;;")
    model.step()
    with pytest.raises(ConfigurationError, match="halted"):
        model.bind_tables({0: (0, 1)})


def test_a_row_past_xmem_in_the_running_format_is_refused_at_binding():
    wide = ReferenceModel(ElementFormat.FP32)
    wide.bind_tables({0: (16383, 2)})  # FP32 rows are 4x wider, XMEM has more of them
    assert wide.tables() == {0: (16383, 2)}


# -- the access observer -----------------------------------------------------


def test_the_observer_names_the_id_and_the_true_row():
    program = "INC LR0 2 ;;\nLD.MULT R0 LR0 CR5 ;;\nLD.MULT R1 LR0 CR6 ;;\nHALT ;;"
    model = _model(program)
    model.bind_tables({9: (20, 8), 4: (1, 3)})
    model.write_register("cr", 9, 5)
    model.write_register("cr", 4, 6)
    seen: list[XmemAccess] = []
    model.observe_xmem_accesses(seen.append)
    model.run()
    assert [(a.cr, a.base, a.table_id, a.storage_row, a.row) for a in seen] == [
        (5, 9, 9, 20, 22),
        (6, 4, 4, 1, 3),
    ]


def test_the_observer_has_no_table_id_when_nothing_is_bound():
    model = _model("LD.MULT R0 LR0 CR5 ;;\nHALT ;;")
    model.write_register("cr", 12, 5)
    seen: list[XmemAccess] = []
    model.observe_xmem_accesses(seen.append)
    model.run()
    assert [(a.base, a.table_id, a.row, a.storage_row) for a in seen] == [(12, None, 12, None)]
