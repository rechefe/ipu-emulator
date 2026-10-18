"""Precise faults (E04-20, IPU-FLT-3, IPU-FLT-8): a faulting word has no
architectural effect, and the lowest-numbered of its faults is recorded."""

from __future__ import annotations

import copy

import pytest
from emu_support import ElementFormat, assemble_program, make_state, new_state

from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.errors import EmulatorFault
from ipu_emu.ipu import plans_for
from ipu_emu.ipu_state import HaltRecord
from ipu_emu.prototyping import build_enabled
from ipu_isa.faults import FaultId
from ipu_isa.registers import REGISTER_DEFINITIONS
from ipu_isa.slots import Slot

# Each faulting word also writes LRs (one of them read by the faulting op,
# forwarded), loads a row and multiplies: none of it may show.
MULT = "MULT.RC.VS LR0 0 LR1 CR15"
WORDS = {
    FaultId.F3: (f"SET LR0 CR1 ; SET LR0 CR2 ; LD.MULT R0 LR7 CR2 ; {MULT} ;;", {}),
    FaultId.F4: (
        f"SET LR1 CR9 ; INC LR6 1 ; LD.MULT.CYCLIC LR7 CR2 LR1 ; {MULT} ;;",
        {9: 64},  # an index off a slot boundary, written by this word's LR slot
    ),
    FaultId.F5: (
        f"SET LR4 CR9 ; INC LR6 1 ; LD.MULT R0 LR7 CR2 ; {MULT} ; ACC.RESHAPE LRD0 LRD2 LR4 ;;",
        {9: 9},  # a mask above RESHAPE_ELEMENT_COUNT
    ),
    FaultId.F6: (
        f"SET LR7 CR9 ; INC LR6 1 ; LD.MULT R0 LR7 CR2 ; {MULT} ;;",
        {9: 16384},  # the first row past narrow mode's 2 MB
    ),
}


def _observable(state):
    return (
        {name: bytes(state.regfile.view(name)) for name in REGISTER_DEFINITIONS},
        {base: bytes(page) for base, page in state.xmem.touched_pages()},
        state.program_counter,
        copy.deepcopy(state.stats),
    )


# Both modes in a prototyping build (--config=proto), the faithful one otherwise.
MODES = [False, True] if build_enabled() else [False]


def _prepared(program, cr, prototyping=False, fmt=None):
    state = new_state(prototyping=prototyping, fmt=fmt)
    load_program(state, program)
    for index, value in cr.items():
        state.regfile.set_cr(index, value)
    state.xmem.write_address(0, bytes(range(1, 129)))  # a load would change R0
    state.regfile.set_lr(7, 0)
    return state


def _faults_at_word_one(program, cr, want, prototyping=False, fmt=None):
    state = _prepared(program, cr, prototyping, fmt)
    # Run the first word (a plain INC) alone: the limit of 1 then faults F-9
    # at word 1, before it runs. Snapshot there.
    with pytest.raises(EmulatorFault, match="F-9"):
        run_until_complete(state, max_cycles=1)
    before = _observable(state)
    with pytest.raises(EmulatorFault) as caught:
        run_until_complete(state, max_cycles=0, reset_stats=False)
    assert (caught.value.fault, caught.value.pc) == (want, 1)
    assert state.halt == HaltRecord("fault", 1, want)
    after = _observable(state)
    assert after[0] == before[0], "a register changed"
    assert after[1] == before[1], "XMEM changed"
    assert after[2] == before[2], "the PC moved"
    after[3].total_cycles = before[3].total_cycles  # the run's count, not the word's
    assert after[3] == before[3], "a statistic changed"


@pytest.mark.parametrize("prototyping", MODES)
@pytest.mark.parametrize("fault", sorted(WORDS))
def test_a_faulting_word_has_no_effect(fault, prototyping):
    asm, cr = WORDS[fault]
    program = assemble_program(f"INC LR7 0 ;;\n{asm}\nHALT ;;")
    _faults_at_word_one(program, cr, fault, prototyping)


@pytest.mark.parametrize("prototyping", MODES)
def test_f1_word_has_no_effect(prototyping):
    # A reserved activation code beside an LR write, a load and a multiply.
    word = dict(
        assemble_program(
            f"SET LR1 CR2 ; LD.MULT R0 LR7 CR2 ; {MULT} ; ACTIVATE.QUANTIZE relu CR15 ;;"
        )[0]
    )
    for field, plans in plans_for(Slot.AAQ):
        plan = plans[word[field]]
        if plan.inst_name == "ACTIVATE.QUANTIZE":
            word[dict(plan.fields)["activation_fn"]] = 15
    program = [assemble_program("INC LR7 0 ;;")[0], word, assemble_program("HALT ;;")[0]]
    _faults_at_word_one(program, {2: 5}, FaultId.F1, prototyping)


@pytest.mark.parametrize(
    ("asm", "cr", "want"),
    [
        # F-3 (two LR writes to LR0) and F-5 (mask 9 in LR4): F-3.
        (
            "SET LR0 CR1 ; SET LR0 CR2 ; SET LR4 CR9 ; ACC.RESHAPE LRD0 LRD2 LR4 ;;",
            {9: 9},
            FaultId.F3,
        ),
        # F-6 in LOAD and F-5 in ACC, LOAD issuing first: F-5, the lower.
        (
            "SET LR7 CR8 ; SET LR4 CR9 ; LD.MULT R0 LR7 CR2 ; ACC.RESHAPE LRD0 LRD2 LR4 ;;",
            {8: 16384, 9: 9},
            FaultId.F5,
        ),
        # F-4 in LOAD and F-6 in STORE: F-4.
        (
            "SET LR1 CR9 ; SET LR7 CR8 ; LD.MULT.CYCLIC LR0 CR2 LR1 ; ST.AAQ LR7 CR2 ;;",
            {8: 16384, 9: 64},
            FaultId.F4,
        ),
    ],
)
@pytest.mark.parametrize("prototyping", MODES)
def test_the_lowest_numbered_fault_wins(asm, cr, want, prototyping):
    program = assemble_program(f"INC LR7 0 ;;\n{asm}\nHALT ;;")
    _faults_at_word_one(program, cr, want, prototyping)


def test_a_clean_word_with_a_check_runs_as_before():
    state = make_state("SET LR7 CR9 ; LD.MULT R0 LR7 CR2 ;;\nHALT ;;", cr={9: 3})
    state.xmem.write_address(3 * 128, bytes([7]) * 128)
    run_until_complete(state)
    assert state.halt == HaltRecord("halt", 1, None)
    assert state.regfile.get_lr(7) == 3 and bytes(state.regfile.get_r(0)) == bytes([7]) * 128


def test_the_instructions_that_can_fault_on_register_values():
    # The predicates are bound by name (check_<name> beside execute_<name>):
    # a misspelt one would silently stop checking. Pin the set.
    checked = {
        (slot.value, plan.inst_name)
        for slot in Slot
        for _, plans in plans_for(slot)
        for plan in plans
        if plan.check is not None
    }
    assert checked == {
        ("load", "LD.MULT"),
        ("load", "LD.MULT.CYCLIC"),
        ("load", "LD.MASK"),
        ("store", "ST.AAQ"),
        ("acc_store", "ST.ACC"),
        ("acc", "ACC.RESHAPE"),
    }


@pytest.mark.filterwarnings("ignore::ipu_emu.errors.DebugOnlyInstructionWarning")
@pytest.mark.parametrize("prototyping", MODES)
@pytest.mark.parametrize("op", ["LD.MASK LR7 CR2", "ST.AAQ LR7 CR2", "ST.ACC LR7 CR2"])
def test_each_load_and_store_faults_f6_with_no_effect(op, prototyping):
    asm = f"SET LR7 CR9 ; INC LR6 1 ; {op} ;;"
    program = assemble_program(f"INC LR7 0 ;;\n{asm}\nHALT ;;")
    _faults_at_word_one(program, {9: 16384}, FaultId.F6, prototyping)


@pytest.mark.parametrize("prototyping", MODES)
def test_f6_follows_the_wide_row_size(prototyping):
    # Wide-vector debug rows are 512 bytes: row 16384 is in XMEM there, and
    # row 1 Mi (byte 512 MiB) is past it.
    word = "SET LR7 CR9 ; INC LR6 1 ; LD.MULT R0 LR7 CR2 ;;"
    program = assemble_program(f"INC LR7 0 ;;\n{word}\nHALT ;;")
    _faults_at_word_one(program, {9: 1 << 20}, FaultId.F6, prototyping, ElementFormat.FP32)
    state = _prepared(program, {9: 16384}, prototyping, ElementFormat.FP32)
    run_until_complete(state)
    assert state.halt == HaltRecord("halt", 2, None)


@pytest.mark.parametrize("prototyping", MODES)
def test_a_cyclic_load_with_a_legal_index_faults_f6_on_its_row(prototyping):
    asm = "SET LR1 CR8 ; SET LR7 CR9 ; INC LR6 1 ; LD.MULT.CYCLIC LR7 CR2 LR1 ;;"
    program = assemble_program(f"INC LR7 0 ;;\n{asm}\nHALT ;;")
    _faults_at_word_one(program, {8: 128, 9: 16384}, FaultId.F6, prototyping)
