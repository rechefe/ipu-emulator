"""The fault model (E04-19): an illegal program raises ``EmulatorFault`` with
its fault ID and the faulting word's PC, and the state records the halt.

F-4, F-5 and F-6 from the handlers are also checked where their instructions
are tested (``test_execute.py``); F-3 in ``test_cycle.py`` and
``test_timing_lr_views.py``. F-7 and F-8 are not modelled; F-9 is E04-21.
"""

from __future__ import annotations

import pytest
from emu_support import assemble_program, make_state, new_state

from ipu_emu import xmem
from ipu_emu.cycle import schedule_word
from ipu_emu.emulator import DebugAction, load_program, run_until_complete, run_with_debug
from ipu_emu.errors import ConfigurationError, EmulatorError, EmulatorFault, XMemBoundsError
from ipu_emu.ipu import Ipu, plans_for
from ipu_emu.ipu_state import HaltRecord
from ipu_isa.faults import ArithmeticEvent, FaultId
from ipu_isa.slots import Slot


def fault_of(state, **kw) -> EmulatorFault:
    with pytest.raises(EmulatorFault) as caught:
        run_until_complete(state, **kw)
    fault = caught.value
    assert state.halt == HaltRecord("fault", fault.pc, fault.fault)
    assert state.is_halted
    return fault


def test_halt_records_the_pc_of_the_halt_word():
    state = make_state("INC LR0 1 ;;\nINC LR0 1 ;;\nHALT ;;")
    assert state.halt is None and not state.is_halted
    assert run_until_complete(state) == 3
    assert state.halt == HaltRecord("halt", 2, None)
    assert state.regfile.get_lr(0) == 2


def test_a_new_run_clears_the_halt():
    state = make_state("INC LR0 1 ;;\nHALT ;;")
    run_until_complete(state)
    state.program_counter = 0
    assert run_until_complete(state) == 2
    assert state.halt == HaltRecord("halt", 1, None) and state.regfile.get_lr(0) == 2


def test_running_past_the_program_is_f2():
    state = make_state("INC LR0 1 ;;\nINC LR0 1 ;;")
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F2, 2)
    assert str(fault).startswith("F-2 at PC 2: fetch outside the program (length 2)")
    assert state.regfile.get_lr(0) == 2  # both words retired
    assert state.stats.total_cycles == 2


def test_a_branch_past_the_program_is_f2_at_the_next_fetch():
    state = make_state("BR CR2 ;;\nHALT ;;", cr={2: 40})
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F2, 40)


@pytest.mark.parametrize("target", [1024, 1029, 0xFFFFFFFF])
def test_a_branch_past_imem_is_f2_not_a_quiet_halt(target):
    state = make_state("BR CR2 ;;\nHALT ;;", cr={2: target})
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F2, target)


def test_raise_on_fault_false_returns_with_the_fault_recorded():
    state = make_state("INC LR0 1 ;;")
    assert run_until_complete(state, raise_on_fault=False) == 1
    assert state.halt == HaltRecord("fault", 1, FaultId.F2)


def test_an_undefined_opcode_is_f1():
    nop = assemble_program("NOP ;;")[0]
    word = dict(nop)
    field, plans = plans_for(Slot.COND)[0]
    word[field] = len(plans)
    state = new_state()
    load_program(state, [nop, word])
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F1, 1)
    assert fault.detail == f"cond slot: opcode {len(plans)} ({field}) is not defined by the ISA"


def with_operand(asm: str, slot: Slot, mnemonic: str, operand: str, value: int) -> dict:
    """The first word of *asm* with *mnemonic*'s *operand* column set to *value*
    (a code the assembler would never emit)."""
    word = dict(assemble_program(asm)[0])
    for field, plans in plans_for(slot):
        plan = plans[word[field]]
        if plan.inst_name == mnemonic:
            word[dict(plan.fields)[operand]] = value
            return word
    raise AssertionError(f"{mnemonic} not in {asm!r}")


@pytest.mark.parametrize(
    ("asm", "slot", "mnemonic", "operand", "value"),
    [
        ("LD.MULT R0 LR0 CR2 ;;", Slot.LOAD, "LD.MULT", "dest", 2),
        ("INC.MODP2 LR0 LR1 4 ;;", Slot.LR, "INC.MODP2", "k", 9),
        ("ACC.STRIDE 16 off off LR0 ;;", Slot.ACC, "ACC.STRIDE", "elements_in_row", 3),
        (
            "ACTIVATE.QUANTIZE relu CR15.L CR14.L CR2 LR0 ;;",
            Slot.AAQ,
            "ACTIVATE.QUANTIZE",
            "activation_fn",
            7,
        ),
        ("LD.LUT 0 CR2 LR0 ;;", Slot.AAQ, "LD.LUT", "lut_addr", 5),
        # SET's source shares a column wider than LcrHalfIdx: a code past CR15.H.
        ("SET LR0 CR1 ;;", Slot.LR, "SET", "src", 200),
    ],
)
def test_a_reserved_operand_encoding_is_f1(asm, slot, mnemonic, operand, value):
    state = new_state()
    nop = assemble_program("NOP ;;")[0]
    load_program(state, [nop, with_operand(asm, slot, mnemonic, operand, value)])
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F1, 1)
    assert f"{mnemonic} {operand} = {value} is a reserved encoding" in fault.detail


def test_an_lr_conflict_is_f3_at_its_word():
    state = make_state("NOP ;;\nSET LR0 CR1 ; SET LR0 CR2 ;;\nHALT ;;")
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F3, 1)


def test_a_cyclic_index_off_a_slot_boundary_is_f4():
    state = make_state("LD.MULT.CYCLIC LR0 CR0 LR1 ;;\nHALT ;;")
    state.regfile.set_lr(1, 64)
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F4, 0)


def test_a_reshape_mask_above_the_element_count_is_f5():
    state = make_state("NOP ;;\nACC.RESHAPE LRD0 LRD2 LR4 ;;\nHALT ;;")
    state.regfile.set_lr(4, 9)
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F5, 1)


def test_an_xmem_row_out_of_range_is_f6():
    state = make_state("LD.MULT R0 LR0 CR2 ;;\nHALT ;;", cr={2: 0xFFFFFFFF})
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F6, 0)


def test_an_access_beyond_the_xmem_allocation_is_f6(monkeypatch):
    # XMem's own bounds check (the row is valid, its bytes run past the end):
    # shrink the allocation XMem checks against, not the row check's.
    state = make_state("LD.MULT R0 LR0 CR2 ;;\nHALT ;;", cr={2: 3})
    monkeypatch.setattr(xmem, "XMEM_SIZE_BYTES", 256)
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F6, 0)
    assert "XMEM read out of bounds" in fault.detail
    # Outside a cycle the same check is a plain error, as before.
    with pytest.raises(XMemBoundsError):
        state.xmem.read_address(512, 1)


def test_an_acc_stride_past_r_acc_is_an_error_before_any_write():
    # Not a fault: the ISA leaves the overrun undefined (an open question).
    state = make_state("ACC.STRIDE 16 off off LR0 ;;\nHALT ;;")
    state.regfile.set_lr(0, 1)  # start at lane 32, then 128 elements
    before = bytes(state.regfile.get_r_acc_bytes())
    with pytest.raises(EmulatorError, match="past the end of R_ACC"):
        run_until_complete(state)
    assert bytes(state.regfile.get_r_acc_bytes()) == before


# --- enable bits and sticky flags (E04-22) ---------------------------------


def test_after_reset_every_fault_is_enabled_and_nothing_is_sticky():
    state = new_state()
    assert state.faults.enable == dict.fromkeys(FaultId, True)
    assert state.faults.sticky == set()
    state.faults.set_enabled(FaultId.F1, False)
    state.faults.sticky.add(FaultId.F1)
    state.faults.reset()
    assert state.faults.enable == dict.fromkeys(FaultId, True) and not state.faults.sticky


def test_an_enabled_fault_is_sticky_too():
    state = make_state("INC LR0 1 ;;")
    fault_of(state)
    assert state.faults.sticky == {FaultId.F2}


def test_f1_disabled_runs_the_slot_as_nop_and_the_word_retires():
    # INC in the LR slot and a reserved activation code in the AAQ slot.
    word = with_operand(
        "INC LR0 1 ; ACTIVATE.QUANTIZE relu CR15.L CR14.L CR2 LR0 ;;",
        Slot.AAQ,
        "ACTIVATE.QUANTIZE",
        "activation_fn",
        7,
    )
    state = new_state()
    load_program(state, [word, assemble_program("HALT ;;")[0]])
    state.faults.set_enabled(FaultId.F1, False)
    assert run_until_complete(state) == 2
    assert state.halt == HaltRecord("halt", 1, None)
    assert state.regfile.get_lr(0) == 1  # the other slot retired
    assert state.faults.sticky == {FaultId.F1}
    # The same word with F-1 enabled faults (the cached schedule is not stale).
    state.faults.set_enabled(FaultId.F1, True)
    state.program_counter = 0
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F1, 0)


def test_f9_disabled_runs_past_the_limit_and_sets_the_flag():
    state = make_state(
        "SET LR1 CR8 ;;\nloop:\nINC LR0 1 ;;\nBNE LR0 LR1 loop ;;\nHALT ;;", cr={8: 10}
    )
    state.faults.set_enabled(FaultId.F9, False)
    assert run_until_complete(state, max_cycles=5) == 22
    assert state.halt == HaltRecord("halt", 3, None)
    assert state.faults.sticky == {FaultId.F9}


@pytest.mark.parametrize("fault", [FaultId.F2, FaultId.F3, FaultId.F4, FaultId.F5, FaultId.F6])
def test_disabling_an_undecided_fault_is_refused_at_setup(fault):
    state = make_state("HALT ;;")
    with pytest.raises(ConfigurationError, match=f"{fault.label} cannot be disabled.*OQ-2"):
        state.faults.set_enabled(fault, False)
    # A direct write past set_enabled is refused when the run starts, before
    # anything changes.
    state.faults.enable[fault] = False
    state.stats.total_cycles = 7
    state.halt = sentinel = HaltRecord("halt", 99, None)
    state.quit_requested = True
    with pytest.raises(ConfigurationError, match="OQ-2"):
        run_until_complete(state)
    assert state.stats.total_cycles == 7 and state.halt is sentinel and state.quit_requested


def test_fault_state_survives_a_step():
    state = make_state("INC LR0 1 ;;\nHALT ;;")
    state.faults.set_enabled(FaultId.F7, False)  # not modelled: allowed, changes nothing
    state.faults.sticky.add(FaultId.F1)
    ipu = Ipu(state)  # held: the cycle keeps only a weak reference to it
    ipu.cycle.step()
    assert state.faults.enable[FaultId.F7] is False and state.faults.sticky == {FaultId.F1}


def _slot_fields():
    """(slot, opcode field, plans) of every HARDWARE slot (and BREAK), each LR sub-slot."""
    return [
        (slot, field, plans)
        for slot in (
            Slot.BREAK,
            Slot.LR,
            Slot.LOAD,
            Slot.MULT,
            Slot.ACC,
            Slot.AAQ,
            Slot.COND,
        )
        for field, plans in plans_for(slot)
    ]


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize(("slot", "field", "plans"), _slot_fields(), ids=lambda v: str(v)[:40])
def test_an_undefined_opcode_schedules_as_that_slots_nop(slot, field, plans, skip):
    word = dict(assemble_program("INC LR0 1 ;;")[0])
    nop = next(code for code, plan in enumerate(plans) if plan.inst_name == "NOP")
    as_nop = dict(word, **{field: nop})
    reserved = schedule_word(dict(word, **{field: len(plans)}), skip_idle_nops=skip)
    assert reserved.reserved is not None
    assert reserved._replace(reserved=None) == schedule_word(as_nop, skip_idle_nops=skip)


def test_f1_disabled_cond_slot_still_advances_the_pc():
    field, plans = plans_for(Slot.COND)[0]
    word = dict(assemble_program("INC LR0 1 ;;")[0], **{field: len(plans)})
    state = new_state()
    load_program(state, [word, assemble_program("HALT ;;")[0]])
    state.faults.set_enabled(FaultId.F1, False)
    assert run_until_complete(state) == 2
    assert state.halt == HaltRecord("halt", 1, None) and state.regfile.get_lr(0) == 1


def test_f1_disabled_word_retires_once_after_a_debugger_stop():
    word = with_operand(
        "INC LR0 1 ; ACTIVATE.QUANTIZE relu CR15.L CR14.L CR2 LR0 ;;",
        Slot.AAQ,
        "ACTIVATE.QUANTIZE",
        "activation_fn",
        7,
    )
    state = new_state()
    load_program(state, [word, assemble_program("HALT ;;")[0]])
    state.faults.set_enabled(FaultId.F1, False)
    stops = []

    def callback(state, cycles):
        stops.append(state.program_counter)
        return DebugAction.CONTINUE

    assert run_with_debug(state, callback, break_on_entry=True) == 2
    assert stops == [0]
    assert state.regfile.get_lr(0) == 1  # retired exactly once
    assert state.faults.sticky == {FaultId.F1} and state.halt == HaltRecord("halt", 1, None)


def test_reset_sets_every_enable_bit_and_clears_the_flags():
    state = make_state("INC LR0 1 ;;")
    fault_of(state)
    state.faults.set_enabled(FaultId.F1, False)
    state.arith_flags.add(ArithmeticEvent.ACC_OVERFLOW)
    state.reset()
    assert state.faults.enable == dict.fromkeys(FaultId, True) and not state.faults.sticky
    assert state.halt is None and not state.arith_flags
