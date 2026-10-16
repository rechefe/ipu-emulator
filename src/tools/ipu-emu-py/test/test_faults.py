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
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.errors import EmulatorError, EmulatorFault, XMemBoundsError
from ipu_emu.ipu import plans_for
from ipu_emu.ipu_state import HaltRecord
from ipu_isa.faults import FaultId
from ipu_isa.slots import Slot


def fault_of(state, **kw) -> EmulatorFault:
    with pytest.raises(EmulatorFault) as caught:
        run_until_complete(state, **kw)
    fault = caught.value
    assert state.halt == HaltRecord("fault", fault.pc, fault.fault)
    assert state.is_halted
    return fault


def test_halt_records_the_pc_of_the_halt_word():
    state = make_state("INC LR0 1 ;;\nINC LR0 1 ;;\nBKPT ;;")
    assert state.halt is None and not state.is_halted
    assert run_until_complete(state) == 3
    assert state.halt == HaltRecord("halt", 2, None)
    assert state.regfile.get_lr(0) == 2


def test_a_new_run_clears_the_halt():
    state = make_state("INC LR0 1 ;;\nBKPT ;;")
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
    state = make_state("BR CR2 ;;\nBKPT ;;", cr={2: 40})
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F2, 40)


@pytest.mark.parametrize("target", [1024, 1029, 0xFFFFFFFF])
def test_a_branch_past_imem_is_f2_not_a_quiet_halt(target):
    state = make_state("BR CR2 ;;\nBKPT ;;", cr={2: target})
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
        ("LDR_MULT_REG R0 LR0 CR2 ;;", Slot.LOAD, "LDR_MULT_REG", "dest", 2),
        ("INCR_MOD_POW2 LR0 LR1 4 ;;", Slot.LR, "INCR_MOD_POW2", "k", 9),
        ("ACC.STRIDE 16 off off LR0 ;;", Slot.ACC, "ACC.STRIDE", "elements_in_row", 3),
        ("ACTIVATE.QUANTIZE relu CR15 ;;", Slot.AAQ, "ACTIVATE.QUANTIZE", "activation_fn", 15),
        # SET's source shares a column wider than CrIdx: a code past CR15.
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
    state = make_state("NOP ;;\nSET LR0 CR1 ; SET LR0 CR2 ;;\nBKPT ;;")
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F3, 1)


def test_a_cyclic_index_off_a_slot_boundary_is_f4():
    state = make_state("LDR_CYCLIC_MULT_REG LR0 CR0 LR1 ;;\nBKPT ;;")
    state.regfile.set_lr(1, 64)
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F4, 0)


def test_a_reshape_mask_above_the_element_count_is_f5():
    state = make_state("NOP ;;\nACC.RESHAPE LRD0 LRD2 LR4 ;;\nBKPT ;;")
    state.regfile.set_lr(4, 9)
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F5, 1)


def test_an_xmem_row_out_of_range_is_f6():
    state = make_state("LDR_MULT_REG R0 LR0 CR2 ;;\nBKPT ;;", cr={2: 0xFFFFFFFF})
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F6, 0)


def test_an_access_beyond_the_xmem_allocation_is_f6(monkeypatch):
    # XMem's own bounds check (the row is valid, its bytes run past the end):
    # shrink the allocation XMem checks against, not the row check's.
    state = make_state("LDR_MULT_REG R0 LR0 CR2 ;;\nBKPT ;;", cr={2: 3})
    monkeypatch.setattr(xmem, "XMEM_SIZE_BYTES", 256)
    fault = fault_of(state)
    assert (fault.fault, fault.pc) == (FaultId.F6, 0)
    assert "XMEM read out of bounds" in fault.detail
    # Outside a cycle the same check is a plain error, as before.
    with pytest.raises(XMemBoundsError):
        state.xmem.read_address(512, 1)


def test_an_acc_stride_past_r_acc_is_an_error_before_any_write():
    # Not a fault: the ISA leaves the overrun undefined (an open question).
    state = make_state("ACC.STRIDE 16 off off LR0 ;;\nBKPT ;;")
    state.regfile.set_lr(0, 1)  # start at lane 32, then 128 elements
    before = bytes(state.regfile.get_r_acc_bytes())
    with pytest.raises(EmulatorError, match="past the end of R_ACC"):
        run_until_complete(state)
    assert bytes(state.regfile.get_r_acc_bytes()) == before
