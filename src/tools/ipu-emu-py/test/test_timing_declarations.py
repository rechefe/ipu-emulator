"""The ISA's timing declarations match the emulator's pinned behaviour (E03-14).

``test_timing_lr_views.py`` (E02-05) and ``test_timing_datapath_views.py``
(E02-06) pin, row by row, which value each read sees today. ``ipu_isa``
declares the same facts as data: each instruction's register reads with their
view, its writes, and the slot execution order. :data:`EXERCISES` maps every
test in those two files to the declarations it exercises, and one case is
generated per declaration: it checks that ``ipu_isa`` declares exactly what the
test pins. A new timing test must be added here (the coverage test fails until
it is), and a declaration that drifts from the pinned behaviour fails here.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from ipu_isa.slots import Slot
from ipu_isa.timing import SLOT_EXECUTION_ORDER, RegisterAccess, View
from ipu_isa.view import isa

CS, FW = View.CYCLE_START, View.FORWARDED
_HERE = Path(__file__).parent
TIMING_TEST_FILES = ("test_timing_lr_views.py", "test_timing_datapath_views.py")


@dataclass(frozen=True)
class Reads:
    """``slot.mnemonic`` reads ``register`` (selected by ``via``) with ``view``."""

    slot: Slot
    mnemonic: str
    register: str
    via: str | None
    view: View


@dataclass(frozen=True)
class Writes:
    """``slot.mnemonic`` writes ``register`` through operand ``via``."""

    slot: Slot
    mnemonic: str
    register: str
    via: str | None


@dataclass(frozen=True)
class Position:
    """``slot`` sits at ``index`` of SLOT_EXECUTION_ORDER (-1: last)."""

    slot: Slot
    index: int


@dataclass(frozen=True)
class NotTiming:
    """The test pins a rule that is not a register view (semantics, faults)."""

    reason: str


Declaration = Reads | Writes | Position | NotTiming


def _reads(slot: Slot, mnemonics: tuple[str, ...], register, via, view) -> tuple[Reads, ...]:
    return tuple(Reads(slot, m, register, via, view) for m in mnemonics)


_AGG = ("AGG.SUM", "AGG.SUM.FIRST", "AGG.MAX", "AGG.MAX.FIRST")
_ACC_MULT_RES = (
    "ACC.ADD",
    "ACC.ADD.FIRST",
    "ACC.SUB",
    "ACC.SUB.FIRST",
    "ACC.MAX",
    "ACC.MAX.FIRST",
    "ACC.STRIDE",
)

#: Each timing test, and the declarations its probes exercise.
EXERCISES: dict[str, tuple[Declaration, ...]] = {
    # ---- E02-05: LR views ----------------------------------------------------
    "test_lr_alu_sources_read_cycle_start": (
        Reads(Slot.LR, "ADD", "lr", "src_a", CS),
        Reads(Slot.LR, "SUB", "lr", "src_a", CS),
        Reads(Slot.LR, "INCR_MOD_POW2", "lr", "step", CS),
        Reads(Slot.LR, "ADDB", "lr", "src_b", CS),
    ),
    "test_lr_subslots_read_cycle_start": (Reads(Slot.LR, "ADD", "lr", "src_a", CS),),
    "test_inc_dec_read_own_dest_at_cycle_start": (
        Reads(Slot.LR, "INC", "lr", "dest", CS),
        Reads(Slot.LR, "DEC", "lr", "dest", CS),
        Reads(Slot.LR, "ADD", "lr", "src_a", CS),
    ),
    "test_cond_operands_read_cycle_start": (
        *_reads(Slot.COND, ("BEQ", "BNE", "BLT", "BGE"), "lr", "reg1", CS),
        Reads(Slot.COND, "BR", "lr", "reg", CS),
    ),
    "test_break_ifeq_reads_cycle_start": (Reads(Slot.BREAK, "BREAK.IFEQ", "lr", "reg", CS),),
    "test_load_offset_and_index_read_forwarded": (
        Reads(Slot.LOAD, "LDR_MULT_REG", "lr", "offset", FW),
        Reads(Slot.LOAD, "LDR_CYCLIC_MULT_REG", "lr", "offset", FW),
        Reads(Slot.LOAD, "LDR_CYCLIC_MULT_REG", "lr", "index", FW),
        Reads(Slot.LOAD, "LDR_MULT_MASK_REG", "lr", "offset", FW),
    ),
    "test_store_offsets_read_forwarded": (
        Reads(Slot.STORE, "STR_POST_AAQ_REG", "lr", "offset", FW),
        Reads(Slot.ACC_STORE, "STR_ACC_REG", "lr", "offset", FW),
    ),
    "test_mult_lr_operands_read_forwarded": (
        Reads(Slot.MULT, "MULT.RC.VS", "lr", "rc_idx", FW),
        Reads(Slot.MULT, "MULT.VE", "lr", "ra_idx", FW),
        Reads(Slot.MULT, "MULT.RC.VS", "lr", "mask_shift", FW),
    ),
    "test_mult_rc_ve_src_reads_forwarded": (Reads(Slot.MULT, "MULT.RC.VE", "lr", "src", FW),),
    "test_acc_stride_offset_reads_forwarded": (Reads(Slot.ACC, "ACC.STRIDE", "lr", "offset", FW),),
    "test_reshape_mask_lr_reads_forwarded": (
        Reads(Slot.ACC, "ACC.RESHAPE", "lr", "reshape_mask", FW),
    ),
    "test_agg_dest_slot_reads_cycle_start": _reads(Slot.ACC, _AGG, "lr", "dest_slot", CS),
    "test_reshape_source_dest_read_cycle_start": (
        Reads(Slot.ACC, "ACC.RESHAPE", "lr", "source", CS),
        Reads(Slot.ACC, "ACC.RESHAPE", "lr", "dest", CS),
    ),
    "test_lr_write_conflict_raises": (
        Writes(Slot.LR, "SET", "lr", "reg"),
        Writes(Slot.LR, "ADDB", "lr", "dest"),
    ),
    "test_lr_subslot_exception_leaves_no_partial_write": (
        NotTiming("every LR sub-slot reads before any writes; a fault ordering rule"),
    ),
    # ---- E02-06: datapath views ------------------------------------------------
    "test_mult_reads_r0_r1_at_cycle_start": (
        Reads(Slot.MULT, "MULT.RC.VV", "r", "ra", CS),
        Reads(Slot.MULT, "MULT.VE", "r", None, CS),
    ),
    "test_mult_reads_r_cyclic_at_cycle_start": (
        Reads(Slot.MULT, "MULT.RC.VS", "r_cyclic", None, CS),
    ),
    "test_mult_reads_r_mask_forwarded": (Reads(Slot.MULT, "MULT.RC.VS", "r_mask", None, FW),),
    "test_acc_reads_this_cycle_mult_res": _reads(Slot.ACC, _ACC_MULT_RES, "mult_res", None, FW),
    "test_agg_reads_this_cycle_mult_res": _reads(Slot.ACC, _AGG, "mult_res", None, FW),
    "test_reshape_reads_previous_mult_res": (Reads(Slot.ACC, "ACC.RESHAPE", "mult_res", None, CS),),
    "test_acc_reads_old_r_acc_at_cycle_start": (Reads(Slot.ACC, "ACC.ADD", "r_acc", None, CS),),
    "test_activate_reads_this_cycle_r_acc": (
        Reads(Slot.AAQ, "ACTIVATE.QUANTIZE", "r_acc", None, FW),
    ),
    "test_store_reads_this_cycle_post_aaq": (
        Reads(Slot.STORE, "STR_POST_AAQ_REG", "post_aaq_reg", None, FW),
    ),
    "test_acc_store_reads_this_cycle_r_acc": (
        Reads(Slot.ACC_STORE, "STR_ACC_REG", "r_acc", None, FW),
    ),
    "test_bkpt_halts_after_whole_word_retires": (Position(Slot.COND, -1),),
    "test_branches_compare_signed": (NotTiming("branch comparison semantics"),),
    "test_break_has_no_side_effects_and_resume_rereads": (Position(Slot.BREAK, 0),),
    "test_idle_word_advances_pc_once": (NotTiming("an empty word runs no slot"),),
}


def _timing_tests() -> set[str]:
    names = set()
    for file_name in TIMING_TEST_FILES:
        tree = ast.parse((_HERE / file_name).read_text())
        names |= {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        }
    return names


def test_every_timing_test_is_mapped():
    assert set(EXERCISES) == _timing_tests()


_CASES = [
    pytest.param(declaration, id=f"{test}-{index}")
    for test, declarations in EXERCISES.items()
    for index, declaration in enumerate(declarations)
    if not isinstance(declaration, NotTiming)
]


@pytest.mark.parametrize("declaration", _CASES)
def test_declaration_matches_the_pinned_behaviour(declaration):
    if isinstance(declaration, Position):
        assert SLOT_EXECUTION_ORDER[declaration.index] is declaration.slot
        return
    instruction = isa().instruction(declaration.slot, declaration.mnemonic)
    if isinstance(declaration, Writes):
        assert RegisterAccess(declaration.register, declaration.via) in instruction.writes()
        return
    declared = [
        access
        for access in instruction.reads()
        if (access.register, access.via_operand) == (declaration.register, declaration.via)
    ]
    assert [access.view for access in declared] == [declaration.view], declared
