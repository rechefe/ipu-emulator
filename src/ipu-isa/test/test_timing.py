"""The timing contract as data (E03-14)."""

from __future__ import annotations

import copy

import pytest

from ipu_isa.instruction_spec import INSTRUCTION_SPEC, PSEUDO_INSTRUCTION_SPEC
from ipu_isa.operand_types import OPERAND_TYPES
from ipu_isa.schema import SpecError
from ipu_isa.slots import SLOTS, Slot
from ipu_isa.timing import (
    CONTROL_SLOTS,
    KNOWN_DEVIATIONS,
    OPERAND_REGISTERS,
    SLOT_EXECUTION_ORDER,
    Deviation,
    RegisterAccess,
    View,
    contract_rule,
    contract_view,
    declared_reads,
    deviations,
    register_writers,
    validate_timing,
)
from ipu_isa.view import isa, load_isa

CS, FW = View.CYCLE_START, View.FORWARDED


def _spec():
    return copy.deepcopy(INSTRUCTION_SPEC)


def _operand(spec, slot, mnemonic, name):
    return next(op for op in spec[slot][mnemonic]["operands"] if op["name"] == name)


def _implicit_read(spec, slot, mnemonic, register):
    return next(r for r in spec[slot][mnemonic]["implicit_reads"] if r["register"] == register)


def _rejects(spec, match):
    with pytest.raises(SpecError, match=match):
        validate_timing(spec, SLOTS)
    with pytest.raises(SpecError, match=match):
        load_isa(spec, PSEUDO_INSTRUCTION_SPEC, SLOTS, OPERAND_TYPES)


def test_the_repository_isa_satisfies_the_validator():
    validate_timing(INSTRUCTION_SPEC, SLOTS)


def test_slot_execution_order():
    assert SLOT_EXECUTION_ORDER == (
        Slot.BREAK,
        Slot.LR,
        Slot.LOAD,
        Slot.MULT,
        Slot.ACC,
        Slot.AAQ,
        Slot.AAQ_EXACT,
        Slot.COND,
    )
    assert set(SLOT_EXECUTION_ORDER) == set(SLOTS)
    assert CONTROL_SLOTS == {Slot.LR, Slot.COND, Slot.BREAK}


def test_operand_registers_cover_the_readable_operand_types():
    readable = {name for name, record in OPERAND_TYPES.items() if record.readable}
    assert set(OPERAND_REGISTERS) == readable


# -- contract_view: one test per Q32 rule ---------------------------------------

#: The repository ISA's writers, as load_isa checks it with.
WRITERS = register_writers(INSTRUCTION_SPEC)


@pytest.mark.parametrize("slot", [Slot.LR, Slot.COND, Slot.BREAK])
def test_rule_1_control_reads_lrs_at_cycle_start(slot):
    assert contract_view(slot, "lr", WRITERS) is CS


@pytest.mark.parametrize("slot", [Slot.LOAD, Slot.MULT, Slot.ACC, Slot.AAQ, Slot.AAQ_EXACT])
def test_rule_1_other_stages_read_lrs_forwarded(slot):
    assert contract_view(slot, "lr", WRITERS) is FW


@pytest.mark.parametrize("register", ["r", "r_cyclic", "r_mask", "r_meta", "r_cyclic_meta"])
def test_rule_2_mult_reads_its_inputs_at_cycle_start(register):
    """LOAD writes them earlier in the word; MULT still reads cycle start."""
    assert contract_view(Slot.MULT, register, WRITERS) is CS


def test_rule_3_mult_res_and_r_acc_are_forwarded():
    assert contract_view(Slot.ACC, "mult_res", WRITERS) is FW
    assert contract_view(Slot.AAQ, "r_acc", WRITERS) is FW


def test_rule_4_lr_subslots_read_cycle_start():
    assert contract_view(Slot.LR, "lr", WRITERS) is CS


def test_other_reads_follow_the_producers():
    """Forwarded when every writer runs earlier; cycle start for a same-slot
    writer or a register nothing writes."""
    assert contract_view(Slot.ACC, "r_acc", WRITERS) is CS  # ACC itself writes it
    assert contract_view(Slot.AAQ_EXACT, "r_acc", WRITERS) is FW
    assert contract_view(Slot.AAQ, "aaq_lut", WRITERS) is CS  # AAQ itself writes it
    assert contract_view(Slot.LOAD, "cr", WRITERS) is CS
    writers = {"r_acc": [Slot.ACC, Slot.COND]}  # a later writer: cycle start
    assert contract_view(Slot.AAQ_EXACT, "r_acc", writers) is CS


@pytest.mark.parametrize(
    ("slot", "register", "rule"),
    [
        (Slot.COND, "lr", 1),
        (Slot.BREAK, "lr", 1),
        (Slot.LOAD, "lr", 1),
        (Slot.ACC, "lr", 1),
        (Slot.MULT, "lr", 1),
        (Slot.MULT, "r", 2),
        (Slot.MULT, "r_cyclic", 2),
        (Slot.MULT, "r_mask", 2),
        (Slot.MULT, "r_meta", 2),
        (Slot.MULT, "r_cyclic_meta", 2),
        (Slot.ACC, "mult_res", 3),
        (Slot.AAQ, "r_acc", 3),
        (Slot.LR, "lr", 4),
        (Slot.LR, "cr", 4),
        (Slot.COND, "cr", None),
        (Slot.ACC, "r_acc", None),
        (Slot.AAQ_EXACT, "r_acc", None),
    ],
)
def test_contract_rule_names_the_rule_that_gives_the_view(slot, register, rule):
    assert contract_rule(slot, register) == rule


def test_every_declared_read_is_labelled_with_a_rule_or_the_slot_order():
    """Each of the four rules labels at least one read of the repository ISA."""
    rules = {
        contract_rule(Slot(slot), access.register)
        for slot, instructions in INSTRUCTION_SPEC.items()
        for inst_def in instructions.values()
        for access in declared_reads(inst_def)
    }
    assert rules == {1, 2, 3, 4, None}


def test_register_writers():
    assert register_writers(INSTRUCTION_SPEC) == {
        "lr": {Slot.LR},
        "r": {Slot.LOAD},
        "r_cyclic": {Slot.LOAD},
        "r_mask": {Slot.LOAD},
        "mult_res": {Slot.MULT},
        "r_acc": {Slot.ACC},
        "r_meta": {Slot.LOAD},
        "r_cyclic_meta": {Slot.LOAD},
        "aaq_lut": {Slot.AAQ},
    }


# -- the declarations -------------------------------------------------------------


def test_acc_add_declares_its_registers():
    add = isa().instruction(Slot.ACC, "ACC.ADD")
    assert add.reads() == (RegisterAccess("mult_res", None, FW), RegisterAccess("r_acc", None, CS))
    assert add.writes() == (RegisterAccess("r_acc"),)


def test_operand_accesses_are_reads_and_writes():
    add = isa().instruction(Slot.LR, "ADD")
    assert add.writes() == (RegisterAccess("lr", "dest"),)
    assert add.reads() == (
        RegisterAccess("lr", "src_a", CS),
        RegisterAccess("lr", "src_b", CS),
        RegisterAccess("cr", "src_b", CS),
    )
    assert add.operand("src_b").type.name == "LcrHalfIdx"
    inc = isa().instruction(Slot.LR, "INC")
    assert inc.reads() == (RegisterAccess("lr", "dest", CS),)


def test_every_non_nop_instruction_but_a_branch_writes_something():
    """COND and BREAK change the PC or halt; the AaQ stores write XMEM (not a
    register); every other instruction writes a register."""
    xmem_stores = {"ACTIVATE.QUANTIZE", "ACTIVATE.EXACT"}
    for slot in (Slot.LR, Slot.LOAD, Slot.MULT, Slot.ACC, Slot.AAQ, Slot.AAQ_EXACT):
        for instruction in isa().instructions(slot):
            if instruction.mnemonic not in {"NOP", *xmem_stores}:
                assert instruction.writes(), f"{slot}.{instruction.mnemonic}"


def test_known_deviations_are_the_issue_table():
    """E04-27 removed the last rows: the emulator follows the contract everywhere."""
    assert KNOWN_DEVIATIONS == ()
    assert deviations(INSTRUCTION_SPEC) == ()


@pytest.mark.parametrize(
    "mnemonic", ["MULT.RC.VV", "MULT.RC.VE", "MULT.RC.VS", "MULT.VE", "MULT.EE"]
)
def test_mult_reads_its_data_inputs_at_cycle_start(mnemonic):
    """Q20, Q32 rule 2: every MULT data input (R0/R1, R_CYCLIC, R_MASK) is read
    at cycle start (E04-27 flipped R_MASK)."""
    reads = {a.register: a.view for a in isa().instruction(Slot.MULT, mnemonic).reads()}
    assert reads["r_mask"] is CS
    for register in ("r", "r_cyclic", "r_meta", "r_cyclic_meta"):
        assert reads.get(register, CS) is CS
    assert reads["r_meta"] is CS or reads["r_cyclic_meta"] is CS


# -- each validator rule fails on a planted violation -----------------------------


def test_forwarded_read_from_a_later_slot_is_rejected():
    spec = _spec()
    read = _implicit_read(spec, "mult", "MULT.RC.VV", "r_mask")
    read["register"], read["view"] = "aaq_lut", "forwarded"
    _rejects(spec, r"mult\.MULT\.RC\.VV: aaq_lut: a forwarded read needs every writer")


def test_forwarded_read_of_a_same_slot_producer_is_rejected():
    spec = _spec()
    _implicit_read(spec, "acc", "ACC.ADD", "r_acc")["view"] = "forwarded"
    _rejects(spec, r"acc\.ACC\.ADD: r_acc: .*written by \['acc'\]")


def test_forwarded_cr_read_is_rejected():
    spec = _spec()
    _operand(spec, "load", "LD.MULT", "base")["read"] = "forwarded"
    _rejects(spec, r"load\.LD\.MULT: cr via 'base': a CR read is at cycle start")


def test_forwarded_control_slot_read_is_rejected():
    spec = _spec()
    _operand(spec, "cond", "BEQ", "reg1")["read"] = "forwarded"
    _rejects(spec, r"cond\.BEQ: lr via 'reg1': a COND \(Control slot\) read is at cycle start")


def test_a_new_undeclared_deviation_is_rejected():
    spec = _spec()
    _implicit_read(spec, "acc", "AGG.SUM", "mult_res")["view"] = "cycle_start"
    _rejects(spec, r"not in KNOWN_DEVIATIONS: acc\.AGG\.SUM mult_res")


def test_a_stale_deviation_row_is_rejected():
    """A row whose read no longer differs from the contract fails until it is deleted."""
    stale = Deviation(Slot.MULT, "MULT.EE", "r_mask", None, FW, CS, "E04-27")
    with pytest.raises(
        SpecError, match=r"KNOWN_DEVIATIONS: row\(s\) no longer differ .*MULT\.EE r_mask via None"
    ):
        validate_timing(INSTRUCTION_SPEC, SLOTS, [stale])


def test_a_listed_deviation_is_accepted():
    """A read that differs from the contract passes once it is listed."""
    spec = _spec()
    _implicit_read(spec, "mult", "MULT.EE", "r_mask")["view"] = "forwarded"
    _rejects(spec, r"not in KNOWN_DEVIATIONS: mult\.MULT\.EE r_mask")
    listed = Deviation(Slot.MULT, "MULT.EE", "r_mask", None, FW, CS, "E04-27")
    validate_timing(spec, SLOTS, [listed])
    assert [(d.mnemonic, d.register, d.today, d.contract) for d in deviations(spec)] == [
        ("MULT.EE", "r_mask", FW, CS)
    ]


def test_lr_write_target_must_be_declared():
    spec = _spec()
    del _operand(spec, "lr", "ADD", "dest")["write"]
    _rejects(spec, r"lr\.ADD: an LR-slot instruction writes one LR")


def test_lr_write_target_is_not_an_implicit_write():
    spec = _spec()
    spec["lr"]["INC"]["implicit_writes"] = [{"register": "lr", "via_operand": "dest"}]
    _rejects(spec, r"lr\.INC\.implicit_writes: an LR-slot instruction declares its LR target")


def test_execution_order_must_hold_every_slot():
    with pytest.raises(SpecError, match="SLOT_EXECUTION_ORDER"):
        validate_timing(INSTRUCTION_SPEC, [*SLOTS, "extra"])


# -- the schema of the new keys ----------------------------------------------------


@pytest.mark.parametrize(
    ("plant", "match"),
    [
        (
            lambda s: _operand(s, "lr", "INC", "imm").__setitem__("write", True),
            r"lr\.INC\.operands\[imm\]\.write: a LrIncDecImmediate operand is not a register",
        ),
        (
            lambda s: _operand(s, "lr", "ADD", "dest").__setitem__("write", 1),
            r"lr\.ADD\.operands\[dest\]\.write: must be True",
        ),
        (
            lambda s: _operand(s, "lr", "ADD", "dest").__setitem__("read", "cycle_start"),
            r"lr\.ADD\.operands\[dest\]: a 'write' operand has no 'read'",
        ),
        (
            lambda s: _implicit_read(s, "acc", "ACC.ADD", "r_acc").__setitem__("register", "racc"),
            r"acc\.ACC\.ADD\.implicit_reads\[1\]\.register: 'racc' is not a register",
        ),
        (
            lambda s: _implicit_read(s, "acc", "ACC.ADD", "r_acc").__setitem__("via_operand", "x"),
            r"acc\.ACC\.ADD\.implicit_reads\[1\]\.via_operand: 'x' is not an operand",
        ),
        (
            lambda s: _implicit_read(s, "acc", "ACC.ADD", "r_acc").__setitem__("view", "live"),
            r"acc\.ACC\.ADD\.implicit_reads\[1\]\.view: 'live' is not one of",
        ),
        (
            lambda s: _implicit_read(s, "acc", "ACC.ADD", "r_acc").pop("view"),
            r"acc\.ACC\.ADD\.implicit_reads\[1\]: missing key\(s\) 'view'",
        ),
        (
            lambda s: s["acc"]["ACC.ADD"]["implicit_writes"][0].__setitem__("view", "forwarded"),
            r"acc\.ACC\.ADD\.implicit_writes\[0\]: unknown key\(s\) 'view'",
        ),
        (
            lambda s: s["acc"]["ACC.ADD"]["implicit_reads"].append(
                {"register": "r_acc", "view": "cycle_start"}
            ),
            r"acc\.ACC\.ADD\.implicit_reads\[2\]: 'r_acc' via None is declared twice",
        ),
    ],
)
def test_schema_of_the_access_keys(plant, match):
    spec = _spec()
    plant(spec)
    with pytest.raises(SpecError, match=match):
        load_isa(spec, PSEUDO_INSTRUCTION_SPEC, SLOTS, OPERAND_TYPES)
