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
    RegisterAccess,
    View,
    contract_view,
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
        Slot.STORE,
        Slot.ACC_STORE,
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


@pytest.mark.parametrize("slot", [Slot.LOAD, Slot.MULT, Slot.ACC, Slot.STORE, Slot.ACC_STORE])
def test_rule_1_other_stages_read_lrs_forwarded(slot):
    assert contract_view(slot, "lr", WRITERS) is FW


@pytest.mark.parametrize("register", ["r", "r_cyclic", "r_mask"])
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
    assert contract_view(Slot.ACC_STORE, "r_acc", WRITERS) is FW
    assert contract_view(Slot.STORE, "post_aaq_reg", WRITERS) is FW
    assert contract_view(Slot.LOAD, "cr", WRITERS) is CS
    writers = {"r_acc": [Slot.ACC, Slot.COND]}  # a later writer: cycle start
    assert contract_view(Slot.ACC_STORE, "r_acc", writers) is CS


def test_register_writers():
    assert register_writers(INSTRUCTION_SPEC) == {
        "lr": {Slot.LR},
        "r": {Slot.LOAD},
        "r_cyclic": {Slot.LOAD},
        "r_mask": {Slot.LOAD},
        "mult_res": {Slot.MULT},
        "r_acc": {Slot.ACC},
        "post_aaq_reg": {Slot.AAQ},
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
    inc = isa().instruction(Slot.LR, "INC")
    assert inc.reads() == (RegisterAccess("lr", "dest", CS),)


def test_every_non_nop_instruction_but_a_branch_writes_something():
    """COND and BREAK change the PC or halt; every other instruction writes a
    register or XMEM (STORE, ACC_STORE)."""
    for slot in (Slot.LR, Slot.LOAD, Slot.MULT, Slot.ACC, Slot.AAQ):
        for instruction in isa().instructions(slot):
            if instruction.mnemonic != "NOP":
                assert instruction.writes(), f"{slot}.{instruction.mnemonic}"


def test_known_deviations_are_the_issue_table():
    rows = {(d.mnemonic, d.register, d.via_operand, d.today, d.fixed_by) for d in KNOWN_DEVIATIONS}
    mult = ("MULT.RC.VV", "MULT.RC.VE", "MULT.RC.VS", "MULT.VE", "MULT.EE")
    assert rows == {
        *((m, "r_mask", None, FW, "E04-27") for m in mult),
    }
    assert all(d.contract is not d.today for d in KNOWN_DEVIATIONS)
    assert set(deviations(INSTRUCTION_SPEC)) == set(KNOWN_DEVIATIONS)


# -- each validator rule fails on a planted violation -----------------------------


def test_forwarded_read_from_a_later_slot_is_rejected():
    spec = _spec()
    _implicit_read(spec, "mult", "MULT.RC.VV", "r_mask")["register"] = "post_aaq_reg"
    _rejects(spec, r"mult\.MULT\.RC\.VV: post_aaq_reg: a forwarded read needs every writer")


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
    """Fixing a deviation (E04-27 flips the read) fails until its row is deleted."""
    spec = _spec()
    _implicit_read(spec, "mult", "MULT.EE", "r_mask")["view"] = "cycle_start"
    _rejects(spec, r"KNOWN_DEVIATIONS: row\(s\) no longer differ .*MULT\.EE r_mask via None")
    remaining = [
        d for d in KNOWN_DEVIATIONS if not (d.mnemonic == "MULT.EE" and d.register == "r_mask")
    ]
    validate_timing(spec, SLOTS, remaining)


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
