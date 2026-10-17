"""Tests for the issue step: one slot op in, one immutable ``IssueFacts`` out (E04-09)."""

from __future__ import annotations

import dataclasses

import pytest
from emu_support import ElementFormat, assemble_program, new_state

from ipu_emu.emulator import load_program
from ipu_emu.ipu import LANES, Ipu
from ipu_emu.issue import IssueFacts, MultIdentity


class _NoInstMem:
    """Stands in for IMEM: any access fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"inst_mem.{name} read during issue")

    def __getitem__(self, key):
        raise AssertionError(f"inst_mem[{key!r}] read during issue")

    def __len__(self):
        raise AssertionError("len(inst_mem) read during issue")


def _engine_at_vv_multiply():
    """An engine stopped (snapshot taken) at a ``MULT.RC.VV`` whose R0 holds declared ones."""
    state = new_state(fmt=ElementFormat.INT8)
    state.write_constant_ones(0)
    state.regfile.set_cr(2, 1)
    code = (
        "LD.MULT R0 LR0 CR0;;\n"
        "LD.MULT.CYCLIC LR0 CR2 LR0;;\n"
        "MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD ;;\n"
        "HALT;;"
    )
    load_program(state, assemble_program(code))
    engine = Ipu(state)
    engine.cycle.step()
    engine.cycle.step()
    engine.cycle.evaluate_break()  # takes the cycle's snapshot; the word has no BREAK
    pc = state.program_counter
    inst = state.inst_mem[pc]
    schedule = engine.cycle.cached_schedule(pc)
    (mult,) = [op for op in schedule.slot_ops if op.slot == "mult"]
    return state, engine, inst, schedule, mult


def test_vv_identity_is_issued_without_reading_inst_mem(monkeypatch):
    state, engine, inst, _, mult = _engine_at_vv_multiply()
    monkeypatch.setattr(state, "_inst_mem", _NoInstMem())
    with pytest.raises(AssertionError):
        state.inst_mem[0]

    facts = engine.issuer.issue(mult, inst)

    assert isinstance(facts, IssueFacts)
    assert facts.slot == "mult"
    assert facts.plan.inst_name == "MULT.RC.VV"
    assert facts.mult_identity == MultIdentity(cr=False, vector=True)
    assert facts.mult_identity.any
    assert facts.alias == "A2_ADD_VV"
    # The raw field and the resolved value of ra are both on the record.
    assert facts.encoded["ra"] == 0
    assert facts.operands["ra"] != facts.encoded["ra"]
    assert state.stats.mult_identity_cycles == 1

    # The handler reads the verdict from the record, not from IMEM.
    engine.issuer.execute(facts)
    assert engine.cycle.current is facts
    assert state.stats.mult_identity_lane_ops == LANES


def test_issue_does_not_mutate_the_schedule_operands():
    _, engine, inst, _, mult = _engine_at_vv_multiply()
    before = dict(mult.operands)
    facts = engine.issuer.issue(mult, inst)
    assert mult.operands == before
    assert facts.encoded is mult.operands
    assert facts.operands is not mult.operands


def test_non_mult_ops_carry_no_identity():
    _, engine, inst, schedule, _ = _engine_at_vv_multiply()
    for op in schedule.slot_ops:
        if op.slot != "mult":
            assert engine.issuer.issue(op, inst).mult_identity is None


def test_records_are_immutable():
    _, engine, inst, _, mult = _engine_at_vv_multiply()
    facts = engine.issuer.issue(mult, inst)
    with pytest.raises(AttributeError):
        facts.alias = None  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        facts.mult_identity.cr = True  # type: ignore[misc]


def test_identity_verdicts_are_shared_instances():
    assert MultIdentity.of(True, False) is MultIdentity.of(True, False)
    assert MultIdentity.of(False, False).any is False
    assert MultIdentity.of(True, False).any and MultIdentity.of(False, True).any


def test_ipu_keeps_no_identity_side_channel():
    engine = Ipu(new_state())
    for name in ("_mult_identity", "_mult_vector_identity", "_mult_cr_identity"):
        assert not hasattr(engine, name)
