"""Commit records and traces (E04-29): the schema contract, the collector's
records against the state, and the trace file."""

from __future__ import annotations

import hashlib
import json
from importlib import resources

import pytest
import timing_probes
from emu_support import assemble_program, make_state, new_state
from jsonschema import Draft202012Validator

from ipu_emu.commit import CommitCollector, CommitRecord, FaultInfo, RegWrite, SlotOp, XmemWrite
from ipu_emu.cycle import CycleOutcome
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.errors import EmulatorFault
from ipu_emu.execute import Ipu
from ipu_emu.trace import TraceWriter, read_trace, trace_header
from ipu_isa.registers import REGISTER_DEFINITIONS
from ipu_isa.trace_schema import SCHEMA_VERSION, load_schema

# The probes include ST.ACC, the emulator-only store: its warning is expected.
_DEBUG_ONLY = pytest.mark.filterwarnings("ignore::ipu_emu.errors.DebugOnlyInstructionWarning")

RECORD = Draft202012Validator(load_schema("commit_record"))
HEADER = Draft202012Validator(load_schema("trace"))

# The schemas' content at their current versions. A change to a schema must
# bump the version in its $id (Q11: minor for an added optional field, major
# otherwise) and then update this table.
SNAPSHOT = {
    "commit_record": "1.0.0",
    "trace": "1.0.0",
}
SNAPSHOT_SHA256 = {
    "commit_record": "d2975ca755b91b66e1484f4946976ba5133273f38be95514bed6e84de5947d99",
    "trace": "6b08e1f1e95b8f2d2fd8f1acd8abe4a6b0c7b6a219613ec5a1e19fec31deab02",
}


def _collect(state):
    with CommitCollector(state) as collector:
        try:
            run_until_complete(state, max_cycles=2_000)
        except EmulatorFault:
            pass
    return collector.records


@_DEBUG_ONLY
@pytest.mark.parametrize("name", sorted(timing_probes.PROBES))
def test_every_record_and_header_validates(name):
    state = timing_probes.prepare(timing_probes.PROBES[name], timing_probes.PROBES[name].modes[0])
    records = _collect(state)
    assert records
    for record in records:
        RECORD.validate(record.to_json())
        assert CommitRecord.from_json(json.loads(json.dumps(record.to_json()))) == record
    HEADER.validate(trace_header(state))


def test_a_record_round_trips():
    record = CommitRecord(
        cycle=3,
        pc_before=7,
        pc_after=8,
        slots=(SlotOp("lr", "SET", {"reg": 1, "src": 2}), SlotOp("acc_store", "X", {}, True)),
        reg_writes=(
            RegWrite("lr", index=1, value=5),
            RegWrite("r_acc", offset=4, data=b"\x01\x02"),
        ),
        xmem_writes=(XmemWrite(128, b"\xff" * 4),),
        outcome="fault",
        fault=FaultInfo("F-6", 7),
    )
    RECORD.validate(record.to_json())
    assert CommitRecord.from_json(record.to_json()) == record
    assert record.to_json()["slots"][1]["tag"] == "emulator"


def _registers(state):
    return {name: bytes(state.regfile.view(name)) for name in REGISTER_DEFINITIONS}


def _xmem(state):
    return {page: bytes(data) for page, data in state.xmem._pages.items()}


def _covered(record, name, byte):
    for w in record.reg_writes:
        if w.register != name:
            continue
        if w.data is None:
            width = REGISTER_DEFINITIONS[name]["size_bytes"]
            if w.index * width <= byte < (w.index + 1) * width:
                return True
        elif w.offset <= byte < w.offset + len(w.data):
            return True
    return False


def _check_registers(record, before, after):
    """Every changed register byte is in a write; every write holds the end value."""
    for reg, old in before.items():
        new = after[reg]
        for byte in (i for i in range(len(new)) if new[i] != old[i]):
            assert _covered(record, reg, byte), (reg, byte)
    for w in record.reg_writes:
        current = after[w.register]
        if w.data is None:
            width = REGISTER_DEFINITIONS[w.register]["size_bytes"]
            value = int.from_bytes(current[w.index * width : (w.index + 1) * width], "little")
            assert value == w.value
        else:
            assert current[w.offset : w.offset + len(w.data)] == w.data


def _check_xmem(record, before, state):
    """The same for XMEM, page by page."""
    after = _xmem(state)
    for page in set(before) | set(after):
        old, new = before.get(page), after.get(page)
        if old == new:
            continue
        size = len(new or old)
        old, new = old or bytes(size), new or bytes(size)
        for i in (i for i in range(size) if old[i] != new[i]):
            address = page * size + i
            assert any(
                w.address <= address < w.address + len(w.data) for w in record.xmem_writes
            ), address
    for w in record.xmem_writes:
        assert state.xmem.snapshot_bytes(w.address, len(w.data)) == w.data


@_DEBUG_ONLY
@pytest.mark.parametrize("name", sorted(timing_probes.PROBES))
def test_the_writes_are_the_state_difference(name):
    probe = timing_probes.PROBES[name]
    state = timing_probes.prepare(probe, probe.modes[0])
    ipu = Ipu(state)
    with CommitCollector(state) as collector:
        for _ in range(200):
            if state.halt is not None:
                break
            regs, mem = _registers(state), _xmem(state)
            seen = len(collector.records)
            try:
                if ipu.cycle.step() is CycleOutcome.BREAK:
                    ipu.cycle.retire()
            except EmulatorFault:
                break
            (record,) = collector.records[seen:]
            _check_registers(record, regs, _registers(state))
            _check_xmem(record, mem, state)


def test_halt_break_and_fault_outcomes():
    state = make_state("INC LR0 1 ;;\nBREAK ;;\nHALT ;;")
    records = _collect(state)
    assert [(r.pc_before, r.pc_after, r.outcome) for r in records] == [
        (0, 1, "retired"),
        (1, 2, "break"),
        (2, 1024, "halt"),
    ]
    assert records[0].slots == (SlotOp("lr", "INC", records[0].slots[0].fields),)
    assert records[0].reg_writes == (RegWrite("lr", index=0, value=1),)


def test_a_fault_cycle_names_the_fault_and_pc():
    # F-3 in the second word: the record of that word is the fault.
    state = make_state("INC LR3 1 ;;\nSET LR0 CR1 ; SET LR0 CR2 ;;\nHALT ;;")
    records = _collect(state)
    assert [r.outcome for r in records] == ["retired", "fault"]
    assert records[-1].fault == FaultInfo("F-3", 1) and records[-1].pc_before == 1
    assert {op.mnemonic for op in records[-1].slots} <= {"SET"}
    # F-2 at fetch: no word began, yet the run's end is recorded.
    state = make_state("INC LR0 1 ;;")
    records = _collect(state)
    assert records[-1].outcome == "fault" and records[-1].fault == FaultInfo("F-2", 1)
    assert records[-1].slots == () and records[-1].reg_writes == ()


def test_no_collector_attached_by_default_and_close_detaches():
    state = make_state("INC LR0 1 ;;\nHALT ;;")
    assert state.cycle_observers == ()
    collector = CommitCollector(state)
    collector.close()
    assert state.cycle_observers == ()
    run_until_complete(state)
    assert collector.records == []


def test_a_trace_file_round_trips(tmp_path):
    state = make_state("INC LR0 1 ;;\nHALT ;;")
    records = _collect(state)
    path = tmp_path / "t.jsonl"
    with TraceWriter(path, trace_header(state)) as trace:
        trace.write_all(records)
    header, back = read_trace(path)
    assert header["schema_version"] == SCHEMA_VERSION and back == records
    HEADER.validate(header)
    lines = path.read_text().splitlines()
    assert len(lines) == 1 + len(records)
    for line in lines[1:]:
        RECORD.validate(json.loads(line))


def test_the_program_digest_names_the_program():
    a, b = make_state("INC LR0 1 ;;\nHALT ;;"), make_state("INC LR0 2 ;;\nHALT ;;")
    assert trace_header(a)["program_digest"] != trace_header(b)["program_digest"]
    c = new_state()
    load_program(c, assemble_program("INC LR0 1 ;;\nHALT ;;"))
    assert trace_header(a) == trace_header(c)


@pytest.mark.parametrize("name", sorted(SNAPSHOT))
def test_a_schema_change_bumps_its_version(name):
    text = resources.files("ipu_isa").joinpath(f"schemas/{name}.v1.json").read_bytes()
    digest = hashlib.sha256(text).hexdigest()
    version = load_schema(name)["$id"].rsplit("/", 1)[1]
    if version == SNAPSHOT[name]:
        assert digest == SNAPSHOT_SHA256[name], (
            f"{name}.v1.json changed without a version bump in its $id (Q11)"
        )
    else:
        pytest.fail(f"{name} is now {version}: update SNAPSHOT and SNAPSHOT_SHA256 here")


@_DEBUG_ONLY
def test_an_emulator_only_op_is_tagged():
    probes = [p for p in timing_probes.PROBES.values() if "ST.ACC" in p.asm]
    assert probes
    state = timing_probes.prepare(probes[0], probes[0].modes[0])
    ops = [op for r in _collect(state) for op in r.slots if op.slot == "acc_store"]
    assert ops and all(op.emulator_only for op in ops)
    assert all(op.to_json()["tag"] == "emulator" for op in ops)
