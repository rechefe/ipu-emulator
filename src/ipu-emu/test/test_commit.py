"""Commit records and traces (E04-29): the schema contract, the collector's
records against the state, and the trace file."""

from __future__ import annotations

import hashlib
import json
import struct
from importlib import resources

import pytest
import timing_probes
from emu_support import assemble_program, make_state, new_state
from jsonschema import Draft202012Validator

from ipu_emu.commit import (
    CommitCollector,
    CommitRecord,
    FaultInfo,
    RegWrite,
    SlotOp,
    XmemMetaWrite,
    XmemWrite,
)
from ipu_emu.cycle import CycleOutcome
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.errors import EmulatorFault
from ipu_emu.ipu import Ipu
from ipu_emu.trace import TraceWriter, read_trace, trace_header
from ipu_isa.codec.word import DecodedWord, word_bytes
from ipu_isa.registers import REGISTER_DEFINITIONS
from ipu_isa.trace_schema import SCHEMA_VERSION, load_schema

# No probe is debug-only since ADR 0007; the marker stays harmless.
_DEBUG_ONLY = pytest.mark.filterwarnings("ignore::ipu_emu.errors.DebugOnlyInstructionWarning")

RECORD = Draft202012Validator(load_schema("commit_record"))
HEADER = Draft202012Validator(load_schema("trace"))

# The schemas' content at their current versions. A change to a schema must
# bump the version in its $id (Q11: minor for an added optional field, major
# otherwise) and then update this table.
SNAPSHOT = {
    "commit_record": "1.1.0",
    "trace": "1.0.0",
}
SNAPSHOT_SHA256 = {
    "commit_record": "797255cc505dcc70c0f27e5794761ef95bb56097a8ebaead8714e465f5b0cadd",
    "trace": "90e26dc04251f4d85de0d750b2cf2a0f9cba7ca1ff97a9c3f10d0e712ff342ee",
}


def _collect(state):
    with CommitCollector(state) as collector:
        try:
            run_until_complete(state, max_cycles=2_000)
        except EmulatorFault:
            pass
    return collector.records


# Every probe in every mode it declares (the FP32/INT32 modes write the
# emulator-only registers).
PROBE_RUNS = [
    (name, mode) for name, probe in sorted(timing_probes.PROBES.items()) for mode in probe.modes
]


@_DEBUG_ONLY
@pytest.mark.parametrize(("name", "mode"), PROBE_RUNS, ids=str)
def test_every_record_and_header_validates(name, mode):
    state = timing_probes.prepare(timing_probes.PROBES[name], mode)
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
    )
    RECORD.validate(record.to_json())
    assert CommitRecord.from_json(record.to_json()) == record
    assert record.to_json()["slots"][1]["tag"] == "emulator"
    fault = CommitRecord(
        cycle=4, pc_before=8, pc_after=8, outcome="fault", fault=FaultInfo("F-6", 8)
    )
    RECORD.validate(fault.to_json())
    assert CommitRecord.from_json(fault.to_json()) == fault


def test_the_schema_ties_the_fault_to_the_outcome():
    ok = CommitRecord(cycle=0, pc_before=0, pc_after=1).to_json()
    for bad in (
        dict(ok, outcome="fault"),  # a fault outcome needs the fault
        dict(ok, fault={"id": "F-2", "pc": 0}),  # a retired word has none
        dict(
            ok,
            outcome="fault",
            fault={"id": "F-2", "pc": 0},
            reg_writes=[{"register": "lr", "index": 0, "value": 1}],
        ),
    ):
        assert not RECORD.is_valid(bad), bad


def _registers(state):
    return {name: bytes(state.regfile.view(name)) for name in REGISTER_DEFINITIONS}


def _xmem(state):
    return {base: bytes(data) for base, data in state.xmem.touched_pages()}


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
    """The same for XMEM, page by page, and for its metadata, unit by unit."""
    before, before_meta = before
    after_meta = dict(state.xmem.touched_metadata())
    written = {w.address: w.value for w in record.xmem_meta_writes}
    for unit in set(before_meta) | set(after_meta):
        if before_meta.get(unit, 0) != after_meta.get(unit, 0):
            assert unit in written, unit
    for unit, value in written.items():
        assert state.xmem.snapshot_metadata(unit) == value
    after = _xmem(state)
    for base in set(before) | set(after):
        old, new = before.get(base), after.get(base)
        if old == new:
            continue
        size = len(new or old)
        old, new = old or bytes(size), new or bytes(size)
        for i in (i for i in range(size) if old[i] != new[i]):
            address = base + i
            assert any(
                w.address <= address < w.address + len(w.data) for w in record.xmem_writes
            ), address
    for w in record.xmem_writes:
        assert state.xmem.snapshot_bytes(w.address, len(w.data)) == w.data


@_DEBUG_ONLY
@pytest.mark.parametrize(("name", "mode"), PROBE_RUNS, ids=str)
def test_the_writes_are_the_state_difference(name, mode):
    state = timing_probes.prepare(timing_probes.PROBES[name], mode)
    ipu = Ipu(state)
    with CommitCollector(state) as collector:
        for _ in range(200):
            if state.halt is not None:
                break
            regs, mem = _registers(state), (_xmem(state), dict(state.xmem.touched_metadata()))
            seen, pc = len(collector.records), state.program_counter
            try:
                if ipu.cycle.step() is CycleOutcome.BREAK:
                    ipu.cycle.retire()
            except EmulatorFault:
                break
            (record,) = collector.records[seen:]
            assert record.pc_before == pc and record.pc_after == state.program_counter
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
    # Nothing of a faulting word retires: no slots, no writes (E04-20).
    assert (
        records[-1].slots == () and records[-1].reg_writes == () and records[-1].xmem_writes == ()
    )
    # F-2 at fetch: no word began, yet the run's end is recorded.
    state = make_state("INC LR0 1 ;;")
    records = _collect(state)
    assert records[-1].outcome == "fault" and records[-1].fault == FaultInfo("F-2", 1)
    assert records[-1].slots == () and records[-1].reg_writes == ()


def test_take_hands_over_the_records_and_numbering_goes_on():
    state = make_state("INC LR0 1 ;;\nINC LR0 1 ;;\nHALT ;;")
    ipu = Ipu(state)
    cycle = ipu.cycle
    with CommitCollector(state) as collector:
        cycle.step()
        taken = collector.take()
        assert [r.cycle for r in taken] == [0] and collector.records == []
        cycle.step()
        cycle.step()
        assert [r.cycle for r in collector.take()] == [1, 2]
        assert collector.take() == []


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
    # The recipe trace.v1.json states: each word's encoding, word_bytes()
    # little-endian, hashed in order.
    words = b"".join(
        DecodedWord(w).to_word().to_bytes(word_bytes(), "little")
        for w in a.inst_mem[: a.program_length]
    )
    assert trace_header(a)["program_digest"] == hashlib.sha256(words).hexdigest()


def test_a_rerun_at_the_same_pc_records_its_break_once():
    from ipu_emu.emulator import DebugAction

    state = make_state("BREAK ;;\nHALT ;;")
    with CommitCollector(state) as collector:
        _debug_run(state, lambda s, c: DebugAction.QUIT, break_on_entry=True)
        state.program_counter = 0
        run_until_complete(state)
    first = collector.records[0]
    assert first.pc_before == 0 and [op.mnemonic for op in first.slots] == ["BREAK"]


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


def test_an_emulator_only_op_is_tagged():
    probes = [p for p in timing_probes.PROBES.values() if "ACTIVATE.EXACT" in p.asm]
    assert probes
    state = timing_probes.prepare(probes[0], probes[0].modes[0])
    ops = [op for r in _collect(state) for op in r.slots if op.slot == "aaq_exact"]
    assert ops and all(op.emulator_only for op in ops)
    assert all(op.to_json()["tag"] == "emulator" for op in ops)


def test_an_aaq_store_records_its_metadata_write():
    """ACTIVATE.QUANTIZE writes a row and its metadata word: both are in the record."""
    state = make_state("ACTIVATE.QUANTIZE identity CR15.L CR14.L CR2 LR0 ;;\nHALT ;;", cr={2: 30})
    state.regfile.set_r_acc_bytes(struct.pack("<128f", *([2.0] * 128)))
    (record, _halt) = _collect(state)
    assert [w.address for w in record.xmem_writes] == [30 * 128]
    assert [(w.address, w.value) for w in record.xmem_meta_writes] == [(30 * 128, 128)]
    RECORD.validate(record.to_json())


def test_a_fault_is_recorded_when_it_happens():
    # Writes after the fault (a host poking registers) are no part of it, and
    # a rerun without close() keeps the fault's record.
    state = make_state("INC LR3 1 ;;\nSET LR0 CR1 ; SET LR0 CR2 ;;\nHALT ;;")
    collector = CommitCollector(state)
    with pytest.raises(EmulatorFault):
        run_until_complete(state)
    state.regfile.set_lr(5, 99)
    state.program_counter = 2
    run_until_complete(state)
    records = collector.close()
    assert [(r.pc_before, r.outcome) for r in records] == [
        (0, "retired"),
        (1, "fault"),
        (2, "halt"),
    ]
    assert records[1].reg_writes == ()


def _debug_run(state, callback, **kw):
    from ipu_emu.emulator import run_with_debug

    return run_with_debug(state, callback, **kw)


def test_a_break_belongs_to_its_own_word_when_the_debugger_moves_the_pc():
    from ipu_emu.emulator import DebugAction

    state = make_state("INC LR0 1 ;;\nBREAK ;;\nINC LR1 1 ;;\nINC LR2 1 ;;\nHALT ;;")

    def callback(state, cycles):
        state.program_counter = 3  # skip to the third INC
        return DebugAction.CONTINUE

    with CommitCollector(state) as collector:
        _debug_run(state, callback)
    by_pc = {r.pc_before: r for r in collector.records}
    assert by_pc[3].outcome == "retired"
    assert [op.mnemonic for op in by_pc[3].slots] == ["INC"]
    assert 1 not in by_pc  # the stopped BREAK word never retired


def test_quit_then_rerun_leaves_no_stale_break():
    from ipu_emu.emulator import DebugAction

    state = make_state("INC LR0 1 ;;\nBREAK ;;\nHALT ;;")
    with CommitCollector(state) as collector:
        _debug_run(state, lambda s, c: DebugAction.QUIT)
        state.program_counter = 2
        run_until_complete(state)
    last = collector.records[-1]
    assert last.outcome == "halt" and [op.mnemonic for op in last.slots] == ["HALT"]


@_DEBUG_ONLY
def test_a_wide_debug_register_write_is_tagged():
    runs = [(n, m) for n, m in PROBE_RUNS if "WIDE" in str(m)]
    assert runs
    tagged = []
    for name, mode in runs:
        state = timing_probes.prepare(timing_probes.PROBES[name], mode)
        for record in _collect(state):
            tagged += [w.to_json() for w in record.reg_writes if w.register.endswith("_wide_debug")]
    assert tagged and all(w["tag"] == "emulator" for w in tagged)


def test_a_metadata_write_is_recorded_per_unit_with_its_end_value():
    state = make_state("INC LR0 1 ;;\nHALT ;;")
    with CommitCollector(state) as collector:
        state.xmem.write_metadata(512, 9)  # a host write: no word, no record
        collector.begin(0)
        state.xmem.write_metadata(256, 0x1AB)
        state.xmem.write_metadata(0, 5)
        state.xmem.write_metadata(256, 0x17F)  # the end-of-cycle value wins
        collector.end()
    (record,) = collector.records
    assert record.xmem_meta_writes == (XmemMetaWrite(0, 5), XmemMetaWrite(256, 0x17F))
    doc = record.to_json()
    assert doc["xmem_meta_writes"] == [{"address": 0, "value": 5}, {"address": 256, "value": 383}]
    RECORD.validate(doc)
    assert CommitRecord.from_json(json.loads(json.dumps(doc))) == record


def test_a_record_without_metadata_writes_leaves_the_field_out():
    record = CommitRecord(cycle=0, pc_before=0, pc_after=1)
    assert "xmem_meta_writes" not in record.to_json()
    assert CommitRecord.from_json(record.to_json()).xmem_meta_writes == ()


def test_the_schema_bounds_a_metadata_write():
    ok = CommitRecord(cycle=0, pc_before=0, pc_after=1).to_json()
    assert RECORD.is_valid(dict(ok, xmem_meta_writes=[{"address": 128, "value": 0xFFFF}]))
    for bad in (
        [{"address": 129, "value": 1}],  # not a 128-byte unit
        [{"address": 128, "value": 0x10000}],  # wider than 16 bits
        [{"address": 128}],
    ):
        assert not RECORD.is_valid(dict(ok, xmem_meta_writes=bad)), bad
    fault = dict(ok, outcome="fault", fault={"id": "F-6", "pc": 0})
    assert RECORD.is_valid(fault)
    assert not RECORD.is_valid(dict(fault, xmem_meta_writes=[{"address": 0, "value": 1}]))
