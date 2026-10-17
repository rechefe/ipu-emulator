"""The reference-model API v1 (E04-30): its public surface, pinned against a
snapshot, and the behaviour each method promises.

Changing the snapshot needs an API_VERSION bump; a breaking change (a name
removed or renamed, a parameter removed, a meaning changed) a major bump
(Q11).
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest
from emu_support import ElementFormat, assemble_program

import ipu_emu.reference as api
from ipu_emu.commit import CommitRecord
from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import API_VERSION, ReferenceModel, RunResult, RunState, Status
from ipu_emu.trace import TraceWriter, read_trace
from ipu_isa.codec.word import DecodedWord, pack_program
from ipu_isa.faults import FaultId
from ipu_isa.registers import REGISTER_DEFINITIONS

SNAPSHOT = {
    "API_VERSION": "1.0.0",
    "__all__": ["API_VERSION", "ReferenceModel", "RunResult", "RunState", "Status"],
    "ReferenceModel": {
        "__init__": "(self, fmt: 'ElementFormat' = <ElementFormat.INT8: 'int8'>) -> 'None'",
        "load_program": "(self, words: 'bytes | Sequence[int]') -> 'None'",
        "read_register": "(self, name: 'str', index: 'int' = 0) -> 'int | bytes'",
        "read_xmem": "(self, address: 'int', size: 'int') -> 'bytes'",
        "reset": "(self, fmt: 'ElementFormat') -> 'None'",
        "run": "(self, max_cycles: 'int | None' = None, *, trace: 'TraceWriter | None' = None)"
        " -> 'RunResult'",
        "set_fault_enable": "(self, fault: 'FaultId', enabled: 'bool') -> 'None'",
        "status": "property",
        "step": "(self) -> 'CommitRecord'",
        "write_register": "(self, name: 'str', value: 'int | bytes', index: 'int' = 0) -> 'None'",
        "write_xmem": "(self, address: 'int', data: 'bytes') -> 'None'",
    },
    "Status": ["state", "pc", "fault"],
    "RunResult": ["status", "cycles"],
    "RunState": ["RUNNING", "HALTED", "FAULTED"],
}


def _surface() -> dict:
    model: dict[str, str] = {}
    for name, member in sorted(vars(ReferenceModel).items()):
        if name.startswith("_") and name != "__init__":
            continue
        model[name] = "property" if isinstance(member, property) else str(inspect.signature(member))
    return {
        "API_VERSION": API_VERSION,
        "__all__": sorted(api.__all__),
        "ReferenceModel": model,
        "Status": [f.name for f in dataclasses.fields(Status)],
        "RunResult": [f.name for f in dataclasses.fields(RunResult)],
        "RunState": [s.name for s in RunState],
    }


def test_the_public_surface_matches_the_snapshot():
    assert _surface() == SNAPSHOT, "the API changed: bump API_VERSION (Q11) and the snapshot"


def _words(asm: str) -> list[int]:
    return [DecodedWord(w).to_word() for w in assemble_program(asm)]


LOOP = "SET LR1 CR2 ;;\nloop:\nINC LR0 1 ;;\nBNE LR0 LR1 loop ;;\nHALT ;;"


def _model(asm: str = LOOP, cr: dict | None = None) -> ReferenceModel:
    model = ReferenceModel()
    model.load_program(_words(asm))
    for index, value in (cr or {2: 5}).items():
        model.write_register("cr", value, index)
    return model


def test_step_records_equal_the_records_of_a_run(tmp_path):
    stepped = _model()
    records = []
    while stepped.status.state is not RunState.HALTED or not records:
        records.append(stepped.step())
    path = tmp_path / "t.jsonl"
    run = _model()
    with TraceWriter(path, {"schema_version": "1.0.0"}) as trace:
        result = run.run(trace=trace)
    _, traced = read_trace(path)
    assert result == RunResult(Status(RunState.HALTED, 3), 12)  # HALT is word 3
    assert [r.to_json() for r in traced] == [r.to_json() for r in records]
    assert all(isinstance(r, CommitRecord) for r in records)


def test_a_program_file_loads_as_its_words():
    model = ReferenceModel()
    model.load_program(pack_program(_words(LOOP)))
    model.write_register("cr", 3, 2)
    assert model.run().cycles == 8 and model.read_register("lr", 0) == 3


@pytest.mark.parametrize("fmt", list(ElementFormat))
def test_reset_gives_the_declared_register_widths(fmt):
    model = _model()
    model.run()
    model.reset(fmt)
    assert model.status == Status(RunState.HALTED, 0)
    for name, definition in REGISTER_DEFINITIONS.items():
        value = model.read_register(name, definition["count"] - 1)
        if definition["vector"]:
            assert isinstance(value, bytes) and len(value) == definition["size_bytes"], name
        else:
            assert isinstance(value, int), name
    assert model.read_register("lr", 0) == 0


def test_host_writes_are_refused_while_running_and_on_cr0_cr1():
    model = _model()
    model.step()
    assert model.status.state is RunState.RUNNING
    with pytest.raises(ConfigurationError, match="only while the model is halted"):
        model.write_register("lr", 1, 0)
    with pytest.raises(ConfigurationError, match="only while the model is halted"):
        model.write_xmem(0, b"\x01")
    model = _model()
    for index in (0, 1):
        with pytest.raises(ConfigurationError, match=f"CR{index} is read-only"):
            model.write_register("cr", 7, index)
    with pytest.raises(ConfigurationError, match="no register"):
        model.read_register("nope")
    with pytest.raises(ConfigurationError, match="takes 128 bytes"):
        model.write_register("r", b"\x00", 0)


def test_a_faulted_model_refuses_everything_until_reset():
    model = _model("SET LR0 CR1 ; SET LR0 CR2 ;;\nHALT ;;")
    record = model.step()
    assert record.outcome == "fault" and record.fault.id == "F-3"
    assert model.status == Status(RunState.FAULTED, 0, FaultId.F3)
    for refused in (model.step, model.run, lambda: model.write_register("lr", 1, 0)):
        with pytest.raises(ConfigurationError, match="(faulted|halted)"):
            refused()
    model.reset(ElementFormat.INT8)
    assert model.status.state is RunState.HALTED
    model.load_program(_words("HALT ;;"))
    assert model.run().status == Status(RunState.HALTED, 0)


def test_step_never_stops_at_break_and_halt_ends_the_run():
    model = _model("BREAK ;;\nINC LR0 1 ;;\nHALT ;;")
    first = model.step()
    # The BREAK slot is not evaluated (Q39): the word just retires.
    assert first.outcome == "retired" and first.pc_after == 1
    assert model.run().status == Status(RunState.HALTED, 2)
    with pytest.raises(ConfigurationError, match="halted after HALT"):
        model.step()


def test_the_cycle_limit_and_fault_enables():
    model = _model(cr={2: 100})
    result = model.run(max_cycles=10)
    # SET, then INC/BNE alternate: after 10 words the next is BNE, word 2.
    assert result.status == Status(RunState.FAULTED, 2, FaultId.F9) and result.cycles == 10
    model.reset(ElementFormat.INT8)
    model.load_program(_words(LOOP))
    model.write_register("cr", 100, 2)
    model.set_fault_enable(FaultId.F9, False)
    assert model.run(max_cycles=10).status.state is RunState.HALTED
    with pytest.raises(ConfigurationError, match="OQ-2"):
        model.set_fault_enable(FaultId.F4, False)


def test_xmem_round_trips_and_bounds_are_checked():
    model = _model()
    model.write_xmem(256, b"\x01\x02\x03")
    assert model.read_xmem(255, 5) == b"\x00\x01\x02\x03\x00"
    with pytest.raises(ConfigurationError):
        model.read_xmem(-1, 4)
    with pytest.raises(ConfigurationError):
        model.write_xmem(1 << 40, b"\x00")


def test_no_method_returns_an_emulator_object():
    model = _model()
    returned = [
        model.read_register("lr", 0),
        model.read_register("r", 0),
        model.read_xmem(0, 4),
        model.step(),
        model.run(),
        model.status,
    ]
    internal = ("IpuState", "RegFile", "Ipu", "Cycle", "XMem")
    assert not any(type(v).__name__ in internal for v in returned)
