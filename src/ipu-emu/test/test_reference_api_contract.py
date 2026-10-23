"""The reference-model API v1 (E04-30): its public surface, pinned against a
snapshot of its version, and the behaviour each method promises.

A change to the surface is a new version's snapshot (Q11): a minor bump for
an addition, a major bump for a breaking change (a name removed or renamed, a
parameter removed, a meaning changed).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import inspect
import struct
import typing

import pytest
from emu_support import QF_E4, ElementFormat, assemble_program

import ipu_emu.reference as api
from ipu_emu.alias_profile import AliasProfile
from ipu_emu.commit import (
    CommitRecord,
    FaultInfo,
    Outcome,
    RegWrite,
    SlotOp,
    XmemMetaWrite,
    XmemWrite,
)
from ipu_emu.emulator import DebugAction
from ipu_emu.errors import ConfigurationError
from ipu_emu.ipu_state import INST_MEM_SIZE
from ipu_emu.reference import API_VERSION, ReferenceModel, RunResult, RunState, Stats, Status
from ipu_emu.trace import TraceWriter, read_trace
from ipu_isa.codec.word import DecodedWord, pack_program, word_bytes
from ipu_isa.faults import ArithmeticEvent, FaultId
from ipu_isa.registers import REGISTER_DEFINITIONS

#: The public surface of each released version. Never edit an existing
#: version's entry: a change to the surface is a new version (Q11), so the
#: test fails until API_VERSION names an entry that matches.
SNAPSHOTS = {
    "1.0.0": {
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
            "write_register": "(self, name: 'str', value: 'int | bytes', index: 'int' = 0)"
            " -> 'None'",
            "write_xmem": "(self, address: 'int', data: 'bytes') -> 'None'",
        },
        "Status": {"state": "RunState", "pc": "int", "fault": "FaultId | None"},
        "RunResult": {"status": "Status", "cycles": "int"},
        "RunState": {"RUNNING": "running", "HALTED": "halted", "FAULTED": "faulted"},
        # What step() returns (ipu_emu.commit), field by field.
        "CommitRecord": {
            "cycle": "int",
            "pc_before": "int",
            "pc_after": "int",
            "slots": "tuple[SlotOp, ...]",
            "reg_writes": "tuple[RegWrite, ...]",
            "xmem_writes": "tuple[XmemWrite, ...]",
            "outcome": "Outcome",
            "fault": "FaultInfo | None",
            "arith_events": "tuple[str, ...]",
        },
        "Outcome": ["retired", "break", "halt", "fault"],
        "SlotOp": {
            "slot": "str",
            "mnemonic": "str",
            "fields": "Mapping[str, int]",
            "emulator_only": "bool",
        },
        "RegWrite": {
            "register": "str",
            "index": "int | None",
            "value": "int | None",
            "offset": "int | None",
            "data": "bytes | None",
        },
        "XmemWrite": {"address": "int", "data": "bytes"},
        "FaultInfo": {"id": "str", "pc": "int"},
    },
    "1.1.0": {
        "__all__": ["API_VERSION", "ReferenceModel", "RunResult", "RunState", "Stats", "Status"],
        "ReferenceModel": {
            "__init__": "(self, fmt: 'ElementFormat' = <ElementFormat.INT8: 'int8'>) -> 'None'",
            "load_program": "(self, words: 'bytes | Sequence[int]') -> 'None'",
            "read_register": "(self, name: 'str', index: 'int' = 0) -> 'int | bytes'",
            "read_xmem": "(self, address: 'int', size: 'int') -> 'bytes'",
            "reset": "(self, fmt: 'ElementFormat') -> 'None'",
            "run": "(self, max_cycles: 'int | None' = None, *, trace: 'TraceWriter | None' = None)"
            " -> 'RunResult'",
            "set_fault_enable": "(self, fault: 'FaultId', enabled: 'bool') -> 'None'",
            "stats": "(self) -> 'Stats'",
            "status": "property",
            "step": "(self) -> 'CommitRecord'",
            "write_register": "(self, name: 'str', value: 'int | bytes', index: 'int' = 0)"
            " -> 'None'",
            "write_xmem": "(self, address: 'int', data: 'bytes') -> 'None'",
        },
        "Status": {"state": "RunState", "pc": "int", "fault": "FaultId | None"},
        "RunResult": {"status": "Status", "cycles": "int"},
        "Stats": {
            "cycles": "int",
            "mult_active_cycles": "int",
            "acc_active_cycles": "int",
            "xmem_reads": "int",
            "xmem_writes": "int",
            "mult_lane_ops": "int",
            "mult_identity_lane_ops": "int",
            "lane_occupancy": "float",
            "effective_mac_utilization": "float",
        },
        "RunState": {"RUNNING": "running", "HALTED": "halted", "FAULTED": "faulted"},
        # What step() returns (ipu_emu.commit), field by field.
        "CommitRecord": {
            "cycle": "int",
            "pc_before": "int",
            "pc_after": "int",
            "slots": "tuple[SlotOp, ...]",
            "reg_writes": "tuple[RegWrite, ...]",
            "xmem_writes": "tuple[XmemWrite, ...]",
            "outcome": "Outcome",
            "fault": "FaultInfo | None",
            "arith_events": "tuple[str, ...]",
        },
        "Outcome": ["retired", "break", "halt", "fault"],
        "SlotOp": {
            "slot": "str",
            "mnemonic": "str",
            "fields": "Mapping[str, int]",
            "emulator_only": "bool",
        },
        "RegWrite": {
            "register": "str",
            "index": "int | None",
            "value": "int | None",
            "offset": "int | None",
            "data": "bytes | None",
        },
        "XmemWrite": {"address": "int", "data": "bytes"},
        "FaultInfo": {"id": "str", "pc": "int"},
    },
    "1.2.0": {
        "__all__": [
            "API_VERSION",
            "ReferenceModel",
            "RunResult",
            "RunState",
            "Stats",
            "Status",
            "TraceWriter",
            "read_trace",
        ],
        "ReferenceModel": {
            "__init__": "(self, fmt: 'ElementFormat' = <ElementFormat.INT8: 'int8'>) -> 'None'",
            "arith_flags": "(self) -> 'frozenset[ArithmeticEvent]'",
            "fault_enabled": "(self, fault: 'FaultId') -> 'bool'",
            "fault_flags": "(self) -> 'frozenset[FaultId]'",
            "load_program": "(self, words: 'bytes | Sequence[int]') -> 'None'",
            "read_register": "(self, name: 'str', index: 'int' = 0) -> 'int | bytes'",
            "read_xmem": "(self, address: 'int', size: 'int') -> 'bytes'",
            "reset": "(self, fmt: 'ElementFormat') -> 'None'",
            "run": "(self, max_cycles: 'int | None' = None, *, trace: 'TraceWriter | None' = None)"
            " -> 'RunResult'",
            "set_fault_enable": "(self, fault: 'FaultId', enabled: 'bool') -> 'None'",
            "stats": "(self) -> 'Stats'",
            "status": "property",
            "step": "(self) -> 'CommitRecord'",
            "trace_header": "(self) -> 'dict[str, Any]'",
            "write_register": "(self, name: 'str', value: 'int | bytes', index: 'int' = 0)"
            " -> 'None'",
            "write_xmem": "(self, address: 'int', data: 'bytes') -> 'None'",
        },
        "Status": {"state": "RunState", "pc": "int", "fault": "FaultId | None"},
        "RunResult": {"status": "Status", "cycles": "int"},
        "Stats": {
            "cycles": "int",
            "mult_active_cycles": "int",
            "acc_active_cycles": "int",
            "xmem_reads": "int",
            "xmem_writes": "int",
            "mult_lane_ops": "int",
            "mult_identity_lane_ops": "int",
            "lane_occupancy": "float",
            "effective_mac_utilization": "float",
        },
        "RunState": {"RUNNING": "running", "HALTED": "halted", "FAULTED": "faulted"},
        # What step() returns (ipu_emu.commit), field by field.
        "CommitRecord": {
            "cycle": "int",
            "pc_before": "int",
            "pc_after": "int",
            "slots": "tuple[SlotOp, ...]",
            "reg_writes": "tuple[RegWrite, ...]",
            "xmem_writes": "tuple[XmemWrite, ...]",
            "outcome": "Outcome",
            "fault": "FaultInfo | None",
            "arith_events": "tuple[str, ...]",
        },
        "Outcome": ["retired", "break", "halt", "fault"],
        "SlotOp": {
            "slot": "str",
            "mnemonic": "str",
            "fields": "Mapping[str, int]",
            "emulator_only": "bool",
        },
        "RegWrite": {
            "register": "str",
            "index": "int | None",
            "value": "int | None",
            "offset": "int | None",
            "data": "bytes | None",
        },
        "XmemWrite": {"address": "int", "data": "bytes"},
        "TraceWriter": {
            "__init__": "(self, path: 'str | Path', header: 'dict[str, Any]') -> 'None'",
            "close": "(self) -> 'None'",
            "write": "(self, record: 'CommitRecord') -> 'None'",
            "write_all": "(self, records: 'Iterable[CommitRecord]') -> 'None'",
        },
        "read_trace": "(path: 'str | Path') -> 'tuple[dict[str, Any], list[CommitRecord]]'",
        "FaultInfo": {"id": "str", "pc": "int"},
    },
    "1.3.0": {
        "__all__": [
            "API_VERSION",
            "ReferenceModel",
            "RunResult",
            "RunState",
            "Stats",
            "Status",
            "TraceWriter",
            "read_trace",
        ],
        "ReferenceModel": {
            "__init__": "(self, fmt: 'ElementFormat' = <ElementFormat.INT8: 'int8'>) -> 'None'",
            "alias_profile": "(self) -> 'AliasProfile | None'",
            "arith_flags": "(self) -> 'frozenset[ArithmeticEvent]'",
            "attach_alias_profile": "(self, profile: 'AliasProfile') -> 'None'",
            "debug_run": "(self, callback: 'DebugCallback', break_on_entry: 'bool' = False, *, "
            "max_cycles: 'int | None' = None, control: 'RunControl | None' = None)"
            " -> 'RunResult'",
            "fault_enabled": "(self, fault: 'FaultId') -> 'bool'",
            "fault_flags": "(self) -> 'frozenset[FaultId]'",
            "load_program": "(self, words: 'bytes | Sequence[int]') -> 'None'",
            "read_register": "(self, name: 'str', index: 'int' = 0) -> 'int | bytes'",
            "read_xmem": "(self, address: 'int', size: 'int') -> 'bytes'",
            "reset": "(self, fmt: 'ElementFormat') -> 'None'",
            "run": "(self, max_cycles: 'int | None' = None, *, trace: 'TraceWriter | None' = None)"
            " -> 'RunResult'",
            "set_fault_enable": "(self, fault: 'FaultId', enabled: 'bool') -> 'None'",
            "stats": "(self) -> 'Stats'",
            "status": "property",
            "step": "(self) -> 'CommitRecord'",
            "trace_header": "(self) -> 'dict[str, Any]'",
            "write_register": "(self, name: 'str', value: 'int | bytes', index: 'int' = 0)"
            " -> 'None'",
            "write_xmem": "(self, address: 'int', data: 'bytes') -> 'None'",
        },
        "Status": {"state": "RunState", "pc": "int", "fault": "FaultId | None"},
        "RunResult": {"status": "Status", "cycles": "int", "quit": "bool"},
        "Stats": {
            "cycles": "int",
            "mult_active_cycles": "int",
            "acc_active_cycles": "int",
            "xmem_reads": "int",
            "xmem_writes": "int",
            "mult_lane_ops": "int",
            "mult_identity_lane_ops": "int",
            "lane_occupancy": "float",
            "effective_mac_utilization": "float",
        },
        "RunState": {"RUNNING": "running", "HALTED": "halted", "FAULTED": "faulted"},
        # What step() returns (ipu_emu.commit), field by field.
        "CommitRecord": {
            "cycle": "int",
            "pc_before": "int",
            "pc_after": "int",
            "slots": "tuple[SlotOp, ...]",
            "reg_writes": "tuple[RegWrite, ...]",
            "xmem_writes": "tuple[XmemWrite, ...]",
            "outcome": "Outcome",
            "fault": "FaultInfo | None",
            "arith_events": "tuple[str, ...]",
        },
        "Outcome": ["retired", "break", "halt", "fault"],
        "SlotOp": {
            "slot": "str",
            "mnemonic": "str",
            "fields": "Mapping[str, int]",
            "emulator_only": "bool",
        },
        "RegWrite": {
            "register": "str",
            "index": "int | None",
            "value": "int | None",
            "offset": "int | None",
            "data": "bytes | None",
        },
        "XmemWrite": {"address": "int", "data": "bytes"},
        "TraceWriter": {
            "__init__": "(self, path: 'str | Path', header: 'dict[str, Any]') -> 'None'",
            "close": "(self) -> 'None'",
            "write": "(self, record: 'CommitRecord') -> 'None'",
            "write_all": "(self, records: 'Iterable[CommitRecord]') -> 'None'",
        },
        "read_trace": "(path: 'str | Path') -> 'tuple[dict[str, Any], list[CommitRecord]]'",
        "FaultInfo": {"id": "str", "pc": "int"},
    },
    "1.4.0": {
        "__all__": [
            "API_VERSION",
            "ReferenceModel",
            "RunResult",
            "RunState",
            "Stats",
            "Status",
            "TraceWriter",
            "read_trace",
        ],
        "ReferenceModel": {
            "__init__": "(self, fmt: 'ElementFormat' = <ElementFormat.INT8: 'int8'>) -> 'None'",
            "alias_profile": "(self) -> 'AliasProfile | None'",
            "arith_flags": "(self) -> 'frozenset[ArithmeticEvent]'",
            "attach_alias_profile": "(self, profile: 'AliasProfile') -> 'None'",
            "debug_run": "(self, callback: 'DebugCallback', break_on_entry: 'bool' = False, *, "
            "max_cycles: 'int | None' = None, control: 'RunControl | None' = None)"
            " -> 'RunResult'",
            "fault_enabled": "(self, fault: 'FaultId') -> 'bool'",
            "fault_flags": "(self) -> 'frozenset[FaultId]'",
            "load_program": "(self, words: 'bytes | Sequence[int]') -> 'None'",
            "observe_xmem_reads": "(self, callback: 'Callable[[int, int], None]')"
            " -> 'Callable[[], None]'",
            "read_register": "(self, name: 'str', index: 'int' = 0) -> 'int | bytes'",
            "read_xmem": "(self, address: 'int', size: 'int') -> 'bytes'",
            "reset": "(self, fmt: 'ElementFormat') -> 'None'",
            "run": "(self, max_cycles: 'int | None' = None, *, trace: 'TraceWriter | None' = None)"
            " -> 'RunResult'",
            "set_fault_enable": "(self, fault: 'FaultId', enabled: 'bool') -> 'None'",
            "stats": "(self) -> 'Stats'",
            "status": "property",
            "step": "(self) -> 'CommitRecord'",
            "trace_header": "(self) -> 'dict[str, Any]'",
            "write_register": "(self, name: 'str', value: 'int | bytes', index: 'int' = 0)"
            " -> 'None'",
            "write_xmem": "(self, address: 'int', data: 'bytes') -> 'None'",
        },
        "Status": {"state": "RunState", "pc": "int", "fault": "FaultId | None"},
        "RunResult": {"status": "Status", "cycles": "int", "quit": "bool"},
        "Stats": {
            "cycles": "int",
            "mult_active_cycles": "int",
            "acc_active_cycles": "int",
            "xmem_reads": "int",
            "xmem_writes": "int",
            "mult_lane_ops": "int",
            "mult_identity_lane_ops": "int",
            "lane_occupancy": "float",
            "effective_mac_utilization": "float",
        },
        "RunState": {"RUNNING": "running", "HALTED": "halted", "FAULTED": "faulted"},
        # What step() returns (ipu_emu.commit), field by field.
        "CommitRecord": {
            "cycle": "int",
            "pc_before": "int",
            "pc_after": "int",
            "slots": "tuple[SlotOp, ...]",
            "reg_writes": "tuple[RegWrite, ...]",
            "xmem_writes": "tuple[XmemWrite, ...]",
            "outcome": "Outcome",
            "fault": "FaultInfo | None",
            "arith_events": "tuple[str, ...]",
        },
        "Outcome": ["retired", "break", "halt", "fault"],
        "SlotOp": {
            "slot": "str",
            "mnemonic": "str",
            "fields": "Mapping[str, int]",
            "emulator_only": "bool",
        },
        "RegWrite": {
            "register": "str",
            "index": "int | None",
            "value": "int | None",
            "offset": "int | None",
            "data": "bytes | None",
        },
        "XmemWrite": {"address": "int", "data": "bytes"},
        "TraceWriter": {
            "__init__": "(self, path: 'str | Path', header: 'dict[str, Any]') -> 'None'",
            "close": "(self) -> 'None'",
            "write": "(self, record: 'CommitRecord') -> 'None'",
            "write_all": "(self, records: 'Iterable[CommitRecord]') -> 'None'",
        },
        "read_trace": "(path: 'str | Path') -> 'tuple[dict[str, Any], list[CommitRecord]]'",
        "FaultInfo": {"id": "str", "pc": "int"},
    },
    # 2.0.0 (ADR 0007): the run modes are NX8 and FP32. ElementFormat lost INT8,
    # E1-E7 and INT32 and NX8's value is "nx8", so a caller that named a removed
    # format breaks: a major bump. It also adds the XMEM metadata accessors and
    # CommitRecord.xmem_meta_writes.
    "2.0.0": {
        "__all__": [
            "API_VERSION",
            "ReferenceModel",
            "RunResult",
            "RunState",
            "Stats",
            "Status",
            "TraceWriter",
            "read_trace",
        ],
        "ReferenceModel": {
            "__init__": "(self, fmt: 'ElementFormat' = <ElementFormat.NX8: 'nx8'>) -> 'None'",
            "alias_profile": "(self) -> 'AliasProfile | None'",
            "arith_flags": "(self) -> 'frozenset[ArithmeticEvent]'",
            "attach_alias_profile": "(self, profile: 'AliasProfile') -> 'None'",
            "debug_run": "(self, callback: 'DebugCallback', break_on_entry: 'bool' = False, *, "
            "max_cycles: 'int | None' = None, control: 'RunControl | None' = None)"
            " -> 'RunResult'",
            "fault_enabled": "(self, fault: 'FaultId') -> 'bool'",
            "fault_flags": "(self) -> 'frozenset[FaultId]'",
            "load_program": "(self, words: 'bytes | Sequence[int]') -> 'None'",
            "observe_xmem_reads": "(self, callback: 'Callable[[int, int], None]')"
            " -> 'Callable[[], None]'",
            "read_register": "(self, name: 'str', index: 'int' = 0) -> 'int | bytes'",
            "read_xmem": "(self, address: 'int', size: 'int') -> 'bytes'",
            "read_xmem_metadata": "(self, address: 'int') -> 'int'",
            "reset": "(self, fmt: 'ElementFormat') -> 'None'",
            "run": "(self, max_cycles: 'int | None' = None, *, trace: 'TraceWriter | None' = None)"
            " -> 'RunResult'",
            "set_fault_enable": "(self, fault: 'FaultId', enabled: 'bool') -> 'None'",
            "stats": "(self) -> 'Stats'",
            "status": "property",
            "step": "(self) -> 'CommitRecord'",
            "trace_header": "(self) -> 'dict[str, Any]'",
            "write_register": "(self, name: 'str', value: 'int | bytes', index: 'int' = 0)"
            " -> 'None'",
            "write_xmem": "(self, address: 'int', data: 'bytes') -> 'None'",
            "write_xmem_metadata": "(self, address: 'int', word: 'int') -> 'None'",
        },
        "Status": {"state": "RunState", "pc": "int", "fault": "FaultId | None"},
        "RunResult": {"status": "Status", "cycles": "int", "quit": "bool"},
        "Stats": {
            "cycles": "int",
            "mult_active_cycles": "int",
            "acc_active_cycles": "int",
            "xmem_reads": "int",
            "xmem_writes": "int",
            "mult_lane_ops": "int",
            "mult_identity_lane_ops": "int",
            "lane_occupancy": "float",
            "effective_mac_utilization": "float",
        },
        "RunState": {"RUNNING": "running", "HALTED": "halted", "FAULTED": "faulted"},
        # What step() returns (ipu_emu.commit), field by field.
        "CommitRecord": {
            "cycle": "int",
            "pc_before": "int",
            "pc_after": "int",
            "slots": "tuple[SlotOp, ...]",
            "reg_writes": "tuple[RegWrite, ...]",
            "xmem_writes": "tuple[XmemWrite, ...]",
            "outcome": "Outcome",
            "fault": "FaultInfo | None",
            "arith_events": "tuple[str, ...]",
            "xmem_meta_writes": "tuple[XmemMetaWrite, ...]",
        },
        "Outcome": ["retired", "break", "halt", "fault"],
        "SlotOp": {
            "slot": "str",
            "mnemonic": "str",
            "fields": "Mapping[str, int]",
            "emulator_only": "bool",
        },
        "RegWrite": {
            "register": "str",
            "index": "int | None",
            "value": "int | None",
            "offset": "int | None",
            "data": "bytes | None",
        },
        "XmemWrite": {"address": "int", "data": "bytes"},
        "XmemMetaWrite": {"address": "int", "value": "int"},
        "TraceWriter": {
            "__init__": "(self, path: 'str | Path', header: 'dict[str, Any]') -> 'None'",
            "close": "(self) -> 'None'",
            "write": "(self, record: 'CommitRecord') -> 'None'",
            "write_all": "(self, records: 'Iterable[CommitRecord]') -> 'None'",
        },
        "read_trace": "(path: 'str | Path') -> 'tuple[dict[str, Any], list[CommitRecord]]'",
        "FaultInfo": {"id": "str", "pc": "int"},
    },
}

_RECORD_TYPES = (CommitRecord, SlotOp, RegWrite, XmemWrite, XmemMetaWrite, FaultInfo)


def _fields(cls: type) -> dict[str, str]:
    return {f.name: str(f.type) for f in dataclasses.fields(cls)}


def _surface() -> dict:
    model: dict[str, str] = {}
    for name, member in sorted(vars(ReferenceModel).items()):
        if name.startswith("_") and name != "__init__":
            continue
        model[name] = "property" if isinstance(member, property) else str(inspect.signature(member))
    return {
        "__all__": sorted(api.__all__),
        "ReferenceModel": model,
        "Status": _fields(Status),
        "RunResult": _fields(RunResult),
        "Stats": _fields(Stats),
        "RunState": {s.name: s.value for s in RunState},
        **{cls.__name__: _fields(cls) for cls in _RECORD_TYPES},
        "Outcome": list(typing.get_args(Outcome)),
        "TraceWriter": {
            name: str(inspect.signature(getattr(api.TraceWriter, name)))
            for name in ("__init__", "close", "write", "write_all")
        },
        "read_trace": str(inspect.signature(api.read_trace)),
    }


def test_the_public_surface_matches_the_snapshot_of_its_version():
    assert API_VERSION in SNAPSHOTS, f"no snapshot of API_VERSION {API_VERSION}"
    assert _surface() == SNAPSHOTS[API_VERSION], (
        "the API changed: add a snapshot under a new API_VERSION (Q11)"
    )


def _words(asm: str) -> list[int]:
    return [DecodedWord(w).to_word() for w in assemble_program(asm)]


LOOP = "SET LR1 CR2 ;;\nloop:\nINC LR0 1 ;;\nBNE LR0 LR1 loop ;;\nHALT ;;"
FAULT = "SET LR0 CR1 ; SET LR0 CR2 ;;\nHALT ;;"  # two writes of LR0: F-3


def _model(asm: str = LOOP, cr: dict | None = None) -> ReferenceModel:
    model = ReferenceModel()
    model.load_program(_words(asm))
    model.write_register("cr", QF_E4, 14)  # CR14.L: the programs' QF half
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
    assert [r.cycle for r in records] == list(range(12))
    assert [r.to_json() for r in traced] == [r.to_json() for r in records]
    assert all(isinstance(r, CommitRecord) for r in records)


def test_a_faulting_run_traces_the_records_of_its_steps(tmp_path):
    stepped = _model("INC LR0 1 ;;\n" + FAULT)
    records = [stepped.step(), stepped.step()]
    path = tmp_path / "t.jsonl"
    with TraceWriter(path, {"schema_version": "1.0.0"}) as trace:
        result = _model("INC LR0 1 ;;\n" + FAULT).run(trace=trace)
    _, traced = read_trace(path)
    assert result == RunResult(Status(RunState.FAULTED, 1, FaultId.F3), 1)
    assert [r.outcome for r in records] == ["retired", "fault"]
    assert [r.to_json() for r in traced] == [r.to_json() for r in records]


def test_a_trace_numbers_its_records_from_zero_and_steps_from_the_load(tmp_path):
    model = _model()
    assert [model.step().cycle for _ in range(2)] == [0, 1]
    path = tmp_path / "t.jsonl"
    with TraceWriter(path, {"schema_version": "1.0.0"}) as trace:
        model.run(trace=trace)
    _, traced = read_trace(path)
    assert [r.cycle for r in traced] == list(range(10))  # the schema's cycle
    model.load_program(_words(LOOP))
    assert model.step().cycle == 0
    model.reset(ElementFormat.NX8)
    assert model.step().cycle == 0


def test_steps_and_runs_add_up_in_the_cycle_count():
    model = _model()
    model.step()
    model.step()
    assert model.run().cycles == 10
    assert model.stats().cycles == 12


def test_a_program_file_loads_as_its_words():
    model = ReferenceModel()
    model.load_program(pack_program(_words(LOOP)))
    model.write_register("cr", 3, 2)
    assert model.run().cycles == 8 and model.read_register("lr", 0) == 3


@pytest.mark.parametrize(
    "words",
    [b"\x00" * (word_bytes() - 1), [-1], [True], [1 << (8 * word_bytes())]],
    ids=["partial-word", "negative", "bool", "too-wide"],
)
def test_a_malformed_program_is_refused(words):
    with pytest.raises(ConfigurationError, match="word"):
        ReferenceModel().load_program(words)


@pytest.mark.parametrize("fmt", list(ElementFormat))
def test_reset_restores_every_reset_value_in_each_format(fmt):
    model = _model()
    model.write_register("r", b"\x05" * 128, 1)
    model.write_xmem(512, b"\x07" * 4)
    model.write_register("cr", 0x1234, 15)
    model.set_fault_enable(FaultId.F9, False)
    model.run(max_cycles=3)  # F-9 off: the run goes past the limit to HALT
    model.reset(fmt)
    assert model.status == Status(RunState.HALTED, 0)
    fresh = ReferenceModel(fmt)
    for name, definition in REGISTER_DEFINITIONS.items():
        for index in range(definition["count"]):
            value = model.read_register(name, index)
            assert value == fresh.read_register(name, index), (name, index)
            if definition["vector"]:
                assert len(value) == definition["size_bytes"], name
            else:
                assert isinstance(value, int), name
    assert model.read_register("lr", 0) == 0 and model.read_register("cr", 15) != 0x1234
    assert model.read_xmem(512, 4) == bytes(4)
    # F-9 is enabled again: a 3-cycle limit faults the 12-cycle loop.
    model.write_register("cr", 5, 2)
    assert model.run(max_cycles=3).status.fault is FaultId.F9
    # The program stayed.
    model.reset(fmt)
    model.write_register("cr", 5, 2)
    assert model.run().cycles == 12


def test_host_writes_are_refused_while_running():
    model = _model()
    model.step()
    assert model.status.state is RunState.RUNNING
    for refused in (
        lambda: model.write_register("lr", 1, 0),
        lambda: model.write_xmem(0, b"\x01"),
        lambda: model.load_program(_words("HALT ;;")),
    ):
        with pytest.raises(ConfigurationError, match="only while the model is halted"):
            refused()


@pytest.mark.parametrize("index", [0, 1])
def test_cr0_and_cr1_are_read_only_to_the_host(index):
    with pytest.raises(ConfigurationError, match=f"CR{index} is read-only"):
        _model().write_register("cr", 7, index)


@pytest.mark.parametrize(
    ("name", "value", "index", "match"),
    [
        ("nope", 0, 0, "no register"),
        ("lr", 0, 16, "registers 0..15"),
        ("r", b"\x00", 0, "takes 128 bytes"),
        ("lr", 1 << 32, 0, "unsigned 32-bit"),
        ("lr", True, 0, "unsigned 32-bit"),
    ],
)
def test_a_register_write_that_does_not_fit_is_refused(name, value, index, match):
    with pytest.raises(ConfigurationError, match=match):
        _model().write_register(name, value, index)


def test_a_faulted_model_refuses_steps_and_writes_until_reset():
    model = _model(FAULT)
    record = model.step()
    assert record.outcome == "fault" and record.fault.id == "F-3"
    assert model.status == Status(RunState.FAULTED, 0, FaultId.F3)
    for refused in (
        model.step,
        model.run,
        lambda: model.write_register("lr", 1, 0),
        lambda: model.write_xmem(0, b"\x01"),
        lambda: model.load_program(_words("HALT ;;")),
    ):
        with pytest.raises(ConfigurationError, match="faulted"):
            refused()
    model.reset(ElementFormat.NX8)
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
    model.load_program(_words("HALT ;;"))  # a new program may follow a HALT
    assert model.step().outcome == "halt"


def test_the_cycle_limit_and_fault_enables():
    model = _model(cr={2: 100})
    result = model.run(max_cycles=10)
    # SET, then INC/BNE alternate: after 10 words the next is BNE, word 2.
    assert result.status == Status(RunState.FAULTED, 2, FaultId.F9) and result.cycles == 10
    model.reset(ElementFormat.NX8)
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


def test_xmem_metadata_round_trips_and_is_checked():
    model = _model()
    assert model.read_xmem_metadata(256) == 0  # never written
    model.write_xmem_metadata(256, 0x1AB)
    assert model.read_xmem_metadata(256) == 0x1AB
    assert model.read_xmem_metadata(128) == 0 and model.read_xmem(256, 4) == bytes(4)
    for address, word in ((100, 1), (1 << 40, 1), (0, 1 << 16), (0, -1)):
        with pytest.raises(ConfigurationError):
            model.write_xmem_metadata(address, word)
    with pytest.raises(ConfigurationError):
        model.read_xmem_metadata(64)
    model.step()
    with pytest.raises(ConfigurationError, match="halted"):
        model.write_xmem_metadata(0, 1)
    model.reset(ElementFormat.NX8)
    assert model.read_xmem_metadata(256) == 0


def test_the_metadata_registers_read_as_bytes_and_reset_to_zero():
    model = _model()
    for name, count, size in (("r_meta", 2, 2), ("r_cyclic_meta", 4, 2), ("aaq_lut", 1, 550)):
        for index in range(count):
            assert model.read_register(name, index) == bytes(size), (name, index)
    model.write_register("r_meta", b"\x7f\x01", 1)
    model.write_register("r_cyclic_meta", b"\x05\x00", 3)
    assert model.read_register("r_meta", 1) == b"\x7f\x01"
    assert model.read_register("r_cyclic_meta", 3) == b"\x05\x00"
    with pytest.raises(ConfigurationError, match="2 bytes"):
        model.write_register("r_meta", b"\x00", 0)
    model.reset(ElementFormat.NX8)
    assert model.read_register("r_meta", 1) == bytes(2)


def test_observe_xmem_reads_reports_the_programs_reads_until_stopped():
    model = _model("LD.MULT R0 LR0 CR2 ;;\nLD.MULT R1 LR0 CR2 ;;\nHALT ;;")
    model.write_register("cr", 2, index=2)  # row 2: byte 256 in INT8
    reads: list[tuple[int, int]] = []
    stop = model.observe_xmem_reads(lambda address, size: reads.append((address, size)))
    model.read_xmem(0, 4)  # a host read: not the program's
    model.step()
    assert reads == [(256, 128)]
    stop()
    model.step()
    assert reads == [(256, 128)]


_PLAIN = (int, float, str, bytes, bool, type(None), enum.Enum, tuple, typing.Mapping)
_API_MODULES = {"ipu_emu.reference", "ipu_emu.commit"}


def _assert_plain(value: object, path: str) -> None:
    """*value* is a plain value or an API dataclass of plain values."""
    if dataclasses.is_dataclass(value):
        assert type(value).__module__ in _API_MODULES, path
        for f in dataclasses.fields(value):
            _assert_plain(getattr(value, f.name), f"{path}.{f.name}")
        return
    assert isinstance(value, _PLAIN), f"{path}: {type(value).__module__}.{type(value).__name__}"
    if isinstance(value, enum.Enum):
        assert type(value).__module__ in _API_MODULES | {"ipu_isa.faults"}, path
    elif isinstance(value, tuple):
        for i, item in enumerate(value):
            _assert_plain(item, f"{path}[{i}]")
    elif isinstance(value, typing.Mapping):
        for key, item in value.items():
            _assert_plain(item, f"{path}[{key!r}]")


def test_no_method_returns_an_emulator_object():
    model = _model("LD.MULT R0 LR0 CR0 ;;\n" + LOOP)
    returned = {
        "read_register(lr)": model.read_register("lr", 0),
        "read_register(r)": model.read_register("r", 0),
        "read_xmem": model.read_xmem(0, 4),
        "read_xmem_metadata": model.read_xmem_metadata(0),
        "step": model.step(),
        "run": model.run(),
        "status": model.status,
        "stats": model.stats(),
    }
    for name, value in returned.items():
        _assert_plain(value, name)
    faulted = _model(FAULT)
    _assert_plain(faulted.step(), "step(fault)")


# Word 0: an identity multiply (CR1 is 1) and an ACC op; word 1: a multiply
# by R0, not an identity. Every lane is valid (CR15's default dstructure).
_STATS_PROGRAM = "MULT.RC.VE LR0 CR1 0 LR0 CR15.L CR14.L ; ACC.ADD.FIRST ;;\nMULT.RC.VV LR0 R0 0 LR0 CR15.L CR14.L ;;\nHALT ;;"


def test_stats_count_lanes_and_identity_multiplies_over_steps_and_runs():
    model = _model(_STATS_PROGRAM)
    assert model.stats() == Stats(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0)
    model.step()
    assert model.stats() == Stats(1, 1, 1, 0, 0, 128, 128, 1.0, 0.0)
    model.run()
    # 3 words, 256 lanes carried data, 128 of them an identity: 1/3 of peak MACs.
    assert model.stats() == Stats(3, 2, 1, 0, 0, 256, 128, 2 / 3, 1 / 3)
    model.load_program(_words(_STATS_PROGRAM))
    assert model.stats() == Stats(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0)
    model.run()
    model.reset(ElementFormat.NX8)
    assert model.stats() == Stats(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0)


def test_a_faulting_word_adds_no_statistics():
    # The faulting word also holds an identity multiply and an ACC op.
    faulting = (
        "MULT.RC.VE LR0 CR1 0 LR0 CR15.L CR14.L ; ACC.ADD.FIRST ; SET LR0 CR1 ; SET LR0 CR2 ;;"
    )
    model = _model(f"INC LR0 1 ;;\n{faulting}\nHALT ;;")
    model.run()
    assert model.status.fault is FaultId.F3
    assert model.stats() == Stats(1, 0, 0, 0, 0, 0, 0, 0.0, 0.0)


def test_the_sticky_fault_and_arithmetic_flags_and_enable_bits():
    model = _model(cr={2: 100})
    assert model.fault_flags() == frozenset() and model.fault_enabled(FaultId.F9)
    model.set_fault_enable(FaultId.F9, False)
    assert not model.fault_enabled(FaultId.F9)
    assert model.run(max_cycles=10).status.state is RunState.HALTED
    # F-9 disabled: no halt, but the flag is set (IPU-FLT-5).
    assert model.fault_flags() == frozenset({FaultId.F9})
    model.reset(ElementFormat.NX8)
    assert model.fault_flags() == frozenset() and model.fault_enabled(FaultId.F9)
    assert model.arith_flags() == frozenset()


def test_the_arithmetic_flags_are_the_runs_events():
    model = ReferenceModel(ElementFormat.FP32)
    model.load_program(_words("ACC.ADD ;;\nHALT ;;"))
    lanes = struct.pack("<128f", *[3.0e38] * 128)
    model.write_register("r_acc", lanes)
    model.write_register("mult_res", lanes)
    model.run()
    assert model.arith_flags() == frozenset({ArithmeticEvent.ACC_OVERFLOW})


def test_the_trace_header_names_the_loaded_program(tmp_path):
    model = _model()
    header = model.trace_header()
    words = pack_program(_words(LOOP))
    assert header["program_digest"] == hashlib.sha256(words).hexdigest()
    assert header["format"] == "nx8"
    path = tmp_path / "t.jsonl"
    with api.TraceWriter(path, header) as trace:
        model.run(trace=trace)
    assert api.read_trace(path)[0] == header


def test_fault_enabled_refuses_a_non_fault():
    with pytest.raises(ConfigurationError, match="no FaultId"):
        ReferenceModel().fault_enabled("F-1")


def test_debug_run_stops_for_the_callback_and_continues():
    model = _model("BREAK ;;\nINC LR0 1 ;;\nHALT ;;")
    stops = []

    def callback(state, cycles):
        stops.append((state.program_counter, cycles))
        return DebugAction.CONTINUE

    result = model.debug_run(callback, break_on_entry=True)
    # The entry and word 0's BREAK slot stop at the same PC: one callback.
    assert stops == [(0, 0)]
    assert result == RunResult(Status(RunState.HALTED, 2), 3)
    assert model.read_register("lr", 0) == 1
    stops.clear()
    model.load_program(_words("INC LR0 1 ;;\nBREAK ;;\nHALT ;;"))
    model.debug_run(callback)
    assert stops == [(1, 1)]  # the BREAK word, before it retires


def test_debug_run_quit_pauses_where_it_stopped():
    model = _model()
    result = model.debug_run(lambda state, cycles: DebugAction.QUIT, break_on_entry=True)
    assert result == RunResult(Status(RunState.RUNNING, 0), 0, quit=True)
    assert model.run().cycles == 12  # the run picks up at the PC the quit left


def test_debug_run_reports_a_fault():
    model = _model("INC LR0 1 ;;\n" + FAULT)
    result = model.debug_run(lambda state, cycles: DebugAction.CONTINUE)
    assert result == RunResult(Status(RunState.FAULTED, 1, FaultId.F3), 1)


def test_an_alias_profile_records_the_runs_after_it_is_attached():
    model = _model("MULT.RC.VE LR0 CR1 0 LR0 CR15.L CR14.L ; ACC.ADD.FIRST ;;\nHALT ;;")
    assert model.alias_profile() is None
    profile = AliasProfile()
    model.attach_alias_profile(profile)
    assert model.alias_profile() is profile
    model.run()
    assert profile.cycles == 2 and profile.events > 0
    other = _model()
    with pytest.raises(ConfigurationError, match="single model"):
        other.attach_alias_profile(profile)
    running = _model()
    running.step()
    with pytest.raises(ConfigurationError, match="halted"):
        running.attach_alias_profile(AliasProfile())


def test_a_quit_at_a_break_word_leaves_no_stop_for_the_next_step():
    program = "INC LR0 1 ;;\nBREAK ; INC LR0 1 ;;\nHALT ;;"
    model = _model(program)
    model.debug_run(lambda state, cycles: DebugAction.QUIT)
    plain = _model(program)
    plain.step()
    # The next word is recorded as a plain step's (outcome retired, no BREAK
    # slot), apart from its cycle number.
    assert model.step().to_json() == plain.step().to_json()


def test_the_debuggers_quit_past_imem_halts_the_model():
    model = _model()

    def quit_like_ipu_debug(state, cycles):
        state.program_counter = INST_MEM_SIZE  # DebugSession.halt()
        return DebugAction.QUIT

    result = model.debug_run(quit_like_ipu_debug, break_on_entry=True)
    assert result == RunResult(Status(RunState.HALTED, INST_MEM_SIZE), 0, quit=True)
    model.write_register("cr", 5, 2)  # the host may write again
    with pytest.raises(ConfigurationError, match="past IMEM"):
        model.step()
    model.load_program(_words(LOOP))
    assert model.run().cycles == 12


def test_attaching_another_profile_detaches_the_one_before():
    model = _model()
    first, second = AliasProfile(), AliasProfile()
    model.attach_alias_profile(first)
    model.attach_alias_profile(second)
    assert first.observer is None  # it no longer watches the model
    model.run()
    assert first.cycles == 0 and second.cycles == 12
    _model().attach_alias_profile(first)  # free again
    with pytest.raises(ConfigurationError, match="no AliasProfile"):
        model.attach_alias_profile(None)


def test_a_program_that_fills_imem_faults_f2_stepped_or_run():
    words = _words("INC LR0 1 ;;") * INST_MEM_SIZE
    stepped = ReferenceModel()
    stepped.load_program(words)
    for _ in range(INST_MEM_SIZE):
        stepped.step()
    assert stepped.status.state is RunState.RUNNING  # not a halt: the next fetch faults
    assert stepped.step().fault == FaultInfo("F-2", INST_MEM_SIZE)
    ran = ReferenceModel()
    ran.load_program(words)
    assert ran.run().status == stepped.status == Status(RunState.FAULTED, INST_MEM_SIZE, FaultId.F2)


def test_ctrl_c_in_a_debug_run_ends_it_as_a_quit():
    model = _model()

    def interrupted(state, cycles):
        raise KeyboardInterrupt

    result = model.debug_run(interrupted, break_on_entry=True)
    assert result.quit and result == RunResult(Status(RunState.RUNNING, 0), 0, quit=True)
    assert model.run().quit is False and model.status.state is RunState.HALTED
