"""One retired word as a commit record, and a collector that builds them (E04-29).

:class:`CommitRecord` is a thin layer over ``ipu_isa``'s commit record schema:
:meth:`~CommitRecord.to_json` gives exactly a schema document, and
:meth:`~CommitRecord.from_json` reads one back. :class:`CommitCollector`
watches a state through public events only (its cycle observers, the register
file's and XMEM's write observers) and builds one record per word. Nothing is
collected, and nothing costs, unless a collector is attached.

A write is a range the word wrote (what its handlers said they may write),
with its value at the end of the cycle: a write that leaves a value unchanged
is still a write. Scalar registers (LR, CR) are recorded per index. A write
to an emulator-only register (the wide-vector debug copies) carries
``"tag": "emulator"``. A faulting word is recorded when the fault is (through
``IpuState.add_fault_observer``), with no writes: faults are precise (E04-20).
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ipu_emu.cycle import CycleOutcome
from ipu_isa.registers import REGISTER_DEFINITIONS
from ipu_isa.slots import SLOTS, Slot
from ipu_isa.tags import Tag

if TYPE_CHECKING:
    from collections.abc import Callable

    from ipu_emu.ipu_state import HaltRecord, IpuState
    from ipu_emu.issue import IssueFacts

__all__ = [
    "CommitCollector",
    "CommitRecord",
    "FaultInfo",
    "Outcome",
    "RegWrite",
    "SlotOp",
    "XmemWrite",
]

Outcome = Literal["retired", "break", "halt", "fault"]


@dataclass(frozen=True)
class SlotOp:
    """One non-NOP op of the word: its slot, mnemonic and encoded fields."""

    slot: str
    mnemonic: str
    fields: Mapping[str, int]
    emulator_only: bool = False

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "slot": self.slot,
            "mnemonic": self.mnemonic,
            "fields": dict(self.fields),
        }
        if self.emulator_only:
            out["tag"] = "emulator"
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> SlotOp:
        return cls(doc["slot"], doc["mnemonic"], dict(doc["fields"]), doc.get("tag") == "emulator")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SlotOp):
            return NotImplemented
        return self.to_json() == other.to_json()

    def __hash__(self) -> int:
        return hash((self.slot, self.mnemonic, tuple(sorted(self.fields.items()))))


#: Registers with no hardware equivalent: wide-vector debug mode's copies.
EMULATOR_REGISTERS = frozenset({"r_wide_debug", "r_cyclic_wide_debug"})


@dataclass(frozen=True)
class RegWrite:
    """A register write: ``value`` of a scalar register's ``index``, or
    ``data`` (bytes) at byte ``offset`` of a vector register."""

    register: str
    index: int | None = None
    value: int | None = None
    offset: int | None = None
    data: bytes | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any]
        if self.data is None:
            out = {"register": self.register, "index": self.index, "value": self.value}
        else:
            out = {"register": self.register, "offset": self.offset, "data": self.data.hex()}
        if self.register in EMULATOR_REGISTERS:
            out["tag"] = "emulator"
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> RegWrite:
        if "data" in doc:
            return cls(doc["register"], offset=doc["offset"], data=bytes.fromhex(doc["data"]))
        return cls(doc["register"], index=doc["index"], value=doc["value"])


@dataclass(frozen=True)
class XmemWrite:
    """``data`` written at XMEM byte ``address``."""

    address: int
    data: bytes

    def to_json(self) -> dict[str, Any]:
        return {"address": self.address, "data": self.data.hex()}

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> XmemWrite:
        return cls(doc["address"], bytes.fromhex(doc["data"]))


@dataclass(frozen=True)
class FaultInfo:
    """The fault a word raised: its label (``"F-3"``) and PC."""

    id: str
    pc: int


@dataclass(frozen=True)
class CommitRecord:
    """One retired word, as the commit record schema describes it."""

    cycle: int
    pc_before: int
    pc_after: int
    slots: tuple[SlotOp, ...] = ()
    reg_writes: tuple[RegWrite, ...] = ()
    xmem_writes: tuple[XmemWrite, ...] = ()
    outcome: Outcome = "retired"
    fault: FaultInfo | None = None
    arith_events: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """The record as a schema document (JSON-ready: numbers, strings, lists)."""
        return {
            "cycle": self.cycle,
            "pc_before": self.pc_before,
            "pc_after": self.pc_after,
            "slots": [op.to_json() for op in self.slots],
            "reg_writes": [w.to_json() for w in self.reg_writes],
            "xmem_writes": [w.to_json() for w in self.xmem_writes],
            "outcome": self.outcome,
            "fault": None if self.fault is None else {"id": self.fault.id, "pc": self.fault.pc},
            "arith_events": list(self.arith_events),
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> CommitRecord:
        """A record from a schema document."""
        fault = doc["fault"]
        return cls(
            cycle=doc["cycle"],
            pc_before=doc["pc_before"],
            pc_after=doc["pc_after"],
            slots=tuple(SlotOp.from_json(op) for op in doc["slots"]),
            reg_writes=tuple(RegWrite.from_json(w) for w in doc["reg_writes"]),
            xmem_writes=tuple(XmemWrite.from_json(w) for w in doc["xmem_writes"]),
            outcome=doc["outcome"],
            fault=None if fault is None else FaultInfo(fault["id"], fault["pc"]),
            arith_events=tuple(doc["arith_events"]),
        )


def _merged(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """``(start, end)`` ranges, sorted, overlapping and touching ones joined."""
    out: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


_EMULATOR_SLOTS = frozenset(slot.value for slot in Slot if SLOTS[slot].tag is Tag.EMULATOR)


@dataclass
class _Pending:
    pc: int
    slots: list[SlotOp] = field(default_factory=list)
    regs: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    xmem: list[tuple[int, int]] = field(default_factory=list)
    broke: bool = False


class CommitCollector:
    """Builds a :class:`CommitRecord` for every word *state* runs.

    Attach it before a run and :meth:`close` it after (or use it as a context
    manager). A fault is recorded when it happens, whether before a word
    began (F-2 at fetch, F-9 at the cycle limit) or inside one.

    Args:
        state: The machine state to watch.
    """

    def __init__(self, state: IpuState) -> None:
        self.state = state
        self.records: list[CommitRecord] = []
        self._pending: _Pending | None = None
        # The BREAK slot runs before ``begin``: its op, and whether it fired,
        # wait for the word at that PC to begin.
        self._pre: list[SlotOp] = []
        self._pre_pc: int | None = None
        self._broke = False
        self._seen_halt: object = state.halt
        self._detach: list[Callable[[], None]] = [
            state.add_cycle_observer(self),
            state.add_fault_observer(self._faulted),
            state.regfile.add_observer(self._register_written),
            state.xmem.add_observer(self._memory_event),
        ]

    def __enter__(self) -> CommitCollector:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> list[CommitRecord]:
        """Stop watching. Returns the records."""
        for detach in self._detach:
            detach()
        self._detach = []
        self._pending, self._pre, self._pre_pc, self._broke = None, [], None, False
        return self.records

    # -- CycleObserver ------------------------------------------------------

    def begin(self, pc: int) -> None:
        # A word that began and never ended raised an emulator error, not a
        # fault (a fault is recorded at once): it leaves no record.
        same_word = self._pre_pc == pc
        self._pending = _Pending(
            pc, slots=self._pre if same_word else [], broke=same_word and self._broke
        )
        self._pre, self._pre_pc, self._broke = [], None, False

    def issued(self, facts: IssueFacts) -> None:
        pass

    def retired(self, facts: IssueFacts, result: Any) -> None:
        op = facts.plan.inst_name
        if self._pending is None and self._pre_pc != self.state.program_counter:
            self._pre, self._pre_pc, self._broke = [], self.state.program_counter, False
        if facts.slot == "break" and result is CycleOutcome.BREAK:
            self._broke = True
        if op == "NOP":
            return
        slot_op = SlotOp(facts.slot, op, dict(facts.encoded), facts.slot in _EMULATOR_SLOTS)
        if self._pending is None:
            self._pre.append(slot_op)  # the BREAK slot runs before ``begin``
        else:
            self._pending.slots.append(slot_op)

    def end(self) -> None:
        pending, self._pending = self._pending, None
        if pending is None:
            return
        halt = self.state.halt
        outcome: Outcome = "retired"
        if halt is not None and halt is not self._seen_halt and halt.reason == "halt":
            outcome = "halt"
            self._seen_halt = halt
        elif pending.broke:
            outcome = "break"
        self._emit(pending, self.state.program_counter, outcome, None)

    # -- write observers ----------------------------------------------------

    def _register_written(self, name: str, start: int, size: int) -> None:
        if self._pending is not None and size:
            self._pending.regs.setdefault(name, []).append((start, start + size))

    def _memory_event(self, kind: str, address: int, size: int, data: object) -> None:
        if self._pending is not None and kind == "write" and size:
            self._pending.xmem.append((address, address + size))

    # -- records ------------------------------------------------------------

    def _faulted(self, halt: HaltRecord) -> None:
        """Record the faulting word (or, for a fault before any word began,
        a record of the PC alone), with no writes."""
        pending, self._pending = self._pending, None
        self._seen_halt = halt
        slots = pending.slots if pending is not None and pending.pc == halt.pc else []
        if pending is None and self._pre_pc == halt.pc:
            slots = self._pre  # faulted at its BREAK stop's retire, before ``begin``
        self._pre, self._pre_pc, self._broke = [], None, False
        assert halt.fault is not None
        self._emit(
            _Pending(halt.pc, slots=slots),
            self.state.program_counter,
            "fault",
            FaultInfo(halt.fault.label, halt.pc),
        )

    def _emit(
        self, pending: _Pending, pc_after: int, outcome: Outcome, fault: FaultInfo | None
    ) -> None:
        regfile = self.state.regfile
        reg_writes: list[RegWrite] = []
        for name in sorted(pending.regs):
            current = regfile.view(name)
            definition = REGISTER_DEFINITIONS.get(name)
            scalar = definition is not None and not definition["vector"]
            for start, end in _merged(pending.regs[name]):
                if scalar:
                    width = definition["size_bytes"]  # type: ignore[index]
                    for index in range(start // width, (end + width - 1) // width):
                        (value,) = struct.unpack_from("<I", current, index * width)
                        reg_writes.append(RegWrite(name, index=index, value=value))
                else:
                    reg_writes.append(RegWrite(name, offset=start, data=bytes(current[start:end])))
        xmem = self.state.xmem
        xmem_writes = tuple(
            XmemWrite(start, xmem.snapshot_bytes(start, end - start))
            for start, end in _merged(pending.xmem)
        )
        self.records.append(
            CommitRecord(
                cycle=len(self.records),
                pc_before=pending.pc,
                pc_after=pc_after,
                slots=tuple(pending.slots),
                reg_writes=tuple(reg_writes),
                xmem_writes=xmem_writes,
                outcome=outcome,
                fault=fault,
            )
        )
