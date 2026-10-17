"""The stable reference-model API, v1 (E04-30, D5, Q10, Q11).

A small, fixed interface for RTL verification: a testbench resets the model,
loads a program, sets registers and XMEM while the model is halted, steps it
cycle by cycle and compares each :class:`~ipu_emu.commit.CommitRecord` with
the RTL. Only the names in ``__all__`` are the contract; everything else in
``ipu_emu`` may change. :data:`API_VERSION` follows Q11: a change to a public
name or signature bumps it (a breaking one, its major number), and
``test_reference_api_contract.py`` holds the snapshot.

Register names are ``ipu_isa``'s (``ipu_isa.registers.REGISTER_DEFINITIONS``):
``lr`` and ``cr`` are scalar banks indexed by register, the others vector
registers indexed by element. No method returns an emulator object.

Until the steppable engine lands, run control is the emulator's own
(``RunClock``, ``Cycle``); the engine will take it over behind this API.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass

from ipu_emu.commit import CommitCollector, CommitRecord
from ipu_emu.emulator import load_program as _load_program
from ipu_emu.errors import ConfigurationError, EmulatorFault
from ipu_emu.ipu import Ipu
from ipu_emu.ipu_state import INST_MEM_SIZE, IpuState
from ipu_emu.run_clock import RunClock
from ipu_emu.trace import TraceWriter
from ipu_emu.xmem import XMEM_SIZE_BYTES
from ipu_isa.codec.word import DecodedWord, unpack_program
from ipu_isa.constants import CR_READ_ONLY_INITIAL_VALUES
from ipu_isa.faults import FaultId
from ipu_isa.formats import ElementFormat
from ipu_isa.registers import REGISTER_DEFINITIONS, RegisterDefinition

__all__ = ["API_VERSION", "ReferenceModel", "RunResult", "RunState", "Status"]

#: The version of this API (Q11).
API_VERSION = "1.0.0"


class RunState(str, enum.Enum):
    """Whether the model runs, stopped normally, or stopped on a fault."""

    RUNNING = "running"
    HALTED = "halted"
    FAULTED = "faulted"


@dataclass(frozen=True)
class Status:
    """The model's run state.

    Attributes:
        state: Running (stepped and not stopped), halted (after reset, a
            program load or a HALT word) or faulted.
        pc: The program counter; after a HALT word or a fault, that word's.
        fault: The fault, when faulted; else ``None``.
    """

    state: RunState
    pc: int
    fault: FaultId | None = None


@dataclass(frozen=True)
class RunResult:
    """How :meth:`ReferenceModel.run` ended.

    Attributes:
        status: The model's status when the run stopped.
        cycles: The words the run retired.
    """

    status: Status
    cycles: int


class ReferenceModel:
    """One IPU, as the RTL verification sees it.

    Args:
        fmt: The element format to start in (:meth:`reset` changes it).
    """

    def __init__(self, fmt: ElementFormat = ElementFormat.INT8) -> None:
        self._state = IpuState(fmt=fmt)
        self._ipu = Ipu(self._state)
        self._collector = CommitCollector(self._state)
        self._started = False

    # -- setup ---------------------------------------------------------------

    def reset(self, fmt: ElementFormat) -> None:
        """Reset in element format *fmt*: registers, XMEM, the PC, the fault
        controls (every fault enabled, no sticky flag), the arithmetic flags
        and the statistics. The loaded program stays.

        Raises:
            ConfigurationError: *fmt* is no element format.
        """
        self._state.reset(fmt)
        self._ipu.cycle.reset()
        self._collector.records.clear()
        self._started = False

    def load_program(self, words: bytes | Sequence[int]) -> None:
        """Load a program: a program file's bytes (``ipu-as`` ``.bin``), or
        its words as integers. The PC goes to 0.

        Raises:
            ConfigurationError: The program is larger than IMEM.
        """
        decoded: list[dict[str, int]]
        if isinstance(words, (bytes, bytearray)):
            decoded = [*unpack_program(bytes(words))]
        else:
            decoded = [DecodedWord.from_word(word) for word in words]
        _load_program(self._state, decoded)
        self._state.program_counter = 0
        self._state.halt = None
        self._started = False

    def set_fault_enable(self, fault: FaultId, enabled: bool) -> None:
        """Set or clear *fault*'s enable bit (IPU-FLT-5).

        Raises:
            ConfigurationError: Clearing F-2 to F-6, whose disabled behaviour
                is not defined (OQ-2).
        """
        self._state.faults.set_enabled(fault, enabled)

    # -- registers and XMEM --------------------------------------------------

    def read_register(self, name: str, index: int = 0) -> int | bytes:
        """Register *name*'s element *index*: an int for ``lr`` and ``cr``,
        the element's bytes for a vector register.

        Raises:
            ConfigurationError: No such register or index.
        """
        definition, size = self._register(name, index)
        data = bytes(self._state.regfile.view(name)[index * size : (index + 1) * size])
        return data if definition["vector"] else int.from_bytes(data, "little")

    def write_register(self, name: str, value: int | bytes, index: int = 0) -> None:
        """Write register *name*'s element *index*; only while halted.

        Raises:
            ConfigurationError: The model is running or faulted; the register
                is read-only to the host (CR0, CR1); no such register or
                index; or *value* does not fit it.
        """
        self._writable()
        definition, size = self._register(name, index)
        if name == "cr" and index in CR_READ_ONLY_INITIAL_VALUES:
            raise ConfigurationError(f"CR{index} is read-only to the host")
        if definition["vector"]:
            if not isinstance(value, (bytes, bytearray)) or len(value) != size:
                raise ConfigurationError(f"{name}[{index}] takes {size} bytes")
            data = bytes(value)
        else:
            if not isinstance(value, int) or not 0 <= value < 1 << (8 * size):
                raise ConfigurationError(f"{name}[{index}] takes an unsigned {8 * size}-bit int")
            data = value.to_bytes(size, "little")
        self._state.regfile.mutable(name, index * size, size)[:] = data

    def read_xmem(self, address: int, size: int) -> bytes:
        """*size* bytes of XMEM at *address*.

        Raises:
            ConfigurationError: The range is outside XMEM.
        """
        if address < 0 or size < 0 or address + size > XMEM_SIZE_BYTES:
            raise ConfigurationError(f"XMEM [{address}, {address + size}) is outside XMEM")
        return self._state.xmem.snapshot_bytes(address, size)

    def write_xmem(self, address: int, data: bytes) -> None:
        """Write *data* at XMEM *address*; only while halted.

        Raises:
            ConfigurationError: The model is running or faulted, or the range
                is outside XMEM.
        """
        self._writable()
        self._state.xmem.write_address(address, bytes(data))

    # -- running -------------------------------------------------------------

    def step(self) -> CommitRecord:
        """Run one word and return its commit record. A BREAK slot never
        stops it (Q39): the word retires.

        Raises:
            ConfigurationError: The model is faulted (IPU-FLT-7) or halted
                after a HALT word: reset or load a program first.
        """
        self._steppable()
        seen = len(self._collector.records)
        try:
            self._ipu.cycle.step(honour_break=False)
        except EmulatorFault:
            pass
        self._started = True
        (record,) = self._collector.records[seen:]
        return record

    def run(self, max_cycles: int | None = None, *, trace: TraceWriter | None = None) -> RunResult:
        """Step until a HALT word, a fault, or *max_cycles* words (fault F-9;
        ``None`` keeps the cycle limit, 0 disables it).

        Args:
            max_cycles: The cycle limit of this run.
            trace: Where to write each word's commit record, if anywhere.

        Raises:
            ConfigurationError: As :meth:`step`.
        """
        self._steppable()
        clock = RunClock(self._state, max_cycles)
        clock.begin_run(reset_stats=not self._started)
        seen = len(self._collector.records)
        try:
            while self._state.halt is None:
                clock.admit()
                self._ipu.cycle.step(honour_break=False)
                clock.count_cycle()
        except EmulatorFault:
            pass
        finally:
            clock.end_run()
        self._started = True
        if trace is not None:
            trace.write_all(self._collector.records[seen:])
        return RunResult(self.status, clock.cycles)

    @property
    def status(self) -> Status:
        """The run state, PC and (when faulted) the fault."""
        state = self._state
        halt = state.halt
        if halt is not None and halt.reason == "fault":
            return Status(RunState.FAULTED, halt.pc, halt.fault)
        if halt is not None:
            return Status(RunState.HALTED, halt.pc)  # the HALT word's PC
        if not self._started:
            return Status(RunState.HALTED, state.program_counter)
        return Status(RunState.RUNNING, state.program_counter)

    # -- helpers -------------------------------------------------------------

    def _register(self, name: str, index: int) -> tuple[RegisterDefinition, int]:
        definition = REGISTER_DEFINITIONS.get(name)
        if definition is None:
            raise ConfigurationError(
                f"no register {name!r}; registers: {sorted(REGISTER_DEFINITIONS)}"
            )
        if not 0 <= index < definition["count"]:
            raise ConfigurationError(
                f"{name} has elements 0..{definition['count'] - 1}, not {index}"
            )
        return definition, definition["size_bytes"]

    def _writable(self) -> None:
        state = self.status.state
        if state is not RunState.HALTED:
            raise ConfigurationError(
                f"the host writes only while the model is halted; it is {state.value}"
            )

    def _steppable(self) -> None:
        status = self.status
        if status.state is RunState.FAULTED:
            raise ConfigurationError(
                f"faulted ({status.fault.label if status.fault else ''} at PC {status.pc}): reset first (IPU-FLT-7)"
            )
        if self._state.halt is not None or self._state.program_counter >= INST_MEM_SIZE:
            raise ConfigurationError("halted after HALT: reset or load a program first")
