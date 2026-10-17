"""The stable reference-model API, v1 (E04-30, D5, Q10, Q11).

A small, fixed interface for RTL verification: a testbench resets the model,
loads a program, sets registers and XMEM while the model is halted, steps it
cycle by cycle and compares each :class:`~ipu_emu.commit.CommitRecord` with
the RTL. Only the names in ``__all__`` and the records :meth:`step` returns
are the contract; everything else in ``ipu_emu`` may change. :data:`API_VERSION`
follows Q11: a change to a public name, signature or record field bumps it (a
breaking one, its major number), and ``test_reference_api_contract.py`` holds
the snapshot.

Register names are ``ipu_isa``'s (``ipu_isa.registers.REGISTER_DEFINITIONS``),
and ``index`` picks the register within its bank (``lr``: LR0-LR15, ``cr``:
CR0-CR15, ``r``: R0-R1). ``lr`` and ``cr`` read and write as unsigned 32-bit
ints, the others as bytes. ``r_wide_debug`` and ``r_cyclic_wide_debug`` are
emulator-only: they hold the operands of the wide formats (FP32, INT32), which
the hardware does not have. No method returns an emulator object.

A host access the IPU refuses with an AXI error (faults.md, "Illegal host
accesses", IPU-FLT-7) raises :class:`~ipu_emu.errors.ConfigurationError`:
the host acts between cycles, never inside one.

Until the steppable engine lands, run control is the emulator's own
(``RunClock``, ``Cycle``); the engine will take it over behind this API.
"""

from __future__ import annotations

import dataclasses
import enum
import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ipu_emu.commit import CommitCollector as _CommitCollector
from ipu_emu.commit import CommitRecord
from ipu_emu.emulator import load_program as _load_program
from ipu_emu.errors import ConfigurationError, EmulatorFault
from ipu_emu.ipu import Ipu as _Ipu
from ipu_emu.ipu_state import INST_MEM_SIZE
from ipu_emu.ipu_state import IpuState as _IpuState
from ipu_emu.run_clock import RunClock as _RunClock
from ipu_emu.trace import TraceWriter
from ipu_isa.codec.word import DecodedWord, unpack_program, word_bytes
from ipu_isa.constants import CR_READ_ONLY_INITIAL_VALUES
from ipu_isa.faults import FaultId
from ipu_isa.formats import ElementFormat
from ipu_isa.registers import REGISTER_DEFINITIONS, RegisterDefinition
from ipu_isa.view import isa

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
        self._state = _IpuState(fmt=fmt)
        self._ipu = _Ipu(self._state)
        self._collector = _CommitCollector(self._state)
        self._started = False

    # -- setup ---------------------------------------------------------------

    def reset(self, fmt: ElementFormat) -> None:
        """Reset the model. The loaded program stays.

        Registers and XMEM go to their reset values (CR15 holds the default
        dstructure), the PC to 0, every fault is enabled with no sticky flag,
        and the arithmetic flags and the statistics clear. Record numbering
        restarts at 0.

        Args:
            fmt: The element format to run in.

        Raises:
            ConfigurationError: *fmt* is no element format.
        """
        self._state.reset(fmt)
        self._ipu.cycle.reset()
        self._new_collector()
        self._started = False

    def load_program(self, words: bytes | Sequence[int]) -> None:
        """Load a program, only while halted. The PC goes to 0 and record
        numbering restarts at 0.

        Args:
            words: A program file's bytes (``ipu-as`` ``.bin``), or its words
                as unsigned ints.

        Raises:
            ConfigurationError: The model is running or faulted (after a
                fault, :meth:`reset` first); the bytes are not whole words; a
                word is wider than the instruction word; or the program is
                larger than IMEM.
        """
        self._writable()
        decoded: list[dict[str, int]]
        if isinstance(words, (bytes, bytearray)):
            width = word_bytes()
            if len(words) % width:
                raise ConfigurationError(
                    f"a program file is whole {width}-byte words; this one has {len(words)} bytes"
                )
            decoded = [*unpack_program(bytes(words))]
        else:
            bits = isa().layout.word_bits()
            for word in words:
                if isinstance(word, bool) or not isinstance(word, int) or not 0 <= word < 1 << bits:
                    raise ConfigurationError(f"{word!r} is not an unsigned {bits}-bit word")
            decoded = [DecodedWord.from_word(word) for word in words]
        _load_program(self._state, decoded)
        self._state.program_counter = 0
        self._state.halt = None
        self._new_collector()
        self._started = False

    def set_fault_enable(self, fault: FaultId, enabled: bool) -> None:
        """Set or clear a fault's enable bit (IPU-FLT-5).

        Args:
            fault: The fault.
            enabled: Whether it halts the IPU.

        Raises:
            ConfigurationError: Clearing F-2 to F-6, whose disabled behaviour
                is not defined (OQ-2).
        """
        self._state.faults.set_enabled(fault, enabled)

    # -- registers and XMEM --------------------------------------------------

    def read_register(self, name: str, index: int = 0) -> int | bytes:
        """Read one register.

        Args:
            name: The bank, as ``ipu_isa`` names it (``lr``, ``cr``, ``r``,
                ``r_acc``, ...).
            index: The register within the bank.

        Returns:
            An unsigned int for ``lr`` and ``cr``; the register's bytes for
            the others.

        Raises:
            ConfigurationError: No such bank or index.
        """
        definition, size = self._register(name, index)
        data = bytes(self._state.regfile.view(name)[index * size : (index + 1) * size])
        return data if definition["vector"] else int.from_bytes(data, "little")

    def write_register(self, name: str, value: int | bytes, index: int = 0) -> None:
        """Write one register, only while halted.

        Args:
            name: The bank, as for :meth:`read_register`.
            value: An unsigned 32-bit int for ``lr`` and ``cr``; exactly the
                register's bytes for the others.
            index: The register within the bank.

        Raises:
            ConfigurationError: The model is running or faulted; the register
                is read-only to the host (CR0, CR1); no such bank or index; or
                *value* does not fit the register.
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
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 1 << (8 * size)
            ):
                raise ConfigurationError(f"{name}[{index}] takes an unsigned {8 * size}-bit int")
            data = value.to_bytes(size, "little")
        self._state.regfile.mutable(name, index * size, size)[:] = data

    def read_xmem(self, address: int, size: int) -> bytes:
        """Read XMEM.

        Args:
            address: The first byte.
            size: The number of bytes.

        Returns:
            The bytes.

        Raises:
            ConfigurationError: The range is outside XMEM.
        """
        return self._state.xmem.snapshot_bytes(address, size)

    def write_xmem(self, address: int, data: bytes) -> None:
        """Write XMEM, only while halted.

        Args:
            address: The first byte.
            data: The bytes.

        Raises:
            ConfigurationError: The model is running or faulted, or the range
                is outside XMEM.
        """
        self._writable()
        self._state.xmem.write_address(address, bytes(data))

    # -- running -------------------------------------------------------------

    def step(self) -> CommitRecord:
        """Run one word. A BREAK slot never stops it (Q39): the word retires.
        A step has no cycle limit (F-9 is :meth:`run`'s); its word counts
        toward the statistics' cycles.

        Returns:
            The word's commit record; its ``outcome`` is ``retired``, ``halt``
            or ``fault`` (never ``break``), and its ``cycle`` counts the words
            since the last :meth:`reset` or :meth:`load_program`.

        Raises:
            ConfigurationError: The model is faulted (IPU-FLT-7: reset first),
                or stopped after a HALT word or past IMEM (reset or load a
                program first).
        """
        self._steppable()
        clock = _RunClock(self._state)
        clock.begin_run(reset_stats=not self._started)
        try:
            self._ipu.cycle.step(honour_break=False)
            clock.count_cycle()
        except EmulatorFault:
            pass  # the fault is in the word's record and in ``status``
        finally:
            clock.end_run()
        self._started = True
        (record,) = self._collector.take()
        return record

    def run(self, max_cycles: int | None = None, *, trace: TraceWriter | None = None) -> RunResult:
        """Step until a HALT word, a fault, or the cycle limit (fault F-9).

        Args:
            max_cycles: The cycle limit of this run: ``None`` keeps the
                model's, 0 disables it.
            trace: Where to write each word's commit record, as it retires;
                ``None`` keeps none. Its records number from 0 (the schema's
                ``cycle``: records before this one in the trace).

        Returns:
            The status the run stopped in and the words it retired.

        Raises:
            ConfigurationError: As :meth:`step`.
        """
        self._steppable()
        clock = _RunClock(self._state, max_cycles)
        clock.begin_run(reset_stats=not self._started)
        numbering = itertools.count()  # the trace's record numbers
        try:
            while self._state.halt is None:
                clock.admit()
                self._ipu.cycle.step(honour_break=False)
                clock.count_cycle()
                self._flush(trace, numbering)
        except EmulatorFault:
            pass  # the fault is in the last record and in ``status``
        finally:
            clock.end_run()
            self._flush(trace, numbering)
        self._started = True
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

    def _new_collector(self) -> None:
        self._collector.close()
        self._collector = _CommitCollector(self._state)

    def _flush(self, trace: TraceWriter | None, numbering: Iterator[int]) -> None:
        records = self._collector.take()
        if trace is not None:
            for record in records:
                trace.write(dataclasses.replace(record, cycle=next(numbering)))

    def _register(self, name: str, index: int) -> tuple[RegisterDefinition, int]:
        definition = REGISTER_DEFINITIONS.get(name)
        if definition is None:
            raise ConfigurationError(
                f"no register {name!r}; registers: {sorted(REGISTER_DEFINITIONS)}"
            )
        if not 0 <= index < definition["count"]:
            raise ConfigurationError(
                f"{name} has registers 0..{definition['count'] - 1}, not {index}"
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
            fault = status.fault.label if status.fault is not None else "?"
            raise ConfigurationError(
                f"faulted ({fault} at PC {status.pc}): reset first (IPU-FLT-7)"
            )
        if self._state.halt is not None:
            raise ConfigurationError("halted after HALT: reset or load a program first")
        if self._state.program_counter >= INST_MEM_SIZE:
            raise ConfigurationError(
                f"PC {self._state.program_counter} is past IMEM: reset or load a program first"
            )
