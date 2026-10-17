"""Top-level IPU state container.

Combines the register file, external memory, program counter, and instruction
memory into a single object — the Python equivalent of ``ipu__obj_t``.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from ipu_emu import activations as _activations
from ipu_emu.fault_control import FaultControl
from ipu_emu.inst_mem import InstMem
from ipu_emu.ipu_config import DEFAULT_DSTRUCTURE_CR
from ipu_emu.ipu_math import DType
from ipu_emu.prototyping import resolve_enabled as _resolve_prototyping
from ipu_emu.regfile import RegFile
from ipu_emu.stats import RunStats
from ipu_emu.xmem import XMem
from ipu_isa.constants import (
    DEFAULT_DSTRUCTURE,
    DStructureConfig,
    PadMode,
    Partition,
    decode_dstructure,
    encode_dstructure,
)
from ipu_isa.faults import ArithmeticEvent, FaultId
from ipu_isa.geometry import INST_MEM_WORDS

if TYPE_CHECKING:
    from ipu_emu.cycle import CycleObserver

#: Instruction-memory depth, as the emulator names it.
INST_MEM_SIZE = INST_MEM_WORDS

#: Cycles a run may take before it faults F-9 (L3). 0 disables the check.
DEFAULT_CYCLE_LIMIT = 1_000_000


def _geometry():
    """The ``geometry`` module: the active codec and the format's registers.

    ``geometry`` imports this module, so importing it at load time would be circular.
    """
    from ipu_emu import geometry

    return geometry


def _ipu():
    """The ``ipu`` module, for its register sizes and row geometry.

    ``ipu`` imports this module, so importing it at load time would be circular.
    """
    from ipu_emu import ipu

    return ipu


class WideVectorArithmetic(str, Enum):
    """How 128-element wide-vector debug math is performed (emulator-only; issue #33).

    FP32: each element is IEEE float32 (default for "no quantization" FP analysis).
    INT32: each element is signed int32 with wrap semantics matching INT8-mode acc ops.
    """

    FP32 = "fp32"
    INT32 = "int32"


@dataclass(frozen=True)
class HaltRecord:
    """Why and where the program stopped (IPU-FLT-4).

    Attributes:
        reason: ``"halt"`` for a HALT word, ``"fault"`` for a fault.
        pc: The PC of the HALT or faulting word.
        fault: The fault, or ``None`` for a HALT.
    """

    reason: Literal["halt", "fault"]
    pc: int
    fault: FaultId | None


class IpuState:
    """Complete IPU processor state.

    Attributes:
        regfile:         The live register file.
        xmem:            External memory (512 MiB, mode-independent allocation).
        program_counter: Current instruction address.
        inst_mem:        Instruction memory: frozen decoded words, written
                         through ``inst_mem[pc] = word`` (see ``InstMem``).
        dtype:           Arithmetic data type (not stored in CR; emulator-only).
        prototyping:     Whether the opt-in fast paths are active (emulator-only).
        quit_requested:  The debugger quit the last run before it finished.
        halt:            Set when a HALT word retires or a word faults; ``None``
                         while the program runs. Cleared when a run starts.
        faults:          Fault enable bits and sticky flags (``FaultControl``);
                         every fault enabled after construction.
        cycle_limit:     Cycles a run may take before it faults F-9; 0 means
                         no limit. A runner's ``max_cycles`` sets it.
        program_length:  Words of the loaded program: a fetch at or past it
                         is fault F-2. ``load_program`` sets it; a state whose
                         IMEM is written directly keeps all of IMEM.
    """

    def __init__(
        self,
        *,
        wide_vector_debug: bool = False,
        wide_vector_arithmetic: WideVectorArithmetic = WideVectorArithmetic.FP32,
        wide_vector_quantize_output: bool = False,
        elu_alpha: float | None = None,
        window_a: float | None = None,
        window_b: float | None = None,
        dtype: DType = DType.INT8,
        prototyping: bool | None = None,
        alias_profile=None,
    ) -> None:
        self.regfile = RegFile()
        self.xmem = XMem()
        self.program_counter: int = 0
        self.halt: HaltRecord | None = None
        self.program_length: int = INST_MEM_SIZE
        self.cycle_limit: int = DEFAULT_CYCLE_LIMIT
        self.faults = FaultControl()
        self.stats = RunStats()
        self.alias_profile = alias_profile
        self.stats.alias_profile = alias_profile
        self._inst_mem = InstMem(INST_MEM_SIZE)
        #: Observers every cycle on this state calls (add_cycle_observer).
        self.cycle_observers: tuple[CycleObserver, ...] = ()
        self.fault_observers: tuple[Callable[[HaltRecord], None], ...] = ()
        # Sticky arithmetic flags (E04-23): set by ACC and AGG, never by a
        # fault, cleared only when the state is built (no run clears them).
        self.arith_flags: set[ArithmeticEvent] = set()
        self.arith_observers: tuple[Callable[[frozenset[ArithmeticEvent]], None], ...] = ()
        # Set when a debugger quit the last run before the program finished
        # (emulator-only; cleared at the start of every run).
        self.quit_requested: bool = False

        # Arithmetic data type — not stored in a CR register (emulator-only).
        self.dtype: DType = dtype

        self.set_cr_dstructure(
            valid_elements=DEFAULT_DSTRUCTURE.valid_elements,
            partition=DEFAULT_DSTRUCTURE.partition,
        )

        # --- Emulator-only wide-vector debug mode (GitHub issue #33) ------------
        self.wide_vector_debug: bool = wide_vector_debug
        self.wide_vector_arithmetic: WideVectorArithmetic = wide_vector_arithmetic
        self.wide_vector_quantize_output: bool = wide_vector_quantize_output

        # --- Prototyping mode (emulator-only; see ipu_emu/prototyping.py) --------
        # Opt-in fast paths that produce bit-identical state while skipping work
        # the hardware really does. The Bazel flag is a mandatory capability;
        # runtime controls may only select the mode inside a flagged build.
        self.prototyping: bool = _resolve_prototyping(prototyping)

        # --- Activation α (emulator-only; not mapped to CR) ----------------------
        self.elu_alpha: float = (
            float(elu_alpha) if elu_alpha is not None else float(_activations._ELU_ALPHA)
        )

        # --- Window bounds [a, b) for the ``window`` activation (emulator-only) ---
        self.window_a: float = (
            float(window_a) if window_a is not None else float(_activations.DEFAULT_WINDOW_A)
        )
        self.window_b: float = (
            float(window_b) if window_b is not None else float(_activations.DEFAULT_WINDOW_B)
        )

    # -- observers -----------------------------------------------------------

    def add_cycle_observer(self, observer: CycleObserver) -> Callable[[], None]:
        """Attach *observer* to every cycle that runs on this state.

        For a caller that never sees the engine (``run_until_complete`` builds
        its own): any ``Cycle`` on this state calls it, before the cycle's own
        observers (``Cycle.add_observer``).

        Args:
            observer: The :class:`~ipu_emu.cycle.CycleObserver` to call.

        Returns:
            A function that detaches it; calling it twice is harmless.
        """
        self.cycle_observers = self.cycle_observers + (observer,)
        # Weak, so an observer that keeps its remover does not pin the state.
        ref = weakref.ref(self)

        def remove() -> None:
            state = ref()
            if state is not None:
                state.cycle_observers = tuple(o for o in state.cycle_observers if o is not observer)

        return remove

    def add_fault_observer(self, fn: Callable[[HaltRecord], None]) -> Callable[[], None]:
        """Call ``fn(halt)`` each time a fault halts this state (:meth:`record_fault`).

        Returns:
            A function that detaches it; calling it twice is harmless.
        """
        self.fault_observers = self.fault_observers + (fn,)
        ref = weakref.ref(self)

        def remove() -> None:
            state = ref()
            if state is not None:
                state.fault_observers = tuple(o for o in state.fault_observers if o is not fn)

        return remove

    def record_arith(self, events: frozenset[ArithmeticEvent]) -> None:
        """Set *events*' sticky flags and tell the arithmetic observers. An
        event never halts (Q53)."""
        self.arith_flags |= events
        for observer in self.arith_observers:
            observer(events)

    def add_arith_observer(
        self, fn: Callable[[frozenset[ArithmeticEvent]], None]
    ) -> Callable[[], None]:
        """Call ``fn(events)`` each time ACC or AGG reports arithmetic events.

        Returns:
            A function that detaches it; calling it twice is harmless.
        """
        self.arith_observers = self.arith_observers + (fn,)
        ref = weakref.ref(self)

        def remove() -> None:
            state = ref()
            if state is not None:
                state.arith_observers = tuple(o for o in state.arith_observers if o is not fn)

        return remove

    def record_fault(self, fault: FaultId, pc: int) -> HaltRecord:
        """Halt on *fault* at *pc*: set its sticky flag and ``halt``, then tell
        the fault observers. The one place a fault is recorded (the cycle's
        faults and the run's F-9 both come here)."""
        self.faults.sticky.add(fault)
        self.halt = HaltRecord("fault", pc, fault)
        for observer in self.fault_observers:
            observer(self.halt)
        return self.halt

    # -- CR dstructure convenience (CR15 = valid_elements[7:0] | partition[11:8]) --

    def get_cr_dstructure(self) -> DStructureConfig:
        """Read CR15 as decoded dstructure configuration fields."""
        return decode_dstructure(self.regfile.get_cr(DEFAULT_DSTRUCTURE_CR))

    def get_dstructure_for(self, cr_idx: int) -> DStructureConfig:
        """Decode the dstructure configuration from an arbitrary CR register.

        Lets mult/acc/aaq stage instructions select which CR supplies their
        valid element mask and dstructure info, instead of being hard-coded to CR15.
        """
        return decode_dstructure(self.regfile.get_cr(cr_idx))

    def set_cr_dstructure(
        self,
        valid_elements: int = DEFAULT_DSTRUCTURE.valid_elements,
        partition: Partition | int = DEFAULT_DSTRUCTURE.partition,
        pad_mode: PadMode | int = DEFAULT_DSTRUCTURE.pad_mode,
        *,
        cr_idx: int = DEFAULT_DSTRUCTURE_CR,
    ) -> None:
        """Write dstructure configuration fields to ``cr_idx`` (CR15 by default)."""
        self.regfile.set_cr(
            cr_idx,
            encode_dstructure(
                valid_elements=valid_elements, partition=partition, pad_mode=pad_mode
            ),
        )

    def set_activation_alphas(
        self,
        *,
        elu_alpha: float | None = None,
        window_a: float | None = None,
        window_b: float | None = None,
    ) -> None:
        """Override α for ``elu`` and the ``[a, b)`` bounds of ``window``
        (emulator-only; not CR).

        Only arguments that are not ``None`` are updated. Values apply to subsequent
        ``ACTIVATE`` instructions executed on this state.
        """
        if elu_alpha is not None:
            self.elu_alpha = float(elu_alpha)
        if window_a is not None:
            self.window_a = float(window_a)
        if window_b is not None:
            self.window_b = float(window_b)

    # -- register file snapshot (for VLIW dispatch) -------------------------

    def snapshot_regfile(self) -> RegFile:
        """Deep-copy the register file for VLIW read-before-write semantics."""
        return self.regfile.snapshot()

    # -- XMEM ↔ register transfers (mirrors ipu__load_r_reg / ipu__store_r_reg) --

    def write_constant_ones(self, xmem_addr: int) -> None:
        """Write a LANES-element ONES row in the active format, with alias provenance.

        Use only for intentional identity constants, never ordinary inputs or
        weights. Any subsequent XMEM write overlapping this row revokes it.
        """
        data = _geometry().active_codec(self).one() * _ipu().LANES
        self.xmem.write_address(xmem_addr, data)
        self.xmem.mark_constant_ones(xmem_addr, data)

    # Debug/test conveniences, not on any instruction execution path. Each
    # takes a ROW number and mirrors its instruction in the active format:
    # LD.MULT, LD.MULT.CYCLIC, ST.ACC, LD.MASK. A row is LANES elements of
    # the active codec; the registers are the format's (``vector_registers``).

    def _row_bytes(self) -> int:
        return _geometry().active_codec(self).element_bytes * _ipu().LANES

    def _row_address(self, xmem_row: int) -> int:
        return xmem_row * self._row_bytes()

    def load_r_reg_from_xmem(self, xmem_row: int, r_index: int) -> None:
        """Load one row from XMEM into R register *r_index* (0=R0, 1=R1)."""
        addr = self._row_address(xmem_row)
        data = self.xmem.read_address(addr, self._row_bytes())
        name = _geometry().vector_registers(self).ra
        self.regfile.set_register_bytes(name, r_index, data)
        if self.xmem.is_constant_ones(addr, data):
            self.regfile.mark_constant_ones(name, r_index * len(data), len(data))

    def store_r_reg_to_xmem(self, xmem_row: int, r_index: int) -> None:
        """Store R register *r_index* (one row) to XMEM."""
        name = _geometry().vector_registers(self).ra
        self.xmem.write_address(
            self._row_address(xmem_row), self.regfile.get_register_bytes(name, r_index)
        )

    def load_r_cyclic_from_xmem(self, xmem_row: int, slot_element_idx: int = 0) -> None:
        """Load one row from XMEM into r_cyclic at *slot_element_idx*.

        An element index on a slot boundary (``R_CYCLIC_VALID_INDICES``), as
        for LD.MULT.CYCLIC. Defaults to slot 0.
        """
        ipu = _ipu()
        if slot_element_idx not in ipu.R_CYCLIC_VALID_INDICES:
            raise ValueError(
                f"slot_element_idx must be one of {ipu.R_CYCLIC_VALID_INDICES}; "
                f"got {slot_element_idx}"
            )
        addr = self._row_address(xmem_row)
        row_bytes = self._row_bytes()
        data = self.xmem.read_address(addr, row_bytes)
        byte_idx = slot_element_idx * _geometry().active_codec(self).element_bytes
        name = _geometry().vector_registers(self).r_cyclic
        # A slot never wraps the ring, so this is the ring's own slot write.
        self.regfile.mutable(name, byte_idx, row_bytes)[:] = data
        if self.xmem.is_constant_ones(addr, data):
            self.regfile.mark_constant_ones(name, byte_idx, len(data))

    def store_acc_to_xmem(self, xmem_row: int) -> None:
        """Store the whole accumulator (R_ACC_SIZE bytes in both modes) to XMEM."""
        self.xmem.write_address(self._row_address(xmem_row), self.regfile.get_r_acc_bytes())

    def load_r_mask_from_xmem(self, xmem_row: int) -> None:
        """Load the mask register (R_REG_SIZE bytes in both modes: 1 bit per lane)."""
        self.regfile.set_r_mask(
            self.xmem.read_address(self._row_address(xmem_row), _ipu().R_REG_SIZE)
        )

    # -- state queries ------------------------------------------------------

    @property
    def inst_mem(self) -> InstMem:
        """Instruction memory. Read-only attribute: write words into it, never
        replace it, so the cycle's per-PC schedules see every write."""
        return self._inst_mem

    @property
    def is_halted(self) -> bool:
        """True once a HALT word retired or a word faulted (``halt`` is set),
        or the PC is past instruction memory (where a debugger's quit moves it)."""
        return self.halt is not None or self.program_counter >= INST_MEM_SIZE

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full IPU state for debug/JSON export."""
        return {
            "program_counter": self.program_counter,
            "regfile": self.regfile.to_dict(),
        }

    def __repr__(self) -> str:
        return f"IpuState(pc={self.program_counter}, halted={self.is_halted})"
