"""Top-level IPU state container.

Combines the register file, external memory, program counter, and instruction
memory into a single object — the state an :class:`~ipu_emu.ipu.Ipu` engine runs on.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from ipu_emu.activations import AaqOptions, ActivationParams
from ipu_emu.errors import ConfigurationError
from ipu_emu.fault_control import FaultControl
from ipu_emu.inst_mem import InstMem
from ipu_emu.ipu_config import DEFAULT_DSTRUCTURE_CR
from ipu_emu.numerics import LaneCodec, codec_for
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
from ipu_isa.formats import FORMATS, ElementFormat, FormatInfo
from ipu_isa.geometry import INST_MEM_WORDS, LANES

if TYPE_CHECKING:
    from ipu_emu.cycle import CycleObserver

#: Instruction-memory depth, as the emulator names it.
INST_MEM_SIZE = INST_MEM_WORDS

#: Cycles a run may take before it faults F-9 (L3). 0 disables the check.
DEFAULT_CYCLE_LIMIT = 1_000_000


def _check_format(fmt: object) -> ElementFormat:
    """*fmt* as an :class:`ElementFormat`, or a :class:`ConfigurationError`."""
    if isinstance(fmt, ElementFormat):
        return fmt
    try:
        return ElementFormat(fmt)
    except ValueError:
        names = ", ".join(f.value for f in ElementFormat)
        raise ConfigurationError(
            f"unknown element format {fmt!r}; expected one of: {names}"
        ) from None


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
        xmem:            External memory (512 MiB, format-independent allocation).
        program_counter: Current instruction address.
        inst_mem:        Instruction memory: frozen decoded words, written
                         through ``inst_mem[pc] = word`` (see ``InstMem``).
        fmt:             The element format (read-only; see :meth:`reset`).
        activation:      The activation parameters ``ACTIVATE`` reads (emulator-only).
        aaq:             The AAQ stage's emulator-only options.
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
        fmt: ElementFormat = ElementFormat.INT8,
        activation: ActivationParams | None = None,
        aaq: AaqOptions | None = None,
        prototyping: bool | None = None,
        alias_profile=None,
    ) -> None:
        """Build a state at reset in element format *fmt*.

        Args:
            fmt: The element format (Q47). It sets the element and row bytes,
                the accumulator type and the vector registers; it is fixed
                for the state's life, except through :meth:`reset`.
            activation: The activation parameters; the module defaults if omitted.
            aaq: The AAQ stage's options; :class:`AaqOptions` defaults if omitted.
            prototyping: Select the opt-in fast paths (see ``ipu_emu.prototyping``).
            alias_profile: An alias profile to record into, or ``None``.

        Raises:
            ConfigurationError: *fmt* is no element format, or prototyping
                was asked for outside a prototyping build.
        """
        self._set_format(fmt)
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
        # fault, cleared only when the state is built or reset (no run clears them).
        self.arith_flags: set[ArithmeticEvent] = set()
        self.arith_observers: tuple[Callable[[frozenset[ArithmeticEvent]], None], ...] = ()
        # Set when a debugger quit the last run before the program finished
        # (emulator-only; cleared at the start of every run).
        self.quit_requested: bool = False

        self.set_cr_dstructure(
            valid_elements=DEFAULT_DSTRUCTURE.valid_elements,
            partition=DEFAULT_DSTRUCTURE.partition,
        )

        # --- Prototyping mode (emulator-only; see ipu_emu/prototyping.py) --------
        # Opt-in fast paths that produce bit-identical state while skipping work
        # the hardware really does. The Bazel flag is a mandatory capability;
        # runtime controls may only select the mode inside a flagged build.
        self.prototyping: bool = _resolve_prototyping(prototyping)

        # --- Activation α and window bounds (emulator-only; not mapped to CR) ----
        self.activation: ActivationParams = (
            activation if activation is not None else ActivationParams()
        )
        # --- AAQ options (emulator-only; the AaQ placeholder, Q35) --------------
        self.aaq: AaqOptions = aaq if aaq is not None else AaqOptions()

    # -- element format --------------------------------------------------------

    def _set_format(self, fmt: object) -> None:
        checked = _check_format(fmt)
        self._fmt = checked
        self._codec = codec_for(checked)

    @property
    def fmt(self) -> ElementFormat:
        """The element format this state runs in. Read-only: build a new state
        or call :meth:`reset` to change it."""
        return self._fmt

    @property
    def format_info(self) -> FormatInfo:
        """What :attr:`fmt` declares in ``ipu_isa.formats``: element and row
        bytes, the accumulator type."""
        return FORMATS[self._fmt]

    @property
    def codec(self) -> LaneCodec:
        """The lane codec of :attr:`fmt`, which every handler decodes and encodes through."""
        return self._codec

    def reset(self, fmt: ElementFormat | None = None) -> None:
        """Return to the reset state, in element format *fmt* if given.

        Clears the registers (to their reset values, CR15 the default
        dstructure), XMEM, the program counter, the halt record, the run
        statistics and the arithmetic flags, and sets every fault enable bit
        and clears the sticky fault flags (IPU-FLT-6). The instruction memory
        and program length, the cycle limit, the activation parameters, the
        AAQ options, the prototyping mode and the observers are kept.

        Args:
            fmt: The new element format; the current one if omitted.

        Raises:
            ConfigurationError: *fmt* is no element format. Nothing is reset.
        """
        if fmt is not None:
            self._set_format(fmt)
        RegFile().copy_into(self.regfile)
        self.set_cr_dstructure(
            valid_elements=DEFAULT_DSTRUCTURE.valid_elements,
            partition=DEFAULT_DSTRUCTURE.partition,
        )
        self.xmem.clear()
        self.program_counter = 0
        self.halt = None
        self.stats.reset()
        self.quit_requested = False
        self.faults.reset()
        self.arith_flags.clear()

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

    # -- CR dstructure convenience (CR15 = valid_elements[7:0] | partition[12:8] | pad_mode[14:13]) --

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
        self.activation = replace(
            self.activation,
            **{
                name: value
                for name, value in (
                    ("elu_alpha", elu_alpha),
                    ("window_a", window_a),
                    ("window_b", window_b),
                )
                if value is not None
            },
        )

    # -- XMEM constants ----------------------------------------------------

    def write_constant_ones(self, xmem_addr: int) -> None:
        """Write a LANES-element ONES row in the active format, with alias provenance.

        Use only for intentional identity constants, never ordinary inputs or
        weights. Any subsequent XMEM write overlapping this row revokes it.
        """
        data = self._codec.one() * LANES
        self.xmem.write_address(xmem_addr, data)
        self.xmem.mark_constant_ones(xmem_addr, data)

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
