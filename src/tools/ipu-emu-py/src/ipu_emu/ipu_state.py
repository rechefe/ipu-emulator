"""Top-level IPU state container.

Combines the register file, external memory, program counter, and instruction
memory into a single object — the Python equivalent of ``ipu__obj_t``.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ipu_emu.activations import AaqOptions, ActivationParams
from ipu_emu.errors import ConfigurationError
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
from ipu_isa.formats import FORMATS, ElementFormat, FormatInfo
from ipu_isa.geometry import INST_MEM_WORDS

if TYPE_CHECKING:
    from ipu_emu.cycle import CycleObserver

#: Instruction-memory depth, as the emulator names it.
INST_MEM_SIZE = INST_MEM_WORDS


def _geometry():
    """The ``geometry`` module: the format's registers.

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
        self.stats = RunStats()
        self.alias_profile = alias_profile
        self.stats.alias_profile = alias_profile
        self._inst_mem = InstMem(INST_MEM_SIZE)
        #: Observers every cycle on this state calls (add_cycle_observer).
        self.cycle_observers: tuple[CycleObserver, ...] = ()
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
        dstructure), XMEM, the program counter and the run statistics. The
        instruction memory, the activation parameters, the AAQ options, the
        prototyping mode and the observers are kept.

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
        self.stats.reset()
        self.quit_requested = False

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
        data = self._codec.one() * _ipu().LANES
        self.xmem.write_address(xmem_addr, data)
        self.xmem.mark_constant_ones(xmem_addr, data)

    # Debug/test conveniences, not on any instruction execution path. Each
    # takes a ROW number and mirrors its instruction in the active format:
    # LD.MULT, LD.MULT.CYCLIC, ST.ACC, LD.MASK. A row is LANES elements of
    # the active codec; the registers are the format's (``vector_registers``).

    def _row_bytes(self) -> int:
        return self._codec.element_bytes * _ipu().LANES

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
        byte_idx = slot_element_idx * self._codec.element_bytes
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
        """True if PC has run past instruction memory."""
        return self.program_counter >= INST_MEM_SIZE

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full IPU state for debug/JSON export."""
        return {
            "program_counter": self.program_counter,
            "regfile": self.regfile.to_dict(),
        }

    def __repr__(self) -> str:
        return f"IpuState(pc={self.program_counter}, halted={self.is_halted})"
