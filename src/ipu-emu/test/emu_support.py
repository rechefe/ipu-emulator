"""Shared helpers for the emulator tests: assemble, load and run a program.

Every emulator test that needs a program goes through this module, so a change
to how state is built or how a program runs is made here once.

``fmt`` names the run mode a state runs in (``IpuState(fmt=...)``: NX8, the
default, or FP32; ADR 0007); ``ElementFormat`` is re-exported from
``ipu_isa.formats``. ``halves`` presets CR halves by their assembly names
(``{"cr14.l": qf(E4)}``); :data:`E4` and :data:`QF_E4` are a common NX8 format
and its QF half, which the programs here keep in ``CR14.L``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ipu_as.lark_tree import assemble
from ipu_emu.emulator import DEFAULT_MAX_CYCLES, load_program, run_until_complete
from ipu_emu.errors import ConfigurationError
from ipu_emu.geometry import R_CYCLIC_VALID_INDICES, R_REG_SIZE, vector_registers
from ipu_emu.ipu import Ipu
from ipu_emu.ipu_state import IpuState
from ipu_emu.outcome import CycleOutcome
from ipu_emu.run_control import RunControl
from ipu_isa.codec.word import DecodedWord
from ipu_isa.cr_halves import qf
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES
from ipu_isa.nx8 import Nx8Format, encode_row

__all__ = [
    "E4",
    "QF_E4",
    "BreakpointRunControl",
    "ElementFormat",
    "write_nx8_row",
    "decode_program",
    "assemble_program",
    "load_r_cyclic_from_xmem",
    "load_r_mask_from_xmem",
    "load_r_reg_from_xmem",
    "new_state",
    "make_state",
    "run",
    "run_both",
    "store_acc_to_xmem",
    "store_r_reg_to_xmem",
]


@dataclass
class BreakpointRunControl(RunControl):
    """A :class:`RunControl` whose breakpoints and run target the emulator's own
    ``run_with_debug`` tests set directly. The debugger's ``DebugControl`` lives
    in ``ipu_debug``, which the emulator's tests do not depend on."""

    def add_breakpoint(self, pc: int) -> None:
        self._breakpoints = self._breakpoints | {pc}

    def run_until(self, pc: int, current_pc: int) -> bool:
        self.run_target = pc if pc != current_pc else None
        return pc != current_pc


def decode_program(words: Iterable[int]) -> list[dict[str, int]]:
    """Decode encoded instruction words into the dicts ``load_program`` takes."""
    return [DecodedWord.from_word(word) for word in words]


def assemble_program(asm: str) -> list[dict[str, int]]:
    """Assemble *asm* and decode it."""
    return decode_program(assemble(asm))


#: A common NX8 format (signed, fe 4, 8 bits) and its QF half.
E4 = Nx8Format(sign=1, exp_format=4, num_bits=8)
QF_E4 = qf(E4)


def _half_name(name: str) -> tuple[int, bool]:
    """``"cr14.l"`` -> ``(14, False)``."""
    bank, _, which = name.lower().partition(".")
    if not bank.startswith("cr") or which not in ("l", "h"):
        raise ConfigurationError(f"{name!r} is not a CR half (CR0.L .. CR15.H)")
    return int(bank[2:]), which == "h"


def new_state(
    *,
    cr: Mapping[int, int] | None = None,
    fmt: ElementFormat | None = None,
    ds: int | None = None,
    halves: Mapping[str, int] | None = None,
    setup: Callable[[IpuState], None] | None = None,
    **state_kw: Any,
) -> IpuState:
    """Build an ``IpuState`` with no program loaded.

    *fmt* selects the run mode. *ds* is written into ``CR15.L`` (the default
    DS half) and *halves* into the halves they name (``{"cr14.l": QF_E4}``),
    before *cr* presets whole configuration registers, so a ``cr`` entry wins.
    ``CR14.L`` holds :data:`QF_E4` unless *halves* or *cr* say otherwise.
    *setup* then runs on the state (XMEM contents, register presets). Other
    keywords go to ``IpuState``.
    """
    kwargs = dict(state_kw)
    if fmt is not None:
        kwargs["fmt"] = fmt
    state = IpuState(**kwargs)
    state.set_cr_half(14, False, QF_E4)
    if ds is not None:
        state.set_cr_half(15, False, ds)
    for name, value in (halves or {}).items():
        state.set_cr_half(*_half_name(name), value)
    if cr:
        for idx, val in cr.items():
            state.regfile.set_cr(idx, val)
    if setup is not None:
        setup(state)
    return state


def write_nx8_row(
    state: IpuState, xmem_row: int, values: Sequence[float], fmt: Nx8Format = E4
) -> bytes:
    """Quantize *values* to one NX8 row (``nx8.encode_row``) and write it and
    its metadata at *xmem_row*; returns the row's bytes."""
    row = encode_row(list(values), fmt)
    addr = xmem_row * LANES
    state.xmem.write_address(addr, row.elements)
    state.xmem.write_metadata(addr, row.metadata)
    return row.elements


def make_state(
    asm: str,
    *,
    cr: Mapping[int, int] | None = None,
    fmt: ElementFormat | None = None,
    **state_kw: Any,
) -> IpuState:
    """Assemble *asm* and return a state with the program loaded, ready to run.

    Keywords are those of :func:`new_state`.
    """
    state = new_state(cr=cr, fmt=fmt, **state_kw)
    load_program(state, assemble_program(asm))
    return state


def run(asm: str, *, max_cycles: int = DEFAULT_MAX_CYCLES, **kw: Any) -> IpuState:
    """:func:`make_state`, then run to halt (breakpoints ignored)."""
    state = make_state(asm, **kw)
    run_until_complete(state, max_cycles)
    return state


def run_both(
    asm: str | None = None,
    *,
    program: Sequence[dict[str, int] | None] | None = None,
    on_cycle: Callable[[Sequence[Ipu]], None] | None = None,
    max_cycles: int = 100_000,
    **kw: Any,
) -> tuple[IpuState, IpuState]:
    """Run *asm* on a faithful and a prototyping state in lockstep.

    Pass *program* (decoded words, ``None`` for an empty entry) instead of
    *asm* to run a program the assembler would not produce.

    Both states are built with :func:`new_state` (keywords *kw*, including
    *setup*, which runs on each). After every cycle the two cycle outcomes must agree and
    *on_cycle* gets both engines (faithful first) to compare state. Breakpoints
    are resumed. Each state's ``stats.total_cycles`` is set to the cycle count.
    Requires a ``--define=ipu_proto=1`` build.
    """
    if (asm is None) == (program is None):
        raise TypeError("run_both takes exactly one of asm and program")
    decoded = assemble_program(asm) if program is None else list(program)
    states = tuple(new_state(prototyping=mode, **kw) for mode in (False, True))
    for state in states:
        load_program(state, [None if inst is None else dict(inst) for inst in decoded])
    engines = [Ipu(state) for state in states]

    cycles = 0
    while not states[0].is_halted:
        assert not states[1].is_halted
        outcomes = [engine.cycle.step() for engine in engines]
        assert outcomes[0] == outcomes[1]
        if outcomes[0] == CycleOutcome.BREAK:
            for engine in engines:
                engine.cycle.retire()
        cycles += 1
        if cycles >= max_cycles:
            raise RuntimeError(f"program exceeded {max_cycles} cycles")
        if on_cycle is not None:
            on_cycle(engines)

    assert states[1].is_halted
    for state in states:
        state.stats.total_cycles = cycles
    return states


# -- XMEM <-> register transfers ------------------------------------------
# Test conveniences, not on any instruction execution path. Each takes a ROW
# number and moves the row's bytes in the state's element format: LD.MULT,
# LD.MULT.CYCLIC and LD.MASK without the metadata, and a plain R_ACC dump. A
# row is LANES elements of the state's codec; the registers are the format's
# (``vector_registers``).


def _row_bytes(state: IpuState) -> int:
    return state.codec.element_bytes * LANES


def _row_address(state: IpuState, xmem_row: int) -> int:
    return xmem_row * _row_bytes(state)


def load_r_reg_from_xmem(state: IpuState, xmem_row: int, r_index: int) -> None:
    """Load one row from XMEM into R register *r_index* (0=R0, 1=R1)."""
    addr = _row_address(state, xmem_row)
    data = state.xmem.read_address(addr, _row_bytes(state))
    name = vector_registers(state).ra
    state.regfile.set_register_bytes(name, r_index, data)
    if state.xmem.is_constant_ones(addr, data):
        state.regfile.mark_constant_ones(name, r_index * len(data), len(data))


def store_r_reg_to_xmem(state: IpuState, xmem_row: int, r_index: int) -> None:
    """Store R register *r_index* (one row) to XMEM."""
    name = vector_registers(state).ra
    state.xmem.write_address(
        _row_address(state, xmem_row), state.regfile.get_register_bytes(name, r_index)
    )


def load_r_cyclic_from_xmem(state: IpuState, xmem_row: int, slot_element_idx: int = 0) -> None:
    """Load one row from XMEM into r_cyclic at *slot_element_idx*.

    An element index on a slot boundary (``R_CYCLIC_VALID_INDICES``), as
    for LD.MULT.CYCLIC. Defaults to slot 0.
    """
    if slot_element_idx not in R_CYCLIC_VALID_INDICES:
        raise ConfigurationError(
            f"slot_element_idx must be one of {R_CYCLIC_VALID_INDICES}; got {slot_element_idx}"
        )
    addr = _row_address(state, xmem_row)
    row_bytes = _row_bytes(state)
    data = state.xmem.read_address(addr, row_bytes)
    byte_idx = slot_element_idx * state.codec.element_bytes
    name = vector_registers(state).r_cyclic
    # A slot never wraps the ring, so this is the ring's own slot write.
    state.regfile.mutable(name, byte_idx, row_bytes)[:] = data
    if state.xmem.is_constant_ones(addr, data):
        state.regfile.mark_constant_ones(name, byte_idx, len(data))


def store_acc_to_xmem(state: IpuState, xmem_row: int) -> None:
    """Dump the whole accumulator (R_ACC_SIZE bytes in every format) to XMEM (a test aid)."""
    state.xmem.write_address(_row_address(state, xmem_row), state.regfile.get_r_acc_bytes())


def load_r_mask_from_xmem(state: IpuState, xmem_row: int) -> None:
    """Load the mask register (R_REG_SIZE bytes in every format: 1 bit per lane)."""
    state.regfile.set_r_mask(state.xmem.read_address(_row_address(state, xmem_row), R_REG_SIZE))
