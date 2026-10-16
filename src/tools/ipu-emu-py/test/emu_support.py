"""Shared helpers for the emulator tests: assemble, load and run a program.

Every emulator test that needs a program goes through this module, so a change
to how state is built or how a program runs is made here once.

``fmt`` names the element format a state runs in. Until E04-17 replaces the
``dtype`` attribute and the three ``wide_vector_*`` constructor flags with one
element format on ``IpuState``, :func:`new_state` maps ``fmt`` onto them. That
mapping is the only place the tests know about the old flags.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from ipu_emu.emulator import DEFAULT_MAX_CYCLES, load_program, run_until_complete
from ipu_emu.execute import BreakResult, Ipu, decode_instruction_word
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_as.lark_tree import assemble

__all__ = [
    "ElementFormat",
    "decode_program",
    "assemble_program",
    "new_state",
    "make_state",
    "run",
    "run_both",
]


class ElementFormat(enum.Enum):
    """Element formats as decided in Q47 (one enum, later owned by ``ipu_isa``).

    INT8 and E1..E7 are hardware formats; FP32 and INT32 are emulator-only
    (today's wide-vector debug mode). Test-side stand-in until E03/E04-17 land.
    """

    INT8 = "int8"
    E1 = "e1"
    E2 = "e2"
    E3 = "e3"
    E4 = "e4"
    E5 = "e5"
    E6 = "e6"
    E7 = "e7"
    FP32 = "fp32"
    INT32 = "int32"


_WIDE = {
    ElementFormat.FP32: WideVectorArithmetic.FP32,
    ElementFormat.INT32: WideVectorArithmetic.INT32,
}


def _format_kwargs(fmt: ElementFormat) -> dict[str, Any]:
    """Map an element format onto today's ``IpuState`` keywords (until E04-17)."""
    if fmt in _WIDE:
        return {
            "wide_vector_debug": True,
            "wide_vector_arithmetic": _WIDE[fmt],
            "dtype": DType.INT8,
        }
    return {"wide_vector_debug": False, "dtype": DType[fmt.name]}


def decode_program(words: Iterable[int]) -> list[dict[str, int]]:
    """Decode encoded instruction words into the dicts ``load_program`` takes."""
    return [decode_instruction_word(word) for word in words]


def assemble_program(asm: str) -> list[dict[str, int]]:
    """Assemble *asm* and decode it."""
    return decode_program(assemble(asm))


def new_state(
    *,
    cr: Mapping[int, int] | None = None,
    fmt: ElementFormat | None = None,
    dstructure: Mapping[str, Any] | None = None,
    **state_kw: Any,
) -> IpuState:
    """Build an ``IpuState`` with no program loaded.

    *fmt* selects the element format. *dstructure* is passed to
    ``set_cr_dstructure`` before *cr* presets the configuration registers, so a
    ``cr`` entry for CR15 wins. Other keywords go to ``IpuState``.
    """
    kwargs = dict(state_kw)
    if fmt is not None:
        kwargs.update(_format_kwargs(fmt))
    state = IpuState(**kwargs)
    if dstructure:
        state.set_cr_dstructure(**dstructure)
    if cr:
        for idx, val in cr.items():
            state.regfile.set_cr(idx, val)
    return state


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
    asm: str,
    *,
    setup: Callable[[IpuState], None] | None = None,
    on_cycle: Callable[[Sequence[Ipu]], None] | None = None,
    max_cycles: int = 100_000,
    **kw: Any,
) -> tuple[IpuState, IpuState]:
    """Run *asm* on a faithful and a prototyping state in lockstep.

    Both states are built with :func:`new_state` (keywords *kw*), then *setup*
    runs on each. After every cycle the two cycle outcomes must agree and
    *on_cycle* gets both engines (faithful first) to compare state. Breakpoints
    are resumed. Each state's ``stats.total_cycles`` is set to the cycle count.
    Requires a ``--define=ipu_proto=1`` build.
    """
    decoded = assemble_program(asm)
    states = tuple(new_state(prototyping=mode, **kw) for mode in (False, True))
    for state in states:
        if setup is not None:
            setup(state)
        load_program(state, [dict(inst) for inst in decoded])
    engines = [Ipu(state) for state in states]

    cycles = 0
    while not states[0].is_halted:
        assert not states[1].is_halted
        outcomes = [engine.execute_vliw_cycle() for engine in engines]
        assert outcomes[0] == outcomes[1]
        if outcomes[0] == BreakResult.BREAK:
            for engine in engines:
                engine.execute_vliw_cycle_skip_break()
        cycles += 1
        if cycles >= max_cycles:
            raise RuntimeError(f"program exceeded {max_cycles} cycles")
        if on_cycle is not None:
            on_cycle(engines)

    assert states[1].is_halted
    for state in states:
        state.stats.total_cycles = cycles
    return states
