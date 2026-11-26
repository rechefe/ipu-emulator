"""The fused MULT -> ACC path (``Issuer.mult_product``): the same state both ways.

A MULT.RC.VE whose float32 product is stored unchanged in MULT_RES hands those
lanes to the ACC of the same word, which then skips decoding MULT_RES again.
Every run here goes twice, once fused and once on the general path (a cycle
observer attached: observed ops never take the hand-over), and every register,
the statistics and the arithmetic flags must be the same.
"""

from __future__ import annotations

import copy
import math
import struct

import numpy as np
import pytest
from emu_support import ElementFormat, assemble_program, new_state

from ipu_emu.emulator import load_program
from ipu_emu.ipu import Ipu
from ipu_emu.prototyping import build_enabled
from ipu_isa.cr_halves import ds

# Every ACC op that reads MULT_RES after a MULT.RC.VE in the same word, an
# unshifted and a shifted lane mask (the shift pads MULT_RES: no hand-over),
# an LR scalar and CR scalars, and a word with the MULT alone.
PROGRAM = """
    SET LR0 CR0 ; SET LR1 CR1 ; SET LR2 CR2 ;;
    LD.MULT R0 LR0 CR2 ;;
    LD.MULT.CYCLIC LR0 CR3 LR0 ;;
    MULT.RC.VE LR0 CR8 0 LR0 CR15.L CR14.L ; ACC.ADD.FIRST ;;
    MULT.RC.VE LR0 CR9 0 LR0 CR15.L CR14.L ; ACC.ADD ;;
    MULT.RC.VE LR0 LR1 0 LR0 CR15.L CR14.L ; ACC.SUB ;;
    MULT.RC.VE LR0 CR8 0 LR0 CR15.L CR14.L ; ACC.MAX ;;
    MULT.RC.VE LR0 CR9 0 LR1 CR15.L CR14.L ; ACC.ADD ;;
    MULT.RC.VE LR0 CR8 0 LR0 CR15.L CR14.L ;;
    ACC.ADD ;;
    MULT.RC.VE LR0 CR9 0 LR0 CR15.L CR14.L ; ACC.ADD ;;
    HALT ;;
"""

_SOURCE = 0x10000

#: Prototyping mode runs in a ``--define=ipu_proto=1`` build only (as in
#: test_prototyping_equivalence); faithful mode takes the hand-over as well.
_MODES = [
    False,
    pytest.param(
        True, marks=pytest.mark.skipif(not build_enabled(), reason="requires --define=ipu_proto=1")
    ),
]


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _seed(fmt: ElementFormat, row: bytes, scalars: tuple[float, float]):
    def setup(state):
        state.xmem.write_address(_SOURCE, row)
        state.xmem.write_address(_SOURCE + len(row), row[::-1])
        state.regfile.set_cr(2, _SOURCE // len(row))  # R0's row
        state.regfile.set_cr(3, _SOURCE // len(row) + 1)  # R_CYCLIC's row
        state.set_cr_half(15, False, ds(32, 4))
        if fmt is ElementFormat.FP32:
            state.regfile.set_cr(8, _float_bits(scalars[0]))
            state.regfile.set_cr(9, _float_bits(scalars[1]))
        else:
            for unit in (_SOURCE, _SOURCE + len(row)):
                state.xmem.write_metadata(unit, 127)

    return setup


def _final(state) -> tuple:
    return (
        state.program_counter,
        {name: bytes(state.regfile.raw_readonly(name)) for name in state.regfile.register_names},
        copy.deepcopy(state.stats),
        sorted(map(str, state.arith_flags)),
    )


class _Watcher:
    """A cycle observer that does nothing: its presence alone turns the hand-over off."""

    def begin(self, pc):
        pass

    def issued(self, facts):
        pass

    def retired(self, facts, result):
        pass

    def end(self):
        pass


def _run(fmt: ElementFormat, row: bytes, scalars, *, fused: bool, prototyping: bool):
    state = new_state(fmt=fmt, prototyping=prototyping, setup=_seed(fmt, row, scalars))
    program = PROGRAM
    if fmt is not ElementFormat.FP32:  # NX8: the CR constants 0.0 and 1.0
        program = program.replace("CR8", "CR0").replace("CR9", "CR1")
    load_program(state, assemble_program(program))
    ipu = Ipu(state)
    if not fused:
        ipu.cycle.add_observer(_Watcher())
    while not state.is_halted:
        ipu.cycle.step()
    return _final(state)


def _fp32_row(values) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


_RNG = np.random.default_rng(11)
_FP32_ROWS = {
    "random": _fp32_row(_RNG.standard_normal(128)),
    "nan and inf": _fp32_row([math.nan, math.inf, -math.inf] + list(_RNG.standard_normal(125))),
    "overflow": _fp32_row([3e38] * 128),
    "subnormal": _fp32_row([1e-40] * 64 + [-1e-39] * 64),
}


@pytest.mark.parametrize("prototyping", _MODES, ids=["faithful", "prototyping"])
@pytest.mark.parametrize("row", sorted(_FP32_ROWS))
@pytest.mark.parametrize("scalars", [(1.5, -0.25), (3e38, 7.0), (1e-30, 1e-30)])
def test_fp32_fused_and_general_paths_agree(row, scalars, prototyping):
    fused = _run(ElementFormat.FP32, _FP32_ROWS[row], scalars, fused=True, prototyping=prototyping)
    general = _run(
        ElementFormat.FP32, _FP32_ROWS[row], scalars, fused=False, prototyping=prototyping
    )
    assert fused == general


@pytest.mark.parametrize("prototyping", _MODES, ids=["faithful", "prototyping"])
def test_nx8_fused_and_general_paths_agree(prototyping):
    row = bytes(_RNG.integers(0, 256, 128, dtype=np.uint8))
    fused = _run(ElementFormat.NX8, row, (0.0, 0.0), fused=True, prototyping=prototyping)
    general = _run(ElementFormat.NX8, row, (0.0, 0.0), fused=False, prototyping=prototyping)
    assert fused == general


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "observed"])
def test_the_hand_over_reaches_only_the_acc_right_after_an_unpadded_mult(fused):
    """Which ACC.ADD words get the MULT's lanes: the one after an unmasked MULT
    in the same word, not one after a shifted (padding) mask, nor an ACC.ADD
    alone in its word; and none while an observer is attached."""
    import ipu_emu.cycle as cycle_mod
    import ipu_emu.ipu as ipu_mod

    seen: list[bool] = []
    saved = dict(ipu_mod._SLOT_PLANS)
    for key, plans in saved.items():
        patched = []
        for plan in plans:
            if plan.inst_name == "ACC.ADD":
                real = plan.fn

                def spy(ipu, _real=real, **operands):
                    seen.append(ipu.issuer.prior_product is not None)
                    return _real(ipu, **operands)

                plan = plan._replace(fn=spy)
            patched.append(plan)
        ipu_mod._SLOT_PLANS[key] = tuple(patched)
    cycle_mod._issue_table.cache_clear()
    try:
        _run(ElementFormat.FP32, _FP32_ROWS["random"], (1.5, -0.25), fused=fused, prototyping=False)
    finally:
        ipu_mod._SLOT_PLANS.update(saved)
        cycle_mod._issue_table.cache_clear()
    # Words 5 (MULT ; ACC.ADD), 8 (shifted mask: padded), 10 (ACC.ADD alone),
    # 11 (MULT ; ACC.ADD).
    assert seen == ([True, False, False, True] if fused else [False] * 4)
