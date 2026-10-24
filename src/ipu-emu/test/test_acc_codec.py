"""ACC, AGG and load handlers against a scalar reference of each mode (E04-16, ADR 0007).

Each handler runs in both run modes (NX8 and FP32; the accumulator is FP32 in
both) on random operands, and R_ACC must equal what the scalar Python
arithmetic the handlers replaced computes: ``struct``-unpacked accumulator
lanes, Python ``+``/``-``/``max`` and one float32 rounding on the store. AGG
folds the first ``valid_elements`` elements of its DS half. A source scan keeps
format branches out of the slot modules (they ask ``geometry.fp32_mode``) and
out of the test load/store helpers.
"""

from __future__ import annotations

import ast
import inspect
import random
import struct
import textwrap
from pathlib import Path

import emu_support
import numpy as np
import pytest
from emu_support import (
    ElementFormat,
    load_r_cyclic_from_xmem,
    load_r_reg_from_xmem,
    make_state,
    store_r_reg_to_xmem,
)

import ipu_emu
from ipu_emu.emulator import run_until_complete
from ipu_emu.errors import EmulatorError
from ipu_emu.ipu_state import IpuState
from ipu_isa.cr_halves import ds
from ipu_isa.geometry import LANES

FORMATS = (ElementFormat.NX8, ElementFormat.FP32)

PACKAGE_DIR = Path(inspect.getsourcefile(ipu_emu)).parent
#: What a handler would read to branch on the element format (E04-17: the
#: state's one format value and what it declares).
_FORMAT_BRANCH_NAMES = ("fmt", "format_info", "ElementFormat", "AccumulatorType")
# ``IpuState.write_constant_ones`` is left out: an NX8 ONES row needs a QF
# and metadata, an FP32 one does not (ADR 0007).
_STATE_HELPERS = (
    "load_r_reg_from_xmem",
    "store_r_reg_to_xmem",
    "load_r_cyclic_from_xmem",
    "store_acc_to_xmem",
    "load_r_mask_from_xmem",
)


def _pack(fmt: ElementFormat, values) -> bytes:
    """The scalar store: ``struct``'s float32 rounding."""
    return struct.pack(f"<{len(values)}f", *values)


def _unpack(fmt: ElementFormat, raw: bytes) -> list:
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def _random_lanes(fmt: ElementFormat, rng: np.random.Generator) -> bytes:
    values = rng.uniform(-8.0, 8.0, LANES).astype("<f4")
    values[:4] = (0.0, -0.0, 0.0, -0.0)  # signed zeros, where 0 - x and max differ from -x
    return values.tobytes()


def _setup(fmt: ElementFormat, asm: str, seed: str) -> tuple[IpuState, list, list]:
    rng = np.random.default_rng(random.Random(seed).getrandbits(64))
    state = make_state(f"{asm};;\nHALT;;\n", fmt=fmt)
    acc_raw, res_raw = _random_lanes(fmt, rng), _random_lanes(fmt, rng)
    state.regfile.mutable("r_acc")[:] = acc_raw
    state.regfile.mutable("mult_res")[:] = res_raw
    return state, _unpack(fmt, acc_raw), _unpack(fmt, res_raw)


def _acc_lanes(state: IpuState) -> bytes:
    return bytes(state.regfile.view("r_acc"))


_ACC_REFERENCE = {
    "ACC.ADD": lambda a, m: a + m,
    "ACC.SUB": lambda a, m: a - m,
    "ACC.MAX": lambda a, m: max(a, m),
    "ACC.ADD.FIRST": lambda a, m: m,
    "ACC.MAX.FIRST": lambda a, m: m,
    "ACC.SUB.FIRST": lambda a, m: 0.0 - m,
}


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.value)
@pytest.mark.parametrize("mnemonic", sorted(_ACC_REFERENCE))
def test_acc_lanewise(fmt, mnemonic):
    state, acc, res = _setup(fmt, mnemonic, f"{mnemonic}-{fmt}")
    run_until_complete(state)
    want = [_ACC_REFERENCE[mnemonic](a, m) for a, m in zip(acc, res, strict=True)]
    assert _acc_lanes(state) == _pack(fmt, want)


def _agg_case(fmt: ElementFormat, mnemonic: str, valid: int, dest: int):
    state, acc, res = _setup(fmt, f"{mnemonic} LR0 CR15.H", f"{mnemonic}-{fmt}-{valid}-{dest}")
    # A DS of 2 groups, both valid: AGG still takes the first valid_elements.
    state.set_cr_half(15, True, ds(min(valid, 64), 2) if valid <= 64 else ds(valid))
    state.regfile.set_lr(0, dest)
    run_until_complete(state)
    return state, acc, res[:valid]


def _fold_sum(fmt: ElementFormat, lanes: list):
    total = 0.0
    for v in lanes:
        total += v
    return total


def _fold_max(lanes: list, seed):
    best = seed
    for v in lanes:
        if v > best:
            best = v
    return best


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.value)
@pytest.mark.parametrize("mnemonic", ["AGG.SUM", "AGG.SUM.FIRST", "AGG.MAX", "AGG.MAX.FIRST"])
@pytest.mark.parametrize("valid", [0, 1, 77, 128])
def test_agg_folds_in_lane_order(fmt, mnemonic, valid):
    dest = 100
    state, acc, active = _agg_case(fmt, mnemonic, valid, dest)
    if mnemonic == "AGG.SUM.FIRST":
        value = _fold_sum(fmt, active)
    elif mnemonic == "AGG.SUM":
        value = float(_fold_sum(fmt, active)) + acc[dest]
    elif mnemonic == "AGG.MAX.FIRST":
        value = _fold_max(active, float("-inf"))
    else:
        value = _fold_max(active, acc[dest])
    raw = _pack(fmt, acc)
    lane = _pack(fmt, [value])
    assert _acc_lanes(state) == raw[: dest * 4] + lane + raw[(dest + 1) * 4 :]


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.value)
def test_acc_stride_past_the_last_lane_raises_before_writing(fmt):
    # An open question (MAS-stage-accumulator 6.8), not a fault: an EmulatorError
    # before any lane is written (E04-19).
    state, acc, res = _setup(fmt, "ACC.STRIDE 16 off off LR0", f"stride-{fmt}")
    state.regfile.set_lr(0, 1)  # start lane 32: lanes 96..127 of the result do not fit
    with pytest.raises(EmulatorError, match="past the end of R_ACC"):
        run_until_complete(state)
    assert _acc_lanes(state) == _pack(fmt, acc)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.value)
def test_state_helpers_move_one_row_of_the_format(fmt):
    state = make_state("HALT;;\n", fmt=fmt)
    wide = fmt is ElementFormat.FP32
    element_bytes = 4 if wide else 1
    row = LANES * element_bytes
    ra, ring = ("r_wide_debug", "r_cyclic_wide_debug") if wide else ("r", "r_cyclic")

    state.write_constant_ones(row, emu_support.E4)  # XMEM row 1 (the QF is NX8's)
    ones = state.xmem.read_address(row, row)
    assert state.xmem.read_address(row, row + 1)[-1] == 0
    if wide:
        assert ones == struct.pack("<f", 1.0) * LANES
    else:
        assert state.xmem.read_metadata(row) != 0  # the row's scale

    load_r_reg_from_xmem(state, 1, 1)
    assert state.regfile.get_register_bytes(ra, 1) == ones
    assert state.regfile.has_constant_ones(ra, row, row)
    store_r_reg_to_xmem(state, 2, 1)
    assert state.xmem.read_address(2 * row, row) == ones

    load_r_cyclic_from_xmem(state, 1, 256)
    start = 256 * element_bytes
    assert bytes(state.regfile.view(ring)[start : start + row]) == ones
    assert state.regfile.has_constant_ones(ring, start, row)


def _source_names(source: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


@pytest.mark.parametrize("module", sorted(p.name for p in (PACKAGE_DIR / "slots").glob("*.py")))
def test_slot_modules_have_no_format_branch(module):
    names = _source_names((PACKAGE_DIR / "slots" / module).read_text(encoding="utf-8"))
    assert names.isdisjoint(_FORMAT_BRANCH_NAMES)


@pytest.mark.parametrize("helper", _STATE_HELPERS)
def test_state_load_store_helpers_have_no_format_branch(helper):
    fn = getattr(IpuState, helper, None) or getattr(emu_support, helper)
    source = textwrap.dedent(inspect.getsource(fn))
    assert _source_names(source).isdisjoint(_FORMAT_BRANCH_NAMES)
