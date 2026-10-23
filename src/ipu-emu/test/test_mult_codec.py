"""MULT handlers against a numpy reference of each run mode (E04-15, ADR 0007).

Each MULT handler runs in NX8 mode (a static and a dynamic QF) and in FP32 mode
on random operands, and MULT_RES must equal what a plain model computes: each
element expanded with its row's metadata (``ipu_isa.nx8.decode_element``, the
``ARCH-stage-mult.md`` §3.2.1 expansion) or read as FP32, multiplied in float64
and rounded once to float32. The CR scalar is a MULT constant: in NX8 one
element in the constant layout with its own scale and type, in FP32 the CR's
float32 bits; CR0 reads 0.0 and CR1 1.0 in both modes.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from emu_support import ElementFormat, make_state

from ipu_emu.cr_contents import constant_value
from ipu_emu.emulator import run_until_complete
from ipu_emu.errors import CrContentError
from ipu_isa.cr_halves import constant, decode_constant, qf
from ipu_isa.geometry import LANES
from ipu_isa.nx8 import Nx8Format, decode_element, metadata_word

RING_LANES = 512
#: Element indices that make the R_CYCLIC and Ra windows wrap across rows.
RC_IDX = 450
RA_IDX = 200
#: CRs the programs read.
SCALAR_CR = 3
#: LRs the programs read: the index operand, the (zero) mask shift, the LR scalar index.
INDEX_LR, SHIFT_LR, SCALAR_LR = 0, 2, 3

E4 = Nx8Format(sign=1, exp_format=4, num_bits=8)
DYNAMIC = Nx8Format(sign=1, exp_format=0, num_bits=8, dynamic=True)
#: (mode, QF) pairs: the QF is ignored in FP32 mode.
MODES = (
    (ElementFormat.NX8, E4),
    (ElementFormat.NX8, DYNAMIC),
    (ElementFormat.FP32, E4),
)
_IDS = ["nx8-e4", "nx8-dynamic", "fp32"]


class Model:
    """The reference of one mode: element bytes (and row metadata) to values."""

    def __init__(self, fmt: ElementFormat, nx8: Nx8Format) -> None:
        self.fmt = fmt
        self.nx8 = nx8
        self.wide = fmt is ElementFormat.FP32
        self.element_bytes = 4 if self.wide else 1

    def random_rows(self, rng: np.random.Generator, rows: int) -> tuple[bytes, list[int]]:
        """Random element bytes and one random metadata word per row."""
        if self.wide:
            return rng.uniform(-8.0, 8.0, rows * LANES).astype("<f4").tobytes(), [0] * rows
        metadata = [
            metadata_word(
                int(rng.integers(100, 150)), int(rng.integers(0, 2)) if self.nx8.dynamic else 0
            )
            for _ in range(rows)
        ]
        return rng.integers(0, 256, rows * LANES, dtype=np.uint8).tobytes(), metadata

    def values(self, raw: bytes, metadata: list[int]) -> np.ndarray:
        if self.wide:
            return np.frombuffer(raw, dtype="<f4").astype(np.float64)
        out = []
        for i, code in enumerate(raw):
            word = metadata[i // LANES]
            out.append(decode_element(code, self.nx8, word & 0xFF, (word >> 8) & 1))
        return np.asarray(out, dtype=np.float32).astype(np.float64)

    def cr_scalar(self, index: int, word: int) -> float:
        if index == 0:
            return 0.0
        if index == 1:
            return 1.0
        if self.wide:
            return float(np.uint32(word).view("<f4"))
        c = decode_constant(word)
        with np.errstate(all="ignore"):
            return float(np.float32(decode_element(c.value, c.fmt, c.scale)))

    @staticmethod
    def mult_res(products: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            return products.astype("<f4")

    @staticmethod
    def read_mult_res(state) -> np.ndarray:
        return np.frombuffer(bytes(state.regfile.view("mult_res")), dtype="<f4")


def _constant_word(model: Model, rng: np.random.Generator) -> int:
    if model.wide:
        return int(np.float32(rng.uniform(-4, 4)).view(np.uint32))
    return constant(int(rng.integers(0, 256)), int(rng.integers(110, 140)), 3)


def _setup(mode, asm: str, seed: str, cr_word: int | None = None):
    model = Model(*mode)
    rng = np.random.default_rng(random.Random(seed).getrandbits(64))
    ring, ring_meta = model.random_rows(rng, RING_LANES // LANES)
    ra, ra_meta = model.random_rows(rng, 2)  # R0 ++ R1
    drawn = _constant_word(model, rng)
    cr_word = drawn if cr_word is None else cr_word
    lr_scalar_index = int(rng.integers(0, 1 << 16))
    state = make_state(
        asm, fmt=model.fmt, cr={SCALAR_CR: cr_word}, halves={"cr14.l": qf(model.nx8)}
    )
    regs = state.regfile
    row = LANES * model.element_bytes
    ra_reg, ring_reg = ("r_wide_debug", "r_cyclic_wide_debug") if model.wide else ("r", "r_cyclic")
    regs.set_register_bytes(ring_reg, 0, ring)
    regs.set_register_bytes(ra_reg, 0, ra[:row])
    regs.set_register_bytes(ra_reg, 1, ra[row:])
    for k, word in enumerate(ring_meta):
        regs.set_register_bytes("r_cyclic_meta", k, word.to_bytes(2, "little"))
    for k, word in enumerate(ra_meta):
        regs.set_register_bytes("r_meta", k, word.to_bytes(2, "little"))
    regs.set_lr(SCALAR_LR, lr_scalar_index)
    ra_values = model.values(ra, ra_meta)
    operands = {
        "ring": model.values(ring, ring_meta),
        "ra": ra_values,
        "cr": model.cr_scalar(SCALAR_CR, cr_word),
        "lr_scalar": ra_values[lr_scalar_index % (2 * LANES)],
    }
    return model, state, operands


def _run(mode, mult: str, index: int, seed: str, cr_word: int | None = None):
    asm = f"{mult};;\nHALT;;\n"
    model, state, operands = _setup(mode, asm, seed, cr_word)
    state.regfile.set_lr(INDEX_LR, index)
    run_until_complete(state)
    return model, model.read_mult_res(state), operands


def _rc_window(operands) -> np.ndarray:
    return operands["ring"][(RC_IDX + np.arange(LANES)) % RING_LANES]


@pytest.mark.parametrize("mode", MODES, ids=_IDS)
@pytest.mark.parametrize("ra", [0, 1], ids=["r0", "r1"])
def test_mult_rc_vv(mode, ra):
    asm = f"MULT.RC.VV LR0 R{ra} 0 LR2 CR15.L CR14.L"
    model, got, ops = _run(mode, asm, RC_IDX, f"vv-{mode}-{ra}")
    want = model.mult_res(_rc_window(ops) * ops["ra"][ra * LANES : (ra + 1) * LANES])
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("mode", MODES, ids=_IDS)
@pytest.mark.parametrize("src", ["CR3", "LR3"])
def test_mult_rc_ve(mode, src):
    asm = f"MULT.RC.VE LR0 {src} 0 LR2 CR15.L CR14.L"
    model, got, ops = _run(mode, asm, RC_IDX, f"ve-{mode}-{src}")
    scalar = ops["cr"] if src.startswith("CR") else ops["lr_scalar"]
    np.testing.assert_array_equal(got, model.mult_res(_rc_window(ops) * scalar))


@pytest.mark.parametrize("mode", MODES, ids=_IDS)
def test_mult_rc_vs(mode):
    model, got, ops = _run(mode, "MULT.RC.VS LR0 0 LR2 CR15.L CR14.L", RC_IDX, f"vs-{mode}")
    window = _rc_window(ops)
    np.testing.assert_array_equal(got, model.mult_res(window * window))


@pytest.mark.parametrize("mode", MODES, ids=_IDS)
def test_mult_ve(mode):
    model, got, ops = _run(mode, "MULT.VE LR0 CR3 0 LR2 CR15.L CR14.L", RA_IDX, f"mve-{mode}")
    window = ops["ra"][(RA_IDX + np.arange(LANES)) % (2 * LANES)]
    np.testing.assert_array_equal(got, model.mult_res(window * ops["cr"]))


@pytest.mark.parametrize("mode", MODES, ids=_IDS)
def test_mult_ee(mode):
    index = 2 * LANES + 44  # wraps to lane 44 of the R0 ++ R1 pair
    model, got, ops = _run(mode, "MULT.EE LR0 CR3 0 LR2 CR15.L CR14.L", index, f"ee-{mode}")
    product = ops["ra"][44] * ops["cr"]
    np.testing.assert_array_equal(got, model.mult_res(np.full(LANES, product)))


def test_nx8_constant_layout_by_hand():
    """CR3 = constant(0x0C, scale 128, fe 4): exp 1, man 4/8, so 1.5 * 2^(128-127-1) (§3.2.1)."""
    word = constant(0x0C, 128, 4)  # exp 1, man 4/8 below 2^1: 0.75 * 2 = 1.5
    model, got, ops = _run(MODES[0], "MULT.VE LR0 CR3 0 LR2 CR15.L CR14.L", RA_IDX, "c", word)
    assert ops["cr"] == 1.5
    window = ops["ra"][(RA_IDX + np.arange(LANES)) % (2 * LANES)]
    np.testing.assert_array_equal(got, model.mult_res(window * 1.5))


@pytest.mark.parametrize(
    ("word", "value"),
    [
        (constant(0b0_0_111111, 255, 1), float("inf")),  # 1.98 * 2^128: beyond FP32
        (constant(0b0_111111_1, 0, 6), 0.0),  # about 2^-190: below FP32's smallest subnormal
    ],
    ids=["overflow", "underflow"],
)
def test_an_nx8_constant_expands_to_fp32_like_an_element(word, value):
    """The constant is expanded to FP32 before the multiply (§3.2.1), as a vector element is."""
    assert constant_value(3, word) == value


@pytest.mark.parametrize("mode", MODES, ids=_IDS)
@pytest.mark.parametrize(
    "mult",
    ["MULT.VE LR0 CR{} 0 LR2 CR15.L CR14.L", "MULT.RC.VE LR0 CR{} 0 LR2 CR15.L CR14.L"],
    ids=["ve", "rc_ve"],
)
@pytest.mark.parametrize(("cr", "value"), [(0, 0.0), (1, 1.0)])
def test_cr0_and_cr1_are_zero_and_one_in_both_modes(mode, mult, cr, value):
    index = RC_IDX if mult.startswith("MULT.RC") else RA_IDX
    model, got, ops = _run(mode, mult.format(cr), index, f"cr{cr}-{mode}-{mult}")
    if mult.startswith("MULT.RC"):
        window = _rc_window(ops)
    else:
        window = ops["ra"][(RA_IDX + np.arange(LANES)) % (2 * LANES)]
    np.testing.assert_array_equal(got, model.mult_res(window * value))


def test_an_invalid_constant_is_a_configuration_error_before_the_word_retires():
    """fe 0 is no NX8 type: the check phase refuses the word, so nothing of it retires."""
    asm = "SET LR5 CR2 ; MULT.VE LR0 CR3 0 LR2 CR15.L CR14.L;;\nHALT;;\n"
    state = make_state(asm, cr={2: 7, SCALAR_CR: 0x00_00_80_38})  # exp_format field 0
    with pytest.raises(CrContentError, match="CR3"):
        run_until_complete(state)
    assert state.regfile.get_lr(5) == 0  # the LR write was undone
