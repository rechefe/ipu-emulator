"""MULT handlers against a numpy reference of the active codec (E04-15).

Each MULT handler runs once per element format the tree's kernels use (INT8,
E4 and FP32) on random operands, and MULT_RES must equal what a plain numpy
model of that format computes: decode both operands, multiply in int64 or
float64, round once to the accumulator type (int32 wrap, float32). The CR
scalar operands run in every format: a CR scalar is the CR's low
element-width bits decoded in the active format, and CR1 reads as 1 (Q49,
E04-28).
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from emu_support import ElementFormat, make_state

from ipu_emu.emulator import run_until_complete
from ipu_isa.formats import ElementFormat as IsaFormat
from ipu_isa.formats import decode_element
from ipu_isa.geometry import LANES

FORMATS = (ElementFormat.INT8, ElementFormat.E4, ElementFormat.FP32)
ALL_FORMATS = tuple(ElementFormat)

RING_LANES = 512
#: Element indices that make the R_CYCLIC and Ra windows wrap.
RC_IDX = 450
RA_IDX = 200
#: CRs the programs read.
SCALAR_CR = 3
#: LRs the programs read: the index operand, the (zero) mask shift, the LR scalar index.
INDEX_LR, SHIFT_LR, SCALAR_LR = 0, 2, 3


class Model:
    """The numpy reference of one format: element bytes to values, and the product's rounding."""

    def __init__(self, fmt: ElementFormat) -> None:
        self.fmt = fmt
        self.wide = fmt in (ElementFormat.FP32, ElementFormat.INT32)
        self.integer = fmt in (ElementFormat.INT8, ElementFormat.INT32)
        self.element_bytes = 4 if self.wide else 1

    def random_elements(self, rng: np.random.Generator, count: int) -> bytes:
        if self.fmt is ElementFormat.FP32:
            return rng.uniform(-8.0, 8.0, count).astype("<f4").tobytes()
        if self.fmt is ElementFormat.INT32:
            return rng.integers(-(1 << 20), 1 << 20, count).astype("<i4").tobytes()
        return rng.integers(0, 256, count, dtype=np.uint8).tobytes()

    def values(self, raw: bytes) -> np.ndarray:
        if self.fmt is ElementFormat.INT8:
            return np.frombuffer(raw, dtype=np.int8).astype(np.int64)
        if self.fmt is ElementFormat.INT32:
            return np.frombuffer(raw, dtype="<i4").astype(np.int64)
        if self.fmt is ElementFormat.FP32:
            return np.frombuffer(raw, dtype="<f4").astype(np.float64)
        table = np.array(
            [decode_element(b, IsaFormat(self.fmt)) for b in range(256)], dtype=np.float64
        )
        return table[np.frombuffer(raw, dtype=np.uint8)]

    def cr_scalar(self, word: int) -> np.generic:
        """A CR scalar: the word's low element-width bits, as one element of the format."""
        return self.values(word.to_bytes(4, "little")[: self.element_bytes])[0]

    def mult_res(self, products: np.ndarray) -> np.ndarray:
        if self.integer:
            return products.astype(np.int64).astype("<i4")
        with np.errstate(all="ignore"):
            return products.astype("<f4")

    def read_mult_res(self, state) -> np.ndarray:
        raw = bytes(state.regfile.view("mult_res"))
        return np.frombuffer(raw, dtype="<i4" if self.integer else "<f4")


def _setup(fmt: ElementFormat, asm: str, seed: str, cr_word: int | None = None):
    model = Model(fmt)
    rng = np.random.default_rng(random.Random(seed).getrandbits(64))
    ring = model.random_elements(rng, RING_LANES)
    ra = model.random_elements(rng, 2 * LANES)  # R0 ++ R1
    drawn = int(rng.integers(0, 1 << 32))
    if cr_word is None:
        cr_word = drawn
    lr_scalar_index = int(rng.integers(0, 1 << 16))
    state = make_state(asm, fmt=fmt, cr={SCALAR_CR: cr_word})
    regs = state.regfile
    row = LANES * model.element_bytes
    ra_reg, ring_reg = ("r_wide_debug", "r_cyclic_wide_debug") if model.wide else ("r", "r_cyclic")
    regs.set_register_bytes(ring_reg, 0, ring)
    regs.set_register_bytes(ra_reg, 0, ra[:row])
    regs.set_register_bytes(ra_reg, 1, ra[row:])
    regs.set_lr(SCALAR_LR, lr_scalar_index)
    operands = {
        "ring": model.values(ring),
        "ra": model.values(ra),
        "cr": model.cr_scalar(cr_word),
        "lr_scalar": model.values(ra)[lr_scalar_index % (2 * LANES)],
    }
    return model, state, operands


def _run(fmt: ElementFormat, mult: str, index: int, seed: str, cr_word: int | None = None):
    asm = f"{mult};;\nHALT;;\n"
    model, state, operands = _setup(fmt, asm, seed, cr_word)
    state.regfile.set_lr(INDEX_LR, index)
    run_until_complete(state)
    return model, model.read_mult_res(state), operands


def _rc_window(operands) -> np.ndarray:
    return operands["ring"][(RC_IDX + np.arange(LANES)) % RING_LANES]


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.value)
@pytest.mark.parametrize("ra", [0, 1], ids=["r0", "r1"])
def test_mult_rc_vv(fmt, ra):
    model, got, ops = _run(fmt, f"MULT.RC.VV LR0 R{ra} 0 LR2 CR15", RC_IDX, f"vv-{fmt}-{ra}")
    want = model.mult_res(_rc_window(ops) * ops["ra"][ra * LANES : (ra + 1) * LANES])
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("fmt", ALL_FORMATS, ids=lambda f: f.value)
@pytest.mark.parametrize("src", ["CR3", "LR3"])
def test_mult_rc_ve(fmt, src):
    model, got, ops = _run(fmt, f"MULT.RC.VE LR0 {src} 0 LR2 CR15", RC_IDX, f"ve-{fmt}-{src}")
    scalar = ops["cr"] if src.startswith("CR") else ops["lr_scalar"]
    np.testing.assert_array_equal(got, model.mult_res(_rc_window(ops) * scalar))


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.value)
def test_mult_rc_vs(fmt):
    model, got, ops = _run(fmt, "MULT.RC.VS LR0 0 LR2 CR15", RC_IDX, f"vs-{fmt}")
    window = _rc_window(ops)
    np.testing.assert_array_equal(got, model.mult_res(window * window))


@pytest.mark.parametrize("fmt", ALL_FORMATS, ids=lambda f: f.value)
def test_mult_ve(fmt):
    model, got, ops = _run(fmt, "MULT.VE LR0 CR3 0 LR2 CR15", RA_IDX, f"mve-{fmt}")
    window = ops["ra"][(RA_IDX + np.arange(LANES)) % (2 * LANES)]
    np.testing.assert_array_equal(got, model.mult_res(window * ops["cr"]))


@pytest.mark.parametrize("fmt", ALL_FORMATS, ids=lambda f: f.value)
def test_mult_ee(fmt):
    index = 2 * LANES + 44  # wraps to lane 44 of the R0 ++ R1 pair
    model, got, ops = _run(fmt, "MULT.EE LR0 CR3 0 LR2 CR15", index, f"ee-{fmt}")
    product = ops["ra"][44] * ops["cr"]
    np.testing.assert_array_equal(got, model.mult_res(np.full(LANES, product)))


#: CR words whose low element-width bits mean different things per format: a
#: byte that is negative as INT8, a word whose low byte and whole word differ,
#: and FP32 2.5 (a small negative INT8 low byte, 0x40200000 as INT32).
CR_WORDS = (0x000000FE, 0x12345681, 0x40200000)


@pytest.mark.parametrize("fmt", ALL_FORMATS, ids=lambda f: f.value)
@pytest.mark.parametrize("word", CR_WORDS, ids=hex)
def test_mult_ve_cr_scalar_is_low_element_bits(fmt, word):
    """MULT.VE's CR scalar is the CR's low element-width bits decoded in the format (Q49)."""
    model, got, ops = _run(fmt, "MULT.VE LR0 CR3 0 LR2 CR15", RA_IDX, f"q49-{fmt}-{word}", word)
    low = word.to_bytes(4, "little")[: model.element_bytes]
    scalar = model.values(low)[0]
    window = ops["ra"][(RA_IDX + np.arange(LANES)) % (2 * LANES)]
    np.testing.assert_array_equal(got, model.mult_res(window * scalar))


@pytest.mark.parametrize("fmt", ALL_FORMATS, ids=lambda f: f.value)
@pytest.mark.parametrize(
    "mult", ["MULT.VE LR0 CR1 0 LR2 CR15", "MULT.RC.VE LR0 CR1 0 LR2 CR15"], ids=["ve", "rc_ve"]
)
def test_cr1_scalar_is_one_in_every_format(fmt, mult):
    """CR1 holds the integer 1, and as an element scalar reads as 1 in the active format."""
    index = RC_IDX if mult.startswith("MULT.RC") else RA_IDX
    model, got, ops = _run(fmt, mult, index, f"cr1-{fmt}-{mult}")
    if mult.startswith("MULT.RC"):
        window = _rc_window(ops)
    else:
        window = ops["ra"][(RA_IDX + np.arange(LANES)) % (2 * LANES)]
    np.testing.assert_array_equal(got, model.mult_res(window))
