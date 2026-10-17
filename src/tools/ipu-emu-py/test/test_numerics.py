"""The lane codecs (``ipu_emu.numerics``) are bit-identical to the scalar arithmetic (E04-13, L5).

The reference is the scalar path the handlers run today: each element decoded
by ``ipu_isa.formats.decode_element`` (or as a signed byte), multiplied or
accumulated as Python ints and floats (``ipu_mult``, ``ipu_add``, ``ipu_sub``,
``max``) and stored with ``struct``. Results are compared as stored bytes, so
the sign of zero and NaN bits count.
"""

from __future__ import annotations

import ast
import random
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

import ipu_emu.numerics
from ipu_emu.ipu_math import ipu_add, ipu_mult, ipu_sub
from ipu_emu.numerics import codec_for
from ipu_isa.faults import ArithmeticEvent
from ipu_isa.formats import FORMATS, ElementFormat, decode_element
from ipu_isa.geometry import LANES

EIGHT_BIT = [f for f, info in FORMATS.items() if info.element_bits == 8]
FLOAT8 = [f for f in EIGHT_BIT if FORMATS[f].is_float]
INT_ACC = [f for f, info in FORMATS.items() if not info.is_float]
FLOAT_ACC = [f for f, info in FORMATS.items() if info.is_float]

# Any float-accumulator format selects ipu_add's float branch.
_INT_FORMAT = ElementFormat.INT8
_FLOAT_FORMAT = ElementFormat.E4

_ROWS = 7813  # 7813 * 128 = 1_000_064 accumulator pairs
_PAIRS = _ROWS * LANES
_SENTINEL = 0xA5


def _scalar_values(fmt: ElementFormat) -> list[int | float]:
    """Every code's value as the scalar path sees it: a Python int (INT8) or float."""
    if FORMATS[fmt].is_float:
        return [decode_element(c, fmt) for c in range(256)]
    return [int(decode_element(c, fmt)) for c in range(256)]


def _lane_fmt(fmt: ElementFormat) -> str:
    return "<f" if FORMATS[fmt].is_float else "<i"


def _byte_grid() -> tuple[bytes, bytes]:
    """Every (a, b) byte pair, as two 65536-byte rows."""
    codes = np.arange(256, dtype=np.uint8)
    return np.repeat(codes, 256).tobytes(), np.tile(codes, 256).tobytes()


# ---------------------------------------------------------------------------
# Exhaustive (L5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", EIGHT_BIT, ids=str)
def test_decode_every_code(fmt):
    codec = codec_for(fmt)
    decoded = codec.decode(bytes(range(256)), 256)
    expected = _scalar_values(fmt)
    if FORMATS[fmt].is_float:
        assert decoded.astype(np.float32).tobytes() == struct.pack("<256f", *expected)
    else:
        assert decoded.tolist() == expected


@pytest.mark.parametrize("fmt", EIGHT_BIT, ids=str)
def test_decode_scalar_reads_the_low_byte(fmt):
    codec = codec_for(fmt)
    decoded = codec.decode(bytes(range(256)), 256)
    for code in range(256):
        word = 0x5A5A_5A00 | code
        assert np.array_equal(
            np.asarray(codec.decode_scalar(word)), decoded[code], equal_nan=True
        ), code


@pytest.mark.parametrize("fmt", EIGHT_BIT, ids=str)
def test_mult_every_byte_pair(fmt):
    codec = codec_for(fmt)
    a_raw, b_raw = _byte_grid()
    count = len(a_raw)
    got = bytearray(4 * count)
    codec.store_acc(got, codec.mult(codec.decode(a_raw, count), codec.decode(b_raw, count)))

    values = _scalar_values(fmt)
    products = [values[a] * values[b] for a, b in zip(a_raw, b_raw, strict=True)]
    lane = _lane_fmt(fmt)[1]
    assert bytes(got) == struct.pack(f"<{count}{lane}", *products)

    dtype = fmt
    via_ipu_mult = [ipu_mult(a, b, dtype) for a, b in zip(a_raw, b_raw, strict=True)]
    assert struct.pack(f"<{count}{lane}", *via_ipu_mult) == bytes(got)


@pytest.mark.parametrize("fmt", EIGHT_BIT, ids=str)
def test_mult_by_a_scalar_broadcasts(fmt):
    codec = codec_for(fmt)
    row = bytes(range(128))
    dtype = fmt
    for scalar in (0x00, 0x01, 0x38, 0x7F, 0x80, 0xC5, 0xFF):
        got = bytearray(4 * LANES)
        codec.store_acc(got, codec.mult(codec.decode(row), codec.decode_scalar(scalar)))
        expected = [ipu_mult(c, scalar, dtype) for c in row]
        assert bytes(got) == struct.pack(f"<{LANES}{_lane_fmt(fmt)[1]}", *expected), scalar


# ---------------------------------------------------------------------------
# Random accumulator pairs
# ---------------------------------------------------------------------------


def _float32_specials() -> list[int]:
    f32 = np.finfo(np.float32)
    values = [0.0, -0.0, np.inf, -np.inf, f32.max, -f32.max, f32.tiny, -f32.tiny, 1.0, -1.0]
    bits = [int(np.float32(v).view(np.uint32)) for v in values]
    bits += [
        0x7FC00000,  # quiet NaN
        0xFFC00000,  # negative quiet NaN
        0x7FA00000,  # signalling NaN
        0x7FC12345,  # NaN with a payload
        0x00000001,  # smallest subnormal
        0x807FFFFF,  # largest negative subnormal
        0x7F7FFFFE,  # one ulp below the maximum
        0xFF7FFFFE,
    ]
    return bits


def _random_float32_bits(rng: random.Random) -> np.ndarray:
    """Bit patterns: mostly uniform (every class), some specials, some near the maximum."""
    specials = _float32_specials()
    out = np.empty(_PAIRS, dtype=np.uint32)
    for i in range(_PAIRS):
        r = rng.random()
        if r < 0.9:
            out[i] = rng.getrandbits(32)
        elif r < 0.97:
            out[i] = rng.choice(specials)
        else:
            # Exponent 254: a sum or a product of two of these can overflow float32.
            out[i] = (rng.getrandbits(1) << 31) | (254 << 23) | rng.getrandbits(23)
    return out


def _random_int32_bits(rng: random.Random) -> np.ndarray:
    edges = [0, 1, -1, 2**31 - 1, -(2**31), 2**31 - 2, -(2**31) + 1]
    values = [
        rng.choice(edges) if rng.random() < 0.05 else rng.randrange(-(2**31), 2**31)
        for _ in range(_PAIRS)
    ]
    return np.array(values, dtype=np.int64).astype(np.int32).view(np.uint32)


def _store_rows_scalar(lane_fmt: str, values: list) -> tuple[bytes, dict[int, str]]:
    """Today's row store: ``struct.pack`` per row, lane by lane when the row fails."""
    out = bytearray([_SENTINEL]) * (4 * len(values))
    failures: dict[int, str] = {}
    for row in range(len(values) // LANES):
        lanes = values[row * LANES : (row + 1) * LANES]
        try:
            out[row * 4 * LANES : (row + 1) * 4 * LANES] = struct.pack(
                f"<{LANES}{lane_fmt[1]}", *lanes
            )
        except OverflowError:
            view = memoryview(out)[row * 4 * LANES :]
            try:
                for i, v in enumerate(lanes):
                    struct.pack_into(lane_fmt, view, 4 * i, v)
            except OverflowError as exc:
                failures[row] = str(exc)
    return bytes(out), failures


def _store_rows_codec(codec, lanes: np.ndarray) -> tuple[bytes, dict[int, str]]:
    out = bytearray([_SENTINEL]) * (4 * lanes.size)
    failures: dict[int, str] = {}
    for row in range(lanes.size // LANES):
        try:
            codec.store_acc(
                memoryview(out)[row * 4 * LANES :], lanes[row * LANES : (row + 1) * LANES]
            )
        except OverflowError as exc:
            failures[row] = str(exc)
    return bytes(out), failures


def _wrap32(value: int) -> int:
    return ((value + 2**31) % 2**32) - 2**31


def _scalar_ops(lane_fmt: str, a_raw: bytes, b_raw: bytes, *, wide_mult: bool) -> dict:
    """Each op's stored bytes and failing rows, computed the scalar way."""
    a = list(struct.unpack(f"<{_PAIRS}{lane_fmt[1]}", a_raw))
    b = list(struct.unpack(f"<{_PAIRS}{lane_fmt[1]}", b_raw))
    dtype = _INT_FORMAT if lane_fmt == "<i" else _FLOAT_FORMAT
    results = {
        "add": [ipu_add(x, y, dtype) for x, y in zip(a, b, strict=True)],
        "sub": [ipu_sub(x, y, dtype) for x, y in zip(a, b, strict=True)],
        "max": [max(x, y) for x, y in zip(a, b, strict=True)],
        "neg": [ipu_sub(0 if lane_fmt == "<i" else 0.0, y, dtype) for y in b],
    }
    if wide_mult:
        if lane_fmt == "<i":
            results["mult"] = [_wrap32(x * y) for x, y in zip(a, b, strict=True)]
        else:
            results["mult"] = [x * y for x, y in zip(a, b, strict=True)]
    return {op: _store_rows_scalar(lane_fmt, values) for op, values in results.items()}


def _codec_op(codec, op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if op == "neg":
        return codec.sub(np.zeros_like(b), b)
    return getattr(codec, op)(a, b)


@pytest.fixture(scope="module")
def float_pairs():
    rng = random.Random(0xE0413)
    a_raw = _random_float32_bits(rng).tobytes()
    b_raw = _random_float32_bits(rng).tobytes()
    return a_raw, b_raw, _scalar_ops("<f", a_raw, b_raw, wide_mult=True)


@pytest.fixture(scope="module")
def int_pairs():
    rng = random.Random(0xE0414)
    a_raw = _random_int32_bits(rng).tobytes()
    b_raw = _random_int32_bits(rng).tobytes()
    return a_raw, b_raw, _scalar_ops("<i", a_raw, b_raw, wide_mult=True)


def _check_accumulator_ops(fmt: ElementFormat, pairs) -> None:
    a_raw, b_raw, expected = pairs
    codec = codec_for(fmt)
    a = codec.load_acc(a_raw, _PAIRS)
    b = codec.load_acc(b_raw, _PAIRS)
    ops = ["add", "sub", "max", "neg"]
    if FORMATS[fmt].element_bits == 32:
        # FP32 and INT32 elements are accumulator-width: their products are random pairs too.
        ops.append("mult")
        a, b = codec.decode(a_raw, _PAIRS), codec.decode(b_raw, _PAIRS)
    for op in ops:
        got_bytes, got_failures = _store_rows_codec(codec, _codec_op(codec, op, a, b))
        want_bytes, want_failures = expected[op]
        assert got_failures == want_failures, op
        assert got_bytes == want_bytes, op


@pytest.mark.parametrize("fmt", FLOAT_ACC, ids=str)
def test_float_accumulator_ops_match_scalar(fmt, float_pairs):
    _check_accumulator_ops(fmt, float_pairs)


def test_random_float_pairs_overflow_some_rows(float_pairs):
    # Guards the data: the overflow path must actually be exercised.
    _, _, expected = float_pairs
    assert expected["add"][1] and expected["mult"][1]
    assert len(expected["add"][1]) < _ROWS // 2


@pytest.mark.parametrize("fmt", INT_ACC, ids=str)
def test_int_accumulator_ops_match_scalar(fmt, int_pairs):
    _check_accumulator_ops(fmt, int_pairs)


def _acc_rows(codec, op: str, a: np.ndarray, b: np.ndarray):
    """The ACC path of *op* (E04-23): acc_add/acc_sub, then store_acc_events, row by row."""
    lanes = codec.acc_sub(np.zeros_like(b), b) if op == "neg" else getattr(codec, "acc_" + op)(a, b)
    for row in range(_ROWS):
        out = bytearray(4 * LANES)
        events = codec.store_acc_events(out, lanes[row * LANES : (row + 1) * LANES])
        yield row, bytes(out), events


@pytest.mark.parametrize("fmt", FLOAT_ACC, ids=str)
def test_the_acc_store_with_events_matches_scalar_where_it_fits(fmt, float_pairs):
    # Where struct's lane-by-lane store succeeds, the ACC path stores the same
    # bits (NaN payloads and quieting included) and reports no overflow; where
    # it raised, the ACC path reports ACC_OVERFLOW instead (its lanes hold inf).
    a_raw, b_raw, expected = float_pairs
    codec = codec_for(fmt)
    a, b = codec.load_acc(a_raw, _PAIRS), codec.load_acc(b_raw, _PAIRS)
    for op in ("add", "sub", "neg"):
        want_bytes, want_failures = expected[op]
        for row, got, events in _acc_rows(codec, op, a, b):
            if row in want_failures:
                assert ArithmeticEvent.ACC_OVERFLOW in events, (op, row)
                continue
            assert got == want_bytes[row * 4 * LANES : (row + 1) * 4 * LANES], (op, row)
            assert ArithmeticEvent.ACC_OVERFLOW not in events, (op, row)
            nan = bool(np.isnan(np.frombuffer(got, dtype="<f4")).any())
            assert (ArithmeticEvent.NAN_RESULT in events) == nan, (op, row)


@pytest.mark.parametrize("fmt", INT_ACC, ids=str)
def test_the_int_acc_store_wraps_and_reports_exactly_the_wraps(fmt, int_pairs):
    a_raw, b_raw, expected = int_pairs
    codec = codec_for(fmt)
    a, b = codec.load_acc(a_raw, _PAIRS), codec.load_acc(b_raw, _PAIRS)
    exact = {"add": a + b, "sub": a - b, "neg": -b}
    for op in ("add", "sub", "neg"):
        want_bytes, _ = expected[op]
        for row, got, events in _acc_rows(codec, op, a, b):
            assert got == want_bytes[row * 4 * LANES : (row + 1) * 4 * LANES], (op, row)
            lanes = exact[op][row * LANES : (row + 1) * LANES]
            wrapped = bool(((lanes < -(2**31)) | (lanes >= 2**31)).any())
            assert (ArithmeticEvent.ACC_OVERFLOW in events) == wrapped, (op, row)


@pytest.mark.parametrize("fmt", INT_ACC, ids=str)
def test_int_accumulate_wraps_at_2_pow_31(fmt):
    codec = codec_for(fmt)
    top, bottom = 2**31 - 1, -(2**31)
    a = np.array([top, bottom, top, bottom, -1], dtype=np.int64)
    b = np.array([1, -1, top, bottom, bottom], dtype=np.int64)
    for op, scalar in (("add", ipu_add), ("sub", ipu_sub)):
        got = bytearray(4 * a.size)
        codec.store_acc(got, getattr(codec, op)(a, b))
        expected = [scalar(int(x), int(y), _INT_FORMAT) for x, y in zip(a, b, strict=True)]
        assert bytes(got) == struct.pack(f"<{a.size}i", *expected), op
    assert codec.add(np.array([top]), np.array([1])).tolist() == [bottom]
    assert codec.sub(np.array([bottom]), np.array([1])).tolist() == [top]


def test_float_store_overflow_leaves_struct_state():
    codec = codec_for(ElementFormat.FP32)
    big = float(np.finfo(np.float32).max)
    lanes = np.array([1.0, 2.0, big * 2, 3.0], dtype=np.float64)
    out = bytearray([_SENTINEL]) * 16
    with pytest.raises(OverflowError, match="float too large to pack with f format"):
        codec.store_acc(out, lanes)
    assert bytes(out) == struct.pack("<2f", 1.0, 2.0) + bytes(4) + bytes([_SENTINEL]) * 4
    # An infinity is not an overflow: it stores as infinity.
    codec.store_acc(out, np.array([np.inf, -np.inf, 0.0, -0.0]))
    assert bytes(out) == struct.pack("<4f", np.inf, -np.inf, 0.0, -0.0)


# ---------------------------------------------------------------------------
# The module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", list(ElementFormat), ids=str)
def test_codec_for_every_format(fmt):
    codec = codec_for(fmt)
    info = FORMATS[fmt]
    assert codec is codec_for(ElementFormat(str(fmt)))  # cached, accepts the string form
    assert codec.fmt is fmt
    assert codec.element_bytes == info.element_bytes
    assert codec.acc_dtype.itemsize == info.accumulator_bytes
    assert codec.acc_dtype.kind == ("f" if info.is_float else "i")
    one = codec.one()
    assert len(one) == info.element_bytes
    assert codec.decode(one, 1).tolist() == [1]


def test_numerics_imports_only_numpy_stdlib_and_ipu_isa():
    allowed = {"numpy", "ipu_isa", "__future__"} | set(sys.stdlib_module_names)
    for path in Path(ipu_emu.numerics.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.startswith("ipu_emu.numerics"):
                    continue
                assert name.split(".")[0] in allowed, f"{path.name} imports {name}"
