"""The lane codecs (``ipu_emu.numerics``) against independent references (E04-13, L5, ADR 0007).

NX8 decode is checked against hand-worked values (``ARCH-stage-mult.md``
§3.2.1) and against ``ipu_isa.nx8.decode_element`` for every code; the FP32
accumulator arithmetic against the scalar path the handlers ran before the
codecs (Python floats stored with ``struct``). Results are compared as stored
bytes, so the sign of zero and NaN bits count.
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
from ipu_emu.numerics import codec_for
from ipu_isa.errors import IsaValueError
from ipu_isa.faults import ArithmeticEvent
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.geometry import LANES
from ipu_isa.nx8 import Nx8Format, decode_element, metadata_word

_ROWS = 7813  # 7813 * 128 = 1_000_064 accumulator pairs
_PAIRS = _ROWS * LANES
_SENTINEL = 0xA5

E4 = Nx8Format(sign=1, exp_format=4, num_bits=8)
DYNAMIC = Nx8Format(sign=1, exp_format=0, num_bits=8, dynamic=True)


# ---------------------------------------------------------------------------
# NX8 decode (ADR 0007)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "scale", "value"),
    [
        (0x00, 127, 1.0),  # sign 0, exp 0 (distance 0 below 2^0), man 0
        (0x08, 127, 0.5),  # exp 1: one binade below the scale
        (0x0C, 127, 0.75),  # exp 1, man 4/8
        (0x80, 127, -1.0),  # the sign bit
        (0x78, 127, 0.0),  # exp 15 (clamp), man 0: zero
        (0x00, 130, 8.0),  # the scale moves the binade: 2^(130-127)
        (0x79, 127, 2.0**-17),  # clamp exp, man 1: 1/8 * 2^(0 - 14)
    ],
)
def test_nx8_decode_hand_worked_codes(code, scale, value):
    codec = codec_for(ElementFormat.NX8)
    assert codec.decode(bytes([code]), 1, E4, metadata_word(scale)).tolist() == [value]


@pytest.mark.parametrize("fmt", [E4, DYNAMIC, Nx8Format(sign=0, exp_format=3, num_bits=6)])
@pytest.mark.parametrize("metadata", [metadata_word(127), metadata_word(140, 1), 0])
def test_nx8_decode_every_code_is_the_isa_expansion_rounded_to_fp32(fmt, metadata):
    codec = codec_for(ElementFormat.NX8)
    decoded = codec.decode(bytes(range(256)), 256, fmt, metadata)
    scale, dyn = metadata & 0xFF, (metadata >> 8) & 1
    expected = [decode_element(c, fmt, scale, dyn) for c in range(256)]
    assert decoded.astype(np.float32).tobytes() == np.asarray(expected, dtype=np.float32).tobytes()


def test_nx8_decode_needs_a_qf():
    with pytest.raises(IsaValueError, match="QF"):
        codec_for(ElementFormat.NX8).decode(bytes(LANES))


def test_fp32_decode_ignores_qf_and_metadata():
    codec = codec_for(ElementFormat.FP32)
    raw = struct.pack("<2f", 1.5, -2.0)
    assert codec.decode(raw, 2).tolist() == codec.decode(raw, 2, E4, 0x185).tolist() == [1.5, -2.0]


def test_mult_by_a_scalar_broadcasts():
    codec = codec_for(ElementFormat.NX8)
    row = codec.decode(bytes(range(128)), LANES, E4, metadata_word(127))
    for scalar in (0.0, 1.0, -0.5, 3.25):
        got = bytearray(4 * LANES)
        codec.store_acc(got, codec.mult(row, scalar))
        assert bytes(got) == struct.pack(f"<{LANES}f", *[float(v) * scalar for v in row]), scalar


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


def _scalar_ops(lane_fmt: str, a_raw: bytes, b_raw: bytes, *, wide_mult: bool) -> dict:
    """Each op's stored bytes and failing rows, computed the scalar way."""
    a = list(struct.unpack(f"<{_PAIRS}{lane_fmt[1]}", a_raw))
    b = list(struct.unpack(f"<{_PAIRS}{lane_fmt[1]}", b_raw))
    results = {
        "add": [x + y for x, y in zip(a, b, strict=True)],
        "sub": [x - y for x, y in zip(a, b, strict=True)],
        "max": [max(x, y) for x, y in zip(a, b, strict=True)],
        "neg": [0.0 - y for y in b],
    }
    if wide_mult:
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


def _check_accumulator_ops(fmt: ElementFormat, pairs) -> None:
    a_raw, b_raw, expected = pairs
    codec = codec_for(fmt)
    a = codec.load_acc(a_raw, _PAIRS)
    b = codec.load_acc(b_raw, _PAIRS)
    ops = ["add", "sub", "max", "neg"]
    if FORMATS[fmt].element_bits == 32:
        # FP32 elements are accumulator-width: their products are random pairs too.
        ops.append("mult")
        a, b = codec.decode(a_raw, _PAIRS), codec.decode(b_raw, _PAIRS)
    for op in ops:
        got_bytes, got_failures = _store_rows_codec(codec, _codec_op(codec, op, a, b))
        want_bytes, want_failures = expected[op]
        assert got_failures == want_failures, op
        assert got_bytes == want_bytes, op


@pytest.mark.parametrize("fmt", list(ElementFormat), ids=str)
def test_float_accumulator_ops_match_scalar(fmt, float_pairs):
    _check_accumulator_ops(fmt, float_pairs)


def test_random_float_pairs_overflow_some_rows(float_pairs):
    # Guards the data: the overflow path must actually be exercised.
    _, _, expected = float_pairs
    assert expected["add"][1] and expected["mult"][1]
    assert len(expected["add"][1]) < _ROWS // 2


def _acc_rows(codec, op: str, a: np.ndarray, b: np.ndarray):
    """The ACC path of *op* (E04-23): acc_add/acc_sub, then store_acc_events, row by row."""
    lanes = codec.acc_sub(np.zeros_like(b), b) if op == "neg" else getattr(codec, "acc_" + op)(a, b)
    for row in range(_ROWS):
        out = bytearray(4 * LANES)
        events = codec.store_acc_events(out, lanes[row * LANES : (row + 1) * LANES])
        yield row, bytes(out), events


@pytest.mark.parametrize("fmt", list(ElementFormat), ids=str)
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
    assert codec.acc_dtype.kind == "f"


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
