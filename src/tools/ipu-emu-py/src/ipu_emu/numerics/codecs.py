"""One lane codec per element format, and :func:`codec_for`, the only format switch.

A codec has three domains:

- **Element**: the bytes a register or XMEM row holds, ``element_bytes`` each.
- **Compute**: what :meth:`LaneCodec.decode` returns. int64 for the integer
  formats, float64 for the float formats. Widening keeps every intermediate
  exactly what the scalar Python arithmetic computed (Python ints and floats).
- **Accumulator**: MULT_RES and R_ACC, stored as int32 or float32
  (``acc_dtype``) and computed on in int64 or float64. :meth:`store_acc` does
  the one rounding the scalar path did when it packed a result with
  ``struct.pack_into``.

NumPy's global error policy belongs to the embedding application, so every
float operation here runs under ``np.errstate(all="ignore")``: NaN, overflow
and signalling-NaN conversion behave like ``struct`` whatever that policy is.
"""

from __future__ import annotations

import struct
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt

from ipu_isa.faults import ArithmeticEvent
from ipu_isa.formats import FORMATS, ElementFormat, FormatInfo, element_values, encode_element
from ipu_isa.geometry import LANES

Buffer = bytes | bytearray | memoryview
Lanes = npt.NDArray[np.int64] | npt.NDArray[np.float64]

_INT32_SIGN: Final = 1 << 31
_INT32_MASK: Final = (1 << 32) - 1

# A CR or LR-indexed scalar is the low byte of the register word, read as a
# signed byte, in every format but the 8-bit floats. E04-28 decodes it in the
# active format instead (Q49).
_SCALAR_BYTE_MASK: Final = 0xFF


def wrap_int32(values: Lanes) -> npt.NDArray[np.int64]:
    """``values`` wrapped to int32 two's complement, kept as int64 (``ipu_add``'s INT8 rule).

    Args:
        values: int64 lanes (or an int64 scalar).

    Returns:
        The wrapped values, in ``[-2**31, 2**31)``.
    """
    return ((values + _INT32_SIGN) & _INT32_MASK) - _INT32_SIGN


def _signed_byte(word: int) -> int:
    byte = word & _SCALAR_BYTE_MASK
    return byte - 0x100 if byte & 0x80 else byte


def _pack_lanes_one_by_one(lane_fmt: str, out: Buffer, values: list[float]) -> None:
    """The scalar store, replayed for a row that does not fit.

    ``struct.pack_into`` zeroes a field before it packs it and raises on a
    value the format cannot hold, so a row that fails at lane *k* leaves lanes
    ``0..k-1`` written, lane *k* zeroed and the rest untouched, and raises
    struct's own ``OverflowError``.
    """
    for i, v in enumerate(values):
        struct.pack_into(lane_fmt, out, 4 * i, v)


class LaneCodec(Protocol):
    """The numerics of one element format, a row of lanes at a time."""

    fmt: ElementFormat
    element_bytes: int  # 1 for INT8/E1..E7, 4 for FP32/INT32
    acc_dtype: np.dtype  # int32 or float32 (accumulator storage)

    def decode(self, raw: Buffer, count: int = LANES) -> Lanes:
        """The first ``count`` elements of ``raw`` in the compute domain."""
        ...

    def decode_scalar(self, word: int) -> np.generic:
        """A CR or LR-indexed scalar (a register word) in the compute domain."""
        ...

    def mult(self, a: Lanes, b: Lanes | np.generic) -> Lanes:
        """Element products, in the accumulator domain."""
        ...

    def add(self, a: Lanes, b: Lanes) -> Lanes:
        """Accumulator sums."""
        ...

    def sub(self, a: Lanes, b: Lanes) -> Lanes:
        """Accumulator differences ``a - b``."""
        ...

    def max(self, a: Lanes, b: Lanes) -> Lanes:
        """Python's ``max(a, b)`` per lane: ``b`` only where ``b > a`` (so a NaN ``a`` stays)."""
        ...

    def load_acc(self, raw: Buffer, count: int = LANES) -> Lanes:
        """The first ``count`` accumulator elements of ``raw``."""
        ...

    def store_acc(self, out: Buffer, lanes: Lanes) -> None:
        """Write ``lanes`` to the start of ``out`` as accumulator elements, rounding once."""
        ...

    def acc_add(self, a: Lanes, b: Lanes) -> Lanes:
        """:meth:`add` as ACC and AGG compute it, for :meth:`store_acc_events`:
        an integer sum is not yet wrapped, so the store can see it overflow."""
        ...

    def acc_sub(self, a: Lanes, b: Lanes) -> Lanes:
        """:meth:`sub`, likewise."""
        ...

    def store_acc_events(self, out: Buffer, lanes: Lanes) -> frozenset[ArithmeticEvent]:
        """:meth:`store_acc` for ACC and AGG (E04-23): store the IEEE result
        (a float overflow is an infinity, a NaN stays; an integer wraps) and
        return the arithmetic events it saw, without raising (Q53)."""
        ...

    def acc_min(self) -> np.generic:
        """The least accumulator value, the identity of :meth:`max` (INT32_MIN or -inf)."""
        ...

    def one(self) -> bytes:
        """One element of value 1."""
        ...


_NO_EVENTS: frozenset[ArithmeticEvent] = frozenset()
_OVERFLOW = frozenset({ArithmeticEvent.ACC_OVERFLOW})


class _Codec:
    """What every codec states about its format."""

    def __init__(self, info: FormatInfo) -> None:
        self.fmt: ElementFormat = info.format
        self.element_bytes: int = info.element_bytes
        self.acc_dtype: np.dtype = np.dtype("<i4" if not info.is_float else "<f4")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.fmt})"


class IntCodec(_Codec):
    """INT8 and INT32: products and sums in int64, wrapped to int32 like ``ipu_add``."""

    def __init__(self, info: FormatInfo) -> None:
        super().__init__(info)
        self._element_dtype = np.dtype(f"<i{info.element_bytes}")

    def decode(self, raw: Buffer, count: int = LANES) -> npt.NDArray[np.int64]:
        """See :meth:`LaneCodec.decode`."""
        return np.frombuffer(raw, dtype=self._element_dtype, count=count).astype(np.int64)

    def decode_scalar(self, word: int) -> np.int64:
        """See :meth:`LaneCodec.decode_scalar`: the word's low byte, signed."""
        return np.int64(_signed_byte(word))

    def mult(self, a: Lanes, b: Lanes | np.generic) -> npt.NDArray[np.int64]:
        """See :meth:`LaneCodec.mult`. An int32 product cannot overflow int64."""
        return wrap_int32(a * b)

    def add(self, a: Lanes, b: Lanes) -> npt.NDArray[np.int64]:
        """See :meth:`LaneCodec.add`."""
        return wrap_int32(a + b)

    def sub(self, a: Lanes, b: Lanes) -> npt.NDArray[np.int64]:
        """See :meth:`LaneCodec.sub`."""
        return wrap_int32(a - b)

    def max(self, a: Lanes, b: Lanes) -> Lanes:
        """See :meth:`LaneCodec.max`."""
        return np.where(b > a, b, a)

    def load_acc(self, raw: Buffer, count: int = LANES) -> npt.NDArray[np.int64]:
        """See :meth:`LaneCodec.load_acc`."""
        return np.frombuffer(raw, dtype="<i4", count=count).astype(np.int64)

    def store_acc(self, out: Buffer, lanes: Lanes) -> None:
        """See :meth:`LaneCodec.store_acc`: wraps to int32, so it never fails."""
        raw = (np.asarray(lanes, dtype=np.int64) & _INT32_MASK).astype("<u4").tobytes()
        out[: len(raw)] = raw

    def acc_add(self, a: Lanes, b: Lanes) -> npt.NDArray[np.int64]:
        """See :meth:`LaneCodec.acc_add`: exact in int64."""
        return np.asarray(a, dtype=np.int64) + b

    def acc_sub(self, a: Lanes, b: Lanes) -> npt.NDArray[np.int64]:
        """See :meth:`LaneCodec.acc_sub`: exact in int64."""
        return np.asarray(a, dtype=np.int64) - b

    def store_acc_events(self, out: Buffer, lanes: Lanes) -> frozenset[ArithmeticEvent]:
        """See :meth:`LaneCodec.store_acc_events`: ACC_OVERFLOW when a lane wraps."""
        values = np.asarray(lanes, dtype=np.int64)
        out[: values.size * 4] = (values & _INT32_MASK).astype("<u4").tobytes()
        if values.size and (values.min() < -_INT32_SIGN or values.max() >= _INT32_SIGN):
            return _OVERFLOW
        return _NO_EVENTS

    def acc_min(self) -> np.int64:
        """See :meth:`LaneCodec.acc_min`."""
        return np.int64(-_INT32_SIGN)

    def one(self) -> bytes:
        """See :meth:`LaneCodec.one`."""
        return (1).to_bytes(self.element_bytes, "little", signed=True)


def _first_nan_wins(a: Lanes | np.generic, result: Lanes) -> Lanes:
    """``result`` with ``a``'s NaN wherever ``a`` is NaN.

    A scalar ``a + b`` on x86-64 returns its first operand's NaN when both are
    NaN (so a payload survives); numpy's vector loops may return either. NaN
    lanes are rare, so the common case costs one ``isnan`` over the result.
    """
    if not np.isnan(result).any():
        return result
    return np.where(np.isnan(a), a, result)


class _FloatCodec(_Codec):
    """Accumulation for every float format: float64 arithmetic, one float32 rounding."""

    def mult(self, a: Lanes, b: Lanes | np.generic) -> npt.NDArray[np.float64]:
        """See :meth:`LaneCodec.mult`."""
        with np.errstate(all="ignore"):
            return _first_nan_wins(a, a * b)

    def add(self, a: Lanes, b: Lanes) -> npt.NDArray[np.float64]:
        """See :meth:`LaneCodec.add`."""
        with np.errstate(all="ignore"):
            return _first_nan_wins(a, a + b)

    def sub(self, a: Lanes, b: Lanes) -> npt.NDArray[np.float64]:
        """See :meth:`LaneCodec.sub`."""
        with np.errstate(all="ignore"):
            return _first_nan_wins(a, a - b)

    def max(self, a: Lanes, b: Lanes) -> Lanes:
        """See :meth:`LaneCodec.max`."""
        with np.errstate(all="ignore"):
            return np.where(b > a, b, a)

    def load_acc(self, raw: Buffer, count: int = LANES) -> npt.NDArray[np.float64]:
        """See :meth:`LaneCodec.load_acc`."""
        with np.errstate(all="ignore"):
            return np.frombuffer(raw, dtype="<f4", count=count).astype(np.float64)

    def store_acc(self, out: Buffer, lanes: Lanes) -> None:
        """See :meth:`LaneCodec.store_acc`.

        Raises:
            OverflowError: A finite lane is too large for float32 (struct's
                error). Lanes before it are written and it is zeroed, as
                lane-by-lane ``struct.pack_into`` leaves them.
        """
        values = np.asarray(lanes, dtype=np.float64)
        # numpy 2 refuses to re-enter one errstate object, so this stays per call.
        with np.errstate(all="ignore"):
            narrowed = values.astype("<f4")
        if not np.isfinite(narrowed).all() and np.any(np.isinf(narrowed) & np.isfinite(values)):
            # struct.pack_into raises where numpy would store an infinity: replay it.
            _pack_lanes_one_by_one("<f", out, values.tolist())
            return
        raw = narrowed.tobytes()
        out[: len(raw)] = raw

    def acc_add(self, a: Lanes, b: Lanes) -> npt.NDArray[np.float64]:
        """See :meth:`LaneCodec.acc_add`: :meth:`add` (rounded once, at the store)."""
        return self.add(a, b)

    def acc_sub(self, a: Lanes, b: Lanes) -> npt.NDArray[np.float64]:
        """See :meth:`LaneCodec.acc_sub`: :meth:`sub`."""
        return self.sub(a, b)

    def store_acc_events(self, out: Buffer, lanes: Lanes) -> frozenset[ArithmeticEvent]:
        """See :meth:`LaneCodec.store_acc_events`: ACC_OVERFLOW when a finite
        lane rounds to an infinity, NAN_RESULT when a lane is NaN."""
        values = np.asarray(lanes, dtype=np.float64)
        with np.errstate(all="ignore"):
            narrowed = values.astype("<f4")
        raw = narrowed.tobytes()
        out[: len(raw)] = raw
        if np.isfinite(narrowed).all():
            return _NO_EVENTS
        events: set[ArithmeticEvent] = set()
        if np.any(np.isinf(narrowed) & np.isfinite(values)):
            events.add(ArithmeticEvent.ACC_OVERFLOW)
        if np.isnan(narrowed).any():
            events.add(ArithmeticEvent.NAN_RESULT)
        return frozenset(events)

    def acc_min(self) -> np.float64:
        """See :meth:`LaneCodec.acc_min`."""
        return np.float64(-np.inf)


class Fp8Codec(_FloatCodec):
    """E1..E7: decoded through a 256-entry table built from ``ipu_isa``'s element codec."""

    def __init__(self, info: FormatInfo) -> None:
        super().__init__(info)
        self._table = np.array(element_values(info.format), dtype=np.float64)
        self._table.flags.writeable = False
        self._one = bytes([encode_element(1.0, info.format)])

    def decode(self, raw: Buffer, count: int = LANES) -> npt.NDArray[np.float64]:
        """See :meth:`LaneCodec.decode`."""
        return self._table[np.frombuffer(raw, dtype=np.uint8, count=count)]

    def decode_scalar(self, word: int) -> np.float64:
        """See :meth:`LaneCodec.decode_scalar`: the word's low byte as an element."""
        return self._table[word & _SCALAR_BYTE_MASK]

    def one(self) -> bytes:
        """See :meth:`LaneCodec.one`."""
        return self._one


class Fp32Codec(_FloatCodec):
    """FP32 (wide-vector debug mode): IEEE binary32 elements, widened to float64."""

    def decode(self, raw: Buffer, count: int = LANES) -> npt.NDArray[np.float64]:
        """See :meth:`LaneCodec.decode`."""
        return self.load_acc(raw, count)

    def decode_scalar(self, word: int) -> np.float64:
        """See :meth:`LaneCodec.decode_scalar`: the word's low byte, signed, as a float."""
        return np.float64(_signed_byte(word))

    def one(self) -> bytes:
        """See :meth:`LaneCodec.one`."""
        return struct.pack("<f", 1.0)


def _make(info: FormatInfo) -> LaneCodec:
    if not info.is_float:
        return IntCodec(info)
    if info.element_bits == 8:
        return Fp8Codec(info)
    return Fp32Codec(info)


_CODECS: Final[dict[ElementFormat, LaneCodec]] = {fmt: _make(FORMATS[fmt]) for fmt in ElementFormat}


def codec_for(fmt: ElementFormat) -> LaneCodec:
    """The lane codec of ``fmt`` (one shared instance per format).

    Args:
        fmt: Any element format.

    Returns:
        Its codec.
    """
    return _CODECS[ElementFormat(fmt)]
