"""The lane codec of each run mode, and :func:`codec_for`, which picks one.

:func:`codec_for` is the only mode switch the slot handlers use for lane
arithmetic. The register layout (:mod:`ipu_emu.geometry`, the operand resolver,
the cycle's read views) and the AaQ stores branch on the element width
themselves.

A codec has three domains:

- **Element**: the bytes a register or XMEM row holds, ``element_bytes`` each.
  An FP32 element is IEEE binary32; an NX8 element is one byte whose value
  depends on the instruction's QF and the row's metadata (ADR 0007), so
  :meth:`LaneCodec.decode` takes both.
- **Compute**: what :meth:`LaneCodec.decode` returns: float64.
- **Accumulator**: MULT_RES and R_ACC, stored as float32 (``acc_dtype``) and
  computed on in float64 in both modes. :meth:`store_acc` does the one
  rounding the scalar path did when it packed a result with
  ``struct.pack_into``.

NumPy's global error policy belongs to the embedding application, so every
float operation here runs under ``np.errstate(all="ignore")``: NaN, overflow
and signalling-NaN conversion behave like ``struct`` whatever that policy is.
"""

from __future__ import annotations

import functools
import struct
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt

from ipu_isa.errors import IsaValueError
from ipu_isa.faults import ArithmeticEvent
from ipu_isa.formats import FORMATS, ElementFormat, FormatInfo
from ipu_isa.geometry import LANES
from ipu_isa.nx8 import Nx8Format, decode_element, split_metadata

Buffer = bytes | bytearray | memoryview
Lanes = npt.NDArray[np.float64]

_INT32_MASK: Final = (1 << 32) - 1


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
    """The numerics of one run mode, a row of lanes at a time."""

    fmt: ElementFormat
    element_bytes: int  # 1 for NX8, 4 for FP32
    acc_dtype: np.dtype  # float32 (accumulator storage)

    def decode(
        self, raw: Buffer, count: int = LANES, qf: Nx8Format | None = None, metadata: int = 0
    ) -> Lanes:
        """The first ``count`` elements of ``raw`` in the compute domain.

        NX8 decodes each byte in format ``qf`` with the row's ``metadata`` word
        (``ARCH-stage-mult.md`` §3.2.1); FP32 ignores both.
        """
        ...

    def mult(self, a: Lanes, b: Lanes | np.generic | float) -> Lanes:
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
        """:meth:`add` as ACC and AGG compute it, for :meth:`store_acc_events`."""
        ...

    def acc_sub(self, a: Lanes, b: Lanes) -> Lanes:
        """:meth:`sub`, likewise."""
        ...

    def store_acc_events(self, out: Buffer, lanes: Lanes) -> frozenset[ArithmeticEvent]:
        """:meth:`store_acc` for ACC and AGG (E04-23): store the IEEE result
        (a float overflow is an infinity, a NaN stays) and return the
        arithmetic events it saw, without raising (Q53)."""
        ...

    def acc_min(self) -> np.generic:
        """The least accumulator value, the identity of :meth:`max` (-inf)."""
        ...


_NO_EVENTS: frozenset[ArithmeticEvent] = frozenset()
_OVERFLOW = frozenset({ArithmeticEvent.ACC_OVERFLOW})


class _Codec:
    """What every codec states about its format."""

    def __init__(self, info: FormatInfo) -> None:
        self.fmt: ElementFormat = info.format
        self.element_bytes: int = info.element_bytes
        self.acc_dtype: np.dtype = np.dtype("<f4")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.fmt})"


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


@functools.lru_cache(maxsize=1024)
def nx8_table(fmt: Nx8Format, metadata: int) -> npt.NDArray[np.float64]:
    """Every code's value in *fmt* for one row's *metadata* word (read-only).

    Entry ``c`` is ``nx8.decode_element(c, fmt, scale, dynamic_exponent)``
    rounded to float32, as MULT expands an element to FP32.
    """
    scale, dynamic_exponent = split_metadata(metadata)
    values = [decode_element(code, fmt, scale, dynamic_exponent) for code in range(256)]
    with np.errstate(all="ignore"):
        table = np.array(values, dtype=np.float64).astype(np.float32).astype(np.float64)
    table.flags.writeable = False
    return table


class Nx8Codec(_FloatCodec):
    """NX8 (the hardware): one byte per element, decoded with a QF and row metadata."""

    def decode(
        self, raw: Buffer, count: int = LANES, qf: Nx8Format | None = None, metadata: int = 0
    ) -> npt.NDArray[np.float64]:
        """See :meth:`LaneCodec.decode`.

        Raises:
            IsaValueError: ``qf`` is missing (an NX8 element has no value without one).
        """
        if qf is None:
            raise IsaValueError("an NX8 element is decoded with a QF format")
        return nx8_table(qf, metadata)[np.frombuffer(raw, dtype=np.uint8, count=count)]


class Fp32Codec(_FloatCodec):
    """FP32 (the emulator-only debug mode): IEEE binary32 elements, widened to float64."""

    def decode(
        self, raw: Buffer, count: int = LANES, qf: Nx8Format | None = None, metadata: int = 0
    ) -> npt.NDArray[np.float64]:
        """See :meth:`LaneCodec.decode`: ``qf`` and ``metadata`` are ignored."""
        return self.load_acc(raw, count)


def _make(info: FormatInfo) -> LaneCodec:
    if info.format is ElementFormat.NX8:
        return Nx8Codec(info)
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
