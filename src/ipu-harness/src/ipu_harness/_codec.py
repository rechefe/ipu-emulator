"""Element values as a variant holds them: its range, its rows' bytes and metadata.

One place for the harness (the manifest's ``range`` check, packing, goldens):
every per-mode branch is here, none in the callers (ADR 0007).

* FP32: a row is 128 IEEE binary32 values (512 bytes), no metadata.
* NX8: a row is 128 elements of the variant's type plus one metadata word
  (``scale``, ``dynamic_exponent``). The scale is chosen per row from the
  row's largest value (``ipu_isa.nx8.encode_row``, the AaQ quantizer), so how
  a value is held depends on the other values of its row: the harness rounds
  rows, never single values. Zero-padded lanes do not change the scale. A
  zero value reads back 0 only when the row scale (biased) is at least
  ``2^fe + fm``; below that, and in a row of zeros, it reads back as a tiny
  nonzero value, as the spec's quantizer gives (OQ-46).
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt

from ipu_harness.variants import Variant
from ipu_isa import cr_halves
from ipu_isa.geometry import LANES
from ipu_isa.nx8 import Nx8Format, decode_element, decode_row, encode_row, split_metadata

Array = npt.NDArray[Any]

_FP32_MAX = float(np.finfo(np.float32).max)

__all__ = [
    "Array",
    "decode_rows",
    "encode_rows",
    "held_rows",
    "limits",
    "row_grid",
    "scalar_neighbours",
    "scalar_word",
]


def limits(v: Variant) -> tuple[float, float]:
    """The lowest and highest finite value ``v`` holds.

    NX8 scales each row, so its range is FP32's (an unsigned type: from 0).
    """
    if v.qf is not None and not v.qf.sign:
        return 0.0, _FP32_MAX
    return -_FP32_MAX, _FP32_MAX


def _as_rows(rows: Array) -> Array:
    out = np.asarray(rows, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != LANES:
        out = out.reshape(-1, LANES)
    return out


def encode_rows(rows: Array, v: Variant) -> tuple[bytes, tuple[int, ...] | None]:
    """``rows`` (``(n, 128)``, finite) as ``v`` stores them.

    Returns:
        The rows' bytes, and each row's metadata word (NX8) or ``None`` (FP32).
        Values beyond FP32's range saturate to the largest float32.
    """
    rows = _as_rows(rows)
    # FP32's range only: NX8 encodes a negative value of an unsigned type as 0
    # and saturates past the row's scale itself (ipu_isa.nx8.encode_row), as
    # ACTIVATE.QUANTIZE does.
    clipped = np.clip(rows, -_FP32_MAX, _FP32_MAX)
    if v.qf is None:
        return clipped.astype("<f4").tobytes(), None
    data = bytearray()
    metadata = []
    for row in clipped.astype(np.float32).astype(np.float64):
        encoded = encode_row(row.tolist(), v.qf)
        data += encoded.elements
        metadata.append(encoded.metadata)
    return bytes(data), tuple(metadata)


def decode_rows(raw: bytes, v: Variant, metadata: tuple[int, ...] | None) -> Array:
    """The ``(n, 128)`` values ``raw`` holds (inverse of :func:`encode_rows`).

    Args:
        raw: Whole rows of ``v``.
        v: The variant.
        metadata: Each row's metadata word (NX8; ``None`` reads as 0).
    """
    if v.qf is None:
        return np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(-1, LANES)
    count = len(raw) // LANES
    words = metadata if metadata is not None else (0,) * count
    out = np.empty((count, LANES), dtype=np.float64)
    for index in range(count):
        scale, dyn = split_metadata(words[index])
        out[index] = decode_row(raw[index * LANES : (index + 1) * LANES], v.qf, scale, dyn)
    return out


def held_rows(rows: Array, v: Variant) -> Array:
    """``rows`` as ``v`` holds them: encoded and decoded, row by row."""
    data, metadata = encode_rows(rows, v)
    return decode_rows(data, v, metadata)


def row_grid(v: Variant, metadata: int) -> Array:
    """Every finite value one NX8 row with ``metadata`` holds, ascending (its
    steps: the "one NX8 step" tolerance measures distances on it)."""
    if v.qf is None:
        raise ValueError("an FP32 row has no NX8 grid")
    scale, dyn = split_metadata(metadata)
    codes = bytes(range(256))
    values = np.asarray(decode_row(codes, v.qf, scale, dyn), dtype=np.float64)
    grid: Array = np.unique(values[np.isfinite(values)])
    return grid


# --- element scalars: a float parameter bound to a CR --------------------------------------


@cache
def _nx8_constants(fe: int, sign: int) -> dict[float, tuple[int, int]]:
    """Every value a MULT constant of type (``fe``, ``sign``) holds, to its
    (code, scale): the smallest scale and then code that hold it. A value
    is expanded to FP32 as the emulator reads a constant (an underflow reads
    0, an overflow an infinity, which no parameter can be)."""
    fmt = Nx8Format(sign=sign, exp_format=fe, num_bits=8)
    out: dict[float, tuple[int, int]] = {}
    for scale in range(256):
        for code in range(256):
            with np.errstate(all="ignore"):
                value = float(np.float32(decode_element(code, fmt, scale)))
            if math.isfinite(value):
                out.setdefault(value, (code, scale))
    return out


def _constant_types(v: Variant) -> list[tuple[int, int]]:
    """The constant types tried for ``v``: its own (``fe``, ``sign``) first
    (a dynamic variant: ``fe`` 2, then 1), then every other signed one."""
    own: list[tuple[int, int]] = []
    if v.qf is not None:
        own = [(2, 1), (1, 1)] if v.qf.dynamic else [(v.qf.exp_format, v.qf.sign)]
    every = [(fe, 1) for fe in range(1, 8)]
    return [*own, *(t for t in every if t not in own)]


def scalar_word(value: float, v: Variant) -> int | None:
    """The CR word a MULT reads as ``value`` in ``v``'s mode, or ``None``.

    FP32: the IEEE float32 bits, when ``value`` is a float32. NX8: the MULT
    constant layout (``ipu_isa.cr_halves.constant``): an 8-bit element of the
    variant's type if one holds ``value`` exactly, else of another signed type
    (ADR 0007). ``CR0`` and ``CR1`` read 0.0 and 1.0 without a word.
    """
    if not math.isfinite(value):
        return None
    if v.qf is None:
        if abs(value) > _FP32_MAX or float(np.float32(value)) != value:
            return None
        return int(np.float32(value).view("<u4"))
    for fe, sign in _constant_types(v):
        found = _nx8_constants(fe, sign).get(value)
        if found is not None:
            code, scale = found
            return cr_halves.constant(code, scale, fe, sign)
    return None


def scalar_neighbours(value: float, v: Variant) -> tuple[float, ...]:
    """The values nearest ``value`` that :func:`scalar_word` encodes, below
    and above (one at the edge of the range)."""
    if v.qf is None:
        if abs(value) > _FP32_MAX:
            return (math.copysign(_FP32_MAX, value),)
        rounded = np.float32(value)
        if float(rounded) < value:
            return (float(rounded), float(np.nextafter(rounded, np.float32(np.inf))))
        return (float(np.nextafter(rounded, np.float32(-np.inf))), float(rounded))
    fe, sign = _constant_types(v)[0]
    values = sorted(_nx8_constants(fe, sign))
    below = [x for x in values if x < value][-1:]
    above = [x for x in values if x > value][:1]
    return (*below, *above)
