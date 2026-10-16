"""Element values as a format holds them: its range, and rounding into it.

One place for the harness (the manifest's ``range`` check, case inputs). The
element codec itself moves to ``ipu_isa.formats`` in E04-13; until then FP8
goes through the emulator's public encode and decode, which round to nearest
and saturate.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import numpy.typing as npt

from ipu_emu.ipu_math import DType, fp8_bytes_to_fp32, fp32_to_fp8_bytes
from ipu_isa.formats import FORMATS, ElementFormat

Array = npt.NDArray[Any]

_FP32_MAX = float(np.finfo(np.float32).max)


@functools.cache
def limits(fmt: ElementFormat) -> tuple[float, float]:
    """The lowest and highest finite value ``fmt`` holds."""
    info = FORMATS[fmt]
    if not info.is_float:
        half = 1 << (info.element_bits - 1)
        return float(-half), float(half - 1)
    if fmt is ElementFormat.FP32:
        return -_FP32_MAX, _FP32_MAX
    decoded = fp8_bytes_to_fp32(bytes(range(256)), DType(info.exp_bits))
    top = float(np.max(decoded[np.isfinite(decoded)]))
    return -top, top


def held(values: Array, fmt: ElementFormat) -> Array:
    """Finite ``values`` as ``fmt`` holds them: rounded to nearest, saturated to its
    range; an integer array for an integer format, else float32."""
    info = FORMATS[fmt]
    lo, hi = limits(fmt)
    clipped = np.clip(np.asarray(values, dtype=np.float64), lo, hi)
    if not info.is_float:
        return np.rint(clipped).astype(np.int8 if info.element_bits == 8 else np.int32)
    as_fp32 = clipped.astype(np.float32)
    if fmt is ElementFormat.FP32:
        return as_fp32
    dtype = DType(info.exp_bits)
    flat: Array = fp8_bytes_to_fp32(fp32_to_fp8_bytes(as_fp32.ravel(), dtype), dtype)
    return np.reshape(flat, as_fp32.shape)
