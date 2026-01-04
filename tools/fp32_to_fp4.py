#!/usr/bin/env python3
"""Convert fp32 binary arrays to compact FP4 formats.

Assumptions:
- Input is a raw binary of little-endian float32 values.
- Output stores one FP4 value per byte (lower nibble carries the value; upper nibble is zero).
- FP4 layout used here is sign:1 | exp:2 | mantissa:1 (often called e2m1) with bias=1.
  This matches the bitfield in src/lib/fp/fp.h (man:1, exp:2, sign:1 in the low nibble).
- Values are assumed representable by FP8 E4M3 (range is small), but we saturate to FP4 range.

If you actually need a different FP4 variant (e.g., different bias or packing two values
per byte), adjust `quantize_fp4_e2m1` and `_pack_byte` accordingly.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np


FP4_EXP_BITS = 2
FP4_MAN_BITS = 1
FP4_SIGN_BITS = 1
FP4_BIAS = 1  # matches e2m1 convention: exp_field = exponent + 1


def quantize_fp4_e2m1(value: float) -> int:
    """Quantize a float to FP4 e2m1 (sign:1, exp:2, man:1) stored in low nibble.

    Rounding: nearest-ties-to-even on the mantissa bit.
    Saturation: clamp to largest finite FP4 value; NaN/inf also clamp.
    """

    if math.isnan(value) or math.isinf(value):
        value = math.copysign(float("inf"), value)

    if value == 0.0:
        return 0

    sign = 1 if value < 0 else 0
    x = abs(value)

    # Smallest normal exponent is -FP4_BIAS (exp_field=0 is treated as subnormal here -> we flush to zero)
    exp = math.floor(math.log2(x))
    exp_field = exp + FP4_BIAS

    # Handle overflow/underflow
    max_exp_field = (1 << FP4_EXP_BITS) - 1
    if exp_field >= max_exp_field:
        # Max finite: mantissa all ones
        exp_field = max_exp_field
        mant_field = (1 << FP4_MAN_BITS) - 1
    elif exp_field <= 0:
        # Too small -> flush to zero (no subnormals supported here)
        return 0
    else:
        # Normalized: compute mantissa fraction
        frac = x / (2 ** exp) - 1.0  # in [0, 1)
        scale = 1 << FP4_MAN_BITS
        mant = frac * scale
        mant_rounded = int(np.rint(mant))
        if mant_rounded == scale:
            # Rounding overflow bumps exponent
            mant_rounded = 0
            exp_field += 1
            if exp_field >= max_exp_field:
                exp_field = max_exp_field
                mant_rounded = scale - 1
        mant_field = mant_rounded

    encoded = (sign << 3) | (exp_field << FP4_MAN_BITS) | mant_field
    return encoded & 0x0F  # lower nibble


def convert_array(values: np.ndarray) -> np.ndarray:
    """Vectorized conversion of float32 array to uint8 FP4 bytes (one per value)."""
    out = np.empty(values.shape, dtype=np.uint8)
    it = np.nditer(values, flags=["multi_index"])
    while not it.finished:
        out[it.multi_index] = quantize_fp4_e2m1(float(it[0]))
        it.iternext()
    return out


def process_file(src: Path, dst: Path) -> None:
    data = np.fromfile(src, dtype=np.float32)
    if data.size == 0:
        raise ValueError(f"No data read from {src}")

    encoded = convert_array(data)
    encoded.tofile(dst)
    print(f"Wrote {encoded.size} FP4 values to {dst}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert fp32 binary to FP4 e2m1 binary")
    p.add_argument("src", type=Path, help="Input binary file of float32 values")
    p.add_argument("dst", type=Path, help="Output binary file (FP4, one value per byte)")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    process_file(args.src, args.dst)


if __name__ == "__main__":
    main()
