#!/usr/bin/env python3
"""Generate the golden parity file for the C99 port of ipu_math / activations.

Imports the *Python reference implementations* directly (no re-derivation) and
dumps every case as a line of text in which each double is written as the exact
uint64 bit pattern of its IEEE-754 binary64 encoding.  Comparison in
``test_math_parity.c`` is therefore bit-exact, never approximate.

Usage:
    python3 gen_golden.py <output-path>

Record formats (whitespace separated, one per line):
    FP8DEC  <byte>      <exp_bits> <result:%016x>
    FP8ENC  <in:%016x>  <exp_bits> <result_byte>
    ONEBYTE <dtype>     <result_byte>
    MULT    <a_byte>    <b_byte>   <dtype> <result:%016x>
    ADD     <a:%016x>   <b:%016x>  <dtype> <result:%016x>
    SUB     <a:%016x>   <b:%016x>  <dtype> <result:%016x>
    WRAP32  <in:%016x>  <result_int32>
    ACT     <fn_id>     <x:%016x>  <alpha:%016x> <result:%016x>
"""

from __future__ import annotations

import importlib.util
import math
import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TOOLS = _HERE.parents[3]  # .../src/tools
_EMU_SRC = _TOOLS / "ipu-emu-py" / "src"
_COMMON_SRC = _TOOLS / "ipu-common" / "src"
sys.path.insert(0, str(_EMU_SRC))
sys.path.insert(0, str(_COMMON_SRC))


def _load(mod_name: str, path: Path):
    """Load a single reference module by file path.

    Loading the file directly (rather than ``from ipu_emu import ipu_math``)
    keeps the generator independent of the emulator package's ``__init__``,
    which drags in the assembler.  Both modules are leaves: ipu_math.py needs
    only math/enum/numpy and activations.py only math.
    """
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("ipu_math_ref", _EMU_SRC / "ipu_emu" / "ipu_math.py")
A = _load("activations_ref", _COMMON_SRC / "ipu_common" / "activations.py")


def bits(x: float) -> str:
    """Exact uint64 bit pattern of a Python float."""
    return "%016x" % struct.unpack("<Q", struct.pack("<d", float(x)))[0]


EXP_BITS = range(1, 8)


# --------------------------------------------------------------------------
# FP8 decode: all 256 byte values x exp_bits 1..7  (1792 cases)
# --------------------------------------------------------------------------
def gen_decode(out: list[str]) -> None:
    for eb in EXP_BITS:
        for b in range(256):
            out.append("FP8DEC %d %d %s" % (b, eb, bits(M._fp8_to_float32_scalar(b, eb))))


# --------------------------------------------------------------------------
# FP8 encode input set
# --------------------------------------------------------------------------
def encode_inputs(eb: int) -> list[float]:
    man_bits = 7 - eb
    bias = (1 << (eb - 1)) - 1
    vals: list[float] = []

    # Special values first.
    vals += [0.0, -0.0, float("inf"), float("-inf"), float("nan")]

    # Every decodable FP8 value round-tripped (both signs, NaN included).
    decoded = [M._fp8_to_float32_scalar(b, eb) for b in range(256)]
    vals += decoded

    # Max finite, and values just above it (overflow-clamp branch).
    max_byte = M._fp8_max_finite(eb, man_bits)
    max_fin = M._fp8_to_float32_scalar(max_byte, eb)
    for scale in (1.0, 1.0000001, 1.5, 2.0, 1e3, 1e30, 1e300):
        vals += [max_fin * scale, -max_fin * scale]
    vals += [math.nextafter(max_fin, math.inf), math.nextafter(max_fin, 0.0)]

    # Exact ties between adjacent representable magnitudes -> banker's rounding.
    finite = sorted({v for v in decoded if math.isfinite(v) and v >= 0.0})
    for lo, hi in zip(finite, finite[1:]):
        mid = (lo + hi) / 2.0
        vals += [mid, -mid]
        vals += [math.nextafter(mid, 0.0), math.nextafter(mid, math.inf)]
    # One tie above the largest finite value too.
    vals += [max_fin * 1.5, -(max_fin * 1.5)]

    # Subnormal-branch ties written directly in the encoder's own units:
    #   man_int = val * 2^(man_bits+bias-1)
    step = 2.0 ** -(man_bits + bias - 1)
    for k in range(0, (1 << man_bits) + 3):
        for frac in (0.0, 0.25, 0.5, 0.75):
            v = (k + frac) * step
            vals += [v, -v]

    # Tiny subnormals / underflow.
    vals += [5e-324, -5e-324, 1e-320, -1e-320, 1e-45, -1e-45, step / 4.0, -step / 4.0]

    # Dense sweep across the dynamic range of this format.
    smallest = min((abs(v) for v in decoded if math.isfinite(v) and v != 0.0), default=1.0)
    lo_e = math.floor(math.log2(smallest)) - 3
    hi_e = math.ceil(math.log2(max_fin)) + 3
    n = 0
    e = float(lo_e)
    while e <= hi_e:
        v = 2.0 ** e
        vals += [v, -v]
        e += 1.0 / 16.0
        n += 1
        if n > 20000:
            break

    # Plain linear sweep near unity and small integers.
    for i in range(-2000, 2001):
        v = i / 128.0
        vals.append(v)

    return vals


def gen_encode(out: list[str]) -> None:
    for eb in EXP_BITS:
        for v in encode_inputs(eb):
            out.append("FP8ENC %s %d %d" % (bits(v), eb, M._float32_to_fp8_scalar(v, eb)))


# --------------------------------------------------------------------------
# dtype_one_byte / mult / add / sub / wrap
# --------------------------------------------------------------------------
def gen_misc(out: list[str]) -> None:
    for d in range(0, 8):
        out.append("ONEBYTE %d %d" % (d, M.dtype_one_byte(M.DType(d))))

    for d in range(0, 8):
        for a in range(0, 256, 1 if d == 0 else 1):
            for b in (0, 1, 2, 3, 5, 17, 63, 64, 100, 127, 128, 129, 200, 254, 255):
                out.append("MULT %d %d %d %s" % (a, b, d, bits(M.ipu_mult(a, b, d))))

    int_words = [0, 1, -1, 2, -2, 7, -7, 1000, -1000, 65535, -65536,
                 2147483647, -2147483648, 123456789, -123456789,
                 2000000000, -2000000000, 1073741824, -1073741824]
    for a in int_words:
        for b in int_words:
            out.append("ADD %s %s 0 %s" % (bits(a), bits(b), bits(M.ipu_add(a, b, 0))))
            out.append("SUB %s %s 0 %s" % (bits(a), bits(b), bits(M.ipu_sub(a, b, 0))))

    fl_words = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1e-300, 1e300, -1e300,
                math.pi, -math.e, float("inf"), float("-inf"), float("nan"),
                5e-324, 1.7976931348623157e308]
    for d in range(1, 8):
        for a in fl_words:
            for b in fl_words:
                out.append("ADD %s %s %d %s" % (bits(a), bits(b), d, bits(M.ipu_add(a, b, d))))
                out.append("SUB %s %s %d %s" % (bits(a), bits(b), d, bits(M.ipu_sub(a, b, d))))

    wrap_vals = [0, 1, -1, 2147483647, 2147483648, -2147483648, -2147483649,
                 4294967295, 4294967296, 4294967297, -4294967296, 8589934592,
                 123456789012, -123456789012, 1e15, -1e15]
    for v in wrap_vals:
        u = int(v) & 0xFFFFFFFF
        signed = u - 0x100000000 if u >= 0x80000000 else u
        out.append("WRAP32 %s %d" % (bits(float(v)), signed))


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------
def activation_inputs() -> list[float]:
    xs: list[float] = []
    # Dense linear grid over a wide span (covers all softplus / sigmoid branches).
    for i in range(-6000, 6001):
        xs.append(i / 100.0)
    # Fine grid near the interesting region.
    for i in range(-800, 801):
        xs.append(i / 1000.0)
    # Branch boundaries.
    xs += [0.0, -0.0, 6.0, -6.0, 20.0, -20.0, 20.000000001, -20.000000001,
           19.999999999, -19.999999999]
    # Log-spaced magnitudes, both signs.
    for e in range(-30, 31):
        v = 2.0 ** e
        xs += [v, -v]
        xs += [v * 1.5, -v * 1.5]
    # Extremes and specials (exp2 overflow cases are filtered out below).
    xs += [1e-300, -1e-300, 5e-324, -5e-324, 1e300, -1e300,
           float("inf"), float("-inf"), float("nan"),
           1.7976931348623157e308, -1.7976931348623157e308]
    return xs


def gen_activations(out: list[str]) -> int:
    skipped = 0
    xs = activation_inputs()
    for alpha in (1.0, 0.5):
        for fn in range(A.ACTIVATION_COUNT):
            for x in xs:
                try:
                    r = A.apply_activation(fn, x, elu_alpha=alpha)
                except (OverflowError, ValueError, ZeroDivisionError):
                    # Python's math.exp raises OverflowError where C's exp()
                    # returns +inf; those inputs are outside the parity domain.
                    skipped += 1
                    continue
                out.append("ACT %d %s %s %s" % (fn, bits(x), bits(alpha), bits(r)))
    # Out-of-range ids fall through to identity.
    for fn in (12, 13, 99, -1, -12345):
        for x in (0.0, -1.5, 3.25, float("nan")):
            out.append("ACT %d %s %s %s"
                       % (fn, bits(x), bits(1.0),
                          bits(A.apply_activation(fn, x, elu_alpha=1.0))))
    return skipped


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: gen_golden.py <output-path>", file=sys.stderr)
        return 2
    out: list[str] = []
    gen_decode(out)
    gen_encode(out)
    gen_misc(out)
    skipped = gen_activations(out)

    path = Path(sys.argv[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("# ipu golden parity vectors\n")
        fh.write("COUNT %d\n" % len(out))
        fh.write("\n".join(out))
        fh.write("\n")
    print("wrote %d cases to %s (%d activation inputs skipped: Python raised)"
          % (len(out), path, skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
