"""Element-format lane codecs: the numerics handlers are written against (E04-13, Q48).

One codec per element format decodes a row of lanes into a numpy array,
computes in the accumulator domain and encodes the result back with the
format's rounding and saturation. :func:`codec_for` picks the codec; it is the
only format switch in ``ipu_emu``. Every codec is bit-identical to the scalar
arithmetic it replaces (``ipu_math.ipu_mult``/``ipu_add``/``ipu_sub`` and the
``struct`` stores): ``test_numerics`` proves it exhaustively.

Imports numpy, the standard library and ``ipu_isa`` only.
"""

from ipu_emu.numerics.codecs import (
    Fp8Codec,
    Fp32Codec,
    IntCodec,
    LaneCodec,
    codec_for,
    wrap_int32,
)

__all__ = [
    "Fp8Codec",
    "Fp32Codec",
    "IntCodec",
    "LaneCodec",
    "codec_for",
    "wrap_int32",
]
