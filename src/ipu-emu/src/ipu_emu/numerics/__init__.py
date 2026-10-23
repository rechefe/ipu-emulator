"""Run-mode lane codecs: the numerics handlers are written against (E04-13, Q48, ADR 0007).

One codec per run mode decodes a row of lanes into a numpy array, computes in
the accumulator domain (FP32 in both modes) and stores the result with
float32 rounding. :func:`codec_for` picks the codec; it is the only mode
switch the slot handlers use for lane arithmetic. An NX8 element's value
depends on the instruction's QF and the row's metadata, so
:meth:`LaneCodec.decode` takes both (the FP32 codec ignores them).

Imports numpy, the standard library and ``ipu_isa`` only.
"""

from ipu_emu.numerics.codecs import (
    Fp32Codec,
    LaneCodec,
    Nx8Codec,
    codec_for,
    nx8_table,
)

__all__ = [
    "Fp32Codec",
    "LaneCodec",
    "Nx8Codec",
    "codec_for",
    "nx8_table",
]
