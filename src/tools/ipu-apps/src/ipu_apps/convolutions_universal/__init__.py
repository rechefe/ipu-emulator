"""Universal convolution apps — standard, depthwise, and pointwise.

Each sub-package contains a single runtime-parameterized assembly binary
and Python harness that replaces all specialized per-configuration apps.

Shared helpers (dtype parsing, mask building, input packing, output dumping)
live here so sub-packages can import them without duplicating code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import dump_xmem_to_binary

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Shared constants --------------------------------------------------------

CHUNK_BYTES = 128
ACC_CHUNK_BYTES = 512  # 128 lanes × 4 bytes (INT32 accumulator)

# -- Shared dtype parsing ----------------------------------------------------

_DTYPE_MAP = {
    "INT8": DType.INT8,
    "int8": DType.INT8,
    "FP8_E4M3": DType.E4,
    "fp8_e4m3": DType.E4,
    "FP8_E5M2": DType.E5,
    "fp8_e5m2": DType.E5,
}


def parse_dtype(dtype: Union[str, DType]) -> DType:
    """Parse a dtype string (or pass through a DType) into a :class:`DType`."""
    if isinstance(dtype, DType):
        return dtype
    dt = _DTYPE_MAP.get(dtype)
    if dt is None:
        raise ValueError(
            f"Invalid dtype '{dtype}'. Supported: INT8, FP8_E4M3, FP8_E5M2"
        )
    return dt


# -- Shared mask builder (universal 3x3 conv / depthwise) --------------------

def build_border_mask_data(cols: int) -> bytes:
    """Build the 128-byte mask register data for a given spatial width.

    Only 3 mask slots are needed. Bottom-border handling is done by loading
    zeros into the S2 cyclic slot for the last chunk instead of using dedicated
    bottom-row masks.

    Layout (8 slots of 16 bytes = 128 bits each):
      slot 0: all zeros    -> no masking (kc=0)
      slot 1: left border  -> zero col 0 of each packed row (kc=-1)
      slot 2: right border -> zero last col of each packed row (kc=+1)
      slots 3-7: unused (zeros)
    """
    rows_per_chunk = 128 // cols
    mask = bytearray(128)

    def set_bit(slot: int, bit: int) -> None:
        byte_idx = slot * 16 + bit // 8
        mask[byte_idx] |= 1 << (bit % 8)

    for r in range(rows_per_chunk):
        set_bit(1, r * cols)
    for r in range(rows_per_chunk):
        set_bit(2, r * cols + cols - 1)

    return bytes(mask)


# -- Shared input packer (8x8 paired-channel layout) -------------------------

def pack_input_paired(
    input_raw: bytes,
    channels: int,
    spatial: int = 64,
) -> bytes:
    """Pack per-channel input into paired-chunk 128-byte layout.

    Input  : ``input_raw[ch * spatial + pos]`` (per-channel, ``spatial`` bytes
             each; default 64 covers 8x8 apps).
    Output : ``channels // 2`` chunks of 128 bytes each. Chunk ``j`` contains
             channel ``2j`` in bytes 0..spatial-1 and channel ``2j+1`` in
             bytes ``spatial..128``.

    The default ``spatial=64`` reproduces the byte output of the previous
    per-app ``_build_input_data`` copies in conv_8x8, depthwise_8x8,
    pointwise_8x8, and residual_add.
    """
    pairs = channels // 2
    packed = bytearray(pairs * 128)
    for j in range(pairs):
        dst = j * 128
        src_even = (2 * j) * spatial
        src_odd = (2 * j + 1) * spatial
        packed[dst:dst + spatial] = input_raw[src_even:src_even + spatial]
        packed[dst + spatial:dst + 128] = input_raw[src_odd:src_odd + spatial]
    return bytes(packed)


# -- Shared output dumper ----------------------------------------------------

def dump_outputs(
    state: "IpuState",
    path,
    base: int,
    chunk_bytes: int,
    count: int,
) -> None:
    """Dump ``count`` consecutive chunks of ``chunk_bytes`` from XMEM to file.

    Thin wrapper over :func:`ipu_emu.emulator.dump_xmem_to_binary`.
    Callers must guard against ``path is None`` before calling.
    """
    dump_xmem_to_binary(state, path, base, chunk_bytes, count)


__all__ = [
    "CHUNK_BYTES",
    "ACC_CHUNK_BYTES",
    "parse_dtype",
    "build_border_mask_data",
    "pack_input_paired",
    "dump_outputs",
]
