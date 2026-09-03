"""Universal convolution apps — standard, depthwise, and pointwise.

Each sub-package contains a single runtime-parameterized assembly binary
and Python harness that replaces all specialized per-configuration apps.

Shared helpers (input packing, output dumping) live here so sub-packages can
import them without duplicating code.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipu_emu.emulator import dump_xmem_to_binary
from ipu_emu.ipu_config import DEFAULT_VALID_ELEMENTS

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Shared constants (derived from IPU architecture parameters) -------------

# One output chunk holds DEFAULT_VALID_ELEMENTS lanes at 1 byte/lane (INT8/FP8).
CHUNK_BYTES = DEFAULT_VALID_ELEMENTS                 # 128
INT32_ELEMENT_BYTES = 4
ACC_CHUNK_BYTES = CHUNK_BYTES * INT32_ELEMENT_BYTES  # 512 (INT32 accumulator)
# Mask register: one 128-bit slot per CHUNK_BYTES lanes, packed 8 lanes/byte.
MASK_SLOT_BYTES = CHUNK_BYTES // 8                   # 16

# -- Shared chunk-interleaved packing (channel-per-128-element-chunk layout) -
#
# Every FP32 wide-vector app in this package uses the same on-device layout
# for a [channels, rows, cols] tensor: rows_per_chunk = CHUNK_ELEMENTS // cols
# spatial rows share one 128-element (512-byte FP32) chunk, chunks are
# grouped by channel. This is strictly internal plumbing -- see
# docs/content/adding-applications.md's "output file must have the same
# layout as the input file" rule -- so these helpers pack a caller's raw
# FP32 tensor on the way IN and unpack it on the way OUT; no app's
# input_path/output_path files are ever in this chunked format.

CHUNK_ELEMENTS = CHUNK_BYTES  # 128 elements/chunk (mode-blind: 512 B in FP32)


def pack_input_chunked(input_chw, cols: int) -> bytes:
    """Pack ``[channels, rows, cols]`` float32 into the chunk-interleaved
    layout every conv/depthwise/pointwise app in this package uses
    internally.

    ``cols`` must divide ``CHUNK_ELEMENTS`` (128). Offset formula (in
    ELEMENTS, not bytes): chunk = r // rows_per_chunk; local_row = r %
    rows_per_chunk; offset = (chunk*channels + ch)*128 + local_row*cols + c.
    """
    import numpy as np

    channels, rows, w = input_chw.shape
    if w != cols:
        raise ValueError(f"input_chw width {w} does not match cols {cols}")
    if CHUNK_ELEMENTS % cols != 0:
        raise ValueError(
            f"cols must divide {CHUNK_ELEMENTS} (a spatial row must tile one "
            f"chunk without straddling its edge), got {cols}"
        )
    rows_per_chunk = CHUNK_ELEMENTS // cols
    num_chunks = (rows + rows_per_chunk - 1) // rows_per_chunk
    packed = np.zeros(num_chunks * channels * CHUNK_ELEMENTS, dtype=np.float32)
    for ch in range(channels):
        for r in range(rows):
            chunk = r // rows_per_chunk
            local_row = r % rows_per_chunk
            row_off = (chunk * channels + ch) * CHUNK_ELEMENTS + local_row * cols
            packed[row_off:row_off + cols] = input_chw[ch, r, :].astype(np.float32)
    return packed.tobytes()


def unpack_output_chunked(raw: bytes, out_channels: int, out_rows: int, cols: int):
    """Inverse of :func:`pack_input_chunked`. Returns ``[out_channels,
    out_rows, cols]`` float32."""
    import numpy as np

    rows_per_chunk = CHUNK_ELEMENTS // cols
    out = np.zeros((out_channels, out_rows, cols), dtype=np.float32)
    arr = np.frombuffer(raw, dtype=np.float32)
    for ch in range(out_channels):
        for r in range(out_rows):
            chunk = r // rows_per_chunk
            local_row = r % rows_per_chunk
            off = (chunk * out_channels + ch) * CHUNK_ELEMENTS + local_row * cols
            out[ch, r, :] = arr[off:off + cols]
    return out


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


# -- Dynamic XMEM region layout ----------------------------------------------

# Total XMEM available to an app, in bytes. Narrow mode caps XMEM at 16384
# rows of 128 B; wide-vector debug mode uses 4 B/element but the *row* count
# is what the ISA addresses, so the row budget below is mode-independent and
# the byte figure here is the narrow-mode equivalent used for host-side
# planning.
XMEM_ROWS = 16384
XMEM_BYTES = XMEM_ROWS * CHUNK_BYTES  # 2 MiB


class XmemOverflow(ValueError):
    """Raised when an app's regions cannot fit inside XMEM.

    Carries the per-region breakdown so callers can see *which* region is
    responsible rather than only that the total is too large.
    """


def allocate_regions(regions, *, xmem_bytes: int = XMEM_BYTES) -> dict:
    """Pack named byte-sized regions into non-overlapping, chunk-aligned bases.

    ``regions`` is a sequence of ``(name, size_bytes)`` pairs, laid out in the
    order given starting at 0. Each base is rounded up to a ``CHUNK_BYTES``
    boundary, because every XMEM operand in the row-addressed ISA is a ROW
    number -- a region starting mid-chunk is not addressable.

    Returns ``{name: base_byte_address}``.

    This replaces per-app hardcoded ``*_BASE_ADDR`` constants, which reserve
    fixed gaps sized for a guessed worst case. Those gaps are silently wrong
    in both directions: too small for large configurations (a region overruns
    its successor and corrupts it with no error -- e.g. a pointwise kernel of
    out_channels * ceil(in_ch/128) * 128 bytes exceeds a 64 KiB gap once
    out_channels > 128 at in_channels >= 512), and needlessly large for small
    ones. Sizing from the actual configuration fixes both.

    Raises :class:`XmemOverflow` if the regions do not fit.
    """
    bases: dict = {}
    cursor = 0
    for name, size in regions:
        if size < 0:
            raise ValueError(f"region {name!r} has negative size {size}")
        bases[name] = cursor
        # Advance past this region, then align the next base to a chunk.
        cursor += size
        cursor = ((cursor + CHUNK_BYTES - 1) // CHUNK_BYTES) * CHUNK_BYTES
    if cursor > xmem_bytes:
        detail = ", ".join(f"{n}={sz} B @ {bases[n]:#x}" for n, sz in regions)
        raise XmemOverflow(
            f"regions need {cursor} bytes ({cursor / 1048576:.2f} MiB) but XMEM "
            f"holds {xmem_bytes} ({xmem_bytes / 1048576:.2f} MiB): {detail}"
        )
    return bases
