"""Row-softmax for N < 128 elements/row, packed P rows per 128-lane chunk.

Logical rows of ``N`` elements are packed into 128-lane physical chunks. The
partition size ``ps`` is the next power of two >= N (clamped to [16, 128]), so
``P = 128 / ps`` rows share one chunk:

    N in 65..128 -> ps=128, P=1     N in 17..32 -> ps=32,  P=4
    N in 33..64  -> ps=64,  P=2     N in  1..16 -> ps=16,  P=8

Row p occupies lanes ``p*ps .. p*ps + N-1`` of its chunk; lanes ``N..ps-1`` are
padding. ``CR15.valid_elements = N`` masks every AGG/ACTIVATE so only the first
N lanes of each partition contribute (max/sum ignore the padding tail).

Reduction trick (probe-validated): to reduce partition p, ``MULT.RC.VV`` reads
r_cyclic at byte offset ``p*ps*4`` so partition p lands in mult_res lanes
0..N-1, then a masked ``AGG`` reduces exactly those into r_acc slot = row index.

Layout:
  * input / output : PACKED (P rows/chunk) -- input as given, output must match.
  * numerators     : UNPACKED (one 512B chunk per logical row, lanes 0..N-1) --
                     intermediate, free to be convenient.
  * maxvec/rvec    : 128-lane scalar vectors, slot per logical row.

Only Pass 4 re-packs: it builds a full chunk in r_acc (row p at lanes p*ps),
then one ACTIVATE+store drains it packed.

The base softmax_rows app is the ps=128/P=1 special case; this app handles the
P>1 packed regimes. Same base-2 / max-subtraction math (see softmax_rows).

Usage::

    from ipu_apps.softmax.softmax_rows_partial import SoftmaxRowsPartialApp
    app = SoftmaxRowsPartialApp(
        inst_path="softmax_rows_partial.bin",
        input_path="logits.bin",   # rows * N float32, row-major
        output_path="probs.bin",
        n=32, rows=100,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import dump_xmem_to_binary
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    pass

# -- Constants --------------------------------------------------------------

LANES = 128                  # lanes per physical chunk
CHUNK_BYTES = LANES * 4      # 512 bytes per FP32 chunk

# XMEM byte-address map. INPUT_BASE non-zero; literal 0 lives in CR0 (read-only).
INPUT_BASE_ADDR = 0x10000    # packed input chunks
NUM_BASE_ADDR = 0x20000      # unpacked numerator chunks (one per logical row)
OUTPUT_BASE_ADDR = 0x30000   # packed output chunks
CVEC_ADDR = 0x40000          # C_VEC = log2(e), resident
MAXVEC_ADDR = 0x40400        # per-row max (1 chunk)
RVEC_ADDR = 0x40600          # per-row 1/sum (1 chunk)

NEG_ONE_BYTE = 0xFF          # 0xFF -> signed -1 -> -1.0 in wide FP32
LOG2E = math.log2(math.e)


def partition_size(n: int) -> int:
    """Next power of two >= n, clamped to [16, 128]."""
    if not 1 <= n <= 128:
        raise ValueError(f"N must be in 1..128; got {n}")
    ps = 16
    while ps < n:
        ps *= 2
    return ps


class SoftmaxRowsPartialApp(IpuApp):
    """Packed row-softmax for N elements/row (N <= 128), P = 128/ps rows/chunk.

    Args:
        inst_path:   Assembled instruction binary.
        input_path:  Input logits, ``rows * N`` float32, row-major (one row's N
                     elements contiguous).
        output_path: Optional output path (packed, same layout as input).
        n:           Elements per logical row (1..128).
        rows:        Number of logical rows (any count; zero-padded up to a
                     multiple of P internally).
    """

    def __init__(self, *, n: int, rows: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.n = int(n)
        self.rows = int(rows)
        self.ps = partition_size(self.n)
        self.parts_per_chunk = LANES // self.ps          # P
        # Pad logical rows up to a multiple of P so the last chunk is full.
        rem = self.rows % self.parts_per_chunk
        self.padded_rows = self.rows + (self.parts_per_chunk - rem) % self.parts_per_chunk
        self.num_chunks = self.padded_rows // self.parts_per_chunk
        # KNOWN BUG (see STATUS.md): P=8 (N<=16) with >=3 chunks produces wrong
        # Pass-4 output. Guard until the multi-chunk P=8 repack is fixed.
        if self.parts_per_chunk == 8 and self.num_chunks >= 3:
            raise NotImplementedError(
                "softmax_rows_partial: N<=16 (P=8) with >=3 chunks is not yet "
                "correct (Pass-4 repack bug; see STATUS.md). Use <=16 rows for "
                f"N<=16, or split the batch. Got n={self.n}, rows={self.rows}."
            )

    # -- wide-vector FP32 state ---------------------------------------------

    @staticmethod
    def make_state() -> IpuState:
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )
        state.dtype = DType.INT8
        return state

    def _pack_input(self) -> bytes:
        """Read row-major (rows x N) float32, zero-pad to padded_rows, pack into
        chunks of P rows; row p at lanes p*ps..p*ps+N-1, padding lanes zero."""
        raw = self.input_path.read_bytes()
        flat = struct.unpack(f"<{self.rows * self.n}f", raw[: self.rows * self.n * 4])
        packed = bytearray(self.num_chunks * CHUNK_BYTES)
        for r in range(self.rows):  # padded rows stay zero
            chunk = r // self.parts_per_chunk
            p = r % self.parts_per_chunk
            base = chunk * CHUNK_BYTES + p * self.ps * 4
            row = flat[r * self.n:(r + 1) * self.n]
            struct.pack_into(f"<{self.n}f", packed, base, *row)
        return bytes(packed)

    def setup(self, state: "IpuState") -> None:
        state.xmem.write_address(INPUT_BASE_ADDR, self._pack_input())
        state.xmem.write_address(CVEC_ADDR, struct.pack("<128f", *([LOG2E] * LANES)))

        # CR15.valid_elements = N masks every AGG/ACTIVATE to the first N lanes.
        state.set_cr_dstructure(valid_elements=self.n)

        # CR0=0, CR1=1 are read-only hardware constants (reused as zero source /
        # 1.0 identity scalar). Writable CRs below.
        # CR1 == 1 (read-only) serves as the generic +1 increment, freeing a slot.
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, CVEC_ADDR)
        state.regfile.set_cr(4, NUM_BASE_ADDR)
        state.regfile.set_cr(5, MAXVEC_ADDR)
        state.regfile.set_cr(6, RVEC_ADDR)
        state.regfile.set_cr(7, CHUNK_BYTES)          # chunk stride (512)
        state.regfile.set_cr(8, LANES)                # 128: R1 byte-index base for maxvec select
        state.regfile.set_cr(9, self.padded_rows)     # logical-row loop bound
        state.regfile.set_cr(10, INPUT_BASE_ADDR)     # input base
        state.regfile.set_cr(11, NEG_ONE_BYTE)        # -1.0 subtract scalar
        state.regfile.set_cr(12, self.ps * 4)         # partition byte stride (ps*4)
        state.regfile.set_cr(13, self.num_chunks)     # chunk loop bound
        state.regfile.set_cr(14, self.parts_per_chunk)  # P (partitions per chunk)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # Dump only the chunks holding real (non-padding) rows.
            dump_xmem_to_binary(
                state, self.output_path, OUTPUT_BASE_ADDR, CHUNK_BYTES, self.num_chunks
            )

    def run(self, **kwargs):
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)
