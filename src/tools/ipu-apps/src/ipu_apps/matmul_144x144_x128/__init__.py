"""Transformer matmul 144×144 harness.

Computes C[j, t] = sum_k W[j, k] * D[k, t]  for all j in [0, 144), t in [0, 256).

Usage::

    from ipu_apps.matmul_144x144_x128 import MatMul144x144x128App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

K     = 144
N_OUT = 144
N_TG  = 2
N_TOK = 128

# Region bases are DERIVED, not hardcoded: every region's byte size scales with
# the element width, so a map that fits in narrow mode (1 B/element) overflows in
# wide-vector debug mode (4 B/element). Laying the regions out by their actual
# row counts keeps them disjoint at any element width.
#
# The old fixed map (D=0x00000, W=0x10000, C=0x20000) worked only in narrow mode:
# in wide mode D alone needs 288 rows x 512 B = 144 KB, so it ran straight through
# WEIGHTS_BASE at 0x10000 and the weight staging overwrote the middle of D. The
# kernel then read zeros for every k >= 64, silently dropping most of the
# contraction.
DATA_BASE = 0x00000

# XMEM .asm operands are ROW numbers, not byte addresses -- see issue #179.
# A row is LANES *elements*, so every row COUNT below is element-width
# independent and therefore identical in narrow and wide-vector debug mode.
# Only byte quantities scale with the element width, and those are computed
# from `elem` at staging time rather than baked in here.
#
# The *_BASE constants stay byte addresses: they only drive this harness's
# direct xmem.write_address/read_address calls, which bypass row translation.
# The CR/LR registers in setup() feed the .asm's XMEM instructions instead, so
# they carry row numbers.
LANES = 128                              # elements per XMEM row (mode-independent)

W_STRIDE_ROWS    = -(-K // LANES)            # 2 rows per output channel (ceil)
DATA_STRIDE_ROWS = (N_TG * N_TOK) // LANES   # 2 rows per input channel

W_STRIDE         = W_STRIDE_ROWS * LANES     # elements per output channel (padded)
OUTPUT_ROW_BYTES = 512                       # r_acc store payload (bytes, both modes)


def _row_bytes(elem: int) -> int:
    """Bytes per XMEM row at the given element width: 128 narrow, 512 wide."""
    return LANES * elem


def _output_stride_rows(elem: int) -> int:
    """Rows spanned by one accumulator store.

    r_acc is 512 B in BOTH modes (128 lanes x 32-bit accumulators) and the store
    writes all of it unconditionally -- unlike the data registers, its width does
    NOT scale with the element width. So its footprint in ROWS is mode-dependent:
    4 rows narrow (512/128) but only 1 row wide (512/512). This is the one place
    a row count does not cancel out.
    """
    return OUTPUT_ROW_BYTES // _row_bytes(elem)


DATA_ROWS = K * N_TG                     # one row per (k, tg)
WEIGHT_ROWS = N_OUT * W_STRIDE_ROWS      # W_STRIDE_ROWS rows per output channel


def _base_rows(elem: int) -> tuple[int, int, int]:
    """The three region bases as ROW numbers, laid out back to back.

    Row COUNTS per region are element-width independent except for the output,
    whose store payload is a fixed 512 B (see ``_output_stride_rows``). Packing
    by row count keeps D/W/C disjoint in both modes; the byte addresses that
    fall out differ per mode, which is exactly the point.
    """
    data_row = DATA_BASE // _row_bytes(elem)
    weights_row = data_row + DATA_ROWS
    output_row = weights_row + WEIGHT_ROWS
    return data_row, weights_row, output_row


def _base_bytes(elem: int) -> tuple[int, int, int]:
    """The three region bases as BYTE addresses (for direct xmem staging)."""
    rb = _row_bytes(elem)
    d, w, o = _base_rows(elem)
    return d * rb, w * rb, o * rb

_DTYPE_MAP = {
    "INT8":     DType.INT8,
    "int8":     DType.INT8,
    "E4": DType.E4,
    "fp8_e4": DType.E4,
    "E5": DType.E5,
    "fp8_e5": DType.E5,
}


def parse_dtype(dtype_str: str) -> DType:
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(f"Invalid dtype '{dtype_str}'. Supported: INT8, E4, E5")
    return dt


def _load_data(state: "IpuState", data_path: str | Path, elem: int) -> None:
    """Stage D. Channel-major and already contiguous, so a straight copy works
    at any element width; ``elem`` only affects how many bytes that is."""
    raw = Path(data_path).read_bytes()
    expected = K * N_TG * N_TOK * elem
    if len(raw) < expected:
        raise ValueError(f"{data_path}: expected >= {expected} B for elem={elem}, got {len(raw)}")
    state.xmem.write_address(DATA_BASE, bytearray(raw[:expected]))


def _load_weights(state: "IpuState", weights_path: str | Path, elem: int, weights_base: int) -> None:
    """Stage W, padding each output channel's K elements out to whole rows.

    ``elem`` is the per-element byte width (1 for INT8/FP8, 4 for wide FP32).
    Row size and the per-channel stride both scale with it, so the padding
    arithmetic is expressed in elements and multiplied through.
    """
    raw = Path(weights_path).read_bytes()
    row_elems = LANES                      # elements per XMEM row
    stride = W_STRIDE_ROWS * row_elems     # elements per output channel (padded)
    for j in range(N_OUT):
        row = raw[j * K * elem : (j * K + K) * elem]
        for chunk in range(W_STRIDE_ROWS):
            lo = chunk * row_elems
            hi = min(lo + row_elems, K)
            buf = bytearray(row_elems * elem)
            if hi > lo:
                buf[: (hi - lo) * elem] = row[lo * elem : hi * elem]
            state.xmem.write_address(
                weights_base + (j * stride + lo) * elem, buf
            )


class MatMul144x144x128App(IpuApp):
    """144×144 transformer matmul application harness."""

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        # Element width is a property of the STATE, not of the kernel: wide-vector
        # debug mode stores 4-byte FP32 elements, narrow mode 1-byte INT8/FP8.
        # The kernel body is identical either way -- quantization belongs at the
        # XMEM write boundary (ACTIVATE.QUANTIZE), not threaded through staging.
        wide = bool(getattr(state, "wide_vector_debug", False))
        elem = 4 if wide else 1
        if not wide:
            state.dtype = self.dtype

        data_base_b, weights_base_b, output_base_b = _base_bytes(elem)
        self._output_base_bytes = output_base_b

        _load_data(state, self.input_path, elem)
        _load_weights(state, self.weights_path, elem, weights_base_b)

        data_base_row, weights_base_row, output_base_row = _base_rows(elem)
        out_stride_rows = _output_stride_rows(elem)

        # CR1 (≡1) is a read-only hardwired constant on the new architecture —
        # writes are silently dropped. WEIGHTS_BASE is moved to CR9 (free).
        # cr0=DATA_BASE is 0x0 (harmless no-op, matches hardwired 0); cr2 is a
        # writable CR and stays. See MIGRATION_CHECKLIST.md Bug #2.
        state.regfile.set_cr(0, data_base_row)
        state.regfile.set_cr(9, weights_base_row)
        state.regfile.set_cr(2, weights_base_row + 1)           # W[j,128..143]: next row
        state.regfile.set_cr(3, output_base_row)                                    # tg=0 output
        state.regfile.set_cr(4, output_base_row + N_OUT * out_stride_rows)       # tg=1 output
        state.regfile.set_cr(5, -DATA_STRIDE_ROWS)              # tg=0 data startup (rows)
        state.regfile.set_cr(6, -(DATA_STRIDE_ROWS // N_TG))    # tg=1 data startup (rows)
        state.regfile.set_cr(7, -1)                             # k-loop1 fixed_idx startup
        state.regfile.set_cr(8, 127)                            # k-loop2 fixed_idx startup
        state.regfile.set_lr(0, 0)                              # r_cyclic write-index 0
        state.regfile.set_lr(2, DATA_STRIDE_ROWS)               # data stride (rows)
        state.regfile.set_lr(3, out_stride_rows)                # output stride (rows)
        state.regfile.set_lr(6, 126)                            # k-loop1 bound: first_index=0, width=128 → 0+128-2=126
        state.regfile.set_lr(7, 0)                              # output pointer
        state.regfile.set_lr(8, 0)                              # weight byte offset
        state.regfile.set_lr(9, 0)                              # j counter
        state.regfile.set_lr(10, N_OUT)                         # j-loop limit
        state.regfile.set_lr(11, 142)                           # k-loop2 bound: first_index=128, width=16 → 128+16-2=142
        state.regfile.set_lr(12, W_STRIDE_ROWS)                 # weight stride per j (rows)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                self._output_base_bytes, OUTPUT_ROW_BYTES, N_OUT * N_TG,
            )
