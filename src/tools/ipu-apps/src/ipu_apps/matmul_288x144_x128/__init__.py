"""Transformer matmul 288×144 harness (FFN linear 1, no activation).

Computes C[j, t] = sum_k W[j, k] * D[k, t]  for all j in [0, 288), t in [0, 256).

Usage::

    from ipu_apps.matmul_288x144_x128 import MatMul288x144x128App
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
N_OUT = 288
N_TG  = 2
N_TOK = 128

# Region bases are DERIVED, not hardcoded: every region's byte size scales with
# the element width, so a map that fits in narrow mode (1 B/element) overflows
# in wide-vector debug mode (4 B/element). Laying the regions out by their
# actual row counts keeps them disjoint at any element width.
#
# XMEM .asm operands are ROW numbers, not byte addresses (issue #179). A row is
# LANES *elements*, so row COUNTS are element-width independent -- except the
# output store, whose payload is a fixed 512 B (see _output_stride_rows).
LANES = 128                                  # elements per XMEM row

W_STRIDE_ROWS    = -(-K // LANES)            # rows per output channel (ceil)
DATA_STRIDE_ROWS = (N_TG * N_TOK) // LANES   # rows per input channel
W_STRIDE         = W_STRIDE_ROWS * LANES     # elements per output channel (padded)
OUTPUT_ROW_BYTES = 512                       # r_acc store payload (bytes, both modes)

DATA_BASE   = 0x00000
DATA_ROWS   = K * N_TG
WEIGHT_ROWS = N_OUT * W_STRIDE_ROWS


def _row_bytes(elem: int) -> int:
    """Bytes per XMEM row at the given element width: 128 narrow, 512 wide."""
    return LANES * elem


def _output_stride_rows(elem: int) -> int:
    """Rows spanned by one accumulator store.

    r_acc is a fixed 512 B in BOTH modes and the store writes all of it, so its
    footprint in ROWS is mode-dependent: 4 narrow, 1 wide. The one row count
    that does not cancel out.
    """
    return OUTPUT_ROW_BYTES // _row_bytes(elem)


def _base_rows(elem: int) -> tuple[int, int, int]:
    """D/W/C bases as ROW numbers, packed back to back and disjoint in both modes."""
    data_row = DATA_BASE // _row_bytes(elem)
    weights_row = data_row + DATA_ROWS
    output_row = weights_row + WEIGHT_ROWS
    return data_row, weights_row, output_row


def _base_bytes(elem: int) -> tuple[int, int, int]:
    """D/W/C bases as BYTE addresses (for direct xmem staging)."""
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
    raw = Path(data_path).read_bytes()
    expected = K * N_TG * N_TOK * elem
    if len(raw) < expected:
        raise ValueError(f"{data_path}: expected >= {expected} B for elem={elem}, got {len(raw)}")
    state.xmem.write_address(DATA_BASE, bytearray(raw[:expected]))


def _load_weights(state: "IpuState", weights_path: str | Path, elem: int, weights_base: int) -> None:
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
            state.xmem.write_address(weights_base + (j * stride + lo) * elem, buf)


class MatMul288x144x128App(IpuApp):
    """288×144 transformer matmul application harness."""

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        # Element width is a property of the STATE, not the kernel: wide-vector
        # debug mode stores 4-byte FP32 elements, narrow mode 1-byte INT8/FP8.
        wide = bool(getattr(state, "wide_vector_debug", False))
        elem = 4 if wide else 1
        if not wide:
            state.dtype = self.dtype

        data_base_b, weights_base_b, output_base_b = _base_bytes(elem)
        self._output_base_bytes = output_base_b

        _load_data(state, self.input_path, elem)
        _load_weights(state, self.weights_path, elem, weights_base_b)

        DATA_BASE_ROW, WEIGHTS_BASE_ROW, OUTPUT_BASE_ROW = _base_rows(elem)
        OUTPUT_STRIDE_ROWS = _output_stride_rows(elem)
        # CR1 (≡1) is a read-only hardwired constant on the new architecture —
        # writes are silently dropped. WEIGHTS_BASE is moved to CR9 (free).
        # cr0=DATA_BASE is 0x0 (harmless no-op); cr2 is writable and stays.
        # See MIGRATION_CHECKLIST.md Bug #2.
        state.regfile.set_cr(0, DATA_BASE_ROW)
        state.regfile.set_cr(9, WEIGHTS_BASE_ROW)
        state.regfile.set_cr(2, WEIGHTS_BASE_ROW + 1)           # W[j,128..143]: next row
        state.regfile.set_cr(3, OUTPUT_BASE_ROW)                                    # tg=0 output
        state.regfile.set_cr(4, OUTPUT_BASE_ROW + N_OUT * OUTPUT_STRIDE_ROWS)       # tg=1 output
        state.regfile.set_cr(5, -DATA_STRIDE_ROWS)              # tg=0 data startup (rows)
        state.regfile.set_cr(6, -(DATA_STRIDE_ROWS // N_TG))    # tg=1 data startup (rows)
        state.regfile.set_cr(7, -1)                             # k-loop1 fixed_idx startup
        state.regfile.set_cr(8, 127)                            # k-loop2 fixed_idx startup
        state.regfile.set_lr(0, 0)                              # r_cyclic write-index 0
        state.regfile.set_lr(2, DATA_STRIDE_ROWS)               # data stride (rows)
        state.regfile.set_lr(3, OUTPUT_STRIDE_ROWS)             # output stride (rows)
        state.regfile.set_lr(6, 126)                            # k-loop1 bound: first_index=0, width=128 → 126
        state.regfile.set_lr(7, 0)                              # output pointer
        state.regfile.set_lr(8, 0)                              # weight byte offset
        state.regfile.set_lr(9, 0)                              # j counter
        state.regfile.set_lr(10, N_OUT)                         # j-loop limit
        state.regfile.set_lr(11, 142)                           # k-loop2 bound: first_index=128, width=16 → 142
        state.regfile.set_lr(12, W_STRIDE_ROWS)                 # weight stride per j (rows)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                self._output_base_bytes, OUTPUT_ROW_BYTES, N_OUT * N_TG,
            )
