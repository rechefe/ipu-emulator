"""Transformer matmul 576×192 harness (Layer 4 QKV, 64 tokens).

Computes C[j, t] = sum_k W[j, k] * D[k, t]  for all j in [0, 576), t in [0, 64).

  D: channel-major [192, 64] input  — K channels × 64 tokens
  W: output-major  [576, 192] weights — N_OUT rows × K cols, stored verbatim (no transpose)
  C: channel-major [576, 64] output  — N_OUT channels × 64 tokens (FP32 accumulators)

Single token group: 64 tokens fit in one 128-lane SIMD vector, so each output
channel needs only one accumulate-and-store pass.

Usage::

    from ipu_apps.matmul_576x192_x128 import MatMul576x192x128App

    app = MatMul576x192x128App(
        inst_path="matmul_576x192_x128.bin",
        input_path="input.bin",
        weights_path="weights.bin",
        output_path="output.bin",
        dtype="INT8",
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Dimensions -------------------------------------------------------------

K      = 192   # input channels
N_OUT  = 576   # output channels
N_TOK  = 64    # tokens (single group, padded to 128 in XMEM)
N_LANE = 128   # SIMD lanes / padded tokens per channel

# -- Memory map -------------------------------------------------------------

DATA_BASE    = 0x00000   # D: K × 128 bytes            =  24,576 B
WEIGHTS_BASE = 0x10000   # W: N_OUT × 256 bytes        = 147,456 B
OUTPUT_BASE  = 0x40000   # C: N_OUT × 256 bytes packed = 147,456 B

W_STRIDE         = 256       # bytes per output channel in XMEM (ceil(K/128)*128)
DATA_STRIDE      = N_LANE     # bytes per channel in XMEM (128, padded)
OUTPUT_ROW_BYTES = N_TOK * 4  # bytes per output channel (64 tokens × 4) = 256

# -- Dtype helper -----------------------------------------------------------

_DTYPE_MAP = {
    "INT8":   DType.INT8,
    "int8":   DType.INT8,
    "E4":     DType.E4,
    "fp8_e4": DType.E4,
    "E5":     DType.E5,
    "fp8_e5": DType.E5,
}


def parse_dtype(dtype_str: str) -> DType:
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(f"Invalid dtype '{dtype_str}'. Supported: INT8, E4, E5")
    return dt


def _load_data(state: "IpuState", data_path: str | Path) -> None:
    """Load channel-major input into XMEM, padding each channel to 128 bytes.

    File layout: K channels × N_TOK bytes (contiguous, D[k][tok] at k*N_TOK + tok).
    XMEM layout:  channel k at DATA_BASE + k*128 (N_TOK valid bytes + zero pad).
    """
    raw = Path(data_path).read_bytes()
    for k in range(K):
        row = raw[k * N_TOK : k * N_TOK + N_TOK]
        padded = bytearray(N_LANE)
        padded[:N_TOK] = row
        state.xmem.write_address(DATA_BASE + k * DATA_STRIDE, padded)


def _load_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Load output-major weights into XMEM (no transposition).

    File layout: W[j][k] at byte j*K + k  (N_OUT rows × K cols).
    XMEM layout per output channel j (two 128-byte chunks):
      WEIGHTS_BASE + j*W_STRIDE        : W[j, 0..127]
      WEIGHTS_BASE + j*W_STRIDE + 128  : W[j, 128..191] + 64 zero bytes
    """
    raw = Path(weights_path).read_bytes()
    for j in range(N_OUT):
        row = raw[j * K : j * K + K]
        state.xmem.write_address(WEIGHTS_BASE + j * W_STRIDE, bytearray(row[:128]))
        tail = bytearray(128)
        tail[:K - 128] = row[128:K]
        state.xmem.write_address(WEIGHTS_BASE + j * W_STRIDE + 128, tail)


class MatMul576x192x128App(IpuApp):
    """576×192 transformer matmul application harness (Layer 4 QKV)."""

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        state.dtype = self.dtype
        _load_data(state, self.input_path)
        _load_weights(state, self.weights_path)
        # CR1 (≡1) is a read-only hardwired constant on the new architecture —
        # writes are silently dropped. WEIGHTS_BASE is moved to CR9 (free).
        # cr0=DATA_BASE is 0x0 (harmless no-op); cr2/cr3 are writable and stay.
        state.regfile.set_cr(0, DATA_BASE)
        state.regfile.set_cr(9, WEIGHTS_BASE)
        state.regfile.set_cr(2, WEIGHTS_BASE + 128)
        state.regfile.set_cr(3, OUTPUT_BASE)
        state.regfile.set_cr(6, -DATA_STRIDE)                  # data startup: -128
        state.regfile.set_cr(8, -1)                            # per-chunk fixed_idx startup
        state.regfile.set_lr(0, 0)                             # r_cyclic write-index 0
        state.regfile.set_lr(2, DATA_STRIDE)                   # data stride (128)
        state.regfile.set_lr(3, OUTPUT_ROW_BYTES)              # output stride (256, packed)
        state.regfile.set_lr(6, 126)                           # chunk0 bound: width=128
        state.regfile.set_lr(7, 0)                             # output pointer
        state.regfile.set_lr(8, 0)                             # weight byte offset
        state.regfile.set_lr(9, 0)                             # j counter
        state.regfile.set_lr(10, N_OUT)                        # j-loop limit (576)
        state.regfile.set_lr(11, (K - 128) - 2)               # chunk1 bound: width=64 → 62
        state.regfile.set_lr(12, W_STRIDE)                     # weight stride per j (256)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, OUTPUT_ROW_BYTES, N_OUT,
            )
