"""Transformer matmul 432×144 harness.

Computes C[j, t] = sum_k W[j, k] * D[k, t]  for all j in [0, 432), t in [0, 256).

  D: grouped channel-major [2, 144, 128] input  — K channels × 256 tokens (2 token groups × 128)
  W: output-major  [432, 144] weights — N_OUT rows × K cols, stored verbatim (no transpose)
  C: channel-major [432, 256] output  — N_OUT channels × 256 tokens (FP32 accumulators)

Usage::

    from ipu_apps.matmul_432x144_x128 import MatMul432x144x128App

    app = MatMul432x144x128App(
        inst_path="matmul_432x144_x128.bin",
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

K     = 144   # input channels
N_OUT = 432   # output channels
N_TG  = 2     # token groups
N_TOK = 128   # tokens per group

# -- Memory map -------------------------------------------------------------

DATA_BASE    = 0x00000   # D: N_TG × K × N_TOK bytes  = 36,864 B (grouped)
WEIGHTS_BASE = 0x10000   # W: N_OUT × 256 bytes        = 110,592 B  (padded to 2×128 per out_ch)
OUTPUT_BASE  = 0x30000   # C: N_OUT × N_TG × 512 bytes = 442,368 B

W_STRIDE        = 256    # bytes per output channel in XMEM (ceil(K/128)*128)
OUTPUT_ROW_BYTES = 512   # bytes per (out_ch, tg) row  = N_TOK × 4

# -- Dtype helper -----------------------------------------------------------

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


def _load_data(state: "IpuState", data_path: str | Path) -> None:
    """Load grouped channel-major input directly into XMEM.

    File layout: 2 tg blocks × K channels × 128 bytes each.
    tg=0 block at offset 0, tg=1 block at offset K*128.
    """
    raw = Path(data_path).read_bytes()
    state.xmem.write_address(DATA_BASE, bytearray(raw))


def _load_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Load output-major weights into XMEM (no transposition).

    File layout: W[j][k] at byte j*K + k  (N_OUT rows × K cols).
    XMEM layout per output channel j:
      WEIGHTS_BASE + j*W_STRIDE         : W[j, 0..127]  (128 bytes)
      WEIGHTS_BASE + j*W_STRIDE + 128   : W[j, 128..143] + 112 zero bytes
    """
    raw = Path(weights_path).read_bytes()
    for j in range(N_OUT):
        row = raw[j * K : j * K + K]
        # First 128-byte chunk
        state.xmem.write_address(WEIGHTS_BASE + j * W_STRIDE, bytearray(row[:128]))
        # Second chunk: remaining K-128=16 bytes, zero-padded to 128
        tail = bytearray(128)
        tail[:K - 128] = row[128:K]
        state.xmem.write_address(WEIGHTS_BASE + j * W_STRIDE + 128, tail)


class MatMul432x144x128App(IpuApp):
    """432×144 transformer matmul application harness.

    Args:
        inst_path:    Path to assembled instruction binary.
        input_path:   Path to channel-major input D (K×256 bytes).
        weights_path: Path to output-major weight W (N_OUT×K bytes, W[j][k]).
        output_path:  Optional path to write output C.
        dtype:        Data type string or :class:`DType`.
    """

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        state.dtype = self.dtype
        _load_data(state, self.input_path)
        _load_weights(state, self.weights_path)
        state.regfile.set_cr(0, DATA_BASE)
        state.regfile.set_cr(1, WEIGHTS_BASE)
        state.regfile.set_cr(2, WEIGHTS_BASE + 128)
        state.regfile.set_cr(3, OUTPUT_BASE)                    # tg=0 output
        state.regfile.set_cr(4, OUTPUT_BASE + N_OUT * 512)      # tg=1 output
        state.regfile.set_cr(5, -256)                           # tg=0 data startup
        state.regfile.set_cr(6, -128)                           # tg=1 data startup
        state.regfile.set_cr(7, -1)                             # k-loop1 fixed_idx startup
        state.regfile.set_cr(8, 127)                            # k-loop2 fixed_idx startup
        state.regfile.set_lr(0, 0)                              # r_cyclic write-index 0
        state.regfile.set_lr(2, 256)                            # data stride
        state.regfile.set_lr(3, 512)                            # output stride
        state.regfile.set_lr(6, 126)                            # k-loop1 bound: first_index=0, width=128 → 126
        state.regfile.set_lr(7, 0)                              # output pointer
        state.regfile.set_lr(8, 0)                              # weight byte offset
        state.regfile.set_lr(9, 0)                              # j counter
        state.regfile.set_lr(10, N_OUT)                         # j-loop limit
        state.regfile.set_lr(11, 142)                           # k-loop2 bound: first_index=128, width=16 → 142
        state.regfile.set_lr(12, W_STRIDE)                      # weight stride per j

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, OUTPUT_ROW_BYTES, N_OUT * N_TG,
            )
