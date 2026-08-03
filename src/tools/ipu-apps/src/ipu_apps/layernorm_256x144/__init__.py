"""LayerNorm 256×144 harness.

Computes output[ch, tg, i] = γ[ch] × (x[ch,tg,i] − μ[tg,i]) / σ[tg,i] + β[ch]
for 144 channels × 2 token groups × 128 tokens/group, using wide-vector FP32.

Data layout in XMEM: DATA_BASE + (ch*N_TG + tg)*512  (channel-major, tg interleaved).
Output layout: OUTPUT_BASE + (ch*N_TG + tg)*512.
γ/β span two 512-byte rows (144 > 128 lanes).

Usage::

    from ipu_apps.layernorm_256x144 import LayerNorm256x144App
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N_CH        = 144
N_TG        = 2
N_TPG       = 128    # tokens per group (SIMD width)
ROW_BYTES   = 512    # wide-vector mode: 128 × FP32

DATA_BASE        = 0x00000   # N_CH × N_TG × ROW_BYTES = 147 456 B
GAMMA_BASE       = 0x24000   # 2 × ROW_BYTES = 1 024 B  (γ[0..127], γ[128..143])
BETA_BASE        = 0x24400   # 2 × ROW_BYTES = 1 024 B
ONES_BASE        = 0x24800   # ROW_BYTES = 512 B  (128 × 1.0f)
NEG_INV_N_BASE   = 0x24A00   # ROW_BYTES = 512 B  (128 × −1/N_CH)
INV_N_BASE       = 0x24C00   # ROW_BYTES = 512 B  (128 ×  1/N_CH)
NEG_MEAN_BASE    = 0x24E00   # ROW_BYTES = 512 B  (reused per tg)
CENTERED_BASE    = 0x25000   # N_CH × ROW_BYTES = 73 728 B  (reused per tg)
TEMP_BASE        = 0x37000   # ROW_BYTES = 512 B
INVSTD_BASE      = 0x37200   # ROW_BYTES = 512 B
OUTPUT_BASE      = 0x37400   # N_CH × N_TG × ROW_BYTES = 147 456 B

# XMEM .asm operands are ROW numbers (one row = LANES elements), not byte
# addresses -- see issue #179. The *_BASE constants above stay byte addresses
# because they only drive the harness's direct xmem read/write calls (which
# bypass row translation); the CR registers in setup() feed the .asm's XMEM
# instructions instead, so they carry addresses converted to rows.
#
# These kernels run in WIDE-VECTOR DEBUG mode (the test builds IpuState with
# wide_vector_debug=True), where an element is 4 B, so one 128-lane row is
# 512 B -- not the 128 B of narrow mode. ROW_BYTES already is that row size.
ROW_SIZE_BYTES = ROW_BYTES               # 512 in wide-vector debug mode
DATA_BASE_ROW      = DATA_BASE // ROW_SIZE_BYTES
GAMMA_BASE_ROW     = GAMMA_BASE // ROW_SIZE_BYTES
BETA_BASE_ROW      = BETA_BASE // ROW_SIZE_BYTES
ONES_BASE_ROW      = ONES_BASE // ROW_SIZE_BYTES
NEG_INV_N_BASE_ROW = NEG_INV_N_BASE // ROW_SIZE_BYTES
INV_N_BASE_ROW     = INV_N_BASE // ROW_SIZE_BYTES
NEG_MEAN_BASE_ROW  = NEG_MEAN_BASE // ROW_SIZE_BYTES
CENTERED_BASE_ROW  = CENTERED_BASE // ROW_SIZE_BYTES
TEMP_BASE_ROW      = TEMP_BASE // ROW_SIZE_BYTES
INVSTD_BASE_ROW    = INVSTD_BASE // ROW_SIZE_BYTES
OUTPUT_BASE_ROW    = OUTPUT_BASE // ROW_SIZE_BYTES
ROW_STRIDE_ROWS    = 1                   # one ROW_BYTES row = exactly 1 XMEM row


def _fp32_row(values: np.ndarray) -> bytes:
    """Pack a 1-D float32 array into 512 bytes (zero-padded to 128 lanes)."""
    assert values.ndim == 1 and len(values) <= 128
    padded = np.zeros(128, dtype=np.float32)
    padded[: len(values)] = values
    return padded.tobytes()


class LayerNorm256x144App(IpuApp):
    """256-token × 144-channel LayerNorm in wide-vector FP32 debug mode."""

    def __init__(
        self,
        *,
        input_path: str | Path,
        gamma_path: str | Path,
        beta_path: str | Path,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(input_path)
        self.gamma_path = Path(gamma_path)
        self.beta_path  = Path(beta_path)

    def setup(self, state: "IpuState") -> None:
        # Data: N_CH × N_TG rows of N_TPG FP32 values
        # File layout: (ch*N_TG + tg) row order, each row 512 bytes
        state.xmem.write_address(DATA_BASE, bytearray(self.input_path.read_bytes()))

        # γ and β: 144 values each → two 512-byte rows
        gamma = np.frombuffer(self.gamma_path.read_bytes(), dtype=np.float32)
        beta  = np.frombuffer(self.beta_path.read_bytes(),  dtype=np.float32)
        assert len(gamma) == N_CH and len(beta) == N_CH

        state.xmem.write_address(GAMMA_BASE,              bytearray(_fp32_row(gamma[:128])))
        state.xmem.write_address(GAMMA_BASE + ROW_BYTES,  bytearray(_fp32_row(gamma[128:])))
        state.xmem.write_address(BETA_BASE,               bytearray(_fp32_row(beta[:128])))
        state.xmem.write_address(BETA_BASE + ROW_BYTES,   bytearray(_fp32_row(beta[128:])))

        # Constants
        ones = np.ones(N_TPG, dtype=np.float32).tobytes()
        state.xmem.write_address(ONES_BASE,     bytearray(ones))
        neg_inv_n = np.full(N_TPG, -1.0 / N_CH, dtype=np.float32).tobytes()
        state.xmem.write_address(NEG_INV_N_BASE, bytearray(neg_inv_n))
        inv_n = np.full(N_TPG,  1.0 / N_CH, dtype=np.float32).tobytes()
        state.xmem.write_address(INV_N_BASE,    bytearray(inv_n))

        # CR registers — must match ASM header.
        # NOTE: CR0 (=0) and CR1 (=1) are read-only hardwired constants in the new
        # architecture; writes are silently dropped. DATA_BASE is 0x0 so CR0 is fine,
        # and GAMMA_BASE moved off CR1 to CR11 (CR11's old const-zero role is served
        # by the hardwired CR0).
        state.regfile.set_cr(0,  DATA_BASE_ROW)
        state.regfile.set_cr(2,  BETA_BASE_ROW)
        state.regfile.set_cr(3,  ONES_BASE_ROW)
        state.regfile.set_cr(4,  NEG_INV_N_BASE_ROW)
        state.regfile.set_cr(5,  INV_N_BASE_ROW)
        state.regfile.set_cr(6,  NEG_MEAN_BASE_ROW)
        state.regfile.set_cr(7,  CENTERED_BASE_ROW)
        state.regfile.set_cr(8,  TEMP_BASE_ROW)
        state.regfile.set_cr(9,  INVSTD_BASE_ROW)
        state.regfile.set_cr(10, OUTPUT_BASE_ROW)
        state.regfile.set_cr(11, GAMMA_BASE_ROW)   # moved off read-only CR1
        state.regfile.set_cr(12, N_CH)      # 144
        state.regfile.set_cr(13, ROW_STRIDE_ROWS) # 1 row
        state.regfile.set_cr(14, N_TPG)     # 128

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, ROW_BYTES, N_CH * N_TG,
            )
