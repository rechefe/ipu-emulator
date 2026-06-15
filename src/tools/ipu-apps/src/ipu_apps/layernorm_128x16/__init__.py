"""LayerNorm 128×16 harness.

Computes output[ch, i] = γ[ch] × (x[ch,i] − μ[i]) / σ[i] + β[ch]
for ch=0..15, i=0..127 (one token group, 16 channels).
Runs in wide-vector FP32 debug mode.

Usage::

    from ipu_apps.layernorm_128x16 import LayerNorm128x16App
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.emulator import dump_xmem_to_binary
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    pass

N_CH        = 16
N_TPG       = 128    # tokens per group (one token group)
ROW_BYTES   = 512    # wide-vector mode: 128 × FP32

DATA_BASE        = 0x00000   # N_CH × ROW_BYTES = 8 192 B
GAMMA_BASE       = 0x02000   # ROW_BYTES = 512 B
BETA_BASE        = 0x02200   # ROW_BYTES = 512 B
ONES_BASE        = 0x02400   # ROW_BYTES = 512 B  (128 × 1.0f)
NEG_INV_N_BASE   = 0x02600   # ROW_BYTES = 512 B  (128 × -1/N_CH)
INV_N_BASE       = 0x02800   # ROW_BYTES = 512 B  (128 ×  1/N_CH)
NEG_MEAN_BASE    = 0x02A00   # ROW_BYTES = 512 B  (written step 1)
CENTERED_BASE    = 0x02C00   # N_CH × ROW_BYTES = 8 192 B
TEMP_BASE        = 0x04E00   # ROW_BYTES = 512 B
INVSTD_BASE      = 0x05000   # ROW_BYTES = 512 B
OUTPUT_BASE      = 0x05200   # N_CH × ROW_BYTES = 8 192 B


def _fp32_row(values: list[float]) -> bytes:
    """Pack a list of float32 values into 512 bytes (zero-padded to 128 lanes)."""
    assert len(values) <= 128
    packed = struct.pack(f"<{len(values)}f", *values)
    return packed + b"\x00" * (ROW_BYTES - len(packed))


class LayerNorm128x16App(IpuApp):
    """128-token × 16-channel LayerNorm in wide-vector FP32 debug mode."""

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
        self.beta_path = Path(beta_path)

    def setup(self, state: "IpuState") -> None:
        # Load input: N_CH rows of N_TPG FP32 values
        raw = self.input_path.read_bytes()
        state.xmem.write_address(DATA_BASE, bytearray(raw))

        # Load γ and β as 512-byte rows (16 meaningful values, rest zero)
        state.xmem.write_address(GAMMA_BASE, bytearray(self.gamma_path.read_bytes()))
        state.xmem.write_address(BETA_BASE,  bytearray(self.beta_path.read_bytes()))

        # Constants
        ones_fp32 = np.ones(N_TPG, dtype=np.float32).tobytes()
        state.xmem.write_address(ONES_BASE,       bytearray(ones_fp32))
        neg_inv_n = np.full(N_TPG, -1.0 / N_CH, dtype=np.float32).tobytes()
        state.xmem.write_address(NEG_INV_N_BASE,  bytearray(neg_inv_n))
        inv_n = np.full(N_TPG, 1.0 / N_CH, dtype=np.float32).tobytes()
        state.xmem.write_address(INV_N_BASE,      bytearray(inv_n))

        # CR registers
        state.regfile.set_cr(0,  DATA_BASE)
        state.regfile.set_cr(1,  GAMMA_BASE)
        state.regfile.set_cr(2,  BETA_BASE)
        state.regfile.set_cr(3,  ONES_BASE)
        state.regfile.set_cr(4,  NEG_INV_N_BASE)
        state.regfile.set_cr(5,  INV_N_BASE)
        state.regfile.set_cr(6,  NEG_MEAN_BASE)
        state.regfile.set_cr(7,  CENTERED_BASE)
        state.regfile.set_cr(8,  TEMP_BASE)
        state.regfile.set_cr(9,  INVSTD_BASE)
        state.regfile.set_cr(10, OUTPUT_BASE)
        state.regfile.set_cr(11, 0)      # const zero
        state.regfile.set_cr(12, N_CH)
        state.regfile.set_cr(13, ROW_BYTES)
        state.regfile.set_cr(14, N_TPG)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, ROW_BYTES, N_CH,
            )
