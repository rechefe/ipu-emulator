"""LayerNorm 256×144 harness.

Computes out[ch, t] = γ[ch] × (x[ch,t] − mean_t) / sqrt(var_t) + β[ch]
for 256 tokens × 144 channels, using FP8_E4M3 inputs with FP32 accumulation.

aaq instructions use INT8 truncation (val >> 24) at 5 points — see
TODO(fp8_aaq) markers in the assembly.

Output layout: (144 ch × 2 tg) rows of 512-byte FP32 accumulators,
row for (ch, tg) at OUTPUT_BASE + (ch*2 + tg)*512.

Usage::

    from ipu_apps.layernorm_256x144 import LayerNorm256x144App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.ipu_math import DType, fp32_to_fp8_bytes
from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N_CH    = 144
N_TOK   = 256
N_TG    = 2
N_TPG   = 128   # tokens per group
ROW_BYTES = 128
OUTPUT_ROW_BYTES = 512

DATA_BASE     = 0x00000
GAMMA_BASE    = 0x0A000
BETA_BASE     = 0x0A100
ONES_BASE     = 0x0A200
NEG_ONES_BASE = 0x0A280
MASK_BASE     = 0x0A300
TEMP_BASE     = 0x0AB00
OUTPUT_BASE   = 0x0AC00

# FP8_E4M3 encoding of 1.0 and -1.0
_FP8_ONE     = int.from_bytes(fp32_to_fp8_bytes(np.array([1.0],  dtype=np.float32), DType.E4), "little")
_FP8_NEG_ONE = int.from_bytes(fp32_to_fp8_bytes(np.array([-1.0], dtype=np.float32), DType.E4), "little")


def _build_masks() -> bytearray:
    """Build 2048-byte mask region: 128 masks × 16 bytes, packed 8 per 128-byte block.

    Mask for token t: 128-bit integer with all bits=1 except bit t=0 (pass lane t through).
    Stored little-endian. 8 masks per 128-byte block.
    """
    buf = bytearray(16 * 128)
    for t in range(128):
        mask_128bit = ((1 << 128) - 1) ^ (1 << t)
        mask_bytes = mask_128bit.to_bytes(16, byteorder="little")
        buf[t * 16 : t * 16 + 16] = mask_bytes
    return buf


class LayerNorm256x144App(IpuApp):
    """LayerNorm 256×144 application harness."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path  = Path(self.input_path)
        self.gamma_path  = Path(self.gamma_path)
        self.beta_path   = Path(self.beta_path)

    def setup(self, state: "IpuState") -> None:
        # Run in INT8 mode so aaq instruction is available.
        # Input bytes are FP8_E4M3-encoded but treated as raw bytes by INT8 arithmetic.
        state.set_cr_dtype(int(DType.INT8))

        # Load input x: [144 ch × 2 tg × 128 tok] = 36,864 bytes
        raw_x = self.input_path.read_bytes()
        state.xmem.write_address(DATA_BASE, bytearray(raw_x))

        # Load γ: two 128-byte rows
        raw_gamma = self.gamma_path.read_bytes()   # 144 bytes FP8_E4M3
        row0 = bytearray(raw_gamma[:128])
        row1 = bytearray(128)
        row1[:N_CH - 128] = raw_gamma[128:N_CH]
        state.xmem.write_address(GAMMA_BASE,       row0)
        state.xmem.write_address(GAMMA_BASE + 128, row1)

        # Load β: same layout
        raw_beta = self.beta_path.read_bytes()     # 144 bytes FP8_E4M3
        brow0 = bytearray(raw_beta[:128])
        brow1 = bytearray(128)
        brow1[:N_CH - 128] = raw_beta[128:N_CH]
        state.xmem.write_address(BETA_BASE,       brow0)
        state.xmem.write_address(BETA_BASE + 128, brow1)

        # Ones and neg-ones (FP8_E4M3)
        state.xmem.write_address(ONES_BASE,     bytearray([_FP8_ONE]     * ROW_BYTES))
        state.xmem.write_address(NEG_ONES_BASE, bytearray([_FP8_NEG_ONE] * ROW_BYTES))

        # Mask region
        state.xmem.write_address(MASK_BASE, _build_masks())

        # CR registers
        state.regfile.set_cr(0,  DATA_BASE)
        state.regfile.set_cr(1,  GAMMA_BASE)
        state.regfile.set_cr(2,  BETA_BASE)
        state.regfile.set_cr(3,  MASK_BASE)
        state.regfile.set_cr(4,  ONES_BASE)
        state.regfile.set_cr(5,  TEMP_BASE)
        state.regfile.set_cr(6,  OUTPUT_BASE)
        state.regfile.set_cr(7,  NEG_ONES_BASE)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, OUTPUT_ROW_BYTES, N_CH * N_TG,
            )
