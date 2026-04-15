"""Hardcoded first-layer 3x3 convolution: 256x256x3 -> 128x128x16, stride 2.

Input:  256x256x3 (INT8, CHW layout)
Output: 128x128x16 (INT8, quantized, interleaved 128-byte chunks)

Usage::

    from ipu_apps.convolutions_universal.conv_first_layer import (
        ConvFirstLayerApp,
    )

    app = ConvFirstLayerApp(
        inst_path="conv_first_layer.bin",
        input_path="input.bin",
        kernel_path="kernel.bin",
        output_path="output.bin",
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Constants ---------------------------------------------------------------

IN_ROWS = 256
IN_COLS = 256
IN_CHANNELS = 3
OUT_ROWS = 128
OUT_COLS = 128
OUT_CHANNELS = 16
KERNEL_SIZE = 3
STRIDE = 2

CHANNEL_STRIDE = IN_ROWS * IN_COLS  # 65536
ROW_STRIDE = IN_COLS                # 256

# -- Memory layout -----------------------------------------------------------

INPUT_BASE_ADDR = 0x000000
# Channel c starts at INPUT_BASE + c * 65536
# Total input: 3 * 65536 = 196608 bytes

KERNEL_BASE_ADDR = 0x040000
# 16 filters x 128 bytes = 2048 bytes

MASK_BASE_ADDR = 0x041000
# 128 bytes (8 slots x 16 bytes)

TEMP_BASE_ADDR = 0x041100
# 256 bytes (temp_left[128] + temp_right[128])

OUTPUT_BASE_ADDR = 0x050000
# 128 rows x 16 channels x 128 bytes = 262144 bytes


def _build_mask_data() -> bytes:
    """Build 128-byte mask register data for cols=128 (1 row per chunk).

    Slots:
      0: all zeros      (no masking, kc=0)
      1: bit 0 set      (zero position 0, kc=-1 left border)
      2: bit 127 set    (zero position 127, kc=+1 right border)
    """
    mask = bytearray(128)
    # Slot 1: zero position 0
    mask[1 * 16 + 0] |= 1 << 0  # bit 0 of slot 1
    # Slot 2: zero position 127
    mask[2 * 16 + 127 // 8] |= 1 << (127 % 8)  # bit 127 of slot 2
    return bytes(mask)


def _build_kernel_data(kernel_raw: bytes) -> bytes:
    """Pack kernel into 16 blocks of 128 bytes.

    Input: 16 filters x 3 channels x 9 taps = 432 bytes (contiguous).
      kernel_raw[f * 27 + c * 9 + tap]

    Output: 16 blocks of 128 bytes.
      Each block: 27 bytes of taps + 101 bytes padding.
    """
    kernel_padded = bytearray(OUT_CHANNELS * 128)
    taps_per_filter = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE  # 27
    for f in range(OUT_CHANNELS):
        src = f * taps_per_filter
        dst = f * 128
        kernel_padded[dst:dst + taps_per_filter] = (
            kernel_raw[src:src + taps_per_filter]
        )
    return bytes(kernel_padded)


class ConvFirstLayerApp(IpuApp):
    """Hardcoded first-layer 3x3 convolution application harness.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input image binary (256x256x3, CHW, INT8).
        kernel_path:  Path to kernel binary (16x3x9 = 432 bytes).
        output_path:  Optional path to write output.
    """

    ASM_PATH = Path(__file__).resolve().parent / "conv_first_layer.asm"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.kernel_path = Path(self.kernel_path)

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(DType.INT8))

        # Load input (CHW: 3 channels x 256 rows x 256 cols)
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (packed into 128-byte blocks)
        kernel_raw = self.kernel_path.read_bytes()
        kernel_padded = _build_kernel_data(kernel_raw)
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_padded)

        # Load mask data
        mask_data = _build_mask_data()
        state.xmem.write_address(MASK_BASE_ADDR, mask_data)

        # CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)                    # ch0 base
        state.regfile.set_cr(1, INPUT_BASE_ADDR + CHANNEL_STRIDE)   # ch1 base
        state.regfile.set_cr(2, INPUT_BASE_ADDR + 2 * CHANNEL_STRIDE)  # ch2 base
        state.regfile.set_cr(3, KERNEL_BASE_ADDR)
        state.regfile.set_cr(4, MASK_BASE_ADDR)
        state.regfile.set_cr(5, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(6, TEMP_BASE_ADDR)
        state.regfile.set_cr(9, 1)  # identity scalar for mult.ve.cr

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_bytes = OUT_ROWS * OUT_CHANNELS * OUT_COLS  # 128*16*128
            data = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
            Path(self.output_path).write_bytes(data)
