"""Concat 16x16x128x128: channel-axis concat of two 128-channel tensors (L4).

Concatenates two same-spatial-shape, different-(or-equal)-channel-count
tensors along the channel axis: ``output = cat((A, B), dim=channel)``,
matching MobileViT-S's ``torch.cat((residual, features), dim=channel)`` at
the end of its L4 transformer block (both branches have 128
channels there, so the concrete shape here is 128 + 128 -> 256).

A real copy kernel, not a no-op: see the ``.asm`` header's "WHY A REAL COPY
KERNEL" note for why this does not just rely on callers pre-arranging shared
memory. A and B may live at arbitrary (non-adjacent) XMEM bases; this kernel
physically copies both into one contiguous output buffer.

Usage::

    from ipu_apps.kernels.reshape.concat_16x16x128x128.app import Concat16x16x128x128App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.kernel_registry.base import IpuApp
from ipu_apps.kernels.reshape.app import concat_spec

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Dimensions -------------------------------------------------------------

H   = 16    # spatial height
W   = 16    # spatial width
C_A = 128    # channels in input A (residual branch)
C_B = 128    # channels in input B (fold-branch features)
C_OUT = C_A + C_B

# -- Memory map ---------------------------------------------------------
#
# Wide-vector FP32 only, same convention as fold/unfold/residual_add: an
# XMEM row is LANES * 4 = 512 bytes unconditionally, and .asm XMEM operands
# are ROW numbers, not byte offsets. One channel = one row.
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

A_BASE_ROW   = 0
B_BASE_ROW   = A_BASE_ROW + C_A
OUT_BASE_ROW = B_BASE_ROW + C_B

A_BASE   = A_BASE_ROW * ROW_BYTES
B_BASE   = B_BASE_ROW * ROW_BYTES
OUT_BASE = OUT_BASE_ROW * ROW_BYTES


class Concat16x16x128x128App(IpuApp):
    """Concat two 128-channel, 16x16-spatial tensors along the channel axis.

    Args:
        inst_path:    Path to assembled instruction binary.
        input_a_path: Path to input A (128 rows of 128 FP32 elements).
        input_b_path: Path to input B (128 rows of 128 FP32 elements).
        output_path:  Optional path to write the 256-row concatenated output.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_a_path = Path(self.input_a_path)
        self.input_b_path = Path(self.input_b_path)

    def setup(self, state: "IpuState") -> None:
        raw_a = Path(self.input_a_path).read_bytes()
        raw_b = Path(self.input_b_path).read_bytes()
        if len(raw_a) != C_A * ROW_BYTES:
            raise ValueError(
                f"A is {len(raw_a)} bytes, expected {C_A * ROW_BYTES} "
                "(one full 128-element FP32 row per channel)"
            )
        if len(raw_b) != C_B * ROW_BYTES:
            raise ValueError(f"B is {len(raw_b)} bytes, expected {C_B * ROW_BYTES}")
        state.xmem.write_address(A_BASE, bytearray(raw_a))
        state.xmem.write_address(B_BASE, bytearray(raw_b))

        # CR0 (=0) and CR1 (=1) are read-only hardwired constants -- writing
        # to either raises EmulatorError, even when the value written matches
        # the hardwired one. A_BASE_ROW is 0 here, so CR0 already holds the
        # correct value without any write; the .asm's ``ZERO`` alias for CR0
        # relies on the hardwired value, not on this setup() writing it.
        state.regfile.set_cr(2, -1)                 # PTR_START: src ptr startup (-1 row)
        state.regfile.set_cr(3, 1)                  # ROW_STRIDE
        state.regfile.set_cr(4, 1)                  # DTYPE_ONE: 1.0 in wide FP32 (low byte -> float)
        state.regfile.set_cr(5, A_BASE_ROW)
        state.regfile.set_cr(6, B_BASE_ROW)
        state.regfile.set_cr(7, OUT_BASE_ROW)                # OUT_A_BASE
        state.regfile.set_cr(8, OUT_BASE_ROW + C_A)          # OUT_B_BASE
        state.regfile.set_cr(9, C_A)                 # C_A_COUNT
        state.regfile.set_cr(10, C_B)                # C_B_COUNT

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUT_BASE, ROW_BYTES, C_OUT,
            )


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list; see
# :func:`~ipu_apps.kernels.reshape.app.concat_spec` for the
# exact-shape `supports`.

SPEC = concat_spec(Concat16x16x128x128App, h=H, w=W, c_a=C_A, c_b=C_B)
