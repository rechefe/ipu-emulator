"""Concat 32x32x96x96: channel-axis concat of two 96-channel tensors (L3).

Concatenates two same-spatial-shape, different-(or-equal)-channel-count
tensors along the channel axis: ``output = cat((A, B), dim=channel)``,
matching MobileViT-S's ``torch.cat((residual, features), dim=channel)`` at
the end of its L3 transformer block (verified: both branches have 96
channels there, so the concrete shape here is 96 + 96 -> 192).

A real copy kernel, not a no-op: see the ``.asm`` header's "WHY A REAL COPY
KERNEL" note for why this does not just rely on callers pre-arranging shared
memory. A and B may live at arbitrary (non-adjacent) XMEM bases; this kernel
physically copies both into one contiguous output buffer.

Usage::

    from ipu_apps.unfold.concat_32x32x96x96 import Concat32x32x96x96App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import KernelSpec, no, yes
from ipu_apps.unfold._concat_spec_support import (
    WIDE_VECTOR_ONLY,
    concat_query,
    positive_dims,
)

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Dimensions -------------------------------------------------------------

H   = 32    # spatial height
W   = 32    # spatial width
C_A = 96    # channels in input A (residual branch)
C_B = 96    # channels in input B (fold-branch features)
C_OUT = C_A + C_B

# -- Memory map ---------------------------------------------------------
#
# Wide-vector FP32 only, same convention as fold/unfold/residual_add: an
# XMEM row is LANES * 4 = 512 bytes unconditionally, and .asm XMEM operands
# are ROW numbers (issue #179), not byte offsets. One channel = one row.
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

A_BASE_ROW   = 0
B_BASE_ROW   = A_BASE_ROW + C_A
OUT_BASE_ROW = B_BASE_ROW + C_B

A_BASE   = A_BASE_ROW * ROW_BYTES
B_BASE   = B_BASE_ROW * ROW_BYTES
OUT_BASE = OUT_BASE_ROW * ROW_BYTES


class Concat32x32x96x96App(IpuApp):
    """Concat two 96-channel, 32x32-spatial tensors along the channel axis.

    Args:
        inst_path:    Path to assembled instruction binary.
        input_a_path: Path to input A (96 rows x 512 bytes).
        input_b_path: Path to input B (96 rows x 512 bytes).
        output_path:  Optional path to write the 192-row concatenated output.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_a_path = Path(self.input_a_path)
        self.input_b_path = Path(self.input_b_path)

    def setup(self, state: "IpuState") -> None:
        raw_a = Path(self.input_a_path).read_bytes()
        raw_b = Path(self.input_b_path).read_bytes()
        assert len(raw_a) == C_A * ROW_BYTES, (
            f"A is {len(raw_a)} bytes, expected {C_A * ROW_BYTES} "
            "(one full 512-byte row per channel)"
        )
        assert len(raw_b) == C_B * ROW_BYTES, (
            f"B is {len(raw_b)} bytes, expected {C_B * ROW_BYTES}"
        )
        state.xmem.write_address(A_BASE, bytearray(raw_a))
        state.xmem.write_address(B_BASE, bytearray(raw_b))

        # cr0 (=0) and cr1 (=1) are read-only hardwired constants -- writing
        # to either now raises EmulatorError (issue #230 / PR #231), even
        # when the value written matches the hardwired one. A_BASE_ROW is 0
        # here, so cr0 already holds the correct value without any write;
        # the .asm's `ZERO = cr0` alias relies on the hardwired value, not
        # on this setup() writing it.
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
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's domain: it is an exact-shape
# match, since C_A/C_B and the loop bounds are baked into cr9/cr10 by
# setup() for this one (H, W, C_A, C_B) tuple -- same convention fold and
# residual_add use.


def _supports(**params):
    q = concat_query(params["shape"])
    bad = positive_dims(q)
    if bad:
        return no(bad)
    if (q.h, q.w, q.c_a, q.c_b) != (H, W, C_A, C_B):
        return no(
            f"handles exactly (H, W, C_A, C_B) = ({H}, {W}, {C_A}, {C_B}); "
            f"got ({q.h}, {q.w}, {q.c_a}, {q.c_b})"
        )
    return yes()


def _build(**params):
    return {}


def _explain(**params):
    return (
        f"(H, W, C_A, C_B) == ({H}, {W}, {C_A}, {C_B}) exactly: channel "
        f"counts and loop bounds are fixed constants loaded by setup() for "
        f"this shape."
    )


SPEC = KernelSpec(
    name="concat_32x32x96x96",
    op="concat",
    variant="32x32x96x96",
    app_class=Concat32x32x96x96App,
    asm="concat_32x32x96x96.asm",
    requires=("shape",),
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=lambda **params: (WIDE_VECTOR_ONLY,),
    bundle=lambda **params: concat_query(params["shape"]).bundle,
    cost=lambda **params: 0.0,
)
