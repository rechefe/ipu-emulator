"""STORE and ACC_STORE slot handlers: ST.AAQ and the debug-only ST.ACC."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from ipu_emu.errors import DebugOnlyInstructionWarning
from ipu_emu.geometry import xmem_row_addr, xmem_row_check
from ipu_isa.faults import FaultId

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu


def execute_store_nop(ipu: Ipu) -> None:
    """Execute NOP in store slot: No operation."""


def execute_acc_store_nop(ipu: Ipu) -> None:
    """Execute NOP in acc_store slot: No operation."""


def check_st_acc(ipu: Ipu, *, offset: int, base: int) -> tuple[FaultId, str] | None:
    """F-6 for a row outside XMEM."""
    return xmem_row_check(ipu.state, offset + base)


def check_st_aaq(ipu: Ipu, *, offset: int, base: int) -> tuple[FaultId, str] | None:
    """F-6 for a row outside XMEM."""
    return xmem_row_check(ipu.state, offset + base)


def execute_st_acc(ipu: Ipu, *, offset: int, base: int) -> None:
    """Execute ST.ACC: Store accumulator to memory (debug only).

    Stores all 512 bytes of R_ACC (128 elements x 32-bit accumulator
    width) unconditionally in both modes -- R_ACC's width does not scale
    with the active element width, so "all of R_ACC" is already the
    mode-blind statement. Only the row address is translated; this spans
    4 rows in narrow mode and 1 row in debug mode, which is accepted
    since this is a debug-only instruction by nature (see warning above).
    """
    warnings.warn(
        "[DEBUG ONLY] ST.ACC is not a hardware instruction and is available "
        "for emulator debugging purposes only",
        DebugOnlyInstructionWarning,
        stacklevel=2,
    )
    addr = xmem_row_addr(ipu.state, offset + base)
    acc_data = bytearray(ipu._view.read("r_acc"))
    ipu.state.xmem.write_address(addr, acc_data)


def execute_st_aaq(ipu: Ipu, *, offset: int, base: int) -> None:
    """Store **POST_AAQ_REG** (512 bytes) to XMEM.

    POST_AAQ_REG's width does not scale with the active element width
    (like R_ACC), so all 512 bytes are written unconditionally in both
    modes; only the row address is translated.
    """
    addr = xmem_row_addr(ipu.state, offset + base)
    ipu.state.xmem.write_address(addr, bytes(ipu._view.read("post_aaq_reg")))
