"""Tests for SystemRDL codegen artifacts."""

from __future__ import annotations

from ipu_ctrl_regs import (
    CTRL_START_PULSE,
    IMEM_BASE,
    MMIO_BASE,
    REG_CTRL_OFFSET,
    REG_CTRL_START_SHIFT,
    REG_ID_MAGIC_MASK,
    REG_ID_OFFSET,
    REG_STATUS_OFFSET,
)


def test_register_offsets_match_rdl_layout() -> None:
    assert REG_ID_OFFSET == 0x000
    assert REG_CTRL_OFFSET == 0x004
    assert REG_STATUS_OFFSET == 0x008
    assert REG_ID_MAGIC_MASK == 0x0000_FFFF
    assert REG_CTRL_START_SHIFT == 0
    assert CTRL_START_PULSE == 1


def test_address_map_constants() -> None:
    assert MMIO_BASE == 0x1000_0000
    assert IMEM_BASE == 0x1001_0000
