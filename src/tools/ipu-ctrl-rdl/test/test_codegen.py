"""Verify generated register metadata matches ipu_ctrl.rdl."""

from __future__ import annotations

from pathlib import Path

import ipu_ctrl_regs as regs


def test_ctrl_start_bit_position() -> None:
    assert regs.CTRL.START == 0
    assert regs.CTRL.START_MASK == 0x1
    assert regs.CTRL.HALT == 1
    assert regs.CTRL.RESET == regs.CTRL.SOFT_RESET
    assert regs.CTRL.RESET_MASK == 0x8


def test_dstructure_field_layout() -> None:
    assert regs.DSTRUCTURE.VALID_ELEMENTS == 0
    assert regs.DSTRUCTURE.VALID_ELEMENTS_MASK == 0xFF
    assert regs.DSTRUCTURE.PARTITION == 8
    assert regs.DSTRUCTURE.PARTITION_MASK == 0xF00


def test_register_offsets_match_rdl() -> None:
    assert regs.ID.OFFSET == 0x000
    assert regs.CTRL.OFFSET == 0x004
    assert regs.STATUS.OFFSET == 0x008
    assert regs.DSTRUCTURE.OFFSET == 0x024
    assert regs.CR0.OFFSET == 0x040
    assert regs.CR15.OFFSET == 0x07C


def test_host_address_constants() -> None:
    assert regs.MMIO_BASE == 0x10000000
    assert regs.IMEM_BASE == 0x10010000
    assert regs.IMEM_MAP_SIZE == regs.IMEM_DEPTH * regs.INSTRUCTION_ALIGNED_BYTES


def test_rdl_source_present() -> None:
    rdl = Path(__file__).resolve().parents[3] / "hw" / "ipu_ctrl.rdl"
    assert rdl.is_file()
    text = rdl.read_text(encoding="utf-8")
    assert "addrmap ipu_host" in text
    assert "dstructure" in text
