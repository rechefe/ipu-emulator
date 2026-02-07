"""Tests for ipu_emu.regfile and ipu_emu.ipu_state."""

import copy
import struct

import numpy as np
import pytest

from ipu_emu.descriptors import REGFILE_SCHEMA, RegDescriptor, RegDtype, RegKind
from ipu_emu.regfile import RegFile
from ipu_emu.ipu_state import IpuState


# ---------------------------------------------------------------------------
# RegFile — scalar (LR / CR) access
# ---------------------------------------------------------------------------

class TestRegFileScalars:
    def test_lr_set_get(self):
        rf = RegFile()
        for i in range(16):
            rf.set_lr(i, i * 100)
        for i in range(16):
            assert rf.get_lr(i) == i * 100

    def test_cr_set_get(self):
        rf = RegFile()
        rf.set_cr(0, 42)
        rf.set_cr(15, 0xDEADBEEF)
        assert rf.get_cr(0) == 42
        assert rf.get_cr(15) == 0xDEADBEEF

    def test_lr_overflow_wraps_to_32bit(self):
        rf = RegFile()
        rf.set_lr(0, 0x1_0000_0001)  # > 32 bits
        assert rf.get_lr(0) == 1  # only low 32 bits kept

    def test_lr_index_out_of_range(self):
        rf = RegFile()
        with pytest.raises(AssertionError):
            rf.set_lr(16, 0)


# ---------------------------------------------------------------------------
# RegFile — R registers (128-byte)
# ---------------------------------------------------------------------------

class TestRegFileRRegs:
    def test_set_get_r0(self):
        rf = RegFile()
        data = bytearray(range(128))
        rf.set_r(0, data)
        assert rf.get_r(0) == data

    def test_set_get_r1(self):
        rf = RegFile()
        data = bytearray([0xFF] * 128)
        rf.set_r(1, data)
        assert rf.get_r(1) == data
        # r0 should still be zero
        assert rf.get_r(0) == bytearray(128)

    def test_r_get_returns_copy(self):
        rf = RegFile()
        data = bytearray([0xAB] * 128)
        rf.set_r(0, data)
        got = rf.get_r(0)
        got[0] = 0x00
        assert rf.get_r(0)[0] == 0xAB  # original unchanged

    def test_r_wrong_size_asserts(self):
        rf = RegFile()
        with pytest.raises(AssertionError):
            rf.set_r(0, bytearray(64))  # too small

    def test_r_index_out_of_range(self):
        rf = RegFile()
        with pytest.raises(AssertionError):
            rf.get_r(2)


# ---------------------------------------------------------------------------
# RegFile — R cyclic (512-byte, wraparound)
# ---------------------------------------------------------------------------

class TestRegFileCyclic:
    def test_no_wrap(self):
        rf = RegFile()
        data = bytearray(range(128))
        rf.set_r_cyclic_at(0, data)
        assert rf.get_r_cyclic_at(0, 128) == data

    def test_wrap_around(self):
        """Write 128 bytes starting at index 500 — should wrap around the 512-byte buffer."""
        rf = RegFile()
        data = bytearray(range(128))
        rf.set_r_cyclic_at(500, data)

        got = rf.get_r_cyclic_at(500, 128)
        assert got == data

    def test_modulo_index(self):
        """Index 512 should be equivalent to index 0."""
        rf = RegFile()
        data = bytearray([0xCC] * 128)
        rf.set_r_cyclic_at(512, data)
        assert rf.get_r_cyclic_at(0, 128) == data

    def test_read_wraps_correctly(self):
        """Fill entire cyclic buffer, read 128 from near the end."""
        rf = RegFile()
        full = bytearray(range(256)) * 2  # 512 bytes
        rf.set_r_cyclic_at(0, full)

        # Read 128 bytes starting at 450 — wraps at 512
        got = rf.get_r_cyclic_at(450)
        expected = full[450:] + full[: 128 - (512 - 450)]
        assert got == expected


# ---------------------------------------------------------------------------
# RegFile — R mask
# ---------------------------------------------------------------------------

class TestRegFileMask:
    def test_set_get_mask(self):
        rf = RegFile()
        data = bytearray([0xFF] * 128)
        rf.set_r_mask(data)
        assert rf.get_r_mask() == data

    def test_mask_wrong_size(self):
        rf = RegFile()
        with pytest.raises(AssertionError):
            rf.set_r_mask(bytearray(64))


# ---------------------------------------------------------------------------
# RegFile — Accumulator
# ---------------------------------------------------------------------------

class TestRegFileAcc:
    def test_acc_bytes(self):
        rf = RegFile()
        data = bytearray([0x01] * 512)
        rf.set_r_acc_bytes(data)
        assert rf.get_r_acc_bytes() == data

    def test_acc_word_access(self):
        rf = RegFile()
        rf.set_r_acc_word(0, 0xDEADBEEF)
        assert rf.get_r_acc_word(0) == 0xDEADBEEF

    def test_acc_word_view_numpy(self):
        rf = RegFile()
        rf.set_r_acc_word(5, 42)
        words = rf.get_r_acc_words()
        assert words[5] == 42
        assert len(words) == 128  # 512 / 4

    def test_acc_returns_copy(self):
        rf = RegFile()
        rf.set_r_acc_word(0, 100)
        data = rf.get_r_acc_bytes()
        data[0] = 0
        assert rf.get_r_acc_word(0) == 100


# ---------------------------------------------------------------------------
# RegFile — Misc forwarding registers
# ---------------------------------------------------------------------------

class TestRegFileMisc:
    def test_mult_res(self):
        rf = RegFile()
        data = bytearray([0xAB] * 512)
        rf.set_mult_res_bytes(data)
        assert rf.get_mult_res_bytes() == data

    def test_mem_bypass(self):
        rf = RegFile()
        data = bytearray([0xCD] * 128)
        rf.set_mem_bypass(data)
        assert rf.get_mem_bypass() == data


# ---------------------------------------------------------------------------
# RegFile — Generic accessor + aliases
# ---------------------------------------------------------------------------

class TestRegFileGeneric:
    def test_alias_r0(self):
        rf = RegFile()
        data = bytearray([0x11] * 128)
        rf.set_r(0, data)
        # "r0" is an alias for "r" index 0
        got = rf.get_register_bytes("r0", 0)
        assert got == data

    def test_alias_acc(self):
        rf = RegFile()
        data = bytearray([0x22] * 512)
        rf.set_r_acc_bytes(data)
        got = rf.get_register_bytes("acc")
        assert got == data


# ---------------------------------------------------------------------------
# RegFile — Snapshot (deep copy)
# ---------------------------------------------------------------------------

class TestRegFileSnapshot:
    def test_snapshot_is_independent(self):
        rf = RegFile()
        rf.set_lr(0, 42)
        snap = rf.snapshot()
        rf.set_lr(0, 999)

        assert snap.get_lr(0) == 42
        assert rf.get_lr(0) == 999

    def test_snapshot_preserves_r_regs(self):
        rf = RegFile()
        data = bytearray([0xEE] * 128)
        rf.set_r(0, data)
        snap = rf.snapshot()
        rf.set_r(0, bytearray(128))

        assert snap.get_r(0) == data
        assert rf.get_r(0) == bytearray(128)


# ---------------------------------------------------------------------------
# RegFile — Serialisation
# ---------------------------------------------------------------------------

class TestRegFileSerialization:
    def test_to_dict_has_all_keys(self):
        rf = RegFile()
        d = rf.to_dict()
        for desc in REGFILE_SCHEMA:
            assert desc.name in d

    def test_to_dict_lr_values(self):
        rf = RegFile()
        for i in range(16):
            rf.set_lr(i, i)
        d = rf.to_dict()
        assert d["lr"] == list(range(16))


# ---------------------------------------------------------------------------
# IpuState
# ---------------------------------------------------------------------------

class TestIpuState:
    def test_init(self):
        state = IpuState()
        assert state.program_counter == 0
        assert not state.is_halted

    def test_halted_at_end(self):
        state = IpuState()
        state.program_counter = 1024
        assert state.is_halted

    def test_cr_dtype(self):
        state = IpuState()
        state.set_cr_dtype(2)
        assert state.get_cr_dtype() == 2

    def test_load_store_r_reg_xmem(self):
        """Load data into XMEM, then load into R register, verify."""
        state = IpuState()
        data = bytearray(range(128))
        state.xmem.write_address(256, data)
        state.load_r_reg_from_xmem(256, 0)
        assert state.regfile.get_r(0) == data

    def test_store_r_reg_to_xmem(self):
        state = IpuState()
        data = bytearray([0xAB] * 128)
        state.regfile.set_r(1, data)
        state.store_r_reg_to_xmem(512, 1)
        assert state.xmem.read_address(512, 128) == data

    def test_snapshot_regfile(self):
        state = IpuState()
        state.regfile.set_lr(3, 77)
        snap = state.snapshot_regfile()
        state.regfile.set_lr(3, 0)
        assert snap.get_lr(3) == 77

    def test_to_dict(self):
        state = IpuState()
        state.regfile.set_lr(0, 10)
        d = state.to_dict()
        assert d["program_counter"] == 0
        assert d["regfile"]["lr"][0] == 10
