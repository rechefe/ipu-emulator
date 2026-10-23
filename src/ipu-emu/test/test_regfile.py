"""Tests for ipu_emu.regfile and ipu_emu.ipu_state."""

import os
import struct
import subprocess
import sys

import numpy as np
import pytest
from emu_support import (
    ElementFormat,
    load_r_cyclic_from_xmem,
    load_r_mask_from_xmem,
    load_r_reg_from_xmem,
    new_state,
    store_acc_to_xmem,
    store_r_reg_to_xmem,
)

from ipu_emu.descriptors import REGFILE_SCHEMA
from ipu_emu.errors import (
    EmulatorError,
    RegisterIndexError,
    RegisterKindError,
    RegisterSizeError,
)
from ipu_emu.ipu_config import DEFAULT_DSTRUCTURE_CR
from ipu_emu.ipu_state import IpuState
from ipu_emu.regfile import RegFile
from ipu_isa.constants import (
    DSTRUCTURE_VALID_ELEMENTS_MASK,
    PadMode,
    decode_dstructure,
    encode_dstructure,
)

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
        rf.set_cr(2, 0xCAFE)  # CR0/CR1 are locked
        assert rf.get_cr(2) == 0xCAFE

    def test_cr0_is_permanently_zero(self):
        rf = RegFile()
        with pytest.raises(EmulatorError, match="CR0 is read-only"):
            rf.set_cr(0, 0xFFFFFF)
        assert rf.get_cr(0) == 0

    def test_cr1_is_permanently_one(self):
        rf = RegFile()
        with pytest.raises(EmulatorError, match="CR1 is read-only"):
            rf.set_cr(1, 0xFFFFFF)
        assert rf.get_cr(1) == 1

    def test_cr_32bit_mask(self):
        rf = RegFile()
        rf.set_cr(3, (1 << 33) | 0xDEADBEEF)  # 34 bits — should be masked to 32 bits
        assert rf.get_cr(3) == 0xDEADBEEF

    def test_lr_overflow_wraps_to_32bit(self):
        rf = RegFile()
        rf.set_lr(0, (1 << 33) | 0xDEADBEEF)  # > 32 bits
        assert rf.get_lr(0) == 0xDEADBEEF  # only low 32 bits kept

    def test_lr_index_out_of_range(self):
        rf = RegFile()
        with pytest.raises(RegisterIndexError):
            rf.set_lr(16, 0)

    def test_lr_negative_index_rejected(self):
        rf = RegFile()
        with pytest.raises(RegisterIndexError, match=r"\[-1\] out of range"):
            rf.set_lr(-1, 0)
        with pytest.raises(RegisterIndexError):
            rf.get_lr(-1)
        assert rf.get_lr(15) == 0  # nothing written to the last LR

    def test_scalar_accessor_on_vector_register(self):
        rf = RegFile()
        with pytest.raises(RegisterKindError, match="not a scalar register"):
            rf.get_scalar("r", 0)
        with pytest.raises(RegisterKindError):
            rf.set_scalar("r", 0, 1)

    def test_register_errors_keep_builtin_bases(self):
        assert issubclass(RegisterIndexError, IndexError)
        assert issubclass(RegisterSizeError, ValueError)
        assert issubclass(RegisterKindError, TypeError)
        for cls in (RegisterIndexError, RegisterSizeError, RegisterKindError):
            assert issubclass(cls, EmulatorError)


class TestRegFileOptimizedMode:
    """Validation must survive ``python -O`` (which strips ``assert``)."""

    def test_checks_hold_under_python_O(self):
        script = "import sys\nassert False, 'asserts are enabled'\n"
        # Sanity: the child really runs with asserts stripped.
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        probe = subprocess.run(
            [sys.executable, "-O", "-c", script], env=env, capture_output=True, text=True
        )
        assert probe.returncode == 0, probe.stderr

        script = """
import sys

from ipu_emu.errors import RegisterIndexError, RegisterSizeError
from ipu_emu.regfile import RegFile

rf = RegFile()
checks = [
    (RegisterIndexError, lambda: rf.set_lr(-1, 0)),
    (RegisterIndexError, lambda: rf.get_cr(999)),
    (RegisterIndexError, lambda: rf.get_r(-1)),
    (RegisterIndexError, lambda: rf.set_r_acc_word(-1, 0)),
    (RegisterSizeError, lambda: rf.set_r(0, bytearray(64))),
    (RegisterSizeError, lambda: rf.set_r_mask(bytearray(64))),
]
for expected, call in checks:
    try:
        call()
    except expected:
        continue
    sys.exit(f"no {expected.__name__} raised")
print("ok")
"""
        result = subprocess.run(
            [sys.executable, "-O", "-c", script], env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip() == "ok"


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

    def test_r_wrong_size_rejected(self):
        rf = RegFile()
        with pytest.raises(RegisterSizeError, match="expected 128 bytes, got 64"):
            rf.set_r(0, bytearray(64))  # too small
        with pytest.raises(RegisterSizeError):
            rf.set_r(0, bytearray(129))  # too large
        assert len(rf.get_r(0)) == 128
        assert len(rf.view("r")) == 256  # storage not resized

    def test_r_index_out_of_range(self):
        rf = RegFile()
        with pytest.raises(RegisterIndexError):
            rf.get_r(2)

    def test_r_negative_index_rejected(self):
        rf = RegFile()
        with pytest.raises(RegisterIndexError):
            rf.get_r(-1)
        with pytest.raises(RegisterIndexError):
            rf.set_r(-1, bytearray([0xFF] * 128))
        assert rf.get_r(1) == bytearray(128)


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
        """Index 512 should be equivalent to index 0 -- r_cyclic is allocated
        exactly 512 B (narrow-only; wide-vector debug mode uses the separate
        r_cyclic_wide_debug register, #180), so the default wrap (the full
        allocation) already lands on the 512-byte boundary with no explicit
        wrap_size needed."""
        rf = RegFile()
        data = bytearray([0xCC] * 128)
        rf.set_r_cyclic_at(512, data)
        assert rf.get_r_cyclic_at(0, 128) == data

    def test_read_wraps_correctly(self):
        """Fill the 512-byte cyclic buffer, read 128 from near the wrap boundary."""
        rf = RegFile()
        full = bytearray(range(256)) * 2  # 512 bytes
        rf.set_r_cyclic_at(0, full)

        # Read 128 bytes starting at 450 — wraps at 512
        got = rf.get_r_cyclic_at(450, 128)
        expected = full[450:] + full[: 128 - (512 - 450)]
        assert got == expected

    def test_default_wrap_is_full_allocation(self):
        """With no wrap_size argument, RegFile wraps at the full 512 B allocation --
        it has no concept of 'mode'; r_cyclic itself is narrow-only storage."""
        rf = RegFile()
        data = bytearray([0xAB] * 128)
        rf.set_r_cyclic_at(512, data)  # no wrap_size -> defaults to 512 (full allocation)
        assert rf.get_r_cyclic_at(0, 128) == data


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
        with pytest.raises(RegisterSizeError, match="expected 128 bytes, got 64"):
            rf.set_r_mask(bytearray(64))
        with pytest.raises(RegisterSizeError):
            rf.set_r_mask(bytearray(256))
        assert len(rf.get_r_mask()) == 128  # storage not resized


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

    def test_acc_word_index_rejected(self):
        rf = RegFile()
        n_words = len(rf.get_r_acc_bytes()) // 4
        for index in (-1, n_words):
            with pytest.raises(RegisterIndexError, match="word"):
                rf.set_r_acc_word(index, 0)
            with pytest.raises(RegisterIndexError):
                rf.get_r_acc_word(index)
        assert rf.get_r_acc_word(n_words - 1) == 0

    def test_acc_bytes_wrong_size_rejected(self):
        rf = RegFile()
        size = len(rf.get_r_acc_bytes())
        with pytest.raises(RegisterSizeError):
            rf.set_r_acc_bytes(bytearray(size - 4))
        assert len(rf.get_r_acc_bytes()) == size

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


    def test_snapshot_and_copy_into_carry_the_metadata_registers(self):
        # ADR 0007: R0/R1 and the four R_CYCLIC slots each hold a 16-bit
        # metadata word; the AaQ LUT holds 4352 table + 46 metadata bits.
        rf = RegFile()
        rf.set_r_meta(1, b"\x7f\x01")
        rf.set_r_cyclic_meta(3, b"\x05\x01")
        rf.set_aaq_lut(bytes(range(256)) * 2 + bytes(38))
        snap = rf.snapshot()
        rf.set_r_meta(1, bytes(2))
        assert snap.get_r_meta(1) == b"\x7f\x01" and rf.get_r_meta(1) == bytes(2)
        assert snap.get_r_cyclic_meta(3) == b"\x05\x01"
        assert snap.get_aaq_lut()[:256] == bytes(range(256))
        fresh = RegFile().snapshot()
        snap.copy_into(fresh)
        assert fresh.get_r_meta(1) == b"\x7f\x01" and fresh.get_aaq_lut() == snap.get_aaq_lut()

    def test_metadata_registers_reset_to_zero_with_their_sizes(self):
        rf = RegFile()
        assert [rf.get_r_meta(i) for i in range(2)] == [bytes(2)] * 2
        assert [rf.get_r_cyclic_meta(i) for i in range(4)] == [bytes(2)] * 4
        assert rf.get_aaq_lut() == bytes((4352 + 46 + 7) // 8)
        with pytest.raises(RegisterIndexError):
            rf.get_r_cyclic_meta(4)
        with pytest.raises(RegisterSizeError):
            rf.set_r_meta(0, b"\x00")


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

    def test_format_attribute(self):
        from ipu_isa.formats import ElementFormat

        assert IpuState().fmt is ElementFormat.INT8  # default
        assert IpuState(fmt=ElementFormat.E4).fmt is ElementFormat.E4

    def test_cr15_default_dstructure(self):
        state = IpuState()
        valid_elements, partition = state.get_dstructure_for(DEFAULT_DSTRUCTURE_CR)
        assert valid_elements == 128
        assert partition == 0

    def test_set_cr_dstructure(self):
        state = IpuState()
        state.set_cr_dstructure(64, 2)
        config = state.get_dstructure_for(DEFAULT_DSTRUCTURE_CR)
        assert config.valid_elements == 64
        assert config.partition == 2

    def test_set_cr_dstructure_accepts_cr_idx(self):
        state = IpuState()
        cr15_before = state.regfile.get_cr(DEFAULT_DSTRUCTURE_CR)

        state.set_cr_dstructure(7, 4, cr_idx=3)

        config = state.get_dstructure_for(3)
        assert config.valid_elements == 7
        assert config.partition == 4
        assert state.regfile.get_cr(DEFAULT_DSTRUCTURE_CR) == cr15_before

    def test_cr_dstructure_masks_valid_elements(self):
        state = IpuState()
        state.set_cr_dstructure(valid_elements=DSTRUCTURE_VALID_ELEMENTS_MASK + 1)
        assert state.get_dstructure_for(DEFAULT_DSTRUCTURE_CR).valid_elements == 0

    def test_cr_dstructure_invalid_partition_raises(self):
        with pytest.raises(ValueError, match="not a valid Partition"):
            encode_dstructure(valid_elements=64, partition=3)

    def test_dstructure_codec_round_trip(self):
        raw = encode_dstructure(valid_elements=17, partition=4)
        decoded = decode_dstructure(raw)
        assert decoded.valid_elements == 17
        assert decoded.partition == 4

    def test_dstructure_pad_mode_defaults_to_zero(self):
        assert IpuState().get_dstructure_for(DEFAULT_DSTRUCTURE_CR).pad_mode == PadMode.ZERO

    def test_dstructure_pad_mode_codec_round_trip(self):
        raw = encode_dstructure(valid_elements=17, partition=4, pad_mode=PadMode.NEG_INF)
        decoded = decode_dstructure(raw)
        assert decoded.valid_elements == 17
        assert decoded.partition == 4
        assert decoded.pad_mode == PadMode.NEG_INF

    def test_dstructure_invalid_pad_mode_raises(self):
        with pytest.raises(ValueError, match="not a valid PadMode"):
            encode_dstructure(valid_elements=64, partition=0, pad_mode=3)

    def test_set_cr_dstructure_accepts_pad_mode(self):
        state = IpuState()
        state.set_cr_dstructure(valid_elements=64, partition=2, pad_mode=PadMode.POS_INF)
        config = state.get_dstructure_for(DEFAULT_DSTRUCTURE_CR)
        assert config.valid_elements == 64
        assert config.partition == 2
        assert config.pad_mode == PadMode.POS_INF

    def test_cr_dstructure_uses_config_register(self):
        state = IpuState()
        state.set_cr_dstructure(7, 4)
        assert state.regfile.get_cr(DEFAULT_DSTRUCTURE_CR) == encode_dstructure(
            valid_elements=7,
            partition=4,
        )

    def test_load_store_r_reg_xmem(self):
        """Load data into XMEM, then load into R register, verify.

        load_r_reg_from_xmem takes a ROW number (narrow row = 128 B), matching
        LD.MULT's addressing (#179): row 2 = byte 256.
        """
        state = IpuState()
        data = bytearray(range(128))
        state.xmem.write_address(256, data)
        load_r_reg_from_xmem(state, 2, 0)
        assert state.regfile.get_r(0) == data

    def test_store_r_reg_to_xmem(self):
        """store_r_reg_to_xmem takes a ROW number: row 4 = byte 512 (narrow)."""
        state = IpuState()
        data = bytearray([0xAB] * 128)
        state.regfile.set_r(1, data)
        store_r_reg_to_xmem(state, 4, 1)
        assert state.xmem.read_address(512, 128) == data

    def test_load_r_reg_from_xmem_debug_mode_row_size(self):
        """In debug mode a row is 512 B, not 128, and lands in r_wide_debug --
        the register LD.MULT loads in that mode -- not the 128-byte r."""
        state = new_state(fmt=ElementFormat.FP32)
        data = struct.pack("<128f", *([2.5] * 128))
        state.xmem.write_address(3 * 512, data)  # row 3, debug row size = 512 B
        load_r_reg_from_xmem(state, 3, 0)
        assert state.regfile.get_r_wide_debug(0) == bytearray(data)

    def test_store_r_reg_from_xmem_debug_mode_round_trip(self):
        """store_r_reg_to_xmem in debug mode reads back from r_wide_debug,
        mirroring the load helper above."""
        state = new_state(fmt=ElementFormat.INT32)
        state.regfile.set_r_wide_debug(1, struct.pack("<128i", *([7] * 128)))
        store_r_reg_to_xmem(state, 2, 1)  # row 2, debug row size = 512 B -> byte 1024
        stored = state.xmem.read_address(2 * 512, 512)
        assert struct.unpack("<128i", stored) == tuple([7] * 128)

    def test_load_r_cyclic_from_xmem_defaults_to_slot_0(self):
        state = IpuState()
        data = bytearray(range(128))
        state.xmem.write_address(256, data)  # row 2, narrow row size = 128 B
        load_r_cyclic_from_xmem(state, 2)
        assert state.regfile.get_r_cyclic_at(0, 128, wrap_size=512) == data

    def test_load_r_cyclic_from_xmem_targets_nonzero_slot(self):
        """slot_element_idx lets the caller target slots 1-3, not just slot 0 --
        the capability gap #180/#181 exist to fix."""
        state = IpuState()
        data = bytearray(range(128))
        state.xmem.write_address(256, data)
        load_r_cyclic_from_xmem(state, 2, slot_element_idx=256)
        written = state.regfile.get_r_cyclic_at(256, 128, wrap_size=512)
        assert written == data
        # slot 0 must be untouched
        assert state.regfile.get_r_cyclic_at(0, 128, wrap_size=512) == bytearray(128)

    def test_load_r_cyclic_from_xmem_debug_mode(self):
        """In debug mode, slot_element_idx=128 writes at byte 512 (4 B/element)."""
        state = new_state(fmt=ElementFormat.FP32)
        data = struct.pack("<128f", *([1.5] * 128))
        state.xmem.write_address(1 * 512, data)  # row 1, debug row size = 512 B
        load_r_cyclic_from_xmem(state, 1, slot_element_idx=128)
        written = state.regfile.get_r_cyclic_wide_debug_at(128 * 4, 512)  # byte 512
        assert written == bytearray(data)

    def test_store_acc_to_xmem_row_addressed(self):
        state = IpuState()
        data = bytearray([0x07] * 512)
        state.regfile.set_r_acc_bytes(data)
        store_acc_to_xmem(state, 5)  # row 5, narrow row size = 128 B -> byte 640
        assert state.xmem.read_address(640, 512) == data

    def test_load_r_mask_from_xmem_row_addressed(self):
        state = IpuState()
        mask_data = bytearray(range(128))
        state.xmem.write_address(384, mask_data)  # row 3, narrow row size = 128 B
        load_r_mask_from_xmem(state, 3)
        assert state.regfile.get_r_mask() == mask_data

    def test_load_r_mask_from_xmem_debug_mode_still_reads_128_bytes(self):
        """The mask is 1 bit/lane and does not scale with element width -- the
        row ADDRESS is translated by the debug row size (512 B), but the READ
        stays 128 bytes in both modes."""
        state = new_state(fmt=ElementFormat.FP32)
        mask_data = bytearray(range(128))
        state.xmem.write_address(2 * 512, mask_data)  # row 2, debug row size = 512 B
        load_r_mask_from_xmem(state, 2)
        assert state.regfile.get_r_mask() == mask_data

    def test_post_aaq_reg_debug_alias_aaq_result(self):
        """Legacy debug name ``aaq_result`` resolves to ``post_aaq_reg``."""
        state = IpuState()
        data = bytearray((i * 3) & 0xFF for i in range(512))
        state.regfile.set_post_aaq_reg(data)
        assert state.regfile.get_register_bytes("aaq_result") == bytearray(data)

    def test_regfile_snapshot(self):
        state = IpuState()
        state.regfile.set_lr(3, 77)
        snap = state.regfile.snapshot()
        state.regfile.set_lr(3, 0)
        assert snap.get_lr(3) == 77

    def test_to_dict(self):
        state = IpuState()
        state.regfile.set_lr(0, 10)
        d = state.to_dict()
        assert d["program_counter"] == 0
        assert d["regfile"]["lr"][0] == 10


def test_regfile_observers_register_and_detach():
    rf = RegFile()
    first, second = [], []
    remove_first = rf.add_observer(lambda *event: first.append(event))
    rf.add_observer(lambda *event: second.append(event))
    rf.mutable("r_acc")
    remove_first()
    remove_first()  # idempotent
    rf.mutable("mult_res")
    assert [name for name, _, _ in first] == ["r_acc"]
    assert [name for name, _, _ in second] == ["r_acc", "mult_res"]


def test_a_scalar_write_is_observed():
    rf = RegFile()
    events = []
    rf.add_observer(lambda *event: events.append(event))
    rf.set_lr(3, 7)
    rf.set_cr(4, 9)
    assert events == [("lr", 12, 4), ("cr", 16, 4)]


def test_regfile_snapshot_does_not_inherit_observers():
    rf = RegFile()
    events = []
    rf.add_observer(lambda *event: events.append(event))
    rf.snapshot().mutable("r_acc")
    assert events == []


class TestReadAndWriteViews:
    """``view`` reads without side effects; ``mutable`` reports exactly what it hands out."""

    def test_view_tells_no_observer_and_keeps_provenance(self):
        rf = RegFile()
        rf.mark_constant_ones("r", 0, 128)
        events = []
        rf.add_observer(lambda *event: events.append(event))
        assert bytes(rf.view("r")[:4]) == bytes(4)
        assert events == []
        assert rf.has_constant_ones("r", 0, 128)

    def test_writing_through_a_view_raises(self):
        rf = RegFile()
        with pytest.raises(TypeError):
            rf.view("r_acc")[0] = 1
        assert not rf.get_r_acc_words().flags.writeable

    def test_mutable_reports_and_forgets_exactly_its_range(self):
        rf = RegFile()
        rf.mark_constant_ones("r", 0, 128)
        events = []
        rf.add_observer(lambda *event: events.append(event))
        rf.mutable("r", 16, 8)[:] = bytes(range(8))
        assert events == [("r", 16, 8)]
        assert rf.view("r")[16:24] == bytes(range(8))
        assert not rf.has_constant_ones("r", 16, 8)
        assert rf.has_constant_ones("r", 0, 16) and rf.has_constant_ones("r", 24, 104)
        # Unlike raw(), it does not make the register untrustworthy for good.
        rf.mark_constant_ones("r", 16, 8)
        assert rf.has_constant_ones("r", 0, 128)

    def test_mutable_range_is_checked(self):
        rf = RegFile()
        with pytest.raises(RegisterIndexError):
            rf.mutable("r_acc", 508, 8)

    def test_set_words_writes_the_whole_register(self):
        rf = RegFile()
        rf.set_r_acc_words(np.arange(128, dtype=np.uint32))
        assert rf.get_r_acc_word(5) == 5
        with pytest.raises(RegisterSizeError):
            rf.set_r_acc_words(np.arange(3, dtype=np.uint32))

    def test_descriptor_resolves_aliases(self):
        rf = RegFile()
        assert rf.descriptor("r_acc").name == "r_acc"
        assert rf.descriptor("r0").name == "r"

    def test_raw_is_deprecated_and_still_escapes(self):
        rf = RegFile()
        with pytest.deprecated_call():
            buffer = rf.raw("r")
        buffer[0] = 7
        assert rf.view("r")[0] == 7
        rf.mark_constant_ones("r", 0, 128)
        assert not rf.has_constant_ones("r", 0, 128), "an escaped buffer is never trusted again"
