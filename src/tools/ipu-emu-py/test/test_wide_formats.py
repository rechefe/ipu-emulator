"""The 4-byte element formats FP32 and INT32 (emulator-only; GitHub issue #33).

XMEM transfer sizes stay architectural (128-byte loads for r in an 8-bit
format); in FP32 and INT32 ``LD.MULT`` / ``LD.MULT.CYCLIC`` consume 512 bytes
per transfer as 128×32-bit elements. LR/CR are unchanged.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from emu_support import ElementFormat, assemble_program, make_state, new_state, run

from ipu_emu.activations import AaqOptions
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.geometry import (
    element_width_bytes,
    row_size_bytes,
    xmem_element_width_bytes,
    xmem_row_size_bytes,
)
from ipu_emu.ipu import LANES, EmulatorError, Ipu
from ipu_emu.ipu_state import IpuState


class TestWideVectorFp32:
    def test_mult_ee_fp32_no_quantization_in_acc(self) -> None:
        """128 float elements multiply element-wise; acc holds float products."""
        r0 = struct.pack("<128f", *([2.0] * 128))
        rc = struct.pack("<128f", *([3.0] * 128))
        state = new_state(fmt=ElementFormat.FP32)
        state.xmem.write_address(0x1000, r0)
        state.xmem.write_address(0x2000, rc)
        asm = """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LD.MULT r0 lr0 cr0;;
LD.MULT.CYCLIC lr1 cr0 lr2;;
MULT.RC.VV lr2 r0 0 lr2 cr15;;
acc.add.first;;
HALT;;
"""
        state.regfile.set_cr(6, 0x1000 // 512)  # row number (debug row = 512 B)
        state.regfile.set_cr(7, 0x2000 // 512)
        state.regfile.set_cr(8, 0)
        load_program(state, assemble_program(asm))
        run_until_complete(state)
        acc = state.regfile.mutable("r_acc")
        for i in range(128):
            v = struct.unpack_from("<f", acc, i * 4)[0]
            assert v == pytest.approx(6.0), f"lane {i}"

    def test_activate_quantize_noop_unless_quantize_flag(self) -> None:
        """ACTIVATE.QUANTIZE without AaqOptions.quantize_output leaves POST_AAQ_REG all zero."""
        state = run(
            """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LD.MULT r0 lr0 cr0;;
LD.MULT.CYCLIC lr1 cr0 lr2;;
MULT.RC.VV lr2 r0 0 lr2 cr15;;
acc.add.first;;
SET lr0 cr9;;
ACTIVATE.QUANTIZE identity cr15;;
HALT;;
""",
            cr={6: 0x1000 // 512, 7: 0x2000 // 512, 8: 0, 9: 128},
            fmt=ElementFormat.FP32,
        )
        assert state.regfile.get_post_aaq_reg() == bytearray(512)

    def test_activate_quantize_fp32_acc_for_comparison(self) -> None:
        """ACTIVATE.QUANTIZE with AaqOptions.quantize_output quantizes FP32 acc elements to INT8."""
        # Use exact float product 3.0 so rounding is unambiguous (round-half-even).
        r0 = struct.pack("<128f", *([1.5] * 128))
        rc = struct.pack("<128f", *([2.0] * 128))
        state = new_state(fmt=ElementFormat.FP32, aaq=AaqOptions(quantize_output=True))
        state.xmem.write_address(0x1000, r0)
        state.xmem.write_address(0x2000, rc)
        asm = """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LD.MULT r0 lr0 cr0;;
LD.MULT.CYCLIC lr1 cr0 lr2;;
MULT.RC.VV lr2 r0 0 lr2 cr15;;
acc.add.first;;
SET lr0 cr9;;
ACTIVATE.QUANTIZE identity cr15;;
HALT;;
"""
        state.regfile.set_cr(6, 0x1000 // 512)  # row number (debug row = 512 B)
        state.regfile.set_cr(7, 0x2000 // 512)
        state.regfile.set_cr(8, 0)
        state.regfile.set_cr(9, 128)
        load_program(state, assemble_program(asm))
        run_until_complete(state)
        out = state.regfile.get_post_aaq_reg()
        assert all(b == 3 for b in out[:128])
        assert out[128:] == bytearray(384)


class TestWideVectorInt32:
    def test_mult_ee_int32_wrap_multiply(self) -> None:
        r0 = struct.pack("<128i", *([70000] * 128))
        rc = struct.pack("<128i", *([70000] * 128))
        state = new_state(fmt=ElementFormat.INT32)
        state.xmem.write_address(0x1000, r0)
        state.xmem.write_address(0x2000, rc)
        asm = """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LD.MULT r0 lr0 cr0;;
LD.MULT.CYCLIC lr1 cr0 lr2;;
MULT.RC.VV lr2 r0 0 lr2 cr15;;
acc.add.first;;
HALT;;
"""
        state.regfile.set_cr(6, 0x1000 // 512)  # row number (debug row = 512 B)
        state.regfile.set_cr(7, 0x2000 // 512)
        state.regfile.set_cr(8, 0)
        load_program(state, assemble_program(asm))
        run_until_complete(state)
        acc = state.regfile.mutable("r_acc")
        expected = (70000 * 70000) & 0xFFFFFFFF
        if expected >= 0x80000000:
            expected -= 0x100000000
        for i in range(128):
            v = struct.unpack_from("<i", acc, i * 4)[0]
            assert v == expected, f"lane {i}"


class TestWideVectorR0R1Isolation:
    """Encoding 0 (R0) and 1 (R1) must not share debug staging storage."""

    def test_r0_and_r1_independent(self) -> None:
        r0 = struct.pack("<128f", *([1.0] * 128))
        r1 = struct.pack("<128f", *([99.0] * 128))
        rc = struct.pack("<128f", *([2.0] * 128))

        def run_mult(which: str) -> float:
            # Debug-mode rows are 512 B; r0/r1 need distinct rows (0 and 1),
            # rc a third (row 2) -- one row-size apart, not the old 256 B gap.
            st = new_state(fmt=ElementFormat.FP32)
            st.xmem.write_address(0 * 512, r0)
            st.xmem.write_address(1 * 512, r1)
            st.xmem.write_address(2 * 512, rc)
            st.regfile.set_cr(6, 0)
            st.regfile.set_cr(7, 1)
            st.regfile.set_cr(8, 2)
            st.regfile.set_cr(9, 0)
            asm = f"""\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
SET lr3 cr9;;
LD.MULT r0 lr0 cr0;;
LD.MULT r1 lr1 cr0;;
LD.MULT.CYCLIC lr2 cr0 lr3;;
MULT.RC.VV lr3 {which} 0 lr3 cr15;;
acc.add.first;;
HALT;;
"""
            load_program(st, assemble_program(asm))
            run_until_complete(st)
            return struct.unpack_from("<f", st.regfile.view("r_acc"), 0)[0]

        assert run_mult("r0") == pytest.approx(2.0)
        assert run_mult("r1") == pytest.approx(198.0)


class TestWideVectorCyclicIndex:
    """``LD.MULT.CYCLIC`` writes must land on one of the four element-index
    slot boundaries (0/128/256/384) in wide-vector debug mode too -- writes only
    ever replace a whole slot, in either mode. A non-boundary index still raises;
    it is no longer restricted to index=0 (r_cyclic's 4-slot ring exists in debug
    mode exactly as it does in narrow mode -- see #180)."""

    def test_ldr_cyclic_non_boundary_index_raises_fp32(self) -> None:
        twos = struct.pack("<128f", *([2.0] * 128))
        st = new_state(fmt=ElementFormat.FP32)
        st.xmem.write_address(0x1000, twos)
        st.regfile.set_cr(6, 0x1000)
        st.regfile.set_cr(7, 1)  # not a slot boundary (0/128/256/384)
        asm = """\
SET lr0 cr6;;
SET lr1 cr7;;
LD.MULT.CYCLIC lr0 cr0 lr1;;
HALT;;
"""
        load_program(st, assemble_program(asm))
        with pytest.raises(EmulatorError, match="index must be one of"):
            run_until_complete(st)

    def test_ldr_cyclic_slot_boundary_index_succeeds_fp32(self) -> None:
        """index=128 (a valid slot boundary) writes to r_cyclic element 128, i.e.
        byte 512 in debug mode (element_width=4)."""
        twos = struct.pack("<128f", *([2.0] * 128))
        st = new_state(fmt=ElementFormat.FP32)
        st.xmem.write_address(0x1000, twos)
        st.regfile.set_cr(6, 0x1000 // 512)  # row number (debug row = 512 B)
        st.regfile.set_cr(7, 128)
        asm = """\
SET lr0 cr6;;
SET lr1 cr7;;
LD.MULT.CYCLIC lr0 cr0 lr1;;
HALT;;
"""
        load_program(st, assemble_program(asm))
        run_until_complete(st)
        written = st.regfile.get_r_cyclic_wide_debug_at(
            128 * 4, 512
        )  # byte 512, wrap_size default (2048)
        assert written == bytearray(twos)


class TestWideVectorWrap:
    """RC reads wrap after the 512th ELEMENT (byte 2048 in debug/FP32 -- 4 B/element),
    the same 512-element wrap as narrow mode, just at a different byte count (#180).
    Reads are allowed at any element index and may cross a slot boundary; only
    LD.MULT.CYCLIC's WRITES are restricted to the four slot boundaries."""

    def test_mult_rc_ve_fp32_cr_scalar_wraps(self) -> None:
        # rc_idx=480 elements (aligned to a 4-byte lane, not a slot boundary --
        # reads don't need one): lane 31 ends at element 511; lane 32 wraps to element 0.
        buf = bytearray(2048)
        for k in range(32):
            struct.pack_into("<f", buf, (480 + k) * 4, 3.0)
        for k in range(96):
            struct.pack_into("<f", buf, k * 4, 5.0)
        st = new_state(fmt=ElementFormat.FP32)
        st.regfile.set_cr(2, 2)  # low byte 2 → scalar 2.0 in wide FP32 path
        st.regfile.set_r_cyclic_wide_debug_at(0, buf)
        st.regfile.set_cr(6, 480)  # rc_idx is an ELEMENT index (issue #182 follow-up)
        st.regfile.set_cr(7, 0)
        asm = """\
SET lr0 cr6;;
SET lr2 cr7;;
MULT.RC.VE lr0 cr2 0 lr2 cr15;;
acc.add.first;;
HALT;;
"""
        load_program(st, assemble_program(asm))
        run_until_complete(st)
        mult_res = st.regfile.mutable("mult_res")
        for i in range(32):
            assert struct.unpack_from("<f", mult_res, i * 4)[0] == pytest.approx(6.0), f"lane {i}"
        for i in range(32, 128):
            assert struct.unpack_from("<f", mult_res, i * 4)[0] == pytest.approx(10.0), f"lane {i}"

    def test_mult_rc_ve_fp32_lr_scalar_wraps(self) -> None:
        """MULT.RC.VE (wide FP32, LR-encoded src): RC elements wrap after element 512."""
        buf = bytearray(2048)
        for k in range(32):
            struct.pack_into("<f", buf, (480 + k) * 4, 3.0)
        for k in range(96):
            struct.pack_into("<f", buf, k * 4, 5.0)
        st = new_state(fmt=ElementFormat.FP32)
        st.regfile.set_r_cyclic_wide_debug_at(0, buf)
        st.xmem.write_address(0x1000, struct.pack("<128f", *([2.0] * 128)))
        st.regfile.set_cr(10, 0x1000 // 512)  # row number (debug row = 512 B)
        st.regfile.set_cr(6, 480)  # rc_idx is an ELEMENT index (issue #182 follow-up)
        st.regfile.set_cr(7, 0)
        st.regfile.set_cr(8, 0)
        asm = """\
SET lr4 cr10;;
LD.MULT r0 lr4 cr0;;
SET lr0 cr6;;
SET lr2 cr7;;
SET lr3 cr8;;
MULT.RC.VE lr0 lr2 0 lr3 cr15;;
acc.add.first;;
HALT;;
"""
        load_program(st, assemble_program(asm))
        run_until_complete(st)
        mult_res = st.regfile.mutable("mult_res")
        for i in range(32):
            assert struct.unpack_from("<f", mult_res, i * 4)[0] == pytest.approx(6.0), f"lane {i}"
        for i in range(32, 128):
            assert struct.unpack_from("<f", mult_res, i * 4)[0] == pytest.approx(10.0), f"lane {i}"


class TestWideVectorRcIdxElementAddressing:
    """rc_idx (MULT.RC.* operand) is an ELEMENT index into r_cyclic, the same
    unit as LD.MULT.CYCLIC's index (issue #182 follow-up). Regression
    coverage for the softmax_rows_partial Pass 4 bug: a reverse-slide placement
    computed as (RING_ELEMENTS - k) must land at r_cyclic element 0 data, not
    at a stale/unwritten byte offset from treating rc_idx as a byte quantity
    against the wrong ring size."""

    def test_reverse_slide_wraps_to_element_zero(self) -> None:
        """rc_idx = RING_ELEMENTS - k, read k elements later, wraps back to the
        real data written at element 0 -- the exact placement trick Pass 4 uses
        to reverse-slide partitions into place."""
        RING_ELEMENTS = 512
        k = 32
        buf = bytearray(2048)
        for i in range(128):
            struct.pack_into("<f", buf, i * 4, 9.0)
        st = new_state(fmt=ElementFormat.FP32)
        st.regfile.set_r_cyclic_wide_debug_at(0, buf)
        st.regfile.set_cr(6, RING_ELEMENTS - k)  # rc_idx in ELEMENTS
        asm = """\
SET lr0 cr6;;
MULT.RC.VE lr0 cr1 0 lr0 cr15;;
HALT;;
"""
        load_program(st, assemble_program(asm))
        run_until_complete(st)
        mult_res = st.regfile.mutable("mult_res")
        # Lanes [0, k) read the ring's stale tail (zero); lanes [k, 128) wrap
        # around to the real data written at element 0.
        for i in range(k):
            assert struct.unpack_from("<f", mult_res, i * 4)[0] == pytest.approx(0.0), f"lane {i}"
        for i in range(k, 128):
            assert struct.unpack_from("<f", mult_res, i * 4)[0] == pytest.approx(9.0), f"lane {i}"

    def test_rc_idx_zero_matches_element_offset_convention(self) -> None:
        """rc_idx=0 needs no special-casing relative to a wrapped rc_idx: both
        read element-indexed positions in the same ring, confirming rc_idx=0
        is not a degenerate/different unit from a nonzero rc_idx."""
        buf = bytearray(2048)
        for i in range(128):
            struct.pack_into("<f", buf, i * 4, 4.0)
        st = new_state(fmt=ElementFormat.FP32)
        st.regfile.set_r_cyclic_wide_debug_at(0, buf)
        st.regfile.set_cr(6, 0)
        asm = """\
SET lr0 cr6;;
MULT.RC.VE lr0 cr1 0 lr0 cr15;;
HALT;;
"""
        load_program(st, assemble_program(asm))
        run_until_complete(st)
        mult_res = st.regfile.mutable("mult_res")
        for i in range(128):
            assert struct.unpack_from("<f", mult_res, i * 4)[0] == pytest.approx(4.0), f"lane {i}"


class TestWideVectorAgg:
    """Wide-vector coverage for the ACC-slot AGG.* instructions."""

    def _run_agg(self, asm: str, fmt: ElementFormat, mult_res: bytearray) -> IpuState:
        def setup(st: IpuState) -> None:
            st.regfile.set_mult_res_bytes(mult_res)
            st.set_cr_dstructure(128)
            st.regfile.set_lr(0, 127)  # dest = r_acc lane 127

        return run(asm, fmt=fmt, setup=setup)

    def test_agg_sum_first_int32_wide(self) -> None:
        """AGG.SUM.FIRST with INT32 wide elements: 128×4 = 512 written to dest element."""
        mult_res = bytearray(512)
        struct.pack_into("<128i", mult_res, 0, *([4] * 128))
        st = self._run_agg("AGG.SUM.FIRST LR0 cr15;;\nHALT;;\n", ElementFormat.INT32, mult_res)
        raw = st.regfile.mutable("r_acc")
        assert struct.unpack_from("<i", raw, 127 * 4)[0] == 512

    def test_agg_sum_first_fp32_wide(self) -> None:
        """AGG.SUM.FIRST with FP32 wide elements: 128×0.5 = 64.0 written to dest element."""
        mult_res = bytearray(512)
        struct.pack_into("<128f", mult_res, 0, *([0.5] * 128))
        st = self._run_agg("AGG.SUM.FIRST LR0 cr15;;\nHALT;;\n", ElementFormat.FP32, mult_res)
        raw = st.regfile.mutable("r_acc")
        assert struct.unpack_from("<f", raw, 127 * 4)[0] == pytest.approx(64.0)

    def test_agg_sum_int32_wide_running(self) -> None:
        """AGG.SUM with INT32 wide elements adds onto the existing r_acc[dest] (issue #182)."""
        mult_res = bytearray(512)
        struct.pack_into("<128i", mult_res, 0, *([4] * 128))
        st = new_state(fmt=ElementFormat.INT32)
        st.regfile.set_mult_res_bytes(mult_res)
        st.set_cr_dstructure(128)
        st.regfile.set_lr(0, 127)  # dest = r_acc lane 127
        struct.pack_into("<i", st.regfile.mutable("r_acc"), 127 * 4, 1000)
        load_program(st, assemble_program("AGG.SUM LR0 cr15;;\nHALT;;\n"))
        run_until_complete(st)
        raw = st.regfile.mutable("r_acc")
        assert struct.unpack_from("<i", raw, 127 * 4)[0] == 1512

    def test_agg_max_first_fp32_wide(self) -> None:
        """AGG.MAX.FIRST with FP32 wide elements picks the largest element."""
        mult_res = bytearray(512)
        struct.pack_into("<128f", mult_res, 0, *([1.0] * 128))
        struct.pack_into("<f", mult_res, 3 * 4, 7.5)
        st = self._run_agg("AGG.MAX.FIRST LR0 cr15;;\nHALT;;\n", ElementFormat.FP32, mult_res)
        raw = st.regfile.mutable("r_acc")
        assert struct.unpack_from("<f", raw, 127 * 4)[0] == pytest.approx(7.5)


class TestWideVectorAlignment:
    def test_rc_idx_element_offset_always_lane_aligned(self) -> None:
        """rc_idx is an ELEMENT index (issue #182 follow-up); scaled to bytes via
        element_width_bytes(), so any rc_idx value produces a byte offset that is
        always a multiple of 4 in wide-vector debug mode -- misalignment via
        rc_idx is no longer reachable (unlike when rc_idx was a raw byte offset).
        The MULT handlers index R_CYCLIC in elements, so no byte guard is needed."""
        st = new_state(fmt=ElementFormat.FP32)
        st.xmem.write_address(0x1000, struct.pack("<128f", *([1.0] * 128)))
        st.xmem.write_address(0x2000, struct.pack("<128f", *([2.0] * 128)))
        st.regfile.set_cr(6, 0x1000 // 512)  # row number (debug row = 512 B)
        st.regfile.set_cr(7, 0x2000 // 512)
        st.regfile.set_cr(8, 0)
        st.regfile.set_cr(9, 1)  # rc_idx = 1 element -> byte offset 4, aligned
        asm = """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LD.MULT r0 lr0 cr0;;
LD.MULT.CYCLIC lr1 cr0 lr2;;
SET lr3 cr9;;
MULT.RC.VV lr3 r0 0 lr3 cr15;;
HALT;;
"""
        load_program(st, assemble_program(asm))
        run_until_complete(st)
        # An r_cyclic element index addresses whole 4-byte elements.
        assert xmem_element_width_bytes(st) % 4 == 0


class TestElementAndRowSizeHelpers:
    """``ipu_emu.geometry``: element bytes, the one primitive the two modes
    differ by (1 vs 4 bytes/element), and the row size (LANES elements) derived
    from it, by element format and by state.
    """

    @pytest.mark.parametrize(
        ("fmt", "element", "row"),
        [
            (ElementFormat.INT8, 1, 128),
            (ElementFormat.FP32, 4, 512),
            (ElementFormat.INT32, 4, 512),
        ],
    )
    def test_sizes(self, fmt: ElementFormat, element: int, row: int) -> None:
        st = new_state(fmt=fmt)
        assert xmem_element_width_bytes(st) == element_width_bytes(st.fmt) == element
        assert xmem_row_size_bytes(st) == row_size_bytes(st.fmt) == row

    def test_row_size_is_lanes_times_element_width(self) -> None:
        """Row size must track element width, not be independently hardcoded."""
        for fmt in (ElementFormat.INT8, ElementFormat.FP32, ElementFormat.INT32):
            st = new_state(fmt=fmt)
            assert xmem_row_size_bytes(st) == LANES * xmem_element_width_bytes(st)


class TestWideVectorCyclicFourSlotRing:
    """r_cyclic's 4-slot ring (element boundaries 0/128/256/384) exists in debug
    mode exactly as in narrow mode -- LD.MULT.CYCLIC writes one slot at a
    time; MULT.RC.* reads may start at any element index, including ones that
    cross a slot boundary (#180)."""

    def _write_slot(
        self, st: IpuState, *, slot_element_idx: int, xmem_byte_addr: int, fill: float
    ) -> None:
        st.xmem.write_address(xmem_byte_addr, struct.pack("<128f", *([fill] * 128)))
        st.regfile.set_cr(6, xmem_byte_addr // 512)  # row number (debug row = 512 B)
        st.regfile.set_cr(7, slot_element_idx)
        asm = """\
SET lr0 cr6;;
SET lr1 cr7;;
LD.MULT.CYCLIC lr0 cr0 lr1;;
HALT;;
"""
        load_program(st, assemble_program(asm))
        st.program_counter = 0  # reusing st across multiple runs -- rewind PC each time
        run_until_complete(st)

    def test_four_slots_written_independently(self) -> None:
        """Writing all four slots (0/128/256/384) with distinct fills leaves each
        readable independently -- confirms debug mode's ring has four live slots,
        not one register that always overwrites the same 512 B."""
        st = new_state(fmt=ElementFormat.FP32)
        fills = {0: 1.0, 128: 2.0, 256: 3.0, 384: 4.0}
        for slot_idx, fill in fills.items():
            self._write_slot(st, slot_element_idx=slot_idx, xmem_byte_addr=0x1000, fill=fill)

        for slot_idx, expected_fill in fills.items():
            byte_off = slot_idx * 4  # 4 B/element in debug mode
            data = st.regfile.get_r_cyclic_wide_debug_at(byte_off, 512)  # 128 elements * 4 B
            values = struct.unpack("<128f", data)
            assert all(v == pytest.approx(expected_fill) for v in values), (
                f"slot {slot_idx}: expected all {expected_fill}, got {values[:4]}..."
            )

    def test_read_crosses_slot_boundary(self) -> None:
        """A 128-element read starting at element 64 returns elements 64..191 --
        it crosses from slot 0 into slot 1 and does NOT wrap, per the design."""
        st = new_state(fmt=ElementFormat.FP32)
        self._write_slot(st, slot_element_idx=0, xmem_byte_addr=0x1000, fill=1.0)
        self._write_slot(st, slot_element_idx=128, xmem_byte_addr=0x2000, fill=2.0)

        # Read 128 elements starting at element 64 -> elements 64..127 (slot 0,
        # fill=1.0) then 128..191 (slot 1, fill=2.0).
        data = st.regfile.get_r_cyclic_wide_debug_at(64 * 4, 128 * 4)
        values = struct.unpack("<128f", data)
        for i in range(64):
            assert values[i] == pytest.approx(1.0), (
                f"element {64 + i} (from slot 0): got {values[i]}"
            )
        for i in range(64, 128):
            assert values[i] == pytest.approx(2.0), (
                f"element {64 + i} (from slot 1): got {values[i]}"
            )


class TestWindowActivationFp32:
    """The ``window`` indicator on fractional FP32 accumulator elements."""

    def test_default_bounds_on_fractional_elements(self) -> None:
        """Default bounds [0.0, 0.1): 0.0 and 0.05 are inside; -0.01 and 0.1 are not."""
        state = new_state(fmt=ElementFormat.FP32, aaq=AaqOptions(quantize_output=True))
        state.set_cr_dstructure(4)
        acc = state.regfile.mutable("r_acc")
        for i, x in enumerate((0.0, 0.05, -0.01, 0.1)):
            struct.pack_into("<f", acc, i * 4, x)
        load_program(state, assemble_program("ACTIVATE.QUANTIZE window cr15;;\nHALT;;\n"))
        run_until_complete(state)

        result = state.regfile.get_post_aaq_reg()
        assert result[0] == 1, "window(0.0) = 1 (a is inclusive)"
        assert result[1] == 1, "window(0.05) = 1"
        assert result[2] == 0, "window(-0.01) = 0 (below a)"
        assert result[3] == 0, "window(0.1) = 0 (b is exclusive)"

    def test_bounds_are_configurable(self) -> None:
        """set_activation_alphas overrides the window bounds for later ACTIVATEs."""
        state = new_state(fmt=ElementFormat.FP32, aaq=AaqOptions(quantize_output=True))
        state.set_activation_alphas(window_a=0.25, window_b=0.75)
        state.set_cr_dstructure(3)
        acc = state.regfile.mutable("r_acc")
        for i, x in enumerate((0.2, 0.25, 0.75)):
            struct.pack_into("<f", acc, i * 4, x)
        load_program(state, assemble_program("ACTIVATE.QUANTIZE window cr15;;\nHALT;;\n"))
        run_until_complete(state)

        result = state.regfile.get_post_aaq_reg()
        assert result[0] == 0, "window(0.2) = 0 (below a = 0.25)"
        assert result[1] == 1, "window(0.25) = 1 (a is inclusive)"
        assert result[2] == 0, "window(0.75) = 0 (b is exclusive)"


class TestWideVectorStoreOverflow:
    """A float32 accumulate that overflows stores +inf in that lane, the IEEE
    result, and sets the sticky ACC_OVERFLOW flag (E04-23, Q53); the other
    lanes are correct and nothing raises, whatever the NumPy error policy."""

    def test_acc_add_overflow_stores_inf_and_sets_the_flag(self) -> None:
        from ipu_isa.faults import ArithmeticEvent

        state = make_state("ACC.ADD ;;", fmt=ElementFormat.FP32)
        k = 5
        acc = [1.0] * LANES
        mult = [2.0] * LANES
        acc[k] = mult[k] = 3e38  # 6e38 does not fit float32
        state.regfile.mutable("r_acc")[:] = struct.pack(f"<{LANES}f", *acc)
        state.regfile.mutable("mult_res")[:] = struct.pack(f"<{LANES}f", *mult)

        with np.errstate(all="raise"):
            Ipu(state).execute_vliw_cycle()

        expected = [3.0] * k + [float("inf")] + [3.0] * (LANES - k - 1)
        assert state.regfile.view("r_acc")[:] == struct.pack(f"<{LANES}f", *expected)
        assert state.arith_flags == {ArithmeticEvent.ACC_OVERFLOW}


class TestWideVectorNumpyErrorPolicy:
    """The embedding application's NumPy policy must not change emulation."""

    def test_acc_invalid_ignores_caller_seterr(self) -> None:
        state = make_state("ACC.ADD ;;", fmt=ElementFormat.FP32)
        state.regfile.mutable("r_acc")[:] = struct.pack(f"<{LANES}f", *([float("inf")] * LANES))
        state.regfile.mutable("mult_res")[:] = struct.pack(f"<{LANES}f", *([float("-inf")] * LANES))
        with np.errstate(all="raise"):
            Ipu(state).execute_vliw_cycle()

        expected_lane = struct.pack("<f", float("inf") + float("-inf"))
        assert state.regfile.view("r_acc") == expected_lane * LANES

    def test_mult_invalid_ignores_caller_seterr(self) -> None:
        # LR0 is 0 at reset: rc index 0, mask shift 0.
        state = make_state("MULT.RC.VV LR0 R0 0 LR0 CR15 ;;", fmt=ElementFormat.FP32)
        state.regfile.mutable("r_wide_debug")[: LANES * 4] = struct.pack(
            f"<{LANES}f", *([0.0] * LANES)
        )
        state.regfile.mutable("r_cyclic_wide_debug")[: LANES * 4] = struct.pack(
            f"<{LANES}f", *([float("inf")] * LANES)
        )
        with np.errstate(all="raise"):
            Ipu(state).execute_vliw_cycle()

        expected_lane = struct.pack("<f", float("inf") * 0.0)
        assert state.regfile.view("mult_res") == expected_lane * LANES
