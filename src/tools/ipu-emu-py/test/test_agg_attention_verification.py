"""Verification of the aggregation datapath for attention kernels (ZDlinear+main).

Re-verified after the shipped AGG fix (origin/master commit d67c441,
"Update AGG instruction references to use MULT_RES lanes"). Each test below is a
decisive probe; the class docstring states the question it settles.

POST-FIX findings (all confirmed PASS):

1. AGG now reduces **mult_res** (the multiply result, pre-accumulator), NOT R_ACC.
   All four handlers read mult_res:
     execute_agg_sum_first   ipu.py:899  mult_res = self.state.regfile.raw("mult_res")
     execute_agg_sum         ipu.py:911
     execute_agg_max_first   ipu.py:930
     execute_agg_max         ipu.py:941
   and pass it to the reducers _agg_sum_lanes (ipu.py:901/915) /
   _agg_max_lanes (ipu.py:933/945).

2. Semantics (matching shipped code, with line refs):
   AGG.SUM.FIRST dest -> R_ACC[dest]  = sum(mult_res[0..n-1])           (ipu.py:901,905)
   AGG.SUM       dest -> R_ACC[dest] += sum(mult_res[0..n-1])           (ipu.py:914-920)
                        (seed read from snapshot R_ACC[dest], ipu.py:914)
   AGG.MAX.FIRST dest -> R_ACC[dest]  = max(mult_res[0..n-1])           (ipu.py:932-935)
   AGG.MAX       dest -> R_ACC[dest]  = max(mult_res[0..n-1], R_ACC[dest])(ipu.py:944-946)
   n = 128 if full_xmem_row else min(CR15.valid_elements,128) (ipu.py:897/_agg_active_lane_count:868)
   masking: deselected mult_res lanes are zeroed by the MULT slot before AGG reads them.
   mult_res is read LIVE -> MULT + AGG co-issue in one bundle (no intervening ACC).

3. Collision-free: full-width MULT -> AGG-into-slot dest=i across queries; earlier
   queries' results parked in other R_ACC slots survive (no ACC.FIRST, no overwrite).

4. Key-major broadcast (ACC over keys, no AGG) still correct + clean 512B store.
   full_xmem_row vs valid_elements honored; R_MASK gating verified narrow INT8.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_as.lark_tree import assemble


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _wide_state(arith: WideVectorArithmetic = WideVectorArithmetic.FP32) -> IpuState:
    st = IpuState(wide_vector_debug=True, wide_vector_arithmetic=arith)
    st.dtype = DType.INT8
    return st


def _load_and_run(st: IpuState, asm: str) -> IpuState:
    encoded = assemble(asm)
    load_program(st, [decode_instruction_word(w) for w in encoded])
    run_until_complete(st)
    return st


def _acc_f(st: IpuState, lane: int) -> float:
    return struct.unpack_from("<f", st.regfile.raw("r_acc"), lane * 4)[0]


def _acc_i(st: IpuState, lane: int) -> int:
    return struct.unpack("<i", struct.pack("<I", st.regfile.get_r_acc_word(lane)))[0]


def _mult_f(st: IpuState, lane: int) -> float:
    return struct.unpack_from("<f", st.regfile.raw("mult_res"), lane * 4)[0]


# ===========================================================================
# FINDING 1: what does AGG reduce -- now mult_res, not R_ACC.
# ===========================================================================

class TestFinding1_AggReducesMultRes:
    """Decisive: AGG.SUM.FIRST must reduce mult_res, not R_ACC.

    Probe: MULT writes a KNOWN pattern into mult_res (sum 768). Pre-seed R_ACC to
    a distinct value (sum 128) WITHOUT accumulating mult_res into it. AGG.SUM.FIRST
    in the SAME bundle, no ACC. Fixed behavior -> result == 768 (sum of mult_res).
    """

    def test_agg_sum_first_reduces_mult_res_768(self) -> None:
        st = _wide_state()
        # R_CYCLIC lane k = 2.0, R0 lane k = 3.0  -> mult_res lane = 6.0 (sum=768)
        st.regfile.set_r_cyclic_at(0, struct.pack("<128f", *([2.0] * 128)))
        st.xmem.write_address(0x1000, struct.pack("<128f", *([3.0] * 128)))
        st.regfile.set_cr(6, 0x1000)
        # Pre-seed R_ACC to 1.0 per lane (sum=128.0) -- distinct from mult sum.
        acc = bytearray(512)
        struct.pack_into("<128f", acc, 0, *([1.0] * 128))
        st.regfile.set_r_acc_bytes(acc)
        st.set_cr_dstructure(128)
        st.regfile.set_lr(0, 100)  # dest lane

        asm = """\
SET lr1 cr6;;
LDR_MULT_REG r0 lr1 cr0;;
MULT.RC.VV lr2 r0 0 lr2; AGG.SUM.FIRST lr0 1;;
BKPT;;
"""
        _load_and_run(st, asm)
        result = _acc_f(st, 100)

        # mult_res really held 6.0/lane this cycle:
        assert _mult_f(st, 0) == pytest.approx(6.0)
        # FIXED verdict: AGG reduced mult_res (768), not R_ACC (128).
        assert result == pytest.approx(768.0), f"expected sum(mult_res)=768, got {result}"
        assert result != pytest.approx(128.0), "AGG still reducing R_ACC (regression)"

    def test_agg_sum_running_adds_to_racc_dest(self) -> None:
        """AGG.SUM (running): R_ACC[dest] += sum(mult_res). Pre-seed dest, expect +768."""
        st = _wide_state()
        st.regfile.set_r_cyclic_at(0, struct.pack("<128f", *([2.0] * 128)))
        st.xmem.write_address(0x1000, struct.pack("<128f", *([3.0] * 128)))
        st.regfile.set_cr(6, 0x1000)
        acc = bytearray(512)
        struct.pack_into("<f", acc, 100 * 4, 1000.0)  # dest pre-seed
        st.regfile.set_r_acc_bytes(acc)
        st.set_cr_dstructure(128)
        st.regfile.set_lr(0, 100)
        asm = """\
SET lr1 cr6;;
LDR_MULT_REG r0 lr1 cr0;;
MULT.RC.VV lr2 r0 0 lr2; AGG.SUM lr0 1;;
BKPT;;
"""
        _load_and_run(st, asm)
        assert _acc_f(st, 100) == pytest.approx(1768.0), "AGG.SUM must add to R_ACC[dest]"

    def test_agg_max_first_reduces_mult_res(self) -> None:
        """AGG.MAX.FIRST: R_ACC[dest] = max(mult_res). One lane spiked to 9.0."""
        st = _wide_state()
        rc = bytearray(512)
        struct.pack_into("<128f", rc, 0, *([1.0] * 128))
        st.regfile.set_r_cyclic_at(0, bytes(rc))
        r0 = bytearray(512)
        struct.pack_into("<128f", r0, 0, *([1.0] * 128))
        struct.pack_into("<f", r0, 7 * 4, 9.0)  # lane 7 product = 9.0
        st.xmem.write_address(0x1000, bytes(r0))
        st.regfile.set_cr(6, 0x1000)
        st.regfile.set_r_acc_bytes(bytearray(512))
        st.set_cr_dstructure(128)
        st.regfile.set_lr(0, 50)
        asm = """\
SET lr1 cr6;;
LDR_MULT_REG r0 lr1 cr0;;
MULT.RC.VV lr2 r0 0 lr2; AGG.MAX.FIRST lr0 1;;
BKPT;;
"""
        _load_and_run(st, asm)
        assert _acc_f(st, 50) == pytest.approx(9.0), "AGG.MAX.FIRST must reduce mult_res max"

    def test_agg_max_running_seeds_from_racc_dest(self) -> None:
        """AGG.MAX (running): seeded with R_ACC[dest]; seed 99 beats all lanes (=6)."""
        st = _wide_state()
        st.regfile.set_r_cyclic_at(0, struct.pack("<128f", *([2.0] * 128)))
        st.xmem.write_address(0x1000, struct.pack("<128f", *([3.0] * 128)))
        st.regfile.set_cr(6, 0x1000)
        acc = bytearray(512)
        struct.pack_into("<f", acc, 50 * 4, 99.0)
        st.regfile.set_r_acc_bytes(acc)
        st.set_cr_dstructure(128)
        st.regfile.set_lr(0, 50)
        asm = """\
SET lr1 cr6;;
LDR_MULT_REG r0 lr1 cr0;;
MULT.RC.VV lr2 r0 0 lr2; AGG.MAX lr0 1;;
BKPT;;
"""
        _load_and_run(st, asm)
        assert _acc_f(st, 50) == pytest.approx(99.0), "AGG.MAX seed (99) must win over lanes(6)"

    def test_agg_sum_first_int8_reduces_mult_res(self) -> None:
        """Narrow INT8: MULT.RC.VV -> mult_res = 3*4=12/lane; AGG.SUM.FIRST -> 12*128=1536."""
        st = IpuState()
        st.dtype = DType.INT8
        st.set_cr_dstructure(128)
        st.xmem.write_address(0x1000, bytes([3] * 128))   # R0
        st.xmem.write_address(0x2000, bytes([4] * 512))   # R_CYCLIC
        st.regfile.set_cr(6, 0x1000)
        st.regfile.set_cr(7, 0x2000)
        st.regfile.set_cr(8, 0)
        st.regfile.set_lr(0, 100)
        # Pre-seed R_ACC distinct so we'd see it if AGG read R_ACC.
        for i in range(128):
            st.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", 1))[0])
        asm = """\
SET lr1 cr6;;
LDR_MULT_REG r0 lr1 cr0;;
SET lr3 cr7;;
SET lr4 cr8;;
LDR_CYCLIC_MULT_REG lr3 cr0 lr4;;
MULT.RC.VV lr4 r0 0 lr4; AGG.SUM.FIRST lr0 1;;
BKPT;;
"""
        _load_and_run(st, asm)
        assert _acc_i(st, 100) == 1536, f"INT8 AGG.SUM.FIRST expected 1536, got {_acc_i(st, 100)}"


# ===========================================================================
# FINDING 2: query-major dot-product is now COLLISION-FREE.
# ===========================================================================

class TestFinding2_QueryMajorAttnV:
    """Decisive end-to-end: query-major O[i] = P[i,:] @ V, 1 channel, 2 chunks.

    The team's intended flow per chunk: full-width MULT of 128 key-products, then
    AGG.SUM[.FIRST] dest=i to reduce-and-accumulate into R_ACC[i] -- NO ACC, NO
    per-query reset. With the fix this produces P@V directly, and query i's result
    parked in R_ACC[i] survives while query i+1 computes (different dest lane).
    """

    N_Q = 4
    N_K = 256

    def _make_problem(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        P = rng.integers(-3, 4, size=(self.N_Q, self.N_K)).astype(np.float32)
        V = rng.integers(-3, 4, size=(self.N_K,)).astype(np.float32)
        return P, V

    def test_intended_flow_matches_and_is_collision_free(self) -> None:
        P, V = self._make_problem()
        ref = P @ V  # (N_Q,)

        st = _wide_state()
        # P query-major, contiguous: (i,c) chunk at 0x10000 + (i*2+c)*512.
        for i in range(self.N_Q):
            for c in range(2):
                st.xmem.write_address(0x10000 + (i * 2 + c) * 512,
                                      struct.pack("<128f", *P[i, c * 128:(c + 1) * 128]))
        for c in range(2):
            st.xmem.write_address(0x20000 + c * 512,
                                  struct.pack("<128f", *V[c * 128:(c + 1) * 128]))

        st.regfile.set_cr(6, 0x10000)   # P base
        st.regfile.set_cr(7, 0x20000)   # V chunk0
        st.regfile.set_cr(8, 0x20200)   # V chunk1
        st.regfile.set_cr(9, 0)         # rc index
        st.regfile.set_cr(10, 512)      # chunk stride
        st.set_cr_dstructure(128)

        # lr0=P ptr, lr3=dest lane(=i, starts 0), lr4=stride 512.
        # Per query i, chunk 0: AGG.SUM.FIRST (clean write dest=i);
        #             chunk 1: AGG.SUM (running add into dest=i).
        lines = ["SET lr0 cr6;;", "SET lr3 cr9;;", "SET lr4 cr10;;"]
        for i in range(self.N_Q):
            for c in range(2):
                v_cr = 7 if c == 0 else 8
                agg = "AGG.SUM.FIRST" if c == 0 else "AGG.SUM"
                lines.append(f"SET lr1 cr{v_cr};;")
                lines.append("LDR_MULT_REG r0 lr0 cr0;;")
                lines.append("LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;")
                lines.append(f"MULT.RC.VV lr2 r0 0 lr2; {agg} lr3 1;;")
                lines.append("ADD lr0 lr0 lr4;;")   # next P chunk
            lines.append("INC lr3 1;;")             # next query -> next dest lane
        lines.append("BKPT;;")
        _load_and_run(st, "\n".join(lines))

        got = np.array([_acc_f(st, i) for i in range(self.N_Q)])
        # (a) every O[i] matches numpy P@V:
        assert np.allclose(got, ref), f"got {got} ref {ref}"
        # (b) collision-free: all 4 results coexist in distinct R_ACC slots, no ACC.FIRST.
        assert len(set(got.tolist())) >= 1 and np.allclose(got, ref), "earlier queries clobbered"


# ===========================================================================
# FINDING 3: key-major broadcast (no AGG path) -- still solid.
# ===========================================================================

class TestFinding3_KeyMajorBroadcast:
    """lanes = query tokens, scalar = V[s], ACC accumulate over keys, no AGG.

    O[t] = sum_s P[t,s] * V[s]. Produces a full 128-lane channel-major R_ACC row
    that stores cleanly via STR_ACC_REG.
    """

    N_Q = 128
    N_K = 4

    def test_key_major_acc_matches_and_stores_clean(self) -> None:
        rng = np.random.default_rng(1)
        P = rng.integers(-2, 3, size=(self.N_Q, self.N_K)).astype(np.float32)
        V = rng.integers(-2, 3, size=(self.N_K,)).astype(np.float32)
        ref = P @ V  # (128,)

        st = _wide_state()
        for s in range(self.N_K):
            st.xmem.write_address(0x10000 + s * 512, struct.pack("<128f", *P[:, s]))
            st.xmem.write_address(0x20000 + s * 512,
                                  struct.pack("<128f", *([float(V[s])] * 128)))
        st.regfile.set_cr(6, 0x10000)   # P col base
        st.regfile.set_cr(7, 0x20000)   # V base
        st.regfile.set_cr(8, 0x30000)   # out addr
        st.regfile.set_cr(9, 512)       # stride
        st.set_cr_dstructure(128)

        lines = ["SET lr0 cr7;;", "SET lr1 cr6;;", "SET lr3 cr9;;"]
        for s in range(self.N_K):
            acc = "ACC.FIRST" if s == 0 else "ACC"
            lines.append("LDR_MULT_REG r0 lr0 cr0;;")
            lines.append("LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;")
            lines.append(f"MULT.RC.VV lr2 r0 0 lr2; {acc};;")
            lines.append("ADD lr0 lr0 lr3;;")
            lines.append("ADD lr1 lr1 lr3;;")
        lines.append("SET lr5 cr8;;")
        lines.append("STR_ACC_REG lr5 cr0;;")
        lines.append("BKPT;;")
        _load_and_run(st, "\n".join(lines))

        got = np.array([_acc_f(st, t) for t in range(self.N_Q)])
        assert np.allclose(got, ref), f"key-major got {got[:6]} ref {ref[:6]}"
        stored = st.xmem.read_address(0x30000, 512)
        stored_vals = np.frombuffer(bytes(stored), dtype="<f4")
        assert np.allclose(stored_vals, ref), "stored row mismatch"


# ===========================================================================
# FINDING 4: supporting checks.
# ===========================================================================

class TestFinding4a_FullXmemRowVsValidElements:
    """AGG full_xmem_row=1 -> 128 lanes (ignores valid_elements);
    full_xmem_row=0 -> CR15.valid_elements. Reduced source is mult_res."""

    def _run(self, full_xmem_row: int) -> int:
        st = IpuState()
        st.dtype = DType.INT8
        st.set_cr_dstructure(10)        # valid_elements=10
        st.xmem.write_address(0x1000, bytes([1] * 128))   # R0
        st.xmem.write_address(0x2000, bytes([1] * 512))   # R_CYCLIC -> mult_res=1/lane
        st.regfile.set_cr(6, 0x1000)
        st.regfile.set_cr(7, 0x2000)
        st.regfile.set_cr(8, 0)
        st.regfile.set_lr(0, 127)
        asm = f"""\
SET lr1 cr6;;
LDR_MULT_REG r0 lr1 cr0;;
SET lr3 cr7;;
SET lr4 cr8;;
LDR_CYCLIC_MULT_REG lr3 cr0 lr4;;
MULT.RC.VV lr4 r0 0 lr4; AGG.SUM.FIRST lr0 {full_xmem_row};;
BKPT;;
"""
        _load_and_run(st, asm)
        return _acc_i(st, 127)

    def test_full_xmem_row_1_uses_128(self) -> None:
        assert self._run(1) == 128, "full_xmem_row=1 must sum all 128 mult_res lanes"

    def test_full_xmem_row_0_uses_valid_elements(self) -> None:
        assert self._run(0) == 10, "full_xmem_row=0 must use valid_elements=10"


class TestFinding4b_MultAggCoIssueLive:
    """mult_res is read live, so MULT and AGG co-issue in ONE bundle with no ACC.

    (This is the core enabler of finding 2; asserted directly here.) Also confirms
    ACC and AGG remain mutually exclusive in the single ACC slot (cannot co-issue).
    """

    def test_mult_and_agg_same_bundle_no_acc(self) -> None:
        st = IpuState()
        st.dtype = DType.INT8
        st.set_cr_dstructure(128)
        st.xmem.write_address(0x1000, bytes([2] * 128))
        st.xmem.write_address(0x2000, bytes([3] * 512))
        st.regfile.set_cr(6, 0x1000)
        st.regfile.set_cr(7, 0x2000)
        st.regfile.set_cr(8, 0)
        st.regfile.set_lr(0, 5)
        asm = """\
SET lr1 cr6;;
LDR_MULT_REG r0 lr1 cr0;;
SET lr3 cr7;;
SET lr4 cr8;;
LDR_CYCLIC_MULT_REG lr3 cr0 lr4;;
MULT.RC.VV lr4 r0 0 lr4; AGG.SUM.FIRST lr0 1;;
BKPT;;
"""
        _load_and_run(st, asm)
        # 2*3=6 per lane, 128 lanes -> 768, computed in a single MULT+AGG bundle.
        assert _acc_i(st, 5) == 768, f"MULT+AGG one bundle expected 768, got {_acc_i(st, 5)}"

    def test_acc_and_agg_cannot_co_issue(self) -> None:
        asm = """\
MULT.RC.VV lr4 r0 0 lr4; ACC.FIRST; AGG.SUM.FIRST lr0 1;;
BKPT;;
"""
        with pytest.raises(SystemExit):
            assemble(asm)


class TestFinding4c_RmaskGatingNarrowMode:
    """R_MASK lane gating zeroes deselected mult_res lanes -- narrow INT8.

    _mult_mask_and_shift returns early in wide mode (ipu.py:374), so gating is only
    exercisable narrow. With AGG reading mult_res, gated (zero) lanes contribute 0
    to the sum -- so AGG over a masked MULT reduces only the active lanes.
    """

    def _run_masked_agg(self, n_set_bytes: int) -> int:
        mask_data = bytearray(128)
        for i in range(n_set_bytes):
            mask_data[i] = 0xFF
        st = IpuState()
        st.dtype = DType.INT8
        st.set_cr_dstructure(128)
        st.xmem.write_address(0x1000, bytes([2] * 128))   # R0
        st.xmem.write_address(0x2000, bytes([3] * 512))   # R_CYCLIC
        st.xmem.write_address(0x3000, bytes(mask_data))
        st.regfile.set_cr(6, 0x1000)
        st.regfile.set_cr(7, 0x2000)
        st.regfile.set_cr(8, 0)
        st.regfile.set_cr(9, 0x3000)
        st.regfile.set_lr(0, 10)
        asm = """\
SET lr1 cr6;;
LDR_MULT_REG r0 lr1 cr0;;
SET lr3 cr7;;
SET lr5 cr8;;
LDR_CYCLIC_MULT_REG lr3 cr0 lr5;;
SET lr6 cr9;;
LDR_MULT_MASK_REG lr6 cr0;;
MULT.RC.VV lr5 r0 0 lr5; AGG.SUM.FIRST lr0 1;;
BKPT;;
"""
        _load_and_run(st, asm)
        return _acc_i(st, 10)

    def test_mask_gates_lanes_before_agg(self) -> None:
        # 8 bytes set -> 64 active lanes, each product 6 -> sum 384 (not 768).
        assert self._run_masked_agg(8) == 384, "masked AGG must reduce only active lanes"

    def test_full_mask_all_lanes(self) -> None:
        assert self._run_masked_agg(16) == 768, "full mask -> all 128 lanes reduced"
