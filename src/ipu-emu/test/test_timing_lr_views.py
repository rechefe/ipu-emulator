"""Which LR value each consumer reads when the same word writes that LR (E02-05).

"Cycle start" is the value the LR held when the word began; "forwarded" is the
value an LR sub-slot wrote earlier in the same word (GLOSSARY.md). Each test
runs one minimal probe from ``timing_probes.py`` in faithful mode and asserts
the one value that tells the two apart. These pin today's behaviour: E04
refactors must keep every test green. E04-25 flipped the two rows decision Q32
changed (``test_agg_dest_slot_reads_forwarded``,
``test_reshape_source_dest_read_forwarded``).
"""

from __future__ import annotations

import pytest
from timing_probes import (
    PROBES,
    RAISES,
    ROW,
    Mode,
    lane,
    prepare,
    r_cyclic_element,
    ra_element,
    run,
    stored_element,
)

from ipu_emu.ipu import Ipu
from ipu_isa.faults import FaultId


def _lr(probe: str, index: int) -> int:
    return run(PROBES[probe]).state.regfile.get_lr(index)


@pytest.mark.parametrize(
    ("probe", "expected"),
    [
        ("lr_alu_add", 5 + 1),
        ("lr_alu_sub", 5 - 1),
        ("lr_alu_incr_mod_pow2", (0 + 5) % 16),
        ("lr_alu_addb", 0x05050505),
    ],
)
def test_lr_alu_sources_read_cycle_start(probe, expected):
    """ADD, SUB, INC.MODP2 and ADDB read their LR sources at cycle start.

    SET's source is pinned by ADD's: both are LcrHalfIdx reads in the LR slot.
    The ADDB probe's dest is LRD2, the pair LR2:LR3; LR2 is checked.
    """
    assert _lr(probe, 2) == expected


def test_lr_subslots_read_cycle_start():
    """An LR sub-slot reads the cycle-start value of an LR another sub-slot writes."""
    assert _lr("lr_subslots", 1) == 0 + 1


@pytest.mark.parametrize(("probe", "own"), [("lr_inc_own_dest", 13), ("lr_dec_own_dest", 7)])
def test_inc_dec_read_own_dest_at_cycle_start(probe, own):
    """INC/DEC update their dest from its cycle-start value, and another sub-slot
    reading that LR sees the cycle-start value too."""
    state = run(PROBES[probe]).state
    assert (state.regfile.get_lr(0), state.regfile.get_lr(1)) == (own, 10)


@pytest.mark.parametrize(
    ("probe", "marker"),
    [
        ("cond_beq", 2),  # 0 == 0: taken (forwarded 5 would fall through)
        ("cond_bne", 1),  # 0 != 0: falls through
        ("cond_blt", 2),  # 0 < 1: taken
        ("cond_bge", 1),  # 0 >= 1: falls through
        ("cond_br", 1),  # jumps to 4, the cycle-start LR0 (forwarded: 6)
    ],
)
def test_cond_operands_read_cycle_start(probe, marker):
    """BEQ, BNE, BLT.S, BGE.S and BR read their LR operands at cycle start."""
    assert _lr(probe, 7) == marker


def test_break_ifeq_reads_cycle_start():
    """BREAK.IFEQ reads its LR at cycle start: the word that sets LR0 = 5 does
    not break, the next one does."""
    assert run(PROBES["break_ifeq"]).breaks == [1]


def test_load_offset_and_index_read_forwarded():
    """LD.MULT, LD.MULT.CYCLIC and LD.MASK read their
    offset and index LRs forwarded."""
    assert ra_element(run(PROBES["load_mult_offset"]).state, 0) == 2
    assert r_cyclic_element(run(PROBES["load_cyclic_offset"]).state, 0) == 2
    state = run(PROBES["load_cyclic_index"]).state
    # Slot 0 is untouched (its bytes stay 0); slot 1 (element 128) got the row.
    assert (state.regfile.view("r_cyclic")[0], r_cyclic_element(state, 128)) == (0, 1)
    assert run(PROBES["load_mask_offset"]).state.regfile.view("r_mask")[0] == 0x22


def test_aaq_offsets_read_forwarded():
    """ACTIVATE.QUANTIZE, ACTIVATE.EXACT and LD.LUT read their table_offset LR
    forwarded: the row lands at (or is read from) ROW + 8 / ROW + 1, not ROW."""
    state = run(PROBES["aaq_store_offset"]).state
    assert state.xmem.snapshot_bytes(ROW * 128, 128) == bytes(128)
    assert stored_element(state, ROW + 8, 0) == 7
    state = run(PROBES["exact_store_offset"], Mode.WIDE_FP32).state
    assert state.xmem.snapshot_bytes(ROW * 512, 4) == bytes(4)
    assert stored_element(state, ROW + 8, 0) == 7
    state = run(PROBES["ld_lut_offset"]).state
    assert state.regfile.view("aaq_lut")[0] == 2


@pytest.mark.parametrize(
    ("probe", "expected"),
    [
        ("mult_rc_idx", 2 * 2),  # R_CYCLIC[128..] (2s) squared, not [0..] (1s)
        ("mult_ra_idx", 2 * 1),  # R1 (2s) x CR1, not R0 (1s)
        ("mult_mask_shift", 0),  # shift 1 masks lane 0; shift 0 would keep 3 * 3
    ],
)
def test_mult_lr_operands_read_forwarded(probe, expected):
    """MULT reads rc_idx, ra_idx and mask_shift forwarded."""
    assert lane(run(PROBES[probe]).state, "mult_res", 0) == expected


def test_mult_rc_ve_src_reads_forwarded():
    """MULT.RC.VE's LR src (no ``read`` flag; the handler reads the live file)
    is forwarded: it selects Ra element 5 (7), not element 0 (3)."""
    assert lane(run(PROBES["mult_rc_ve_src"]).state, "mult_res", 0) == 7


def test_acc_stride_offset_reads_forwarded():
    """ACC.STRIDE reads its offset LR forwarded: the row lands at lane 32, so
    lane 32 holds MULT_RES[0] (1) rather than MULT_RES[64] (65)."""
    state = run(PROBES["acc_stride_offset"]).state
    assert lane(state, "r_acc", 32) == 1


def test_reshape_mask_lr_reads_forwarded():
    """ACC.RESHAPE reads an LR reshape_mask forwarded: mask 8 moves no element,
    so R_ACC stays zero."""
    assert lane(run(PROBES["reshape_mask_lr"]).state, "r_acc", 0) == 0


@pytest.mark.parametrize(
    ("op", "value"), [("sum", 128), ("sum_first", 128), ("max", 1), ("max_first", 1)]
)
def test_agg_dest_slot_reads_forwarded(op, value):
    """AGG.SUM/AGG.MAX(.FIRST) read dest_slot forwarded (Q32 rule 1, E04-25):
    the result goes to lane 5, the value the same word's SET wrote, not lane 0."""
    state = run(PROBES[f"agg_dest_slot_{op}"]).state
    assert (lane(state, "r_acc", 0), lane(state, "r_acc", 5)) == (0, value)


@pytest.mark.parametrize(
    ("probe", "dest", "value"), [("reshape_source", 0, 6), ("reshape_dest", 5, 1)]
)
def test_reshape_source_dest_read_forwarded(probe, dest, value):
    """ACC.RESHAPE reads its source and dest LRD pairs forwarded (Q32 rule 1,
    E04-25): element 7 sees the same word's SET (byte 5), so it copies
    MULT_RES[5] (6) to R_ACC[0] (source probe) or MULT_RES[0] (1) to R_ACC[5]
    (dest probe)."""
    assert lane(run(PROBES[probe]).state, "r_acc", dest) == value


@pytest.mark.parametrize("probe", ["lr_conflict_same_lr", "lr_conflict_lrd_overlap"])
def test_lr_write_conflict_raises(probe):
    """Two LR sub-slots writing one LR (directly, or through an ADDB LRD pair)
    fault F-3 before any LR is written."""
    with pytest.raises(RAISES[probe], match="LR conflict") as caught:
        run(PROBES[probe])
    assert caught.value.fault is FaultId.F3


def test_lr_subslot_exception_leaves_no_partial_write():
    """A word whose LR sub-slot 1 source is a reserved encoding faults F-1 at
    fetch, before any slot runs, so sub-slot 0's write to LR1 never happens.
    (Before E04-19 this word raised in the operand read; no encodable word
    can make an LR read raise now.)
    """
    state = prepare(PROBES["lr_subslot_exception"])
    with pytest.raises(RAISES["lr_subslot_exception"]):
        engine = Ipu(state)
        engine.cycle.step()
    assert state.regfile.get_lr(1) == 0
