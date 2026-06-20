"""Tests for RunStats counters (issue #84)."""

from __future__ import annotations

from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_state import IpuState
from ipu_emu.stats import RunStats

from ipu_as.lark_tree import assemble


# ---------------------------------------------------------------------------
# Helpers (mirrored from test_execute.py)
# ---------------------------------------------------------------------------


def _run(asm_code: str, *, cr: dict[int, int] | None = None) -> IpuState:
    encoded = assemble(asm_code)
    decoded = [decode_instruction_word(w) for w in encoded]
    state = IpuState()
    if cr:
        for idx, val in cr.items():
            state.regfile.set_cr(idx, val)
    load_program(state, decoded)
    run_until_complete(state)
    return state


# ---------------------------------------------------------------------------
# RunStats unit tests
# ---------------------------------------------------------------------------


class TestRunStatsProperties:
    def test_utilization_zero_cycles(self):
        s = RunStats()
        assert s.mult_utilization == 0.0
        assert s.acc_utilization == 0.0

    def test_utilization_full(self):
        s = RunStats(total_cycles=10, mult_active_cycles=10, acc_active_cycles=10)
        assert s.mult_utilization == 1.0
        assert s.acc_utilization == 1.0

    def test_utilization_partial(self):
        s = RunStats(total_cycles=4, mult_active_cycles=1, acc_active_cycles=2)
        assert s.mult_utilization == 0.25
        assert s.acc_utilization == 0.5

    def test_xmem_accesses_sum(self):
        s = RunStats(xmem_reads=3, xmem_writes=2)
        assert s.xmem_accesses == 5

    def test_format_summary_contains_key_fields(self):
        s = RunStats(total_cycles=100, mult_active_cycles=80, acc_active_cycles=60,
                     xmem_reads=10, xmem_writes=5)
        text = s.format_summary()
        assert "100" in text
        assert "80" in text
        assert "60" in text
        assert "10" in text
        assert "5" in text
        assert "15" in text  # xmem_accesses


# ---------------------------------------------------------------------------
# Integration: stats are populated by run_until_complete
# ---------------------------------------------------------------------------


class TestStatsCountedDuringExecution:
    def test_nop_program_has_zero_active_cycles(self):
        # All NOPs: no mult, acc, or xmem activity
        state = _run("BKPT;;")
        assert state.stats.mult_active_cycles == 0
        assert state.stats.acc_active_cycles == 0
        assert state.stats.xmem_reads == 0
        assert state.stats.xmem_writes == 0

    def test_total_cycles_matches_instruction_count(self):
        # BKPT halts but counts as one cycle
        state = _run("BKPT;;")
        assert state.stats.total_cycles == 1

    def test_mult_active_counted(self):
        # LDR_MULT_REG (1 read) + MULT.EE in one cycle, then BKPT
        # MULT.EE syntax: ra, cyclic_offset(LR), mask_offset(imm), mask_shift(LR)
        asm = """\
SET lr0 cr0;;
LDR_MULT_REG r0 lr0 cr0;MULT.EE r0 lr0 0 lr0;;
BKPT;;
"""
        state = IpuState()
        encoded = assemble(asm)
        decoded = [decode_instruction_word(w) for w in encoded]
        state.regfile.set_cr(0, 0)
        state.xmem.write_address(0, bytearray(128))
        load_program(state, decoded)
        run_until_complete(state)
        assert state.stats.mult_active_cycles >= 1

    def test_acc_active_counted(self):
        asm = """\
ACC.FIRST;;
BKPT;;
"""
        state = _run(asm)
        assert state.stats.acc_active_cycles >= 1

    def test_xmem_read_counted(self):
        # LDR_MULT_REG reads from xmem
        asm = """\
SET lr0 cr0;;
LDR_MULT_REG r0 lr0 cr1;;
BKPT;;
"""
        state = IpuState()
        encoded = assemble(asm)
        decoded = [decode_instruction_word(w) for w in encoded]
        state.regfile.set_cr(0, 0)
        state.regfile.set_cr(1, 0x1000)
        state.xmem.write_address(0x1000, bytearray(128))
        load_program(state, decoded)
        run_until_complete(state)
        assert state.stats.xmem_reads == 1
        assert state.stats.xmem_writes == 0

    def test_xmem_write_counted(self):
        # STR_ACC_REG writes to xmem
        asm = """\
STR_ACC_REG lr0 cr1;;
BKPT;;
"""
        state = IpuState()
        encoded = assemble(asm)
        decoded = [decode_instruction_word(w) for w in encoded]
        state.regfile.set_lr(0, 0)
        state.regfile.set_cr(1, 0x1000)
        import warnings
        load_program(state, decoded)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_until_complete(state)
        assert state.stats.xmem_writes == 1
        assert state.stats.xmem_reads == 0

    def test_multiple_cycles_accumulate(self):
        # 3 independent acc cycles (ACC.FIRST, ACC, ACC), then BKPT
        asm = """\
ACC.FIRST;;
ACC;;
ACC;;
BKPT;;
"""
        state = _run(asm)
        assert state.stats.total_cycles == 4
        assert state.stats.acc_active_cycles == 3

    def test_nop_slots_not_counted(self):
        # Three pure-LR instructions: no mult/acc/xmem
        asm = """\
SET lr0 cr0;;
ADD lr1 lr0 cr1;;
ADD lr2 lr1 cr1;;
BKPT;;
"""
        state = _run(asm)
        assert state.stats.mult_active_cycles == 0
        assert state.stats.acc_active_cycles == 0
        assert state.stats.xmem_reads == 0
        assert state.stats.xmem_writes == 0
