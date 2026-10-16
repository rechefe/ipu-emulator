"""Tests for ISA alias tracking and lane-accurate MAC accounting.

The ISA has no vector move/add/subtract/broadcast, so kernels emulate them with
a multiply against a constant 1.0. These tests pin the detection of those idioms
and the counters that separate real MAC work from data routing.
"""

from __future__ import annotations

import pytest

from ipu_emu.debug_control import DebugControl
from ipu_emu.emulator import DebugAction, load_program, run_until_complete, run_with_debug
from ipu_emu.errors import DebugOnlyInstructionWarning
from ipu_emu.ipu import Ipu, LANES
from ipu_emu.ipu_state import IpuState

from emu_support import ElementFormat, assemble_program, make_state, new_state, run

# A declared row narrower than LANES, for the padding cases.
NARROW_ELEMENTS = 16


# CR1 is hardwired to 1 and read-only (writes raise), so every `... CR1 ...`
# multiply below is a genuine multiply-by-one with no setup required.


class TestAliasAttribution:
    """Each identity multiply attributes to exactly one alias — first match wins."""

    def test_mov_rc_is_the_catch_all(self):
        state = run("MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.ADD.FIRST ;;\nBKPT;;")
        assert state.stats.alias_hits == {"A1_MOV_RC": 1}

    def test_running_acc_add_is_a_vector_add(self):
        state = run("MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.ADD ;;\nBKPT;;")
        assert state.stats.alias_hits == {"A2_ADD_VV": 1}

    def test_acc_sub_is_a_vector_subtract(self):
        state = run("MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.SUB ;;\nBKPT;;")
        assert state.stats.alias_hits == {"A3_SUB_VV": 1}

    def test_agg_sum_is_a_register_reduce(self):
        state = run(
            "MULT.RC.VE LR0 CR1 0 LR0 CR15 ; AGG.SUM.FIRST LR3 CR15 ;;\nBKPT;;",
        )
        assert state.stats.alias_hits == {"A4_AGG_RC": 1}

    def test_mult_ee_is_a_broadcast(self):
        state = run("MULT.EE LR1 CR1 0 LR0 CR15 ; ACC.ADD.FIRST ;;\nBKPT;;")
        assert state.stats.alias_hits == {"A5_BCAST": 1}

    def test_nonzero_mask_offset_is_a_scatter(self):
        state = run("MULT.RC.VE LR0 CR1 3 LR0 CR15 ; ACC.ADD ;;\nBKPT;;")
        assert state.stats.alias_hits == {"A6_GATHER_SCATTER": 1}

    def test_simulation_only_store_is_flagged(self):
        with pytest.warns(DebugOnlyInstructionWarning):
            state = run("STR_ACC_REG LR2 CR2 ;;\nBKPT;;")
        assert state.stats.alias_hits == {"A16_STR_ACC_REG": 1}

    def test_hits_partition_identity_cycles(self):
        state = run(
            "MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.ADD.FIRST ;;\n"
            "MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.ADD ;;\n"
            "MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.SUB ;;\n"
            "BKPT;;",
        )
        stats = state.stats
        mult_hits = sum(
            count for alias, count in stats.alias_hits.items() if alias != "A16_STR_ACC_REG"
        )
        assert mult_hits == stats.mult_identity_cycles == 3


class TestGenuineWorkIsNotFlagged:
    def test_weight_indexed_multiply_is_not_an_alias(self):
        # src is an LR selecting an Ra element — a real weight, not a constant.
        state = run("MULT.RC.VE LR0 LR2 0 LR0 CR15 ; ACC.ADD ;;\nBKPT;;")
        stats = state.stats
        assert stats.alias_hits == {}
        assert stats.mult_identity_cycles == 0
        assert stats.mult_active_cycles == 1

    def test_vector_by_vector_multiply_has_no_scalar_to_flag(self):
        state = run("MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD ;;\nBKPT;;")
        assert state.stats.alias_hits == {}
        assert state.stats.mult_identity_cycles == 0

    def test_real_multiply_does_not_inherit_a_preceding_identity_verdict(self):
        # MULT.RC.VV has no scalar operand to probe, so it must not be judged
        # by the identity multiply that ran the cycle before it.
        state = run(
            "MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.ADD.FIRST ;;\n"
            "MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD ;;\n"
            "BKPT;;"
        )
        stats = state.stats
        assert stats.mult_active_cycles == 2
        assert stats.mult_identity_cycles == 1
        # Only the first multiply's lanes count as routing.
        assert stats.mult_lane_ops == 2 * LANES
        assert stats.mult_identity_lane_ops == LANES


class TestRealWeightsAreNotIdentity:
    """A weight that happens to equal 1 is still a MAC.

    Only a *constant register* supplying 1.0 is a pass-through. An LR selecting
    an element of R0/R1 is selecting data, whose value is not the emulator's
    business — quantized kernels are full of weights equal to 1.
    """

    def test_weight_of_value_one_is_still_real_work(self):
        # R0[5] = 1 (a perfectly ordinary INT8 weight); LR2 = 5 selects it.
        state = make_state(
            "SET LR2 CR5 ;;\nMULT.RC.VE LR0 LR2 0 LR0 CR15 ; ACC.ADD ;;\nBKPT;;",
            cr={5: 5},
        )
        # Write the raw combined Ra buffer (R0 ++ R1) so LR2 finds the weight.
        state.regfile.raw("r")[5] = 1
        run_until_complete(state)
        stats = state.stats
        assert stats.mult_identity_cycles == 0
        assert stats.alias_hits == {}
        assert stats.effective_mac_utilization > 0.0

    def test_constant_register_of_value_one_is_a_pass_through(self):
        state = run("MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.ADD ;;\nBKPT;;")
        assert state.stats.mult_identity_cycles == 1


class TestLaneAccounting:
    def test_full_width_multiply_counts_every_lane(self):
        state = run("MULT.RC.VE LR0 LR2 0 LR0 CR15 ; ACC.ADD ;;\nBKPT;;")
        assert state.stats.mult_lane_ops == LANES

    def test_declared_narrow_row_counts_only_declared_lanes(self):
        state = run(
            "MULT.RC.VE LR0 LR2 0 LR0 CR15 ; ACC.ADD ;;\nBKPT;;",
            dstructure={"valid_elements": NARROW_ELEMENTS},
        )
        stats = state.stats
        assert stats.mult_lane_ops == NARROW_ELEMENTS
        # The cycle is still fully occupied — that is the distortion being exposed.
        assert stats.mult_active_cycles == 1

    def test_undeclared_dstructure_register_does_not_zero_the_lane_count(self):
        # Naming a CR that carries no dstructure config decodes valid_elements
        # as 0. The multiply still retires every masked-in lane, so 0 would be a silent
        # under-report of exactly the metric this feature exists to fix.
        state = run("MULT.RC.VE LR0 LR2 0 LR0 CR3 ; ACC.ADD ;;\nBKPT;;")
        stats = state.stats
        assert stats.mult_lane_ops == LANES
        assert stats.effective_mac_utilization > 0.0

    def test_identity_work_is_excluded_from_effective_mac(self):
        state = run("MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.ADD.FIRST ;;\nBKPT;;")
        stats = state.stats
        assert stats.mult_lane_ops == stats.mult_identity_lane_ops == LANES
        assert stats.effective_mac_utilization == 0.0
        # ...even though the multiplier looks busy by the old metric.
        assert stats.mult_utilization > 0.0

    def test_idle_cycles_are_total_minus_active(self):
        state = run("ADD LR1 LR1 CR1 ;;\nBKPT;;")
        stats = state.stats
        assert stats.mult_active_cycles == 0
        assert stats.mult_idle_cycles == stats.total_cycles


class TestIdleAccounting:
    @pytest.mark.parametrize("debug", [False, True])
    def test_embedded_break_counts_only_the_resumed_instruction(self, debug):
        state = make_state("BREAK;;\nBKPT;;")
        if debug:

            def callback(state, cycles):
                assert cycles == state.stats.mult_idle_cycles == 0
                return DebugAction.CONTINUE

            run_with_debug(state, callback)
        else:
            run_until_complete(state)
        assert state.stats.mult_idle_cycles == state.stats.total_cycles == 2

    def test_idle_count_is_current_at_debugger_stops(self):
        state = make_state(
            "MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.ADD.FIRST ;;\nADD LR1 LR1 CR1 ;;\nBKPT;;"
        )
        control = DebugControl()
        control.add_breakpoint(1)
        control.add_breakpoint(2)
        observed = []

        def callback(state, cycles):
            observed.append((cycles, state.stats.mult_idle_cycles))
            return DebugAction.CONTINUE

        run_with_debug(state, callback, control=control)
        assert observed == [(1, 0), (2, 1)]
        assert state.stats.mult_idle_cycles == 2
        assert state.stats.mult_idle_cycles + state.stats.mult_active_cycles == 3

    @pytest.mark.parametrize("skip_break", [False, True])
    def test_empty_instruction_counts_as_idle(self, skip_break):
        state = IpuState()
        ipu = Ipu(state)
        if skip_break:
            ipu.execute_vliw_cycle_skip_break()
        else:
            ipu.execute_vliw_cycle()
        assert state.program_counter == 1
        assert state.stats.mult_idle_cycles == 1


class TestSummaryRendering:
    def test_alias_block_only_appears_when_there_are_hits(self):
        clean = run("MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD ;;\nBKPT;;")
        assert "ISA alias hits" not in clean.stats.format_summary()

        faked = run("MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.ADD.FIRST ;;\nBKPT;;")
        summary = faked.stats.format_summary()
        assert "ISA alias hits" in summary
        assert "A1_MOV_RC" in summary


class TestDeclaredVectorIdentities:
    @staticmethod
    def prepare(*, wide=False, dtype=None, declared=True, ring=False, tail=""):
        from ipu_emu.ipu_math import DType

        fmt = ElementFormat.FP32 if wide else ElementFormat[(dtype or DType.INT8).name]
        state = new_state(fmt=fmt)
        state.write_constant_ones(0)
        size = 512 if wide else 128
        ones = state.xmem.read_address(0, size)
        if not declared:
            # Even writing the identical values revokes their constant intent.
            state.xmem.write_address(0, ones)
        state.xmem.write_address(size, ones)  # ordinary data equal to one
        state.regfile.set_cr(2, 1)
        first = "CR2" if ring else "CR0"
        second = "CR0" if ring else "CR2"
        code = f"LDR_MULT_REG R0 LR0 {first};;\nLDR_CYCLIC_MULT_REG LR0 {second} LR0;;\n" + tail
        load_program(state, assemble_program(code))
        return state

    @pytest.mark.parametrize("wide", [False, True])
    @pytest.mark.parametrize("ring", [False, True])
    @pytest.mark.parametrize("declared", [False, True])
    def test_only_declared_vector_ones_count(self, wide, ring, declared):
        state = self.prepare(
            wide=wide,
            ring=ring,
            declared=declared,
            tail="MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD ;;\nBKPT;;",
        )
        run_until_complete(state)
        assert state.stats.alias_hits == ({"A2_ADD_VV": 1} if declared else {})
        assert state.stats.mult_identity_lane_ops == (128 if declared else 0)

    def test_declared_ring_ones_with_lr_selected_beta(self):
        state = self.prepare(ring=True, tail="MULT.RC.VE LR0 LR0 0 LR0 CR15 ; ACC.ADD ;;\nBKPT;;")
        run_until_complete(state)
        assert state.stats.alias_hits == {"A2_ADD_VV": 1}

    def test_same_cycle_overwrite_uses_snapshot_then_revokes(self):
        state = self.prepare(
            tail=(
                "LDR_MULT_REG R0 LR0 CR2 ; MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD ;;\n"
                "MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD ;;\nBKPT;;"
            )
        )
        run_until_complete(state)
        assert state.stats.alias_hits == {"A2_ADD_VV": 1}
        assert state.stats.mult_active_cycles == 2

    @pytest.mark.parametrize("mutation", ["setter", "raw", "memory"])
    def test_mutation_revokes_provenance_even_if_values_are_ones(self, mutation):
        state = self.prepare(tail="MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD ;;\nBKPT;;")
        engine = Ipu(state)
        if mutation == "memory":
            state.xmem[0:128] = bytes([1]) * 128
        engine.execute_vliw_cycle()
        engine.execute_vliw_cycle()
        if mutation == "setter":
            state.regfile.set_r(0, bytes([1]) * 128)
        elif mutation == "raw":
            state.regfile.raw("r")[:128] = bytes([1]) * 128
        run_until_complete(state)
        assert state.stats.alias_hits == {}

    @pytest.mark.parametrize("dtype_name", ["INT8", "E4", "E5"])
    def test_declared_ones_are_dtype_correct(self, dtype_name):
        from ipu_emu.ipu_math import DType

        state = self.prepare(
            dtype=getattr(DType, dtype_name),
            tail="MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD.FIRST ;;\nBKPT;;",
        )
        run_until_complete(state)
        assert state.stats.alias_hits == {"A1_MOV_RC": 1}

    @pytest.mark.parametrize("valid,expected", [(128, "A6_GATHER_SCATTER"), (16, "A1_MOV_RC")])
    def test_mask_zero_window_is_distinct_from_declared_padding(self, valid, expected):
        state = make_state(
            "MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.ADD.FIRST ;;\nBKPT;;",
            dstructure={"valid_elements": valid},
        )
        state.regfile.set_r_mask(((1 << 16) - 1).to_bytes(16, "little") + bytes(112))
        run_until_complete(state)
        assert state.stats.alias_hits == {expected: 1}

    def test_shifted_mask_zero_uses_the_effective_window(self):
        state = run("MULT.RC.VE LR0 CR1 0 LR1 CR15 ; ACC.ADD.FIRST ;;\nBKPT;;")
        # The default LR1 is zero; ordinary full-width move remains A1.
        assert state.stats.alias_hits == {"A1_MOV_RC": 1}
        shifted = run("SET LR1 CR1;; MULT.RC.VE LR0 CR1 0 LR1 CR15 ; ACC.ADD.FIRST ;; BKPT;;")
        assert shifted.stats.alias_hits == {"A6_GATHER_SCATTER": 1}

    def test_stride_routes_a_declared_vector(self):
        state = self.prepare(
            ring=True, tail="MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.STRIDE 16 on off LR0 ;;\nBKPT;;"
        )
        run_until_complete(state)
        assert state.stats.alias_hits == {"A6_GATHER_SCATTER": 1}

    def test_cyclic_window_requires_provenance_across_all_source_lanes(self):
        state = IpuState()
        state.write_constant_ones(0)
        state.xmem.write_address(128, bytes([1]) * 128)
        state.regfile.set_cr(2, 128)
        state.regfile.set_cr(3, 64)
        state.regfile.set_cr(4, 1)
        code = """SET LR2 CR2;; SET LR3 CR3;;
LDR_CYCLIC_MULT_REG LR0 CR0 LR0;;
LDR_CYCLIC_MULT_REG LR0 CR0 LR2;;
MULT.RC.VE LR3 LR0 0 LR0 CR15 ; ACC.ADD;;
LDR_CYCLIC_MULT_REG LR0 CR4 LR2;;
MULT.RC.VE LR3 LR0 0 LR0 CR15 ; ACC.ADD;; BKPT;;"""
        load_program(state, assemble_program(code))
        run_until_complete(state)
        assert state.stats.alias_hits == {"A2_ADD_VV": 1}

    def test_retained_mutable_handle_cannot_reacquire_constant_provenance(self):
        state = self.prepare(tail="MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD ;;\nBKPT;;")
        escaped = state.regfile.raw("r")
        engine = Ipu(state)
        engine.execute_vliw_cycle()  # declared load after the handle escaped
        escaped[:128] = bytes([1]) * 128
        run_until_complete(state)
        assert state.stats.alias_hits == {}

    def test_same_cycle_constant_load_does_not_mark_the_old_snapshot(self):
        state = IpuState()
        state.write_constant_ones(0)
        state.xmem.write_address(128, bytes([1]) * 128)
        state.regfile.set_cr(2, 1)
        code = """LDR_MULT_REG R0 LR0 CR2;;
LDR_CYCLIC_MULT_REG LR0 CR0 LR0 ; MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD;;
MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD;; BKPT;;"""
        load_program(state, assemble_program(code))
        run_until_complete(state)
        assert state.stats.alias_hits == {"A2_ADD_VV": 1}
        assert state.stats.mult_active_cycles == 2
