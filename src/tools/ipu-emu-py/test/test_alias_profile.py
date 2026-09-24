"""Integration and extension-contract tests for optional hardware measurements."""
import gc
import json
import math
import struct
import weakref

import pytest
from ipu_as.lark_tree import assemble
from ipu_emu.alias_profile import AliasDefinition, AliasProfile, DetectorRegistry, Event, Match
from ipu_emu.alias_detectors import default_registry
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_state import IpuState


def run(code, *, enabled=True, setup=None, wide=False):
    state = IpuState(alias_profile=AliasProfile() if enabled else None, wide_vector_debug=wide)
    if setup:
        setup(state)
    load_program(state, [decode_instruction_word(w) for w in assemble(code)])
    run_until_complete(state)
    return state


def rows(state, alias):
    return state.alias_profile.to_dict()["aliases"][alias]["measurements"]


def count(state, alias, status="verified", subtype=None):
    return sum(r["count"] for r in rows(state, alias) if r["status"] == status and
               (subtype is None or r["subtype"] == subtype))


def test_profiled_state_is_freed_without_the_cycle_collector():
    """state -> profile -> observer must not point back at the state: a
    reference cycle would keep every profiled state's XMEM alive until the
    cyclic garbage collector happens to run."""
    state = run("MULT.RC.VE LR0 CR1 0 LR0 CR15; ACC.ADD.FIRST;;\nBKPT;;")
    profile = state.alias_profile
    released = weakref.ref(state)
    gc.disable()
    try:
        del state
        assert released() is None
    finally:
        gc.enable()
    assert profile.to_dict()["totals"]["cycles"] > 0


def test_registry_extension_needs_no_emulator_changes():
    class Custom:
        def observe(self, event):
            if event.instruction == "ACC.ADD.FIRST":
                yield Match("custom_move", (event,), metrics={"observed": 1})
    registry = default_registry().copy().register(AliasDefinition("custom_move", "Custom"), Custom)
    state = IpuState(alias_profile=AliasProfile(registry))
    load_program(state, [decode_instruction_word(w) for w in assemble(
        "MULT.RC.VE LR0 CR1 0 LR0 CR15; ACC.ADD.FIRST;;\nBKPT;;")])
    run_until_complete(state)
    assert count(state, "custom_move") == 1
    assert "custom_move" not in default_registry().entries
    assert {f'A{i}' for i in range(1, 17)} <= {key.split('_')[0] for key in default_registry().entries}
    with pytest.raises(ValueError):
        registry.register(AliasDefinition("custom_move", "Duplicate"), Custom)


def test_profile_preserves_architecture_and_identity_accounting():
    code = "MULT.RC.VE LR0 CR1 0 LR0 CR15; ACC.ADD.FIRST;;\nBKPT;;"
    on, off = run(code), run(code, enabled=False)
    for field in ("total_cycles", "mult_active_cycles", "mult_identity_cycles", "mult_lane_ops", "alias_hits"):
        assert getattr(on.stats, field) == getattr(off.stats, field)
    assert on.regfile.get_r_acc_bytes() == off.regfile.get_r_acc_bytes()
    assert count(on, "A1_MOV_RC") == count(on, "A13_NARROW_MULT") == 1
    data = json.loads(on.alias_profile.to_json())
    assert any({"A1_MOV_RC", "A13_NARROW_MULT"} <= set(group["aliases"])
               for group in data["overlap_instruction_groups"])
    assert data["totals"]["cycles"] == on.stats.total_cycles


TRANSFER = """
MULT.RC.VE LR0 CR1 0 LR0 CR15; ACC.ADD.FIRST;;
ACTIVATE.QUANTIZE identity CR15; STR_POST_AAQ_REG LR0 CR0;;
LDR_CYCLIC_MULT_REG LR0 CR0 LR0;;
MULT.RC.VV LR0 R0 0 LR0 CR15; ACC.ADD.FIRST;;
BKPT;;
"""


def test_roundtrip_and_acc_multiply_share_evidence():
    state = run(TRANSFER, wide=True)
    assert count(state, "A8_MOV_ACC") == 1
    assert count(state, "A11_ACC_MUL") == 1
    row = rows(state, "A8_MOV_ACC")[0]
    assert row["metrics"]["read_bytes"] == 512
    assert row["metrics"]["write_bytes"] == 512
    assert row["metrics"]["lossy_transfers"] == 0


def test_nonidentity_activation_is_not_a_move():
    state = run(TRANSFER.replace("identity", "relu"), wide=True)
    assert count(state, "A8_MOV_ACC") == count(state, "A11_ACC_MUL") == 0


def test_same_cycle_load_not_visible_to_mult():
    state = run(TRANSFER.replace("LR0;;\nMULT.RC.VV", "LR0; MULT.RC.VV"), wide=True)
    assert count(state, "A8_MOV_ACC") == 1
    assert count(state, "A11_ACC_MUL") == 0


def test_constant_exp_and_fractional():
    """A setup-written row verifies on its own; the program never writes address 0."""
    def setup(state):
        state.xmem.write_address(0, struct.pack("<f", math.log2(math.e)) * 128)
    state = run("""
LDR_MULT_REG R0 LR0 CR0;;
MULT.RC.VV LR0 R0 0 LR0 CR15; ACC.ADD.FIRST;;
ACTIVATE.QUANTIZE exp2 CR15;;
BKPT;;
""", setup=setup, wide=True)
    assert count(state, "A9_EXP", "verified") == 1
    assert count(state, "A10_FRACTIONAL_SCALAR", "verified") == 1
    assert state.stats.mult_identity_cycles == 0


def test_program_written_row_is_not_an_automatic_constant():
    """Only rows the program never wrote self-verify.

    log2(e) starts at row 1; the program copies it to row 0 and the exp2 chain
    reads both of its operands back from row 0. A9 still matches, but on data this
    run produced, so it stays a candidate -- which is what stops the automatic rule
    from declaring anything that merely sits still.
    """
    def setup(state):
        state.xmem.write_address(512, struct.pack("<f", math.log2(math.e)) * 128)
        state.regfile.set_lr(1, 1)   # source row 1 == address 512
        state.regfile.set_cr(2, 0)   # store destination row 0
    state = run("""
LDR_CYCLIC_MULT_REG LR1 CR0 LR0;;
MULT.RC.VE LR0 CR1 0 LR0 CR15; ACC.ADD.FIRST;;
ACTIVATE.QUANTIZE identity CR15; STR_POST_AAQ_REG LR0 CR2;;
LDR_MULT_REG R0 LR0 CR0;;
LDR_CYCLIC_MULT_REG LR0 CR0 LR0;;
MULT.RC.VV LR0 R0 0 LR0 CR15; ACC.ADD.FIRST;;
ACTIVATE.QUANTIZE exp2 CR15;;
BKPT;;
""", setup=setup, wide=True)
    assert count(state, "A9_EXP", "candidate") == 1
    assert count(state, "A9_EXP", "verified") == 0


def test_repeated_reads_and_write_invalidation():
    state = run("""
LDR_MULT_REG R0 LR0 CR0;;
LDR_MULT_REG R1 LR0 CR0;;
STR_POST_AAQ_REG LR0 CR0;;
LDR_MULT_REG R0 LR0 CR0;;
BKPT;;
""")
    assert count(state, "A14_MULTIPLE_ACC") == 1
    assert state.alias_profile.read_bytes == 384
    assert state.alias_profile.write_bytes == 512


def test_mask_and_scalar_add_subtypes():
    def setup(state):
        state.xmem.write_address(0, bytes([0, 1]) * 64)
    state = run("""
LDR_MULT_REG R0 LR0 CR0;;
MULT.RC.VV LR0 R0 0 LR0 CR15; ACC.ADD.FIRST;;
ADD LR6 LR6 LR7;;
ADD LR6 LR6 LR7;;
BKPT;;
""", setup=setup)
    assert count(state, "A15_SELECT", subtype="vector_mask") == 1
    assert count(state, "A15_SELECT", subtype="scalar_add_chain") == 1


def test_address_update_dependency():
    state = run("""
ADD LR2 LR2 CR1;;
LDR_MULT_REG R0 LR2 CR0;;
MULT.RC.VV LR0 R0 0 LR0 CR15; ACC.ADD.FIRST;;
BKPT;;
""")
    assert count(state, "A12_ADDRESSING", subtype="address_update") == 1
    assert count(state, "A12_ADDRESSING", subtype="load_to_mult") == 1


def test_overlap_and_bounded_examples():
    registry = DetectorRegistry()
    class Empty:
        def observe(self, event):
            return ()
    for key in ("a", "b"):
        registry.register(AliasDefinition(key, key), Empty)
    profile = AliasProfile(registry, examples=1)
    a, b = Event(1, 1, 0, "mult", "MULT"), Event(2, 1, 0, "acc", "ACC")
    profile.record(Match("a", (a, b)))
    profile.record(Match("b", (a,)))
    profile.record(Match("a", (a, b)))
    report = profile.to_dict()
    assert report["totals"]["unique_profiled_instructions"] == 2
    assert report["totals"]["unique_profiled_cycles"] == 1
    assert len(report["aliases"]["a"]["measurements"][0]["examples"]) == 1


def test_segmented_fold_geometry():
    def setup(state):
        state.set_cr_dstructure(valid_elements=64, partition=2)
        state.regfile.set_r_mask(((1 << 64) - 1).to_bytes(16, 'little') + bytes(112))
        state.regfile.set_cr(2, 64)
    state = run("""
MULT.RC.VE LR0 CR1 0 LR0 CR15; ACC.ADD.FIRST;;
ACTIVATE.QUANTIZE identity CR15; STR_POST_AAQ_REG LR0 CR0;;
LDR_CYCLIC_MULT_REG LR0 CR0 LR0;;
MULT.RC.VE LR0 CR1 0 LR0 CR15; ACC.ADD.FIRST;;
SET LR3 CR2;;
MULT.RC.VE LR3 CR1 0 LR0 CR15; ACC.ADD;;
BKPT;;
""", setup=setup, wide=True)
    assert count(state, "A7_REDUCE_SEG") == 1
    assert rows(state, "A7_REDUCE_SEG")[0]["metrics"]["partitions"] == 2


@pytest.mark.parametrize("taken", [True, False])
def test_conditional_select_both_paths(taken):
    def setup(state):
        state.regfile.set_lr(1, 0 if taken else 1)
    state = run("""
BLT LR1 CR1 full;;
SET LR6 CR0;;
B joined;;
full:
SET LR6 CR1;;
joined:
BKPT;;
""", setup=setup)
    assert count(state, "A15_SELECT", subtype="conditional_select") == 1
    assert state.regfile.get_lr(6) == int(taken)


def test_register_overwrite_revokes_transfer_provenance():
    from ipu_emu.ipu import Ipu
    state = IpuState(alias_profile=AliasProfile(), wide_vector_debug=True)
    load_program(state, [decode_instruction_word(w) for w in assemble(TRANSFER)])
    engine = Ipu(state)
    for _ in range(3):
        engine.execute_vliw_cycle()
    # Same bytes, new register value: it is no longer the recorded load.
    state.regfile.set_r_cyclic_wide_debug_at(0, bytes(512))
    engine.execute_vliw_cycle()
    assert count(state, "A11_ACC_MUL") == 0


def test_empty_cycles_and_break_resume_count_once():
    from ipu_emu.ipu import Ipu, BreakResult
    state = IpuState(alias_profile=AliasProfile())
    engine = Ipu(state)
    engine.execute_vliw_cycle()
    assert state.alias_profile.cycles == 1
    load_program(state, [decode_instruction_word(w) for w in assemble("BREAK; MULT.RC.VE LR0 CR1 0 LR0 CR15;;")])
    state.program_counter = 0
    assert engine.execute_vliw_cycle() == BreakResult.BREAK
    assert state.alias_profile.cycles == 1
    engine.execute_vliw_cycle_skip_break()
    assert state.alias_profile.cycles == 2
    assert count(state, "A1_MOV_RC") == 1


def test_rotating_fold_requires_complete_duplicated_input():
    code = """
MULT.RC.VE LR0 CR1 0 LR0 CR15; ACC.ADD.FIRST;;
ACTIVATE.QUANTIZE identity CR15; STR_POST_AAQ_REG LR0 CR0;;
LDR_CYCLIC_MULT_REG LR0 CR0 LR0;;
SET LR3 CR2;;
LDR_CYCLIC_MULT_REG LR0 CR0 LR3;;
MULT.RC.VE LR0 CR1 0 LR0 CR15; ACC.MAX.FIRST;;
SET LR3 CR3;;
MULT.RC.VE LR3 CR1 0 LR0 CR15; ACC.MAX;;
BKPT;;
"""
    def setup(state):
        state.regfile.set_cr(2, 128)
        state.regfile.set_cr(3, 64)
    yes = run(code, setup=setup, wide=True)
    assert count(yes, "A7_REDUCE_SEG", subtype="rotating_partition_fold") == 1
    no = run(code.replace('LDR_CYCLIC_MULT_REG LR0 CR0 LR3;;', 'NOP;;'), setup=setup, wide=True)
    assert count(no, "A7_REDUCE_SEG") == 0


def test_hardware_only_alias_and_dtype_support():
    with pytest.warns(UserWarning):
        state = run('STR_ACC_REG LR0 CR0;;\nBKPT;;')
    assert count(state, 'A16_STR_ACC_REG') == 1
    assert rows(state, 'A16_STR_ACC_REG')[0]['metrics']['write_bytes'] == 512
    assert state.alias_profile.to_dict()['aliases']['A10_FRACTIONAL_SCALAR']['status'] == 'unsupported_mode'


def test_roundtrip_does_not_include_store_padding_or_mask_loads():
    code = TRANSFER.replace('LDR_CYCLIC_MULT_REG LR0 CR0 LR0;;', 'LDR_MULT_MASK_REG LR0 CR0;;')
    assert count(run(code), 'A8_MOV_ACC') == 0
    code = TRANSFER.replace('LDR_CYCLIC_MULT_REG LR0 CR0 LR0;;', 'LDR_CYCLIC_MULT_REG LR0 CR1 LR0;;')
    assert count(run(code), 'A8_MOV_ACC') == 0


def test_exp_rebase_with_offset_is_candidate_until_offset_is_proven():
    def setup(state):
        state.xmem.write_address(0, struct.pack('<f', math.log2(math.e)) * 128)
    state = run('''
LDR_MULT_REG R0 LR0 CR0;;
MULT.RC.VV LR0 R0 0 LR0 CR15; ACC.ADD.FIRST;;
MULT.EE LR0 CR1 0 LR0 CR15; ACC.SUB;;
ACTIVATE.QUANTIZE exp2 CR15;;
BKPT;;
''', setup=setup, wide=True)
    assert count(state, 'A9_EXP', 'candidate', 'rebased_with_offset') == 1
    assert count(state, 'A9_EXP') == 0


@pytest.mark.parametrize('slot,activation,verified', [(0, 'identity', True), (1, 'identity', False), (0, 'relu', False)])
def test_exp_offset_traces_packed_reduction_and_memory(slot, activation, verified):
    def setup(state):
        state.xmem.write_address(0, struct.pack('<f', math.log2(math.e)) * 128)
        state.regfile.set_lr(1, 128 + slot)
        state.regfile.set_cr(2, 1)
    state = run(f'''
LDR_MULT_REG R0 LR0 CR0;;
MULT.RC.VV LR0 R0 0 LR0 CR15; AGG.MAX.FIRST LR0 CR15;;
ACTIVATE.QUANTIZE {activation} CR15; STR_POST_AAQ_REG LR0 CR2;;
LDR_MULT_REG R1 LR0 CR2;;
MULT.RC.VV LR0 R0 0 LR0 CR15; ACC.ADD.FIRST;;
MULT.EE LR1 CR1 0 LR0 CR15; ACC.SUB;;
ACTIVATE.QUANTIZE exp2 CR15;;
BKPT;;
''', setup=setup, wide=True)
    assert count(state, 'A9_EXP', 'verified' if verified else 'candidate') == 1


@pytest.mark.parametrize('overwrite', ['none', 'memory', 'register'])
def test_exp_offset_same_bytes_do_not_preserve_overwritten_provenance(overwrite):
    from ipu_emu.ipu import Ipu
    state = IpuState(alias_profile=AliasProfile(), wide_vector_debug=True)
    state.xmem.write_address(0, struct.pack('<f', math.log2(math.e)) * 128)
    state.regfile.set_lr(1, 128)
    state.regfile.set_cr(2, 1)
    load_program(state, [decode_instruction_word(w) for w in assemble('''
LDR_MULT_REG R0 LR0 CR0;;
MULT.RC.VV LR0 R0 0 LR0 CR15; AGG.MAX.FIRST LR0 CR15;;
ACTIVATE.QUANTIZE identity CR15; STR_POST_AAQ_REG LR0 CR2;;
LDR_MULT_REG R1 LR0 CR2;;
MULT.RC.VV LR0 R0 0 LR0 CR15; ACC.ADD.FIRST;;
MULT.EE LR1 CR1 0 LR0 CR15; ACC.SUB;;
ACTIVATE.QUANTIZE exp2 CR15;;
BKPT;;
''')])
    engine = Ipu(state)
    for _ in range(3):
        engine.execute_vliw_cycle()
    data = state.xmem.read_address(512, 512)
    if overwrite == 'memory':
        state.xmem.write_address(512, data)
    engine.execute_vliw_cycle()
    if overwrite == 'register':
        state.regfile.set_r_wide_debug(1, data)
    for _ in range(3):
        engine.execute_vliw_cycle()
    assert count(state, 'A9_EXP', 'verified' if overwrite == 'none' else 'candidate') == 1


@pytest.mark.parametrize('dtype_name', ['INT8', 'E4', 'E5'])
def test_binary_vector_mask_uses_active_dtype(dtype_name):
    from ipu_emu.ipu_math import DType, dtype_one_byte
    dtype = DType[dtype_name]
    state = IpuState(dtype=dtype, alias_profile=AliasProfile())
    state.xmem.write_address(0, bytes([0, dtype_one_byte(dtype)]) * 64)
    load_program(state, [decode_instruction_word(w) for w in assemble('''
LDR_MULT_REG R0 LR0 CR0;;
MULT.RC.VV LR0 R0 0 LR0 CR15;;
BKPT;;
''')])
    run_until_complete(state)
    assert count(state, 'A15_SELECT', subtype='vector_mask') == 1


def test_address_update_overwritten_before_live_read_is_not_a_dependency():
    state = run('''
ADD LR2 LR2 CR1;;
SET LR2 CR0; LDR_MULT_REG R0 LR2 CR0;;
BKPT;;
''')
    assert count(state, 'A12_ADDRESSING', subtype='address_update') == 0


def test_mask_shift_update_is_not_address_arithmetic():
    state = run('''
ADD LR2 LR2 CR1;;
MULT.RC.VE LR0 CR1 0 LR2 CR15;;
BKPT;;
''')
    assert count(state, 'A12_ADDRESSING', subtype='address_update') == 0


@pytest.mark.parametrize('address, expected', [(1, 0), (128, 1)])
def test_indexed_memory_versions_distinguish_overlap(address, expected):
    from ipu_emu.ipu import Ipu
    state = IpuState(alias_profile=AliasProfile())
    load_program(state, [decode_instruction_word(w) for w in assemble('''
LDR_MULT_REG R0 LR0 CR0;;
LDR_MULT_REG R1 LR0 CR0;;
BKPT;;
''')])
    engine = Ipu(state)
    engine.execute_vliw_cycle()
    state.xmem.write_address(address, bytes(1))
    engine.execute_vliw_cycle()
    assert count(state, 'A14_MULTIPLE_ACC') == expected


@pytest.mark.parametrize('instruction, expected', [
    ('MULT.RC.VE LR0 CR1 0 LR0 CR15; ACC.ADD.FIRST', 'A1_MOV_RC'),
    ('MULT.RC.VE LR0 CR1 0 LR0 CR15; ACC.ADD', 'A2_ADD_VV'),
    ('MULT.RC.VE LR0 CR1 0 LR0 CR15; ACC.SUB', 'A3_SUB_VV'),
    ('MULT.RC.VE LR0 CR1 0 LR0 CR15; AGG.SUM.FIRST LR3 CR15', 'A4_AGG_RC'),
    ('MULT.EE LR0 CR1 0 LR0 CR15; ACC.ADD.FIRST', 'A5_BCAST'),
    ('MULT.RC.VE LR0 CR1 1 LR0 CR15; ACC.ADD.FIRST', 'A6_GATHER_SCATTER'),
])
def test_legacy_alias_profile_includes_coissued_consumer(instruction, expected):
    state = run(instruction + ';;\nBKPT;;')
    assert count(state, expected) == 1
    assert rows(state, expected)[0]['metrics']['participating_instructions'] == 2
    assert rows(state, expected)[0]['metrics']['participating_cycles'] == 1


def test_rotation_cannot_mix_accumulator_memory_versions():
    def setup(state):
        state.regfile.set_cr(2, 128)
        state.regfile.set_cr(3, 64)
    state = run('''
MULT.RC.VE LR0 CR1 0 LR0 CR15; ACC.ADD.FIRST;;
ACTIVATE.QUANTIZE identity CR15; STR_POST_AAQ_REG LR0 CR0;;
LDR_CYCLIC_MULT_REG LR0 CR0 LR0;;
SET LR3 CR2;;
LDR_CYCLIC_MULT_REG LR0 CR0 LR3;;
MULT.RC.VE LR0 CR1 0 LR0 CR15; ACC.MAX.FIRST;;
ACTIVATE.QUANTIZE identity CR15; STR_POST_AAQ_REG LR0 CR0;;
LDR_CYCLIC_MULT_REG LR0 CR0 LR0;;
LDR_CYCLIC_MULT_REG LR0 CR0 LR3;;
SET LR3 CR3;;
MULT.RC.VE LR3 CR1 0 LR0 CR15; ACC.MAX;;
BKPT;;
''', setup=setup, wide=True)
    assert count(state, 'A7_REDUCE_SEG') == 0
