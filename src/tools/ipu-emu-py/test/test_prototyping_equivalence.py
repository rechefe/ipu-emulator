"""Prototyping mode must change speed and nothing else.

Its shortcuts (skipping NOP slots, caching per-PC dispatch work, short-circuiting
an all-ones multiply mask) are only defensible if the state they leave behind is
*identical* to the faithful path's — not close, identical. In a flagged build,
every program runs through faithful and prototype engines in lockstep; every
register, all 8 MiB of XMEM, snapshots, cycle outcomes and run statistics are
compared after every cycle.

The Bazel flag is also a capability boundary. A default build verifies that the
fast paths cannot be selected; only a flagged build executes the two-mode cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from ipu_emu.emulator import load_program
from ipu_emu.execute import BreakResult, Ipu, decode_instruction_word
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.prototyping import build_enabled, default_enabled

from ipu_as.lark_tree import assemble


_REQUIRES_PROTO_BUILD = pytest.mark.skipif(
    not build_enabled(), reason="requires --define=ipu_proto=1"
)


# Programs chosen for what they make the two paths do differently: NOP-heavy
# slots, branches (the cond slot's NOP advances the PC), an explicitly loaded
# mask, all three LR sub-slots at once, and both wide arithmetic modes.
PROGRAMS = {
    "mult_acc_agg": """
        SET LR0 CR0 ; SET LR1 CR1 ;;
        LDR_MULT_REG R0 LR0 CR2 ;;
        LDR_CYCLIC_MULT_REG LR0 CR3 LR0 ;;
        MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD.FIRST ;;
        MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.ADD ;;
        MULT.RC.VS LR0 0 LR0 CR15 ; ACC.MAX ;;
        MULT.VE LR0 CR1 0 LR0 CR15 ; ACC.SUB ;;
        MULT.EE LR0 CR1 0 LR0 CR15 ; AGG.SUM.FIRST LR1 CR15 ;;
        MULT.RC.VV LR0 R0 0 LR0 CR15 ; AGG.MAX LR1 CR15 ;;
        ACTIVATE.QUANTIZE relu CR15 ; STR_POST_AAQ_REG LR1 CR4 ;;
        BKPT ;;
    """,
    "masked": """
        SET LR0 CR0 ; SET LR2 CR1 ;;
        LDR_MULT_MASK_REG LR0 CR6 ;;
        LDR_CYCLIC_MULT_REG LR0 CR3 LR0 ;;
        MULT.RC.VS LR0 0 LR2 CR15 ; ACC.ADD.FIRST ;;
        MULT.RC.VS LR0 1 LR0 CR15 ; ACC.ADD ;;
        ACTIVATE.QUANTIZE identity CR15 ; STR_POST_AAQ_REG LR0 CR4 ;;
        BKPT ;;
    """,
    "loop_and_lr_ops": """
        SET LR0 CR0 ; SET LR1 CR1 ; SET LR3 CR7 ;;
        INC LR0 1 ; DEC LR3 1 ; INCR_MOD_POW2 LR1 CR1 3 ;;
        ADDB LRD4 LR1 ;;
        ADDBI LRD6 3 ;;
    loop:
        ADD LR0 LR0 CR1 ; SUB LR3 LR3 CR1 ;;
        BLT LR0 CR7 loop ;;
        BEQ LR0 CR7 done ;;
        SET LR0 CR0 ;;
    done:
        BKPT ;;
    """,
}


def _state(*, wide: bool, arithmetic, prototyping: bool, seed: int) -> IpuState:
    if wide:
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=arithmetic,
            wide_vector_quantize_output=False,
            prototyping=prototyping,
        )
    else:
        state = IpuState(prototyping=prototyping)
    state.dtype = DType.INT8

    rng = np.random.RandomState(seed)
    row = 512 if wide else 128
    if wide and arithmetic == WideVectorArithmetic.FP32:
        payload = rng.randn(128 * 6).astype(np.float32).tobytes()
    elif wide:
        payload = rng.randint(-2000, 2000, 128 * 6).astype(np.int32).tobytes()
    else:
        payload = bytes(rng.randint(0, 256, 128 * 6, dtype=np.uint8))
    state.xmem.write_address(0x10000, payload)

    state.regfile.set_cr(2, 0x10000 // row)          # R0 source row
    state.regfile.set_cr(3, 0x10000 // row + 1)      # cyclic source row
    state.regfile.set_cr(4, 0x30000 // row)          # post-AAQ store row
    state.regfile.set_cr(6, 0x10000 // row + 2)      # mask row
    state.regfile.set_cr(7, 5)                       # loop bound
    state.set_cr_dstructure(valid_elements=128, partition=4)
    return state


def _assert_regfiles_identical(regfile_a, regfile_b) -> None:
    assert regfile_a._storage.keys() == regfile_b._storage.keys()
    for name, buf in regfile_a._storage.items():
        assert bytes(buf) == bytes(regfile_b._storage[name]), name


def _assert_states_identical(state_a: IpuState, state_b: IpuState) -> None:
    assert state_a.program_counter == state_b.program_counter
    _assert_regfiles_identical(state_a.regfile, state_b.regfile)
    assert state_a.xmem._data == state_b.xmem._data
    assert state_a.stats == state_b.stats


def _run_both(asm: str, *, wide: bool, arithmetic=None, seed: int = 5):
    decoded = [decode_instruction_word(word) for word in assemble(asm)]
    states = [
        _state(wide=wide, arithmetic=arithmetic, prototyping=mode, seed=seed)
        for mode in (False, True)
    ]
    for state in states:
        load_program(state, [dict(inst) for inst in decoded])
    engines = [Ipu(state) for state in states]

    retained_snapshot = None
    retained_bytes = None
    cycles = 0
    while not states[0].is_halted:
        assert not states[1].is_halted
        outcomes = [engine.execute_vliw_cycle() for engine in engines]
        assert outcomes[0] == outcomes[1]
        if outcomes[0] == BreakResult.BREAK:
            for engine in engines:
                engine.execute_vliw_cycle_skip_break()

        cycles += 1
        if cycles >= 100_000:
            raise RuntimeError("equivalence program exceeded 100000 cycles")
        _assert_states_identical(*states)

        snapshots = [engine.snapshot for engine in engines]
        assert (snapshots[0] is None) == (snapshots[1] is None)
        if snapshots[0] is not None:
            _assert_regfiles_identical(*snapshots)
            if retained_snapshot is None:
                retained_snapshot = snapshots[1]
                retained_bytes = {
                    name: bytes(buf)
                    for name, buf in retained_snapshot._storage.items()
                }
            else:
                assert all(
                    bytes(retained_snapshot._storage[name]) == original
                    for name, original in retained_bytes.items()
                )

    assert states[1].is_halted
    for state in states:
        state.stats.total_cycles = cycles
    _assert_states_identical(*states)
    return [(states[0], cycles), (states[1], cycles)]


def _assert_identical(faithful, fast) -> None:
    (state_a, cycles_a), (state_b, cycles_b) = faithful, fast
    assert cycles_a == cycles_b
    _assert_states_identical(state_a, state_b)


@_REQUIRES_PROTO_BUILD
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_narrow_int8_programs_are_identical(name):
    faithful, fast = _run_both(PROGRAMS[name], wide=False)
    _assert_identical(faithful, fast)


@_REQUIRES_PROTO_BUILD
@pytest.mark.parametrize("name", sorted(PROGRAMS))
@pytest.mark.parametrize(
    "arithmetic", [WideVectorArithmetic.FP32, WideVectorArithmetic.INT32]
)
def test_wide_vector_programs_are_identical(name, arithmetic):
    faithful, fast = _run_both(PROGRAMS[name], wide=True, arithmetic=arithmetic)
    _assert_identical(faithful, fast)


@_REQUIRES_PROTO_BUILD
def test_the_cond_slots_nop_still_advances_the_pc():
    """The one NOP that is not a no-op.

    Skipping it would leave the PC parked on the same instruction forever, so a
    program with no branch at all is the sharpest check that prototyping mode
    does not skip it.
    """
    asm = "SET LR0 CR1 ;;\nADD LR0 LR0 CR1 ;;\nBKPT ;;"
    faithful, fast = _run_both(asm, wide=False)
    _assert_identical(faithful, fast)
    assert fast[1] == 3


def _wide_fp32_activation_state(
    activation: str, payload: bytes, prototyping: bool
) -> tuple[IpuState, Ipu]:
    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
        wide_vector_quantize_output=False,
        prototyping=prototyping,
    )
    state.dtype = DType.INT8
    state.set_cr_dstructure(valid_elements=128)
    state.regfile.raw("r_acc")[:] = payload
    state.regfile.raw("post_aaq_reg")[:] = b"\xa5" * 512
    decoded = [
        decode_instruction_word(word)
        for word in assemble(f"ACTIVATE.QUANTIZE {activation} CR15 ;;\n")
    ]
    load_program(state, decoded)
    return state, Ipu(state)


@_REQUIRES_PROTO_BUILD
@pytest.mark.parametrize("activation", ["identity", "exp2", "reciprocal"])
def test_full_row_fp32_activation_fast_path_is_bit_exact(activation):
    rng = np.random.RandomState(37)
    lanes = rng.uniform(-20.0, 20.0, 128).astype(np.float32)
    bits = lanes.view(np.uint32)
    bits[:8] = np.array(
        [
            0x00000000,  # +0
            0x80000000,  # -0
            0x7F800000,  # +inf
            0xFF800000,  # -inf
            0x7FC00001,  # quiet NaN with payload
            0x7F800001,  # signaling NaN with payload
            0x3F800000,  # +1
            0xBF800000,  # -1
        ],
        dtype=np.uint32,
    )
    payload = bits.tobytes()
    states = []
    outcomes = []

    for prototyping in (False, True):
        state, ipu = _wide_fp32_activation_state(
            activation, payload, prototyping
        )
        outcomes.append(ipu.execute_vliw_cycle())
        states.append(state)

    assert outcomes[0] == outcomes[1]
    _assert_states_identical(*states)


@_REQUIRES_PROTO_BUILD
@pytest.mark.parametrize(
    ("activation", "overflow_bits"),
    [
        ("exp2", 0x447A0000),        # 1000.0: math.exp overflows
        ("reciprocal", 0x00000001),  # reciprocal cannot narrow to float32
    ],
)
def test_full_row_fp32_activation_exception_state_is_bit_exact(
    activation, overflow_bits
):
    bits = np.full(128, 0x3F800000, dtype=np.uint32)
    bits[2] = overflow_bits
    payload = bits.tobytes()
    states = []
    exceptions = []

    for prototyping in (False, True):
        state, ipu = _wide_fp32_activation_state(
            activation, payload, prototyping
        )
        with pytest.raises(OverflowError) as caught:
            ipu.execute_vliw_cycle()
        states.append(state)
        exceptions.append(caught.value)

    assert type(exceptions[0]) is type(exceptions[1])
    assert exceptions[0].args == exceptions[1].args
    _assert_states_identical(*states)


@_REQUIRES_PROTO_BUILD
def test_cached_scalar_read_invalid_index_exception_is_bit_exact():
    inst = decode_instruction_word(assemble("SET LR0 CR1 ;;\n")[0])
    cr_field = next(
        name
        for name in inst
        if name.startswith("lr_inst_0_") and name.endswith("_addbi_immediate")
    )
    inst[cr_field] = 999
    states = []
    exceptions = []

    for prototyping in (False, True):
        state = IpuState(prototyping=prototyping)
        load_program(state, [dict(inst)])
        with pytest.raises(AssertionError) as caught:
            Ipu(state).execute_vliw_cycle()
        states.append(state)
        exceptions.append(caught.value)

    assert type(exceptions[0]) is type(exceptions[1])
    assert exceptions[0].args == exceptions[1].args
    _assert_states_identical(*states)


@_REQUIRES_PROTO_BUILD
def test_an_lr_conflict_still_raises():
    """Conflict detection is precomputed per PC; it must still fire every time."""
    asm = "SET LR0 CR1 ; SET LR0 CR2 ;;\nBKPT ;;"
    decoded = [decode_instruction_word(word) for word in assemble(asm)]
    states = []
    exceptions = []
    for prototyping in (False, True):
        state = IpuState(prototyping=prototyping)
        load_program(state, [dict(inst) for inst in decoded])
        with pytest.raises(RuntimeError, match="LR conflict") as caught:
            Ipu(state).execute_vliw_cycle()
        states.append(state)
        exceptions.append(caught.value)

    assert type(exceptions[0]) is type(exceptions[1])
    assert exceptions[0].args == exceptions[1].args
    _assert_states_identical(*states)


def test_bazel_flag_is_a_mandatory_capability(monkeypatch):
    monkeypatch.delenv("IPU_EMU_PROTOTYPING", raising=False)
    assert IpuState(prototyping=False).prototyping is False
    assert IpuState().prototyping == default_enabled()

    if build_enabled():
        assert IpuState().prototyping is True
        assert IpuState(prototyping=True).prototyping is True
        monkeypatch.setenv("IPU_EMU_PROTOTYPING", "0")
        assert IpuState().prototyping is False
        monkeypatch.setenv("IPU_EMU_PROTOTYPING", "1")
        assert IpuState().prototyping is True
    else:
        assert IpuState().prototyping is False
        with pytest.raises(ValueError, match="--define=ipu_proto=1"):
            IpuState(prototyping=True)
        monkeypatch.setenv("IPU_EMU_PROTOTYPING", "1")
        with pytest.raises(ValueError, match="--define=ipu_proto=1"):
            IpuState()


@_REQUIRES_PROTO_BUILD
@pytest.mark.parametrize("patch_style", ["replace", "in_place"])
def test_debug_instruction_patch_invalidates_cached_plan(patch_style):
    """Continuing from BREAK must execute the debugger's current instruction."""
    break_inst = decode_instruction_word(assemble("BREAK ;;\n")[0])
    replacement = decode_instruction_word(assemble("SET LR0 CR2 ;;\n")[0])
    states = []

    for prototyping in (False, True):
        state = IpuState(prototyping=prototyping)
        state.regfile.set_cr(2, 99)
        state.inst_mem[0] = dict(break_inst)
        ipu = Ipu(state)

        assert ipu.execute_vliw_cycle() == BreakResult.BREAK
        retained = ipu.snapshot
        assert retained is not None
        retained_bytes = {
            name: bytes(buf) for name, buf in retained._storage.items()
        }

        if patch_style == "replace":
            state.inst_mem[0] = dict(replacement)
        else:
            state.inst_mem[0].clear()
            state.inst_mem[0].update(replacement)

        ipu.execute_vliw_cycle_skip_break()
        assert state.regfile.get_lr(0) == 99
        assert state.program_counter == 1
        assert all(
            bytes(retained._storage[name]) == original
            for name, original in retained_bytes.items()
        )
        states.append(state)

    _assert_states_identical(*states)
