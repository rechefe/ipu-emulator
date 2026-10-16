"""Prototyping mode must change speed and nothing else.

Its shortcuts (skipping NOP slots, caching per-PC dispatch work, short-circuiting
an all-ones multiply mask) are only defensible if the state they leave behind is
*identical* to the faithful path's — not close, identical. In a flagged build,
every program runs through faithful and prototype engines in lockstep; every
register, all 512 MiB of XMEM, snapshots, cycle outcomes and run statistics are
compared after every cycle.

The Bazel flag is also a capability boundary. A default build verifies that the
fast paths cannot be selected; only a flagged build executes the two-mode cases.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import timing_probes
from emu_support import ElementFormat, assemble_program, make_state, new_state, run_both

from ipu_emu.emulator import load_program
from ipu_emu.errors import DebugOnlyInstructionWarning, RegisterIndexError
from ipu_emu.execute import BreakResult, Ipu
from ipu_emu.ipu_state import IpuState
from ipu_emu.prototyping import (
    # TODO(E04-08): prototyping becomes an adapter; test its env parsing publicly.
    _env_requests_prototyping,
    build_enabled,
    default_enabled,
)

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


def _seed(fmt: ElementFormat, seed: int):
    """Return a setup callback that fills XMEM and the CRs the programs read."""
    wide = fmt in (ElementFormat.FP32, ElementFormat.INT32)

    def setup(state: IpuState) -> None:
        rng = np.random.RandomState(seed)
        row = 512 if wide else 128
        if fmt == ElementFormat.FP32:
            payload = rng.randn(128 * 6).astype(np.float32).tobytes()
        elif wide:
            payload = rng.randint(-2000, 2000, 128 * 6).astype(np.int32).tobytes()
        else:
            payload = bytes(rng.randint(0, 256, 128 * 6, dtype=np.uint8))
        state.xmem.write_address(0x10000, payload)

        state.regfile.set_cr(2, 0x10000 // row)  # R0 source row
        state.regfile.set_cr(3, 0x10000 // row + 1)  # cyclic source row
        state.regfile.set_cr(4, 0x30000 // row)  # post-AAQ store row
        state.regfile.set_cr(6, 0x10000 // row + 2)  # mask row
        state.regfile.set_cr(7, 5)  # loop bound
        state.set_cr_dstructure(valid_elements=128, partition=4)

    return setup


def _regfile_bytes(regfile) -> dict[str, bytes]:
    return {name: regfile.raw_readonly(name) for name in regfile.register_names}


def _assert_regfiles_identical(regfile_a, regfile_b) -> None:
    assert _regfile_bytes(regfile_a) == _regfile_bytes(regfile_b)


def _assert_states_identical(state_a: IpuState, state_b: IpuState) -> None:
    assert state_a.program_counter == state_b.program_counter
    _assert_regfiles_identical(state_a.regfile, state_b.regfile)
    assert state_a.xmem == state_b.xmem
    assert state_a.stats == state_b.stats


def _cycle_comparer():
    """An ``on_cycle`` callback for ``run_both``: the two states must be
    identical after every cycle, and a snapshot handed out by the prototyping
    engine must stay unchanged by later cycles."""
    retained = {}

    def compare(engines) -> None:
        _assert_states_identical(*(engine.state for engine in engines))
        snapshots = [engine.snapshot for engine in engines]
        assert (snapshots[0] is None) == (snapshots[1] is None)
        if snapshots[0] is None:
            return
        _assert_regfiles_identical(*snapshots)
        if not retained:
            retained["snapshot"] = snapshots[1]
            retained["bytes"] = _regfile_bytes(snapshots[1])
        else:
            assert _regfile_bytes(retained["snapshot"]) == retained["bytes"]

    return compare


def _run_equivalence(asm: str, *, fmt: ElementFormat = ElementFormat.INT8, seed: int = 5):
    """Run *asm* faithful and prototyping in lockstep; compare after every cycle."""
    faithful, fast = run_both(asm, fmt=fmt, setup=_seed(fmt, seed), on_cycle=_cycle_comparer())
    _assert_states_identical(faithful, fast)
    return faithful, fast


@_REQUIRES_PROTO_BUILD
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_narrow_int8_programs_are_identical(name):
    _run_equivalence(PROGRAMS[name])


@_REQUIRES_PROTO_BUILD
@pytest.mark.parametrize("name", sorted(PROGRAMS))
@pytest.mark.parametrize("fmt", [ElementFormat.FP32, ElementFormat.INT32])
def test_wide_vector_programs_are_identical(name, fmt):
    _run_equivalence(PROGRAMS[name], fmt=fmt)


# The timing probes (E02-05, E02-06): one minimal program per row of the
# register-timing table, each in every element format it is meaningful in.
_PROBE_RUNS = [
    pytest.param(name, mode, id=f"{name}-{mode.value}")
    for name, probe in timing_probes.PROBES.items()
    if name not in timing_probes.RAISES
    for mode in probe.modes
]


_PROBE_FORMATS = {
    timing_probes.Mode.NARROW: ElementFormat.INT8,
    timing_probes.Mode.WIDE_FP32: ElementFormat.FP32,
}


def _probe_states(name: str, mode) -> list[IpuState]:
    return [
        timing_probes.prepare(timing_probes.PROBES[name], mode, prototyping=prototyping)
        for prototyping in (False, True)
    ]


def _probe_setup(probe):
    def setup(state: IpuState) -> None:
        for index, value in probe.cr.items():
            state.regfile.set_cr(index, value)
        probe.setup(state)

    return setup


@_REQUIRES_PROTO_BUILD
@pytest.mark.parametrize(("name", "mode"), _PROBE_RUNS)
def test_timing_probes_are_identical(name, mode):
    probe = timing_probes.PROBES[name]
    # STR_ACC_REG is emulator-only and says so on every run.
    expected = (
        pytest.warns(DebugOnlyInstructionWarning)
        if "STR_ACC_REG" in probe.asm
        else contextlib.nullcontext()
    )
    with expected:
        faithful, fast = run_both(
            program=timing_probes.decoded_program(probe),
            fmt=_PROBE_FORMATS[mode],
            dstructure={"valid_elements": timing_probes.LANES},
            setup=_probe_setup(probe),
            on_cycle=_cycle_comparer(),
        )
    _assert_states_identical(faithful, fast)


#: Raising probes whose two modes are known to end in different states.
_KNOWN_DIVERGENT = {"lr_subslot_exception"}


def _run_until_raise(name: str):
    """Run a probe that raises in both modes; return each mode's exception and
    final state."""
    (mode,) = timing_probes.PROBES[name].modes
    states = _probe_states(name, mode)
    exceptions = []
    for state in states:
        engine = Ipu(state)
        with pytest.raises(timing_probes.RAISES[name]) as caught:
            for _ in range(timing_probes.MAX_CYCLES):
                engine.execute_vliw_cycle()
        exceptions.append(caught.value)
    return states, exceptions


@_REQUIRES_PROTO_BUILD
@pytest.mark.parametrize("name", sorted(set(timing_probes.RAISES) - _KNOWN_DIVERGENT))
def test_raising_timing_probes_are_identical(name):
    states, exceptions = _run_until_raise(name)
    assert type(exceptions[0]) is type(exceptions[1])
    assert exceptions[0].args == exceptions[1].args
    _assert_states_identical(*states)


@_REQUIRES_PROTO_BUILD
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="E04-07 closes the LR sub-slot divergence"
)
def test_lr_subslot_exception_partial_state_matches():
    """When an LR sub-slot's operand read raises, faithful mode has written no
    LR; prototyping mode runs sub-slots one at a time and has already written
    sub-slot 0's LR1. E04-07 makes the two agree; the strict xfail then fails
    and that PR removes the marker."""
    states, _ = _run_until_raise("lr_subslot_exception")
    _assert_states_identical(*states)


@_REQUIRES_PROTO_BUILD
def test_the_cond_slots_nop_still_advances_the_pc():
    """The one NOP that is not a no-op.

    Skipping it would leave the PC parked on the same instruction forever, so a
    program with no branch at all is the sharpest check that prototyping mode
    does not skip it.
    """
    asm = "SET LR0 CR1 ;;\nADD LR0 LR0 CR1 ;;\nBKPT ;;"
    _, fast = _run_equivalence(asm)
    assert fast.stats.total_cycles == 3


def _wide_fp32_activation_state(
    activation: str, payload: bytes, prototyping: bool
) -> tuple[IpuState, Ipu]:
    state = make_state(
        f"ACTIVATE.QUANTIZE {activation} CR15 ;;\n",
        fmt=ElementFormat.FP32,
        dstructure={"valid_elements": 128},
        prototyping=prototyping,
    )
    state.regfile.mutable("r_acc")[:] = payload
    state.regfile.mutable("post_aaq_reg")[:] = b"\xa5" * 512
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
        state, ipu = _wide_fp32_activation_state(activation, payload, prototyping)
        outcomes.append(ipu.execute_vliw_cycle())
        states.append(state)

    assert outcomes[0] == outcomes[1]
    _assert_states_identical(*states)


@_REQUIRES_PROTO_BUILD
@pytest.mark.parametrize(
    ("activation", "overflow_bits"),
    [
        ("exp2", 0x447A0000),  # 1000.0: math.exp overflows
        ("reciprocal", 0x00000001),  # reciprocal cannot narrow to float32
    ],
)
def test_full_row_fp32_activation_exception_state_is_bit_exact(activation, overflow_bits):
    bits = np.full(128, 0x3F800000, dtype=np.uint32)
    bits[2] = overflow_bits
    payload = bits.tobytes()
    states = []
    exceptions = []

    for prototyping in (False, True):
        state, ipu = _wide_fp32_activation_state(activation, payload, prototyping)
        with pytest.raises(OverflowError) as caught:
            ipu.execute_vliw_cycle()
        states.append(state)
        exceptions.append(caught.value)

    assert type(exceptions[0]) is type(exceptions[1])
    assert exceptions[0].args == exceptions[1].args
    _assert_states_identical(*states)


@_REQUIRES_PROTO_BUILD
def test_cached_scalar_read_invalid_index_exception_is_bit_exact():
    inst = assemble_program("SET LR0 CR1 ;;\n")[0]
    cr_field = next(
        name for name in inst if name.startswith("lr_inst_0_") and name.endswith("_addbi_immediate")
    )
    inst[cr_field] = 999
    states = []
    exceptions = []

    for prototyping in (False, True):
        state = new_state(prototyping=prototyping)
        load_program(state, [dict(inst)])
        with pytest.raises(RegisterIndexError) as caught:
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
    states = []
    exceptions = []
    for prototyping in (False, True):
        state = make_state(asm, prototyping=prototyping)
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
        monkeypatch.setenv("IPU_EMU_PROTOTYPING", "FALSE")
        assert IpuState().prototyping is False
        monkeypatch.setenv("IPU_EMU_PROTOTYPING", "1")
        assert IpuState().prototyping is True
    else:
        assert IpuState().prototyping is False
        with pytest.raises(ValueError, match="--define=ipu_proto=1"):
            IpuState(prototyping=True)
        monkeypatch.setenv("IPU_EMU_PROTOTYPING", "FALSE")
        assert IpuState().prototyping is False
        monkeypatch.setenv("IPU_EMU_PROTOTYPING", "1")
        with pytest.raises(ValueError, match="--define=ipu_proto=1"):
            IpuState()


@pytest.mark.parametrize("raw", ["FALSE", "False", "false", "NO", "Off", " 0 ", ""])
def test_env_false_values_are_case_insensitive(raw):
    assert _env_requests_prototyping(raw) is False


@pytest.mark.parametrize("raw", ["1", "true", "YES", "on", " On "])
def test_env_true_values_are_case_insensitive(raw):
    assert _env_requests_prototyping(raw) is True


@pytest.mark.parametrize("raw", ["maybe", "2", "enable"])
def test_env_unrecognised_value_raises(raw, monkeypatch):
    with pytest.raises(ValueError, match="IPU_EMU_PROTOTYPING"):
        _env_requests_prototyping(raw)
    monkeypatch.setenv("IPU_EMU_PROTOTYPING", raw)
    with pytest.raises(ValueError, match="IPU_EMU_PROTOTYPING"):
        IpuState()


@_REQUIRES_PROTO_BUILD
@pytest.mark.parametrize("patch_style", ["replace", "in_place"])
def test_debug_instruction_patch_invalidates_cached_plan(patch_style):
    """Continuing from BREAK must execute the debugger's current instruction."""
    break_inst = assemble_program("BREAK ;;\n")[0]
    replacement = assemble_program("SET LR0 CR2 ;;\n")[0]
    states = []

    for prototyping in (False, True):
        state = new_state(prototyping=prototyping, cr={2: 99})
        state.inst_mem[0] = dict(break_inst)
        ipu = Ipu(state)

        assert ipu.execute_vliw_cycle() == BreakResult.BREAK
        retained = ipu.snapshot
        assert retained is not None
        retained_bytes = _regfile_bytes(retained)

        if patch_style == "replace":
            state.inst_mem[0] = dict(replacement)
        else:
            state.inst_mem[0].clear()
            state.inst_mem[0].update(replacement)

        ipu.execute_vliw_cycle_skip_break()
        assert state.regfile.get_lr(0) == 99
        assert state.program_counter == 1
        assert _regfile_bytes(retained) == retained_bytes
        states.append(state)

    _assert_states_identical(*states)
