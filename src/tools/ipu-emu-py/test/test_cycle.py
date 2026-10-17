"""The Cycle module (E04-07): one implementation of one VLIW word.

``step()`` is ``evaluate_break()`` then ``retire()``; ``evaluate_break()`` has no
side effects; an idle word counts one idle multiplier cycle; ``retire()`` takes
the snapshot again, so a debugger's edit made at a break is visible.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from emu_support import assemble_program, make_state, new_state

from ipu_emu.cycle import Cycle, CycleOutcome, decode_word, schedule_word
from ipu_emu.emulator import load_program
from ipu_emu.errors import EmulatorError, EmulatorFault
from ipu_emu.execute import BreakResult, Ipu
from ipu_emu.geometry import effective_mult_mask, partition_vector
from ipu_emu.ipu_state import HaltRecord, IpuState
from ipu_emu.regfile import RegFile
from ipu_isa.faults import FaultId

# Every slot kind, a BREAK in the middle and a branch back: loads, multiplies,
# accumulates, aggregation, a store, LR arithmetic and the cond slot.
PROGRAM = """
    SET LR0 CR0 ; SET LR1 CR1 ;;
    LD.MULT R0 LR0 CR2 ;;
    LD.MULT.CYCLIC LR0 CR3 LR0 ;;
loop:
    MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD.FIRST ;;
    MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.ADD ; BREAK ;;
    MULT.EE LR0 CR1 0 LR0 CR15 ; AGG.SUM.FIRST LR1 CR15 ;;
    ACTIVATE.QUANTIZE relu CR15 ; ST.AAQ LR1 CR4 ; INC LR2 1 ;;
    BLT.S LR2 CR7 loop ;;
    HALT ;;
"""

STORE_ROW_ADDRESS = 0x30000


def _seed(state: IpuState) -> None:
    rng = np.random.RandomState(7)
    state.xmem.write_address(0x10000, bytes(rng.randint(0, 256, 128 * 4, dtype=np.uint8)))
    state.regfile.set_cr(2, 0x10000 // 128)  # R0 source row
    state.regfile.set_cr(3, 0x10000 // 128 + 1)  # cyclic source row
    state.regfile.set_cr(4, STORE_ROW_ADDRESS // 128)  # post-AAQ store row
    state.regfile.set_cr(7, 3)  # loop bound
    state.set_cr_dstructure(valid_elements=128, partition=4)


def _program_state(asm: str = PROGRAM, **kw) -> IpuState:
    state = new_state(setup=_seed, **kw)
    load_program(state, assemble_program(asm))
    return state


def _observable(state: IpuState) -> tuple:
    """Everything a word can change: PC, every register, the store row, stats."""
    return (
        state.program_counter,
        {name: bytes(state.regfile.raw_readonly(name)) for name in state.regfile.register_names},
        state.xmem.snapshot_bytes(STORE_ROW_ADDRESS, 128),
        copy.deepcopy(state.stats),
    )


def _cycle(state: IpuState, observer=None) -> Cycle:
    cycle = Cycle(state, Ipu(state))
    if observer is not None:
        cycle.add_observer(observer)
    return cycle


class _Recorder:
    """A CycleObserver that records its calls."""

    def __init__(self, state: IpuState) -> None:
        self.state = state
        self.calls: list[tuple] = []

    def begin(self, pc: int) -> None:
        self.calls.append(("begin", pc))

    def issued(self, facts) -> None:
        self.calls.append(("issued", facts.slot, facts.plan.inst_name))

    def retired(self, facts, result) -> None:
        self.calls.append(("retired", facts.slot, facts.plan.inst_name))

    def end(self) -> None:
        self.calls.append(("end", self.state.program_counter))


def test_step_equals_evaluate_break_then_retire():
    stepped, split = _program_state(), _program_state()
    by_step, by_parts = _cycle(stepped), _cycle(split)
    outcomes = []
    while not stepped.is_halted:
        outcome = by_step.step()
        if outcome is CycleOutcome.BREAK:
            by_step.retire()
        assert by_parts.evaluate_break() is outcome
        by_parts.retire()
        outcomes.append(outcome)
        assert _observable(stepped) == _observable(split)
    assert split.is_halted
    assert outcomes.count(CycleOutcome.BREAK) == 3  # once per loop iteration
    assert len(outcomes) > 8


@pytest.mark.parametrize("fires", [True, False])
def test_evaluate_break_has_no_side_effects(fires):
    # The word loads, multiplies, accumulates, stores and writes LRs.
    breaking = "BREAK" if fires else "BREAK.IFEQ LR1 5"
    asm = (
        "LD.MULT R0 LR0 CR2 ; MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD.FIRST ; "
        f"ST.AAQ LR0 CR4 ; SET LR0 CR3 ; INC LR3 2 ; {breaking} ;;\nHALT ;;"
    )
    state, twin = _program_state(asm), _program_state(asm)
    before = _observable(twin)

    outcome = _cycle(state).evaluate_break()

    assert outcome is (CycleOutcome.BREAK if fires else CycleOutcome.RETIRED)
    assert _observable(state) == before
    assert state.xmem == twin.xmem


def test_an_idle_word_counts_one_idle_cycle_and_advances_the_pc():
    state = new_state()
    load_program(state, [None, *assemble_program("HALT ;;")])
    cycle = _cycle(state)

    assert cycle.step() is CycleOutcome.RETIRED

    assert state.program_counter == 1
    assert state.stats.mult_idle_cycles == 1
    assert state.stats.mult_active_cycles == 0


def test_an_idle_word_split_at_its_break_still_counts_once():
    state = new_state()
    load_program(state, [None, *assemble_program("HALT ;;")])
    cycle = _cycle(state)

    assert cycle.evaluate_break() is CycleOutcome.RETIRED
    assert (state.program_counter, state.stats.mult_idle_cycles) == (0, 0)
    cycle.retire()
    assert (state.program_counter, state.stats.mult_idle_cycles) == (1, 1)


def test_without_honour_break_the_word_retires():
    state = make_state("SET LR0 CR8 ; BREAK ;;\nHALT ;;", cr={8: 42})
    cycle = _cycle(state)

    assert cycle.step(honour_break=False) is CycleOutcome.RETIRED

    assert state.regfile.get_lr(0) == 42
    assert state.program_counter == 1


def test_a_break_stops_before_any_side_effect():
    state = make_state("SET LR0 CR8 ; BREAK ;;\nHALT ;;", cr={8: 42})
    cycle = _cycle(state)

    assert cycle.step() is CycleOutcome.BREAK

    assert state.regfile.get_lr(0) == 0
    assert state.program_counter == 0


def test_an_edit_made_at_a_break_is_visible_to_the_resumed_word():
    state = make_state("SET LR0 CR8 ; ADD LR1 LR2 CR8 ; BREAK ;;\nHALT ;;", cr={8: 42})
    cycle = _cycle(state)
    assert cycle.evaluate_break() is CycleOutcome.BREAK

    # What a debugger does while stopped at the break.
    state.regfile.set_cr(8, 7)
    state.regfile.set_lr(2, 100)
    cycle.retire()

    assert state.regfile.get_lr(0) == 7
    assert state.regfile.get_lr(1) == 107


def test_reset_empties_the_schedule_cache_and_recopies_the_register_file(monkeypatch):
    state = make_state("SET LR0 CR8 ;;\nSET LR1 CR8 ;;\nHALT ;;", cr={8: 1})
    ipu = Ipu(state)
    ipu.cycle.step()
    ipu.cycle.step()
    assert ipu.cycle.cached_schedule(0) is not None
    assert ipu.cycle.cached_schedule(1) is not None

    copies = []
    real_snapshot = RegFile.snapshot

    def counting_snapshot(regfile):
        copies.append(regfile)
        return real_snapshot(regfile)

    monkeypatch.setattr(RegFile, "snapshot", counting_snapshot)
    ipu.cycle.reset()

    assert ipu.cycle.cached_schedule(0) is None
    assert ipu.cycle.cached_schedule(1) is None
    ipu.cycle.step()
    assert copies == [state.regfile]


def test_a_word_is_scheduled_once_per_pc(monkeypatch):
    import ipu_emu.cycle as cycle_mod

    state = make_state("loop:\nINC LR0 1 ;;\nBLT.S LR0 CR8 loop ;;\nHALT ;;", cr={8: 5})
    scheduled = []
    real = cycle_mod.schedule_word

    def counting(inst, **kw):
        scheduled.append(dict(inst))
        return real(inst, **kw)

    monkeypatch.setattr(cycle_mod, "schedule_word", counting)
    cycle = _cycle(state)
    while not state.is_halted:
        cycle.step()

    assert state.regfile.get_lr(0) == 5
    # Five loop iterations, but each of the three words is decoded once.
    assert len(scheduled) == 3


def test_writing_imem_drops_only_that_pcs_schedule():
    state = make_state("SET LR0 CR8 ;;\nSET LR1 CR8 ;;\nSET LR2 CR8 ;;\nHALT ;;", cr={8: 4})
    cycle = _cycle(state)
    for _ in range(3):
        cycle.step()
    kept = [cycle.cached_schedule(pc) for pc in (0, 2)]
    assert cycle.cached_schedule(1) is not None

    state.inst_mem[1] = assemble_program("SET LR3 CR8 ;;")[0]

    assert cycle.cached_schedule(1) is None
    assert all(cycle.cached_schedule(pc) is old for pc, old in zip((0, 2), kept, strict=True))
    state.program_counter = 1
    cycle.step()
    assert state.regfile.get_lr(3) == 4


def test_imem_words_are_frozen_and_writes_are_versioned():
    state = make_state("SET LR0 CR8 ;;", cr={8: 4})
    word = state.inst_mem[0]
    version = state.inst_mem.version(0)
    with pytest.raises(TypeError):
        word["anything"] = 1  # type: ignore[index]

    state.inst_mem[0] = dict(word)

    assert state.inst_mem.version(0) == version + 1
    assert state.inst_mem.version(1) == 0
    assert state.inst_mem[0] == word


def test_imem_write_observers_are_called_and_removed():
    state = new_state()
    seen = []
    remove = state.inst_mem.add_write_observer(seen.append)
    state.inst_mem[3] = None
    state.inst_mem[-1] = None
    remove()
    remove()
    state.inst_mem[4] = None
    assert seen == [3, len(state.inst_mem) - 1]


def test_imem_does_not_keep_a_cycle_alive():
    import gc
    import weakref

    state = make_state("SET LR0 CR8 ;;", cr={8: 4})
    ipu = Ipu(state)
    cycle = Cycle(state, ipu)
    cycle.step()
    ref = weakref.ref(cycle)
    del cycle
    gc.collect()
    assert ref() is None
    state.inst_mem[0] = None  # the dead cycle's observer is gone, not called


def test_a_debugger_word_written_at_a_break_is_retired():
    state = make_state("SET LR0 CR8 ; BREAK ;;\nHALT ;;", cr={8: 5, 9: 6})
    cycle = _cycle(state)
    assert cycle.evaluate_break() is CycleOutcome.BREAK

    state.inst_mem[0] = assemble_program("SET LR1 CR9 ;;")[0]
    cycle.retire()

    assert state.regfile.get_lr(0) == 0
    assert state.regfile.get_lr(1) == 6
    assert state.program_counter == 1


def test_the_observer_brackets_each_retired_word_and_not_a_break():
    state = make_state("SET LR0 CR8 ; BREAK ;;\nSET LR1 CR8 ;;\nHALT ;;", cr={8: 3})
    recorder = _Recorder(state)
    cycle = _cycle(state, recorder)

    assert cycle.step() is CycleOutcome.BREAK
    assert _brackets(recorder) == []
    cycle.retire()
    cycle.step()
    assert _brackets(recorder) == [("begin", 0), ("end", 1), ("begin", 1), ("end", 2)]


def _brackets(recorder: _Recorder) -> list[tuple]:
    return [call for call in recorder.calls if call[0] in ("begin", "end")]


def _non_nop(recorder: _Recorder) -> list[tuple]:
    return [call for call in recorder.calls if call[0] in ("begin", "end") or call[2] != "NOP"]


def test_ops_are_issued_and_retired_inside_the_brackets_and_break_before_them():
    state = make_state("SET LR0 CR8 ; BREAK ;;\nSET LR1 CR8 ;;\nHALT ;;", cr={8: 3})
    recorder = _Recorder(state)
    cycle = _cycle(state, recorder)

    assert cycle.step() is CycleOutcome.BREAK
    assert recorder.calls == [("issued", "break", "BREAK"), ("retired", "break", "BREAK")]
    recorder.calls.clear()
    cycle.retire()
    assert _non_nop(recorder) == [
        ("begin", 0),
        ("issued", "lr", "SET"),
        ("retired", "lr", "SET"),
        ("end", 1),
    ]


def test_state_observers_run_before_the_cycles_and_detach():
    state = make_state("SET LR0 CR8 ;;\nSET LR1 CR8 ;;\nHALT ;;", cr={8: 3})
    order: list[str] = []

    class _Named(_Recorder):
        def __init__(self, name: str) -> None:
            super().__init__(state)
            self.name = name

        def begin(self, pc: int) -> None:
            order.append(self.name)

    cycle = Cycle(state, Ipu(state))
    remove_own = cycle.add_observer(_Named("cycle"))
    remove_state = state.add_cycle_observer(_Named("state"))
    cycle.step()
    assert order == ["state", "cycle"]
    remove_own()
    remove_state()
    remove_state()  # harmless
    cycle.step()
    assert order == ["state", "cycle"]
    assert cycle.observers == ()


def test_issue_facts_carry_lr_writes_and_the_mult_mask():
    state = make_state(
        "SET LR0 CR8 ; ADDBI LRD2 1 ;;\nMULT.EE LR4 CR1 2 LR5 CR15 ;;\nHALT ;;",
        cr={8: 3},
        setup=lambda s: s.regfile.set_r_mask(bytes(range(128))),
    )
    seen = {}

    class _Facts(_Recorder):
        def issued(self, facts) -> None:
            if facts.plan.inst_name != "NOP":
                seen[facts.plan.inst_name, facts.lr_writes] = facts.mask

    cycle = _cycle(state, _Facts(state))
    cycle.step()
    state.regfile.set_lr(5, 1)  # mask_shift +1
    cycle.step()
    assert ("SET", (0,)) in seen and seen["SET", (0,)] is None
    assert ("ADDBI", (2, 3)) in seen
    mask = seen["MULT.EE", ()]
    slot = int.from_bytes(bytes(range(32, 48)), "little")
    partition = state.get_dstructure_for(15).partition
    assert mask == (slot << 1) & ((1 << 128) - 1) & partition_vector(partition)
    assert mask == effective_mult_mask(state.regfile, state.get_dstructure_for(15), 2, 1)


def test_an_lr_conflict_raises_before_any_lr_write():
    state = make_state("SET LR0 CR8 ; SET LR0 CR9 ; INC LR3 1 ;;\nHALT ;;", cr={8: 5, 9: 6})
    with pytest.raises(EmulatorFault, match=r"F-3 at PC 0: LR conflict.*targets: \[0, 0, 3\]"):
        _cycle(state).step()
    assert state.halt == HaltRecord("fault", 0, FaultId.F3)
    assert state.regfile.get_lr(0) == 0
    assert state.regfile.get_lr(3) == 0


def test_schedule_of_a_word():
    word = assemble_program("SET LR0 CR1 ; INC LR2 1 ; ACC.ADD ; BREAK ;;")[0]

    schedule = schedule_word(word)

    assert schedule.break_op is not None
    assert schedule.break_op.plan.inst_name == "BREAK"
    assert schedule.lr_conflict is None
    assert [op.plan.inst_name for op in schedule.lr_ops if op.plan.inst_name != "NOP"] == [
        "SET",
        "INC",
    ]
    assert [op.slot for op in schedule.slot_ops] == [
        op.slot for op in decode_word(word, pc=0) if op.slot not in ("break", "lr")
    ]
    assert "acc" in {op.slot for op in schedule.slot_ops if op.plan.inst_name != "NOP"}


def test_a_nop_break_slot_is_not_scheduled_and_conflicts_are_found_once():
    word = assemble_program("SET LR0 CR1 ; SET LR0 CR2 ;;")[0]

    schedule = schedule_word(word)

    assert schedule.break_op is None
    assert schedule.lr_conflict == (0, 0)


def test_break_result_is_an_alias_of_cycle_outcome():
    assert BreakResult is CycleOutcome
    assert BreakResult.CONTINUE is CycleOutcome.RETIRED


def test_skipping_idle_nops_keeps_counters_and_the_cond_nop_in_place():
    word = assemble_program("SET LR0 CR1 ; ACC.ADD ;;")[0]

    faithful = schedule_word(word)
    skipping = schedule_word(word, skip_idle_nops=True)

    assert skipping.break_op is faithful.break_op is None
    assert [op.plan.inst_name for op in skipping.lr_ops] == ["SET"]
    kept = [(op.slot, op.plan.inst_name, op.operands is None) for op in skipping.slot_ops]
    # The idle multiplier stays as a counter-only op, the cond NOP (it advances
    # the PC) is issued, ACC.ADD is issued; every other NOP is gone.
    assert ("mult", "NOP", True) in kept
    assert ("cond", "NOP", False) in kept
    assert ("acc", "ACC.ADD", False) in kept
    assert all(name != "NOP" or slot in ("mult", "cond") for slot, name, _ in kept)
    # Whatever is kept keeps the faithful order, LR included.
    order = [op.slot for op in faithful.slot_ops]
    assert [op.slot for op in skipping.slot_ops] == [
        slot for slot in order if slot in {op.slot for op in skipping.slot_ops}
    ]
    lr_before = {op.slot for op in faithful.slot_ops[: faithful.lr_position]}
    assert {op.slot for op in skipping.slot_ops[: skipping.lr_position]} <= lr_before


def _lockstep_modes(program: list, **kw) -> list[tuple[IpuState, Cycle]]:
    """A faithful and a NOP-skipping cycle on twin states. The second is a real
    prototyping state (its resolver too) in a ``--define=ipu_proto=1`` build."""
    from ipu_emu.prototyping import build_enabled

    runs = []
    for skip in (False, True):
        state = new_state(prototyping=skip and build_enabled(), **kw)
        load_program(state, program)
        runs.append((state, Cycle(state, Ipu(state), skip_idle_nops=skip)))
    return runs


def test_both_modes_agree_cycle_by_cycle():
    runs = []
    for skip in (False, True):
        state = _program_state()
        runs.append((state, Cycle(state, Ipu(state), skip_idle_nops=skip)))
    while not runs[0][0].is_halted:
        outcomes = [cycle.step() for _, cycle in runs]
        assert outcomes[0] is outcomes[1]
        if outcomes[0] is CycleOutcome.BREAK:
            for _, cycle in runs:
                cycle.retire()
        assert _observable(runs[0][0]) == _observable(runs[1][0])
    assert runs[1][0].is_halted


def test_lr_subslot_exception_atomicity():
    """A word whose LR sub-slot 2 names a CR past the operand's width faults
    F-1 (reserved encoding) before any slot runs: sub-slot 1's target stays
    unwritten, in both modes, compared in lockstep."""
    program = assemble_program("SET LR1 CR2 ; SET LR2 CR3 ; SET LR4 CR5 ;;\nHALT ;;")
    word = program[0]
    # SET's source shares the ADDBI immediate's union column: point sub-slot
    # 2's source at a CR that does not exist (a value the CrIdx width cannot hold).
    source = next(
        name for name in word if name.startswith("lr_inst_1_") and name.endswith("_addbi_immediate")
    )
    assert word[source] == 3
    word[source] = 999
    runs = _lockstep_modes(program, cr={2: 5, 3: 6, 5: 7})

    failures = []
    for state, cycle in runs:
        with pytest.raises(EmulatorFault, match="SET src = 999 is a reserved encoding") as caught:
            cycle.step()
        assert (caught.value.fault, caught.value.pc) == (FaultId.F1, 0)
        failures.append((type(caught.value), caught.value.args))
        assert state.regfile.get_lr(1) == 0  # sub-slot 1's target
        assert state.regfile.get_lr(4) == 0

    assert failures[0] == failures[1]
    assert _observable(runs[0][0]) == _observable(runs[1][0])


def test_lr_subslot_read_error_leaves_no_lr_written(monkeypatch):
    """Every LR sub-slot's reads are resolved before any of them writes
    (``Issuer.dispatch_all``, E04-08): when sub-slot 2's operand resolution
    raises, sub-slot 1 has not written its LR and sub-slot 3 never runs. Both
    resolver adapters, so the faithful and the prototyping path (a real one in
    a ``--define=ipu_proto=1`` build), compared in lockstep."""
    from ipu_emu.resolver import FaithfulResolver, PackedResolver

    def failing(original):
        def resolve(self, plan, encoded, snapshot):
            # Sub-slot 2 is the only SET whose source is CR3.
            if plan.inst_name == "SET" and encoded.get("src") == 3:
                raise EmulatorError("injected read failure")
            return original(self, plan, encoded, snapshot)

        return resolve

    for adapter in (FaithfulResolver, PackedResolver):
        monkeypatch.setattr(adapter, "resolve", failing(adapter.resolve))
    program = assemble_program("SET LR1 CR2 ; SET LR2 CR3 ; SET LR4 CR5 ;;\nHALT ;;")
    runs = _lockstep_modes(program, cr={2: 5, 3: 6, 5: 7})

    for state, cycle in runs:
        with pytest.raises(EmulatorError, match="injected read failure"):
            cycle.step()
        assert [state.regfile.get_lr(i) for i in (1, 2, 4)] == [0, 0, 0]

    assert _observable(runs[0][0]) == _observable(runs[1][0])
