"""The measuring fixture counts the same in faithful and prototyping mode (E04-11).

``count_instructions`` is a CycleObserver on the state, so it sees the ops the
one cycle module issues whichever mode drives it. Prototyping's schedules skip
idle NOP slots (``skip_idle_nops``), which the count ignores; everything else is
the same cycle. A default build drives the prototyping schedule through
``Cycle(..., skip_idle_nops=True)``; a ``--config=proto`` build also runs the
real prototyping mode end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fixture_packed_l5_measure import count_instructions

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_emu.cycle import Cycle, CycleOutcome
from ipu_emu.emulator import load_program_from_binary, run_until_complete
from ipu_emu.ipu import Ipu
from ipu_emu.ipu_state import IpuState
from ipu_emu.prototyping import build_enabled
from ipu_isa.constants import encode_dstructure
from ipu_isa.formats import ElementFormat

_PACKED_ASM = Path(__file__).parent / "asm_packed_residual_add_240x16.asm"
_N_PACKED_ROWS = 30
_ROW_BYTES = 512

#: A word with a BREAK, so a cycle stops and resumes; mostly idle slots.
_BREAKING_ASM = """
loop:
    INC LR1 1 ; BREAK ;;
    BLT.S LR1 CR8 loop ;;
    HALT ;;
"""


def _residual_add_state(binary: Path, *, prototyping: bool = False) -> IpuState:
    state = IpuState(
        fmt=ElementFormat.FP32,
        prototyping=prototyping,
    )
    load_program_from_binary(state, binary)
    rng = np.random.RandomState(7)
    rows = rng.uniform(-3.0, 3.0, size=(2 * _N_PACKED_ROWS, 128)).astype(np.float32)
    state.xmem.write_address(0, bytearray(rows.tobytes()))
    for cr, value in {3: 2 * _N_PACKED_ROWS, 4: 0, 5: -1, 6: _N_PACKED_ROWS}.items():
        state.regfile.set_cr(cr, value)
    for cr, value in {7: 1, 8: 1, 9: _N_PACKED_ROWS, 10: 1}.items():
        state.regfile.set_cr(cr, value)
    state.regfile.set_cr(15, encode_dstructure(valid_elements=128))
    return state


def _breaking_state(binary: Path, *, prototyping: bool = False) -> IpuState:
    state = IpuState(prototyping=prototyping)
    load_program_from_binary(state, binary)
    state.regfile.set_cr(8, 4)
    return state


_PROGRAMS = {
    "packed_residual_add_240x16": (lambda: _PACKED_ASM.read_text("utf-8"), _residual_add_state),
    "break_and_resume": (lambda: _BREAKING_ASM, _breaking_state),
}


def _run_schedule(state: IpuState, *, skip_idle_nops: bool) -> None:
    """Run *state* to HALT on a cycle with the given schedule, resuming at each BREAK."""
    ipu = Ipu(state)
    cycle = Cycle(state, ipu, skip_idle_nops=skip_idle_nops)
    for _ in range(100_000):
        if state.is_halted:
            break
        if cycle.step() is CycleOutcome.BREAK:
            cycle.retire()
    assert state.is_halted


@pytest.mark.parametrize("program", sorted(_PROGRAMS))
def test_counts_are_identical_in_faithful_and_prototyping_mode(tmp_path: Path, program: str):
    source, make_state = _PROGRAMS[program]
    binary = tmp_path / "program.bin"
    assemble_to_bin_file(source(), str(binary))

    counts = {}
    for name, skip in (("faithful", False), ("prototyping schedule", True)):
        state = make_state(binary)
        with count_instructions(state) as counts[name]:
            _run_schedule(state, skip_idle_nops=skip)
        assert state.cycle_observers == ()  # detached on exit
    state = make_state(binary)
    with count_instructions(state) as counts["run_until_complete"]:
        run_until_complete(state, max_cycles=100_000)
    if build_enabled():
        state = make_state(binary, prototyping=True)
        with count_instructions(state) as counts["prototyping"]:
            run_until_complete(state, max_cycles=100_000)

    reference = counts["faithful"].by_slot
    assert reference and counts["faithful"].total > 0
    assert {name: c.by_slot for name, c in counts.items()} == dict.fromkeys(counts, reference)
    if program == "break_and_resume":
        assert reference["break"] == 4
