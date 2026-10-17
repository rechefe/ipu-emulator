"""Every declared implicit read is honoured by the emulator (E04-12).

The cases are generated from the spec: one per (instruction, implicit read) in
``instruction_spec.py``. Each one runs a two-bundle probe program where a
producer in another slot writes the register in the same word as the reader:

* **probe**: ``producer ; reader ;;`` then ``HALT``;
* **cycle-start reference**: ``reader ;;`` alone, from the same start state;
* **forwarded reference**: ``producer ;;`` then ``reader ;;``.

The reader's outputs (the registers it writes, or the XMEM row it stores) must
differ between the two references, so the probe can tell the views apart, and
the probe must match the reference its declared view names. A read no other
slot can produce in the same word (a CR, a register only the reader's own slot
writes, an LR read by an LR sub-slot) cannot be told apart; its case checks
that this is so and that the plan carries the declared view.

This is the long-term guard on the timing contract. The hand-written E02
characterization tests (``test_timing_*_views.py``) pin the same views with
hand-picked probes; E04-25 flipped their two Q32 rows with the spec.
"""

from __future__ import annotations

import struct
import warnings
from collections.abc import Callable

import numpy as np
import pytest
from timing_probes import ROW, Mode, Probe, decoded_program, run

from ipu_as.lark_tree import assemble
from ipu_emu import prototyping
from ipu_emu.cycle import CycleView, IssuedOp, decode_word
from ipu_emu.emulator import load_program
from ipu_emu.errors import DebugOnlyInstructionWarning, EmulatorError, UndeclaredRead
from ipu_emu.execute import Ipu, decode_instruction_word
from ipu_emu.ipu import read_views
from ipu_emu.ipu_state import IpuState
from ipu_isa.slots import Slot
from ipu_isa.timing import RegisterAccess, View
from ipu_isa.view import Instruction, isa

LANES = 128
#: XMEM row the store readers write (the producers load from ``ROW``).
STORE_ROW = ROW + 8
#: The value an LR producer writes (CR3): an index the readers can see.
LR_PRODUCED = 8
CRS = {2: ROW, 3: LR_PRODUCED, 4: STORE_ROW, 5: 3}

#: The reader instruction of each mnemonic with implicit reads. Operands that
#: select a register through an implicit LR read name LR6 (0 at the start,
#: LR_PRODUCED once the LR producer writes it); loads read row CR2, stores
#: write row CR4.
READERS: dict[str, str] = {
    "ST.AAQ": "ST.AAQ LR0 CR4",
    "ST.ACC": "ST.ACC LR0 CR4",
    "INC.MODP2": "INC.MODP2 LR1 LR0 4",
    "INC": "INC LR1 1",
    "DEC": "DEC LR1 1",
    "ADDB": "ADDB LRD2 CR5",
    "ADDBI": "ADDBI LRD2 3",
    "MULT.RC.VV": "MULT.RC.VV LR0 R0 0 LR0 CR15",
    "MULT.RC.VE": "MULT.RC.VE LR0 LR6 0 LR0 CR15",
    "MULT.RC.VS": "MULT.RC.VS LR0 0 LR0 CR15",
    "MULT.VE": "MULT.VE LR0 CR5 0 LR0 CR15",
    "MULT.EE": "MULT.EE LR0 CR5 0 LR0 CR15",
    "ACC.ADD": "ACC.ADD",
    "ACC.ADD.FIRST": "ACC.ADD.FIRST",
    "ACC.MAX": "ACC.MAX",
    "ACC.MAX.FIRST": "ACC.MAX.FIRST",
    "ACC.SUB": "ACC.SUB",
    "ACC.SUB.FIRST": "ACC.SUB.FIRST",
    "ACC.STRIDE": "ACC.STRIDE 16 off off LR0",
    "AGG.SUM.FIRST": "AGG.SUM.FIRST LR0 CR15",
    "AGG.SUM": "AGG.SUM LR0 CR15",
    "AGG.MAX.FIRST": "AGG.MAX.FIRST LR0 CR15",
    "AGG.MAX": "AGG.MAX LR0 CR15",
    "ACC.RESHAPE": "ACC.RESHAPE LRD0 LRD2 LR6",
    "ACTIVATE.QUANTIZE": "ACTIVATE.QUANTIZE identity CR15",
}

#: For each register a reader reads, an instruction of another slot that writes
#: it to a value different from the start state's: (slot, instruction).
PRODUCERS: dict[str, tuple[Slot, str]] = {
    "r": (Slot.LOAD, "LD.MULT R0 LR0 CR2"),
    "r_cyclic": (Slot.LOAD, "LD.MULT.CYCLIC LR0 CR2 LR0"),
    "r_mask": (Slot.LOAD, "LD.MASK LR0 CR2"),
    "mult_res": (Slot.MULT, "MULT.RC.VS LR1 0 LR1 CR15"),
    "r_acc": (Slot.ACC, "ACC.ADD.FIRST"),
    "post_aaq_reg": (Slot.AAQ, "ACTIVATE.QUANTIZE identity CR15"),
    "lr": (Slot.LR, "SET LR6 CR3"),
}

#: The register that holds a declared register in wide-vector debug mode.
_WIDE_STORAGE = {"r": "r_wide_debug", "r_cyclic": "r_cyclic_wide_debug"}


def _register_writers() -> dict[str, frozenset[Slot]]:
    """For each register, the slots with an instruction that writes it."""
    writers: dict[str, set[Slot]] = {}
    for slot in isa().spec_slot_order:
        for instruction in isa().instructions(slot):
            for access in instruction.writes():
                writers.setdefault(access.register, set()).add(slot)
    return {register: frozenset(slots) for register, slots in writers.items()}


_WRITERS = _register_writers()


def _cases() -> list[tuple[Instruction, RegisterAccess]]:
    return [
        (instruction, access)
        for slot in isa().spec_slot_order
        for instruction in isa().instructions(slot)
        for access in instruction.implicit_reads
    ]


def _case_id(case: tuple[Instruction, RegisterAccess]) -> str:
    instruction, access = case
    via = f"@{access.via_operand}" if access.via_operand else ""
    return f"{instruction.slot}.{instruction.mnemonic}:{access.register}{via}"


CASES = _cases()


def _co_issuable_producers(instruction: Instruction, register: str) -> frozenset[Slot]:
    """The slots that can write *register* in the same word as *instruction*.

    The reader's own slot holds the reader, so it is not one; for an LR
    sub-slot, another sub-slot writing the same LR is an LR write conflict.
    """
    return _WRITERS.get(register, frozenset()) - {instruction.slot}


# -- the start state ------------------------------------------------------------


def _start_state(mode: Mode) -> Callable[[IpuState], None]:
    """Deterministic, distinct contents for every register a reader reads and
    for the XMEM row the producers load."""

    def setup(state: IpuState) -> None:
        rng = np.random.default_rng(12)
        regfile = state.regfile
        if mode is Mode.WIDE_FP32:

            def elements(n: int) -> bytes:
                return struct.pack(f"<{n}f", *(rng.integers(-16, 17, n) / 4.0))

            regfile.mutable("r_wide_debug")[:] = elements(2 * LANES)
            regfile.mutable("r_cyclic_wide_debug")[:] = elements(4 * LANES)
            regfile.mutable("mult_res")[:] = elements(LANES)
            regfile.mutable("r_acc")[:] = elements(LANES)
            row = elements(LANES)
        else:

            def lanes(n: int) -> bytes:
                return struct.pack(f"<{n}i", *rng.integers(-60, 61, n).tolist())

            regfile.mutable("r")[:] = rng.integers(0, 256, 2 * LANES, dtype=np.uint8).tobytes()
            regfile.mutable("r_cyclic")[:] = rng.integers(
                0, 256, 4 * LANES, dtype=np.uint8
            ).tobytes()
            regfile.mutable("mult_res")[:] = lanes(LANES)
            regfile.mutable("r_acc")[:] = lanes(LANES)
            row = rng.integers(0, 256, LANES, dtype=np.uint8).tobytes()
        regfile.mutable("post_aaq_reg")[:] = rng.integers(
            0, 256, 4 * LANES, dtype=np.uint8
        ).tobytes()
        state.xmem.write_address(ROW * len(row), row)

    return setup


# -- running a probe ------------------------------------------------------------


def _outputs(state: IpuState, instruction: Instruction, mode: Mode) -> tuple[bytes, ...]:
    """What the reader wrote: its written registers, and the row a store wrote."""
    wide = mode is Mode.WIDE_FP32
    out = [
        bytes(
            state.regfile.view(
                _WIDE_STORAGE.get(access.register, access.register) if wide else access.register
            )
        )
        for access in instruction.writes()
    ]
    if instruction.slot in (Slot.STORE, Slot.ACC_STORE):
        row_bytes = LANES * (4 if wide else 1)
        out.append(bytes(state.xmem.read_address(STORE_ROW * row_bytes, 4 * LANES)))
    return tuple(out)


def _run_outputs(asm: str, instruction: Instruction, mode: Mode, prototyping_on: bool):
    probe = Probe(asm, cr=CRS, setup=_start_state(mode), modes=(mode,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DebugOnlyInstructionWarning)
        warnings.filterwarnings("ignore", message=".*simulation-only acc_store slot")
        result = run(probe, mode, prototyping=prototyping_on)
    return _outputs(result.state, instruction, mode)


# -- the tests ------------------------------------------------------------------


def test_every_reader_and_producer_is_a_real_instruction():
    """The tables above are spec data: each reader is its mnemonic, each
    producer writes its register from its slot."""
    for mnemonic, asm in READERS.items():
        assert asm.split()[0] == mnemonic
    for register, (slot, asm) in PRODUCERS.items():
        producer = isa().instruction(slot, asm.split()[0])
        assert register in {access.register for access in producer.writes()}, asm


def test_there_is_a_case_for_every_declared_implicit_read():
    ids = [_case_id(case) for case in CASES]
    assert len(ids) == len(set(ids))
    assert len(CASES) == sum(
        len(instruction.implicit_reads)
        for slot in isa().spec_slot_order
        for instruction in isa().instructions(slot)
    )
    # Every register some instruction reads implicitly is covered.
    assert {access.register for _, access in CASES} >= {"r_mask", "mult_res", "r_acc", "lr"}


@pytest.mark.parametrize("prototyping_on", [False, True], ids=["faithful", "prototyping"])
@pytest.mark.parametrize("mode", [Mode.NARROW, Mode.WIDE_FP32], ids=lambda m: m.value)
@pytest.mark.parametrize("case", CASES, ids=[_case_id(case) for case in CASES])
def test_implicit_read_honours_its_declared_view(case, mode, prototyping_on):
    instruction, access = case
    if prototyping_on and not prototyping.build_enabled():
        pytest.skip("prototyping needs --config=proto")
    # The plan the emulator dispatches carries the declared view.
    views = read_views(instruction)
    assert views[access.register, access.via_operand] is (access.view is View.CYCLE_START)

    producers = _co_issuable_producers(instruction, access.register)
    if not producers:
        # Nothing in another slot writes it during the word, so both views hold
        # the same value: there is nothing a program could observe.
        assert access.register == "cr" or _WRITERS.get(access.register, frozenset()) <= {
            instruction.slot
        }
        return

    assert access.register in PRODUCERS, (
        f"add a producer of {access.register} to PRODUCERS (written by {sorted(producers)})"
    )
    assert instruction.mnemonic in READERS, f"add a reader for {instruction.mnemonic} to READERS"
    slot, producer = PRODUCERS[access.register]
    assert slot in producers
    reader = READERS[instruction.mnemonic]

    cycle_start = _run_outputs(f"{reader} ;;\nHALT ;;", instruction, mode, prototyping_on)
    forwarded = _run_outputs(
        f"{producer} ;;\n{reader} ;;\nHALT ;;", instruction, mode, prototyping_on
    )
    assert cycle_start != forwarded, "the probe cannot tell the two views apart"

    probe = _run_outputs(f"{producer} ; {reader} ;;\nHALT ;;", instruction, mode, prototyping_on)
    expected = cycle_start if access.view is View.CYCLE_START else forwarded
    assert probe == expected, f"declared {access.view}, but the reader saw the other view"


# -- reads outside the declaration -----------------------------------------------


def _plan(slot: Slot, mnemonic: str):
    """The plan the emulator dispatches for *mnemonic* (from its READERS word)."""
    word = decode_instruction_word(assemble(f"{READERS[mnemonic]} ;;")[0])
    return next(op.plan for op in decode_word(word) if op.slot == slot.value)


def test_an_undeclared_read_raises():
    view = CycleView(IpuState())
    view.plan = _plan(Slot.ACC, "ACC.ADD")
    view.snapshot = IpuState().regfile
    view.read("mult_res")  # declared
    with pytest.raises(UndeclaredRead, match="ACC.ADD reads r_cyclic"):
        view.read("r_cyclic")
    with pytest.raises(UndeclaredRead):
        view.lr(0, "dest_slot")


def test_a_handler_reading_an_undeclared_register_raises():
    """A handler that reads a register its instruction does not declare fails
    when it runs, rather than silently getting either view."""
    state = IpuState()
    load_program(state, decoded_program(Probe("ACC.ADD ;;\nHALT ;;")))
    engine = Ipu(state)
    engine.execute_vliw_cycle()  # a cycle has taken its snapshot

    def reads_r_mask(self: Ipu) -> None:
        self._view.read("r_mask")

    plan = _plan(Slot.ACC, "ACC.ADD")._replace(fn=reads_r_mask)
    with pytest.raises(UndeclaredRead, match="ACC.ADD reads r_mask"):
        engine.issuer.dispatch(IssuedOp("acc", plan, {}), {})


def test_a_read_with_no_op_issued_is_an_error():
    view = CycleView(IpuState())
    with pytest.raises(EmulatorError, match="outside a cycle"):
        view.read("mult_res")
