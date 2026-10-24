"""Minimal programs that show which value of a register a consumer reads (E02-05, E02-06).

Each probe writes a register and reads it in the same instruction word, so the
value the program ends with tells which version the consumer saw:

* **cycle start**: the value the register held when the word began, or
* **forwarded**: the value written earlier in the same cycle.

The probes are data, so the timing tests (``test_timing_lr_views.py``,
``test_timing_datapath_views.py``) assert on their results, and the
prototyping lockstep test runs the same programs in both engines (E02-08).

Every probe runs in faithful mode (``prototyping=False``) unless a caller asks
otherwise, so the timing tests pin the faithful engine in every build.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from ipu_as.lark_tree import assemble
from ipu_emu.emulator import load_program
from ipu_emu.errors import EmulatorFault
from ipu_emu.ipu import Ipu
from ipu_emu.ipu_state import IpuState
from ipu_emu.outcome import CycleOutcome
from ipu_isa.codec.word import DecodedWord
from ipu_isa.cr_halves import qf
from ipu_isa.formats import ElementFormat
from ipu_isa.nx8 import Nx8Format, decode_element, encode_row, split_metadata

#: XMEM row the probes load from and store to.
ROW = 16
LANES = 128
MAX_CYCLES = 1_000

#: The NX8 format the probes' vectors are in (QF half in CR14.L): signed, fe 4,
#: 8 bits. The small integers the probes use are exact in it.
E4 = Nx8Format(sign=1, exp_format=4, num_bits=8)


class Mode(Enum):
    """The run modes a probe runs in."""

    NX8 = "nx8"  # NX8 elements (QF in CR14.L), FP32 MULT_RES / R_ACC lanes
    WIDE_FP32 = "wide_fp32"  # FP32 elements and lanes (emulator-only)


def new_state(mode: Mode, *, prototyping: bool = False) -> IpuState:
    fmt = ElementFormat.FP32 if mode is Mode.WIDE_FP32 else ElementFormat.NX8
    state = IpuState(fmt=fmt, prototyping=prototyping)
    state.set_cr_half(14, False, qf(E4))  # CR15.L holds the default DS (128 elements)
    return state


# -- reading and writing registers and XMEM, in elements -----------------------


def _wide(state: IpuState) -> bool:
    return state.fmt is ElementFormat.FP32


def _rows(state: IpuState, values, rows: int) -> tuple[bytes, list[int]]:
    """*values* (padded with 0 to *rows* rows) as element bytes, and each
    row's metadata word (NX8: quantized row by row; FP32: float32, no metadata)."""
    values = list(values) + [0] * (rows * LANES - len(values))
    if _wide(state):
        return struct.pack(f"<{len(values)}f", *values), [0] * rows
    data, metadata = b"", []
    for k in range(rows):
        row = encode_row(values[k * LANES : (k + 1) * LANES], E4)
        data += row.elements
        metadata.append(row.metadata)
    return data, metadata


def write_row(state: IpuState, row: int, values) -> None:
    """Write one XMEM row of input elements (and, in NX8, its metadata)."""
    data, (metadata,) = _rows(state, values, 1)
    state.xmem.write_address(row * len(data), data)
    if not _wide(state):
        state.xmem.write_metadata(row * len(data), metadata)


def write_raw_row(state: IpuState, row: int, data: bytes) -> None:
    """Write raw bytes at the start of an XMEM row (a mask row)."""
    state.xmem.write_address(row * LANES * (4 if _wide(state) else 1), data)


def _fill(state: IpuState, name: str, meta: str, rows: int, values) -> None:
    data, metadata = _rows(state, values, rows)
    if _wide(state):
        name = f"{name}_wide_debug"
    state.regfile.mutable(name)[: len(data)] = data
    for index, word in enumerate(metadata):
        state.regfile.set_register_bytes(meta, index, word.to_bytes(2, "little"))


def fill_ra(state: IpuState, values) -> None:
    """R0 ++ R1 (256 elements), with their metadata."""
    _fill(state, "r", "r_meta", 2, values)


def fill_r_cyclic(state: IpuState, values) -> None:
    """R_CYCLIC (512 elements), with each slot's metadata."""
    _fill(state, "r_cyclic", "r_cyclic_meta", 4, values)


def fill_lanes(state: IpuState, register: str, values) -> None:
    """MULT_RES or R_ACC lanes (FP32 in both modes)."""
    state.regfile.mutable(register)[: 4 * len(values)] = struct.pack(f"<{len(values)}f", *values)


def lane(state: IpuState, register: str, index: int):
    """One MULT_RES or R_ACC lane."""
    return struct.unpack_from("<f", state.regfile.view(register), 4 * index)[0]


def _element(state: IpuState, register: str, meta: str, index: int):
    if _wide(state):
        return struct.unpack_from("<f", state.regfile.view(f"{register}_wide_debug"), 4 * index)[0]
    raw = state.regfile.view(meta)
    word = int.from_bytes(raw[2 * (index // LANES) : 2 * (index // LANES) + 2], "little")
    scale, dynamic = split_metadata(word)
    return decode_element(state.regfile.view(register)[index], E4, scale, dynamic)


def ra_element(state: IpuState, index: int):
    return _element(state, "r", "r_meta", index)


def r_cyclic_element(state: IpuState, index: int):
    return _element(state, "r_cyclic", "r_cyclic_meta", index)


def stored_element(state: IpuState, row: int, index: int):
    """Element *index* of the row an AaQ store wrote at *row*, decoded."""
    if _wide(state):
        return struct.unpack_from(
            "<f", state.xmem.snapshot_bytes(row * 4 * LANES, 4 * LANES), 4 * index
        )[0]
    scale, dynamic = split_metadata(state.xmem.snapshot_metadata(row * LANES))
    return decode_element(state.xmem.snapshot_bytes(row * LANES, LANES)[index], E4, scale, dynamic)


# -- probes ---------------------------------------------------------------------


def _no_setup(state: IpuState) -> None:
    pass


@dataclass(frozen=True)
class Probe:
    """A program, the CRs it needs, and any register or XMEM contents it starts
    from. ``edit`` changes the decoded program before it is loaded (for a word
    the assembler would not produce). ``modes`` lists the element formats the
    probe is meaningful in."""

    asm: str
    cr: dict[int, int] = field(default_factory=dict)
    setup: Callable[[IpuState], None] = _no_setup
    edit: Callable[[list], None] | None = None
    modes: tuple[Mode, ...] = (Mode.NX8,)


@dataclass
class Run:
    state: IpuState
    #: PCs of the words whose break slot stopped the run (then resumed).
    breaks: list[int]
    cycles: int


def decoded_program(probe: Probe) -> list:
    program: list = [DecodedWord.from_word(word) for word in assemble(probe.asm)]
    if probe.edit is not None:
        probe.edit(program)
    return program


def prepare(probe: Probe, mode: Mode = Mode.NX8, *, prototyping: bool = False) -> IpuState:
    state = new_state(mode, prototyping=prototyping)
    for index, value in probe.cr.items():
        state.regfile.set_cr(index, value)
    probe.setup(state)
    load_program(state, decoded_program(probe))
    return state


def run(probe: Probe, mode: Mode = Mode.NX8, *, prototyping: bool = False) -> Run:
    """Run to the end, resuming past every break, as ``run_until_complete`` does.

    It is not ``run_until_complete`` because the break tests need the PC of
    each break; run statistics are not reset or totalled here.
    """
    state = prepare(probe, mode, prototyping=prototyping)
    ipu = Ipu(state)
    breaks: list[int] = []
    cycles = 0
    while not state.is_halted:
        if cycles >= MAX_CYCLES:
            raise RuntimeError(f"probe ran past {MAX_CYCLES} cycles")
        if ipu.cycle.step() == CycleOutcome.BREAK:
            breaks.append(state.program_counter)
            ipu.cycle.retire()
        cycles += 1
    return Run(state, breaks, cycles)


def _setup(*steps: Callable[[IpuState], None]) -> Callable[[IpuState], None]:
    def setup(state: IpuState) -> None:
        for step in steps:
            step(state)

    return setup


def _ra(values) -> Callable[[IpuState], None]:
    return lambda state: fill_ra(state, values)


def _r_cyclic(values) -> Callable[[IpuState], None]:
    return lambda state: fill_r_cyclic(state, values)


def _lanes(register: str, values) -> Callable[[IpuState], None]:
    return lambda state: fill_lanes(state, register, values)


def _row(row: int, values) -> Callable[[IpuState], None]:
    return lambda state: write_row(state, row, values)


def _raw_row(row: int, data: bytes) -> Callable[[IpuState], None]:
    return lambda state: write_raw_row(state, row, data)


def _break_lr_subslot_1_source(program: list) -> None:
    """Point LR sub-slot 1's SET source at a CR index that does not exist."""
    word = program[0]
    field_name = next(
        name for name in word if name.startswith("lr_inst_1_") and name.endswith("_addbi_immediate")
    )
    # SET's src (LcrHalfIdx: CR3 is code 16 + 3) shares the ADDBI immediate's
    # union column; make sure the field found is that operand before breaking it.
    assert word[field_name] == 19, f"{field_name} is not sub-slot 1's SET source"
    word[field_name] = 999


def _idle_second_word(program: list) -> None:
    program[1] = None


_ONES = [1] * LANES
_TWOS = [2] * LANES
_THREES = [3] * LANES
_COUNTING = list(range(1, LANES + 1))

# The cond probes branch to `taken` or fall through; LR7 records which (2 or 1).
_BRANCH = """
    SET LR0 CR2 ; {branch} taken ;;
    SET LR7 CR3 ;;
    HALT ;;
taken:
    SET LR7 CR4 ;;
    HALT ;;
"""
_BRANCH_CR = {2: 5, 3: 1, 4: 2}

# The signed-branch probes: LR0 = -1 (0xFFFFFFFF) compared with CR0 = 0.
_SIGNED_BRANCH = """
    SET LR0 CR2 ;;
    {branch} LR0 CR0 taken ;;
    SET LR7 CR3 ;;
    HALT ;;
taken:
    SET LR7 CR4 ;;
    HALT ;;
"""
_SIGNED_BRANCH_CR = {2: 0xFFFFFFFF, 3: 1, 4: 2}

_ACC_OPS = {
    "add": "ACC.ADD",
    "add_first": "ACC.ADD.FIRST",
    "sub": "ACC.SUB",
    "sub_first": "ACC.SUB.FIRST",
    "max": "ACC.MAX",
    "max_first": "ACC.MAX.FIRST",
    "stride": "ACC.STRIDE 16 off off LR0",
}
_AGG_OPS = {
    "sum": "AGG.SUM",
    "sum_first": "AGG.SUM.FIRST",
    "max": "AGG.MAX",
    "max_first": "AGG.MAX.FIRST",
}

PROBES: dict[str, Probe] = {
    # ---- E02-05: LR views ----------------------------------------------------
    # LR ALU sources: LR1 is 5 at cycle start and 100 once sub-slot 0 writes it.
    "lr_alu_add": Probe(
        "SET LR1 CR2 ;;\nSET LR1 CR3 ; ADD LR2 LR1 CR1 ;;\nHALT ;;", cr={2: 5, 3: 100}
    ),
    "lr_alu_sub": Probe(
        "SET LR1 CR2 ;;\nSET LR1 CR3 ; SUB LR2 LR1 CR1 ;;\nHALT ;;", cr={2: 5, 3: 100}
    ),
    "lr_alu_incr_mod_pow2": Probe(
        "SET LR1 CR2 ;;\nSET LR1 CR3 ; INC.MODP2 LR2 LR1 4 ;;\nHALT ;;", cr={2: 5, 3: 100}
    ),
    "lr_alu_addb": Probe(
        "SET LR1 CR2 ;;\nSET LR1 CR3 ; ADDB LRD2 LR1 ;;\nHALT ;;", cr={2: 5, 3: 100}
    ),
    "lr_subslots": Probe("INC LR0 1 ; ADD LR1 LR0 CR1 ;;\nHALT ;;"),
    "lr_inc_own_dest": Probe("SET LR0 CR2 ;;\nINC LR0 3 ; ADD LR1 LR0 CR0 ;;\nHALT ;;", cr={2: 10}),
    "lr_dec_own_dest": Probe("SET LR0 CR2 ;;\nDEC LR0 3 ; ADD LR1 LR0 CR0 ;;\nHALT ;;", cr={2: 10}),
    "cond_beq": Probe(_BRANCH.format(branch="BEQ LR0 CR0"), cr=_BRANCH_CR),
    "cond_bne": Probe(_BRANCH.format(branch="BNE LR0 CR0"), cr=_BRANCH_CR),
    "cond_blt": Probe(_BRANCH.format(branch="BLT.S LR0 CR1"), cr=_BRANCH_CR),
    "cond_bge": Probe(_BRANCH.format(branch="BGE.S LR0 CR1"), cr=_BRANCH_CR),
    # BR LR0: LR0 holds 4 at cycle start and 6 once the same word writes it.
    "cond_br": Probe(
        """
        SET LR0 CR5 ;;
        SET LR0 CR6 ; BR LR0 ;;
        HALT ;;
        HALT ;;
        SET LR7 CR3 ;;
        HALT ;;
        SET LR7 CR4 ;;
        HALT ;;
        """,
        cr={3: 1, 4: 2, 5: 4, 6: 6},
    ),
    # Word 0 writes LR0 = 5 and tests it; word 1 tests it again.
    "break_ifeq": Probe(
        "SET LR0 CR2 ; BREAK.IFEQ LR0 5 ;;\nBREAK.IFEQ LR0 5 ;;\nHALT ;;", cr={2: 5}
    ),
    # Loads: LR0 (offset) is 0 at cycle start and 1 once written; row ROW
    # holds 1s, row ROW + 1 holds 2s.
    "load_mult_offset": Probe(
        "SET LR0 CR2 ; LD.MULT R0 LR0 CR3 ;;\nHALT ;;",
        cr={2: 1, 3: ROW},
        setup=_setup(_row(ROW, _ONES), _row(ROW + 1, _TWOS)),
    ),
    "load_cyclic_offset": Probe(
        "SET LR0 CR2 ; LD.MULT.CYCLIC LR0 CR3 LR1 ;;\nHALT ;;",
        cr={2: 1, 3: ROW},
        setup=_setup(_row(ROW, _ONES), _row(ROW + 1, _TWOS)),
    ),
    # LR1 (index) is 0 at cycle start and 128 once written.
    "load_cyclic_index": Probe(
        "SET LR1 CR4 ; LD.MULT.CYCLIC LR0 CR3 LR1 ;;\nHALT ;;",
        cr={3: ROW, 4: 128},
        setup=_row(ROW, _ONES),
    ),
    "load_mask_offset": Probe(
        "SET LR0 CR2 ; LD.MASK LR0 CR3 ;;\nHALT ;;",
        cr={2: 1, 3: ROW},
        setup=_setup(_raw_row(ROW, b"\x11" * LANES), _raw_row(ROW + 1, b"\x22" * LANES)),
    ),
    # AaQ stores: LR0 (table_offset) is 0 at cycle start and 8 once written;
    # R_ACC is 7s.
    "aaq_store_offset": Probe(
        "SET LR0 CR2 ; ACTIVATE.QUANTIZE identity CR15.L CR14.L CR3 LR0 ;;\nHALT ;;",
        cr={2: 8, 3: ROW},
        setup=_lanes("r_acc", [7] * LANES),
    ),
    "exact_store_offset": Probe(
        "SET LR0 CR2 ; ACTIVATE.EXACT identity CR15.L CR3 LR0 ;;\nHALT ;;",
        cr={2: 8, 3: ROW},
        setup=_lanes("r_acc", [7] * LANES),
        modes=(Mode.WIDE_FP32,),
    ),
    # LD.LUT's table_offset: LR0 is 0, then 1; row ROW is 1s, row ROW + 1 is 2s.
    "ld_lut_offset": Probe(
        "SET LR0 CR2 ; LD.LUT 0 CR3 LR0 ;;\nHALT ;;",
        cr={2: 1, 3: ROW},
        setup=_setup(_raw_row(ROW, b"\x01" * LANES), _raw_row(ROW + 1, b"\x02" * LANES)),
    ),
    # MULT LR operands. rc_idx: LR0 is 0, then 128; R_CYCLIC is 1s then 2s.
    "mult_rc_idx": Probe(
        "SET LR0 CR2 ; MULT.RC.VS LR0 0 LR1 CR15.L CR14.L ;;\nHALT ;;",
        cr={2: 128},
        setup=_r_cyclic(_ONES + _TWOS),
    ),
    # ra_idx: LR0 is 0, then 128; R0 is 1s, R1 is 2s; CR1 = 1 is the scalar.
    "mult_ra_idx": Probe(
        "SET LR0 CR2 ; MULT.VE LR0 CR1 0 LR1 CR15.L CR14.L ;;\nHALT ;;",
        cr={2: 128},
        setup=_ra(_ONES + _TWOS),
    ),
    # mask_shift: LR1 is 0, then 1; a shift of 1 masks lane 0 (padded with 0).
    "mult_mask_shift": Probe(
        "SET LR1 CR2 ; MULT.RC.VS LR0 0 LR1 CR15.L CR14.L ;;\nHALT ;;",
        cr={2: 1},
        setup=_r_cyclic(_THREES),
    ),
    # MULT.RC.VE src as an LR: LR2 is 0, then 5; it selects Ra element 0 (3)
    # or element 5 (7) as the scalar; R_CYCLIC is 1s.
    "mult_rc_ve_src": Probe(
        "SET LR2 CR2 ; MULT.RC.VE LR0 LR2 0 LR1 CR15.L CR14.L ;;\nHALT ;;",
        cr={2: 5},
        setup=_setup(_ra([3, 0, 0, 0, 0, 7]), _r_cyclic(_ONES)),
    ),
    # ACC.STRIDE offset: LR0 is 0, then 1 (start lane 0 or 32); MULT_RES[i] = i + 1.
    # The horizontal stride halves the row to 64 lanes, so lane 32 + 64 fits.
    "acc_stride_offset": Probe(
        "SET LR0 CR2 ; ACC.STRIDE 16 on off LR0 ;;\nHALT ;;",
        cr={2: 1},
        setup=_lanes("mult_res", _COUNTING),
    ),
    # ACC.RESHAPE mask as an LR: LR6 is 0 (all 8 elements move), then 8 (none).
    "reshape_mask_lr": Probe(
        "SET LR6 CR2 ; ACC.RESHAPE LRD0 LRD2 LR6 ;;\nHALT ;;",
        cr={2: 8},
        setup=_lanes("mult_res", _COUNTING),
    ),
    # AGG dest_slot: LR0 is 0, then 5; MULT_RES is all 1s.
    **{
        f"agg_dest_slot_{name}": Probe(
            f"SET LR0 CR2 ; {op} LR0 CR15.L ;;\nHALT ;;",
            cr={2: 5},
            setup=_lanes("mult_res", _ONES),
        )
        for name, op in _AGG_OPS.items()
    },
    # ACC.RESHAPE LRD pairs (LRDn is the pair LR(n+1):LR(n)), element 7 only (mask 7): its
    # source byte is LR1's top byte and its dest byte LR3's; each is 0, then 5.
    # MULT_RES[i] = i + 1.
    "reshape_source": Probe(
        "SET LR1 CR2 ; ACC.RESHAPE LRD0 LRD2 7 ;;\nHALT ;;",
        cr={2: 0x05000000},
        setup=_lanes("mult_res", _COUNTING),
    ),
    "reshape_dest": Probe(
        "SET LR3 CR2 ; ACC.RESHAPE LRD0 LRD2 7 ;;\nHALT ;;",
        cr={2: 0x05000000},
        setup=_lanes("mult_res", _COUNTING),
    ),
    "lr_conflict_same_lr": Probe("SET LR0 CR1 ; SET LR0 CR2 ;;\nHALT ;;"),
    "lr_conflict_lrd_overlap": Probe("ADDB LRD0 CR1 ; SET LR1 CR2 ;;\nHALT ;;"),
    # Sub-slot 0 writes LR1; sub-slot 1's source read raises.
    "lr_subslot_exception": Probe(
        "SET LR1 CR2 ; SET LR2 CR3 ;;\nHALT ;;",
        cr={2: 5, 3: 6},
        edit=_break_lr_subslot_1_source,
    ),
    # ---- E02-06: datapath views ----------------------------------------------
    # R0 is 1s at cycle start; the same word loads 2s into it. MULT.VE reads Ra
    # inside its handler; MULT.RC.VV reads it through the ``ra`` operand.
    "mult_ra_operand": Probe(
        "LD.MULT R0 LR0 CR2 ; MULT.RC.VV LR0 R0 0 LR0 CR15.L CR14.L ;;\nHALT ;;",
        cr={2: ROW},
        setup=_setup(_ra(_ONES), _row(ROW, _TWOS), _r_cyclic(_ONES)),
        modes=(Mode.NX8, Mode.WIDE_FP32),
    ),
    "mult_r0_r1": Probe(
        "LD.MULT R0 LR0 CR2 ; MULT.VE LR0 CR1 0 LR0 CR15.L CR14.L ;;\nHALT ;;",
        cr={2: ROW},
        setup=_setup(_ra(_ONES), _row(ROW, _TWOS)),
        modes=(Mode.NX8, Mode.WIDE_FP32),
    ),
    # R_CYCLIC is 1s at cycle start; the same word loads 3s into it.
    "mult_r_cyclic": Probe(
        "LD.MULT.CYCLIC LR0 CR2 LR0 ; MULT.RC.VS LR0 0 LR0 CR15.L CR14.L ;;\nHALT ;;",
        cr={2: ROW},
        setup=_setup(_r_cyclic(_ONES), _row(ROW, _THREES)),
        modes=(Mode.NX8, Mode.WIDE_FP32),
    ),
    # R_MASK is all ones at reset; the same word loads an all-zero mask.
    "mult_r_mask": Probe(
        "LD.MASK LR0 CR2 ; MULT.RC.VS LR0 0 LR0 CR15.L CR14.L ;;\nHALT ;;",
        cr={2: ROW},
        setup=_r_cyclic(_THREES),
        modes=(Mode.NX8, Mode.WIDE_FP32),
    ),
    # The same load one word before the multiply, which then sees it.
    "mult_r_mask_next_word": Probe(
        "LD.MASK LR0 CR2 ;;\nMULT.RC.VS LR0 0 LR0 CR15.L CR14.L ;;\nHALT ;;",
        cr={2: ROW},
        setup=_r_cyclic(_THREES),
        modes=(Mode.NX8, Mode.WIDE_FP32),
    ),
    # MULT_RES is 1s at cycle start; the same word's multiply makes it 9s
    # (R_CYCLIC 3s squared).
    **{
        f"acc_mult_res_{name}": Probe(
            f"MULT.RC.VS LR0 0 LR0 CR15.L CR14.L ; {op} ;;\nHALT ;;",
            setup=_setup(_r_cyclic(_THREES), _lanes("mult_res", _ONES)),
            modes=(Mode.NX8, Mode.WIDE_FP32),
        )
        for name, op in _ACC_OPS.items()
    },
    **{
        f"agg_mult_res_{name}": Probe(
            f"MULT.RC.VS LR0 0 LR0 CR15.L CR14.L ; {op} LR0 CR15.L ;;\nHALT ;;",
            setup=_setup(_r_cyclic(_THREES), _lanes("mult_res", _ONES)),
        )
        for name, op in _AGG_OPS.items()
    },
    "reshape_mult_res": Probe(
        "MULT.RC.VS LR0 0 LR0 CR15.L CR14.L ; ACC.RESHAPE LRD0 LRD2 7 ;;\nHALT ;;",
        setup=_setup(_r_cyclic(_THREES), _lanes("mult_res", _ONES)),
    ),
    # R_ACC is 5s at cycle start; the multiply makes MULT_RES 9s.
    "acc_old_r_acc": Probe(
        "MULT.RC.VS LR0 0 LR0 CR15.L CR14.L ; ACC.ADD ;;\nHALT ;;",
        setup=_setup(_r_cyclic(_THREES), _lanes("r_acc", [5] * LANES)),
    ),
    # R_ACC is 2s at cycle start; MULT_RES is 7s, so ACC makes R_ACC 7s or 9s.
    "activate_r_acc_add_first": Probe(
        "ACC.ADD.FIRST ; ACTIVATE.QUANTIZE identity CR15.L CR14.L CR2 LR0 ;;\nHALT ;;",
        cr={2: ROW},
        setup=_setup(_lanes("mult_res", [7] * LANES), _lanes("r_acc", [2] * LANES)),
    ),
    "activate_r_acc_add": Probe(
        "ACC.ADD ; ACTIVATE.QUANTIZE identity CR15.L CR14.L CR2 LR0 ;;\nHALT ;;",
        cr={2: ROW},
        setup=_setup(_lanes("mult_res", [7] * LANES), _lanes("r_acc", [2] * LANES)),
    ),
    # R_ACC is 1s at cycle start; ACC.ADD.FIRST makes it 7s (MULT_RES 7s);
    # ACTIVATE.EXACT stores the forwarded 7s (FP32 only).
    "exact_r_acc": Probe(
        "ACC.ADD.FIRST ; ACTIVATE.EXACT identity CR15.L CR2 LR0 ;;\nHALT ;;",
        cr={2: ROW},
        setup=_setup(_lanes("mult_res", [7] * LANES), _lanes("r_acc", [1] * LANES)),
        modes=(Mode.WIDE_FP32,),
    ),
    # One word: an LR write, an AaQ store of R_ACC (7s) and HALT; word 1
    # would overwrite LR0 if it ran.
    "halt": Probe(
        "SET LR0 CR2 ; ACTIVATE.QUANTIZE identity CR15.L CR14.L CR3 LR1 ; HALT ;;\n"
        "SET LR0 CR4 ;;\nHALT ;;",
        cr={2: 5, 3: ROW, 4: 9},
        setup=_lanes("r_acc", [7] * LANES),
    ),
    "signed_blt": Probe(_SIGNED_BRANCH.format(branch="BLT.S"), cr=_SIGNED_BRANCH_CR),
    "signed_bge": Probe(_SIGNED_BRANCH.format(branch="BGE.S"), cr=_SIGNED_BRANCH_CR),
    # A BREAK word that also writes LR0 = CR2.
    "break_resume": Probe("SET LR0 CR2 ; BREAK ;;\nHALT ;;", cr={2: 5}),
    # Word 1 is an empty instruction-memory entry.
    "idle_word": Probe(
        "SET LR0 CR2 ;;\nSET LR1 CR2 ;;\nHALT ;;", cr={2: 5}, edit=_idle_second_word
    ),
}

#: Probes whose program raises in faithful mode, with the exception type.
RAISES: dict[str, type[Exception]] = {
    "lr_conflict_same_lr": EmulatorFault,  # F-3
    "lr_conflict_lrd_overlap": EmulatorFault,
    "lr_subslot_exception": EmulatorFault,  # F-1: the source is past CrIdx's width
}
