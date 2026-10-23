"""The conformance suite's cases, as the generator builds them (E04-31).

Each case is a program, the element format it runs in and a starting state.
:func:`all_cases` lists them and :func:`files_of` gives each one's input
files; ``update_conformance.py`` writes those and the trace a run of them
gives. The cases are:

* every HARDWARE instruction (Q17), in NX8, and in FP32 too when its slot is
  a datapath slot (mode-dependent): ``insn_<mnemonic>_<format>``;
* every timing probe (E04-12) in each mode it declares: ``probe_<name>_<format>``;
* each modelled fault and a disabled F-1 and F-9: ``fault_*``;
* the arithmetic events (E04-23): ``arith_*``.

Only this module is Python; the files it leads to are data.
"""

from __future__ import annotations

import json
import random
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import conformance
import timing_probes

from ipu_as.lark_tree import assemble
from ipu_emu.geometry import EIGHT_BIT_ROWS
from ipu_emu.ipu_state import IpuState
from ipu_isa.codec.word import DecodedWord, pack_program
from ipu_isa.constants import CR_READ_ONLY_INITIAL_VALUES
from ipu_isa.cr_halves import constant, qf
from ipu_isa.faults import reserved_encodings
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.geometry import LANES
from ipu_isa.nx8 import Nx8Format
from ipu_isa.registers import REGISTER_DEFINITIONS
from ipu_isa.slots import Slot
from ipu_isa.tags import Tag
from ipu_isa.view import isa

#: The cycle limit of every case but the F-9 ones.
MAX_CYCLES = 1_000
#: XMEM rows the load cases fill (their rows are CR3 + LR1 and CR3 + LR2).
ROWS = 8
#: The NX8 metadata word of XMEM row k in the cases whose load copies it:
#: dynamic exponent 1 and scale 0x80 + k (rows 5 and 6 read 0x0185 and 0x0186).
ROW_METADATA = 0x0180


@dataclass(frozen=True)
class Case:
    """One conformance case, before it is written.

    Attributes:
        name: The folder name.
        description: One line, for ``program.asm`` and ``setup.json``.
        asm: The program source.
        fmt: The element format.
        prepare: Sets the starting state of a fresh ``IpuState(fmt)``: the
            generator writes what differs from reset into ``setup.json``.
        edit: Changes the decoded program before it is encoded (a word the
            assembler does not produce), or ``None``.
        fault_enables: Fault enable bits other than the reset value (set).
        max_cycles: The run's cycle limit.
    """

    name: str
    description: str
    asm: str
    fmt: ElementFormat
    prepare: Callable[[IpuState], None]
    edit: Callable[[list[dict[str, int] | None]], None] | None = None
    fault_enables: Mapping[str, bool] = field(default_factory=dict)
    max_cycles: int = MAX_CYCLES

    def words(self) -> list[int]:
        """The program's words, edited."""
        program = [DecodedWord.from_word(word) for word in assemble(self.asm)]
        if self.edit is not None:
            self.edit(program)
        return [DecodedWord(word).to_word() for word in program]


# -- starting states ----------------------------------------------------------

#: CR values of the instruction cases (CR0, CR1 and CR15 keep their reset
#: values; CR15.L is the default DS). CR11 is a MULT constant (3.0 in NX8:
#: E4 code 0x34 at scale 134) and CR14.L the QF half of E4.
CRS = {
    2: 1,
    3: 4,
    4: 2,
    5: 7,
    6: 128,
    7: 0xFFFFFFFE,
    8: 100,
    9: 3,
    11: constant(0x34, 134, 4),
    14: qf(Nx8Format(sign=1, exp_format=4, num_bits=8)),
}
#: LR values: small row offsets, a cyclic slot boundary (LR6) and a branch target (LR7).
LRS = {1: 1, 2: 2, 3: 3, 4: 5, 6: 128, 7: 3}


def _floats(rng: random.Random, count: int) -> bytes:
    return struct.pack(f"<{count}f", *(rng.uniform(-8.0, 8.0) for _ in range(count)))


def _elements(rng: random.Random, fmt: ElementFormat, count: int) -> bytes:
    """*count* input elements: any byte in NX8, small float32 values in FP32."""
    if fmt is ElementFormat.FP32:
        return _floats(rng, count)
    return rng.randbytes(count)


def _metadata(rng: random.Random, count: int) -> bytes:
    """*count* NX8 metadata words: a scale near 127 (2^-7 .. 2^7), no dynamic exponent."""
    return b"".join(rng.randint(120, 134).to_bytes(2, "little") for _ in range(count))


#: Where each register an instruction reads lives in FP32 mode.
_WIDE_NAMES = {"r": "r_wide_debug", "r_cyclic": "r_cyclic_wide_debug"}


def _fill(rng: random.Random, fmt: ElementFormat, register: str) -> bytes:
    """Random contents for *register* in *fmt*: lanes, elements or bytes."""
    if register in ("mult_res", "r_acc"):
        return _floats(rng, LANES)
    definition = REGISTER_DEFINITIONS[register]
    if register in ("r_meta", "r_cyclic_meta"):
        return _metadata(rng, definition["size_bytes"] * definition["count"] // 2)
    if register in ("r", "r_cyclic"):
        return _elements(rng, fmt, definition["size_bytes"] * definition["count"])
    return rng.randbytes(definition["size_bytes"])


def random_state(
    seed: str, reads: frozenset[str], xmem: bool, metadata: bool = False
) -> Callable[[IpuState], None]:
    """An instruction case's starting state: CRS and LRS, the vector
    registers in *reads* and, when *xmem*, the first ROWS XMEM rows, filled
    from a generator seeded by *seed*. The registers the instruction does not
    read keep their reset values, which keeps the suite small. When
    *metadata* (a load that copies row metadata) and in NX8, those rows'
    metadata words are ROW_METADATA + row, nonzero so a trace shows the copy;
    otherwise XMEM row metadata stays 0."""

    def prepare(state: IpuState) -> None:
        rng = random.Random(seed)
        fmt = state.fmt
        scalars(state)
        for register in sorted(reads - {"lr", "cr"}):
            name = (
                _WIDE_NAMES.get(register, register)
                if FORMATS[fmt].tag is Tag.EMULATOR
                else register
            )
            state.regfile.mutable(name)[:] = _fill(rng, fmt, register)
        if xmem:
            state.xmem.write_address(0, _elements(rng, fmt, ROWS * LANES))
        if metadata and fmt is ElementFormat.NX8:
            for row in range(ROWS):
                state.xmem.write_metadata(row * LANES, ROW_METADATA + row)

    return prepare


def scalars(state: IpuState) -> None:
    """CRS and LRS only: the fault cases' starting state."""
    for index, value in CRS.items():
        state.regfile.set_cr(index, value)
    for index, value in LRS.items():
        state.regfile.set_lr(index, value)


# -- instruction cases --------------------------------------------------------

_NARROW = (ElementFormat.NX8,)
_DATAPATH = (*_NARROW, ElementFormat.FP32)

#: Each HARDWARE instruction's program (HALT ends every one) and formats.
INSTRUCTIONS: dict[str, tuple[str, tuple[ElementFormat, ...]]] = {
    "LD.MULT": ("LD.MULT R0 LR1 CR3 ;;\nLD.MULT R1 LR2 CR3 ;;", _DATAPATH),
    "LD.MULT.CYCLIC": ("LD.MULT.CYCLIC LR1 CR3 LR0 ;;\nLD.MULT.CYCLIC LR2 CR3 LR6 ;;", _DATAPATH),
    "LD.MASK": ("LD.MASK LR1 CR3 ;;", _DATAPATH),
    "SET": ("SET LR0 CR2 ;;\nSET LR5 CR7 ;;", _NARROW),
    "ADD": ("ADD LR0 LR1 LR2 ;;\nADD LR5 LR4 CR7 ;;", _NARROW),
    "SUB": ("SUB LR0 LR1 LR4 ;;\nSUB LR5 LR4 CR5 ;;", _NARROW),
    "INC.MODP2": ("INC.MODP2 LR2 LR3 2 ;;\nINC.MODP2 LR4 CR5 3 ;;", _NARROW),
    "INC": ("INC LR0 7 ;;\nINC LR4 1 ;;", _NARROW),
    "DEC": ("DEC LR0 3 ;;\nDEC LR4 1 ;;", _NARROW),
    "ADDB": ("ADDB LRD0 LR3 ;;\nADDB LRD2 CR5 ;;", _NARROW),
    "ADDBI": ("ADDBI LRD0 200 ;;\nADDBI LRD2 7 ;;", _NARROW),
    "MULT.RC.VV": (
        "MULT.RC.VV LR0 R0 0 LR2 CR15.L CR14.L ;;\nMULT.RC.VV LR4 R1 1 LR0 CR15.L CR14.L ;;",
        _DATAPATH,
    ),
    "MULT.RC.VE": (
        "MULT.RC.VE LR0 LR3 0 LR2 CR15.L CR14.L ;;\nMULT.RC.VE LR1 CR11 1 LR0 CR15.L CR14.L ;;",
        _DATAPATH,
    ),
    "MULT.RC.VS": (
        "MULT.RC.VS LR0 0 LR2 CR15.L CR14.L ;;\nMULT.RC.VS LR4 1 LR0 CR15.L CR14.L ;;",
        _DATAPATH,
    ),
    "MULT.VE": (
        "MULT.VE LR0 CR11 0 LR2 CR15.L CR14.L ;;\nMULT.VE LR1 CR1 1 LR0 CR15.L CR14.L ;;",
        _DATAPATH,
    ),
    "MULT.EE": (
        "MULT.EE LR0 CR11 0 LR2 CR15.L CR14.L ;;\nMULT.EE LR1 CR1 1 LR0 CR15.L CR14.L ;;",
        _DATAPATH,
    ),
    "ACC.ADD": ("ACC.ADD ;;", _DATAPATH),
    "ACC.ADD.FIRST": ("ACC.ADD.FIRST ;;", _DATAPATH),
    "ACC.MAX": ("ACC.MAX ;;", _DATAPATH),
    "ACC.MAX.FIRST": ("ACC.MAX.FIRST ;;", _DATAPATH),
    "ACC.SUB": ("ACC.SUB ;;", _DATAPATH),
    "ACC.SUB.FIRST": ("ACC.SUB.FIRST ;;", _DATAPATH),
    "ACC.STRIDE": ("ACC.STRIDE 16 off off LR0 ;;", _DATAPATH),
    "AGG.SUM.FIRST": ("AGG.SUM.FIRST LR4 CR15.L ;;", _DATAPATH),
    "AGG.SUM": ("AGG.SUM LR4 CR15.L ;;", _DATAPATH),
    "AGG.MAX.FIRST": ("AGG.MAX.FIRST LR4 CR15.L ;;", _DATAPATH),
    "AGG.MAX": ("AGG.MAX LR4 CR15.L ;;", _DATAPATH),
    "ACC.RESHAPE": ("ACC.RESHAPE LRD0 LRD2 0 ;;", _DATAPATH),
    # NX8 quantizes to CR14.L's format and stores 128 bytes and the metadata;
    # FP32 stores the activated row as 512 bytes (ADR 0007).
    "ACTIVATE.QUANTIZE": ("ACTIVATE.QUANTIZE relu CR15.L CR14.L CR3 LR1 ;;", _DATAPATH),
    "LD.LUT": ("LD.LUT 0 CR3 LR1 ;;\nLD.LUT 4 CR3 LR2 ;;", _DATAPATH),
    "BEQ": (
        "BEQ LR0 LR1 skip ;;\nINC LR5 1 ;;\nskip:\nBEQ LR0 CR0 end ;;\nINC LR5 2 ;;\nend:",
        _NARROW,
    ),
    "BNE": (
        "BNE LR0 CR0 skip ;;\nINC LR5 1 ;;\nskip:\nBNE LR0 LR1 end ;;\nINC LR5 2 ;;\nend:",
        _NARROW,
    ),
    # CR7 is -2: signed, it is below every LR.
    "BLT.S": (
        "BLT.S LR1 CR7 skip ;;\nINC LR5 1 ;;\nskip:\nBLT.S CR7 LR1 end ;;\nINC LR5 2 ;;\nend:",
        _NARROW,
    ),
    "BGE.S": (
        "BGE.S CR7 LR1 skip ;;\nINC LR5 1 ;;\nskip:\nBGE.S LR1 CR7 end ;;\nINC LR5 2 ;;\nend:",
        _NARROW,
    ),
    # LR7 is 3: BR jumps past both HALTs.
    "BR": ("BR LR7 ;;\nHALT ;;\nHALT ;;\nINC LR5 1 ;;", _NARROW),
    "HALT": ("", _NARROW),
}


def _slug(text: str) -> str:
    return text.lower().replace(".", "_")


def instruction_cases() -> list[Case]:
    """Each HARDWARE instruction in each of its formats (INSTRUCTIONS)."""
    view = isa()
    every = [i for slot in view.spec_slot_order for i in view.instructions(slot)]
    reads = {i.mnemonic: frozenset(a.register for a in i.reads()) for i in every}
    # The instructions that read XMEM: the LOAD slot's and LD.LUT.
    loads = {i.mnemonic for i in every if i.slot is Slot.LOAD or i.mnemonic == "LD.LUT"}
    # The loads that copy a row's metadata word (into r_meta or r_cyclic_meta).
    copies_metadata = {
        i.mnemonic
        for i in every
        if i.mnemonic in loads and {w.register for w in i.writes()} & {"r_meta", "r_cyclic_meta"}
    }
    cases = []
    for mnemonic, (body, formats) in INSTRUCTIONS.items():
        for fmt in formats:
            name = f"insn_{_slug(mnemonic)}_{fmt.value}"
            asm = f"{body}\nHALT ;;\n" if body else "HALT ;;\n"
            state = random_state(
                name, reads[mnemonic], mnemonic in loads, mnemonic in copies_metadata
            )
            cases.append(Case(name, f"{mnemonic} in {fmt.value.upper()}", asm, fmt, state))
    return cases


# -- timing probes ------------------------------------------------------------

_MODE_FORMAT = {
    timing_probes.Mode.NX8: ElementFormat.NX8,
    timing_probes.Mode.WIDE_FP32: ElementFormat.FP32,
}
#: Probes the suite leaves out, and why.
EXCLUDED_PROBES = {
    "idle_word": "an empty IMEM entry has no program-file encoding",
    "break_ifeq": "the reference API never evaluates BREAK (Q39)",
    "break_resume": "the reference API never evaluates BREAK (Q39)",
}


def _probe_state(
    probe: timing_probes.Probe, mode: timing_probes.Mode
) -> Callable[[IpuState], None]:
    def prepare(state: IpuState) -> None:
        prepared = timing_probes.prepare(probe, mode)
        prepared.regfile.copy_into(state.regfile)
        for base, page in prepared.xmem.touched_pages():
            state.xmem.write_address(base, bytes(page))
        for address, word in prepared.xmem.touched_metadata():
            state.xmem.write_metadata(address, word)

    return prepare


def probe_cases() -> list[Case]:
    """Each timing probe in each mode it declares, but the EXCLUDED_PROBES."""
    cases = []
    for name, probe in sorted(timing_probes.PROBES.items()):
        if name in EXCLUDED_PROBES:
            continue
        for mode in probe.modes:
            fmt = _MODE_FORMAT[mode]
            cases.append(
                Case(
                    f"probe_{name}_{fmt.value}",
                    f"timing probe {name} in {fmt.value.upper()}",
                    probe.asm,
                    fmt,
                    _probe_state(probe, mode),
                    edit=probe.edit,
                )
            )
    return cases


# -- faults and arithmetic events ---------------------------------------------


def _reserved_lr_opcode(program: list) -> None:
    """Give word 0's LR sub-slot 0 the first opcode no LR instruction has (F-1)."""
    word = program[0]
    (field_name,) = (
        name for name in word if name.startswith("lr_inst_0_") and name.endswith("_opcode")
    )
    word[field_name] = reserved_encodings(isa())[Slot.LR].opcodes[0]


_LOOP = "SET LR1 CR8 ;;\nloop:\nINC LR0 1 ;;\nBNE LR0 LR1 loop ;;\nHALT ;;\n"


def _lanes_of(code: str, acc: float, res: float) -> Callable[[IpuState], None]:
    """R_ACC lanes all *acc* and MULT_RES lanes all *res*, struct type *code*."""

    def prepare(state: IpuState) -> None:
        scalars(state)
        state.regfile.mutable("r_acc")[:] = struct.pack(f"<{LANES}{code}", *[acc] * LANES)
        state.regfile.mutable("mult_res")[:] = struct.pack(f"<{LANES}{code}", *[res] * LANES)

    return prepare


def _with_cr(index: int, value: int) -> Callable[[IpuState], None]:
    """CRS and LRS, and CR *index* = *value*."""

    def prepare(state: IpuState) -> None:
        scalars(state)
        state.regfile.set_cr(index, value)

    return prepare


def fault_cases() -> list[Case]:
    """Each modelled fault, and F-1 and F-9 disabled."""
    nx8 = ElementFormat.NX8
    return [
        Case(
            "fault_f1_reserved_opcode",
            "F-1: an undefined LR opcode",
            "INC LR0 1 ;;\nHALT ;;\n",
            nx8,
            scalars,
            edit=_reserved_lr_opcode,
        ),
        Case(
            "fault_f1_disabled",
            "F-1 disabled: the slot runs as NOP and the run goes on",
            "INC LR0 1 ;;\nINC LR4 1 ;;\nHALT ;;\n",
            nx8,
            scalars,
            edit=_reserved_lr_opcode,
            fault_enables={"F-1": False},
        ),
        Case(
            "fault_f2_past_the_program",
            "F-2: the program ends without HALT",
            "INC LR0 1 ;;\n",
            nx8,
            scalars,
        ),
        Case(
            "fault_f3_lr_conflict",
            "F-3: two writes of LR0 in one word",
            "SET LR0 CR2 ; SET LR0 CR5 ;;\nHALT ;;\n",
            nx8,
            scalars,
        ),
        Case(
            "fault_f4_cyclic_index",
            "F-4: an R_CYCLIC index off a slot boundary",
            "LD.MULT.CYCLIC LR0 CR3 LR1 ;;\nHALT ;;\n",
            nx8,
            scalars,
        ),
        Case(
            "fault_f5_reshape_index",
            "F-5: an ACC.RESHAPE destination index past the last lane",
            "ACC.RESHAPE LRD0 LRD6 0 ;;\nHALT ;;\n",
            nx8,
            scalars,
        ),
        Case(
            "fault_f6_xmem_row",
            f"F-6: XMEM row {EIGHT_BIT_ROWS}, past the NX8 mode's rows",
            "LD.MULT R0 LR0 CR10 ;;\nHALT ;;\n",
            nx8,
            _with_cr(10, EIGHT_BIT_ROWS),
        ),
        Case("fault_f9_cycle_limit", "F-9: the cycle limit", _LOOP, nx8, scalars, max_cycles=20),
        Case(
            "fault_f9_disabled",
            "F-9 disabled: the run passes the limit to HALT",
            _LOOP,
            nx8,
            scalars,
            fault_enables={"F-9": False},
            max_cycles=20,
        ),
    ]


def arith_cases() -> list[Case]:
    """The arithmetic events the emulator records (E04-23)."""
    nx8, fp32 = ElementFormat.NX8, ElementFormat.FP32
    return [
        Case(
            "arith_acc_overflow_nx8",
            "ACC_OVERFLOW: float32 lanes that round to infinity",
            "ACC.ADD ;;\nHALT ;;\n",
            nx8,
            _lanes_of("f", 3.0e38, 3.0e38),
        ),
        Case(
            "arith_quant_saturation_nx8",
            "QUANT_SATURATION: an infinite R_ACC lane quantized to E4",
            "ACTIVATE.QUANTIZE identity CR15.L CR14.L CR3 LR1 ;;\nHALT ;;\n",
            nx8,
            _lanes_of("f", float("inf"), 0.0),
        ),
        Case(
            "arith_acc_overflow_fp32",
            "ACC_OVERFLOW: float32 lanes that round to infinity",
            "ACC.ADD ;;\nHALT ;;\n",
            fp32,
            _lanes_of("f", 3.0e38, 3.0e38),
        ),
        Case(
            "arith_nan_result_fp32",
            "NAN_RESULT: NaN lanes",
            "ACC.ADD ;;\nHALT ;;\n",
            fp32,
            _lanes_of("f", 1.0, float("nan")),
        ),
    ]


# -- the files of a case ------------------------------------------------------


def _registers(state: IpuState, fresh: IpuState) -> list[dict[str, Any]]:
    """Every register a model must write so that the unlisted ones are zero:
    each one not zero at the start or not zero at reset (such as CR15's
    default dstructure), in the order of ``ipu_isa.registers``. CR0 and CR1
    are the hardware's constants and never listed."""
    writes = []
    for name, definition in REGISTER_DEFINITIONS.items():
        size = definition["size_bytes"]
        have, reset = state.regfile.view(name), fresh.regfile.view(name)
        for index in range(definition["count"]):
            if name == "cr" and index in CR_READ_ONLY_INITIAL_VALUES:
                continue
            data = bytes(have[index * size : (index + 1) * size])
            zero = bytes(size)
            if data == zero and bytes(reset[index * size : (index + 1) * size]) == zero:
                continue
            if definition["vector"]:
                writes.append({"name": name, "index": index, "hex": data.hex()})
            else:
                writes.append(
                    {"name": name, "index": index, "value": int.from_bytes(data, "little")}
                )
    return writes


def _xmem(state: IpuState) -> list[dict[str, Any]]:
    """The XMEM bytes that are not zero, one region per run (zero runs of
    fewer than 64 bytes stay inside a region)."""
    regions = []
    for base, page in state.xmem.touched_pages():
        data = bytes(page)
        start = None
        zeros = 0
        for offset, byte in enumerate(data):
            if byte:
                if start is None:
                    start = offset
                zeros = 0
            elif start is not None:
                zeros += 1
                if zeros == 64:
                    end = offset - zeros + 1
                    regions.append({"address": base + start, "hex": data[start:end].hex()})
                    start, zeros = None, 0
        if start is not None:
            end = len(data) - zeros
            regions.append({"address": base + start, "hex": data[start:end].hex()})
    return regions


def _emulator_only(case: Case, data: bytes) -> bool:
    """True when the hardware cannot run the case: FP32 mode, or an
    EMULATOR slot's op other than NOP."""
    if FORMATS[case.fmt].tag is Tag.EMULATOR:
        return True
    emulator = set(isa().layout.slots_with_tag(Tag.EMULATOR))
    return any(slot in emulator and op != "NOP" for slot, op in conformance.slot_ops(data))


def _xmem_metadata(state: IpuState) -> list[dict[str, Any]]:
    """The XMEM row metadata words that are not zero, by row: the word's
    value in four hex digits, most significant first."""
    return [
        {"row": address // LANES, "word": f"{word:04x}"}
        for address, word in state.xmem.touched_metadata()
    ]


def setup_of(case: Case, data: bytes) -> dict[str, Any]:
    """The ``setup.json`` of *case*: its starting state, every register,
    XMEM byte and XMEM row metadata word not listed being zero
    (``xmem_metadata`` only when one is not)."""
    state, fresh = IpuState(fmt=case.fmt), IpuState(fmt=case.fmt)
    case.prepare(state)
    setup = {
        "description": case.description,
        "format": case.fmt.value,
        "emulator_only": _emulator_only(case, data),
        "max_cycles": case.max_cycles,
        "fault_enables": dict(case.fault_enables),
        "registers": _registers(state, fresh),
        "xmem": _xmem(state),
    }
    metadata = _xmem_metadata(state)
    if metadata:
        setup["xmem_metadata"] = metadata
    return setup


def source_of(case: Case) -> str:
    """The ``program.asm`` of *case*: its name and description, what the
    edit of ``program.bin`` changes (when there is one), then the source."""
    lines = [f"# {case.name}: {case.description}"]
    if case.edit is not None:
        what = " ".join((case.edit.__doc__ or "").split())
        lines.append(f"# program.bin edits the assembled words: {what}")
    return "\n".join(lines) + "\n" + case.asm.strip() + "\n"


def files_of(case: Case) -> dict[str, bytes]:
    """The input files of *case* (all but the expected trace), by name."""
    data = pack_program(case.words())
    setup = json.dumps(setup_of(case, data), indent=1) + "\n"
    return {
        "program.asm": source_of(case).encode(),
        "program.bin": data,
        "setup.json": setup.encode(),
    }


def all_cases() -> list[Case]:
    """Every case, by name."""
    cases = instruction_cases() + probe_cases() + fault_cases() + arith_cases()
    names = [case.name for case in cases]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"case names used twice: {sorted(duplicates)}")
    return sorted(cases, key=lambda case: case.name)
