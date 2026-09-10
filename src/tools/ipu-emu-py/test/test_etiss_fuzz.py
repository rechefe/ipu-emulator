"""Randomised differential test: random VLIW words must behave identically.

The hand-written corpus in :mod:`etiss_corpus` covers every instruction, but
only with operand values a person thought to write down.  This test builds
random instruction *words* directly and requires the two backends to agree on
each one: either both raise, or both produce the same machine state.

Two generators run, because they probe different things:

``valid``
    Opcodes chosen at random, then each operand filled with a random value from
    the range its type actually allows.  These programs mostly run to
    completion, so the comparison covers real execution — every slot
    combination, not just the ones a person would write.

``raw``
    Every field uniformly random, including encodings the assembler cannot
    produce (an out-of-range cyclic index, a reserved stride encoding, a
    register index wider than the register file).  These almost always raise;
    the point is that both backends must reject the same words.  This is what
    caught the generated C indexing past the end of the register file where the
    Python emulator asserts.
"""

from __future__ import annotations

import random

import pytest

from ipu_as.compound_inst import CompoundInst
# The field-naming helpers are shared with the code generator on purpose: the
# test must address fields exactly the way the generated decoder does.
from ipu_as.gen_etiss import (
    _opcode_field_name,
    _operand_field_name,
    _slot_prefixes,
)
from ipu_common.activations import ACTIVATION_COUNT
from ipu_common.acc_stride_enums import ELEMENTS_IN_ROW_VALUES
from ipu_common.incr_mod_pow2_k import LR_MOD_POW2_K_ENCODED_MAX
from ipu_common.instruction_spec import (
    COMPOUND_LAYOUT_SLOT_ORDER,
    INSTRUCTION_SPEC,
    SLOT_UNIONS,
)
from ipu_common.registers import REGISTER_DEFINITIONS
from ipu_common.reshape_mask import RESHAPE_ELEMENT_COUNT, RESHAPE_MASK_LR_OFFSET

from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.etiss import EtissRunner, is_available
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState
from ipu_emu.xmem import XMEM_SIZE_BYTES

pytestmark = pytest.mark.skipif(
    not is_available(), reason="ETISS runner not built (set $IPU_ETISS_RUN)"
)

#: Keep runs short: a random branch can loop, and both backends hitting the
#: cycle limit is itself a valid outcome to compare.
_MAX_CYCLES = 3000

_WORDS_PER_PROGRAM = 3
_PROGRAMS_PER_DTYPE = 80

_DTYPES = (DType.INT8, DType.E4, DType.E5)

_COMPARED_REGISTERS = (
    "r", "r_cyclic", "r_mask", "r_acc", "post_aaq_reg", "lr", "cr",
    "mult_res", "mem_bypass",
)

_LR_COUNT = REGISTER_DEFINITIONS["lr"]["count"]
_CR_COUNT = REGISTER_DEFINITIONS["cr"]["count"]

#: Number of distinct values each operand type may legally encode.
_OPERAND_RANGE = {
    "LrIdx": _LR_COUNT,
    "CrIdx": _CR_COUNT,
    "DstructureCrIdx": _CR_COUNT,
    "LcrIdx": _LR_COUNT + _CR_COUNT,
    "LrdIdx": _LR_COUNT // 2,
    "MultStageReg": 2,
    "MultMaskOffsetImmediate": 8,
    "ElementsInRow": len(ELEMENTS_IN_ROW_VALUES),
    "HorizontalStride": 3,
    "VerticalStride": 3,
    "ActivationFn": ACTIVATION_COUNT,
    "LrModPow2KImmediate": LR_MOD_POW2_K_ENCODED_MAX + 1,
    "LrOrReshapeMaskImmediate": RESHAPE_MASK_LR_OFFSET + 1,
    "LrIncDecImmediate": 16,
    "AddbiImmediate": 256,
    "BreakImmediate": 1 << 16,
}

#: LR seed values. Cyclic loads only accept a slot boundary, so those appear
#: often enough that LDR_CYCLIC_MULT_REG actually executes sometimes; the small
#: values keep XMEM row operands and RESHAPE byte indices in range.
_LR_SEEDS = [0, 0, 0, 1, 2, 3, 4, 8, 16, 32, 128, 256, 384]

_FIELD_WIDTHS = dict(CompoundInst.get_fields())


def _slot_instances() -> list[tuple[str, str]]:
    """(slot, field-name prefix) for every sub-slot of the compound word."""
    return [
        (slot, prefix)
        for slot in COMPOUND_LAYOUT_SLOT_ORDER
        for prefix in _slot_prefixes(slot)
    ]


_SLOT_INSTANCES = _slot_instances()


def _random_valid_word(rng: random.Random, *, branch_target_limit: int,
                       dtype: DType) -> dict[str, int]:
    """A word whose operands are all inside the range their type allows."""
    word = {name: 0 for name, _ in CompoundInst.get_fields()}
    for slot, prefix in _SLOT_INSTANCES:
        names = list(INSTRUCTION_SPEC[slot])
        if slot == "aaq" and dtype != DType.INT8:
            # ACTIVATE.QUANTIZE is INT8-only by design and raises on both
            # backends under an FP8 dtype; leaving it in would make nearly
            # every FP8 program raise before exercising anything else.
            choice = names.index("NOP")
        else:
            choice = rng.randrange(len(names))
        word[_opcode_field_name(slot, prefix)] = choice

        instruction_name = names[choice]
        union = SLOT_UNIONS[slot]
        spec_operands = {
            op["name"]: op["type"]
            for op in INSTRUCTION_SPEC[slot][instruction_name]["operands"]
        }
        for field_idx, operand_name in union.opcode_bindings.get(instruction_name, []):
            field = _operand_field_name(prefix, field_idx,
                                        union.fields[field_idx].canonical_type)
            op_type = spec_operands[operand_name]
            if op_type == "Label":
                value = rng.randrange(branch_target_limit)
            else:
                limit = _OPERAND_RANGE.get(op_type, 1 << _FIELD_WIDTHS[field])
                value = rng.randrange(min(limit, 1 << _FIELD_WIDTHS[field]))
            word[field] = value
    return word


def _random_raw_word(rng: random.Random, *, branch_target_limit: int,
                     dtype: DType) -> dict[str, int]:
    """A word with every field uniformly random, malformed encodings included."""
    word: dict[str, int] = {}
    opcode_fields = {
        _opcode_field_name(slot, prefix): len(INSTRUCTION_SPEC[slot])
        for slot, prefix in _SLOT_INSTANCES
    }
    aaq_opcode = _opcode_field_name("aaq", _slot_prefixes("aaq")[0])
    for name, width in CompoundInst.get_fields():
        if name == aaq_opcode and dtype != DType.INT8:
            word[name] = list(INSTRUCTION_SPEC["aaq"]).index("NOP")
        elif name in opcode_fields:
            word[name] = rng.randrange(opcode_fields[name])
        elif name.endswith("label_token"):
            word[name] = rng.randrange(branch_target_limit)
        else:
            word[name] = rng.randrange(1 << width)
    return word


def _seed_state(rng: random.Random, dtype: DType) -> IpuState:
    state = IpuState(dtype=dtype)
    # Small CR values double as valid XMEM row bases and as valid dstructure
    # words (partition 0, pad mode zero), so both uses of a CR stay legal.
    for idx in range(2, _CR_COUNT):
        state.regfile.set_cr(idx, rng.randrange(0, 40))
    state.regfile.set_cr(
        _CR_COUNT - 1,
        rng.choice([128, 64, 17, (2 << 8) | 128, (4 << 8) | 96]),
    )
    for idx in range(_LR_COUNT):
        state.regfile.set_lr(idx, rng.choice(_LR_SEEDS))
    for name in ("r", "r_cyclic", "r_mask", "r_acc", "mult_res", "post_aaq_reg"):
        raw = state.regfile.raw(name)
        raw[:] = bytes(rng.randrange(256) for _ in range(len(raw)))
    for row in range(16):
        state.xmem.write_address(
            row * 128, bytes(rng.randrange(256) for _ in range(128))
        )
    return state


def _run(state: IpuState, backend: str) -> tuple[bool, str | None]:
    """(ran to completion, exception type name)."""
    try:
        if backend == "python":
            run_until_complete(state, max_cycles=_MAX_CYCLES)
        else:
            EtissRunner().run(state, max_cycles=_MAX_CYCLES)
    except Exception as exc:  # noqa: BLE001 - any failure is a comparable outcome
        return False, type(exc).__name__
    return True, None


def _compare(index: int, py: IpuState, et: IpuState) -> str | None:
    if py.program_counter != et.program_counter:
        return f"program {index}: PC python={py.program_counter} etiss={et.program_counter}"
    for name in _COMPARED_REGISTERS:
        if bytes(py.regfile.raw(name)) != bytes(et.regfile.raw(name)):
            return f"program {index}: register {name} differs"
    if bytes(py.xmem.read_address(0, XMEM_SIZE_BYTES)) != bytes(
        et.xmem.read_address(0, XMEM_SIZE_BYTES)
    ):
        return f"program {index}: XMEM differs"
    if py.stats != et.stats:
        return f"program {index}: stats python={py.stats} etiss={et.stats}"
    return None


def _fuzz(dtype: DType, generator, seed: int) -> tuple[list[str], int]:
    rng = random.Random(seed)
    mismatches: list[str] = []
    ran = 0

    for index in range(_PROGRAMS_PER_DTYPE):
        words = [
            generator(rng, branch_target_limit=_WORDS_PER_PROGRAM, dtype=dtype)
            for _ in range(_WORDS_PER_PROGRAM)
        ]
        state_seed = rng.randrange(1 << 30)

        states: dict[str, IpuState] = {}
        outcomes: dict[str, tuple[bool, str | None]] = {}
        for backend in ("python", "etiss"):
            state = _seed_state(random.Random(state_seed), dtype)
            load_program(state, [dict(w) for w in words])
            outcomes[backend] = _run(state, backend)
            states[backend] = state

        py_ok = outcomes["python"][0]
        et_ok = outcomes["etiss"][0]
        if py_ok != et_ok:
            mismatches.append(
                f"program {index}: python "
                f"{'ran' if py_ok else 'raised ' + str(outcomes['python'][1])}, "
                f"etiss {'ran' if et_ok else 'raised ' + str(outcomes['etiss'][1])}"
            )
            continue
        if not py_ok:
            continue
        ran += 1
        problem = _compare(index, states["python"], states["etiss"])
        if problem:
            mismatches.append(problem)
    return mismatches, ran


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: d.name)
def test_random_valid_words_agree(dtype: DType) -> None:
    mismatches, ran = _fuzz(dtype, _random_valid_word, 0xC0FFEE ^ int(dtype))
    assert not mismatches, "\n".join(mismatches[:10])
    # The comparison above is only meaningful if programs actually execute.
    assert ran >= _PROGRAMS_PER_DTYPE // 4, (
        f"only {ran} of {_PROGRAMS_PER_DTYPE} random programs ran to completion"
    )


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: d.name)
def test_random_raw_words_agree(dtype: DType) -> None:
    """Malformed encodings must be rejected by both backends, not just Python."""
    mismatches, _ran = _fuzz(dtype, _random_raw_word, 0xBADC0DE ^ int(dtype))
    assert not mismatches, "\n".join(mismatches[:10])
