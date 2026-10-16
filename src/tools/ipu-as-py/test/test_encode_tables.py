"""The ISA tables the encoder reads are built once and shared read-only (E05-01)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import lark
import pytest

import ipu_as.inst as inst
import ipu_as.ipu_token as ipu_token
import ipu_as.utils as utils
from ipu_as.compound_inst import SLOT_TO_INST, CompoundInst
from ipu_as.opcodes import Opcode
from ipu_as.session import AssemblySession

# Bazel runs the test from the runfiles root of the main repository.
ENUM_SNAPSHOT = Path.cwd() / "src/tools/ipu-as-py/test/baselines/enum_descriptors.json"

#: One word per line, touching most slots, registers, immediates and a local
#: label (so the program can be repeated in one source).
PROGRAM = """\
1:
SET LR0 CR1; LDR_MULT_REG R0 LR0 CR0; MULT.RC.VV LR0 R0 0 LR2 CR15; ACC.ADD;;
ADD LR0 LR1 LR2; SUB LR3 LR1 CR5; STR_POST_AAQ_REG LR0 CR0; ACC.MAX;;
ACTIVATE.QUANTIZE relu CR15; BREAK.IFEQ LR0 10; BNE LR0 CR0 1b;;
NOP;;
"""


def _assemble(source: str) -> tuple[int, ...]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AssemblySession(source).assemble().words


INST_CLASSES = sorted(set(SLOT_TO_INST.values()), key=lambda cls: cls.__name__)


@pytest.mark.parametrize("inst_class", INST_CLASSES, ids=lambda cls: cls.__name__)
def test_struct_table_is_one_shared_read_only_object(inst_class) -> None:
    table = inst_class.struct_by_opcode_table()
    assert table is inst_class.struct_by_opcode_table()
    with pytest.raises(TypeError):
        table["NEW_OPCODE"] = inst.InstructionFormat(operands=())
    with pytest.raises(TypeError):
        del table[next(iter(table))]


def test_no_table_is_rebuilt_per_instruction() -> None:
    _assemble(PROGRAM)  # warm: every table is built by now
    struct_before = inst._build_struct_table.cache_info()
    reverse_maps_before = dict(ipu_token._REVERSE_MAPS)
    fields_before = CompoundInst.get_fields()

    words = _assemble(PROGRAM * 25)

    assert len(words) == 100
    struct_after = inst._build_struct_table.cache_info()
    assert struct_after.misses == struct_before.misses, "a slot table was rebuilt"
    assert struct_after.hits > struct_before.hits
    assert ipu_token._REVERSE_MAPS == reverse_maps_before
    assert all(
        ipu_token._REVERSE_MAPS[cls] is reverse for cls, reverse in reverse_maps_before.items()
    )
    assert CompoundInst.get_fields() is fields_before


def test_cached_tables_encode_like_fresh_ones() -> None:
    """Clearing every cache and assembling again gives the same words."""
    expected = _assemble(PROGRAM)
    inst._build_struct_table.cache_clear()
    inst._inst_type_by_opcode.cache_clear()
    inst._default_token.cache_clear()
    ipu_token._REVERSE_MAPS.clear()
    Opcode._class_by_opcode = None
    assert _assemble(PROGRAM) == expected


def _enum_classes() -> dict[str, type[ipu_token.EnumToken]]:
    """Every EnumToken subclass, by the descriptor name the code generators use."""
    classes: dict[str, type[ipu_token.EnumToken]] = {}
    pending = list(ipu_token.EnumToken.__subclasses__())
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        try:
            cls.enum_array()
        except NotImplementedError:
            continue
        classes[utils.camel_case_to_snake_case(cls.__name__)] = cls
    return classes


def _enum_cases() -> list[tuple[str, str, int]]:
    """(descriptor, member as written, value pinned at the base commit)."""
    snapshot = json.loads(ENUM_SNAPSHOT.read_text())
    descriptors = ipu_token.EnumToken.get_all_enum_descriptors()
    assert set(descriptors) == set(snapshot)
    classes = _enum_classes()
    cases = []
    for name, members in sorted(descriptors.items()):
        assert [[value, member] for value, member in members] == snapshot[name]
        for value, member in enumerate(classes[name].enum_array()):
            pinned = [v for v, m in snapshot[name] if m == member.upper().replace(".", "_")]
            # A name that occurs twice encodes as its last occurrence.
            cases.append((name, member, max(pinned)))
            assert value in pinned
    return cases


@pytest.mark.parametrize(("descriptor", "member", "value"), _enum_cases())
def test_enum_member_encodes_to_its_pinned_value(descriptor, member, value) -> None:
    cls = _enum_classes()[descriptor]
    for spelling in (member, member.lower(), member.upper()):
        token = ipu_token.AnnotatedToken(lark.Token("ENUM", spelling), 0)
        assert cls(token).encode() == value


def test_opcode_lookups_resolve_every_opcode() -> None:
    for inst_class in INST_CLASSES:
        for opcode in inst_class.struct_by_opcode_table():
            if opcode == "NOP":
                continue
            assert inst.Inst.find_inst_type_by_opcode(opcode.lower()) is inst_class
            token = lark.Token("OPCODE", opcode)
            assert Opcode.find_opcode_class(token) is inst_class.opcode_type()
    # NOP is in every slot; both lookups keep the first slot defined.
    assert inst.Inst.find_inst_type_by_opcode("nop") is inst.Inst.__subclasses__()[0]
    nop = lark.Token("OPCODE", "NOP")
    assert Opcode.find_opcode_class(nop) is Opcode.__subclasses__()[0]
