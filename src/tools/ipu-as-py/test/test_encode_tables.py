"""The ISA tables the encoder reads are built once and shared read-only (E05-01)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import lark
import pytest

import ipu_as.ipu_token as ipu_token
from ipu_as.compound_inst import CompoundInst
from ipu_as.opcodes import Opcode
from ipu_as.session import AssemblySession
from ipu_isa.operand_types import snake_name
from ipu_isa.slots import LAYOUT_ORDER, SLOTS
from ipu_isa.view import isa

# Bazel runs the test from the runfiles root of the main repository.
ENUM_SNAPSHOT = Path.cwd() / "src/tools/ipu-as-py/test/baselines/enum_descriptors.json"

#: One word per line, touching most slots, registers, immediates and a local
#: label (so the program can be repeated in one source).
PROGRAM = """\
1:
SET LR0 CR1; LD.MULT R0 LR0 CR0; MULT.RC.VV LR0 R0 0 LR2 CR15; ACC.ADD;;
ADD LR0 LR1 LR2; SUB LR3 LR1 CR5; ST.AAQ LR0 CR0; ACC.MAX;;
ACTIVATE.QUANTIZE relu CR15; BREAK.IFEQ LR0 10; BNE LR0 CR0 1b;;
NOP;;
"""


def _assemble(source: str) -> tuple[int, ...]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AssemblySession(source).assemble().words


def test_no_table_is_rebuilt_per_instruction() -> None:
    _assemble(PROGRAM)  # warm: every table is built by now
    reverse_maps_before = dict(ipu_token._REVERSE_MAPS)
    fields_before = CompoundInst.get_fields()

    words = _assemble(PROGRAM * 25)

    assert len(words) == 100
    assert ipu_token._REVERSE_MAPS == reverse_maps_before
    assert all(
        ipu_token._REVERSE_MAPS[cls] is reverse for cls, reverse in reverse_maps_before.items()
    )
    assert CompoundInst.get_fields() is fields_before


def test_cached_tables_encode_like_fresh_ones() -> None:
    """Clearing the token caches and assembling again gives the same words."""
    expected = _assemble(PROGRAM)
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
        classes[snake_name(cls.__name__)] = cls
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
    for slot in LAYOUT_ORDER:
        for instruction in isa().instructions(slot):
            if instruction.mnemonic == "NOP":
                continue
            token = lark.Token("OPCODE", instruction.mnemonic.lower())
            assert Opcode.find_opcode_class(token).__name__ == SLOTS[slot].asm_opcode_class
    # NOP is in every slot; the lookup keeps the first slot defined.
    nop = lark.Token("OPCODE", "NOP")
    assert Opcode.find_opcode_class(nop) is Opcode.__subclasses__()[0]
