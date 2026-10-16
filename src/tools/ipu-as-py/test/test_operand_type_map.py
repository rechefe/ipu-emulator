"""The assembler's token classes agree with the operand-type registry (E03-05).

The fake token classes below subclass ``EnumToken``, which keeps a list of its
subclasses for the SV codegen. Each test drops its fakes and collects them, so
none outlives the test in that list.
"""

from __future__ import annotations

import gc

import pytest

from ipu_as import immediate, ipu_token, reg
from ipu_as.inst import OPERAND_TYPE_MAP, check_operand_type_map


@pytest.fixture(autouse=True)
def _forget_fake_token_classes():
    yield
    gc.collect()


def _fake_enum(name: str, members: list[str]) -> type:
    return type(name, (ipu_token.EnumToken,), {"enum_array": classmethod(lambda cls: members)})


def _rejects(match: str, **replacements) -> None:
    with pytest.raises(ValueError, match=match):
        check_operand_type_map(dict(OPERAND_TYPE_MAP, **replacements))


def test_the_real_map_agrees():
    check_operand_type_map(OPERAND_TYPE_MAP)


def test_a_missing_token_class_names_the_type():
    copy = dict(OPERAND_TYPE_MAP)
    del copy["LrdIdx"]
    with pytest.raises(ValueError, match=r"missing \['LrdIdx'\]"):
        check_operand_type_map(copy)


def test_a_class_with_another_name_is_rejected():
    _rejects(
        r"OPERAND_TYPE_MAP\['LrIdx'\] is LoopRegField.*LrRegField",
        LrIdx=_fake_enum("LoopRegField", reg.LrRegField.enum_array()),
    )


def test_enum_members_must_match_the_registry_domain():
    _rejects(
        "members .* differ from the operand-type registry domain of LrIdx",
        LrIdx=_fake_enum("LrRegField", list(reversed(reg.LrRegField.enum_array()))),
    )


def test_width_must_match_the_registry():
    wide = type("AddbiImmediate", (immediate.AddbiImmediate,), {"bits": classmethod(lambda cls: 9)})
    _rejects(r"AddbiImmediate: AddbiImmediate is 9 bits, but the registry says 8", AddbiImmediate=wide)


def test_value_range_must_match_the_registry():
    narrow = type(
        "MultMaskOffsetImmediate",
        (immediate.MultMaskOffsetImmediate,),
        {"value_range": classmethod(lambda cls: (0, 3))},
    )
    _rejects(
        r"MultMaskOffsetImmediate: .* accepts \(0, 3\), but the registry says \(0, 7\)",
        MultMaskOffsetImmediate=narrow,
    )

