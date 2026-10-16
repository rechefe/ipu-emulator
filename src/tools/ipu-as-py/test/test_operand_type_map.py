"""The assembler's token classes agree with the operand-type registry (E03-05)."""

from __future__ import annotations

import pytest

from ipu_as import ipu_token, reg
from ipu_as.inst import OPERAND_TYPE_MAP, check_operand_type_map


def test_the_real_map_agrees():
    check_operand_type_map(OPERAND_TYPE_MAP)


def test_a_missing_token_class_names_the_type():
    copy = dict(OPERAND_TYPE_MAP)
    del copy["LrdIdx"]
    with pytest.raises(ValueError, match=r"missing \['LrdIdx'\]"):
        check_operand_type_map(copy)


def test_a_class_with_another_name_is_rejected():
    class LoopRegField(ipu_token.EnumToken):
        @classmethod
        def enum_array(cls):
            return reg.LrRegField.enum_array()

    copy = dict(OPERAND_TYPE_MAP, LrIdx=LoopRegField)
    with pytest.raises(ValueError, match=r"OPERAND_TYPE_MAP\['LrIdx'\] is LoopRegField.*LrRegField"):
        check_operand_type_map(copy)


def test_enum_members_must_match_the_registry_domain():
    class LrRegField(ipu_token.EnumToken):
        @classmethod
        def enum_array(cls):
            return ["lr0", "lr1"]

    copy = dict(OPERAND_TYPE_MAP, LrIdx=LrRegField)
    with pytest.raises(ValueError, match="members .* differ from the operand-type registry domain of LrIdx"):
        check_operand_type_map(copy)
