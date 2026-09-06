"""Tests for the ISA alias table.

The table only references INSTRUCTION_SPEC — it must never restate instruction
metadata — so these tests mostly guard that the references stay live and that
the constraint vocabulary is enforced.
"""

from __future__ import annotations

import copy

import pytest

from ipu_common.instruction_spec import INSTRUCTION_SPEC
from ipu_common.isa_alias_spec import (
    ISA_ALIAS_SPEC,
    aliases_for,
    validate_isa_alias_spec,
)


class TestTableIsWellFormed:
    def test_shipped_table_validates(self):
        validate_isa_alias_spec()

    def test_every_reference_resolves(self):
        for alias_id, entry in ISA_ALIAS_SPEC.items():
            for match in entry["matches"]:
                slot, instruction = match["slot"], match["instruction"]
                assert instruction in INSTRUCTION_SPEC[slot], (
                    f"{alias_id} references {slot}/{instruction}"
                )

    def test_operand_references_are_real_operands(self):
        for alias_id, entry in ISA_ALIAS_SPEC.items():
            for match in entry["matches"]:
                names = {
                    op["name"]
                    for op in INSTRUCTION_SPEC[match["slot"]][match["instruction"]][
                        "operands"
                    ]
                }
                for constraint in match["where"]:
                    if "operand" in constraint:
                        assert constraint["operand"] in names, alias_id


class TestValidatorRejectsMistakes:
    """The validator is the thing standing between a typo and a silent miscount."""

    def _validate_with(self, mutate):
        original = copy.deepcopy(ISA_ALIAS_SPEC)
        try:
            mutate(ISA_ALIAS_SPEC)
            with pytest.raises(ValueError):
                validate_isa_alias_spec()
        finally:
            ISA_ALIAS_SPEC.clear()
            ISA_ALIAS_SPEC.update(original)

    def test_unknown_instruction(self):
        self._validate_with(
            lambda spec: spec.__setitem__(
                "BOGUS",
                {
                    "missing_op": "x",
                    "summary": "x",
                    "matches": [
                        {"slot": "mult", "instruction": "NO_SUCH_INSTRUCTION",
                         "where": []}
                    ],
                },
            )
        )

    def test_unknown_operand(self):
        self._validate_with(
            lambda spec: spec.__setitem__(
                "BOGUS",
                {
                    "missing_op": "x",
                    "summary": "x",
                    "matches": [
                        {"slot": "mult", "instruction": "MULT.RC.VE",
                         "where": [{"operand": "no_such_operand", "scalar_is": 1}]}
                    ],
                },
            )
        )

    def test_unsupported_constraint(self):
        self._validate_with(
            lambda spec: spec.__setitem__(
                "BOGUS",
                {
                    "missing_op": "x",
                    "summary": "x",
                    "matches": [
                        {"slot": "mult", "instruction": "MULT.RC.VE",
                         "where": [{"vibes": "good"}]}
                    ],
                },
            )
        )

    def test_existing_operand_that_is_not_the_scalar(self):
        self._validate_with(
            lambda spec: spec["A1_MOV_RC"]["matches"][0]["where"][0].update(
                operand="rc_idx"
            )
        )

    @pytest.mark.parametrize("operand", ["mask_shift", "rc_idx", "src", "cr_idx"])
    def test_immediate_constraint_rejects_register_operands(self, operand):
        # Cover both auto-resolved LR operands and raw register indices. Merely
        # rejecting operands with a "read" key would miss src and cr_idx.
        self._validate_with(
            lambda spec: spec["A1_MOV_RC"]["matches"][0]["where"].append(
                {"operand": operand, "immediate_ne": 0}
            )
        )

    @pytest.mark.parametrize("constraints", [
        [{"co_slot": "acc", "instruction_in": ["ACC.ADD"]},
         {"co_slot": "acc", "instruction_in": ["ACC.SUB"]}],
        [{"operand": "mask_offset", "immediate_ne": 0},
         {"operand": "mask_offset", "immediate_ne": 1}],
    ])
    def test_multiple_constraints_of_the_same_kind(self, constraints):
        self._validate_with(
            lambda spec: spec["A1_MOV_RC"]["matches"][0]["where"].extend(constraints)
        )

    def test_hardware_gate_cannot_ignore_other_constraints(self):
        self._validate_with(
            lambda spec: spec["A16_STR_ACC_REG"]["matches"][0]["where"].append(
                {"co_slot": "acc", "instruction_in": ["ACC.ADD"]}
            )
        )

    def test_identical_matches_are_rejected_regardless_of_constraint_order(self):
        def duplicate(spec):
            entry = copy.deepcopy(spec["A2_ADD_VV"])
            for match in entry["matches"]:
                match["where"].reverse()
            spec["DUPLICATE"] = entry
        self._validate_with(duplicate)

    def test_hardware_only_on_a_real_hardware_slot(self):
        # 'store' is real hardware, so hardware_only makes no sense there.
        self._validate_with(
            lambda spec: spec.__setitem__(
                "BOGUS",
                {
                    "missing_op": "x",
                    "summary": "x",
                    "matches": [
                        {"slot": "store", "instruction": "STR_POST_AAQ_REG",
                         "where": [{"hardware_only": True}]}
                    ],
                },
            )
        )


class TestPrecedence:
    """Order is precedence: hits must partition, so the order is load-bearing."""

    def test_constrained_entries_precede_catch_alls(self):
        order = [alias for alias, _ in aliases_for("mult", "MULT.RC.VE")]
        assert order.index("A4_AGG_RC") < order.index("A1_MOV_RC")
        assert order.index("A3_SUB_VV") < order.index("A1_MOV_RC")
        assert order.index("A2_ADD_VV") < order.index("A1_MOV_RC")

    def test_mult_ee_falls_through_to_broadcast_last(self):
        order = [alias for alias, _ in aliases_for("mult", "MULT.EE")]
        # A MULT.EE co-issued with ACC.SUB is a subtract that happens to
        # broadcast, not a bare broadcast.
        assert order.index("A3_SUB_VV") < order.index("A5_BCAST")
        assert order[-1] == "A5_BCAST"

    def test_vector_by_vector_requires_declared_ones(self):
        matches = aliases_for("mult", "MULT.RC.VV")
        assert matches
        assert all({"vector_ones": True} in match["where"] for _, match in matches)


@pytest.mark.parametrize('instruction,where', [
    ('MULT.RC.VS', [{'vector_ones': True}]),
    ('MULT.RC.VV', [{'vector_ones': False}]),
    ('MULT.RC.VV', [{'vector_ones': True}, {'masked_window': False}]),
    ('MULT.RC.VE', [{'vector_ones': True}, {'operand': 'src', 'scalar_is': 1}]),
])
def test_invalid_provenance_constraints_are_rejected(instruction, where):
    TestValidatorRejectsMistakes()._validate_with(lambda spec: spec.__setitem__('BAD', {
        'missing_op': 'bad', 'summary': 'bad',
        'matches': [{'slot': 'mult', 'instruction': instruction, 'where': where}],
    }))
