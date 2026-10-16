"""The strict schema check of the instruction-spec dicts (E03-03).

Each test deep-copies the real spec, plants one violation and checks that the
validator rejects it with the path of the offending entry.
"""

from __future__ import annotations

import copy

import pytest

from ipu_isa.instruction_spec import (
    INSTRUCTION_SPEC,
    PSEUDO_INSTRUCTION_SPEC,
    SLOT_METADATA,
    validate_instruction_spec,
    validate_pseudo_instruction_spec,
)
from ipu_isa.schema import SpecError


def _spec():
    return copy.deepcopy(INSTRUCTION_SPEC)


def _rejects(spec, match, slot_metadata=SLOT_METADATA):
    with pytest.raises(SpecError, match=match):
        validate_instruction_spec(spec, slot_metadata)


def _rejects_pseudo(pseudo, match):
    with pytest.raises(SpecError, match=match):
        validate_pseudo_instruction_spec(pseudo, INSTRUCTION_SPEC)


def _operand(spec, slot, inst, name):
    return next(op for op in spec[slot][inst]["operands"] if op["name"] == name)


def test_the_real_spec_passes():
    validate_instruction_spec(INSTRUCTION_SPEC, SLOT_METADATA)
    validate_pseudo_instruction_spec(PSEUDO_INSTRUCTION_SPEC, INSTRUCTION_SPEC)


def test_spec_error_is_an_ipu_error_and_a_value_error():
    from ipu_isa.errors import IpuError

    assert issubclass(SpecError, IpuError) and issubclass(SpecError, ValueError)


def test_unknown_operand_key():
    spec = _spec()
    operand = _operand(spec, "acc", "ACC.RESHAPE", "source")
    operand["raed"] = operand.pop("read")
    _rejects(spec, r"acc\.ACC\.RESHAPE\.operands\[source\]: unknown key\(s\) 'raed'")


def test_unknown_instruction_key():
    spec = _spec()
    spec["lr"]["ADD"]["tags"] = "hardware"
    _rejects(spec, r"lr\.ADD: unknown key\(s\) 'tags'")


def test_missing_instruction_key():
    spec = _spec()
    del spec["lr"]["ADD"]["execute_fn"]
    _rejects(spec, r"lr\.ADD: missing key\(s\) 'execute_fn'")


@pytest.mark.parametrize(
    ("level", "match"),
    [
        ("instruction", r"lr\.ADD: missing key\(s\) 'execute_fn'"),
        ("operand", r"lr\.ADD\.operands\[dest\]: missing key\(s\) 'type'"),
        ("slot_metadata", r"SLOT_METADATA\.lr: missing key\(s\) 'description'"),
        ("pseudo", r"pseudo\.B: missing key\(s\) 'doc'"),
        ("expands_to", r"pseudo\.B\.expands_to: missing key\(s\) 'slot'"),
    ],
)
def test_missing_key_at_every_level(level, match):
    spec, metadata = _spec(), copy.deepcopy(SLOT_METADATA)
    pseudo = copy.deepcopy(PSEUDO_INSTRUCTION_SPEC)
    {
        "instruction": lambda: spec["lr"]["ADD"].pop("execute_fn"),
        "operand": lambda: spec["lr"]["ADD"]["operands"][0].pop("type"),
        "slot_metadata": lambda: metadata["lr"].pop("description"),
        "pseudo": lambda: pseudo["B"].pop("doc"),
        "expands_to": lambda: pseudo["B"]["expands_to"].pop("slot"),
    }[level]()
    with pytest.raises(SpecError, match=match):
        validate_instruction_spec(spec, metadata)
        validate_pseudo_instruction_spec(pseudo, spec)


def test_unknown_slot_metadata_key():
    metadata = copy.deepcopy(SLOT_METADATA)
    metadata["mult"]["hardwear"] = True
    _rejects(_spec(), r"SLOT_METADATA\.mult: unknown key\(s\) 'hardwear'", metadata)


def test_duplicate_operand_name():
    spec = _spec()
    spec["lr"]["ADD"]["operands"][1]["name"] = "dest"
    _rejects(spec, r"lr\.ADD: duplicate operand name 'dest'")


def test_operand_name_pattern():
    spec = _spec()
    spec["lr"]["ADD"]["operands"][0]["name"] = "Dest"
    _rejects(spec, r"lr\.ADD\.operands\[Dest\]\.name: 'Dest' does not match")


def test_unknown_operand_type():
    spec = _spec()
    spec["lr"]["ADD"]["operands"][0]["type"] = "LrIndex"
    _rejects(spec, r"lr\.ADD\.operands\[dest\]\.type: unknown operand type 'LrIndex'")


def test_bad_read_source():
    spec = _spec()
    _operand(spec, "lr", "ADD", "src_a")["read"] = "cycle_start"
    _rejects(spec, r"lr\.ADD\.operands\[src_a\]\.read: 'cycle_start' is not one of")


def test_read_on_a_non_register_operand():
    spec = _spec()
    _operand(spec, "aaq", "ACTIVATE.QUANTIZE", "activation_fn")["read"] = "snapshot"
    _rejects(
        spec,
        r"aaq\.ACTIVATE\.QUANTIZE\.operands\[activation_fn\]\.read: "
        r"a ActivationFn operand is not a register",
    )


def test_doc_operand_names_must_match():
    spec = _spec()
    doc = spec["lr"]["ADD"]["doc"]
    doc.operands = ["dst: wrong name"] + list(doc.operands[1:])
    _rejects(spec, r"lr\.ADD\.doc\.operands: names \['dst', .*\] do not match")


def test_example_has_no_comma():
    spec = _spec()
    spec["lr"]["ADD"]["doc"].example = "ADD LR0, LR1, LR2;;"
    _rejects(spec, r"lr\.ADD\.doc\.example: separate operands with spaces")


def test_mnemonics_unique_after_upper_case():
    spec = _spec()
    spec["lr"]["Nop"] = spec["lr"]["NOP"]
    _rejects(spec, r"lr: duplicate mnemonic \(compared in upper case\) 'NOP'")


def test_exactly_one_nop_per_slot():
    spec = _spec()
    del spec["break"]["NOP"]
    _rejects(spec, r"break: must have exactly one NOP, has 0")


def test_spec_slots_must_be_the_registered_slots():
    spec = _spec()
    del spec["aaq"]
    _rejects(spec, "INSTRUCTION_SPEC: slots .* do not match the slot registry")


def test_slot_metadata_covers_every_slot():
    metadata = copy.deepcopy(SLOT_METADATA)
    del metadata["aaq"]
    _rejects(_spec(), r"SLOT_METADATA: slots .* do not match", metadata)


def test_pseudo_unknown_key():
    pseudo = copy.deepcopy(PSEUDO_INSTRUCTION_SPEC)
    pseudo["B"]["execute_fn"] = "execute_b"
    _rejects_pseudo(pseudo, r"pseudo\.B: unknown key\(s\) 'execute_fn'")


def test_pseudo_unknown_expansion_key():
    pseudo = copy.deepcopy(PSEUDO_INSTRUCTION_SPEC)
    pseudo["B"]["expands_to"]["arguments"] = []
    _rejects_pseudo(pseudo, r"pseudo\.B\.expands_to: unknown key\(s\) 'arguments'")


def test_pseudo_literal_must_be_a_valid_register():
    pseudo = copy.deepcopy(PSEUDO_INSTRUCTION_SPEC)
    pseudo["BZ"]["expands_to"]["args"][1] = "CR16"
    _rejects_pseudo(pseudo, r"pseudo\.BZ\.expands_to\.args\[reg2\]: literal 'CR16' is neither")


def test_pseudo_literal_may_be_any_register_in_the_target_domain():
    """A register-pair literal is valid for an LrdIdx target (ADDB's dest)."""
    pseudo = copy.deepcopy(PSEUDO_INSTRUCTION_SPEC)
    pseudo["B"] = copy.deepcopy(pseudo["B"])
    pseudo["B"]["operands"] = [{"name": "src", "type": "LcrIdx"}]
    pseudo["B"]["doc"].operands = ["src: source"]
    pseudo["B"]["expands_to"] = {"slot": "lr", "instruction": "ADDB", "args": ["LRD0", "src"]}
    validate_pseudo_instruction_spec(pseudo, INSTRUCTION_SPEC)


def test_pseudo_expansion_target_must_exist():
    pseudo = copy.deepcopy(PSEUDO_INSTRUCTION_SPEC)
    pseudo["B"]["expands_to"]["instruction"] = "BEQQ"
    _rejects_pseudo(pseudo, r"pseudo\.B\.expands_to\.instruction: 'BEQQ' is not in slot")


def test_pseudo_args_count():
    pseudo = copy.deepcopy(PSEUDO_INSTRUCTION_SPEC)
    pseudo["B"]["expands_to"]["args"].append("label")
    _rejects_pseudo(pseudo, r"pseudo\.B\.expands_to\.args: must be a list of 3 entries")
