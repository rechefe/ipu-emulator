"""Geometry and ISA constants live in ipu_isa (E03-12)."""

from __future__ import annotations

import copy
import itertools

import pytest

from ipu_isa.constants import (
    CR_ONE_REG_INDEX,
    CR_READ_ONLY_INITIAL_VALUES,
    CR_ZERO_REG_INDEX,
    DEFAULT_DSTRUCTURE,
    DStructureConfig,
    PadMode,
    Partition,
    decode_dstructure,
    encode_dstructure,
)
from ipu_isa.geometry import INST_MEM_WORDS, LANES, R_CYCLIC_ELEMENTS, TAGS, XMEM_BYTES
from ipu_isa.instruction_spec import (
    INSTRUCTION_SPEC,
    PSEUDO_INSTRUCTION_SPEC,
    validate_pseudo_instruction_spec,
)
from ipu_isa.schema import SpecError
from ipu_isa.tags import Tag


def test_geometry():
    assert LANES == 128
    assert INST_MEM_WORDS == 1024
    assert R_CYCLIC_ELEMENTS == 512
    assert XMEM_BYTES == 1 << 29
    assert TAGS["XMEM_BYTES"] is Tag.EMULATOR
    assert TAGS["LANES"] is Tag.HARDWARE


def test_constant_registers():
    assert dict(CR_READ_ONLY_INITIAL_VALUES) == {CR_ZERO_REG_INDEX: 0, CR_ONE_REG_INDEX: 1}
    with pytest.raises(TypeError):
        CR_READ_ONLY_INITIAL_VALUES[2] = 2


@pytest.mark.parametrize("partition, pad_mode", list(itertools.product(Partition, PadMode)))
@pytest.mark.parametrize("valid_elements", [0, 1, 64, LANES])
def test_dstructure_round_trip(valid_elements, partition, pad_mode):
    word = encode_dstructure(valid_elements=valid_elements, partition=partition, pad_mode=pad_mode)
    config = DStructureConfig(valid_elements, partition, pad_mode)
    assert decode_dstructure(word) == config
    assert config.to_register_value() == word


def test_default_dstructure():
    assert DEFAULT_DSTRUCTURE == DStructureConfig(LANES, Partition.P0, PadMode.ZERO)


def _pseudo_with_b_literal(literal: str) -> dict:
    pseudo = copy.deepcopy(PSEUDO_INSTRUCTION_SPEC)
    pseudo["B"]["expands_to"]["args"] = [literal, literal, "label"]
    return pseudo


def test_a_pseudo_literal_must_be_a_constant_register():
    with pytest.raises(SpecError, match=r"pseudo\.B\.expands_to\.args\[.*'CR2' is not a constant"):
        validate_pseudo_instruction_spec(_pseudo_with_b_literal("CR2"), INSTRUCTION_SPEC)


def test_a_pseudo_literal_must_be_documented():
    """CR1 is a constant register, but B's docs only name CR0."""
    with pytest.raises(SpecError, match=r"pseudo\.B\.expands_to\.args\[.*docs do not say so"):
        validate_pseudo_instruction_spec(_pseudo_with_b_literal("CR1"), INSTRUCTION_SPEC)
