"""Geometry and ISA constants live in ipu_isa (E03-12)."""

from __future__ import annotations

import copy
import dataclasses

import pytest

from ipu_isa.constants import (
    CR_ONE_REG_INDEX,
    CR_READ_ONLY_INITIAL_VALUES,
    CR_ZERO_REG_INDEX,
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


def test_every_geometry_size_is_tagged():
    import ipu_isa.geometry as geometry

    sizes = {
        name for name, value in vars(geometry).items() if name.isupper() and isinstance(value, int)
    }
    assert sizes == set(TAGS)


def test_constant_registers():
    assert dict(CR_READ_ONLY_INITIAL_VALUES) == {CR_ZERO_REG_INDEX: 0, CR_ONE_REG_INDEX: 1}
    with pytest.raises(TypeError):
        CR_READ_ONLY_INITIAL_VALUES[2] = 2


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


def test_a_longer_register_name_does_not_document_a_literal():
    """Docs naming CR15 (or a range ending in CR15) do not name CR1."""
    pseudo = _pseudo_with_b_literal("CR1")
    doc = pseudo["B"]["doc"]
    pseudo["B"]["doc"] = dataclasses.replace(
        doc, notes="Compares against CR15; any of CR0–CR15 works."
    )
    with pytest.raises(SpecError, match="docs do not say so"):
        validate_pseudo_instruction_spec(pseudo, INSTRUCTION_SPEC)
