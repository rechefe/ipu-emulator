"""Smoke tests for the union layout SVG renderer.

We don't try to assert visual fidelity here — just that the renderer produces
a syntactically valid-looking SVG that mentions every slot, every opcode, and
every field's canonical type.  Visual review is up to the docs build.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from ipu_common.instruction_spec import (
    COMPOUND_LAYOUT_SLOT_ORDER,
    SLOT_COUNT,
    SLOT_UNIONS,
)
from ipu_common.union_layout_svg import render_union_layout_svg


def _render() -> str:
    return render_union_layout_svg(
        SLOT_UNIONS,
        slot_order=COMPOUND_LAYOUT_SLOT_ORDER,
        slot_counts=dict(SLOT_COUNT),
    )


def test_renders_valid_xml():
    svg = _render()
    # ElementTree raises ParseError on malformed XML.
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert root.attrib["width"]
    assert root.attrib["height"]


def test_every_slot_appears():
    svg = _render()
    for slot in COMPOUND_LAYOUT_SLOT_ORDER:
        assert slot.upper() in svg, f"slot {slot!r} missing from rendered SVG"


def test_every_opcode_appears():
    svg = _render()
    for su in SLOT_UNIONS.values():
        for opcode in su.opcode_bindings:
            assert opcode in svg, f"opcode {opcode!r} missing from rendered SVG"


def test_every_canonical_type_appears():
    svg = _render()
    for su in SLOT_UNIONS.values():
        for f in su.fields:
            assert f.canonical_type in svg, (
                f"canonical type {f.canonical_type!r} missing from rendered SVG"
            )


def test_lr_multiplicity_annotated():
    svg = _render()
    # SLOT_COUNT["lr"] is 3 — title should call this out.
    assert "×3" in svg


def test_msb_to_lsb_slot_order():
    """Slot block titles appear in compound-instruction MSB → LSB order."""
    svg = _render()
    positions: list[int] = []
    for slot in COMPOUND_LAYOUT_SLOT_ORDER:
        # Match slot title lines from union_layout_svg._render_slot, not opcode
        # substrings (e.g. MULT.VE.AAQ must not satisfy slot "aaq").
        needle = f">{slot.upper()} "
        pos = svg.find(needle)
        if pos == -1:
            pos = svg.find(f">{slot.upper()} (")
        assert pos != -1, f"slot title for {slot!r} not found in SVG"
        positions.append(pos)
    assert positions == sorted(positions), (
        f"slots out of order: {list(zip(COMPOUND_LAYOUT_SLOT_ORDER, positions))}"
    )
