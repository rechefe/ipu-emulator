"""Snapshots of the decoded-word names the assembler and emulator agree on (E02-02).

Three things are wire format:

* ``word_fields.json``: ``CompoundInst.get_fields()``, the name and width of
  every field of an instruction word, LSB first, plus the total width (from
  ``ipu_isa.codec.word.WORD_LAYOUT`` since E05-02).
* ``emulator_field_maps.json``: for every slot (and LR sub-slot) and
  instruction, the operand-name to field-name map the emulator dispatches on,
  and the field each slot instance reads its opcode from.
* ``enum_descriptors.json``: every enum descriptor, with its members in value
  order, as ``EnumToken.get_all_enum_descriptors()`` builds them for the code
  generators.

A refactor must leave all three unchanged. ``test_decode_snapshots`` compares
them; ``bazel run //src/tools/ipu-as-py:update_decode_snapshots`` rewrites
them after an intended change.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from ipu_as.compound_inst import CompoundInst
from ipu_as.ipu_token import EnumToken
from ipu_emu import ipu
from ipu_isa.slots import Slot
from ipu_isa.timing import SLOT_EXECUTION_ORDER

#: Where the snapshots live, repo-relative.
SNAPSHOT_DIR = "src/tools/ipu-as-py/test/baselines"

#: The one command that rewrites them from the current code.
UPDATE_COMMAND = "bazel run //src/tools/ipu-as-py:update_decode_snapshots"


def word_fields() -> dict:
    fields = CompoundInst.get_fields()
    return {
        "total_bits": CompoundInst.bits(),
        "fields_lsb_first": [[name, bits] for name, bits in fields],
    }


def emulator_field_maps() -> dict:
    # Read from the emulator's public decode tables (ipu.plans_for): each plan
    # holds its (operand name, field name) pairs. Pinning them is the point.
    field_maps: dict[str, dict[str, dict[str, str]]] = {}
    opcode_fields: dict[str, str] = {}
    for slot in SLOT_EXECUTION_ORDER:
        positions = ipu.plans_for(slot)
        for index, (opcode_field, plans) in enumerate(positions):
            instance = f"{slot.value}[{index}]" if slot is Slot.LR else slot.value
            opcode_fields[instance] = opcode_field
            for plan in plans:
                field_maps.setdefault(instance, {})[plan.inst_name] = dict(plan.fields)
    return {"opcode_fields": opcode_fields, "operand_fields": field_maps}


def enum_descriptors() -> dict:
    return {
        name: [[value, member] for value, member in members]
        for name, members in EnumToken.get_all_enum_descriptors().items()
    }


#: Snapshot file name -> the function producing its content.
SNAPSHOTS: dict[str, Callable[[], dict]] = {
    "word_fields.json": word_fields,
    "emulator_field_maps.json": emulator_field_maps,
    "enum_descriptors.json": enum_descriptors,
}


def render(content: dict) -> str:
    """Indented JSON with each ``[name, bits]`` / ``[value, member]`` pair on
    one line, so a review diff shows one changed line per field or member."""
    text = json.dumps(content, indent=2, sort_keys=True)
    return re.sub(r"\[\s+(\S+),\s+(\S+)\s+\]", r"[\1, \2]", text) + "\n"


def write_all(root: Path) -> None:
    for name, produce in SNAPSHOTS.items():
        (root / SNAPSHOT_DIR / name).write_text(render(produce()))
