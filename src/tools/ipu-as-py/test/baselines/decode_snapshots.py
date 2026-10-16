"""Snapshots of the decoded-word names the assembler and emulator agree on (E02-02).

Three things are wire format today but are derived from Python class names:

* ``word_fields.json``: ``CompoundInst.get_fields()``, the name and width of
  every field of an instruction word, LSB first, plus the total width.
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


def _slot_instance(key: tuple) -> str:
    """``("lr", "ADD", 1)`` -> ``"lr[1]"``; ``("mult", "MULT.VE")`` -> ``"mult"``."""
    return f"{key[0]}[{key[2]}]" if len(key) == 3 else key[0]


def emulator_field_maps() -> dict:
    # The emulator keeps these tables private; pinning them is the point.
    field_maps: dict[str, dict[str, dict[str, str]]] = {}
    for key, operand_fields in ipu._INSTRUCTION_FIELD_MAP.items():
        field_maps.setdefault(_slot_instance(key), {})[key[1]] = dict(operand_fields)
    opcode_fields = dict(ipu._SLOT_OPCODE_FIELDS)
    for slot_idx, field in enumerate(ipu._LR_OPCODE_FIELDS):
        opcode_fields[f"lr[{slot_idx}]"] = field
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
