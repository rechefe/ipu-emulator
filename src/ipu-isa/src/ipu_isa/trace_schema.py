"""The commit record and trace contract: versioned JSON Schemas (E04-29).

``schemas/commit_record.v1.json`` describes one retired word and
``schemas/trace.v1.json`` a trace's header line; a trace is JSON Lines (the
header, then one record per line). Each schema's ``$id`` ends in its semantic
version (Q11): an optional field added is a minor bump, a field removed or
renamed or a meaning changed is a major one. The schemas hold no
Python-specific type, so a cocotb testbench or another model reads the same
files.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from functools import cache
from importlib import resources
from typing import Any, Literal

from ipu_isa.codec.word import word_bytes
from ipu_isa.registers import REGISTER_DEFINITIONS
from ipu_isa.view import isa

__all__ = ["SCHEMA_VERSION", "isa_digest", "load_schema", "program_digest", "schema_version"]

SchemaName = Literal["commit_record", "trace"]


@cache
def load_schema(name: SchemaName) -> dict[str, Any]:
    """The schema *name* (``commit_record`` or ``trace``), parsed."""
    text = resources.files("ipu_isa").joinpath(f"schemas/{name}.v1.json").read_text("utf-8")
    schema: dict[str, Any] = json.loads(text)
    return schema


def schema_version(name: SchemaName) -> str:
    """The semantic version at the end of the schema's ``$id``."""
    return str(load_schema(name)["$id"]).rsplit("/", 1)[1]


#: The commit record schema's version, written in every trace header.
SCHEMA_VERSION = schema_version("commit_record")


def program_digest(words: Iterable[int]) -> str:
    """SHA-256 of a program's emulator words, each little-endian in
    ``word_bytes()`` bytes: what any reader of the program file can recompute
    (``trace.v1.json``'s ``program_digest``)."""
    width = word_bytes()
    digest = hashlib.sha256()
    for word in words:
        digest.update(word.to_bytes(width, "little"))
    return digest.hexdigest()


@cache
def isa_digest() -> str:
    """SHA-256 of the ISA definition: every instruction (slot, opcode,
    mnemonic, tag, operands with their types and widths) and every register.
    A trace records it, so a reader knows which ISA the words decode with."""
    view = isa()
    instructions = [
        {
            "slot": i.slot.value,
            "opcode": i.opcode,
            "mnemonic": i.mnemonic,
            "tag": i.tag.value,
            "operands": [
                [op.name, op.type.name, view.layout.operand_bits[op.type.name]] for op in i.operands
            ],
        }
        for slot in view.slots
        for i in view.instructions(slot)
    ]
    registers = {
        name: {
            "kind": d["kind"].value,
            "count": d["count"],
            "size_bytes": d["size_bytes"],
            "vector": d["vector"],
        }
        for name, d in REGISTER_DEFINITIONS.items()
    }
    text = json.dumps({"instructions": instructions, "registers": registers}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()
