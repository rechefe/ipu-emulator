"""Opcode definitions for IPU assembler — auto-generated from instruction_spec.

This module builds the Opcode token classes from ipu_isa.instruction_spec
at import. No manual opcode definitions here — everything flows from the
single source of truth (INSTRUCTION_SPEC).

All opcode values are AUTOMATICALLY DERIVED from instruction position.
When you add or reorder instructions in INSTRUCTION_SPEC, opcodes update
automatically without touching this file.
"""

from typing import Dict, Type

import lark

import ipu_as.ipu_token as ipu_token
from ipu_isa.instruction_spec import INSTRUCTION_SPEC
from ipu_isa.slots import SLOTS, Slot


def create_assembler_opcodes() -> Dict[str, Type]:
    """Create Opcode classes at runtime (no code generation needed).

    Returns a dict of dynamically created Opcode subclasses:
        {
            "LoadInstOpcode": <class>,
            "LrInstOpcode": <class>,
            ...
        }

    Each class has an enum_array() classmethod returning instruction names
    in order (position = opcode).
    """
    class Opcode(ipu_token.EnumToken):
        """Base class for all opcode types."""
        @classmethod
        def find_opcode_class(cls, opcode_token) -> Type["Opcode"]:
            tv = opcode_token.value.lower()
            for subclass in cls.__subclasses__():
                if any(tv == name.lower() for name in subclass.enum_array()):
                    return subclass
            raise ValueError(
                f"Opcode '{opcode_token.value}' not found in any Opcode subclass"
            )

    result = {}
    # INSTRUCTION_SPEC order: it fixes the order the token classes are created
    # in, which the SV package's enum order follows.
    for slot_type, instructions in INSTRUCTION_SPEC.items():
        class_name = SLOTS[Slot(slot_type)].asm_opcode_class
        enum_array = list(instructions.keys())

        # Create class dynamically
        opcode_class = type(
            class_name,
            (Opcode,),
            {
                "enum_array": classmethod(lambda cls, ea=enum_array: ea),
            }
        )
        result[class_name] = opcode_class

    return result


# Create Opcode classes at runtime from instruction_spec
_opcode_classes = create_assembler_opcodes()

# Export classes with their original names
LoadInstOpcode = _opcode_classes["LoadInstOpcode"]
StoreInstOpcode = _opcode_classes["StoreInstOpcode"]
AccStoreInstOpcode = _opcode_classes["AccStoreInstOpcode"]
LrInstOpcode = _opcode_classes["LrInstOpcode"]
MultInstOpcode = _opcode_classes["MultInstOpcode"]
AccInstOpcode = _opcode_classes["AccInstOpcode"]
AaqInstOpcode = _opcode_classes["AaqInstOpcode"]
CondInstOpcode = _opcode_classes["CondInstOpcode"]
BreakInstOpcode = _opcode_classes["BreakInstOpcode"]

# Export base Opcode class (parent of all the above)
Opcode = LoadInstOpcode.__bases__[0]


def validate_unique_opcodes() -> None:
    """Validate that all opcodes are unique across all Opcode subclasses.

    NOP is intentionally shared across all slot classes; the assembler resolves
    it to the correct slot by context. All other opcodes must be globally unique.
    """
    SHARED_OPCODES = {"NOP"}
    opcodes_subclasses = Opcode.__subclasses__()
    opcode_to_class = {}

    for cls in opcodes_subclasses:
        for opcode in cls.enum_array():
            if opcode in SHARED_OPCODES:
                continue
            if opcode in opcode_to_class:
                existing_class = opcode_to_class[opcode]
                raise AssertionError(
                    f"Duplicate opcode '{opcode}' found in classes "
                    f"'{existing_class.__name__}' and '{cls.__name__}'"
                )
            opcode_to_class[opcode] = cls


# Validate on module load
validate_unique_opcodes()
