"""Opcode definitions for IPU assembler — auto-generated from instruction_spec.

This module re-exports Opcode classes created from ipu_common.instruction_spec
at runtime. No manual opcode definitions here — everything flows from the
single source of truth (INSTRUCTION_SPEC).

All opcode values are AUTOMATICALLY DERIVED from instruction position.
When you add or reorder instructions in INSTRUCTION_SPEC, opcodes update
automatically without touching this file.
"""

import lark
from ipu_common.instruction_spec import create_assembler_opcodes

# Create Opcode classes at runtime from instruction_spec
_opcode_classes = create_assembler_opcodes()

# Export classes with their original names
XmemInstOpcode = _opcode_classes["XmemInstOpcode"]
LrInstOpcode = _opcode_classes["LrInstOpcode"]
MultInstOpcode = _opcode_classes["MultInstOpcode"]
AccInstOpcode = _opcode_classes["AccInstOpcode"]
AaqInstOpcode = _opcode_classes["AaqInstOpcode"]
CondInstOpcode = _opcode_classes["CondInstOpcode"]
BreakInstOpcode = _opcode_classes["BreakInstOpcode"]

# Export base Opcode class (parent of all the above)
Opcode = XmemInstOpcode.__bases__[0]


def validate_unique_opcodes() -> None:
    """Validate that all opcodes are unique across all Opcode subclasses.
    
    This is a sanity check to ensure INSTRUCTION_SPEC has no duplicates.
    """
    opcodes_subclasses = Opcode.__subclasses__()
    opcode_to_class = {}

    for cls in opcodes_subclasses:
        for opcode in cls.enum_array():
            if opcode in opcode_to_class:
                existing_class = opcode_to_class[opcode]
                raise AssertionError(
                    f"Duplicate opcode '{opcode}' found in classes "
                    f"'{existing_class.__name__}' and '{cls.__name__}'"
                )
            opcode_to_class[opcode] = cls


# Validate on module load
validate_unique_opcodes()
