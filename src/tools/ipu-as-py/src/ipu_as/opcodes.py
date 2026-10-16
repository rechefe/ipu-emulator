"""Opcode definitions for IPU assembler — auto-generated from the ISA view.

This module builds the Opcode token classes from ``ipu_isa.view.isa()`` at
import. No manual opcode definitions here — everything flows from the single
source of truth (``instruction_spec.py``).

All opcode values are AUTOMATICALLY DERIVED from instruction position.
When you add or reorder instructions in the spec, opcodes update
automatically without touching this file.
"""

import ipu_as.ipu_token as ipu_token
from ipu_as.errors import SlotError, SourceLocation
from ipu_isa.slots import SLOTS
from ipu_isa.view import isa


def create_assembler_opcodes() -> dict[str, type]:
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

        #: Lower-case opcode -> the Opcode subclass that has it, built on first use.
        _class_by_opcode: dict[str, type["Opcode"]] | None = None

        @classmethod
        def find_opcode_class(cls, opcode_token) -> type["Opcode"]:
            if Opcode._class_by_opcode is None:
                # In subclass order: the shared NOP maps to the first slot's class,
                # as the scan this replaces did.
                table: dict[str, type[Opcode]] = {}
                for subclass in Opcode.__subclasses__():
                    for name in subclass.enum_array():
                        table.setdefault(name.lower(), subclass)
                Opcode._class_by_opcode = table
            subclass = Opcode._class_by_opcode.get(opcode_token.value.lower())
            if subclass is not None:
                return subclass
            raise SlotError(
                f"Opcode '{opcode_token.value}' not found in any Opcode subclass",
                location=SourceLocation.of(opcode_token),
                token=opcode_token.value,
            )

    result = {}
    # Spec order: it fixes the order the token classes are created in, which
    # the SV package's enum order follows.
    for slot in isa().spec_slot_order:
        class_name = SLOTS[slot].asm_opcode_class
        enum_array = [instruction.mnemonic for instruction in isa().instructions(slot)]

        # Create class dynamically
        opcode_class = type(
            class_name,
            (Opcode,),
            {
                "enum_array": classmethod(lambda cls, ea=enum_array: ea),
            },
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
