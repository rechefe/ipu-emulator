"""Master register definitions for IPU — single source of truth.

This module defines all IPU registers in one place, with complete metadata
used by both assembler and emulator:

- `REGISTER_DEFINITIONS`: Master dict with all register metadata
- `create_regfile_schema()`: Generate RegDescriptor list for emulator
- `create_assembler_reg_classes()`: Generate EnumToken subclasses for assembler

The key insight: Define each register once, let the factory functions
adapt it for each package. This eliminates duplication and ensures
consistency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipu_common.types import RegDtype, RegKind, RegDescriptor

if TYPE_CHECKING:
    pass


# ===========================================================================
# MASTER REGISTER DEFINITIONS
# ===========================================================================
#
# Each register is defined with complete metadata that serves both packages:
#
# - `bit_width`: Bits needed to encode the register (for assembler)
# - `encoding_suffix`: Optional suffix for enum class names
# - Rest: RegDescriptor metadata for emulator
#
# ===========================================================================

REGISTER_DEFINITIONS = {
    # -----------------------------------------------------------------------
    # Multiply-stage registers
    # -----------------------------------------------------------------------
    "r": {
        "kind": RegKind.MULT,
        "vector": True,
        "size_bytes": 128,
        "count": 2,
        "dtype": RegDtype.UINT8,
        "debug_aliases": ("r0", "r1"),
        "assembler_values": ["r0", "r1", "mem_bypass"],
        "encoding_class": "MultStageRegField",
    },
    "r_cyclic": {
        "kind": RegKind.MULT,
        "vector": True,
        "cyclic": True,
        "size_bytes": 512,
        "count": 1,
        "dtype": RegDtype.UINT8,
        "debug_aliases": ("rcyclic",),
    },
    "r_mask": {
        "kind": RegKind.MULT,
        "vector": True,
        "size_bytes": 128,
        "count": 1,
        "dtype": RegDtype.UINT128,
        "debug_aliases": ("rmask",),
    },
    # -----------------------------------------------------------------------
    # Accumulator-stage registers
    # -----------------------------------------------------------------------
    "r_acc": {
        "kind": RegKind.ACC,
        "vector": True,
        "word_view": True,
        "size_bytes": 512,
        "count": 1,
        "dtype": RegDtype.UINT8,
        "debug_aliases": ("acc",),
    },
    # -----------------------------------------------------------------------
    # Activation & Quantization (AAQ) stage — 4 × 32-bit general-purpose regs
    # -----------------------------------------------------------------------
    "aaq": {
        "kind": RegKind.AAQ,
        "vector": False,
        "size_bytes": 4,
        "count": 4,
        "dtype": RegDtype.UINT32,
        "assembler_values": [f"aaq{i}" for i in range(4)],
        "encoding_class": "AaqRegField",
    },
    # -----------------------------------------------------------------------
    # LR / CR scalar registers
    # -----------------------------------------------------------------------
    "lr": {
        "kind": RegKind.LR,
        "vector": False,
        "size_bytes": 4,
        "count": 16,
        "dtype": RegDtype.UINT32,
        "assembler_values": [f"lr{i}" for i in range(16)],
        "encoding_class": "LrRegField",
    },
    "cr": {
        "kind": RegKind.CR,
        "vector": False,
        "size_bytes": 4,
        "count": 16,
        "dtype": RegDtype.UINT32,
        "assembler_values": [f"cr{i}" for i in range(16)],
        "encoding_class": "CrRegField",
    },
    # -----------------------------------------------------------------------
    # Miscellaneous / forwarding registers
    # -----------------------------------------------------------------------
    "mult_res": {
        "kind": RegKind.MISC,
        "vector": True,
        "word_view": True,
        "size_bytes": 512,
        "count": 1,
        "dtype": RegDtype.UINT8,
    },
    "mem_bypass": {
        "kind": RegKind.MISC,
        "vector": True,
        "size_bytes": 128,
        "count": 1,
        "dtype": RegDtype.UINT8,
        # Note: mem_bypass is included in MultStageRegField, not a separate register
    },
}


# ===========================================================================
# Factory Functions
# ===========================================================================


def get_register_sizes() -> dict[str, dict[str, int | bool]]:
    """Return metadata for each register from master definitions.

    Returns a dict keyed by register name with the keys:
        size_bytes: size of one element in bytes
        count:      number of elements
        vector:     True for byte-blob registers, False for scalar integers
        cyclic:     True if the register wraps around
        word_view:  True if the register supports uint32 word-level access

    This is the single source of truth for register dimensions used by
    both assembler and emulator.
    """
    result = {}
    for name, meta in REGISTER_DEFINITIONS.items():
        result[name] = {
            "size_bytes": meta["size_bytes"],
            "count": meta.get("count", 1),
            "vector": meta["vector"],
            "cyclic": meta.get("cyclic", False),
            "word_view": meta.get("word_view", False),
        }
    return result


def get_mult_stage_map() -> list[tuple[str, int]]:
    """Return the MultStageRegField encoding as a list of (register, element_index).

    Index in the returned list == the encoded integer value in the VLIW word.
    Each entry maps to a (canonical_register_name, element_index) pair in the
    register file.

    Derived from ``REGISTER_DEFINITIONS["r"]["assembler_values"]``.

    Example::

        [("r", 0), ("r", 1), ("mem_bypass", 0)]
        #  0→r0     1→r1     2→mem_bypass
    """
    r_def = REGISTER_DEFINITIONS["r"]
    aliases = r_def["debug_aliases"]  # ("r0", "r1")
    result: list[tuple[str, int]] = []
    for val in r_def["assembler_values"]:
        if val in aliases:
            # e.g. "r0" → ("r", 0), "r1" → ("r", 1)
            idx = aliases.index(val)
            result.append(("r", idx))
        elif val in REGISTER_DEFINITIONS:
            # e.g. "mem_bypass" is its own register
            result.append((val, 0))
        else:
            raise ValueError(f"Unknown mult-stage value: {val}")
    return result


def create_regfile_schema() -> list[RegDescriptor]:
    """Generate the REGFILE_SCHEMA for the emulator from master definitions.
    
    Returns a list of RegDescriptors in the correct order for initializing
    the emulator's register file.
    """
    schema = []
    
    for name, metadata in REGISTER_DEFINITIONS.items():
        # Skip fields that aren't for RegDescriptor
        descriptor_fields = {
            k: v for k, v in metadata.items()
            if k in {
                "kind", "size_bytes", "count", "dtype",
                "cyclic", "word_view", "debug_aliases"
            }
        }
        
        schema.append(RegDescriptor(
            name=name,
            **descriptor_fields
        ))
    
    return schema


def create_assembler_reg_enums() -> dict[str, list[str]]:
    """Generate assembler register enum definitions from master definitions.
    
    Returns a dict mapping encoding_class names to their enum values.
    Used to generate EnumToken subclasses in ipu-as-py.
    
    Example:
        {
            "MultStageRegField": ["r0", "r1", "mem_bypass"],
            "LrRegField": ["lr0", "lr1", ..., "lr15"],
            "CrRegField": ["cr0", "cr1", ..., "cr15"],
            "LcrRegField": ["lr0", ..., "lr15", "cr0", ..., "cr15"],
        }
    """
    enums = {}
    
    # First pass: collect values for each encoding class
    for name, metadata in REGISTER_DEFINITIONS.items():
        if "encoding_class" not in metadata:
            continue
        
        enc_class = metadata["encoding_class"]
        values = metadata.get("assembler_values", [])
        
        if enc_class not in enums:
            enums[enc_class] = []
        
        enums[enc_class].extend(values)
    
    # Special case: LcrRegField combines LR and CR
    if "LrRegField" in enums and "CrRegField" in enums:
        enums["LcrRegField"] = enums["LrRegField"] + enums["CrRegField"]
    
    return enums


# ===========================================================================
# Dynamic EnumToken Class Generation
# ===========================================================================

def create_assembler_reg_classes(ipu_token_module=None):
    """Dynamically generate EnumToken subclasses from master definitions.
    
    This function creates Python classes that inherit from EnumToken, each with
    their own enum_array() implementation based on the master register definitions.
    
    Args:
        ipu_token_module: The ipu_as.ipu_token module (imported for base class).
                         If None, attempts to import it automatically.
    
    Returns:
        dict[str, type]: Mapping of class names to generated EnumToken subclasses.
        
    Example:
        classes = create_assembler_reg_classes()
        lr_field_class = classes["LrRegField"]
        # Now use lr_field_class as a normal class
    
    When used in ipu-as-py:
        classes = create_assembler_reg_classes(ipu_as.ipu_token)
        # Add to ipu_as namespace:
        for class_name, cls in classes.items():
            globals()[class_name] = cls
    """
    if ipu_token_module is None:
        try:
            from ipu_as import ipu_token as ipu_token_module
        except ImportError:
            raise ImportError(
                "Please pass ipu_token module or import ipu_as first"
            )
    
    enums = create_assembler_reg_enums()
    generated_classes = {}
    
    for class_name, enum_values in enums.items():
        # Create a new EnumToken subclass dynamically
        def make_class(name, values):
            class GeneratedRegField(ipu_token_module.EnumToken):
                @classmethod
                def enum_array(cls):
                    return values
            
            GeneratedRegField.__name__ = name
            GeneratedRegField.__qualname__ = name
            return GeneratedRegField
        
        generated_classes[class_name] = make_class(class_name, enum_values)
    
    return generated_classes


# ===========================================================================
# Validation
# ===========================================================================

def validate_register_definitions() -> None:
    """Validate that master definitions are consistent.
    
    Checks:
    - RegDescriptor creation succeeds for all definitions
    - Assembler enum values don't conflict
    - Required fields are present
    
    Raises ValueError if validation fails.
    """
    try:
        schema = create_regfile_schema()
    except Exception as e:
        raise ValueError(f"Failed to create regfile schema: {e}")
    
    if not schema:
        raise ValueError("Register schema is empty!")
    
    try:
        enums = create_assembler_reg_enums()
    except Exception as e:
        raise ValueError(f"Failed to create assembler enums: {e}")
    
    # Check for duplicate enum values across different classes
    # (except LcrRegField which is a composite of LR and CR)
    all_values = {}
    for enc_class, values in enums.items():
        # Skip checking LcrRegField as it's a composite
        if enc_class == "LcrRegField":
            continue
        
        for value in values:
            if value in all_values:
                raise ValueError(
                    f"Register value '{value}' defined in both "
                    f"'{all_values[value]}' and '{enc_class}'"
                )
            all_values[value] = enc_class
    
    # Validate count in LcrRegField
    if "LcrRegField" in enums:
        expected_lcr = len(enums.get("LrRegField", [])) + len(enums.get("CrRegField", []))
        actual_lcr = len(enums["LcrRegField"])
        if expected_lcr != actual_lcr:
            raise ValueError(
                f"LcrRegField expected {expected_lcr} values, got {actual_lcr}"
            )


# Run validation on module import
validate_register_definitions()
