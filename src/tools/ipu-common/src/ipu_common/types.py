"""Shared types for IPU register file, instruction descriptors, and data formats.

This module defines the foundational types used across all IPU Python packages.
It serves as the single source of truth for register metadata, data types, and
register kinds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RegDtype(Enum):
    """Data type for register storage and debug display.
    
    Each register stores bytes that are interpreted as one of these types.
    UINT128 is used for mask registers that store 128-bit patterns.
    """
    UINT8 = "uint8"
    INT8 = "int8"
    UINT32 = "uint32"
    INT32 = "int32"
    UINT128 = "uint128"  # mask register


class RegKind(Enum):
    """Which pipeline stage / functional group the register belongs to.
    
    MULT:  Multiply-stage registers (r0, r1, r_cyclic, r_mask)
    ACC:   Accumulator-stage registers (r_acc, mult_res)
    LR:    Long-register file (lr0-lr15)
    CR:    Control-register file (cr0-cr15)
    MISC:  Miscellaneous registers
    """
    MULT = "mult"
    ACC = "acc"
    LR = "lr"
    CR = "cr"
    MISC = "misc"


@dataclass(frozen=True)
class RegDescriptor:
    """Declarative description of a single register (or register array).
    
    This type defines the metadata for a register needed by both the assembler
    (for validation) and emulator (for storage and access). Each register in the
    IPU register file has exactly one RegDescriptor.
    
    Attributes:
        name:           Human-readable name (e.g., "r0", "lr", "cr1").
                       Used in assembly syntax and debug CLI.
        kind:           Pipeline stage this register belongs to (RegKind enum).
        size_bytes:     Size of one register element in bytes.
                       E.g., 4 for lr0, 128 for r0.
        count:          Number of elements in this register.
                       E.g., count=2 for r (two 128-byte registers r0, r1),
                       count=16 for lr (16 32-bit registers lr0-lr15).
        dtype:          Element data type for storage and debug display (RegDtype).
        cyclic:         If True, index wraps around (e.g., r_cyclic is cyclic).
        word_view:      If True, also expose a uint32 word view (for r_acc, mult_res).
        debug_aliases:  Extra names accepted by debug CLI (e.g., "r0" for r[0]).
    """
    name: str
    kind: RegKind
    size_bytes: int
    count: int = 1
    dtype: RegDtype = RegDtype.UINT8
    cyclic: bool = False
    word_view: bool = False
    debug_aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_vector(self) -> bool:
        """True for byte-blob registers, False for scalar integer registers.

        Scalar registers (LR, CR) store integer values accessed by index.
        Vector registers (R, R_ACC, etc.) store byte blobs.
        """
        return self.dtype not in (RegDtype.UINT32, RegDtype.INT32)
