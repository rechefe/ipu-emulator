"""The IPU instruction set, the single source of truth: instructions, slots,
operand types, registers, element formats, the timing contract and the binary codec.
"""

from ipu_isa.mnemonics import RESERVED_MNEMONICS
from ipu_isa.types import RegDescriptor, RegDtype, RegKind

__all__ = ["RESERVED_MNEMONICS", "RegDtype", "RegKind", "RegDescriptor"]
