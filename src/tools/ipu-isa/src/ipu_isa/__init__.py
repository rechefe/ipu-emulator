"""The IPU instruction set: instructions, registers, operand types and slots."""

from ipu_isa.mnemonics import RESERVED_MNEMONICS
from ipu_isa.types import RegDescriptor, RegDtype, RegKind

__all__ = ["RESERVED_MNEMONICS", "RegDtype", "RegKind", "RegDescriptor"]
