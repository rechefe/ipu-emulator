"""ISA constants that are not instructions (E03-12).

The read-only constant registers CR0 and CR1 and the register word widths.
The 16-bit DS and QF halves and the MULT constant layout are in
``ipu_isa.cr_halves`` (ADR 0007).
"""

from __future__ import annotations

from types import MappingProxyType

CR_REGISTER_NAME = "cr"

#: CR0 and CR1 are read-only hardware constants 0 and 1 (0.0 and 1.0 as MULT constants).
CR_ZERO_REG_INDEX = 0
CR_ONE_REG_INDEX = 1

CR_ZERO_REG_VALUE = 0
CR_ONE_REG_VALUE = 1
CR_READ_ONLY_INITIAL_VALUES = MappingProxyType(
    {
        CR_ZERO_REG_INDEX: CR_ZERO_REG_VALUE,
        CR_ONE_REG_INDEX: CR_ONE_REG_VALUE,
    }
)

REGISTER_WORD_BITS = 32
REGISTER_WORD_VALUE_MASK = (1 << REGISTER_WORD_BITS) - 1

LR_CR_SCALAR_BITS = 32
LR_CR_SCALAR_VALUE_MASK = (1 << LR_CR_SCALAR_BITS) - 1
