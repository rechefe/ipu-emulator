"""INC/DEC immediate operand — bit width derived from the LR slot union layout.

The field width is not hardcoded here. The union solver packs the
``LrIncDecImmediate`` operand into a shared LR field (same bin as ``LcrIdx``
for ``ADD``/``SUB`` ``src_b`` and ``INCR_MOD_POW2`` ``step``), and
:func:`lr_inc_dec_imm_bits` reads that field's width from
:func:`ipu_isa.layout.isa_layout`.

Valid immediates are unsigned: ``0`` .. ``lr_inc_dec_imm_max()``.
"""

from __future__ import annotations

from ipu_isa.layout import isa_layout


def lr_inc_dec_imm_bits() -> int:
    """Width of the INC/DEC immediate field."""
    return isa_layout().operand_bits["LrIncDecImmediate"]


def lr_inc_dec_imm_max() -> int:
    """Return the maximum unsigned immediate value (inclusive)."""
    return (1 << lr_inc_dec_imm_bits()) - 1
