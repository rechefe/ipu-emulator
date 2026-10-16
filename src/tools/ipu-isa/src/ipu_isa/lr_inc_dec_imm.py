"""INC/DEC immediate operand — bit width derived from the LR slot union layout.

The field width is not hardcoded here.  After ``compute_slot_layouts`` runs, the
union solver writes ``LR_INC_DEC_IMM_FIELD_BITS`` from the shared union field
that carries the ``LrIncDecImmediate`` operand (same bin as ``LcrIdx`` for
``ADD``/``SUB`` ``src_b`` and ``INCR_MOD_POW2`` ``step``).

Valid immediates are unsigned: ``0`` .. ``(1 << LR_INC_DEC_IMM_FIELD_BITS) - 1``.
"""

LR_INC_DEC_IMM_FIELD_BITS: int = 0


def lr_inc_dec_imm_max() -> int:
    """Return the maximum unsigned immediate value (inclusive)."""
    if LR_INC_DEC_IMM_FIELD_BITS <= 0:
        raise RuntimeError(
            "LR_INC_DEC_IMM_FIELD_BITS is unset; union layout must be computed first"
        )
    return (1 << LR_INC_DEC_IMM_FIELD_BITS) - 1
