"""Single source of truth for acc.agg operand enums.

Used by:
- ipu-as: enum names for assembly syntax (AggModeField, PostFnField)
- ipu-emu: interpretation of encoded values in execute_acc_agg

Encoding:
- agg_mode: 0 = SUM, 1 = MAX
- post_fn: 0 = value (identity), 1 = value_cr (VALUE*CR), 2 = inv (1/VALUE), 3 = inv_sqrt (1/SQRT(VALUE))
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Aggregation mode (acc.agg first operand)
# ---------------------------------------------------------------------------

AGG_MODE_NAMES: tuple[str, ...] = ("sum", "max")
AGG_MODE_SUM = 0
AGG_MODE_MAX = 1


def get_agg_mode(encoded: int) -> int:
    """Return aggregation mode: 0 = SUM, 1 = MAX."""
    return encoded & 1


def is_agg_sum(encoded: int) -> bool:
    return get_agg_mode(encoded) == AGG_MODE_SUM


def is_agg_max(encoded: int) -> bool:
    return get_agg_mode(encoded) == AGG_MODE_MAX


# ---------------------------------------------------------------------------
# Post function (acc.agg second operand)
# ---------------------------------------------------------------------------

POST_FN_NAMES: tuple[str, ...] = (
    "value",      # 0: identity VALUE
    "value_cr",   # 1: VALUE * CR[cr_idx]
    "inv",        # 2: 1/VALUE
    "inv_sqrt",   # 3: 1/SQRT(VALUE)
)

POST_FN_VALUE = 0
POST_FN_VALUE_CR = 1
POST_FN_INV = 2
POST_FN_INV_SQRT = 3


def get_post_fn(encoded: int) -> int:
    """Return post function index 0..3."""
    return encoded & 3
