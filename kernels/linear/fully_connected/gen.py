"""fully_connected's case generators (E07-22), beside the harness's default one.

The default case takes the default generator (E06-08): uniform values in each
input's per-format ``range`` (``kernel.toml``). These two give structure the
default one does not:

* ``extremes``: every element at an end of its input's range, with random
  signs, so each sum is as large as the ranges allow. The ranges are chosen
  so that sum stays inside the accumulator (no saturation): the case proves
  they are.
* ``every_code``: for E4 and E5, each element a code of the format drawn
  uniformly among those inside the input's range, so the small exponents and
  the subnormals (rare under a uniform draw of values) are in every row and
  their products reach the FP32 accumulator.

Both draw from ``rng``, the case's seeded stream, so the data are the same in
every process.
"""

import numpy as np

from ipu_isa.formats import FORMATS, element_values

#: The inputs every generator makes.
INPUTS = ("x", "w")


def _bounds(plan, name):
    """Input ``name``'s ``range`` in the case's format (``kernel.toml``
    declares one for every format)."""
    return plan.manifest.tensors[name].value_range[plan.format]


def extremes(plan, rng):
    """``x`` and ``w`` with every element ``lo`` or ``hi`` of its range."""
    out = {}
    for name in INPUTS:
        lo, hi = _bounds(plan, name)
        out[name] = np.where(rng.random(plan.shapes[name]) < 0.5, lo, hi).astype(np.float64)
    return out


def every_code(plan, rng):
    """``x`` and ``w`` drawn uniformly over the format's finite codes in range.

    Raises:
        ValueError: The format is not an 8-bit float format.
    """
    if not FORMATS[plan.format].is_float:
        raise ValueError(
            f"every_code draws codes of an 8-bit float format, not {plan.format.value}"
        )
    table = np.array(element_values(plan.format), dtype=np.float64)
    out = {}
    for name in INPUTS:
        lo, hi = _bounds(plan, name)
        values = np.unique(table[np.isfinite(table) & (table >= lo) & (table <= hi)])
        out[name] = rng.choice(values, size=plan.shapes[name])
    return out
