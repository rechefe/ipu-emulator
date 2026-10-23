"""fully_connected's case generators (E07-22), beside the harness's default one.

The default case takes the default generator (E06-08): uniform values in each
input's ``range`` (``kernel.toml``), which the harness then holds in the case's
NX8 variant row by row (ADR 0007). These two give structure the default one
does not:

* ``extremes``: every element at an end of its input's range, with random
  signs, so each sum is as large as the ranges allow (and every row's scale is
  the range's).
* ``every_code``: for a static NX8 variant (E4, E5), each XMEM row drawn
  uniformly among the codes of one row grid inside the input's range, so the
  small exponents and the clamp-exponent (subnormal) codes, rare under a
  uniform draw of values, are in every row and their products reach the FP32
  accumulator. Each row has one element at the grid's top exponent, so the
  row's scale is the grid's and the harness holds every value unchanged.

Both draw from ``rng``, the case's seeded stream, so the data are the same in
every process.
"""

import numpy as np

from ipu_isa.geometry import LANES
from ipu_isa.nx8 import SCALE_BIAS, decode_row

#: The inputs every generator makes.
INPUTS = ("x", "w")

#: Every byte: the codes of an NX8 row.
_CODES = bytes(range(256))


def _bounds(plan, name):
    """Input ``name``'s ``range`` in the case's variant (``kernel.toml``
    declares one for NX8)."""
    return plan.manifest.tensors[name].value_range.get(plan.format)


def extremes(plan, rng):
    """``x`` and ``w`` with every element ``lo`` or ``hi`` of its range."""
    out = {}
    for name in INPUTS:
        lo, hi = _bounds(plan, name)
        out[name] = np.where(rng.random(plan.shapes[name]) < 0.5, lo, hi).astype(np.float64)
    return out


def _grid(fmt, lo, hi):
    """The finite values of the highest-scale row grid of ``fmt`` inside
    ``[lo, hi]``, and its scale."""
    for scale in range(SCALE_BIAS + 16, 0, -1):
        values = np.asarray(decode_row(_CODES, fmt, scale), dtype=np.float64)
        if values.min() >= lo and values.max() <= hi:
            return np.unique(values), scale
    raise ValueError(f"no row grid of {fmt} fits [{lo}, {hi}]")


def _rows(rng, grid, top, count, width):
    """``count`` rows of ``LANES`` lanes: the first ``width`` drawn from
    ``grid``, one of them from ``top`` (the grid's top exponent), the rest 0."""
    rows = np.zeros((count, LANES), dtype=np.float64)
    rows[:, :width] = rng.choice(grid, size=(count, width))
    rows[np.arange(count), rng.integers(0, width, size=count)] = rng.choice(top, size=count)
    return rows


def every_code(plan, rng):
    """``x`` and ``w``, each XMEM row drawn uniformly over one row grid's codes in range.

    ``x``'s rows are its samples; ``w``'s are its columns (``pack =
    "transposed"``: row ``i`` is input ``i``'s 64 weights, then zeros).

    Raises:
        ValueError: The variant is FP32 or dynamic (no fixed row grid).
    """
    fmt = plan.format.qf
    if fmt is None or fmt.dynamic:
        raise ValueError(f"every_code draws the codes of a static NX8 variant, not {plan.format}")
    out = {}
    for name in INPUTS:
        lo, hi = _bounds(plan, name)
        grid, scale = _grid(fmt, lo, hi)
        top = grid[np.abs(grid) >= 2.0 ** (scale - SCALE_BIAS)]
        if name == "x":
            samples, features = plan.shapes["x"]
            out["x"] = _rows(rng, grid, top, samples, features)[:, :features]
        else:
            n_out, features = plan.shapes["w"]
            out["w"] = _rows(rng, grid, top, features, n_out)[:, :n_out].T.copy()
    return out
