"""Named tolerance tiers: how close a kernel's output must be, and why (E06-10).

A kernel picks a tier per format in ``[golden] tolerance``, by name, or
overrides one with its own numbers and a required ``why``
(``{ tier = "FP32_ORDER", rtol = 3e-5, why = "measured on case X" }``), so no
tolerance literal goes unexplained. The values are provisional: the E07 pilot
(E07-03) measures them in FP32, INT8 and E4 and records the numbers here.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["DEFAULT_TIER", "TIERS", "Tier"]


class Tier(NamedTuple):
    """One tolerance.

    Attributes:
        name: The tier's name (a kernel's override keeps the tier it starts from).
        rtol: Relative tolerance.
        atol: Absolute tolerance, in output units (LSBs for INT8).
        ulps: For an 8-bit float format, the allowed distance in codes; else None.
        why: What the tier is for, or why a kernel overrides it.
    """

    name: str
    rtol: float
    atol: float
    ulps: int | None
    why: str


TIERS: dict[str, Tier] = {
    t.name: t
    for t in (
        Tier("EXACT", 0, 0, 0, "data movement or a single correctly rounded op"),
        Tier("FP32_ORDER", 1e-5, 1e-6, None, "FP32 accumulation order differs from the reference"),
        Tier(
            "FP32_LONG", 1e-4, 1e-5, None, "long reductions (K >= 256) or reciprocal/rsqrt chains"
        ),
        Tier(
            "TRANSCENDENT", 1e-3, 1e-4, None, "libm vs future AaQ LUT headroom (Q25: provisional)"
        ),
        Tier("INT8_LSB", 0, 1, None, "one LSB after round-to-nearest"),
        Tier("FP8_ULP", 0, 0, 1, "one code step in the 8-bit float format"),
    )
}

#: The tier of a format ``[golden] tolerance`` does not name: the strictest.
DEFAULT_TIER = TIERS["EXACT"]
