"""Named tolerance tiers: how close a kernel's output must be, and why (E06-10).

A kernel picks a tier per variant in ``[golden] tolerance``, by name, or
overrides one with its own numbers and a required ``why``
(``{ tier = "FP32_ORDER", rtol = 3e-5, why = "measured on case X" }``), so no
tolerance literal goes unexplained.

What the projections pilot (E07-03, the 8 ``proj_p4`` entries, K = 192 to
480) measured against ``ops.linear`` rounded to the output format:
``FP32_LONG``: FP32, largest error 8.6e-6 on the template's default case
(K = 192, SiLU); the entries pass it too.

ADR 0007 removed ``INT8_LSB`` and ``FP8_ULP`` with their formats and added
``NX8_STEP``: an NX8 output may be one step of its row's grid away from the
golden (the golden is quantized the same way, row by row, but a row's scale
follows its largest value, which the kernel's accumulation order can move).
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["DEFAULT_TIER", "TIERS", "Tier"]


class Tier(NamedTuple):
    """One tolerance.

    Attributes:
        name: The tier's name (a kernel's override keeps the tier it starts from).
        rtol: Relative tolerance.
        atol: Absolute tolerance, in output units.
        ulps: In NX8, the allowed distance in steps of the golden row's grid
            (``NX8_STEP``); else None.
        why: What the tier is for, or why a kernel overrides it.
    """

    name: str
    rtol: float
    atol: float
    ulps: int | None
    why: str

    @property
    def counts_steps(self) -> bool:
        """True when the tier allows NX8 steps (``ulps`` of 1 or more); EXACT's
        0 means no difference at all, in either mode."""
        return bool(self.ulps)


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
        Tier("NX8_STEP", 0, 0, 1, "one step of the NX8 row's grid (ADR 0007)"),
    )
}

#: The tier of a variant ``[golden] tolerance`` does not name: the strictest.
DEFAULT_TIER = TIERS["EXACT"]
