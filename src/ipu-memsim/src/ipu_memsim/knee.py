"""The knee: the smallest XMEM at which more memory stops helping.

Given a sweep over ``num_banks`` (``ipu_memsim.experiment.run_sweep``),
:func:`knees` reports two knees per workload group:

- the **threshold knee**: the smallest bank count from which stall stays at
  or under ``threshold_percent`` of the ideal cycles (default 10) for every
  larger bank count swept. It does not exist when stall is above the
  threshold even with the most banks: then the stall is the cost of moving
  the data once, which no amount of XMEM removes.
- the **plateau knee**: the smallest bank count from which stall stays
  within ``plateau_percent`` (default 1) of the stall at the largest bank
  count swept. It always exists and says where adding memory stops helping.

Both are given in banks, in rows (banks x rows per bank) and in MB (128
bytes per row). Example: stall of 900%, 40%, 4.3%, 4.3%, 4.3% at 3, 4, 5, 8
and 16 banks of 1024 rows: both knees are 5 banks = 5120 rows = 0.625 MB.
"""

from __future__ import annotations

from typing import Any

from ipu_memsim.experiment import SweepTable
from ipu_memsim.params import ROW_BYTES

__all__ = ["knees"]


def _first_from_which(values: list[tuple[int, float | None]], good: Any) -> int | None:
    """The smallest bank count from which every swept point is ``good``."""
    knee = None
    for banks, stall in sorted(values, reverse=True):  # from the most banks down
        if stall is None or not good(stall):
            break
        knee = banks
    return knee


def knees(
    table: SweepTable,
    bank_rows: int = 1024,
    threshold_percent: float = 10.0,
    plateau_percent: float = 1.0,
) -> list[dict[str, Any]]:
    """Per group of a ``num_banks`` sweep, its two knees (see the module).

    Args:
        table: A sweep whose (only) swept parameter is ``num_banks``.
        bank_rows: Rows per bank in that sweep, to convert banks to rows and MB.
        threshold_percent: The threshold knee's level.
        plateau_percent: How close to the final stall counts as "on the
            plateau", in percent of that stall (and at least 0.01 points).

    Raises:
        ValueError: The table does not sweep ``num_banks`` alone.
    """
    if [p.name for p in table.params] != ["num_banks"]:
        raise ValueError("the knee finder needs a sweep of num_banks alone")
    result = []
    for group in table.groups:
        points = [
            (row["num_banks"], row["stall_percent"]) for row in table.rows if row["group"] == group
        ]
        most_banks, final = max(points)
        entry: dict[str, Any] = {
            "group": group,
            "stall_at_most_banks": final,
            "threshold_percent": threshold_percent,
        }
        margin = None if final is None else max(0.01, final * plateau_percent / 100.0)
        found = {
            "threshold": _first_from_which(points, lambda stall: stall <= threshold_percent),
            "plateau": None
            if final is None
            else _first_from_which(points, lambda stall: stall <= final + margin),
        }
        for name, banks in found.items():
            entry[f"{name}_knee_banks"] = banks
            entry[f"{name}_knee_rows"] = None if banks is None else banks * bank_rows
            entry[f"{name}_knee_mb"] = (
                None if banks is None else banks * bank_rows * ROW_BYTES / 2**20
            )
        result.append(entry)
    return result
