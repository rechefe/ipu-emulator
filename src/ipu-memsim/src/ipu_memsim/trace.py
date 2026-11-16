"""The XMEM access trace: what one kernel run touched, cycle by cycle.

A trace is five equal-length columns, one entry per XMEM access the program
made, in the order the emulator made them:

========  =====================================================================
column    meaning
========  =====================================================================
cycle     the word (= cycle, the IPU runs one word per cycle) that made it
op        ``READ`` (0) or ``WRITE`` (1)
region    index into ``header["regions"]``, or ``OTHER`` (-1) when the row
          lies in no declared region
row       the flat XMEM row (byte address // bytes per row)
nbytes    the bytes the emulator moved (one row is 128 bytes in the 8-bit
          formats and 512 bytes in FP32/INT32)
========  =====================================================================

plus a small header (a JSON-ready dict): the kernel, its case and parameter
values, the element format, the ideal cycle count, the regions and the
emulator commit.

On disk it is one CSV file whose first line is ``# `` followed by the header
as JSON; a ``.gz`` suffix compresses it. Traces are large (hundreds of
thousands of lines) and are kept out of git (``/memsim-out/`` in
``.gitignore``).
"""

from __future__ import annotations

import gzip
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["OTHER", "READ", "WRITE", "Trace", "load_trace", "save_trace"]

READ = 0
WRITE = 1
#: The region index of an access outside every declared region.
OTHER = -1

_OP_LETTER = {READ: "R", WRITE: "W"}
_COLUMNS = "cycle,op,region,row,bytes"


@dataclass
class Trace:
    """One kernel run's XMEM accesses (see the module docstring).

    Attributes:
        header: Run facts. Keys the replay relies on: ``ideal_cycles`` (the
            emulator's cycle count), ``regions`` (a list of
            ``{"name", "role", "base_row", "rows"}``) and ``row_bytes``.
        cycle, op, region, row, nbytes: The columns.
    """

    header: dict[str, Any]
    cycle: np.ndarray
    op: np.ndarray
    region: np.ndarray
    row: np.ndarray
    nbytes: np.ndarray

    def __len__(self) -> int:
        return len(self.cycle)

    @property
    def ideal_cycles(self) -> int:
        """Cycles the emulator ran: the runtime with a perfect memory."""
        return int(self.header["ideal_cycles"])

    @property
    def region_names(self) -> list[str]:
        """Region names, by region index."""
        return [r["name"] for r in self.header["regions"]]

    def region_name(self, index: int) -> str:
        """The name of region ``index`` (``"other"`` for ``OTHER``)."""
        return "other" if index == OTHER else self.header["regions"][index]["name"]


def _open(path: Path, mode: str) -> io.TextIOBase:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")  # type: ignore[return-value]
    return open(path, mode, encoding="utf-8", newline="")  # type: ignore[return-value]


def save_trace(trace: Trace, path: Path) -> None:
    """Write ``trace`` to ``path`` (CSV; gzip when the name ends in ``.gz``).

    The output depends only on the trace, so the same run gives the same file
    (gzip is written without a timestamp).
    """
    names = trace.region_names
    lines = ["# " + json.dumps(trace.header, sort_keys=True), _COLUMNS]
    for c, o, g, r, b in zip(
        trace.cycle.tolist(),
        trace.op.tolist(),
        trace.region.tolist(),
        trace.row.tolist(),
        trace.nbytes.tolist(),
        strict=True,
    ):
        lines.append(f"{c},{_OP_LETTER[o]},{'other' if g == OTHER else names[g]},{r},{b}")
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
            zipped.write(text.encode("utf-8"))
    else:
        path.write_text(text, encoding="utf-8")


def load_trace(path: Path) -> Trace:
    """Read a trace written by :func:`save_trace`.

    Raises:
        ValueError: The file is not a trace (no header line, wrong columns).
    """
    with _open(path, "r") as handle:
        first = handle.readline()
        if not first.startswith("# "):
            raise ValueError(f"{path}: not a memsim trace (no '# {{header}}' first line)")
        header = json.loads(first[2:])
        if handle.readline().strip() != _COLUMNS:
            raise ValueError(f"{path}: expected the columns {_COLUMNS}")
        index = {r["name"]: i for i, r in enumerate(header["regions"])}
        index["other"] = OTHER
        cycle, op, region, row, nbytes = [], [], [], [], []
        for line in handle:
            c, o, g, r, b = line.rstrip("\n").split(",")
            cycle.append(int(c))
            op.append(READ if o == "R" else WRITE)
            region.append(index[g])
            row.append(int(r))
            nbytes.append(int(b))
    return Trace(
        header,
        np.asarray(cycle, dtype=np.int64),
        np.asarray(op, dtype=np.int8),
        np.asarray(region, dtype=np.int16),
        np.asarray(row, dtype=np.int64),
        np.asarray(nbytes, dtype=np.int32),
    )
