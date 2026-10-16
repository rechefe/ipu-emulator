"""One measurement, and the two result files ``github-action-benchmark`` reads."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Times and memory (``customSmallerIsBetter``) and rates (``customBiggerIsBetter``).
SMALLER_FILE = "bench-smaller.json"
BIGGER_FILE = "bench-bigger.json"

# Set by the smoke test: every module runs its smallest useful measurement, so
# the modules cannot rot without the default tier noticing.
TINY_ENV = "IPU_BENCH_TINY"

_NAME = re.compile(r"[a-z0-9_]+(/[a-z0-9_.]+)+")


@dataclass(frozen=True)
class Result:
    """``name`` is ``<area>/<measurement>``, e.g. ``assemble/all_kernels``."""

    name: str
    unit: str
    value: float
    extra: str = ""
    bigger_is_better: bool = False

    def __post_init__(self):
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"benchmark name {self.name!r} is not <area>/<measurement>")
        if not self.unit:
            raise ValueError(f"{self.name}: unit is empty")
        if not isinstance(self.value, int | float) or not math.isfinite(self.value):
            raise ValueError(f"{self.name}: value {self.value!r} is not a finite number")

    def to_json(self) -> dict:
        entry = {"name": self.name, "unit": self.unit, "value": round(float(self.value), 6)}
        if self.extra:
            entry["extra"] = self.extra
        return entry


def tiny_budget() -> bool:
    """True under the smoke test: measure as little as still exercises the code."""
    return os.environ.get(TINY_ENV) == "1"


def write_results(results: Iterable[Result], out_dir: Path) -> tuple[Path, Path]:
    """Write the two files into ``out_dir``; return their paths (smaller, bigger)."""
    results = list(results)
    names = [r.name for r in results]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate benchmark names: {duplicates}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for file_name, bigger in ((SMALLER_FILE, False), (BIGGER_FILE, True)):
        entries = [r.to_json() for r in results if r.bigger_is_better is bigger]
        path = out_dir / file_name
        path.write_text(json.dumps(entries, indent=2) + "\n")
        paths.append(path)
    return paths[0], paths[1]


def append_results(results: Iterable[Result], out_dir: Path) -> None:
    """Add ``results`` to the files ``write_results`` wrote (``ci_times``)."""
    existing = []
    for file_name, bigger in ((SMALLER_FILE, False), (BIGGER_FILE, True)):
        path = out_dir / file_name
        if path.exists():
            existing += [
                Result(bigger_is_better=bigger, **entry) for entry in json.loads(path.read_text())
            ]
    write_results(existing + list(results), out_dir)
