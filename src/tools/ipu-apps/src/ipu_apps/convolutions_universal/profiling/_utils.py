"""Shared utilities for convolutions_universal profiling scripts.

Provides:
  - ``assemble_if_needed``: assemble a .asm -> .bin alongside the .asm file,
    reusing the cached .bin if already present.
  - ``print_profile_table``: format and print profiler results.
  - ``run_profile_safe``: run ``run_profile`` and catch exceptions, returning
    a result dict or an error string.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Allow running scripts directly without installing the packages
_ROOT = Path(__file__).resolve().parents[6]  # repo root
for _pkg in ("ipu-emu-py/src", "ipu-common/src", "ipu-apps/src", "ipu-as-py/src"):
    _p = str(_ROOT / "src" / "tools" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ipu_emu.profiling_ipu import run_profile  # noqa: E402  (import after path setup)


def assemble_if_needed(asm_path: Path) -> Path:
    """Return path to the binary for *asm_path*, assembling it if not present.

    The binary is placed alongside the .asm file with a .bin extension.
    If the .bin already exists it is reused without reassembling.
    """
    bin_path = asm_path.with_suffix(".bin")
    if bin_path.exists():
        return bin_path

    from ipu_as import lark_tree  # type: ignore
    print(f"  Assembling {asm_path.name} -> {bin_path.name} ...")
    lark_tree.assemble_to_bin_file(asm_path.read_text(), str(bin_path))
    return bin_path


def run_profile_safe(
    app: Any,
    cr_names: dict[int, str],
    max_cycles: int = 5_000_000,
) -> dict[str, int] | str:
    """Run *app* through the profiler, returning results or an error string."""
    try:
        return run_profile(app, cr_names, max_cycles=max_cycles)
    except Exception as exc:
        return str(exc)


def make_tmp_bin(data: bytes) -> str:
    """Write *data* to a temporary file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".bin")
    os.write(fd, data)
    os.close(fd)
    return path


def cleanup(*paths: str) -> None:
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def print_profile_table(
    headers: list[str],
    rows: list[tuple[str, dict[str, int] | str]],
    col_width: int = 9,
) -> None:
    """Print a profiling results table.

    Args:
        headers:   Region labels in column order (e.g. ["inputs", "kernels", ...]).
        rows:      List of (config_label, result_or_error_str) pairs.
        col_width: Width of each data column.
    """
    label_w = max(len(r[0]) for r in rows) + 2

    header_line = f"{'Config':<{label_w}}" + "".join(
        f"{h:>{col_width}}" for h in headers
    )
    print(header_line)
    print("-" * len(header_line))

    for label, result in rows:
        if isinstance(result, str):
            # Error — print inline
            print(f"{label:<{label_w}} ERROR: {result}")
        else:
            values = "".join(
                f"{result.get(h, '-'):>{col_width}}" for h in headers
            )
            print(f"{label:<{label_w}}{values}")
