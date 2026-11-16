"""Build a memory model from a TOML config file.

A config has one ``[model]`` table, whose ``kind`` picks the model and whose
other keys are that model's constructor arguments, and for ``cache_unit`` an
optional ``[dram]`` table (``ipu_memsim.dram.make_dram``). Example::

    [model]
    kind = "cache_unit"
    num_banks = 16
    bank_rows = 1024
    jump_back = "auto"

    [dram]
    kind = "fixed"
    latency_cycles = 100
    rows_per_cycle = 1.0

XMEM size is not a key of its own: it is ``num_banks * bank_rows * 128``
bytes for ``cache_unit`` and ``size_rows * 128`` bytes for ``cache``. The
helper :func:`rows_for_bytes` converts.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ipu_memsim.cache_unit import CacheUnit
from ipu_memsim.dram import make_dram
from ipu_memsim.models import AllMiss, Banked, Cache, Ideal, MemoryModel, PageFirstTouch
from ipu_memsim.params import ROW_BYTES, validate_config

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

__all__ = ["KINDS", "load_config", "make_model", "rows_for_bytes"]

#: Model kinds a config may name.
KINDS: Mapping[str, type[MemoryModel]] = {
    "ideal": Ideal,
    "all_miss": AllMiss,
    "page_first_touch": PageFirstTouch,
    "banked": Banked,
    "cache": Cache,
    "cache_unit": CacheUnit,
}


def rows_for_bytes(size_bytes: float) -> int:
    """XMEM rows in ``size_bytes`` (2 MB = 2 * 2**20 bytes = 16384 rows)."""
    return int(size_bytes) // ROW_BYTES


def make_model(config: Mapping[str, Any]) -> MemoryModel:
    """The model a config describes (``{"model": {...}, "dram": {...}}``).

    Raises:
        ValueError: No ``[model]``, an unknown ``kind``, a key the parameter
            registry (``ipu_memsim.params``) does not list for the model, or
            a value outside its range.
    """
    validate_config(config)  # every key and value, against the parameter registry
    arguments = dict(config["model"])
    kind = arguments.pop("kind")
    if kind == "cache_unit":
        arguments["dram"] = make_dram(config.get("dram"))
    if kind == "cache" and arguments.get("size_rows") == "inf":
        arguments["size_rows"] = None  # TOML has no null: "inf" is the infinite cache
    try:
        return KINDS[kind](**arguments)
    except TypeError as error:
        raise ValueError(f"[model] kind = {kind!r}: {error}") from error


def load_config(path: Path) -> dict[str, Any]:
    """Read a TOML config file."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)
