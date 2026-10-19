"""Write the Cycle Inspector's kernel list, ``kernels.json``, from ``curated.toml`` (E08-16).

    kernel_list.py curated.toml kernels.json TRACE.json...

Each ``[[kernel]]`` of ``curated.toml`` names a ``kernel_trace`` target by its
``id``; the TRACE files are those targets' outputs. The list keeps the TOML's
order and adds what each trace says about itself (kernel, case, source, element
format, cycles), so the page needs no hard-coded list. It fails when a listed
id has no trace or a trace is not listed: adding a kernel is one target and one
TOML entry, and neither can be forgotten.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import tomli

#: The keys each ``[[kernel]]`` entry holds (all required).
ENTRY_KEYS = frozenset({"id", "group", "title", "label", "description"})

#: The page's kernel groups, in display order: the hardware element formats,
#: then the emulator-only ones.
GROUPS = ("hardware", "emulator")

#: The kernel list's layout version; the page checks it (``KERNELS_SCHEMA``).
SCHEMA_VERSION = 1


def kernel_list(curated: dict[str, Any], traces: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The page's kernel list.

    Args:
        curated: ``curated.toml``, parsed.
        traces: Trace id (its file's stem) -> the trace document.

    Returns:
        One entry per curated kernel, in the TOML's order.

    Raises:
        ValueError: An entry with missing or unknown keys or an unknown
            group, a duplicate id, or the ids and the traces disagree.
    """
    entries = curated.get("kernel", [])
    ids = [entry.get("id") for entry in entries]
    for entry in entries:
        keys = set(entry)
        if keys != ENTRY_KEYS:
            raise ValueError(
                f"{entry.get('id')!r}: missing {sorted(ENTRY_KEYS - keys)}, "
                f"unknown {sorted(keys - ENTRY_KEYS)}"
            )
        if entry["group"] not in GROUPS:
            raise ValueError(f"{entry['id']!r}: group must be one of {GROUPS}")
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate ids in curated.toml: {duplicates}")
    unlisted, untraced = sorted(set(traces) - set(ids)), sorted(set(ids) - set(traces))
    if unlisted or untraced:
        raise ValueError(
            f"curated.toml and the kernel_trace targets disagree: traces not listed "
            f"{unlisted}, listed ids with no trace {untraced}"
        )
    return [_listed(entry, traces[entry["id"]]) for entry in entries]


def _listed(entry: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """A curated entry plus what its trace says about itself."""
    return {
        **entry,
        "kernel": trace["kernel"],
        "case": trace["case"],
        "source_path": trace["source_path"],
        "format": trace["meta"]["format"],
        "total_cycles": trace["total_cycles"],
    }


def main(argv: list[str]) -> int:
    """Write ``kernels.json``: ``argv`` is curated.toml, the output, then the traces."""
    curated_path, output, *trace_paths = (Path(arg) for arg in argv)
    curated = tomli.loads(curated_path.read_text(encoding="utf-8"))
    traces = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in trace_paths}
    document = {"schemaVersion": SCHEMA_VERSION, "kernels": kernel_list(curated, traces)}
    output.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
