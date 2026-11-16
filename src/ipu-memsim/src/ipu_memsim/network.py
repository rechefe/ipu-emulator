"""Network-level aggregation: a per-layer-type table for a whole network.

A network is a TOML file listing its layers. Each ``[[layer]]`` names a layer
type (the row of the thesis table it belongs to), the kernel that computes
it, the case or parameter values that give the layer's real shape, and how
many times the layer occurs::

    [[layer]]
    type = "pointwise_conv"
    name = "MV2-1 project 64->32 @128x128"
    kernel = "pointwise_conv_unified"
    values = { height = 128, width = 128, in_channels = 64, out_channels = 32 }
    count = 1

``[[missing]]`` entries name layer types the network has but no kernel
computes, so the table shows the gap instead of hiding it. A new network
(YOLO, another ViT) is a new file of this kind and nothing else.

:func:`run_network` records each layer's kernel once (the trace is kept and
reused), computes its working set, replays it under every memory config
given, and writes ``layers.csv`` (one row per layer and config),
``layer_types.csv`` and ``layer_types.md`` (summed per layer type, each layer
weighted by its count). A layer whose kernel cannot be recorded stays in the
tables with the reason.

All stall numbers are TOY-MODEL results (see ``ipu_memsim.sweep``), and each
kernel is replayed on its own: data handed from one layer to the next inside
XMEM is not modelled yet (question Q2).
"""

from __future__ import annotations

import csv
import re
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ipu_memsim.config import load_config, make_model
from ipu_memsim.replay import replay
from ipu_memsim.trace import Trace, load_trace, save_trace
from ipu_memsim.workset import working_set

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

__all__ = ["LayerResult", "run_network"]


@dataclass
class LayerResult:
    """One layer of the network, recorded and replayed.

    Attributes:
        type, name, kernel, count: From the network file.
        error: Why the layer has no numbers, if it has none.
        ideal_cycles: The emulator's cycle count for one occurrence.
        accesses: XMEM accesses of one occurrence.
        footprint_rows: Distinct rows touched, all regions summed.
        peak_live_rows: Peak live rows, all regions summed (an upper bound on
            the peak of the sum).
        input_rows, output_rows: Rows read from input regions and written to
            output regions (distinct), the data that crosses to DRAM.
        stall_cycles: Stall cycles of one occurrence, by config name.
    """

    type: str
    name: str
    kernel: str
    count: int
    error: str = ""
    ideal_cycles: int = 0
    accesses: int = 0
    footprint_rows: int = 0
    peak_live_rows: int = 0
    input_rows: int = 0
    output_rows: int = 0
    stall_cycles: dict[str, int] = field(default_factory=dict)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def _trace_for(layer: dict[str, Any], traces: Path, commit: str) -> Trace:
    """The layer's trace: loaded when already recorded, else recorded and saved."""
    values = layer.get("values") or {}
    label = "_".join(
        [layer["kernel"], layer.get("case", "default"), *(f"{k}{v}" for k, v in values.items())]
    )
    path = traces / f"{_slug(label)}.csv.gz"
    if path.exists():
        return load_trace(path)
    from ipu_memsim.record import record  # imports the harness: only when recording

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # emulator-only instructions warn on every use
        trace = record(
            layer["kernel"],
            layer.get("case", "default"),
            values,
            layer.get("format"),
            commit=commit,
        )
    save_trace(trace, path)
    return trace


def _measure(
    layer: dict[str, Any], traces: Path, configs: dict[str, dict[str, Any]], commit: str
) -> LayerResult:
    result = LayerResult(
        layer["type"],
        layer.get("name", layer["kernel"]),
        layer["kernel"],
        int(layer.get("count", 1)),
    )
    try:
        trace = _trace_for(layer, traces, commit)
    except Exception as error:  # noqa: BLE001 (any failure becomes the layer's row)
        result.error = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
        return result
    sets = working_set(trace)
    result.ideal_cycles = trace.ideal_cycles
    result.accesses = len(trace)
    result.footprint_rows = sum(w.footprint for w in sets)
    result.peak_live_rows = sum(w.peak_live for w in sets)
    result.input_rows = sum(w.footprint for w in sets if w.role == "input")
    result.output_rows = sum(w.footprint for w in sets if w.role == "output")
    for name, config in configs.items():
        try:
            result.stall_cycles[name] = replay(trace, make_model(config)).stall_cycles
        except ValueError as error:
            result.stall_cycles[name] = -1  # this configuration cannot run the layer
            result.error = f"{name}: {error}"
    return result


def run_network(
    network: Path, out: Path, configs: Sequence[Path], *, commit: str = "unknown"
) -> list[Path]:
    """Measure every layer of ``network`` under every config; write the tables.

    Args:
        network: The network description (TOML, see the module docstring).
        out: Output folder; traces go to ``out/traces``.
        configs: Model config files; a config's name is its file stem.
        commit: The emulator commit, kept in new traces.

    Returns:
        The files written.
    """
    with open(network, "rb") as handle:
        description = tomllib.load(handle)
    loaded = {path.stem: load_config(path) for path in configs}
    out.mkdir(parents=True, exist_ok=True)
    results = [
        _measure(layer, out / "traces", loaded, commit) for layer in description.get("layer", [])
    ]
    names = list(loaded)

    layers_csv = out / "layers.csv"
    with open(layers_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["type", "name", "kernel", "count", "ideal_cycles", "accesses", "footprint_rows",
                         "peak_live_rows", "input_rows", "output_rows",
                         *(f"stall_{n}" for n in names), "note"])  # fmt: skip
        for r in results:
            writer.writerow([r.type, r.name, r.kernel, r.count, r.ideal_cycles, r.accesses, r.footprint_rows,
                             r.peak_live_rows, r.input_rows, r.output_rows,
                             *(r.stall_cycles.get(n, "") for n in names), r.error])  # fmt: skip

    # Per layer type: sums over its layers, each weighted by its count.
    types = list(dict.fromkeys(r.type for r in results))
    header = ["layer type", "layers", "ideal cycles", "max footprint (rows)", "max peak live (rows)",
              *(f"stall: {n}" for n in names), "not measured"]  # fmt: skip
    rows = []
    for layer_type in types:
        group = [r for r in results if r.type == layer_type]
        measured = [r for r in group if r.ideal_cycles]
        ideal = sum(r.ideal_cycles * r.count for r in measured)
        stalls = []
        for n in names:
            usable = [r for r in measured if r.stall_cycles.get(n, -1) >= 0]
            total = sum(r.stall_cycles[n] * r.count for r in usable)
            share = f" ({100 * total / ideal:.1f}%)" if ideal else ""
            stalls.append(f"{total}{share}" if len(usable) == len(measured) else f"{total}{share}*")
        rows.append([
            layer_type, sum(r.count for r in group), ideal,
            max((r.footprint_rows for r in measured), default=0),
            max((r.peak_live_rows for r in measured), default=0),
            *stalls, "; ".join(r.name for r in group if not r.ideal_cycles),
        ])  # fmt: skip
    for gap in description.get("missing", []):
        rows.append(
            [gap["type"], 0, 0, 0, 0, *[""] * len(names), "NO KERNEL: " + gap.get("note", "")]
        )

    types_csv = out / "layer_types.csv"
    with open(types_csv, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([header, *rows])
    lines = [
        f"# {description.get('name', network.stem)}: per-layer-type memory table",
        "",
        "TOY-MODEL stall numbers (placeholder DRAM speed; each kernel replayed alone). "
        "Stall = cycles, and percent of the type's ideal cycles. A `*` marks a config "
        "that could not run every layer of the type. Sizes are in XMEM rows (128 bytes on hardware).",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows),
        "",
    ]
    types_md = out / "layer_types.md"
    types_md.write_text("\n".join(lines), encoding="utf-8")
    return [layers_csv, types_csv, types_md]
