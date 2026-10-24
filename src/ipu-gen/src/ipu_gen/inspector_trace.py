"""The Cycle Inspector's per-kernel JSON, from an E04-29 trace (E08-16).

The trace (``ipu_isa/schemas``: a header line, then one commit record per
retired word) is the contract; this file is one presentation of it, for the
docs page's trace viewer. ``kernel_trace`` (``docs/inspector/defs.bzl``) records
the trace and its context with ``run_kernel --trace --context``, then runs
``ipu-gen inspector-trace`` on them.

The document (``schemas/inspector-trace.schema.json``):

* ``schemaVersion``: :data:`SCHEMA_VERSION`. The page refuses a version it
  does not know.
* ``kernel``, ``case``, ``options``, ``source_path``, ``source``: what ran.
* ``meta``: the run mode (``nx8`` or ``fp32``, ADR 0007), the case's variant
  and, in NX8, its element type (``nx8``: the QF half's fields) and ``lut``:
  per ``dynamic_exponent`` (one table, two for a dynamic type), each of the
  256 codes' values at unit scale (scale 127). An element's value is its
  code's entry times ``2 ** (scale - 127)``, with ``scale`` and
  ``dynamic_exponent`` from its row's metadata word. ``null`` in FP32.
* ``trace``: the trace header's provenance (schema version, ISA and program
  digests).
* ``program``: per word, its label (or ``null``), the word in hex and its
  non-NOP slots as ``{slot, instance, op, args}``, decoded by the ``ipu_isa``
  codec, operands as assembly writes them.
* ``layout``: every field of the word, MSB to LSB, as
  ``[name, width, slot]`` (``ipu_isa.codec.word.WORD_LAYOUT``).
* ``init``: every register's bytes when the program started, in hex.
* ``cycles``: per retired word, ``pc``; ``d``, its register writes (``lr``
  and ``cr`` as ``{index: value}``, any other register as
  ``[[byte offset, hex], ...]``); ``m``, its XMEM accesses as
  ``[["R" | "W", address, bytes], ...]``; ``M``, the XMEM metadata words it
  wrote, when any, as ``[[address, value], ...]``; ``L``, the multiplier lanes
  it used, when any; ``o``, its outcome when it is not ``retired``.
* ``total_cycles``: the trace's records; ``truncated``: whether ``cycles``
  holds only the first ``max_cycles`` of them.

``L``, the XMEM reads and the variant come from the context file: the commit
record has none of them (it records writes, and its header names the mode
only), so the recorder takes them from the run.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ipu_isa.codec.disasm import operand_text
from ipu_isa.codec.slot import codecs
from ipu_isa.codec.word import WORD_LAYOUT
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.nx8 import SCALE_BIAS, Nx8Format, decode_element
from ipu_isa.operand_types import OperandKind
from ipu_isa.tags import Tag
from ipu_isa.trace_schema import isa_digest, program_digest
from ipu_isa.view import isa

__all__ = [
    "DEFAULT_MAX_CYCLES",
    "SCHEMA_VERSION",
    "TraceError",
    "convert",
    "read_inputs",
    "schema",
]

#: The viewer document's version. Bump it on any change the page must know of.
SCHEMA_VERSION = 2

#: The commit record schema major version this adapter reads.
TRACE_MAJOR = 1

#: The recorder's context layout (``ipu_harness.inspector``) this adapter reads.
CONTEXT_VERSION = 2

#: Cycles kept per trace unless the caller says otherwise: enough for every
#: curated kernel's interesting part, small enough for the page to load.
DEFAULT_MAX_CYCLES = 5000

#: The label a branch target gets when the source's labels are unknown.
GENERATED_LABEL = "w{:04x}"

_SCALAR_REGISTERS = ("lr", "cr")


class TraceError(ValueError):
    """The trace or its context is not one this adapter reads."""


def read_inputs(trace_path: Path, context_path: Path) -> tuple[dict, list[dict], dict]:
    """The trace's header and records, and the context, parsed.

    Raises:
        TraceError: The trace is empty.
    """
    with trace_path.open(encoding="utf-8") as lines:
        docs = [json.loads(line) for line in lines if line.strip()]
    if not docs:
        raise TraceError(f"{trace_path}: empty trace (no header)")
    context = json.loads(context_path.read_text(encoding="utf-8"))
    return docs[0], docs[1:], context


def _check(header: Mapping[str, Any], records: Sequence[Mapping], context: Mapping) -> None:
    major = str(header.get("schema_version", "")).split(".")[0]
    if major != str(TRACE_MAJOR):
        raise TraceError(
            f"commit record schema {header.get('schema_version')!r}; this adapter reads "
            f"{TRACE_MAJOR}.x"
        )
    if header.get("isa_digest") != isa_digest():
        raise TraceError("the trace was recorded with another ISA; record it again")
    if context.get("context_version") != CONTEXT_VERSION:
        raise TraceError(
            f"context version {context.get('context_version')!r}; this adapter reads "
            f"{CONTEXT_VERSION}"
        )
    if header.get("program_digest") != program_digest(int(w, 16) for w in context["program"]):
        raise TraceError("the context's program is not the program the trace ran")
    if len(context["cycles"]) != len(records):
        raise TraceError(f"{len(records)} records but {len(context['cycles'])} context cycles")


def _nx8_type(variant: Mapping[str, Any]) -> Nx8Format | None:
    """The variant's NX8 element type, or ``None`` (an FP32 run)."""
    qf = variant.get("qf")
    if qf is None:
        return None
    return Nx8Format(qf["sign"], qf["exp_format"], qf["num_bits"], bool(qf["dynamic"]))


def _lut(element: Nx8Format) -> list[list[float]]:
    """Each code's value at unit scale, per ``dynamic_exponent`` the type uses."""
    exponents = (0, 1) if element.dynamic else (0,)
    return [
        [decode_element(code, element, SCALE_BIAS, dyn) for code in range(256)] for dyn in exponents
    ]


def _meta(fmt: ElementFormat, variant: Mapping[str, Any]) -> dict[str, Any]:
    info = FORMATS[fmt]
    element = _nx8_type(variant)
    if (element is None) != (fmt is ElementFormat.FP32):
        raise TraceError(f"variant {variant.get('name')!r} does not fit a {fmt.value} run")
    return {
        "format": fmt.value,
        "variant": variant["name"],
        "element_bits": info.element_bits,
        "accumulator": info.accumulator.value,
        # FP32 exists in the emulator only.
        "emulator_only": info.tag is Tag.EMULATOR,
        "nx8": None
        if element is None
        else {
            "sign": element.sign,
            "exp_format": element.exp_format,
            "num_bits": element.num_bits,
            "dynamic": element.dynamic,
        },
        "lut": None if element is None else _lut(element),
    }


def _slots(word: int, labels: Mapping[int, str]) -> Iterator[dict[str, Any]]:
    """The word's non-NOP slot instances, MSB first, as the viewer lists them."""
    view = isa()
    unpacked = codecs(view).word.unpack(word)
    for slot in view.layout.layout_order:
        for instance in range(view.slots[slot].count):
            decoded = unpacked[(slot, instance)]
            instruction = decoded.instruction
            if instruction.mnemonic == "NOP":
                continue
            args = []
            for operand, value in zip(instruction.operands, decoded.operands, strict=True):
                if operand.type.kind is OperandKind.LABEL and value in labels:
                    args.append(labels[value])
                else:
                    args.append(operand_text(operand, value).upper())
            yield {
                "slot": slot.value,
                "instance": instance,
                "op": instruction.mnemonic,
                "args": args,
            }


def _branch_targets(words: Sequence[int]) -> dict[int, str]:
    """A generated label for each word a branch of the program names."""
    targets: dict[int, str] = {}
    view = isa()
    for word in words:
        for decoded in codecs(view).word.unpack(word).values():
            for operand, value in zip(decoded.instruction.operands, decoded.operands, strict=True):
                if operand.type.kind is OperandKind.LABEL and value < len(words):
                    targets[value] = GENERATED_LABEL.format(value)
    return targets


def _program(words: Sequence[int], source_labels: Mapping[str, int]) -> list[dict[str, Any]]:
    labels: dict[int, str] = {}
    for name, address in source_labels.items():
        labels.setdefault(address, name)  # several at one word: the first defined
    labels = labels or _branch_targets(words)
    return [
        {"label": labels.get(pc), "word": f"{word:x}", "slots": list(_slots(word, labels))}
        for pc, word in enumerate(words)
    ]


def _cycle(record: Mapping[str, Any], facts: Mapping[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for write in record["reg_writes"]:
        register = write["register"]
        if register in _SCALAR_REGISTERS:
            deltas.setdefault(register, {})[str(write["index"])] = write["value"]
        else:
            deltas.setdefault(register, []).append([write["offset"], write["data"]])
    memory = [["R", address, size] for address, size in facts["xmem_reads"]]
    memory += [["W", w["address"], len(w["data"]) // 2] for w in record["xmem_writes"]]
    cycle: dict[str, Any] = {"pc": record["pc_before"], "d": deltas, "m": memory}
    metadata = [[w["address"], w["value"]] for w in record.get("xmem_meta_writes", [])]
    if metadata:
        cycle["M"] = metadata
    if facts["lanes"]:
        cycle["L"] = facts["lanes"]
    if record["outcome"] != "retired":
        cycle["o"] = record["outcome"]
    return cycle


def convert(
    header: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    *,
    max_cycles: int = DEFAULT_MAX_CYCLES,
) -> dict[str, Any]:
    """The viewer document of one recorded case.

    Args:
        header: The trace's header line.
        records: Its commit records, in order.
        context: The recorder's context document (``run_kernel --context``).
        max_cycles: Keep at most this many cycles (``truncated`` says so).

    Returns:
        The document (see the module docstring).

    Raises:
        TraceError: The trace's schema, ISA or context is not one this adapter
            reads, or the context does not match the trace.
    """
    if max_cycles < 1:
        raise TraceError(f"max_cycles must be positive, not {max_cycles}")
    _check(header, records, context)
    words = [int(word, 16) for word in context["program"]]
    kept = min(len(records), max_cycles)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kernel": context["kernel"],
        "case": context["case"],
        "options": context["options"],
        "source_path": context["source_path"],
        "source": context["source"],
        "meta": _meta(ElementFormat(header["format"]), context["variant"]),
        "trace": {
            "schema_version": header["schema_version"],
            "isa_digest": header["isa_digest"],
            "program_digest": header["program_digest"],
        },
        "program": _program(words, context["labels"]),
        "layout": [[f.name, f.bits, f.slot.value] for f in reversed(WORD_LAYOUT)],
        "init": dict(context["init"]),
        "cycles": [_cycle(records[i], context["cycles"][i]) for i in range(kept)],
        "total_cycles": len(records),
        "truncated": kept < len(records),
    }


def schema() -> dict[str, Any]:
    """``schemas/inspector-trace.schema.json``, parsed."""
    path = Path(__file__).with_name("schemas") / "inspector-trace.schema.json"
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded
