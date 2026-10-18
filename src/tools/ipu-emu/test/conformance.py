"""The conformance suite's files, read and run through the reference-model API
(E04-31).

A case is a folder under ``src/tools/ipu-isa/conformance/``:

* ``program.asm``: the source, for reading;
* ``program.bin``: the assembled words, what a model loads;
* ``setup.json``: the format, register and XMEM contents, fault enables and
  cycle limit (``setup.schema.json`` beside the cases);
* ``expected.trace.jsonl``: the E04-29 trace of the run.

This module is one consumer of those files: it runs them through
``ipu_emu.reference`` and reads traces with ``ipu_emu.trace``; from
``ipu_isa`` it takes the trace header's fields and the decoding of a program
file. The generator (``regen_conformance.py``) writes the files and runs them
with the same :func:`run_case`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ipu_emu.commit import CommitRecord
from ipu_emu.reference import ReferenceModel
from ipu_emu.trace import TraceWriter
from ipu_isa.codec.word import opcode_field, unpack_program
from ipu_isa.errors import IsaLookupError
from ipu_isa.faults import FaultId
from ipu_isa.formats import ElementFormat
from ipu_isa.slots import SLOTS, Slot
from ipu_isa.view import isa

#: The suite, relative to the repository root (and to the runfiles root).
SUITE = Path("src/tools/ipu-isa/conformance")
SETUP_SCHEMA = "setup.schema.json"
UPDATE_COMMAND = "bazel run //src/tools/ipu-emu:regen_conformance"
FILES = ("program.asm", "program.bin", "setup.json", "expected.trace.jsonl")
#: Header fields that name the producer, not the behaviour: not compared.
PRODUCER_FIELDS = ("emulator_version",)

_FAULTS = {fault.label: fault for fault in FaultId}


def slot_ops(program: bytes) -> list[tuple[Slot, str]]:
    """Each slot instance's mnemonic in every word of a program file, in word
    order (an opcode no instruction has reads as ``"?"``)."""
    view = isa()
    ops = []
    for word in unpack_program(program):
        for slot, info in SLOTS.items():
            for instance in range(info.count):
                code = word.get(opcode_field(slot, instance), 0)
                try:
                    ops.append((slot, view.by_opcode(slot, code).mnemonic))
                except IsaLookupError:
                    ops.append((slot, "?"))
    return ops


def cases(suite: Path) -> list[Path]:
    """Every case folder of *suite*, by name."""
    return sorted(path for path in suite.iterdir() if path.is_dir())


def load(model: ReferenceModel, case: Path) -> dict[str, Any]:
    """Put *model* in *case*'s starting state, as the suite's README says.

    Args:
        model: The model; it is reset.
        case: The case folder.

    Returns:
        The case's ``setup.json``.
    """
    setup = json.loads((case / "setup.json").read_text(encoding="utf-8"))
    model.reset(ElementFormat(setup["format"]))
    model.load_program((case / "program.bin").read_bytes())
    for write in setup["registers"]:
        value = write["value"] if "value" in write else bytes.fromhex(write["hex"])
        model.write_register(write["name"], value, write["index"])
    for region in setup["xmem"]:
        model.write_xmem(region["address"], bytes.fromhex(region["hex"]))
    for label, enabled in setup["fault_enables"].items():
        model.set_fault_enable(_FAULTS[label], enabled)
    return setup


def run_case(case: Path, trace_path: Path) -> None:
    """Run *case* from its files until HALT, a fault or its cycle limit.

    Args:
        case: The case folder.
        trace_path: Where to write the trace of the run.
    """
    model = ReferenceModel()
    setup = load(model, case)
    with TraceWriter(trace_path, model.trace_header()) as trace:
        model.run(setup["max_cycles"], trace=trace)


def _differences(expected: Any, actual: Any, path: str) -> Iterator[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual), key=str):
            if key not in actual:
                yield f"{path}.{key}: expected {expected[key]!r}, missing"
            elif key not in expected:
                yield f"{path}.{key}: unexpected {actual[key]!r}"
            else:
                yield from _differences(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        for index, (want, got) in enumerate(zip(expected, actual, strict=False)):
            yield from _differences(want, got, f"{path}[{index}]")
        if len(expected) != len(actual):
            yield f"{path}: expected {len(expected)} items, got {len(actual)}"
    elif expected != actual:
        yield f"{path}: expected {expected!r}, got {actual!r}"


def first_difference(
    expected: tuple[dict[str, Any], list[CommitRecord]],
    actual: tuple[dict[str, Any], list[CommitRecord]],
) -> str | None:
    """The first field where two traces differ.

    Args:
        expected: A trace's header and records (``ipu_emu.trace.read_trace``).
        actual: Another's.

    Returns:
        ``header.<field>: ...`` or ``cycle N (record N): record.<path>:
        expected ..., got ...``; ``None`` when they agree, the header's
        producer fields aside.
    """
    (want_header, want), (got_header, got) = expected, actual
    keep = {k: v for k, v in want_header.items() if k not in PRODUCER_FIELDS}
    seen = {k: v for k, v in got_header.items() if k not in PRODUCER_FIELDS}
    for difference in _differences(keep, seen, "header"):
        return difference
    for index, (record, other) in enumerate(zip(want, got, strict=False)):
        for difference in _differences(record.to_json(), other.to_json(), "record"):
            return f"cycle {record.cycle} (record {index}): {difference}"
    if len(want) != len(got):
        return f"expected {len(want)} records, got {len(got)}"
    return None
