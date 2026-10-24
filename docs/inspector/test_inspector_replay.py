"""A recorded trace replays to the run's final registers (E08-16).

The page rebuilds every cycle's registers from the document's ``init`` and the
cycles' deltas. Recording a real kernel-tree case through the runner's
recorder (E07-26) and replaying the whole document must give the registers the
model ended with.
"""

from __future__ import annotations

import struct

import pytest

from ipu_emu.trace import read_trace
from ipu_gen import inspector_trace
from ipu_gen.inspector_trace import convert
from ipu_harness import inspector
from ipu_harness.cases import expand
from ipu_harness.family import load_kernel
from ipu_harness.runner import RunOptions, run_kernel
from ipu_isa.registers import REGISTER_DEFINITIONS


def _replay(document: dict) -> dict[str, bytearray]:
    state = {name: bytearray.fromhex(data) for name, data in document["init"].items()}
    for cycle in document["cycles"]:
        for register, writes in cycle["d"].items():
            if register in ("lr", "cr"):
                for index, value in writes.items():
                    struct.pack_into("<I", state[register], int(index) * 4, value)
            else:
                for offset, data in writes:
                    chunk = bytes.fromhex(data)
                    state[register][offset : offset + len(chunk)] = chunk
    return state


def _recorded(folder, case_name, tmp_path):
    """The document of one kernel-tree case, and the registers the run ended with."""
    manifest = load_kernel(folder)
    case = next(c for c in expand(manifest) if c.name == case_name)
    recorder = inspector.InspectorRecorder(tmp_path / "trace.jsonl")
    final = {}

    def record(context):
        recorder(context)
        for name, definition in REGISTER_DEFINITIONS.items():
            final[name] = b"".join(
                _bytes(context.model.read_register(name, i), definition["size_bytes"])
                for i in range(definition["count"])
            )

    result = run_kernel(manifest, case, case.formats[0], RunOptions(executor=record))
    assert result.passed, result.describe()
    header, records = read_trace(recorder.trace_path)
    document = convert(header, [r.to_json() for r in records], recorder.context(manifest, case))
    return document, final


#: Kernel-tree cases, recorded by the runner's recorder (``--trace --context``):
#: NX8 vector registers with row metadata (a fixed and a dynamic variant), a
#: loop over FP32 rows, and LR/CR in a small loop.
#: Each with a word the context's labels name, as (pc, label).
TREE_CASES = [
    ("kernels/conformance/vector_tour", "dynamic", (15, "matvec")),
    ("kernels/conformance/vector_tour", "e4", (15, "matvec")),
    ("kernels/softmax/softmax_rows", "constant_row", None),
    ("kernels/reshape/identity", "default", (1, "row_loop")),
]


@pytest.mark.parametrize(("folder", "case_name", "label"), TREE_CASES)
def test_a_kernel_tree_case_replays_to_its_final_state(folder, case_name, label, tmp_path):
    document, final = _recorded(folder, case_name, tmp_path)
    assert not document["truncated"]
    if label is not None:
        pc, name = label
        assert document["program"][pc]["label"] == name
    replayed = _replay(document)
    for name in REGISTER_DEFINITIONS:
        assert bytes(replayed[name]) == final[name], name


def _bytes(value: int | bytes, size: int) -> bytes:
    return value.to_bytes(size, "little") if isinstance(value, int) else value


def test_the_document_covers_the_whole_run(tmp_path):
    document, _ = _recorded("kernels/reshape/identity", "default", tmp_path)
    assert not document["truncated"]
    assert len(document["cycles"]) == document["total_cycles"]
    assert document["program"][1]["label"] == "row_loop"
    assert any(cycle.get("L") for cycle in document["cycles"])
    assert any(m[0] == "R" for cycle in document["cycles"] for m in cycle["m"])
    assert any(m[0] == "W" for cycle in document["cycles"] for m in cycle["m"])


def test_recorder_and_adapter_agree_on_the_context_version():
    # Packages that may not import each other: the version is written in each.
    assert inspector.CONTEXT_VERSION == inspector_trace.CONTEXT_VERSION
