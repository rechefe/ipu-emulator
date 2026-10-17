"""A recorded trace replays to the run's final registers (E08-16).

The page rebuilds every cycle's registers from the document's ``init`` and the
cycles' deltas. Recording a real case and replaying the whole document must
give the registers the emulator ended with: an ``ipu_apps`` case through
``trace_case``, a kernel-tree case through the runner's recorder (E07-26).
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from ipu_apps.kernel_registry import trace_case
from ipu_apps.kernel_registry.cases import load_cases, run_case
from ipu_apps.kernel_registry.trace_case import CaseTrace, case_context, record_case
from ipu_emu.trace import read_trace
from ipu_gen import inspector_trace
from ipu_gen.inspector_trace import convert
from ipu_harness import inspector
from ipu_harness.cases import expand
from ipu_harness.family import load_kernel
from ipu_harness.runner import RunOptions, run_kernel
from ipu_isa.registers import REGISTER_DEFINITIONS

# Small cases that exercise LR/CR, vector registers and a loop.
CASES = [("identity", "default", {}), ("softmax_rows", "default", {"rows": 2})]


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


def _document(kernel, case, options, tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    context = record_case(kernel, case, trace_path, options)
    header, records = read_trace(trace_path)
    return convert(header, [r.to_json() for r in records], context)


@pytest.mark.parametrize(("kernel", "case", "options"), CASES)
def test_init_plus_every_delta_is_the_final_state(kernel, case, options, tmp_path):
    recorder = CaseTrace(tmp_path / "trace.jsonl")
    with recorder.watch():
        final, _ = run_case(kernel, load_cases(kernel)[case], options=options)
    header, records = read_trace(recorder.trace_path)
    context = case_context(kernel, case, options, recorder)
    replayed = _replay(convert(header, [r.to_json() for r in records], context))
    for name in REGISTER_DEFINITIONS:
        assert bytes(replayed[name]) == bytes(final.regfile.view(name)), name


#: A kernel-tree case: the runner's recorder (``--trace --context``).
VECTOR_TOUR = Path("kernels/conformance/vector_tour")


@pytest.mark.parametrize("case_name", ["int8", "e4"])
def test_a_kernel_tree_case_replays_to_its_final_state(case_name, tmp_path):
    manifest = load_kernel(str(VECTOR_TOUR))
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
    assert not document["truncated"] and document["program"][15]["label"] == "matvec"
    replayed = _replay(document)
    for name in REGISTER_DEFINITIONS:
        assert bytes(replayed[name]) == final[name], name


def _bytes(value: int | bytes, size: int) -> bytes:
    return value.to_bytes(size, "little") if isinstance(value, int) else value


def test_the_document_covers_the_whole_run(tmp_path):
    document = _document("identity", "default", {}, tmp_path)
    assert not document["truncated"]
    assert len(document["cycles"]) == document["total_cycles"]
    assert document["program"][1]["label"] == "row_loop"
    assert any(cycle.get("L") for cycle in document["cycles"])
    assert any(m[0] == "R" for cycle in document["cycles"] for m in cycle["m"])
    assert any(m[0] == "W" for cycle in document["cycles"] for m in cycle["m"])


def test_recorder_and_adapter_agree_on_the_context_version():
    # Packages that may not import each other: the version is written in each.
    assert trace_case.CONTEXT_VERSION == inspector_trace.CONTEXT_VERSION
    assert inspector.CONTEXT_VERSION == inspector_trace.CONTEXT_VERSION
