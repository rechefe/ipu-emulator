"""The reference-model API (E04-30) runs kernels as ``run_until_complete``
does (moved from ipu-apps, E07-25).

Each kernel's default case is staged on a ``ReferenceModel`` by the harness
(everything ``run_kernel`` does before the run), copied into an emulator
``IpuState`` through the API's reads alone, and both run: their registers,
XMEM and cycle counts agree. Stepping the staged kernel word by word gives
the records its traced run writes.
"""

from __future__ import annotations

import pytest

from ipu_emu import emulator
from ipu_emu.ipu_state import DEFAULT_CYCLE_LIMIT, IpuState
from ipu_emu.reference import RunState, TraceWriter, read_trace
from ipu_harness.cases import expand
from ipu_harness.registry import default_registry
from ipu_harness.runner import Staged, stage
from ipu_isa.codec.word import DecodedWord
from ipu_isa.constants import CR_READ_ONLY_INITIAL_VALUES
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.registers import REGISTER_DEFINITIONS

#: A copy loop (LR/CR, one vector path), a 3x3 convolution (masks, a ring of
#: rows, ACC.STRIDE) and a packed column softmax (the exponential and the
#: reciprocal), each in FP32 as the old harness ran them.
KERNELS = ["identity", "conv_universal", "softmax_columns_packed"]


def _staged(kernel: str) -> Staged:
    manifest = default_registry().kernel(kernel)
    case = next(c for c in expand(manifest) if c.name == "default")
    fmt = ElementFormat.FP32 if ElementFormat.FP32 in case.formats else case.formats[0]
    return stage(manifest, case, fmt)


def _register_bytes(value: int | bytes, size: int) -> bytes:
    return value.to_bytes(size, "little") if isinstance(value, int) else value


def _copied(staged: Staged) -> IpuState:
    """An emulator state holding the staged model's program, registers and the
    XMEM its tensors occupy, read through the API."""
    model = staged.model
    state = IpuState(fmt=staged.plan.format)
    emulator.load_program(state, [DecodedWord.from_word(w) for w in staged.program.words])
    for name, definition in REGISTER_DEFINITIONS.items():
        for index in range(definition["count"]):
            if name == "cr" and index in CR_READ_ONLY_INITIAL_VALUES:
                continue
            data = _register_bytes(model.read_register(name, index), definition["size_bytes"])
            state.regfile.set_register_bytes(name, index, data)
    used = staged.placement.rows_used * FORMATS[staged.plan.format].row_bytes
    state.xmem.write_address(0, model.read_xmem(0, used))
    return state


@pytest.mark.parametrize("kernel", KERNELS)
def test_a_kernel_runs_the_same_through_the_api(kernel, tmp_path):
    staged = _staged(kernel)
    state = _copied(staged)
    path = tmp_path / "run.jsonl"
    model = staged.model
    with TraceWriter(path, model.trace_header()) as trace:
        result = model.run(trace=trace)
    cycles = emulator.run_until_complete(state)
    assert result.status.state is RunState.HALTED and result.cycles == cycles
    for name, definition in REGISTER_DEFINITIONS.items():
        size = definition["size_bytes"]
        for index in range(definition["count"]):
            got = _register_bytes(model.read_register(name, index), size)
            want = bytes(state.regfile.view(name)[index * size : (index + 1) * size])
            assert got == want, (name, index)
    for base, page in state.xmem.touched_pages():
        assert model.read_xmem(base, len(page)) == bytes(page), base

    # Stepped word by word, the kernel gives the records its run traced.
    stepped = _staged(kernel).model
    records = []
    for _ in range(DEFAULT_CYCLE_LIMIT):
        records.append(stepped.step().to_json())
        if stepped.status.state is RunState.HALTED:
            break
    assert [r.to_json() for r in read_trace(path)[1]] == records
