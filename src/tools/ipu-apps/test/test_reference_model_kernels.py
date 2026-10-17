"""The reference-model API (E04-30) runs kernels as ``run_until_complete``
does: each case's prepared state is copied into a ``ReferenceModel`` through
the API alone, both run, and their registers, XMEM and cycle counts agree;
stepping the kernel word by word gives the records its run traces."""

from __future__ import annotations

import pytest

from ipu_apps.kernel_registry.cases import load_cases, run_case
from ipu_emu import emulator
from ipu_emu.ipu_state import DEFAULT_CYCLE_LIMIT
from ipu_emu.reference import ReferenceModel, RunState, TraceWriter, read_trace
from ipu_isa.codec.word import DecodedWord
from ipu_isa.constants import CR_READ_ONLY_INITIAL_VALUES
from ipu_isa.registers import REGISTER_DEFINITIONS

CASES = [
    ("identity", "default"),
    ("conv1x1", "full_group"),
    ("softmax_columns_packed", "default"),
]


def _copied(state) -> ReferenceModel:
    """A model holding *state*'s program, registers and XMEM, set through the API."""
    model = ReferenceModel(state.fmt)
    model.load_program(
        [DecodedWord(word or {}).to_word() for word in state.inst_mem[: state.program_length]]
    )
    for name, definition in REGISTER_DEFINITIONS.items():
        for index in range(definition["count"]):
            if name == "cr" and index in CR_READ_ONLY_INITIAL_VALUES:
                continue
            size = definition["size_bytes"]
            data = bytes(state.regfile.view(name)[index * size : (index + 1) * size])
            value = data if definition["vector"] else int.from_bytes(data, "little")
            model.write_register(name, value, index)
    for base, page in state.xmem.touched_pages():
        model.write_xmem(base, bytes(page))
    return model


@pytest.mark.parametrize(("kernel", "case_id"), CASES)
def test_a_kernel_runs_the_same_through_the_api(kernel, case_id, monkeypatch, tmp_path):
    original = emulator.run_until_complete
    compared = []

    def both(state, max_cycles=None, **kwargs):
        # Stepped word by word, the kernel gives the records its run traces.
        stepped = _copied(state)
        records = []
        for _ in range(max_cycles or DEFAULT_CYCLE_LIMIT):
            records.append(stepped.step().to_json())
            if stepped.status.state is RunState.HALTED:
                break
        model = _copied(state)
        path = tmp_path / f"{len(compared)}.jsonl"
        with TraceWriter(path, model.trace_header()) as trace:
            result = model.run(max_cycles, trace=trace)
        assert [r.to_json() for r in read_trace(path)[1]] == records
        cycles = original(state, max_cycles, **kwargs)
        assert result.status.state is RunState.HALTED and result.cycles == cycles
        for name, definition in REGISTER_DEFINITIONS.items():
            size = definition["size_bytes"]
            for index in range(definition["count"]):
                got = model.read_register(name, index)
                data = bytes(state.regfile.view(name)[index * size : (index + 1) * size])
                want = data if definition["vector"] else int.from_bytes(data, "little")
                assert got == want, (name, index)
        for base, page in state.xmem.touched_pages():
            assert model.read_xmem(base, len(page)) == bytes(page), base
        compared.append(cycles)
        return cycles

    monkeypatch.setattr(emulator, "run_until_complete", both)
    run_case(kernel, load_cases(kernel)[case_id])
    assert compared  # the case ran a program through the run, and it was compared
