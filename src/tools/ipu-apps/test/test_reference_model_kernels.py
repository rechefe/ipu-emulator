"""The reference-model API (E04-30) runs kernels as ``run_until_complete``
does: each case's prepared state is copied into a ``ReferenceModel`` through
the API alone, both run, and their registers, XMEM and cycle counts agree."""

from __future__ import annotations

import pytest

from ipu_apps.kernel_registry.cases import load_cases, run_case
from ipu_emu import emulator
from ipu_emu.reference import ReferenceModel, RunState
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
def test_a_kernel_runs_the_same_through_the_api(kernel, case_id, monkeypatch):
    original = emulator.run_until_complete
    compared = []

    def both(state, max_cycles=None, **kwargs):
        model = _copied(state)
        result = model.run(max_cycles)
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
