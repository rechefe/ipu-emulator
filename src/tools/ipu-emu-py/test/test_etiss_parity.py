"""Differential tests: the ETISS backend must match the Python emulator exactly.

The Python emulator is the reference.  For every program in
:mod:`etiss_corpus` both backends run the same initial state and the resulting
register file, XMEM, program counter, cycle count and run statistics must be
identical -- byte for byte, not approximately.

These tests skip when the ETISS runner is not built.  Point ``$IPU_ETISS_RUN``
at the ``ipu_etiss_run`` binary (or put it on ``$PATH``) to run them.
"""

from __future__ import annotations

import pytest

from ipu_as.lark_tree import assemble
from ipu_common.instruction_spec import INSTRUCTION_SPEC

from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.etiss import EtissRunner, is_available
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.xmem import XMEM_SIZE_BYTES

from etiss_corpus import CASES, Case, UNCOVERED_OK

pytestmark = pytest.mark.skipif(
    not is_available(), reason="ETISS runner not built (set $IPU_ETISS_RUN)"
)

_DTYPES = {
    "int8": DType.INT8,
    "fp8_e2": DType.E2,
    "fp8_e4": DType.E4,
    "fp8_e5": DType.E5,
}

_MAX_CYCLES = 200_000

#: Registers compared after a run, by their name in the register file.
_COMPARED_REGISTERS = (
    "r", "r_wide_debug", "r_cyclic", "r_cyclic_wide_debug", "r_mask", "r_acc",
    "post_aaq_reg", "lr", "cr", "mult_res", "mem_bypass",
)


def _build_state(case: Case) -> IpuState:
    state = IpuState(
        dtype=_DTYPES[case.dtype],
        elu_alpha=case.elu_alpha,
        wide_vector_debug=case.wide,
        wide_vector_arithmetic=WideVectorArithmetic(case.wide_arith),
        wide_vector_quantize_output=case.wide_quantize,
    )
    for idx, value in case.cr.items():
        state.regfile.set_cr(idx, value)
    for addr, data in case.xmem.items():
        state.xmem.write_address(addr, data)
    decoded = [decode_instruction_word(word) for word in assemble(case.asm)]
    load_program(state, decoded)
    return state


def _describe(state: IpuState) -> dict[str, object]:
    return {
        "pc": state.program_counter,
        "stats": (
            state.stats.total_cycles,
            state.stats.mult_active_cycles,
            state.stats.acc_active_cycles,
            state.stats.xmem_reads,
            state.stats.xmem_writes,
        ),
        **{name: bytes(state.regfile.raw(name)) for name in _COMPARED_REGISTERS},
    }


def _first_difference(a: bytes, b: bytes) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"first difference at byte {i}: python={x} etiss={y}"
    return f"lengths differ: python={len(a)} etiss={len(b)}"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_etiss_matches_python(case: Case) -> None:
    python_state = _build_state(case)
    python_cycles = run_until_complete(python_state, max_cycles=_MAX_CYCLES)

    etiss_state = _build_state(case)
    etiss_cycles = EtissRunner().run(etiss_state, max_cycles=_MAX_CYCLES)

    assert etiss_cycles == python_cycles, "cycle count differs"

    expected = _describe(python_state)
    actual = _describe(etiss_state)
    for key, want in expected.items():
        got = actual[key]
        if isinstance(want, bytes) and want != got:
            pytest.fail(f"register {key!r} differs: {_first_difference(want, got)}")
        assert got == want, f"{key} differs"

    python_xmem = bytes(python_state.xmem.read_address(0, XMEM_SIZE_BYTES))
    etiss_xmem = bytes(etiss_state.xmem.read_address(0, XMEM_SIZE_BYTES))
    if python_xmem != etiss_xmem:
        pytest.fail(f"XMEM differs: {_first_difference(python_xmem, etiss_xmem)}")


def test_corpus_covers_every_instruction() -> None:
    """Every instruction in the spec is exercised by at least one program.

    Guards against an instruction being added to ``INSTRUCTION_SPEC`` and
    implemented on one backend but never compared against the other.
    """
    text = " ".join(case.asm for case in CASES).upper()
    missing = []
    for slot, instructions in INSTRUCTION_SPEC.items():
        for name in instructions:
            if name in UNCOVERED_OK:
                continue
            if name.upper() not in text:
                missing.append(f"{slot}/{name}")
    assert not missing, (
        "instructions with no ETISS parity coverage: " + ", ".join(sorted(missing))
    )
