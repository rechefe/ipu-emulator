"""The ISA-wide program checks over every kernel of the kernel tree (E07-25).

The standalone corpus tests in ipu-as (``test_assembly_round_trip``,
``test_slot_codec_corpus``, ``test_word_codec_corpus``,
``test_hardware_target``) run these checks on programs that assemble from
their file. A ``kernels/`` program renders only with a case, so here each
kernel's and template's default case is rendered in every format it runs in
(``kernel_bin.default_stage_programs``: the program ``:assemble_<name>``
writes, and a multi-stage kernel's every stage, E06-14), and the same checks
(``program_checks``) run on the rendered text and its words.
"""

from __future__ import annotations

import warnings

import pytest
from program_checks import (
    assembles_for_hardware,
    check_disassembly_round_trip,
    check_hardware_target,
    check_slot_codec,
    check_word_codec,
)

from ipu_as.session import AssemblySession
from ipu_harness.kernel_bin import default_programs, default_stage_programs
from ipu_harness.registry import Registry, kernel_tree

_REGISTRY = Registry.load(kernel_tree())
_KERNELS = sorted(
    [*_REGISTRY.manifests.values(), *_REGISTRY.templates.values()], key=lambda m: m.name
)

#: Each kernel's default case in each of its formats, each stage's program for
#: a multi-stage kernel: (id, rendered program).
PROGRAMS = [
    pytest.param(
        program,
        id=f"{manifest.name}-{fmt.value}" + (f"-{i}" if len(stages) > 1 else ""),
    )
    for manifest in _KERNELS
    for fmt, stages in default_stage_programs(manifest).items()
    for i, (_, program) in enumerate(stages)
]


def test_the_tree_has_kernels():
    assert not _REGISTRY.invalid, _REGISTRY.invalid
    assert len(_KERNELS) > 40 and len(PROGRAMS) > len(_KERNELS)


def test_a_multi_stage_kernel_contributes_every_stage():
    """A multi-stage kernel (``[kernel] stages``) runs more programs than its
    default one: each is checked, and ``:assemble_<name>`` writes its own."""
    staged = [m for m in _KERNELS if m.stages is not None]
    assert staged
    for manifest in staged:
        for fmt, stages in default_stage_programs(manifest).items():
            assert len(stages) > 1, manifest.name
            own = [program for name, program in stages if name == manifest.asm.name]
            assert own and default_programs(manifest)[fmt].words == own[-1].words


def test_the_nx8_tour_is_a_hardware_program():
    # FP32 kernels store through the emulator-only ACTIVATE.EXACT (ADR 0007), so
    # only NX8 programs without BREAK assemble for hardware; vector_tour is one.
    tours = [p for p in PROGRAMS if p.id.startswith("vector_tour-")]
    assert tours and all(assembles_for_hardware(p.values[0].rendered) for p in tours)


@pytest.mark.parametrize("program", PROGRAMS)
def test_the_rendered_text_is_the_program(program):
    """The checks below assemble the rendered text: it is the program's words."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*-only .* slot")
        words = AssemblySession(program.rendered).assemble().words
    assert tuple(words) == tuple(program.words)


@pytest.mark.parametrize("target", ["emulator", "hardware"])
@pytest.mark.parametrize("program", PROGRAMS)
def test_the_disassembly_assembles_to_the_same_words(program, target):
    if not check_disassembly_round_trip(program.rendered, target):
        pytest.skip("an emulator-only instruction: no hardware words")


@pytest.mark.parametrize("program", PROGRAMS)
def test_the_slot_and_word_codecs_reproduce_the_program(program):
    check_slot_codec(program.words)
    check_word_codec(program.words, program.to_bin())


@pytest.mark.parametrize("program", PROGRAMS)
def test_the_hardware_words_are_the_emulator_words_top_bits(program):
    check_hardware_target(program.rendered)
