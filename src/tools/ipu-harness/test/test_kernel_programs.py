"""The ISA-wide program checks over every kernel of the kernel tree (E07-25).

The standalone corpus tests in ipu-as-py (``test_assembly_round_trip``,
``test_slot_codec_corpus``, ``test_word_codec_corpus``,
``test_hardware_target``) run these checks on programs that assemble from
their file. A ``kernels/`` program renders only with a case, so here each
kernel's and template's default case is rendered in every format it runs in
(``kernel_bin.default_programs``, the program ``:assemble_<name>`` writes), and
the same checks (``program_checks``) run on the rendered text and its words.
"""

from __future__ import annotations

import pytest
from program_checks import (
    check_disassembly_round_trip,
    check_hardware_target,
    check_slot_codec,
    check_word_codec,
)

from ipu_as.session import AssemblySession
from ipu_harness.kernel_bin import default_programs
from ipu_harness.registry import Registry, kernel_tree

_REGISTRY = Registry.load(kernel_tree())
_KERNELS = sorted(
    [*_REGISTRY.manifests.values(), *_REGISTRY.templates.values()], key=lambda m: m.name
)

#: Each kernel's default case in each of its formats: (id, rendered program).
PROGRAMS = [
    pytest.param(program, id=f"{manifest.name}-{fmt.value}")
    for manifest in _KERNELS
    for fmt, program in default_programs(manifest).items()
]


def test_the_tree_has_kernels():
    assert not _REGISTRY.invalid, _REGISTRY.invalid
    assert len(_KERNELS) > 40 and len(PROGRAMS) > len(_KERNELS)


@pytest.mark.parametrize("program", PROGRAMS)
def test_the_rendered_text_is_the_program(program):
    """The checks below assemble the rendered text: it is the program's words."""
    assert tuple(AssemblySession(program.rendered).assemble().words) == tuple(program.words)


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
