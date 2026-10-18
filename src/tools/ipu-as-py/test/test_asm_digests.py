"""The assembled `.bin` and `.mem` of every program match the recorded digests.

The manifest (``test/baselines/asm_digests.json``) is today's output. A refactor
of the assembler, the ISA tables or the templates must leave every digest
unchanged; a PR that changes assembled output on purpose regenerates the
manifest with ``bazel run //src/tools/ipu-as-py:update_asm_digests`` and says
why. The corpus is defined in ``src/tools/ipu-as-py/test/corpus/asm_corpus.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import asm_corpus
import pytest

# Bazel runs the test from the runfiles root of the main repository, where the
# corpus files and the manifest sit at their repo-relative paths.
ROOT = Path.cwd()
MANIFEST = asm_corpus.load_manifest(ROOT)
ENTRIES = asm_corpus.discover(ROOT)

_PROGRAM = "vscode-ipu-asm/test/integration/fixture/kernels/demo/demo_kernel/demo_kernel.asm"
_IMMEDIATE = "MULT.VE {{lr_zero}} cr1 0 "


def test_manifest_lists_exactly_the_corpus():
    drift = asm_corpus.corpus_drift(set(MANIFEST), {entry.key for entry in ENTRIES})
    assert drift is None, drift


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.key)
def test_assembled_output_matches_digest(entry):
    if entry.key not in MANIFEST:
        pytest.fail(f"{entry.key} has no recorded digest; run {asm_corpus.UPDATE_COMMAND}")
    actual = asm_corpus.record(asm_corpus.assemble(ROOT, entry))
    mismatch = asm_corpus.digest_mismatch(entry.key, MANIFEST[entry.key], actual)
    assert mismatch is None, mismatch


def test_every_render_context_is_in_the_corpus():
    keys = {entry.key for entry in ENTRIES}
    for path, contexts in asm_corpus.RENDER_CONTEXTS.items():
        assert (ROOT / path).is_file(), f"RENDER_CONTEXTS names a missing program: {path}"
        for context in contexts:
            assert asm_corpus.CorpusEntry.rendered(path, context).key in keys


# The baseline check itself, on a temporary corpus.


def _copy_into(tmp_path: Path, path: str) -> Path:
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / path, target)
    return target


def test_planted_immediate_change_fails_and_names_the_file(tmp_path):
    asm = _copy_into(tmp_path, _PROGRAM)
    entry = asm_corpus.CorpusEntry(_PROGRAM)
    expected = MANIFEST[entry.key]
    assert (
        asm_corpus.digest_mismatch(
            entry.key, expected, asm_corpus.record(asm_corpus.assemble(tmp_path, entry))
        )
        is None
    )

    # Change one immediate: the MULT.VE mask offset, 0 -> 1.
    source = asm.read_text()
    assert source.count(_IMMEDIATE) == 1
    asm.write_text(source.replace(_IMMEDIATE, _IMMEDIATE.replace(" 0 ", " 1 ")))

    actual = asm_corpus.record(asm_corpus.assemble(tmp_path, entry))
    mismatch = asm_corpus.digest_mismatch(entry.key, expected, actual)
    assert mismatch is not None
    assert _PROGRAM in mismatch
    assert "first differing word: " in mismatch
    assert asm_corpus.UPDATE_COMMAND in mismatch


def test_unlisted_program_fails_and_names_the_update_command(tmp_path):
    _copy_into(tmp_path, _PROGRAM)
    extra = "vscode-ipu-asm/test/integration/fixture/kernels/demo/new_kernel/new_kernel.asm"
    (tmp_path / extra).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / _PROGRAM, tmp_path / extra)

    corpus = {entry.key for entry in asm_corpus.discover(tmp_path)}
    drift = asm_corpus.corpus_drift({_PROGRAM}, corpus)
    assert drift is not None
    assert extra in drift
    assert asm_corpus.UPDATE_COMMAND in drift


def test_removed_program_fails():
    drift = asm_corpus.corpus_drift({_PROGRAM, "gone.asm"}, {_PROGRAM})
    assert drift is not None and "gone.asm" in drift
