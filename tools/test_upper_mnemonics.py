"""tools/upper_mnemonics.py: upper case, binary-identical, templates untouched (E07-01)."""

from __future__ import annotations

from pathlib import Path

import asm_corpus
import pytest
import upper_mnemonics
from upper_mnemonics import upper_text

from ipu_as.jinja_tags import tag_spans

# Bazel runs the test from the runfiles root, where the corpus sits at its
# repo-relative paths.
ROOT = Path.cwd()
MANIFEST = asm_corpus.load_manifest(ROOT)
ENTRIES = asm_corpus.discover(ROOT)
PATHS = sorted({entry.path for entry in ENTRIES})


def test_mnemonics_and_registers_become_upper_case():
    text = "loop:\n    ld.mult r0 lr1 cr2 ;;\n    acc.add.first ; set lr0 cr0 ;;\n    b loop ;;\n"
    new, count = upper_text(text)
    assert new == (
        "loop:\n    LD.MULT R0 LR1 CR2 ;;\n    ACC.ADD.FIRST ; SET LR0 CR0 ;;\n    B loop ;;\n"
    )
    assert count == 9


def test_enum_values_labels_numbers_and_comments_are_kept():
    text = (
        "cr9: activate.quantize identity cr15 ; st.aaq lr1 cr3 ;;  # set lr0\n"
        "    mult.ve lr0 cr1 0x1f lr0 cr15 ;;  // add lr0\n"
        "    bz cr9 cr9 ;;\n"
    )
    new, _ = upper_text(text)
    assert new == (
        "cr9: ACTIVATE.QUANTIZE identity CR15 ; ST.AAQ LR1 CR3 ;;  # set lr0\n"
        "    MULT.VE LR0 CR1 0x1f LR0 CR15 ;;  // add lr0\n"
        # cr9 is a label of this file, so it keeps its spelling as an operand too.
        "    BZ cr9 cr9 ;;\n"
    )


def test_template_tags_and_tokens_touching_them_are_kept():
    text = (
        '{#- set lr0 -#}{%- set lr_row = "lr1" -%}\n'
        "{% for i in range(2) %}\n"
        "    add lr{{ i }} {{ lr_row }} cr1 ;;\n"
        "{% endfor %}\n"
        "    {{ op }} lr0 cr2 ;;\n"
        "    {{ block(0) }}\n"
        "    halt ;;\n"
        "    set lr0 cr0 ; # a comment {{ x }} set lr0\n"
        "    halt ;;\n"
    )
    new, _ = upper_text(text)
    assert new == (
        '{#- set lr0 -#}{%- set lr_row = "lr1" -%}\n'
        "{% for i in range(2) %}\n"
        "    ADD lr{{ i }} {{ lr_row }} CR1 ;;\n"
        "{% endfor %}\n"
        "    {{ op }} LR0 CR2 ;;\n"
        "    {{ block(0) }}\n"
        "    HALT ;;\n"
        "    SET LR0 CR0 ; # a comment {{ x }} set lr0\n"
        "    HALT ;;\n"
    )


def test_upper_case_text_is_unchanged():
    text = "    SET LR0 CR0 ;;\n    HALT ;;\n"
    assert upper_text(text) == (text, 0)


def test_check_reports_and_writes_nothing(tmp_path, capsys):
    source = tmp_path / "k.asm"
    source.write_text("    halt ;;\n")
    assert upper_mnemonics.main(["--check", str(source)]) == 1
    assert source.read_text() == "    halt ;;\n"
    assert upper_mnemonics.main([str(source)]) == 0
    assert source.read_text() == "    HALT ;;\n"
    assert upper_mnemonics.main(["--check", str(source)]) == 0
    assert "1 file(s) would change" in capsys.readouterr().out


# --- the kernel tree: every program already in upper case (E07-25) ---------------


def test_every_kernel_source_is_upper_case(capsys):
    sources = sorted((ROOT / "kernels").rglob("*.asm")) + sorted(
        (ROOT / "kernels").rglob("*.asm.j2")
    )
    assert len(sources) > 40, "the kernel tree is missing from the test data"
    assert upper_mnemonics.main(["--check", *map(str, sources)]) == 0, capsys.readouterr().out


# --- the whole corpus: a dry run, nothing written to the tree ----------------------


@pytest.fixture(scope="module")
def rewritten(tmp_path_factory) -> Path:
    """Every corpus program, rewritten, in a copy of its repo-relative layout."""
    root = tmp_path_factory.mktemp("upper")
    for path in PATHS:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(upper_text((ROOT / path).read_text())[0])
    return root


def test_the_rewrite_changes_the_lower_case_programs(rewritten):
    # The editor's corpus keeps lower-case programs (the assembler accepts any case).
    changed = [p for p in PATHS if (ROOT / p).read_text() != (rewritten / p).read_text()]
    assert changed


@pytest.mark.parametrize("path", PATHS)
def test_template_tags_are_byte_identical(rewritten, path):
    old, new = (ROOT / path).read_text(), (rewritten / path).read_text()
    assert len(old) == len(new)
    spans = tag_spans(old)
    assert [old[s.start : s.end] for s in spans] == [new[s.start : s.end] for s in tag_spans(new)]
    # Only case changes: the same text read case-insensitively; and a second run
    # finds nothing left to do.
    assert old.upper() == new.upper()
    assert upper_text(new)[1] == 0


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.key)
def test_every_binary_digest_is_unchanged(rewritten, entry):
    actual = asm_corpus.record(asm_corpus.assemble(rewritten, entry))
    assert asm_corpus.digest_mismatch(entry.key, MANIFEST[entry.key], actual) is None
