"""diagnostics.check runs the AssemblySession stages (E05-07)."""

from __future__ import annotations

import json

from click.testing import CliRunner

from ipu_as import diagnostics, encode
from ipu_as.cli import cli


def test_a_plain_value_error_in_the_encoder_is_internal_not_assemble(monkeypatch) -> None:
    def broken(compound, target):
        raise ValueError("a bug in the encoder")

    monkeypatch.setattr(encode, "encode_word", broken)
    result = CliRunner().invoke(cli, ["check", "--json", "--input", "-"], input="NOP;;\n")
    (found,) = json.loads(result.stdout)
    assert found["stage"] == "internal"
    assert "a bug in the encoder" in found["message"]


def test_consecutive_checks_with_the_same_labels_do_not_interfere() -> None:
    source = "loop:\nNOP;;\nB loop;;\n"
    assert diagnostics.check(source) == []
    assert diagnostics.check(source) == []


def test_defines_reach_the_template() -> None:
    source = "{#- ipu:params\nrows: int\n-#}\n{% for _ in range(rows) %}NOP;;\n{% endfor %}BKPT;;\n"
    assert diagnostics.check(source, defines={"rows": 2}) == []
    (found,) = diagnostics.check(source)
    assert found.stage == "template"
    assert "missing required define 'rows'" in found.message
