"""`ipu-as` runs without prompting and takes template values (E05-10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from ipu_as import session
from ipu_as.cli import cli

TEMPLATE = """{#- ipu:params
cpr:  int = 3
rows: int = 8
-#}
{% for i in range(cpr * rows) %}SET LR{{ i % 16 }} CR2;;
{% endfor %}HALT;;
"""


def _invoke(*argv: str) -> Result:
    # input="" closes stdin: a prompt would abort instead of waiting.
    return CliRunner().invoke(cli, list(argv), input="")


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_assemble_infers_bin_from_the_output_suffix(tmp_path):
    source = _write(tmp_path / "x.asm", "SET LR0 CR2;;\nHALT;;\n")
    inferred, explicit = tmp_path / "x.bin", tmp_path / "explicit.out"
    assert _invoke("assemble", "--input", source, "--output", str(inferred)).exit_code == 0
    result = _invoke("assemble", "--input", source, "--output", str(explicit), "--format", "bin")
    assert result.exit_code == 0
    assert inferred.read_bytes() == explicit.read_bytes()


def test_assemble_writes_mem_for_any_other_suffix(tmp_path):
    source = _write(tmp_path / "x.asm", "HALT;;\n")
    output = tmp_path / "x.mem"
    assert _invoke("assemble", "--input", source, "--output", str(output)).exit_code == 0
    assert output.read_text() == session.assemble("HALT;;\n").to_mem()


def test_defines_render_like_the_same_values_given_to_the_assembler(tmp_path):
    source = _write(tmp_path / "k.asm", TEMPLATE)
    output = tmp_path / "k.bin"
    argv = ["assemble", "--input", source, "--output", str(output), "--define", "cpr=2"]
    result = _invoke(*argv, "--define", "rows=5")
    assert result.exit_code == 0, result.stdout
    assert output.read_bytes() == session.assemble(TEMPLATE, defines={"cpr": 2, "rows": 5}).to_bin()


def test_defines_equal_to_the_header_defaults_change_nothing(tmp_path):
    source = _write(tmp_path / "k.asm", TEMPLATE)
    output = tmp_path / "k.bin"
    argv = ["assemble", "--input", source, "--output", str(output)]
    assert _invoke(*argv, "--define", "cpr=3", "--define", "rows=8").exit_code == 0
    assert output.read_bytes() == session.assemble(TEMPLATE).to_bin()


def test_an_undeclared_define_is_a_template_error(tmp_path):
    source = _write(tmp_path / "k.asm", TEMPLATE)
    output = str(tmp_path / "k.bin")
    result = _invoke("assemble", "--input", source, "--output", output, "--define", "x=1")
    assert result.exit_code == 1
    assert "Error rendering template: undeclared define 'x'" in result.stdout


@pytest.mark.parametrize("define", ["rows", "=8"])
def test_a_define_without_a_key_and_value_is_a_usage_error(tmp_path, define):
    source = _write(tmp_path / "k.asm", TEMPLATE)
    output = str(tmp_path / "k.bin")
    result = _invoke("assemble", "--input", source, "--output", output, "--define", define)
    assert result.exit_code == 2
    assert "is not KEY=VALUE" in result.stderr


def test_check_uses_the_defines(tmp_path):
    # Without a define, `op` renders as nothing and the line does not parse.
    source = _write(tmp_path / "k.asm", "{#- ipu:params\nop: str = ''\n-#}\n{{ op }} LR0 CR2;;\n")
    without = _invoke("check", "--json", "--input", source)
    assert without.exit_code == 1
    result = _invoke("check", "--json", "--input", source, "--define", "op=SET")
    assert (result.exit_code, json.loads(result.stdout)) == (0, [])


def test_includes_resolve_in_the_input_directory_then_each_include_dir(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    _write(tmp_path / "own.inc", "SET LR0 CR2;;\n")
    _write(lib / "shared.inc", "HALT;;\n")
    source = _write(tmp_path / "k.asm", '{% include "own.inc" %}{% include "shared.inc" %}')
    output = tmp_path / "k.bin"
    result = _invoke("assemble", "--input", source, "--output", str(output), "-I", str(lib))
    assert result.exit_code == 0, result.stdout
    assert output.read_bytes() == session.assemble("SET LR0 CR2;;\nHALT;;\n").to_bin()
    checked = _invoke("check", "--input", source, "--include-dir", str(lib))
    assert checked.exit_code == 0, checked.stderr


def test_disassemble_infers_the_format_from_the_input_suffix(tmp_path):
    binary = tmp_path / "x.bin"
    binary.write_bytes(session.assemble("SET LR0 CR2;;\nHALT;;\n").to_bin())
    inferred, explicit = tmp_path / "inferred.asm", tmp_path / "explicit.asm"
    assert _invoke("disassemble", "--input", str(binary), "--output", str(inferred)).exit_code == 0
    argv = ["disassemble", "--input", str(binary), "--output", str(explicit), "--format", "bin"]
    assert _invoke(*argv).exit_code == 0
    assert inferred.read_text() == explicit.read_text()
