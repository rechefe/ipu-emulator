"""`assemble_asm(defines=, includes=)` assembles what the CLI does (E05-12)."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from ipu_as.cli import cli
from ipu_as.session import AssemblySession, instruction_aligned_bytes_len

HERE = Path("src/ipu-as/test/asm_rule")


def test_the_bazel_target_equals_the_cli_with_the_same_define_and_include(tmp_path):
    output = tmp_path / "cli.bin"
    argv = ["assemble", "--input", str(HERE / "template.asm"), "--output", str(output)]
    # Runfiles are symlinks; the include root is where the file really is.
    lib = (HERE / "lib" / "body.asm.j2").resolve().parent
    argv += ["--define", "cpr=3", "-I", str(lib)]
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, result.stdout
    built = Path("src/ipu-as/asm_rule_defines.bin").read_bytes()
    assert built == output.read_bytes()
    # Three SETs (cpr=3, not the header default 1) and the included HALT.
    assert len(built) == 4 * instruction_aligned_bytes_len()


def test_a_hardware_target_builds_the_hardware_words():
    source = (HERE / "hardware.asm").read_text()
    built = Path("src/ipu-as/asm_rule_hardware.bin").read_bytes()
    assert built == AssemblySession(source).assemble("hardware").to_bin()
