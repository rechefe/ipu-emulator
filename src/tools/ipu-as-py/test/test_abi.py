"""`cr` / `lr` ABI namespaces in the render context (E05-13)."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from ipu_as import template
from ipu_as.cli import cli
from ipu_as.errors import TemplateError
from ipu_as.session import AssemblySession
from ipu_as.template import TemplateInputs

ABI = {"cr": {"data_base": "CR2"}, "lr": {"data_ptr": "LR0"}}
SOURCE = "SET {{ lr.data_ptr }} {{ cr.data_base }};;\nBKPT;;\n"
LITERAL = "SET LR0 CR2;;\nBKPT;;\n"


def test_names_render_to_their_registers_and_assemble_like_literals() -> None:
    assert template.render(SOURCE, TemplateInputs(namespaces=ABI)) == LITERAL.rstrip("\n")
    words = AssemblySession(SOURCE, abi=ABI).assemble().words
    assert words == AssemblySession(LITERAL).assemble().words


def test_two_names_may_name_the_same_register() -> None:
    abi = {"cr": {"a": "CR2", "b": "CR2"}}
    assert template.render("{{ cr.a }} {{ cr.b }}", TemplateInputs(namespaces=abi)) == "CR2 CR2"


def test_an_unknown_name_fails_at_its_template_line() -> None:
    with pytest.raises(TemplateError, match=r"unknown ABI name cr\.data_bse") as info:
        template.render("NOP;;\nSET LR0 {{ cr.data_bse }};;\n", TemplateInputs(namespaces=ABI))
    assert info.value.location.line == 2


@pytest.mark.parametrize(
    "abi",
    [{"cr": {"x": "CR16"}}, {"cr": {"x": "R0"}}, {"cr": {"x": "LR2"}}, {"lr": {"x": "LR16"}}],
)
def test_a_value_that_is_not_a_register_of_its_namespace_fails_at_render_start(abi) -> None:
    with pytest.raises(TemplateError, match="is not a register") as info:
        template.render("NOP;;\n", TemplateInputs(namespaces=abi))
    assert info.value.location.line == 1


def test_an_unknown_namespace_is_rejected() -> None:
    with pytest.raises(TemplateError, match="unknown ABI namespace 'r'"):
        template.render("NOP;;", TemplateInputs(namespaces={"r": {"x": "R0"}}))


def test_cr_and_lr_are_reserved_names() -> None:
    with pytest.raises(TemplateError, match="define 'cr' is reserved"):
        template.render("{{ x }}", TemplateInputs(defines={"cr": 1}))
    with pytest.raises(TemplateError, match="parameter 'lr' is reserved"):
        template.render("{#- ipu:params\nlr: int = 1\n-#}{{ lr }}")


def test_cli_reads_the_abi_table_of_a_kernel_toml(tmp_path: Path) -> None:
    (tmp_path / "kernel.toml").write_text(
        '[kernel]\nop = "x"\n\n[abi]\ndata_base = "CR2"\ndata_ptr = "LR0"\n'
    )
    (tmp_path / "prog.asm").write_text(SOURCE)
    out = tmp_path / "prog.bin"
    result = CliRunner().invoke(
        cli,
        [
            "assemble",
            "--input",
            str(tmp_path / "prog.asm"),
            "--output",
            str(out),
            "--format",
            "bin",
            "--abi",
            str(tmp_path / "kernel.toml"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.read_bytes() == AssemblySession(LITERAL).assemble().to_bin()


def test_cli_rejects_an_abi_value_that_is_not_a_register(tmp_path: Path) -> None:
    (tmp_path / "kernel.toml").write_text('[abi]\nx = "R0"\n')
    (tmp_path / "prog.asm").write_text("NOP;;\n")
    result = CliRunner().invoke(
        cli,
        [
            "assemble",
            "--input",
            str(tmp_path / "prog.asm"),
            "--output",
            str(tmp_path / "o"),
            "--format",
            "bin",
            "--abi",
            str(tmp_path / "kernel.toml"),
        ],
    )
    assert result.exit_code == 2
    assert "is not a CR or LR register" in result.output
