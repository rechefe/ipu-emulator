"""``ipu-gen``: write the artifacts generated from the ISA.

bazel run //src/tools/ipu-gen:ipu-gen -- sv-package --output /tmp/ipu_instr_pkg.sv
bazel run //src/tools/ipu-gen:ipu-gen -- docs-pages /tmp/docs
bazel run //src/tools/ipu-gen:ipu-gen -- instruction-format-md /tmp/instruction-format.md
bazel run //src/tools/ipu-gen:ipu-gen -- union-layout-svg /tmp/union_layout.svg
"""

from __future__ import annotations

from pathlib import Path

import click

from ipu_gen import docs_pages, layout_svg
from ipu_gen import sv_package as sv

#: The pages `docs-pages` writes, in the order //docs:generate_mkdocs_md lists them.
DOCS_PAGES = ("instructions.md", "operand-types.md", "assembly-syntax.md", "programmer-guide.md")


@click.group()
def cli() -> None:
    """Generators that read the instruction set (ipu_isa)."""


@cli.command("sv-package")
@click.option("--output", type=click.Path(), required=True, help="Output .sv file path")
def sv_package(output: str) -> None:
    """Generate the SystemVerilog package for the IPU instruction format."""
    sv.generate_sv_package(output)
    click.echo(f"Wrote SystemVerilog package to {output}")


@cli.command("docs-pages")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
def docs_pages_command(out_dir: Path) -> None:
    """Write the instruction, operand-type, assembly-syntax and programmer's guide pages."""
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_pages.generate_all_docs(*(out_dir / page for page in DOCS_PAGES))


@cli.command("instruction-format-md")
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
def instruction_format_md(output: Path) -> None:
    """Write the instruction-format page."""
    output.parent.mkdir(parents=True, exist_ok=True)
    docs_pages.generate_instruction_format_md(output)


@cli.command("union-layout-svg")
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
def union_layout_svg(output: Path) -> None:
    """Write the per-slot union layout SVG (slots MSB → LSB)."""
    svg = layout_svg.union_layout_svg()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg)
    click.echo(f"Wrote {len(svg):,} chars to {output}")


if __name__ == "__main__":
    cli()
