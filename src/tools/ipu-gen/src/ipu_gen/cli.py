"""``ipu-gen``: write the artifacts generated from the ISA.

bazel run //src/tools/ipu-gen:ipu-gen -- sv-package --output /tmp/ipu_instr_pkg.sv
"""

from __future__ import annotations

import click

from ipu_gen import sv_package as sv


@click.group()
def cli() -> None:
    """Generators that read the instruction set (ipu_isa)."""


@cli.command("sv-package")
@click.option("--output", type=click.Path(), required=True, help="Output .sv file path")
def sv_package(output: str) -> None:
    """Generate the SystemVerilog package for the IPU instruction format."""
    sv.generate_sv_package(output)
    click.echo(f"Wrote SystemVerilog package to {output}")


if __name__ == "__main__":
    cli()
