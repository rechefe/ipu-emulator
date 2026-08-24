import json
import sys

import click
from ipu_as import diagnostics, gen_codegen, lark_tree


@click.group()
def cli():
    pass


@click.command()
@click.option("--input", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option(
    "--format",
    prompt="Choose a file format",
    type=click.Choice(["mem", "bin"]),
    default="mem",
)
def assemble(input: click.Path, output: click.Path, format: str):
    """Assembles the given input file."""
    click.echo(f"Assembling file: {input}")
    if output:
        click.echo(f"Output will be saved to: {output}")
    if format == "mem":
        lark_tree.assemble_to_mem_file(open(input).read(), output)
    elif format == "bin":
        lark_tree.assemble_to_bin_file(open(input).read(), output)


@click.command()
@click.option("--input", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option(
    "--format",
    prompt="Choose a file format",
    type=click.Choice(["mem", "bin"]),
    default="mem",
)
def disassemble(input: click.Path, output: click.Path, format: str):
    """Disassembles the given input file."""
    click.echo(f"Disassembling file: {input}")
    if output:
        click.echo(f"Output will be saved to: {output}")
    if format == "mem":
        lark_tree.disassemble_from_mem_file(input, output)
    elif format == "bin":
        lark_tree.disassemble_from_bin_file(input, output)


@click.command("sv-package")
@click.option("--output", type=click.Path(), required=True, help="Output .sv file path")
def sv_package(output: str):
    """Generate a SystemVerilog package for the IPU instruction format."""
    gen_codegen.generate_sv_package(output)
    click.echo(f"Wrote SystemVerilog package to {output}")


@click.command()
@click.option(
    "--input",
    required=True,
    help='Source file to check, or "-" to read from stdin.',
)
@click.option(
    "--json/--text",
    "as_json",
    default=False,
    help="Emit machine-readable diagnostics (used by the VS Code extension).",
)
def check(input: str, as_json: bool):
    """Report errors in an assembly source without producing output.

    Unlike `assemble`, this never exits on the first problem and never writes a
    file — it reports position, so an editor can place a squiggle. Exit code is
    0 when the source assembles cleanly and 1 when it does not.

    Reading from stdin matters for the editor: it checks the buffer as typed,
    which is not what is on disk.
    """
    source = sys.stdin.read() if input == "-" else open(input).read()

    try:
        found = diagnostics.check(source)
    except Exception as error:  # noqa: BLE001 - see below
        # A crash in the checker must not read as "this program is fine". The
        # editor treats non-JSON output as a broken toolchain and clears the
        # squiggles, so an unhandled exception here silently hides real errors.
        # Report it as a diagnostic instead, and keep the exit code non-zero.
        import traceback

        traceback.print_exc(file=sys.stderr)
        found = [
            diagnostics.Diagnostic(
                line=0,
                column=0,
                end_line=0,
                end_column=0,
                message=(
                    f"internal error while checking this file: "
                    f"{type(error).__name__}: {error}"
                ),
                severity="error",
                stage="internal",
                approximate=True,
            )
        ]

    if as_json:
        # stdout is the protocol here; keep it pure JSON.
        json.dump([d.to_dict() for d in found], sys.stdout)
        sys.stdout.write("\n")
    else:
        for d in found:
            where = f"{input}:{d.line + 1}:{d.column + 1}"
            suffix = " (position approximate)" if d.approximate else ""
            click.echo(f"{where}: {d.severity}: {d.message}{suffix}", err=True)
        if not found:
            click.echo(f"{input}: ok")

    sys.exit(1 if found else 0)


cli.add_command(assemble)
cli.add_command(disassemble)
cli.add_command(sv_package)
cli.add_command(check)

if __name__ == "__main__":
    cli()
