import click
import ipu_as.lark_tree as lark_tree


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


cli.add_command(assemble)
cli.add_command(disassemble)

if __name__ == "__main__":
    cli()
