import click


@click.group()
def cli():
    pass


@click.command()
@click.option("--input", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path(exists=False), required=True)
def assemble(input: click.Path, output: click.Path):
    """Assembles the given input file."""
    click.echo(f"Assembling file: {input}")
    if output:
        click.echo(f"Output will be saved to: {output}")
    with open(output, "w") as f:
        f.write(f"Assembled content from {input}")


cli.add_command(assemble)


if __name__ == "__main__":
    cli()
