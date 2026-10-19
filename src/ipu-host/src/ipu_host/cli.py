"""``ipu-host``: write the artifacts generated from the register maps.

bazel run //src/ipu-host:ipu-host -- sv OUT_DIR
bazel run //src/ipu-host:ipu-host -- docs OUT.md

Under ``bazel run`` a relative path is the caller's (``BUILD_WORKING_DIRECTORY``).
"""

from __future__ import annotations

import os
from pathlib import Path

import click

from ipu_host import generate


def _caller_path(path: Path) -> Path:
    """``path`` as the caller meant it: ``bazel run`` starts in the runfiles tree."""
    return Path(os.environ.get("BUILD_WORKING_DIRECTORY", ".")) / path


@click.group()
def cli() -> None:
    """Generators that read the register maps (ipu_host/rdl)."""


@cli.command("sv")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
def sv(out_dir: Path) -> None:
    """Write the SystemVerilog register blocks and their packages."""
    for path in generate.write_register_blocks(_caller_path(out_dir)):
        click.echo(f"Wrote {path}")


@cli.command("docs")
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
def docs(output: Path) -> None:
    """Write the register tables as Markdown."""
    path = _caller_path(output)
    path.write_text(generate.register_map_markdown())
    click.echo(f"Wrote {path}")


if __name__ == "__main__":
    cli()
