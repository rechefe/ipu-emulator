"""``ipu-gen``: write the artifacts generated from the ISA.

bazel run //src/tools/ipu-gen:ipu-gen -- sv-package --output /tmp/ipu_instr_pkg.sv
bazel run //src/tools/ipu-gen:ipu-gen -- docs-pages /tmp/docs
bazel run //src/tools/ipu-gen:ipu-gen -- instruction-format-md /tmp/instruction-format.md
bazel run //src/tools/ipu-gen:ipu-gen -- union-layout-svg /tmp/union_layout.svg
bazel run //src/tools/ipu-gen:ipu-gen -- vscode-data [OUT_DIR]
bazel run //src/tools/ipu-gen:ipu-gen -- verify-sv-package path/to/ipu_instr_pkg.sv
bazel run //src/tools/ipu-gen:ipu-gen -- inspector-trace TRACE CONTEXT OUTPUT [--max-cycles N]

Under ``bazel run`` a relative path is the caller's (``BUILD_WORKING_DIRECTORY``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

from ipu_gen import docs_pages, inspector_trace, layout_svg, vscode_data
from ipu_gen import sv_package as sv


def _caller_path(path: Path) -> Path:
    """``path`` as the caller meant it: ``bazel run`` starts in the runfiles tree."""
    return Path(os.environ.get("BUILD_WORKING_DIRECTORY", ".")) / path


@click.group()
def cli() -> None:
    """Generators that read the instruction set (ipu_isa)."""


@cli.command("sv-package")
@click.option("--output", type=click.Path(), required=True, help="Output .sv file path")
def sv_package(output: str) -> None:
    """Generate the SystemVerilog package for the IPU instruction format."""
    path = _caller_path(Path(output))
    sv.generate_sv_package(path)
    click.echo(f"Wrote SystemVerilog package to {path}")


@cli.command("docs-pages")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
def docs_pages_command(out_dir: Path) -> None:
    """Write the instruction, operand-type, assembly-syntax and programmer's guide pages."""
    out_dir = _caller_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_pages.generate_all_docs(
        instructions_path=out_dir / "instructions.md",
        operand_types_path=out_dir / "operand-types.md",
        assembly_syntax_path=out_dir / "assembly-syntax.md",
        programmer_guide_path=out_dir / "programmer-guide.md",
    )


@cli.command("instruction-format-md")
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
def instruction_format_md(output: Path) -> None:
    """Write the instruction-format page."""
    output = _caller_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    docs_pages.generate_instruction_format_md(output)


@cli.command("union-layout-svg")
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
def union_layout_svg(output: Path) -> None:
    """Write the per-slot union layout SVG (slots MSB → LSB)."""
    svg = layout_svg.union_layout_svg()
    output = _caller_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg)
    click.echo(f"Wrote {len(svg):,} chars to {output}")


@cli.command("vscode-data")
@click.argument("out_dir", required=False, type=click.Path(file_okay=False, path_type=Path))
def vscode_data_command(out_dir: Path | None) -> None:
    """Write the TextMate grammar, language-configuration.json and data/isa.json.

    Without OUT_DIR, into vscode-ipu-asm/ of the workspace `bazel run` started in.
    """
    if out_dir is None:
        workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
        if not workspace:
            raise click.UsageError(
                "No output directory given and BUILD_WORKSPACE_DIRECTORY is unset. "
                "Run via `bazel run //vscode-ipu-asm:gen_vscode`, or pass a directory."
            )
        out_dir = Path(workspace) / "vscode-ipu-asm"
    vscode_data.generate_all(_caller_path(out_dir), vscode_data.REGISTER_FACTS)


@cli.command("inspector-trace")
@click.argument("trace", type=click.Path(dir_okay=False, exists=True, path_type=Path))
@click.argument("context", type=click.Path(dir_okay=False, exists=True, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--max-cycles",
    type=click.IntRange(min=1),
    default=inspector_trace.DEFAULT_MAX_CYCLES,
    show_default=True,
    help="Keep at most this many cycles; the document says it is truncated.",
)
def inspector_trace_command(trace: Path, context: Path, output: Path, max_cycles: int) -> None:
    """Write the Cycle Inspector's JSON for one recorded case (E08-16).

    TRACE and CONTEXT are what //src/tools/ipu-apps:trace_case wrote.
    """
    try:
        header, records, ctx = inspector_trace.read_inputs(
            _caller_path(trace), _caller_path(context)
        )
        document = inspector_trace.convert(header, records, ctx, max_cycles=max_cycles)
    except inspector_trace.TraceError as error:
        raise click.ClickException(str(error)) from None
    _caller_path(output).write_text(json.dumps(document, separators=(",", ":")) + "\n", "utf-8")


@cli.command("verify-sv-package")
@click.argument("path", type=click.Path(dir_okay=False, path_type=Path))
def verify_sv_package(path: Path) -> None:
    """Check a vendored ipu_instr_pkg.sv against the content hash in its header.

    Exits 0 when the body is exactly what the header's hash names, 1 when it was edited.
    """
    path = _caller_path(path)
    # Bytes, not text mode: a copy with its line endings rewritten is not unmodified.
    if sv.verify_sv_package(path.read_bytes().decode("utf-8")):
        click.echo(f"{path}: unmodified")
        return
    click.echo(f"{path}: does not match the content hash in its header", err=True)
    sys.exit(1)


if __name__ == "__main__":
    cli()
