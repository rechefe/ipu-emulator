import functools
import json
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import click
import lark
import tomli

from ipu_as import diagnostics, formatter, language_service, lark_tree, session, template
from ipu_as.errors import AssemblerError, FormatError, ParseError, TemplateError
from ipu_isa.codec.slot import ReservedEncoding


def _exit_with(error: AssemblerError) -> NoReturn:
    """Print one line for a bad source and exit 1.

    The parse and assemble lines are the text the assembler always printed. A
    parse-stage error is chained to the lark exception it came from (the
    session raises it `from` that exception), and that exception's text is
    the one printed.
    """
    cause = error.__cause__
    if isinstance(error, TemplateError):
        click.echo(f"Error rendering template: {error}")
    elif isinstance(error, ParseError) or isinstance(cause, lark.exceptions.LarkError):
        reported = cause if isinstance(cause, lark.exceptions.LarkError) else error
        click.echo(f"Error parsing code: {reported}")
    else:
        click.echo(f"Error assembling code: {error}")
    sys.exit(1)


def _parse_defines(
    _ctx: click.Context, _param: click.Parameter, values: tuple[str, ...]
) -> dict[str, str]:
    """`KEY=VALUE` options as a mapping; the header's types coerce the strings."""
    defines: dict[str, str] = {}
    for value in values:
        key, sep, text = value.partition("=")
        if not sep or not key:
            raise click.BadParameter(f"{value!r} is not KEY=VALUE", param_hint="--define")
        defines[key] = text
    return defines


@dataclass(frozen=True)
class TemplateArgs:
    """The template inputs a command line gives.

    Attributes:
        defines: The `--define KEY=VALUE` values, as strings.
        include_dirs: The `-I` directories, in order.
    """

    defines: dict[str, str]
    include_dirs: tuple[str, ...]

    def search_path(self, input: str) -> tuple[Path, ...]:
        """The template search path for `input`.

        Args:
            input: The `--input` path, or `-` for stdin.

        Returns:
            The input's own directory (unless it is stdin), then each `-I` directory.
        """
        own = () if input == "-" else (Path(input).resolve().parent,)
        return own + tuple(Path(d) for d in self.include_dirs)


def _template_options(command: Callable) -> Callable:
    """Add `--define` and `-I` to `command`, which receives them as one `template_args`."""

    # functools.wraps shares command.__dict__, so the options click has already
    # attached to `command` (those written below this decorator) stay with it.
    @functools.wraps(command)
    def with_template(*args, defines: dict[str, str], include_dirs: tuple[str, ...], **kwargs):
        return command(*args, template_args=TemplateArgs(defines, include_dirs), **kwargs)

    with_template = click.option(
        "-I",
        "--include-dir",
        "include_dirs",
        multiple=True,
        type=click.Path(exists=True, file_okay=False),
        help="A directory {% include %} and {% import %} resolve in, after the input's "
        "own directory. Repeatable.",
    )(with_template)
    return click.option(
        "--define",
        "defines",
        multiple=True,
        callback=_parse_defines,
        metavar="KEY=VALUE",
        help="A template value, checked against the source's ipu:params header. Repeatable.",
    )(with_template)


_target_option = click.option(
    "--target",
    type=click.Choice(["emulator", "hardware"]),
    default="emulator",
    show_default=True,
    help="The word to assemble: every slot, as the emulator runs it, or the hardware "
    "slots only (an emulator-only instruction is then an error).",
)


@click.group()
def cli():
    pass


@click.command()
@click.option("--input", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option(
    "--format",
    type=click.Choice(["mem", "bin"]),
    default=None,
    help="Output format; by default bin for a .bin --output, else mem.",
)
@_template_options
@click.option(
    "--abi",
    "abi_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A kernel.toml whose [abi] table names the registers the template reads "
    "as {{ cr.<name> }} / {{ lr.<name> }}.",
)
@_target_option
def assemble(
    input: str,
    output: str,
    format: str | None,
    template_args: TemplateArgs,
    abi_path: str | None,
    target: str,
) -> None:
    """Assembles the given input file."""
    click.echo(f"Assembling file: {input}")
    if output:
        click.echo(f"Output will be saved to: {output}")
    with open(input) as source:
        text = source.read()
    abi = _read_abi(abi_path) if abi_path else None
    try:
        result = session.AssemblySession(
            text,
            defines=template_args.defines,
            search_path=template_args.search_path(input),
            abi=abi,
        ).assemble(target)
    except AssemblerError as error:
        _exit_with(error)
    if (format or _format_of(output)) == "bin":
        with open(output, "wb") as f:
            f.write(result.to_bin())
    else:
        with open(output, "w") as f:
            f.write(result.to_mem())


def _format_of(path: str) -> str:
    """The file format a path's suffix names: `bin` for `.bin`, else `mem`."""
    return "bin" if Path(path).suffix == ".bin" else "mem"


def _read_abi(path: str) -> dict[str, dict[str, str]]:
    """The `cr` / `lr` namespaces from the `[abi]` table of a kernel.toml (name = "CR2")."""
    try:
        with open(path, "rb") as f:
            table = tomli.load(f).get("abi")
    except (tomli.TOMLDecodeError, UnicodeDecodeError) as error:
        raise click.BadParameter(f"{path}: {error}", param_hint="--abi") from None
    if not isinstance(table, dict):
        raise click.BadParameter(f"{path} has no [abi] table", param_hint="--abi")
    try:
        return template.abi_namespaces(table)
    except TemplateError as error:
        raise click.BadParameter(f"{path}: {error}", param_hint="--abi") from None


@click.command()
@click.option("--input", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option(
    "--format",
    type=click.Choice(["mem", "bin"]),
    default=None,
    help="Input format; by default bin for a .bin --input, else mem.",
)
@_target_option
@click.option(
    "--syntax",
    type=click.Choice(lark_tree.SYNTAXES),
    default="fields",
    show_default=True,
    help="fields: every field of every slot, by word; assembly: source that assembles "
    "back to the same words.",
)
def disassemble(input: str, output: str, format: str | None, target: str, syntax: str) -> None:
    """Disassembles the given input file."""
    click.echo(f"Disassembling file: {input}")
    if output:
        click.echo(f"Output will be saved to: {output}")
    try:
        if (format or _format_of(input)) == "bin":
            lark_tree.disassemble_from_bin_file(input, output, target, syntax)
        else:
            lark_tree.disassemble_from_mem_file(input, output, target, syntax)
    except AssemblerError as error:
        _exit_with(error)
    except ReservedEncoding:
        # A slot holds an opcode no instruction has: most often a file of the
        # other target's words.
        click.echo(f"Error disassembling: {input} is not a file of {target} words")
        sys.exit(1)


#: The `check --json` versions this assembler can print; the contract is in
#: vscode-ipu-asm/schema/check-json.md.
JSON_VERSIONS = (1, 2)


def _print_json_versions(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if value and not ctx.resilient_parsing:
        click.echo(json.dumps(list(JSON_VERSIONS)))
        ctx.exit(0)


#: The `check --json --json-version 2` options that each add one analysis to the
#: output, by the key they add (`ipu_as.language_service.FEATURES`).
_ANALYSIS_FLAGS = {
    "tokens": "Add the semantic tokens of the source.",
    "words": "Add whether each word's instructions fit its slots.",
    "jinja": "Add the Jinja regions, with the names each defines and uses.",
}


def _analysis_options(command: Callable) -> Callable:
    """Add `--tokens`, `--words` and `--jinja` to `command`, which receives them as one `features`."""

    @functools.wraps(command)
    def with_features(*args, **kwargs):
        features = frozenset(name for name in _ANALYSIS_FLAGS if kwargs.pop(name))
        return command(*args, features=features, **kwargs)

    for name, text in reversed(_ANALYSIS_FLAGS.items()):
        with_features = click.option(f"--{name}", is_flag=True, help=text + " Needs v2.")(
            with_features
        )
    return with_features


def _internal_error(error: Exception) -> diagnostics.Diagnostic:
    """The diagnostic for a crash in the checker itself."""
    return diagnostics.Diagnostic(
        line=0,
        column=0,
        end_line=0,
        end_column=0,
        message=f"internal error while checking this file: {type(error).__name__}: {error}",
        severity="error",
        stage="internal",
        approximate=True,
    )


def _analyse(
    source: str, input: str, template_args: TemplateArgs, target: str, features: frozenset[str]
) -> tuple[list[diagnostics.Diagnostic], dict[str, list[dict]]]:
    """The diagnostics of `source`, and each of `features` as its JSON key and value.

    A crash in the checker must not read as "this program is fine". The editor
    treats non-JSON output as a broken toolchain and clears the squiggles, so
    an unhandled exception here would silently hide real errors. It is
    reported as an `internal` diagnostic instead, without the features.
    """
    try:
        if not features:
            found = diagnostics.check(
                source,
                defines=template_args.defines,
                search_path=template_args.search_path(input),
                target=target,
            )
            return found, {}
        analysis = language_service.analyse(
            source,
            defines=template_args.defines,
            search_path=template_args.search_path(input),
            target=target,
            want=features | {"diagnostics"},
        )
        extra = analysis.to_dict()
    except Exception as error:  # noqa: BLE001 - see the docstring
        traceback.print_exc(file=sys.stderr)
        return [_internal_error(error)], {}
    extra.pop("diagnostics")
    return list(analysis.diagnostics or ()), extra


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
@click.option(
    "--json-version",
    type=click.IntRange(min(JSON_VERSIONS), max(JSON_VERSIONS)),
    help="Which --json layout to print: 1 (the default) is a bare array of "
    'diagnostics, 2 is {"schemaVersion": 2, "diagnostics": [...]}.',
)
@click.option(
    "--json-versions",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_print_json_versions,
    help="Print the --json versions this assembler supports, as a JSON array, and exit.",
)
@_template_options
@_target_option
@_analysis_options
def check(
    input: str,
    as_json: bool,
    json_version: int | None,
    template_args: TemplateArgs,
    target: str,
    features: frozenset[str],
):
    """Report errors in an assembly source without producing output.

    Unlike `assemble`, this never exits on the first problem and never writes a
    file — it reports position, so an editor can place a squiggle. Exit code is
    0 when the source assembles cleanly and 1 when it does not.

    Reading from stdin matters for the editor: it checks the buffer as typed,
    which is not what is on disk.

    With `--json --json-version 2`, `--tokens`, `--words` and `--jinja` each add
    one more analysis of the same source to the output, under its own key.
    """
    if json_version is not None and not as_json:
        raise click.UsageError("--json-version applies to --json output only")
    if features and json_version != 2:
        raise click.UsageError("--tokens, --words and --jinja need --json --json-version 2")
    source = sys.stdin.read() if input == "-" else open(input).read()
    found, extra = _analyse(source, input, template_args, target, features)

    if as_json:
        # stdout is the protocol here; keep it pure JSON.
        listed = [d.to_dict() for d in found]
        json.dump(
            listed
            if json_version in (None, 1)
            else {"schemaVersion": 2, "diagnostics": listed, **extra},
            sys.stdout,
        )
        sys.stdout.write("\n")
    else:
        for d in found:
            where = f"{input}:{d.line + 1}:{d.column + 1}"
            suffix = " (position approximate)" if d.approximate else ""
            click.echo(f"{where}: {d.severity}: {d.message}{suffix}", err=True)
        if not found:
            click.echo(f"{input}: ok")

    sys.exit(1 if found else 0)


def _parse_range(
    _ctx: click.Context, _param: click.Parameter, value: str | None
) -> tuple[int, int] | None:
    """`L1:L2` (1-based, inclusive) as the 0-based (first, last) `format_source` takes."""
    if value is None:
        return None
    first, sep, last = value.partition(":")
    if not (sep and first.isdigit() and last.isdigit() and 1 <= int(first) <= int(last)):
        raise click.BadParameter(f"{value!r} is not L1:L2 with 1 <= L1 <= L2", param_hint="--range")
    return int(first) - 1, int(last) - 1


def _parse_indent(_ctx: click.Context, _param: click.Parameter, value: str) -> str:
    """`N` spaces, or `tab`, as the indent string."""
    if value == "tab":
        return "\t"
    if not value.isdigit():
        raise click.BadParameter(
            f"{value!r} is not a number of spaces or 'tab'", param_hint="--indent"
        )
    return " " * int(value)


#: `format`'s exit code when it cannot format the input (refused, or not UTF-8).
_CANNOT_FORMAT = 3


@click.command("format")
@click.option(
    "--input",
    required=True,
    type=click.Path(exists=True, dir_okay=False, allow_dash=True),
    help='Source file to format, or "-" to read from stdin.',
)
@click.option(
    "--check",
    is_flag=True,
    help="Write nothing; exit 1 if formatting would change the source, 0 if not.",
)
@click.option(
    "--in-place",
    is_flag=True,
    help="Rewrite the input file instead of printing the result.",
)
@click.option(
    "--indent",
    default="4",
    show_default=True,
    callback=_parse_indent,
    metavar="N|tab",
    help="One level of indentation: N spaces, or a tab.",
)
@click.option(
    "--blank-line",
    is_flag=True,
    help="Separate each ;; word from the next by an empty line.",
)
@click.option(
    "--range",
    "line_range",
    callback=_parse_range,
    metavar="L1:L2",
    help="Format only lines L1 to L2 (1-based, inclusive); the others stay as they are.",
)
def format_command(
    input: str,
    check: bool,
    in_place: bool,
    indent: str,
    blank_line: bool,
    line_range: tuple[int, int] | None,
):
    """Print an assembly source in the repository's layout.

    A label at column 0 on its own line, each instruction line one indent in,
    and code after a `;;` on a line of its own. Only whitespace changes; Jinja
    tags and comments stay as written. The result is checked to be the same
    program (the same words for a source without Jinja, the same tokens for a
    template); if it is not, or the input is not UTF-8, nothing is written
    and the exit code is 3, so a `--check` gate tells "cannot format" (3)
    from "would reformat" (1).
    """
    if input == "-" and in_place:
        raise click.UsageError("--in-place needs a file --input, not stdin")
    if check and in_place:
        raise click.UsageError("--check writes nothing; drop --in-place")
    try:
        if input == "-":
            text = sys.stdin.buffer.read().decode("utf-8")
        else:
            # newline="": a CRLF file is formatted, and written back, as CRLF.
            with open(input, encoding="utf-8", newline="") as source:
                text = source.read()
        out = formatter.format_source(
            text, indent=indent, blank_line=blank_line, line_range=line_range
        )
    except (FormatError, UnicodeDecodeError) as error:
        click.echo(f"Error formatting {input}: {error}", err=True)
        sys.exit(_CANNOT_FORMAT)
    if check:
        if out != text:
            click.echo(f"{input}: would be reformatted", err=True)
        sys.exit(1 if out != text else 0)
    if not in_place:
        sys.stdout.buffer.write(out.encode("utf-8"))
    elif out != text:
        with open(input, "w", encoding="utf-8", newline="") as target:
            target.write(out)


cli.add_command(assemble)
cli.add_command(disassemble)
cli.add_command(check)
cli.add_command(format_command)

if __name__ == "__main__":
    cli()
