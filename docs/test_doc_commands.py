"""Every documented `ipu-as` command uses only options the CLI declares, and so
does every documented `bazel run` of a kernel target (`//kernels/...`),
`run_kernel` or `query`.

The options are read from the click command and the argparse parsers
themselves, so the test follows each CLI as it changes.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import click
import pytest

from docs.doc_files import doc_files
from ipu_as.cli import cli

# `ipu-as <sub>` or `ipu-as -- <sub>` (as after `bazel run ...:ipu-as`); not
# `ipu-as` or `ipu-asm`.
_INVOCATION = re.compile(r"\bipu-as(?![\w-])(?:[ \t]+--)?[ \t]+(?P<rest>[a-z].*)$")
_SHELL_OPERATORS = {"|", "||", "&&", ";", ">", ">>", "<", "&", "2>&1"}


# `bazel run [--flag ...] //label [-- args]`; a `//kernels/...` label may leave
# out its target name (`//kernels/elementwise/scale`).
_BAZEL_RUN = re.compile(
    r"\bbazel run(?:[ \t]+--?[\w=.-]+)*[ \t]+"
    r"(?P<label>//[\w/.-]+:[\w.-]+|//kernels/[\w/.-]+)(?P<rest>.*)$"
)
#: An ipu_kernel() or registry entry target runs the harness CLI with its
#: kernel filled in (E06-13, E06-17).
_KERNELS = "//kernels/"
_RUN_KERNEL = "//src/ipu-harness:run_kernel"


@dataclass(frozen=True)
class Command:
    path: str
    line: int
    argv: list[str]


def _logical_lines(path: Path) -> list[tuple[int, str]]:
    """Lines of `path` with backslash continuations joined, as (1-based line, text)."""
    result: list[tuple[int, str]] = []
    pending: tuple[int, str] | None = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        start, text = pending if pending else (number, "")
        if line.rstrip().endswith("\\"):
            pending = (start, text + line.rstrip()[:-1] + " ")
            continue
        result.append((start, text + line))
        pending = None
    if pending:
        result.append(pending)
    return result


def _argv(rest: str) -> list[str]:
    """The arguments in `rest`, up to an inline code span's end or a shell operator."""
    rest = rest.split("`", 1)[0]
    try:
        tokens = shlex.split(rest, comments=True)
    except ValueError:  # an unbalanced quote in prose after the command
        tokens = rest.split()
    argv = []
    for token in tokens:
        if token in _SHELL_OPERATORS:
            break
        argv.append(token)
    return argv


def _commands(path: Path) -> list[Command]:
    commands = []
    for number, text in _logical_lines(path):
        match = _INVOCATION.search(text)
        if match:
            commands.append(Command(str(path), number, _argv(match.group("rest"))))
    return commands


def _bazel_runs(path: Path) -> list[Command]:
    """Each `bazel run` in `path`, as the label followed by the arguments after `--`."""
    runs = []
    for number, text in _logical_lines(path):
        match = _BAZEL_RUN.search(text)
        if not match:
            continue
        argv = _argv(match.group("rest"))
        if argv and argv[0] != "--":  # a Bazel option after the label: no tool arguments
            argv = []
        runs.append(Command(str(path), number, [match.group("label"), *argv[1:]]))
    return runs


def _options_by_flag(command: click.Command) -> dict[str, click.Option]:
    """Each option's flags (`--json`, `--text`, ...) mapped to the option."""
    options: dict[str, click.Option] = {}
    for param in command.params:
        if isinstance(param, click.Option):
            for opt in (*param.opts, *param.secondary_opts):
                options[opt] = param
    return options


def _check_given(command: click.Command, args: list[str]) -> tuple[list[str], set[str]]:
    """Problems with the arguments in `args`, and the names of the options given."""
    options = _options_by_flag(command)
    takes_arguments = any(isinstance(p, click.Argument) for p in command.params)
    problems: list[str] = []
    given: set[str] = set()
    tokens = iter(args)
    for token in tokens:
        if not token.startswith("-") or token == "-":
            if not takes_arguments:
                problems.append(f"{command.name} takes no positional argument, got {token!r}")
            continue
        opt, eq, value = token.partition("=")
        param = options.get(opt)
        if param is None:
            problems.append(f"{command.name} has no option {opt}")
            continue
        given.add(param.name)
        if param.is_flag or param.count:
            continue
        if not eq:
            value = next(tokens, None)
            if value is None:
                problems.append(f"{opt} needs a value")
                continue
        if isinstance(param.type, click.Choice) and value not in param.type.choices:
            problems.append(f"{opt} {value!r} is not one of {', '.join(param.type.choices)}")
    return problems, given


def _check_missing(command: click.Command, given: set[str]) -> list[str]:
    """Problems with the options `given` leaves out: required ones, and ones that prompt."""
    problems = []
    for param in command.params:
        if not isinstance(param, click.Option) or param.name in given:
            continue
        if param.required:
            problems.append(f"missing required option {param.opts[0]}")
        elif param.prompt:
            problems.append(f"{param.opts[0]} is not given, so the command prompts for it")
    return problems


def check_command(argv: list[str]) -> list[str]:
    """Problems with running `ipu-as` with `argv` non-interactively; empty when it would run."""
    if not argv:
        return ["no subcommand"]
    name, *args = argv
    command = cli.commands.get(name) if isinstance(cli, click.Group) else None
    if command is None:
        return [f"unknown subcommand {name!r} (have: {', '.join(sorted(cli.commands))})"]
    if "--help" in args:
        return []
    problems, given = _check_given(command, args)
    return problems + _check_missing(command, given)


def _parse(parser: argparse.ArgumentParser, args: list[str]) -> list[str]:
    """argparse's complaint about `args`, or nothing when `parser` accepts them."""
    errors = io.StringIO()
    try:
        with contextlib.redirect_stderr(errors):
            parser.parse_args(args)
    except SystemExit:
        return [errors.getvalue().strip().splitlines()[-1]]
    return []


def _tool_parsers() -> dict[str, Callable[[], argparse.ArgumentParser]]:
    from ipu_harness import cli as run_kernel
    from ipu_harness import query

    return {
        _RUN_KERNEL: run_kernel.build_parser,
        "//kernels:query": query.build_parser,
    }


def check_bazel_run(label: str, args: list[str]) -> list[str] | None:
    """Problems with `bazel run label -- args`; None for a label this test does not know."""
    args = [a for a in args if a != "..."]  # a placeholder for "your options"
    if "--help" in args or "-h" in args:
        return []
    parsers = _tool_parsers()
    if label in parsers:
        return _parse(parsers[label](), args)
    if label.startswith(_KERNELS):
        return _parse(parsers[_RUN_KERNEL](), ["kernel", *args])
    return None


def _cases() -> list:
    cases = []
    for path in doc_files():
        for command in _commands(path):
            case_id = f"{command.path}:{command.line}"
            cases.append(pytest.param(command, id=case_id))
    return cases


def _run_cases() -> list:
    return [
        pytest.param(command, id=f"{command.path}:{command.line}")
        for path in doc_files()
        for command in _bazel_runs(path)
    ]


@pytest.mark.parametrize("command", _cases())
def test_documented_command_runs(command: Command) -> None:
    problems = check_command(command.argv)
    assert not problems, f"{command.path}:{command.line}: " + "; ".join(problems)


@pytest.mark.parametrize("command", _run_cases())
def test_documented_bazel_run_parses(command: Command) -> None:
    label, *args = command.argv
    problems = check_bazel_run(label, args)
    assert not problems, f"{command.path}:{command.line}: " + "; ".join(problems)


def test_bazel_runs_are_found() -> None:
    # Guards the scan: the CLI reference runs every argparse tool it documents.
    labels = {c.argv[0] for c in _bazel_runs(Path("docs/content/reference/cli.md"))}
    assert set(_tool_parsers()) <= labels
    assert "//kernels/softmax/softmax_rows" in labels


@pytest.mark.parametrize(
    ("label", "args", "problem"),
    [
        (_RUN_KERNEL, ["k", "--inputs", "x=a.npy"], "unrecognized"),
        (_RUN_KERNEL, ["--case", "default"], "required: kernel"),
        (_RUN_KERNEL, ["k", "--max-cycles", "two"], "--max-cycles"),
        ("//kernels:query", ["scale", "--sweeps", "n=1..9"], "unrecognized"),
        ("//kernels/elementwise/scale", ["--rows", "3"], "unrecognized"),
    ],
)
def test_check_bazel_run_rejects(label: str, args: list[str], problem: str) -> None:
    assert any(problem in p for p in check_bazel_run(label, args) or [])


def test_check_bazel_run_accepts() -> None:
    assert check_bazel_run("//kernels/reshape/identity", ["--case", "single_row"]) == []
    assert check_bazel_run("//kernels:query", ["scale", "rows=n", "--sweep", "n=1..9"]) == []
    assert check_bazel_run("//tools:ruff", ["check", "--fix"]) is None
    assert check_bazel_run("//kernels/elementwise/scale", ["--case", "one_row", "--debug"]) == []


def test_commands_are_found() -> None:
    # Guards the scan itself: CLAUDE.md documents at least one assemble command.
    assert any(c.argv[:1] == ["assemble"] for c in _commands(Path("CLAUDE.md")))


@pytest.mark.parametrize(
    ("argv", "problem"),
    [
        (["assemble", "--input", "a", "--output", "b", "--format", "hex"], "not one of"),
        (["assemble", "--input", "a", "--format", "bin"], "missing required option --output"),
        (["assemble", "--input", "a", "--output", "b", "--defines", "X=1"], "no option --defines"),
        (["assemble", "--input", "a", "--output", "b", "--format", "bin", "x"], "positional"),
        (["link"], "unknown subcommand"),
    ],
)
def test_check_command_rejects(argv: list[str], problem: str) -> None:
    assert any(problem in p for p in check_command(argv))


def test_check_command_accepts() -> None:
    argv = ["assemble", "--input", "a", "--output", "b", "--format=bin"]
    assert check_command(argv) == []
    argv = ["assemble", "--input", "a", "--output", "b.bin", "--define", "rows=8", "-I", "lib"]
    assert check_command(argv) == []
