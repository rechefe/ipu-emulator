"""Every documented `ipu-as` command uses only options the CLI declares.

The options are read from the click command itself, so the test follows the
CLI as it changes.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import click
import pytest
from ipu_as.cli import cli

from docs.doc_files import doc_files

# `ipu-as <sub>` or `ipu-as -- <sub>` (as after `bazel run ...:ipu-as`); not
# `ipu-as-py` or `ipu-asm`.
_INVOCATION = re.compile(r"\bipu-as(?![\w-])(?:[ \t]+--)?[ \t]+(?P<rest>[a-z].*)$")
_SHELL_OPERATORS = {"|", "||", "&&", ";", ">", ">>", "<", "&", "2>&1"}


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


def _commands(path: Path) -> list[Command]:
    commands = []
    for number, text in _logical_lines(path):
        match = _INVOCATION.search(text)
        if not match:
            continue
        rest = match.group("rest").split("`", 1)[0]  # an inline code span ends at a backtick
        try:
            tokens = shlex.split(rest, comments=True)
        except ValueError:  # an unbalanced quote in prose after the command
            tokens = rest.split()
        argv = []
        for token in tokens:
            if token in _SHELL_OPERATORS:
                break
            argv.append(token)
        commands.append(Command(str(path), number, argv))
    return commands


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


def _cases() -> list:
    cases = []
    for path in doc_files():
        for command in _commands(path):
            case_id = f"{command.path}:{command.line}"
            cases.append(pytest.param(command, id=case_id))
    return cases


@pytest.mark.parametrize("command", _cases())
def test_documented_command_runs(command: Command) -> None:
    problems = check_command(command.argv)
    assert not problems, f"{command.path}:{command.line}: " + "; ".join(problems)


def test_commands_are_found() -> None:
    # Guards the scan itself: CLAUDE.md documents at least one assemble command.
    assert any(c.argv[:1] == ["assemble"] for c in _commands(Path("CLAUDE.md")))


@pytest.mark.parametrize(
    ("argv", "problem"),
    [
        (["assemble", "--input", "a", "--output", "b"], "prompts"),
        (["assemble", "--input", "a", "--output", "b", "--format", "hex"], "not one of"),
        (["assemble", "--input", "a", "--format", "bin"], "missing required option --output"),
        (["assemble", "--define", "X=1"], "no option --define"),
        (["assemble", "--input", "a", "--output", "b", "--format", "bin", "x"], "positional"),
        (["link"], "unknown subcommand"),
    ],
)
def test_check_command_rejects(argv: list[str], problem: str) -> None:
    assert any(problem in p for p in check_command(argv))


def test_check_command_accepts() -> None:
    argv = ["assemble", "--input", "a", "--output", "b", "--format=bin"]
    assert check_command(argv) == []
