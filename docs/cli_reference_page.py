"""The generated parts of the command-line tools page (E09-15).

    bazel run //docs:cli_reference_page -- "$PWD/out"

writes one Markdown snippet per tool into the given directory, read from the
tool's own option declarations (click for ``ipu-as``, argparse for the rest),
so the page lists exactly what each tool parses. ``reference/cli.md`` holds the
hand-written intro and example of each tool and includes these snippets through
``pymdownx.snippets``, as the reference-model API page does.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import click


def _cell(text: object) -> str:
    """``text`` on one line, safe in a Markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|").replace("<", "&lt;")


def _code(text: object) -> str:
    return f"`{text}`".replace("|", "\\|")


def _table(rows: Iterable[tuple[str, str, str, str]]) -> str:
    lines = ["| Option | Value | Default | Description |", "|---|---|---|---|"]
    lines += [f"| {flag} | {value} | {default} | {text} |" for flag, value, default, text in rows]
    return "\n".join(lines) + "\n"


def _click_value(param: click.Parameter) -> str:
    if isinstance(param, click.Option) and (param.is_flag or param.count):
        value = "flag"
    elif isinstance(param.type, click.Choice):
        value = " \\| ".join(_code(choice) for choice in param.type.choices)
    elif param.metavar:
        value = _code(param.metavar)
    elif isinstance(param.type, click.IntRange):
        value = f"{param.type.min} to {param.type.max}"
    else:
        value = _code(param.type.name.upper())
    return value + (", repeatable" if param.multiple else "")


def _click_default(param: click.Parameter) -> str:
    if param.required:
        return "required"
    # Only a value a user could type: not a callable, an empty tuple or click's
    # own "no default" marker.
    default = param.default
    if isinstance(default, bool) or not isinstance(default, (str, int, float)):
        return ""
    return _code(default)


def _click_rows(command: click.Command) -> Iterable[tuple[str, str, str, str]]:
    for param in command.params:
        if isinstance(param, click.Option) and param.hidden:
            continue
        if isinstance(param, click.Option):
            flag = " / ".join(_code(o) for o in (*param.opts, *param.secondary_opts))
            yield flag, _click_value(param), _click_default(param), _cell(param.help or "")
        else:
            yield _code(param.human_readable_name), "positional", _click_default(param), ""


def click_reference(group: click.Group, prog: str) -> str:
    """Every subcommand of ``group``: usage, first help paragraph, options."""
    parts = []
    for name, command in group.commands.items():
        context = click.Context(command, info_name=f"{prog} {name}")
        summary = inspect.cleandoc(command.help or "").split("\n\n")[0]
        usage = command.get_usage(context).removeprefix("Usage: ")
        parts.append(f"### `{prog} {name}`\n\n{_cell(summary)}\n\n```text\n{usage}\n```\n")
        parts.append(_table(_click_rows(command)))
    return "\n".join(parts)


def _argparse_value(action: argparse.Action) -> str:
    if action.nargs == 0:
        return "flag"
    if action.choices:
        return " \\| ".join(_code(choice) for choice in action.choices)
    value = (
        "positional" if not action.option_strings else _code(action.metavar or action.dest.upper())
    )
    if action.type in (int, float):
        value += f" ({action.type.__name__})"
    if isinstance(action, argparse._AppendAction) or action.nargs in ("*", "+"):
        value += ", repeatable"
    return value


def _argparse_default(action: argparse.Action) -> str:
    if action.required and action.nargs not in ("?", "*"):
        return "required"
    if action.default in (None, False, [], argparse.SUPPRESS):
        return ""
    return _code(action.default)


def argparse_reference(
    parser: argparse.ArgumentParser, prog: str, usage: str | None = None, skip: tuple[str, ...] = ()
) -> str:
    """``parser``'s usage line and every argument it declares but ``--help`` and ``skip``.

    ``usage`` replaces argparse's own usage line, for a tool whose launcher
    passes some options itself.
    """
    parser.prog = prog
    rows = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction) or set(action.option_strings) & set(skip):
            continue
        if action.option_strings:
            flag = " / ".join(_code(o) for o in action.option_strings)
        else:
            flag = _code(action.metavar or action.dest)
        text = _cell(action.help or "")
        rows.append((flag, _argparse_value(action), _argparse_default(action), text))
    usage = usage or " ".join(parser.format_usage().removeprefix("usage: ").split())
    return f"```text\n{usage}\n```\n\n" + _table(rows)


def _case_values(values: dict) -> str:
    """A case's parameter values; a tuple is a sweep axis."""
    if not values:
        return "none"
    return ", ".join(_code(f"{name}={value}") for name, value in values.items())


def kernel_cases_reference() -> str:
    """Each kernel's cases (``kernel.toml`` ``[cases]``): values and formats."""
    from ipu_harness.registry import default_registry

    lines = ["| Kernel | Case | Values | Formats |", "|---|---|---|---|"]
    registry = default_registry()
    for name, manifest in sorted(registry.manifests.items()):
        for case in manifest.cases.values():
            formats = case.formats or manifest.formats
            listed = ", ".join(f.value for f in formats)
            lines.append(
                f"| `{name}` | `{case.name}` | {_case_values(dict(case.values))} | {listed} |"
            )
    return "\n".join(lines) + "\n"


def debugger_keys_reference() -> str:
    """Every debugger control, by scope, from the TUI's own key table."""
    from ipu_debug.tui.keymap import HELP_SCOPE_TITLES, KEY_BINDINGS, PANE_FOCUS_ORDER

    parts = []
    for scope, title in HELP_SCOPE_TITLES:
        rows = [
            f"| {' / '.join(_code(k) for k in b.keys)} | {_cell(b.description)} | "
            + ("all" if b.panes == PANE_FOCUS_ORDER else ", ".join(b.panes))
            + " |"
            for b in KEY_BINDINGS
            if b.scope == scope
        ]
        if rows:
            header = "| Key | Action | Panes |\n|---|---|---|"
            parts.append(f"**{title}**\n\n{header}\n" + "\n".join(rows) + "\n")
    return "\n".join(parts)


def _ipu_as() -> str:
    from ipu_as.cli import cli

    return click_reference(cli, "ipu-as")


def _run_kernel() -> str:
    from ipu_harness.cli import build_parser

    return argparse_reference(build_parser(), "run_kernel")


def _query() -> str:
    from ipu_harness.query import build_parser

    return argparse_reference(build_parser(), "query")


#: Each snippet file and the function that renders it.
SNIPPETS: dict[str, Callable[[], str]] = {
    "ipu-as.md": _ipu_as,
    "run-kernel.md": _run_kernel,
    "kernel-cases.md": kernel_cases_reference,
    "query.md": _query,
    "debugger-keys.md": debugger_keys_reference,
}


def write(out_dir: Path) -> list[Path]:
    """Write every snippet into ``out_dir``; return the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, render in SNIPPETS.items():
        path = out_dir / name
        path.write_text(render(), encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: cli_reference_page.py OUT_DIR", file=sys.stderr)
        return 2
    for path in write(Path(argv[0])):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
