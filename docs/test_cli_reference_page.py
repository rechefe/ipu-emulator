"""The Command-line tools page: each table is generated from the tool's own
parser, so an option added to a tool shows up with no doc edit, and the page
includes every generated snippet. Its examples are checked by
``test_doc_commands``."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import click
import pytest

from docs import cli_reference_page as page

CLI_PAGE = Path("docs/content/reference/cli.md")


def test_an_added_click_option_shows_up() -> None:
    @click.group()
    def tool() -> None:
        pass

    @tool.command()
    @click.option("--input", required=True, help="The source.")
    def build(input: str) -> None:
        """Build a thing.

        More text that is not the summary.
        """

    before = page.click_reference(tool, "tool")
    assert "### `tool build`" in before and "Build a thing." in before
    assert "More text" not in before and "--level" not in before

    build = click.option("--level", type=click.Choice(["1", "2"]), default="1", help="How hard.")(
        build
    )
    after = page.click_reference(tool, "tool")
    assert "| `--level` | `1` \\| `2` | `1` | How hard. |" in after
    assert "| `--input` | `TEXT` | required | The source. |" in after


def test_an_added_argparse_option_shows_up() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="the kernel")
    before = page.argparse_reference(parser, "run")
    assert "--seed" not in before
    parser.add_argument("--seed", type=int, default=3, help="the generator's seed")
    parser.add_argument("--input", action="append", metavar="NAME=PATH", help="an input")
    after = page.argparse_reference(parser, "run")
    assert "| `--seed` | `SEED` (int) | `3` | the generator's seed |" in after
    assert "| `--input` | `NAME=PATH`, repeatable |  | an input |" in after
    assert "| `kernel` | positional | required | the kernel |" in after
    assert "run [-h] [--seed SEED]" in after


def test_skip_and_usage_replace_what_a_launcher_passes() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel")
    parser.add_argument("--case")
    text = page.argparse_reference(
        parser, "x", usage="bazel run //:k -- [OPTIONS]", skip=("--kernel",)
    )
    assert "--kernel" not in text and "`--case`" in text
    assert "bazel run //:k -- [OPTIONS]" in text


def test_a_cell_cannot_break_the_table() -> None:
    assert page._cell("a | b\n  {{ cr.<name> }}") == "a \\| b {{ cr.&lt;name> }}"


@pytest.fixture(scope="module")
def snippets() -> dict[str, str]:
    return {name: render() for name, render in page.SNIPPETS.items()}


def test_every_ipu_as_option_is_listed(snippets: dict[str, str]) -> None:
    from ipu_as.cli import cli

    text = snippets["ipu-as.md"]
    for name, command in cli.commands.items():
        assert f"### `ipu-as {name}`" in text
        for param in command.params:
            for opt in getattr(param, "opts", []):
                assert f"`{opt}`" in text, (name, opt)


@pytest.mark.parametrize(
    ("snippet", "module"),
    [
        ("kernel-targets.md", "ipu_apps.kernel_registry.runner"),
        ("run-kernel.md", "ipu_harness.cli"),
        ("trace-case.md", "ipu_apps.kernel_registry.trace_case"),
        ("query.md", "ipu_apps.kernel_registry.query"),
    ],
)
def test_every_argparse_option_is_listed(
    snippets: dict[str, str], snippet: str, module: str
) -> None:
    import importlib

    parser = importlib.import_module(module).build_parser()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in ("-h", "--help", "--kernel"):
                continue
            assert f"`{opt}`" in snippets[snippet], opt


def test_every_kernel_case_is_listed(snippets: dict[str, str]) -> None:
    from ipu_apps.kernel_registry import registry
    from ipu_apps.kernel_registry.cases import load_cases

    registry.load()
    text = snippets["case-options.md"]
    for spec in registry.kernels():
        for case in load_cases(spec.name):
            assert f"| `{spec.name}` | `{case}` |" in text


def test_every_debugger_control_is_listed(snippets: dict[str, str]) -> None:
    from ipu_debug.tui.keymap import KEY_BINDINGS

    text = snippets["debugger-keys.md"]
    for binding in KEY_BINDINGS:
        assert page._cell(binding.description) in text


def test_the_run_summary_is_what_a_run_prints(snippets: dict[str, str]) -> None:
    text = snippets["run-summary.md"]
    assert text.startswith("```text\n=== Run summary ===\n")
    assert "Effective MAC:" in text


def test_the_page_includes_every_snippet() -> None:
    text = CLI_PAGE.read_text(encoding="utf-8")
    included = re.findall(r'^--8<-- "generated/cli/([^"]+)"$', text, flags=re.M)
    assert sorted(included) == sorted(page.SNIPPETS)


def test_each_tool_section_has_an_example() -> None:
    sections = re.split(r"^## ", CLI_PAGE.read_text(encoding="utf-8"), flags=re.M)[1:]
    assert len(sections) >= 6
    for section in sections:
        assert "```bash\nbazel run " in section, section.splitlines()[0]


def test_write_puts_each_snippet_in_the_directory(tmp_path: Path) -> None:
    written = page.write(tmp_path / "cli")
    assert sorted(p.name for p in written) == sorted(page.SNIPPETS)
    assert all(p.read_text(encoding="utf-8") for p in written)
