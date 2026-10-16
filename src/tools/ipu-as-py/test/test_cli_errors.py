"""`ipu-as assemble` reports a bad source the way it always has (E05-06).

The golden (test/golden/cli_assemble_errors.json) is the CLI's output over the
check corpus before the assembler library stopped printing and exiting:
stdout, stderr, exit code, and the exception that escaped, if any. Only the
template stage changes on purpose: a template error used to escape as a jinja2
traceback and now prints one `Error rendering template:` line.

A difference writes the new snapshot to the test's undeclared outputs
(bazel-testlogs/src/tools/ipu-as-py/test_cli_errors/test.outputs/).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from check_corpus import HERE, sources
from click.testing import CliRunner

from ipu_as.cli import cli

GOLDEN = HERE / "golden" / "cli_assemble_errors.json"

#: lark lists the terminals it expected in set order, which varies between runs.
_EXPECTED_BLOCK = re.compile(r"(?:\t\* [^\n]*\n)+")


def _stable(text: str) -> str:
    """`text` with each block of lark's expected terminals sorted."""
    return _EXPECTED_BLOCK.sub(lambda m: "".join(sorted(m.group(0).splitlines(True))), text)


def _run(source: str, tmp: Path) -> dict:
    src = tmp / "prog.asm"
    out = tmp / "prog.bin"
    src.write_text(source, encoding="utf-8")
    result = CliRunner().invoke(
        cli, ["assemble", "--input", str(src), "--output", str(out), "--format", "bin"]
    )
    exception = result.exception
    escaped = None if exception is None or isinstance(exception, SystemExit) else exception
    return {
        "exit_code": result.exit_code,
        "stdout": _stable(result.stdout.replace(str(src), "<input>").replace(str(out), "<output>")),
        "stderr": result.stderr,
        "exception": type(escaped).__name__ if escaped is not None else None,
    }


def _snapshot(tmp: Path) -> dict[str, dict]:
    return {name: _run(source, tmp) for name, source in sources().items()}


def test_cli_output_is_unchanged(tmp_path):
    actual = _snapshot(tmp_path)
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    outputs = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if actual != expected and outputs:
        Path(outputs, GOLDEN.name).write_text(
            json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    assert actual.keys() == expected.keys()
    for name, before in expected.items():
        if before["exception"] is not None:
            # A template error escaped as a jinja2 traceback; now it is one line.
            header = before["stdout"]
            assert actual[name]["exit_code"] == 1, name
            assert actual[name]["exception"] is None, name
            assert actual[name]["stdout"].startswith(header + "Error rendering template: "), name
            assert actual[name]["stdout"].count("\n") == header.count("\n") + 1, name
        else:
            assert actual[name] == before, name
