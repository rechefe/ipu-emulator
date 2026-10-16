"""`ipu-as check --json` is a contract with the VS Code extension: pin it.

The contract is written down in vscode-ipu-asm/schema/ (check-json.md and a
JSON Schema per version). v1, the bare array, must stay byte-identical: the
snapshot below is its output over the extension's corpus and every source
test_diagnostics.py checks. A difference fails; if it is intended, it is a
contract change (see check-json.md), and the new snapshot is written to the
test's undeclared outputs
(bazel-testlogs/src/tools/ipu-as-py/test_check_json/test.outputs/) to copy over
test/golden/check_json_v1.json.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import jsonschema
from click.testing import CliRunner
from referencing import Registry, Resource

from ipu_as.cli import cli

# The workspace root: the runfiles tree under Bazel, the checkout under plain pytest.
ROOT = Path(__file__).parents[4]
HERE = ROOT / "src" / "tools" / "ipu-as-py" / "test"
SCHEMAS = ROOT / "vscode-ipu-asm" / "schema"
GOLDEN = HERE / "golden" / "check_json_v1.json"


def _diagnostics_test_sources() -> list[str]:
    """The sources test_diagnostics.py checks: the literal passed to `check(...)`,
    assigned to `source`, or listed for a `source` parameter (alone, or first of
    several: `"source,expected_line"`)."""
    tree = ast.parse((HERE / "test_diagnostics.py").read_text(encoding="utf-8"))
    found: list[str] = []

    def literal(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value not in found:
            found.append(node.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "check" and node.args:
            literal(node.args[0])
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "source" for t in node.targets):
            literal(node.value)
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "parametrize":
            names, values = node.args[0], node.args[1]
            if not (isinstance(names, ast.Constant) and isinstance(values, ast.List)):
                continue
            argnames = [name.strip() for name in names.value.split(",")]
            if argnames[0] != "source":
                continue
            for element in values.elts:
                literal(element.elts[0] if len(argnames) > 1 and isinstance(element, ast.Tuple) else element)
    return found


def _sources() -> dict[str, str]:
    """Name -> source text: the corpus by path, test_diagnostics' sources by their own text
    (so adding one leaves the others' entries as they were)."""
    corpus = {
        str(path.relative_to(ROOT)): path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "vscode-ipu-asm" / "test" / "corpus").rglob("*.asm"))
    }
    inline = {f"test_diagnostics.py: {text!r}": text for text in _diagnostics_test_sources()}
    return corpus | inline


def _check(source: str, *options: str) -> tuple[int, str]:
    result = CliRunner().invoke(cli, ["check", "--json", *options, "--input", "-"], input=source)
    return result.exit_code, result.stdout


def _validate(output: object, version: int) -> None:
    """Validate against version *version*'s schema; v2 refers to v1's by file name."""
    schemas = {
        path.name: json.loads(path.read_text(encoding="utf-8")) for path in SCHEMAS.glob("check-json-v*.schema.json")
    }
    registry = Registry().with_resources((name, Resource.from_contents(schema)) for name, schema in schemas.items())
    schema = schemas[f"check-json-v{version}.schema.json"]
    jsonschema.validators.validator_for(schema)(schema, registry=registry).validate(output)


def _snapshot() -> dict[str, dict]:
    snapshot = {}
    for name, source in _sources().items():
        exit_code, stdout = _check(source)
        snapshot[name] = {"exit_code": exit_code, "stdout": stdout}
    return snapshot


def test_v1_output_is_unchanged():
    actual = _snapshot()
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if actual != expected:
        outputs = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if outputs:
            Path(outputs, GOLDEN.name).write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert actual == expected


def test_the_schemas_are_valid_json_schemas():
    for path in SCHEMAS.glob("check-json-v*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.validators.validator_for(schema).check_schema(schema)


def test_the_snapshot_covers_every_parametrized_source():
    # Including those listed with other parameters ("source,expected_line").
    assert 'BKPT;;\nBKPT;;\n{{ 1 + "a" }}\nBKPT;;\n' in _diagnostics_test_sources()


def test_v1_output_matches_its_schema():
    for source in _sources().values():
        _, stdout = _check(source)
        _validate(json.loads(stdout), 1)


def test_v2_wraps_the_same_diagnostics_and_matches_its_schema():
    found_some = False
    for source in _sources().values():
        v1_code, v1 = _check(source)
        v2_code, v2 = _check(source, "--json-version", "2")
        output = json.loads(v2)
        _validate(output, 2)
        assert output["schemaVersion"] == 2
        assert output["diagnostics"] == json.loads(v1)
        assert v2_code == v1_code
        found_some = found_some or bool(output["diagnostics"])
    assert found_some, "the corpus has sources with diagnostics"


def test_explicit_version_1_is_the_default():
    for source in _sources().values():
        assert _check(source, "--json-version", "1") == _check(source)


def test_a_json_version_without_json_is_a_usage_error():
    result = CliRunner().invoke(cli, ["check", "--json-version", "2", "--input", "-"], input="BKPT;;\n")
    assert result.exit_code == 2
    assert "--json-version" in result.output


def test_an_unknown_version_is_a_usage_error():
    result = CliRunner().invoke(cli, ["check", "--json", "--json-version", "3", "--input", "-"], input="BKPT;;\n")
    assert result.exit_code == 2
    assert "Usage:" in result.output and "--json-version" in result.output


def test_supported_versions_are_listed():
    result = CliRunner().invoke(cli, ["check", "--json-versions"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == [1, 2]
    schemas = sorted(int(p.name.split("-v")[1].split(".")[0]) for p in SCHEMAS.glob("check-json-v*.schema.json"))
    assert schemas == [1, 2], "a schema for every version printed"

