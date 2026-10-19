"""The assembly sources the CLI contract tests run: the VS Code extension's corpus
and every source test_diagnostics.py checks."""

from __future__ import annotations

import ast
from pathlib import Path

# The workspace root: the runfiles tree under Bazel, the checkout under plain pytest.
ROOT = Path(__file__).parents[3]
HERE = ROOT / "src" / "ipu-as" / "test"


def diagnostics_test_sources() -> list[str]:
    """The sources test_diagnostics.py checks: the literal passed to `check(...)`,
    assigned to `source`, or listed for a `source` parameter (alone, or first of
    several: `"source,expected_line"`)."""
    tree = ast.parse((HERE / "test_diagnostics.py").read_text(encoding="utf-8"))
    found: list[str] = []

    def literal(node: ast.AST) -> None:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value not in found
        ):
            found.append(node.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "check" and node.args:
            literal(node.args[0])
        elif isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "source" for t in node.targets
        ):
            literal(node.value)
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "parametrize":
            names, values = node.args[0], node.args[1]
            if not (isinstance(names, ast.Constant) and isinstance(values, ast.List)):
                continue
            argnames = [name.strip() for name in names.value.split(",")]
            if argnames[0] != "source":
                continue
            for element in values.elts:
                literal(
                    element.elts[0]
                    if len(argnames) > 1 and isinstance(element, ast.Tuple)
                    else element
                )
    return found


def sources() -> dict[str, str]:
    """Name -> source text: the corpus by path, test_diagnostics' sources by their own text
    (so adding one leaves the others' entries as they were)."""
    corpus = {
        str(path.relative_to(ROOT)): path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "vscode-ipu-asm" / "test" / "corpus").rglob("*.asm"))
    }
    inline = {f"test_diagnostics.py: {text!r}": text for text in diagnostics_test_sources()}
    return corpus | inline
