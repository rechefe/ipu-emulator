"""Generates the package table and dependency diagram of Explanation: architecture (E09-17).

``docs/content/explanation/architecture.md`` is hand-written prose that
includes two generated snippets (``pymdownx.snippets``), so the list of
packages and their edges can never drift from the code:

* ``dependencies.md``: a mermaid diagram of the ``[allowed]`` rows of
  ``tools/import_graph.toml``, the table ``//tools:test_import_graph``
  enforces, without the edges another path already implies;
* ``packages.md``: one row per package, with its directory, what it owns
  (the first paragraph of its ``__init__.py`` docstring), what it may import
  and what may import it, then the table's subpackage and deferred rows.

    bazel run //docs:architecture_page -- OUT_DIR

The ``build_docs`` genrule runs it into ``generated/architecture/``. Paths
are read from the current directory, which is the workspace root in a
genrule and in a test's runfiles.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

TABLE = Path("tools/import_graph.toml")
SOURCE_ROOT = Path("src/tools")


@dataclass(frozen=True)
class Package:
    """One first-party package: a top-level key of the import graph's ``[allowed]`` table."""

    name: str
    directory: str
    owns: str
    imports: tuple[str, ...]
    imported_by: tuple[str, ...]


def load_rows(table: Path, section: str) -> dict[str, tuple[str, ...]]:
    with table.open("rb") as f:
        rows = tomllib.load(f).get(section, {})
    return {key: tuple(sorted(deps)) for key, deps in rows.items()}


def summary(init: Path) -> str:
    """The first paragraph of a package's docstring, on one line, in Markdown code style."""
    doc = ast.get_docstring(ast.parse(init.read_text(encoding="utf-8")))
    if not doc:
        raise ValueError(f"{init}: a package needs a docstring that says what it owns")
    first = doc.split("\n\n", 1)[0]
    return re.sub(r"``([^`]+)``", r"`\1`", " ".join(first.split()))


def _depth(name: str, allowed: Mapping[str, tuple[str, ...]], seen: tuple[str, ...] = ()) -> int:
    if name in seen:
        raise ValueError(f"import cycle: {' -> '.join((*seen, name))}")
    deps = [d for d in allowed.get(name, ()) if d in allowed]
    return 1 + max((_depth(d, allowed, (*seen, name)) for d in deps), default=-1)


def packages(table: Path = TABLE, root: Path = SOURCE_ROOT) -> list[Package]:
    """Every package of the table, the ones that import nothing first, then by depth and name."""
    allowed = {k: v for k, v in load_rows(table, "allowed").items() if "." not in k}
    found = []
    for name in sorted(allowed, key=lambda n: (_depth(n, allowed), n)):
        inits = sorted(root.glob(f"*/src/{name}/__init__.py"))
        if len(inits) != 1:
            raise ValueError(f"{name}: expected one {root}/*/src/{name}/__init__.py, found {inits}")
        found.append(
            Package(
                name=name,
                directory=f"{inits[0].parents[2].as_posix()}/",
                owns=summary(inits[0]),
                imports=allowed[name],
                imported_by=tuple(sorted(k for k, deps in allowed.items() if name in deps)),
            )
        )
    return found


def _codes(names: tuple[str, ...]) -> str:
    return ", ".join(f"`{n}`" for n in names) or "nothing"


def _reaches(graph: Mapping[str, tuple[str, ...]], start: str) -> set[str]:
    seen: set[str] = set()
    todo = list(graph.get(start, ()))
    while todo:
        name = todo.pop()
        if name not in seen:
            seen.add(name)
            todo += graph.get(name, ())
    return seen


def drawn_edges(pkgs: list[Package]) -> list[tuple[str, str]]:
    """The edges of the diagram: each allowed import that no other allowed import
    already implies (the transitive reduction)."""
    graph = {p.name: p.imports for p in pkgs}
    return [
        (p.name, dep)
        for p in pkgs
        for dep in p.imports
        if not any(dep in _reaches(graph, other) for other in p.imports if other != dep)
    ]


def render_dependencies(pkgs: list[Package]) -> str:
    lines = ["```mermaid", "flowchart TD"]
    lines += [f"    {p.name}[{p.name}]" for p in pkgs]
    lines += [f"    {src} --> {dst}" for src, dst in drawn_edges(pkgs)]
    return "\n".join([*lines, "```", ""])


def render_packages(pkgs: list[Package], table: Path = TABLE) -> str:
    lines = [
        "| Package | Directory | What it owns | Imports | Imported by |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| `{p.name}` | `{p.directory}` | {p.owns} | {_codes(p.imports)} | {_codes(p.imported_by)} |"
        for p in pkgs
    ]
    notes = [
        f"- `{key}` may import only {_codes(deps)}."
        if deps
        else f"- `{key}` imports no other first-party package."
        for key, deps in load_rows(table, "allowed").items()
        if "." in key
    ]
    notes += [
        f"- `{key}` imports {_codes(mods)} only inside a function."
        for key, mods in load_rows(table, "deferred").items()
    ]
    if notes:
        lines += ["", *notes]
    return "\n".join([*lines, ""])


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: architecture_page.py OUT_DIR", file=sys.stderr)
        return 2
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    pkgs = packages()
    (out / "dependencies.md").write_text(render_dependencies(pkgs), encoding="utf-8")
    (out / "packages.md").write_text(render_packages(pkgs), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
