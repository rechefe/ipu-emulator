"""Explanation: architecture's generated parts (E09-17).

The package table and the dependency diagram come from
``tools/import_graph.toml`` and each package's docstring, so every package the
import graph lists is on the page, a renamed package changes the page with no
doc edit, and the hand-written page includes both snippets.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from docs import architecture_page as page

PAGE = Path("docs/content/explanation/architecture.md")
INDEX = Path("docs/content/explanation/index.md")


@pytest.fixture(scope="module")
def pkgs() -> list[page.Package]:
    return page.packages()


def _table_names(text: str) -> list[str]:
    return re.findall(r"^\| `(\w+)` \|", text, flags=re.MULTILINE)


def test_every_package_of_the_import_graph_has_a_row(pkgs):
    allowed = page.load_rows(page.TABLE, "allowed")
    top = {k for k in allowed if "." not in k}
    assert set(_table_names(page.render_packages(pkgs))) == top
    assert {"ipu_isa", "ipu_as", "ipu_emu", "ipu_debug", "ipu_gen", "ipu_harness"} <= top


def test_every_package_says_what_it_owns(pkgs):
    assert all(p.owns and "|" not in p.owns for p in pkgs)


def test_the_isa_comes_first_and_nothing_imports_up(pkgs):
    order = [p.name for p in pkgs]
    assert order[0] == "ipu_isa"
    for p in pkgs:
        assert all(order.index(dep) < order.index(p.name) for dep in p.imports)


def test_the_diagram_implies_every_allowed_import(pkgs):
    drawn: dict[str, tuple[str, ...]] = {}
    for src, dst in page.drawn_edges(pkgs):
        drawn[src] = (*drawn.get(src, ()), dst)
    for p in pkgs:
        assert set(p.imports) <= page._reaches(drawn, p.name)
    diagram = page.render_dependencies(pkgs)
    assert diagram.startswith("```mermaid\nflowchart TD\n")
    assert "ipu_as --> ipu_isa" in diagram
    assert "ipu_harness --> ipu_isa" not in diagram  # implied through ipu_as


def _fake_tree(tmp_path: Path, name: str) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    table = tmp_path / "import_graph.toml"
    table.write_text(
        f'[allowed]\nipu_isa = []\n{name} = ["ipu_isa"]\n"{name}.sub" = []\n'
        '[deferred]\nipu_isa = ["torch"]\n',
        encoding="utf-8",
    )
    for pkg, doc in (("ipu_isa", "The ISA."), (name, "The tool.\n\nMore.")):
        init = tmp_path / "src" / pkg.replace("_", "-") / "src" / pkg / "__init__.py"
        init.parent.mkdir(parents=True)
        init.write_text(f'"""{doc}"""\n', encoding="utf-8")
    return table, tmp_path / "src"


def test_renaming_a_package_changes_the_page(tmp_path):
    before_tree = _fake_tree(tmp_path / "a", "ipu_old")
    before = page.render_packages(page.packages(*before_tree), before_tree[0])
    after_tree = _fake_tree(tmp_path / "b", "ipu_new")
    after = page.render_packages(page.packages(*after_tree), after_tree[0])
    assert "`ipu_old`" in before and "`ipu_old`" not in after
    assert "| `ipu_new` | `" in after and "src/ipu-new/" in after
    assert "| The tool. |" in after  # the first paragraph only
    assert "- `ipu_new.sub` imports no other first-party package." in after
    assert "- `ipu_isa` imports `torch` only inside a function." in after
    diagram = page.render_dependencies(page.packages(*after_tree))
    assert "ipu_new --> ipu_isa" in diagram and "ipu_old" not in diagram


def test_a_package_without_a_docstring_fails(tmp_path):
    table, root = _fake_tree(tmp_path, "ipu_tool")
    (root / "ipu-tool" / "src" / "ipu_tool" / "__init__.py").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="docstring"):
        page.packages(table, root)


def test_a_package_without_sources_fails(tmp_path):
    table, root = _fake_tree(tmp_path, "ipu_tool")
    table.write_text(
        table.read_text(encoding="utf-8").replace("[deferred]", "ipu_ghost = []\n[deferred]"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ipu_ghost"):
        page.packages(table, root)


def test_the_page_includes_both_snippets_and_no_stub():
    text = PAGE.read_text(encoding="utf-8")
    assert '--8<-- "generated/architecture/dependencies.md"' in text
    assert '--8<-- "generated/architecture/packages.md"' in text
    assert "fills this page" not in text
    assert "prototyping-mode.md" not in INDEX.read_text(encoding="utf-8")
