"""Snapshots of every artifact generated from instruction_spec.py except the
SV package (E02-04; the SV package is pinned by E02-03).

Each artifact is regenerated in-process by the function its Bazel wrapper
calls, not by the genrule, so the result does not depend on stamping. The
generators embed no timestamps or paths today, so no normalization is applied;
if one starts to, normalize it here and list it in ``NORMALIZATIONS``.

``test_generated_artifacts`` compares them byte for byte;
``bazel run //src/tools/ipu-as-py:update_generated_artifacts`` rewrites them.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ipu_gen import docs_pages, layout_svg, vscode_data

#: Where the snapshots live, repo-relative.
SNAPSHOT_DIR = "src/tools/ipu-as-py/test/baselines/generated"

#: The one command that rewrites them from the current code.
UPDATE_COMMAND = "bazel run //src/tools/ipu-as-py:update_generated_artifacts"

#: Text substitutions applied before comparing. None are needed today.
NORMALIZATIONS: tuple[tuple[str, str], ...] = ()


def _all_docs(out: Path) -> None:
    docs_pages.generate_all_docs(out)


def _instruction_format(out: Path) -> None:
    docs_pages.generate_instruction_format_md(out / "instruction-format.md")


def _vscode(out: Path) -> None:
    vscode_data.generate_all(out, vscode_data.REGISTER_FACTS)


def _union_layout_svg(out: Path) -> None:
    (out / "union_layout.svg").write_text(layout_svg.union_layout_svg())


@dataclass(frozen=True)
class Producer:
    """One generator call: the function its wrapper calls (for the reader), how
    to call it into a directory, and the files it writes there."""

    function: str
    write: Callable[[Path], None]
    outputs: tuple[str, ...]


PRODUCERS = (
    Producer(
        "ipu_gen.docs_pages.generate_all_docs (//docs:generate_mkdocs_md)",
        _all_docs,
        docs_pages.REFERENCE_PAGES,
    ),
    Producer(
        "ipu_gen.docs_pages.generate_instruction_format_md (//docs:generate_instruction_format_md)",
        _instruction_format,
        ("instruction-format.md",),
    ),
    Producer(
        "ipu_gen.vscode_data.generate_all (//vscode-ipu-asm:generate_grammar)",
        _vscode,
        ("syntaxes/ipu-asm.tmLanguage.json", "language-configuration.json", "data/isa.json"),
    ),
    Producer(
        "ipu_gen.layout_svg.union_layout_svg (ipu-gen union-layout-svg)",
        _union_layout_svg,
        ("union_layout.svg",),
    ),
)


def normalize(text: str) -> str:
    for old, new in NORMALIZATIONS:
        text = text.replace(old, new)
    return text


def generate() -> dict[str, tuple[str, str]]:
    """Snapshot name -> (producing function, normalized content)."""
    artifacts = {}
    with tempfile.TemporaryDirectory() as tmp:
        for producer in PRODUCERS:
            out = Path(tmp) / str(len(artifacts))
            out.mkdir()
            producer.write(out)
            written = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
            if written != set(producer.outputs):
                raise RuntimeError(
                    f"{producer.function} wrote {sorted(written)}, "
                    f"expected {sorted(producer.outputs)}"
                )
            for name in producer.outputs:
                text = (out / name).read_text(encoding="utf-8")
                artifacts[name] = (producer.function, normalize(text))
    return artifacts


def write_all(root: Path) -> int:
    directory = root / SNAPSHOT_DIR
    artifacts = generate()
    for name, (_, text) in artifacts.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text, encoding="utf-8")
    return len(artifacts)
