"""The generated coverage tables of the Kernel coverage page (E09-10).

    bazel run //docs:kernel_coverage_page -- "$PWD/out"

writes ``kernels.md`` into the given directory: every operation of the
``kernels/`` routing registry (:func:`ipu_harness.registry.default_registry`)
and every kernel that computes it, with its Bazel label, formats and one-line
description, then every ``kernel.toml`` or ``registry.toml`` that did not load.

``explanation/kernel-coverage.md`` includes it through ``pymdownx.snippets``.
The table is read from the registry, never written by hand, so a kernel added
or removed changes the page.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from ipu_harness.registry import Registry, default_registry

KERNELS_FILE = "kernels.md"


def _cell(text: str) -> str:
    """Text safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def kernels_table(registry: Registry) -> str:
    """The ``kernels/`` coverage table.

    Args:
        registry: The kernels.

    Returns:
        Markdown: one row per kernel, grouped by operation, then a list of the
        files that did not load (a kernel missing from coverage).
    """
    lines = [
        f"{len(registry.manifests)} kernels, {len(registry.operations())} operations.",
        "",
        "| Operation | Kernel | Bazel target | Formats | Description |",
        "|---|---|---|---|---|",
    ]
    for op in sorted(registry.operations()):
        for m in sorted(registry.kernels(op), key=lambda m: m.name):
            formats = ", ".join(f.value for f in m.formats)
            lines.append(
                f"| `{op}` | `{m.name}` | `{m.label}` | {formats} | {_cell(m.doc or '')} |"
            )
    if registry.invalid:
        lines += ["", "These files did not load, so their kernels are missing above:", ""]
        lines += [
            f"- `{path}`: {_cell(str(error))}"
            for path, error in sorted(registry.invalid.items(), key=lambda item: str(item[0]))
        ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Write the snippet into the directory ``argv[0]`` names."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: kernel_coverage_page OUT_DIR", file=sys.stderr)
        return 2
    out = Path(args[0])
    out.mkdir(parents=True, exist_ok=True)
    (out / KERNELS_FILE).write_text(kernels_table(default_registry()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
