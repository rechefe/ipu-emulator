"""Regenerate the whole-word compound instruction fields SVG.

Usage::

    bazel run //src/tools/ipu-as-py:gen_instruction_fields_layout_svg -- \\
        src/tools/ipu-as-py/instruction_fields_layout.svg
"""
from __future__ import annotations

import sys
from pathlib import Path

from ipu_as.compound_inst import CompoundInst


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(
            f"Usage: {sys.argv[0]} <output.svg>",
            file=sys.stderr,
        )
        return 2
    out_path = Path(argv[0])
    svg = CompoundInst.generate_struct_layout_svg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg)
    print(f"Wrote {len(svg):,} chars to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
