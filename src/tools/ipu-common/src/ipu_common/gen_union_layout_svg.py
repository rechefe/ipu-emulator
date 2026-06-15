"""Standalone CLI to regenerate the union layout SVG to a file.

Usage::

    bazel run //src/tools/ipu-common:gen_union_layout_svg -- /tmp/union_layout.svg

The slot order matches the compound instruction binary layout (MSB → LSB).
"""
from __future__ import annotations

import sys
from pathlib import Path

from ipu_common.instruction_spec import (
    COMPOUND_LAYOUT_SLOT_ORDER,
    SLOT_COUNT,
    SLOT_UNIONS,
)
from ipu_common.union_layout_svg import render_union_layout_svg


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(
            f"Usage: {sys.argv[0]} <output.svg>",
            file=sys.stderr,
        )
        return 2
    out_path = Path(argv[0])
    svg = render_union_layout_svg(
        SLOT_UNIONS,
        slot_order=COMPOUND_LAYOUT_SLOT_ORDER,
        slot_counts=dict(SLOT_COUNT),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg)
    print(f"Wrote {len(svg):,} chars to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
