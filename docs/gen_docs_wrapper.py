#!/usr/bin/env python3
"""Wrapper script to regenerate MkDocs pages owned by the assembler toolchain."""

import sys
from pathlib import Path

from ipu_as.gen_docs import generate_all_docs

if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) == 4:
        generate_all_docs(Path(argv[0]), Path(argv[1]), Path(argv[2]), Path(argv[3]))
    elif len(argv) == 1:
        out_dir = Path(argv[0])
        out_dir.mkdir(parents=True, exist_ok=True)
        generate_all_docs(
            out_dir / "instructions.md",
            out_dir / "operand-types.md",
            out_dir / "assembly-syntax.md",
            out_dir / "programmer-guide.md",
        )
    else:
        print(
            "Usage: gen_docs_wrapper.py <instructions.md> <operand-types.md> <assembly-syntax.md> <programmer-guide.md>\n"
            "   or: gen_docs_wrapper.py <output_directory>/",
            file=sys.stderr,
        )
        sys.exit(1)
