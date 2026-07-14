#!/usr/bin/env python3
"""Bazel wrapper: generate instruction-format MkDocs page."""

import sys
from pathlib import Path

from ipu_as.gen_docs import generate_instruction_format_md

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: gen_instruction_format_md_wrapper.py <instruction-format.md>",
            file=sys.stderr,
        )
        sys.exit(1)
    generate_instruction_format_md(Path(sys.argv[1]))
