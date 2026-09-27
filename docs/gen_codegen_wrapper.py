#!/usr/bin/env python3
"""Bazel wrapper: generate instruction-format SystemVerilog package."""

import sys
from pathlib import Path

from ipu_as.gen_codegen import generate_sv_package, source_commit_from_status

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: gen_codegen_wrapper.py <output.sv> <stable-status.txt>",
            file=sys.stderr,
        )
        sys.exit(1)
    status = Path(sys.argv[2]).read_text(encoding="utf-8")
    generate_sv_package(Path(sys.argv[1]), source_commit_from_status(status))
