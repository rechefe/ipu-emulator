#!/usr/bin/env python3
"""Bazel wrapper: generate instruction-format C header and SystemVerilog package."""

import sys
from pathlib import Path

from ipu_as.gen_codegen import generate_c_header, generate_sv_package

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: gen_codegen_wrapper.py <output.h> <output.sv>",
            file=sys.stderr,
        )
        sys.exit(1)
    generate_c_header(Path(sys.argv[1]))
    generate_sv_package(Path(sys.argv[2]))
