#!/usr/bin/env python3
"""Check a vendored ipu_instr_pkg.sv against the content hash in its header.

    bazel run //docs:verify_sv_package -- path/to/ipu_instr_pkg.sv

Exits 0 when the body is exactly what the header's hash names, 1 when it was
edited, 2 on a usage error.
"""

import os
import sys
from pathlib import Path

from ipu_gen.sv_package import verify_sv_package

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: verify_sv_package <ipu_instr_pkg.sv>", file=sys.stderr)
        sys.exit(2)
    # `bazel run` starts in the runfiles tree; a relative path is the caller's.
    path = Path(os.environ.get("BUILD_WORKING_DIRECTORY", ".")) / sys.argv[1]
    # Bytes, not text mode: a copy with its line endings rewritten is not unmodified.
    if verify_sv_package(path.read_bytes().decode("utf-8")):
        print(f"{path}: unmodified")
        sys.exit(0)
    print(f"{path}: does not match the content hash in its header", file=sys.stderr)
    sys.exit(1)
