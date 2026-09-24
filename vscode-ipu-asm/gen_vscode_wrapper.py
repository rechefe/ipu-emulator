#!/usr/bin/env python3
"""py_binary entry point for generating the VS Code extension's grammar.

With no argument this writes into the source tree, which is safe because
`bazel run` is unsandboxed and exports BUILD_WORKSPACE_DIRECTORY:

    bazel run //vscode-ipu-asm:gen_vscode
"""

import os
import sys
from pathlib import Path

from ipu_as.gen_vscode import generate_all

if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) == 1:
        generate_all(Path(argv[0]))
    elif not argv:
        workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
        if not workspace:
            print(
                "No output directory given and BUILD_WORKSPACE_DIRECTORY is unset.\n"
                "Run via `bazel run //vscode-ipu-asm:gen_vscode`, or pass a directory.",
                file=sys.stderr,
            )
            sys.exit(1)
        generate_all(Path(workspace) / "vscode-ipu-asm")
    else:
        print("Usage: gen_vscode_wrapper.py [<output_directory>]", file=sys.stderr)
        sys.exit(1)
