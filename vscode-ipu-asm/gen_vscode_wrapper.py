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
from ipu_as.reg import LRD_REG_FIELDS
from ipu_emu.ipu import Ipu
from ipu_emu.ipu_config import CR_READ_ONLY_INITIAL_VALUES, CR_REGISTER_NAME

# What only the emulator knows, for register hovers: fixed registers (CR0 = 0,
# CR1 = 1), and the LR pair behind each LRD name (indexed by its enum position).
EMULATOR = {
    "fixed": {f"{CR_REGISTER_NAME}{i}": v for i, v in CR_READ_ONLY_INITIAL_VALUES.items()},
    "pairs": {name: Ipu._lrd_lr_indices(n) for n, name in enumerate(LRD_REG_FIELDS)},
}

if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) == 1:
        generate_all(Path(argv[0]), EMULATOR)
    elif not argv:
        workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
        if not workspace:
            print(
                "No output directory given and BUILD_WORKSPACE_DIRECTORY is unset.\n"
                "Run via `bazel run //vscode-ipu-asm:gen_vscode`, or pass a directory.",
                file=sys.stderr,
            )
            sys.exit(1)
        generate_all(Path(workspace) / "vscode-ipu-asm", EMULATOR)
    else:
        print("Usage: gen_vscode_wrapper.py [<output_directory>]", file=sys.stderr)
        sys.exit(1)
