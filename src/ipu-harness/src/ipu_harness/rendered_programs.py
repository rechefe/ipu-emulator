"""Every kernel's default-case program, rendered, as ``.asm`` files (E07-25).

    bazel run //kernels:rendered_programs -- OUT_DIR

writes ``OUT_DIR/<family>/<name>.asm`` for each kernel and registry entry of
the kernel tree: the program ``:assemble_<name>`` assembles (the default case,
its first format; :func:`ipu_harness.kernel_bin.default_programs`), with every
template value filled in. A ``kernels/`` source renders only with a case, so a
tool that reads program text without the harness reads these: the editor's
grammar agreement (``.github/workflows/vscode-extension.yml``) tokenizes them.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from ipu_emu.errors import ConfigurationError
from ipu_harness.kernel_bin import default_programs
from ipu_harness.registry import Registry, kernel_tree
from ipu_isa.errors import IpuError


def write_rendered(out_dir: Path, root: Path | None = None) -> list[Path]:
    """Write every kernel's rendered default program under ``out_dir``.

    Args:
        out_dir: The directory to write into (created if missing).
        root: The kernel tree (default: :func:`ipu_harness.registry.kernel_tree`).

    Returns:
        The files written, sorted.

    Raises:
        ConfigurationError: A kernel of the tree does not load.
        ipu_as.errors.AssemblerError: A program does not render or assemble.
    """
    registry = Registry.load(root or kernel_tree())
    for path, error in registry.invalid.items():
        raise ConfigurationError(f"{path}: {error}")
    written = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*-only .* slot")
        for name, manifest in sorted(registry.manifests.items()):
            program = next(iter(default_programs(manifest).values()))
            path = out_dir / manifest.directory.parent.name / f"{name}.asm"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(program.rendered, encoding="utf-8")
            written.append(path)
    return sorted(written)


def main(argv: Sequence[str] | None = None) -> int:
    """The command line. Returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="rendered_programs.py", description=__doc__.splitlines()[0]
    )
    parser.add_argument("out_dir", type=Path, help="the directory to write into")
    args = parser.parse_args(argv)
    try:
        written = write_rendered(args.out_dir)
    except (IpuError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {len(written)} programs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
