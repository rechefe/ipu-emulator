"""Keep each family's generated ``registry.bzl`` in step with its ``registry.toml`` (E06-17).

    bazel run //kernels:update_registry_bzl [-- ROOT ...]   # rewrite the stale ones under ROOTs
    registry_bzl.py check REGISTRY.toml REGISTRY.bzl          # ipu_kernel_family()'s test

Bazel cannot read TOML at load time, so ``ipu_kernel_family()`` loads the
entries from the checked-in ``registry.bzl`` beside the TOML
(:func:`ipu_harness.family.render_bzl`). ``update`` rewrites every one under
the roots (default: the workspace's ``kernels/``) and names the files it
changed; ``check`` fails when the file is not what ``update`` would write.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from ipu_emu.errors import ConfigurationError
from ipu_harness.family import BZL_NAME, REGISTRY_NAME, render_bzl

UPDATE = "bazel run //kernels:update_registry_bzl"
USAGE = (
    "usage: registry_bzl.py check REGISTRY.toml REGISTRY.bzl\n"
    "       registry_bzl.py update [ROOT ...]"
)


def check(registry: Path, bzl: Path) -> str | None:
    """Why ``bzl`` is stale, or None when it is what ``update`` writes."""
    wanted = render_bzl(registry)
    have = bzl.read_text(encoding="utf-8") if bzl.is_file() else None
    if have == wanted:
        return None
    return f"{bzl}: stale, not generated from {registry.name}; run `{UPDATE} -- {registry.parent}`"


def update(roots: Sequence[Path]) -> list[Path]:
    """Rewrite every stale ``registry.bzl`` under ``roots``; returns those it wrote."""
    written = []
    for root in roots:
        for registry in sorted(root.rglob(REGISTRY_NAME)):
            bzl = registry.with_name(BZL_NAME)
            if check(registry, bzl) is not None:
                bzl.write_text(render_bzl(registry), encoding="utf-8")
                written.append(bzl)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """The command line. Returns the exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(args) == 3 and args[0] == "check":
            stale = check(Path(args[1]), Path(args[2]))
            if stale is not None:
                print(stale, file=sys.stderr)
                return 1
            return 0
        if args and args[0] == "update":
            workspace = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", "."))
            roots = [workspace / root for root in args[1:]] or [workspace / "kernels"]
            for bzl in update(roots):
                print(bzl)
            return 0
    except ConfigurationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
