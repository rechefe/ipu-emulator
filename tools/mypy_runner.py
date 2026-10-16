"""Run the locked mypy over one py_library, for a mypy_test (tools/typecheck.bzl).

    bazel test //src/tools/ipu-isa:mypy

Arguments (supplied by the macro):

    --config-file <mypy.ini> --manifest <json> [--strict]

The manifest lists the library's own sources and the first-party import roots
of the library and its deps, both relative to the workspace root. The roots go
into MYPYPATH, so imports of other repository packages resolve; third-party
packages resolve from this process's sys.path, which Bazel built from the same
deps. mypy runs in-process, so it sees exactly that sys.path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from mypy import api


def mypy_args(config: str, manifest: dict, strict: bool, cache_dir: str) -> list[str]:
    args = ["--config-file", config, "--cache-dir", cache_dir]
    if strict:
        args.append("--strict")
    # Sources stay relative to the workspace root (the test's working
    # directory) so errors read `src/tools/<pkg>/...py:<line>: error: ...`.
    return args + list(manifest["sources"])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--strict", action="store_true")
    opts = parser.parse_args(argv)

    manifest = json.loads(Path(opts.manifest).read_text())
    # A py_test runs in the workspace root inside its runfiles, the directory
    # the manifest paths are relative to.
    os.environ["MYPYPATH"] = os.pathsep.join(os.path.abspath(r) for r in manifest["roots"])
    cache_dir = os.path.join(os.environ.get("TEST_TMPDIR", "/tmp"), "mypy_cache")

    args = mypy_args(opts.config_file, manifest, opts.strict, cache_dir)
    mode = "strict" if opts.strict else "default"
    print(f"$ mypy ({mode}) <{len(manifest['sources'])} files>", flush=True)
    stdout, stderr, status = api.run(args)
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
