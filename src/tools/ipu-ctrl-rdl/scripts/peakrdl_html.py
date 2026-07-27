#!/usr/bin/env python3
"""Run peakrdl-html and package the site as a tar archive."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdl", type=Path, required=True)
    parser.add_argument("--top", default="ipu_host")
    parser.add_argument("--output", type=Path, required=True, help="Output .tar path")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "html"
        cmd = [
            sys.executable,
            "-m",
            "peakrdl",
            "html",
            "-o",
            str(out_dir),
            "-t",
            args.top,
            str(args.rdl),
        ]
        subprocess.run(cmd, check=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.output, "w") as tar:
            tar.add(out_dir, arcname=".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
