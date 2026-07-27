#!/usr/bin/env python3
"""Wrapper script to extract the generated docs site and serve it locally."""

import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: serve_docs_wrapper.py <site.tar> [port]", file=sys.stderr)
        sys.exit(1)

    tar_path = Path(sys.argv[1])
    port = sys.argv[2] if len(sys.argv) > 2 else "8000"

    site_dir = Path(tempfile.mkdtemp(prefix="ipu_docs_site_"))
    with tarfile.open(tar_path) as tar:
        tar.extractall(site_dir)

    print(f"Serving docs from {site_dir} at http://localhost:{port}")
    try:
        subprocess.run(
            [sys.executable, "-m", "http.server", port],
            cwd=site_dir,
            check=True,
        )
    finally:
        shutil.rmtree(site_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
