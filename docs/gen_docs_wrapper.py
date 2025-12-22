#!/usr/bin/env python3
"""Wrapper script to call gen_docs from ipu_as module."""

import sys
from pathlib import Path
from ipu_as.gen_docs import generate_instruction_docs

if __name__ == "__main__":
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_instruction_docs(output_dir / "instructions.md")
