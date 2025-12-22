#!/usr/bin/env python3
"""Wrapper script to call mkdocs."""

import sys
from mkdocs.__main__ import cli

if __name__ == "__main__":
    sys.exit(cli())
