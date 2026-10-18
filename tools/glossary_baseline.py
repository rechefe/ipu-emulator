"""Rewrite tools/glossary_baseline.json: `bazel run //tools:glossary_baseline`."""

import sys

from tools import glossary_check

if __name__ == "__main__":
    sys.exit(glossary_check.main())
