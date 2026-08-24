"""Print instruction_aligned_bytes_len() for Bazel codegen rules."""

from __future__ import annotations

import sys

from ipu_as.lark_tree import instruction_aligned_bytes_len


def main() -> None:
    value = str(instruction_aligned_bytes_len())
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="ascii") as fh:
            fh.write(value)
    else:
        print(value)


if __name__ == "__main__":
    main()
