#!/usr/bin/env python3
"""py_binary entry point for dumping the parser's tokenization of a corpus."""

import sys

from ipu_as.dump_parser_tokens import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
