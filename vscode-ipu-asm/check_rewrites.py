#!/usr/bin/env python3
"""Assemble every rewrite test/rewrites.js wrote: neither names nor layout reach
the encoding, so a correct rename or format assembles to the original's bytes.

    bazel run //vscode-ipu-asm:check_rewrites -- <rewrites.json>
"""

import contextlib
import io
import json
import pathlib
import sys
import warnings

import ipu_as.lark_tree as lark_tree


def assemble(text):
    """The encoded words, or the error that stopped assembly."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # simulation-only slot notices
            return lark_tree.assemble(text), None
    except SystemExit:
        return None, (out.getvalue().strip().splitlines() or ["assembly failed"])[-1]
    except Exception as exc:  # the template layer raises its own errors
        return None, f"{type(exc).__name__}: {exc}"


def main(argv):
    if len(argv) != 1:
        print("usage: check_rewrites.py <rewrites.json>", file=sys.stderr)
        return 2
    files = json.loads(pathlib.Path(argv[0]).read_text(encoding="utf-8"))
    failures, checked = [], 0
    for entry in files:
        expected, error = assemble(entry["original"])
        if error:
            failures.append(f"{entry['file']}: the original does not assemble: {error}")
            continue
        for v in entry["variants"]:
            what = f"{entry['file']}: {v['kind']} {v['name']}"
            if "error" in v:
                failures.append(f"{what}: {v['error']}")
                continue
            checked += 1
            words, error = assemble(v["text"])
            if error or words != expected:
                failures.append(f"{what} ({v['uses']} uses) {'breaks assembly: ' + error if error else 'changes the binary'}")
    if failures:
        head = f"{len(failures)} rewrite(s) would change the program:"
        print(head, *("  " + f for f in failures[:25]), sep="\n", file=sys.stderr)
        return 1
    print(f"Every rewrite leaves the program unchanged: {checked} rewrites across {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
