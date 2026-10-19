#!/usr/bin/env python3
"""Judge every text test/verdicts.js wrote, and hold the editor's verdict to the
assembler's: each must accept a text exactly when the other does, but for the
reviewed gaps in test/corpus/negative/allowed-gaps.json.

The assembler's verdict is ``ipu_as.diagnostics.check``, what ``ipu-as check
--json`` prints: it accepts a text when there are no diagnostics. A listed gap
that no longer occurs is an error too, so the list stays reviewed.

allowed-gaps.json is a list of ``{"case": <the case's name in verdicts.json>,
"editor": "accepts" | "rejects", "why": <the reason it is allowed>}``; it is
reviewed with every change to it.

    bazel run //vscode-ipu-asm:check_verdicts -- <verdicts.json>
"""

import json
import pathlib
import sys
import warnings

import ipu_as.diagnostics as diagnostics

GAPS = pathlib.Path(__file__).parent / "test" / "corpus" / "negative" / "allowed-gaps.json"


def assembler_reasons(text):
    """The first line of each diagnostic the assembler gives ``text``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # simulation-only slot notices
        return [d.message.splitlines()[0] for d in diagnostics.check(text)]


def main(argv):
    if len(argv) != 1:
        print("usage: check_verdicts.py <verdicts.json>", file=sys.stderr)
        return 2
    cases = json.loads(pathlib.Path(argv[0]).read_text(encoding="utf-8"))
    listed = json.loads(GAPS.read_text(encoding="utf-8"))
    gaps = {gap["case"]: gap for gap in listed}
    if len(gaps) != len(listed):
        print(f"{GAPS.name} lists a case more than once", file=sys.stderr)
        return 1
    failures, gapped, rejected = [], set(), 0
    for case in cases:
        editor, assembler = case["reasons"], assembler_reasons(case["text"])
        rejected += bool(assembler)
        if bool(editor) == bool(assembler):
            continue
        side = "rejects" if editor else "accepts"
        gap = gaps.get(case["name"])
        if gap is not None and gap["editor"] == side:
            gapped.add(case["name"])
            continue
        why = editor[0] if editor else assembler[0]
        failures.append(f"{case['name']}: the editor {side} it, the assembler does not ({why})")
    failures += [
        f"{name}: listed as an allowed gap, but the two agree" for name in gaps.keys() - gapped
    ]
    if failures:
        print(
            f"{len(failures)} verdict(s) disagree:",
            *("  " + f for f in failures[:25]),
            sep="\n",
            file=sys.stderr,
        )
        return 1
    print(
        f"The editor and the assembler agree on {len(cases)} texts "
        f"({rejected} rejected, {len(gapped)} allowed gaps)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
