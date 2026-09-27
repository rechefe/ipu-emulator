#!/usr/bin/env python3
"""Emit the parser's own tokenization of a corpus, as ground truth for the
TextMate agreement test.

The editor and the assembler must agree about what every byte of a program is.
Only the parser can say authoritatively, so this dumps exactly what it sees:
token boundaries from ``lex(dont_ignore=True)`` (which covers whitespace and
comments too), and each token's structural role from the parse tree.

Note the text emitted here is **Jinja-rendered**. ``lark_tree`` renders the
template away before parsing, so the parser never sees the raw file, and byte
offsets only line up on the rendered form. The agreement test therefore
compares on rendered text; the Jinja layer is covered separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jinja2
import lark

from ipu_as.lark_tree import get_parser

#: Parse-tree rule -> role for the token it owns.
_ROLE_RULES = {"label": "LABEL", "operand": "OPERAND"}


def token_roles(tree: lark.Tree) -> dict[int, str]:
    """Map each token's start offset to LABEL / MNEMONIC / OPERAND."""
    roles: dict[int, str] = {}
    for sub in tree.iter_subtrees():
        if sub.data in _ROLE_RULES:
            token = sub.children[0]
            roles[token.start_pos] = _ROLE_RULES[sub.data]
        elif sub.data == "instruction":
            # The one direct Token child is the mnemonic; operands are subtrees.
            for child in sub.children:
                if isinstance(child, lark.Token):
                    roles[child.start_pos] = "MNEMONIC"
                    break
    return roles


def describe(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    text = jinja2.Template(raw).render() if any(
        m in raw for m in ("{{", "{%", "{#")
    ) else raw

    parser = get_parser()
    roles = token_roles(parser.parse(text))

    tokens = [
        {
            "start": t.start_pos,
            "end": t.end_pos,
            "terminal": t.type,
            "text": str(t),
            "role": roles.get(t.start_pos),
        }
        for t in parser.lex(text, dont_ignore=True)
    ]

    covered = sum(t["end"] - t["start"] for t in tokens)
    if covered != len(text):
        raise SystemExit(
            f"{path}: lexer covered {covered} of {len(text)} bytes; the agreement "
            f"test needs total coverage to compare against TextMate."
        )

    return {"file": str(path), "text": text, "tokens": tokens}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "Usage: dump_parser_tokens.py <output.json> <corpus.asm>...",
            file=sys.stderr,
        )
        return 1

    out = Path(argv[0])
    corpus = [describe(Path(p)) for p in argv[1:]]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(corpus, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
