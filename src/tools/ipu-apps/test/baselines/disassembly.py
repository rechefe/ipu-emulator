"""Disassembly snapshots of the assembled-program corpus (E02-02).

One text file per ``asm_corpus`` entry under ``disassembly/``, one line per
instruction word: ``<index>: <ipu_isa.codec.disasm.disassemble(word)>``, with the
line breaks between slots folded into single spaces. Text rather than a digest,
so a mnemonic rename or a formatting change shows as a readable diff, local to
the programs it touches.

``test_disassembly_snapshots`` compares them;
``bazel run //src/tools/ipu-apps:update_disassembly_snapshots`` rewrites them.
"""

from __future__ import annotations

import re
from pathlib import Path

import asm_corpus

from ipu_isa.codec.disasm import disassemble as disassemble_word

#: Where the snapshots live, repo-relative.
SNAPSHOT_DIR = "src/tools/ipu-apps/test/baselines/disassembly"

#: The one command that rewrites them from the current tree.
UPDATE_COMMAND = "bazel run //src/tools/ipu-apps:update_disassembly_snapshots"

#: Corpus root -> the short directory its snapshots go in.
_SHORT_ROOTS = {
    "src/tools/ipu-apps/src/ipu_apps/kernels": "kernels",
    "vscode-ipu-asm/test": "vscode",
}
if set(_SHORT_ROOTS) != set(asm_corpus.CORPUS_ROOTS):
    raise RuntimeError(
        f"_SHORT_ROOTS {sorted(_SHORT_ROOTS)} != CORPUS_ROOTS {sorted(asm_corpus.CORPUS_ROOTS)}"
    )


def snapshot_name(entry: asm_corpus.CorpusEntry) -> str:
    """The snapshot's path under ``SNAPSHOT_DIR``, for example
    ``kernels/softmax/softmax_rows/softmax_rows.txt`` or
    ``<root>/<program>.<define>-<value>.txt`` for a rendered entry."""
    for root, short in _SHORT_ROOTS.items():
        if entry.path.startswith(root + "/"):
            stem = short + entry.path[len(root) : -len(".asm")]
            break
    else:
        raise ValueError(f"{entry.path} is outside every corpus root")
    if entry.context_json is not None:
        stem += "." + re.sub(r"[^A-Za-z0-9_]+", "-", entry.context_json).strip("-")
    return stem + ".txt"


def disassemble(words: list[int]) -> str:
    lines = []
    for index, word in enumerate(words):
        text = " ".join(part.strip() for part in disassemble_word(word).splitlines())
        lines.append(f"{index}: {text}\n")
    return "".join(lines)


def snapshots(root: Path) -> dict[str, str]:
    """Snapshot name -> content, for the whole corpus under ``root``."""
    return {
        snapshot_name(entry): disassemble(asm_corpus.assemble(root, entry).words)
        for entry in asm_corpus.discover(root)
    }


def recorded(root: Path) -> set[str]:
    """Names of the snapshot files on disk."""
    directory = root / SNAPSHOT_DIR
    return {path.relative_to(directory).as_posix() for path in directory.rglob("*.txt")}


def write_all(root: Path) -> int:
    directory = root / SNAPSHOT_DIR
    for name in recorded(root):
        (directory / name).unlink()
    contents = snapshots(root)
    for name, text in contents.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text)
    return len(contents)
