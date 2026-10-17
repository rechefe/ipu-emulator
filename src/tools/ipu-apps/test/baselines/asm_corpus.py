"""Every assembly program in the repo, assembled the way its users assemble it.

This is the corpus the assembled-binary baseline pins (E02-01, see
``test/test_asm_digests.py``): the `.bin` and `.mem` bytes of every `.asm` file
under ``CORPUS_ROOTS``, plus one entry per render context that today's tests
pass to a program that only assembles correctly with one. A refactor that keeps
every digest proves the encoded programs are byte-identical. Each record also
keeps a short fingerprint per word, used only to name the first word that
changed.

Render contexts are copied here as literals, not imported, because the tests
that use them are manual whole-model runs. If one of those tests changes the
context it renders with, change the matching entry in ``RENDER_CONTEXTS`` in
the same PR.

Programs rendered per run by a kernel's own ``app.py`` (``conv_universal_wide384``,
``depthwise_conv_stride2_*``) are pinned here only in their standalone form,
with their ``| default(...)`` values; the per-case programs are pinned by the
kernel case baselines (E02-07).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ipu_as import lark_tree
from ipu_as.session import AssemblySession

#: Repo-relative directories whose `.asm` files (recursively) form the corpus.
CORPUS_ROOTS = (
    "src/tools/ipu-apps/src/ipu_apps/kernels",
    "vscode-ipu-asm/test",
)

#: Directories under a corpus root that hold no programs: the editor's negative
#: corpus, texts most of which do not assemble (vscode-ipu-asm check_verdicts;
#: its asm_fixtures filegroup leaves it out under Bazel).
NOT_PROGRAMS = ("vscode-ipu-asm/test/corpus/negative/",)

#: Programs that only assemble when rendered with a Jinja context, and the
#: contexts today's tests render them with. Each context is one corpus entry;
#: these programs have no standalone entry. None is left: the MobileViT packed
#: linears that needed one are //kernels/projections templates now (E07-23),
#: whose entries' assemble_* targets pin their programs.
RENDER_CONTEXTS: dict[str, tuple[dict, ...]] = {}

#: Programs that also assemble standalone (from their ``ipu:params`` header
#: defaults), and the other defines their harness tests assemble them with.
#: Each set of defines is one more corpus entry. ``conv_universal_wide384``'s
#: test sweeps rows=16 at width 384 and width 512 (cpr=4) at rows=4;
#: ``depthwise_conv_stride2_narrow``'s runs cols 64 and 16. (rows=16 assembles
#: the same words as the default rows=8: rows changes the program only at 1.
#: The entry records what the harness runs all the same.)
_CONVOLUTIONS = "src/tools/ipu-apps/src/ipu_apps/kernels/convolutions/"
HARNESS_DEFINES: dict[str, tuple[dict, ...]] = {
    _CONVOLUTIONS + "conv_universal_wide384/conv_universal_wide384.asm": (
        {"cpr": 3, "rows": 16},
        {"cpr": 4, "rows": 4},
    ),
    _CONVOLUTIONS + "depthwise_conv_stride2_narrow/depthwise_conv_stride2_narrow.asm": (
        {"elements_in_row": 16},
        {"elements_in_row": 64},
    ),
}

#: Where the digests live, repo-relative.
MANIFEST_PATH = "src/tools/ipu-apps/test/baselines/asm_digests.json"

#: The one command that rewrites the manifest from the current tree.
UPDATE_COMMAND = "bazel run //src/tools/ipu-apps:update_asm_digests"


@dataclass(frozen=True)
class CorpusEntry:
    """One program to assemble: a repo-relative `.asm` path and, optionally,
    the Jinja context a test renders it with."""

    path: str
    #: The context as canonical JSON (sorted keys, no spaces), or None.
    context_json: str | None = None

    @classmethod
    def rendered(cls, path: str, context: dict) -> CorpusEntry:
        return cls(path, json.dumps(context, sort_keys=True, separators=(",", ":")))

    @property
    def context(self) -> dict | None:
        return None if self.context_json is None else json.loads(self.context_json)

    @property
    def key(self) -> str:
        """The manifest key: the path, plus ``?<context-json>`` if rendered."""
        if self.context_json is None:
            return self.path
        return f"{self.path}?{self.context_json}"


@dataclass(frozen=True)
class AssembledProgram:
    """The two output formats of one entry, as the assembler writes them."""

    bin_bytes: bytes
    mem_bytes: bytes

    @property
    def words(self) -> list[int]:
        """The encoded instruction words, read back from the `.bin` bytes."""
        width = lark_tree.instruction_aligned_bytes_len()
        return [
            int.from_bytes(self.bin_bytes[i : i + width], byteorder="little")
            for i in range(0, len(self.bin_bytes), width)
        ]


def discover(root: Path) -> list[CorpusEntry]:
    """Every corpus entry found under ``root`` (a checkout or runfiles tree),
    sorted by key."""
    entries: list[CorpusEntry] = []
    for corpus_root in CORPUS_ROOTS:
        for asm in sorted((root / corpus_root).rglob("*.asm")):
            path = asm.relative_to(root).as_posix()
            if path.startswith(NOT_PROGRAMS):
                continue
            if path in RENDER_CONTEXTS:
                entries.extend(CorpusEntry.rendered(path, c) for c in RENDER_CONTEXTS[path])
            else:
                entries.append(CorpusEntry(path))
                entries.extend(CorpusEntry.rendered(path, c) for c in HARNESS_DEFINES.get(path, ()))
    return sorted(entries, key=lambda entry: entry.key)


def assemble(root: Path, entry: CorpusEntry) -> AssembledProgram:
    """Assemble one entry through the assembler's two file entry points.

    A context is rendered as defines for the source's ``ipu:params`` header,
    as the tests that use it render it; a standalone program goes to
    the assembler as is, which renders any Jinja layer itself. Each assembly
    has its own label table (``ipu_as.session``), so entries do not see each
    other's labels.
    """
    text = (root / entry.path).read_text()
    if entry.context is not None:
        text = AssemblySession(text, defines=entry.context).render()
    with tempfile.TemporaryDirectory() as tmp:
        bin_path = Path(tmp) / "program.bin"
        mem_path = Path(tmp) / "program.mem"
        lark_tree.assemble_to_bin_file(text, str(bin_path))
        lark_tree.assemble_to_mem_file(text, str(mem_path))
        return AssembledProgram(bin_path.read_bytes(), mem_path.read_bytes())


def word_fingerprints(words: list[int]) -> str:
    """A short per-word hash, 8 hex digits per word, concatenated.

    Only used to name the first word that changed; the full-file digests are
    what the baseline compares.
    """
    width = lark_tree.instruction_aligned_bytes_len()
    return "".join(
        hashlib.sha256(word.to_bytes(width, byteorder="little")).hexdigest()[:8] for word in words
    )


def record(program: AssembledProgram) -> dict:
    """The manifest record of one assembled program."""
    words = program.words
    return {
        "bin_sha256": hashlib.sha256(program.bin_bytes).hexdigest(),
        "mem_sha256": hashlib.sha256(program.mem_bytes).hexdigest(),
        "words": len(words),
        "word_fingerprints": word_fingerprints(words),
    }


def compute_manifest(root: Path) -> dict[str, dict]:
    """Assemble the whole corpus under ``root`` and return its manifest."""
    return {entry.key: record(assemble(root, entry)) for entry in discover(root)}


def load_manifest(root: Path) -> dict[str, dict]:
    return json.loads((root / MANIFEST_PATH).read_text())


def write_manifest(root: Path, manifest: dict[str, dict]) -> None:
    (root / MANIFEST_PATH).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def corpus_drift(manifest_keys: set[str], corpus_keys: set[str]) -> str | None:
    """Explain how the manifest and the corpus disagree, or None if they match."""
    unlisted = sorted(corpus_keys - manifest_keys)
    stale = sorted(manifest_keys - corpus_keys)
    if not unlisted and not stale:
        return None
    lines = []
    if unlisted:
        lines.append("Programs with no recorded digest:")
        lines.extend(f"  {key}" for key in unlisted)
    if stale:
        lines.append("Manifest entries whose program no longer exists:")
        lines.extend(f"  {key}" for key in stale)
    lines.append(f"Record the change in this PR with: {UPDATE_COMMAND}")
    return "\n".join(lines)


def first_differing_word(expected: dict, actual: dict) -> int | None:
    """Index of the first word whose fingerprint differs, or of the first word
    one program has and the other lacks; None if every word matches."""
    old, new = expected.get("word_fingerprints", ""), actual["word_fingerprints"]
    for index in range(0, max(len(old), len(new)), 8):
        if old[index : index + 8] != new[index : index + 8]:
            return index // 8
    return None


def digest_mismatch(key: str, expected: dict, actual: dict) -> str | None:
    """Explain how one entry's assembled output changed, or None if it did not."""
    changed = [
        name for name in ("bin_sha256", "mem_sha256", "words") if expected.get(name) != actual[name]
    ]
    if not changed:
        return None
    lines = [f"{key}: assembled output changed"]
    for name in changed:
        lines.append(f"  {name}: {expected.get(name)} -> {actual[name]}")
    index = first_differing_word(expected, actual)
    if index is not None:
        lines.append(f"  first differing word: {index}")
    lines.append(f"If this change is intended, record it with: {UPDATE_COMMAND}")
    return "\n".join(lines)
