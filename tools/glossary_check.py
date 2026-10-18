"""Glossary adoption check: docs do not add uses of terms `GLOSSARY.md` avoids.

`GLOSSARY.md` ends some entries with an `_Avoid_:` line listing discouraged
synonyms. This module reads those lists, counts each avoided term in the
hand-written Markdown (outside code), and compares the counts with
`tools/glossary_baseline.json`. A count may go down but never up; a lower count
must be written back to the baseline so it ratchets.

Run as a program (`bazel run //:glossary_baseline`) it rewrites the baseline
from the working tree. The test is `//:glossary_adoption_test`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

GLOSSARY = Path("GLOSSARY.md")
BASELINE = Path("tools/glossary_baseline.json")
BASELINE_COMMAND = "bazel run //:glossary_baseline"

# Root files checked besides docs/content/**/*.md (the same set as the doc tests).
ROOT_FILES = ("README.md", "SKILL.md", "CLAUDE.md")

# Pages //docs:generate_mkdocs_md writes into docs/content/ (git-ignored). They
# are not hand-written; E08 (ipu_gen) owns their wording.
GENERATED_PAGES = frozenset(
    {
        "docs/content/instructions.md",
        "docs/content/operand-types.md",
        "docs/content/assembly-syntax.md",
        "docs/content/programmer-guide.md",
    }
)

# Committed pages a workflow rewrites, not a person. .github/workflows/release-log.yml
# rebuilds the release log from merged pull request titles and pushes it to
# master without touching the baseline; the titles are historical records that
# nobody can reword.
MACHINE_WRITTEN_PAGES = frozenset({"docs/content/release-log.md"})

# Avoided terms that are ordinary English words. A regex cannot tell the
# discouraged sense from the ordinary one, so they are not counted. This list is
# the only place to tune the check.
SKIPPED_TERMS: Mapping[str, str] = {
    "line": "a line of text or of a table, not only an instruction word or row",
    "word": "any machine word or English word, not only an XMEM row",
    "op": "an `op` field and a binary op are real code names; the avoid is for prose",
    "inst": "a short code identifier; in prose it is almost always a variable name left unquoted",
    "instr": "a short code identifier; in prose it is almost always a variable name left unquoted",
    "app": "app is a key of the registry verdict's JSON, which tools read",
    "application": "ordinary English (an application of a rule)",
    "harness": "the Harness is itself a glossary term; only its use for a kernel is avoided",
    "buffer": "ordinary English and a host-side buffer, not only an XMEM region",
    "image": "a picture or Docker image, not only an XMEM region",
    "blob": "a binary file in general, not only an XMEM region",
    "exception": "a Python exception is not a fault",
    "trap": "ordinary English (a pitfall)",
    "scenario": "ordinary English (a use scenario)",
    "lane": "the Lane is itself a glossary term; only its use for a slot is avoided",
    "chunk": "ordinary English for a part of a file or loop, not only an XMEM row",
}

_ENTRY = re.compile(r"^\*\*(?P<term>.+?)\*\*:\s*$")
_AVOID = re.compile(r"^_Avoid_:\s*(?P<list>.+?)\s*$")
_QUALIFIER = re.compile(r"^(?P<term>.+?)\s*\((?P<qualifier>[^)]*)\)$")
_FENCE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"(?P<ticks>`+)(?:(?!(?P=ticks)).)+?(?P=ticks)", re.DOTALL)
_LINK_TARGET = re.compile(r"\]\([^)\s]*(?:\s+\"[^\"]*\")?\)")


@dataclass(frozen=True)
class AvoidedTerm:
    #: The avoided term as the glossary writes it, without its qualifier.
    term: str
    #: The glossary entry to use instead.
    use_instead: str


@dataclass(frozen=True)
class Violation:
    path: str
    term: str
    message: str


def parse_glossary(text: str) -> list[AvoidedTerm]:
    """Every checked avoided term in `GLOSSARY.md`, in glossary order.

    A term with a parenthesised qualifier is checked only when the qualifier is
    `in prose`; other qualifiers need context a regex cannot see. Terms in
    SKIPPED_TERMS are left out.
    """
    terms: list[AvoidedTerm] = []
    seen: set[str] = set()
    current: str | None = None
    for line in text.splitlines():
        entry = _ENTRY.match(line)
        if entry:
            current = entry.group("term")
            continue
        avoid = _AVOID.match(line)
        if not avoid or current is None:
            continue
        for raw in avoid.group("list").split(","):
            item = raw.strip()
            qualified = _QUALIFIER.match(item)
            if qualified:
                if qualified.group("qualifier").strip().lower() != "in prose":
                    continue
                item = qualified.group("term").strip()
            if not item or item.lower() in SKIPPED_TERMS or item.lower() in seen:
                continue
            seen.add(item.lower())
            terms.append(AvoidedTerm(term=item, use_instead=current))
    return terms


def _blank(text: str) -> str:
    """Replace every character except newlines with a space."""
    return re.sub(r"[^\n]", " ", text)


def prose(text: str) -> str:
    """`text` with fenced code, inline code and link targets blanked out.

    Line structure is kept, so an offset in the result is a line in `text`.
    """
    out: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        match = _FENCE.match(line)
        if fence is None and match:
            fence = match.group("fence")
            out.append(_blank(line))
        elif fence is not None:
            if (
                match
                and match.group("fence")[0] == fence[0]
                and len(match.group("fence")) >= len(fence)
            ):
                if not line.strip().lstrip(fence[0]):
                    fence = None
            out.append(_blank(line))
        else:
            out.append(line)
    result = "".join(out)
    result = _INLINE_CODE.sub(lambda m: _blank(m.group(0)), result)
    return _LINK_TARGET.sub(lambda m: "](" + _blank(m.group(0)[2:]), result)


def _pattern(term: str) -> re.Pattern[str]:
    """Whole-word, case-insensitive match of `term`, allowing a plural and line breaks."""
    words = r"\s+".join(re.escape(word) for word in term.split())
    return re.compile(rf"(?<!\w){words}(?:e?s)?(?!\w)", re.IGNORECASE)


def find_uses(text: str, terms: Iterable[AvoidedTerm]) -> dict[str, list[int]]:
    """For each avoided term used in `text`, the 1-based lines of its uses."""
    checked = prose(text)
    uses: dict[str, list[int]] = {}
    for avoided in terms:
        lines = [
            checked.count("\n", 0, m.start()) + 1 for m in _pattern(avoided.term).finditer(checked)
        ]
        if lines:
            uses[avoided.term] = lines
    return uses


def doc_files(root: Path) -> list[str]:
    """Every checked Markdown file under `root`, as a repository-relative path."""
    content = sorted(
        path.relative_to(root).as_posix() for path in (root / "docs/content").rglob("*.md")
    )
    content = [
        path
        for path in content
        if path not in GENERATED_PAGES and path not in MACHINE_WRITTEN_PAGES
    ]
    return content + [name for name in ROOT_FILES if (root / name).is_file()]


def scan(root: Path, terms: list[AvoidedTerm]) -> dict[str, dict[str, list[int]]]:
    """Uses of each avoided term in each checked file: {path: {term: [line, ...]}}."""
    result: dict[str, dict[str, list[int]]] = {}
    for path in doc_files(root):
        uses = find_uses((root / path).read_text(encoding="utf-8"), terms)
        if uses:
            result[path] = uses
    return result


def counts(uses: Mapping[str, Mapping[str, list[int]]]) -> dict[str, dict[str, int]]:
    """The baseline form of `uses`: {path: {term: count}}, sorted."""
    return {
        path: {term: len(lines) for term, lines in sorted(per_file.items())}
        for path, per_file in sorted(uses.items())
    }


def compare(
    uses: Mapping[str, Mapping[str, list[int]]],
    baseline: Mapping[str, Mapping[str, int]],
    terms: list[AvoidedTerm],
) -> list[Violation]:
    """Differences between the current uses and the baseline counts.

    A count above the baseline (or a use with no baseline entry) names each
    use's `file:line`, the avoided term and the glossary term to use instead. A
    count below the baseline asks for the baseline to be lowered.
    """
    use_instead = {avoided.term: avoided.use_instead for avoided in terms}
    violations: list[Violation] = []
    paths = sorted(set(uses) | set(baseline))
    for path in paths:
        current = uses.get(path, {})
        expected = baseline.get(path, {})
        for term in sorted(set(current) | set(expected)):
            lines = current.get(term, [])
            allowed = expected.get(term, 0)
            if len(lines) > allowed:
                where = ", ".join(f"{path}:{line}" for line in lines)
                violations.append(
                    Violation(
                        path,
                        term,
                        f"{path}: {len(lines)} use(s) of avoided term {term!r}, baseline "
                        f"{allowed}; use the glossary term {use_instead.get(term, '?')!r} "
                        f"instead. Uses: {where}",
                    )
                )
            elif len(lines) < allowed:
                violations.append(
                    Violation(
                        path,
                        term,
                        f"{path}: {len(lines)} use(s) of {term!r}, baseline {allowed}. "
                        f"Uses went down, so lower the baseline: run `{BASELINE_COMMAND}`",
                    )
                )
    return violations


def check(root: Path, baseline_path: Path | None = None) -> list[Violation]:
    """Check the tree at `root` against its baseline."""
    terms = parse_glossary((root / GLOSSARY).read_text(encoding="utf-8"))
    baseline = json.loads((baseline_path or root / BASELINE).read_text(encoding="utf-8"))
    return compare(scan(root, terms), baseline, terms)


def write_baseline(root: Path, baseline_path: Path | None = None) -> dict[str, dict[str, int]]:
    """Rewrite the baseline from the tree at `root` and return it."""
    terms = parse_glossary((root / GLOSSARY).read_text(encoding="utf-8"))
    baseline = counts(scan(root, terms))
    target = baseline_path or root / BASELINE
    target.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    return baseline


def main() -> int:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not workspace:
        print(f"run through Bazel: {BASELINE_COMMAND}", file=sys.stderr)
        return 2
    root = Path(workspace)
    baseline = write_baseline(root)
    total = sum(sum(per_file.values()) for per_file in baseline.values())
    print(f"wrote {BASELINE}: {total} use(s) in {len(baseline)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
