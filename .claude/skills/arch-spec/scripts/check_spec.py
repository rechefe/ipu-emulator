#!/usr/bin/env python3
"""Mechanical checks for an ARCH stage spec (docs/content/specs/ARCH-*.md).

Usage: python3 -I check_spec.py <spec.md>

Errors (exit 1): broken in-page anchor links, missing images, duplicated
numbered headings (usually a bad splice), missing Table of Contents.
Review items (exit 0): sentences worth a second look for the style rules
(negations, history, meta), tiny headings, TBD markers to report.
"""
import re
import sys
import unicodedata
from pathlib import Path


def slugify(value: str) -> str:
    # Same as python-markdown's toc slugify, which mkdocs uses.
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def strip_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> label
    return text.replace("`", "").replace("*", "")


REVIEW_PATTERNS = [
    (r"\bnot\b|\bn't\b", "negation: is the 'not B' part needed?"),
    (r"\bno longer\b|\bused to\b|\bpreviously\b|\bwas renamed\b|\bformerly\b", "history: say what is, not what was"),
    (r"\bthis section (describes|covers|explains|defines)\b|\bas described above\b|\bas mentioned\b|\bsee below\b", "meta: cut it or replace with a section number"),
    (r"\bin other words\b|\bthat is to say\b|\bnote that\b", "restatement: say it once"),
]


def main(path_str: str) -> int:
    path = Path(path_str)
    lines = path.read_text().split("\n")

    headings = []          # (line_no, level, text, slug)
    body = []              # (line_no, text) outside code fences
    in_fence = False
    for n, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
        if m:
            text = strip_inline(m.group(2))
            headings.append((n, len(m.group(1)), text, slugify(text)))
        else:
            body.append((n, line))

    errors, review = [], []

    # Anchors, with mkdocs' _1, _2 suffixes for repeated slugs.
    anchors, seen = set(), {}
    for _, _, _, slug in headings:
        if slug in seen:
            seen[slug] += 1
            anchors.add(f"{slug}_{seen[slug]}")
        else:
            seen[slug] = 0
            anchors.add(slug)

    for n, line in body:
        for target in re.findall(r"\]\(#([^)\s]+)\)", line):
            if target not in anchors:
                errors.append(f"{n}: link #{target} matches no heading")
        for img in re.findall(r"!\[[^\]]*\]\(([^)\s#]+)\)", line):
            if not img.startswith("http") and not (path.parent / img).exists():
                errors.append(f"{n}: image {img} does not exist")

    numbered = {}
    for n, level, text, _ in headings:
        m = re.match(r"^(\d+(?:\.\d+)*)\.?\s", text)
        if m:
            numbered.setdefault(m.group(1), []).append(n)
    for num, where in numbered.items():
        if len(where) > 1:
            errors.append(f"heading number {num} appears {len(where)} times (lines {where}): duplicated section?")

    if not any(t.lower() == "table of contents" for _, _, t, _ in headings):
        errors.append("no 'Table of Contents' heading")

    for n, level, text, _ in headings:
        if level == 6:
            review.append(f"{n}: '###### {text}' renders very small; use ##### or restructure")
        if text.lower() == "summary":
            review.append(f"{n}: 'Summary' heading; put the description directly under the instruction title")

    for n, line in body:
        if line.lstrip().startswith("![") or "| Not used. |" in line:
            continue  # images are not prose; "Not used." is the N/A-row convention
        for pattern, why in REVIEW_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                review.append(f"{n}: {why}\n      {line.strip()[:110]}")
                break
        if "TBD" in line:
            review.append(f"{n}: TBD to report to the user\n      {line.strip()[:110]}")

    print(f"{path}: {len(headings)} headings, {len(errors)} errors, {len(review)} review items")
    for e in errors:
        print(f"  ERROR  {e}")
    for r in review:
        print(f"  review {r}")
    return 1 if errors else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
