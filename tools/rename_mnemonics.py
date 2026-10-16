"""Rename the old mnemonics to the mnemonic standard (E03-19, Q33/Q45).

    bazel run //tools:rename_mnemonics -- [--check] [ROOT]   # ROOT: the workspace

Kept in the repository for open branches and downstream users: run it on a
checkout to move its sources to the new names. Opcodes are unchanged, so every
binary stays byte-identical.

* ``.asm``, ``.j2``, ``.py`` and ``.js`` files: whole-word, case-insensitive,
  keeping the case style (``ldr_mult_reg`` becomes ``ld.mult``), because they
  hold assembly text in either case.
* ``.md`` and ``.tex`` files: whole-word, upper case only.
* A name that a dot joins to another word (``BLT.S``, ``x.BLT``) is not
  touched, so nothing is renamed twice and the script is idempotent; a
  full stop after a name does not protect it; one right after a string
  escape such as ``\\n`` is renamed.
* ``.py`` files also rename the handlers (``execute_ldr_mult_reg`` becomes
  ``execute_ld_mult``).
* Skipped: ``planning/``, the release log, this script and its test, the
  rename table and its test, and everything under ``/test/baselines/``,
  ``/test/golden/`` and ``/test/fixtures/``. Regenerate the baselines with
  ``tools/update_baselines.sh``; rename hand-kept goldens and fixtures (the
  check-JSON and CLI-error goldens, TUI screens, the view snapshot) by hand
  or with ``rename_text``, keeping what they recorded.

Prints the number of replacements per changed file. ``--check`` prints the
same and changes nothing; it exits 1 if anything would change.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

#: Old mnemonic -> new mnemonic (Q45).
RENAMES = {
    "LDR_CYCLIC_MULT_REG": "LD.MULT.CYCLIC",
    "LDR_MULT_MASK_REG": "LD.MASK",
    "LDR_MULT_REG": "LD.MULT",
    "STR_POST_AAQ_REG": "ST.AAQ",
    "STR_ACC_REG": "ST.ACC",
    "INCR_MOD_POW2": "INC.MODP2",
    "BLT": "BLT.S",
    "BGE": "BGE.S",
    "BGT": "BGT.S",
    "BLE": "BLE.S",
    "BKPT": "HALT",
}

#: Old handler -> new handler.
HANDLER_RENAMES = {
    "execute_ldr_mult_reg": "execute_ld_mult",
    "execute_ldr_cyclic_mult_reg": "execute_ld_mult_cyclic",
    "execute_ldr_mult_mask_reg": "execute_ld_mask",
    "execute_str_post_aaq_reg": "execute_st_aaq",
    "execute_str_acc_reg": "execute_st_acc",
    "execute_lr_incr_mod_pow2": "execute_lr_inc_modp2",
    "execute_blt": "execute_blt_s",
    "execute_bge": "execute_bge_s",
    "execute_bkpt": "execute_halt",
}

ANY_CASE_SUFFIXES = {".asm", ".j2", ".py", ".js"}
UPPER_CASE_SUFFIXES = {".md", ".tex"}

SKIPPED_PREFIXES = ("planning/",)
SKIPPED_FILES = {
    "docs/content/release-log.md",
    "tools/rename_mnemonics.py",
    "tools/test_rename_mnemonics.py",
    "docs/content/specs/mnemonic-rename.md",  # the rename table itself
    "src/tools/ipu-as-py/test/test_sv_rename_table.py",
}
SKIPPED_PARTS = ("/test/baselines/", "/test/golden/", "/test/fixtures/")

_NAMES = "|".join(sorted(RENAMES, key=len, reverse=True))
# Not inside a longer word or a dotted name; a string escape (``\nBKPT``) counts
# as a boundary.
_BEFORE = r"(?:(?<![\w.])|(?<=\\[nrt]))"
# A dot that continues a name (``BLT.S``, or ``BLT\.S`` in a regex) protects
# it; a full stop does not.
_AFTER = r"(?!\w|\.\w|\\\.\w)"
_ANY_CASE = re.compile(rf"{_BEFORE}({_NAMES}){_AFTER}", re.IGNORECASE)
_UPPER = re.compile(rf"{_BEFORE}({_NAMES}){_AFTER}")
_HANDLERS = re.compile(rf"\b({'|'.join(HANDLER_RENAMES)})\b")


def _new_name(match: re.Match[str]) -> str:
    old = match.group(1)
    new = RENAMES[old.upper()]
    return new.lower() if old.islower() else new


def rename_text(text: str, suffix: str) -> tuple[str, int]:
    """Rename the old mnemonics in one file's text.

    Args:
        text: The file's contents.
        suffix: The file's suffix; it picks the rule (any case, upper case
            only, or none).

    Returns:
        The new text and the number of replacements.
    """
    if suffix in ANY_CASE_SUFFIXES:
        text, count = _ANY_CASE.subn(_new_name, text)
    elif suffix in UPPER_CASE_SUFFIXES:
        text, count = _UPPER.subn(_new_name, text)
    else:
        return text, 0
    if suffix == ".py":
        text, handlers = _HANDLERS.subn(lambda m: HANDLER_RENAMES[m.group(1)], text)
        count += handlers
    return text, count


def _skipped(path: str) -> bool:
    return (
        path.startswith(SKIPPED_PREFIXES)
        or path in SKIPPED_FILES
        or any(part in "/" + path for part in SKIPPED_PARTS)
    )


def rename_tree(root: Path, *, check: bool = False) -> dict[str, int]:
    """Rename every tracked file under a checkout.

    Args:
        root: The checkout (``git ls-files`` runs there).
        check: Report only; write nothing.

    Returns:
        Repository path -> replacements, for each file that changes.
    """
    listed = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    changed = {}
    for path in listed:
        file = root / path
        if _skipped(path) or file.suffix not in ANY_CASE_SUFFIXES | UPPER_CASE_SUFFIXES:
            continue
        if not file.is_file():
            continue
        text = file.read_text(encoding="utf-8")
        new, count = rename_text(text, file.suffix)
        if count:
            changed[path] = count
            if not check:
                file.write_text(new, encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_root = os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")
    parser.add_argument("root", nargs="?", default=default_root, type=Path)
    parser.add_argument("--check", action="store_true", help="report only; exit 1 on changes")
    args = parser.parse_args(argv)
    changed = rename_tree(args.root, check=args.check)
    for path, count in sorted(changed.items()):
        print(f"{count:6d}  {path}")
    print(f"{sum(changed.values())} replacement(s) in {len(changed)} file(s)")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
