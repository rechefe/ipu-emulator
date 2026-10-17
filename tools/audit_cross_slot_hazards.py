"""Find the bundles whose result depends on a same-word register view (E04-25).

    bazel run //tools:audit_cross_slot_hazards              # deviations only
    bazel run //tools:audit_cross_slot_hazards -- --all     # every same-word read after write

A *cross-slot hazard* is one instruction word in which a slot reads a register
that another slot of the same word writes (for an indexed register, the same
index: ``INC LR3 1`` and ``AGG.SUM LR3 CR15`` share LR3). Only in such a word
does it matter whether the read sees the register at cycle start or forwarded
(GLOSSARY.md, ``ipu_isa.timing``).

The audit assembles the whole program corpus (``asm_corpus``: every kernel and
test program, with the render contexts its tests use), decodes every word with
the ISA's codecs and lists its hazards. For each one it prints the view the
spec declares today and the view the timing contract gives. By default it
prints only the hazards where the two differ: the bundles whose result changes
when the deviation is removed (``KNOWN_DEVIATIONS``, the issue in the last
column). Run it before a timing flip (E04-25, E04-26, E04-27) to find the
kernels to fix, and after it to show none are left.

Register granularity: an LR, an LRD pair, an LcrIdx LR, an ``LrOrReshapeMask``
LR and R0/R1 resolve to their indices; every other register is one unit. The
exit status is 0 unless the corpus fails to assemble or decode.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import asm_corpus

from ipu_isa.codec.slot import DecodedSlot, codecs
from ipu_isa.registers import lrd_pair
from ipu_isa.reshape_mask import RESHAPE_MASK_LR_OFFSET
from ipu_isa.slots import Slot
from ipu_isa.timing import KNOWN_DEVIATIONS, SLOT_EXECUTION_ORDER, View, contract_view

#: The LR count: an ``LcrIdx`` value below it is an LR, at or above it a CR.
_LR_COUNT = 16

#: (slot, mnemonic, register, via operand) -> the issue that removes the deviation.
_FIXED_BY = {(d.slot, d.mnemonic, d.register, d.via_operand): d.fixed_by for d in KNOWN_DEVIATIONS}


@dataclass(frozen=True, slots=True)
class Hazard:
    """One read in a word whose register (index) another slot of the word writes."""

    program: str  # corpus key
    word: int  # word index in the program
    reader: str  # "acc.AGG.SUM.FIRST"
    register: str  # "lr", "mult_res", ...
    via_operand: str | None
    indices: tuple[int, ...] | None  # None: the whole register
    writers: tuple[str, ...]  # "lr[1].INC", ...
    declared: View
    contract: View
    fixed_by: str  # KNOWN_DEVIATIONS issue, or ""

    @property
    def deviates(self) -> bool:
        return self.declared is not self.contract

    def describe(self) -> str:
        where = self.register.upper()
        if self.indices is not None:
            where += "[" + ",".join(str(i) for i in self.indices) + "]"
        if self.via_operand:
            where += f" via {self.via_operand}"
        fixed = f" ({self.fixed_by})" if self.fixed_by else ""
        return (
            f"{self.program}: word {self.word}: {self.reader} reads {where}, "
            f"written by {' '.join(self.writers)}; declared {self.declared}, "
            f"contract {self.contract}{fixed}"
        )


def _indices(decoded: DecodedSlot, register: str, via: str | None) -> frozenset[int] | None:
    """The indices of ``register`` that ``via`` selects in ``decoded``; ``None``
    for the whole register; an empty set when ``via`` selects another register
    (a CR through an ``LcrIdx``, or an immediate reshape mask)."""
    if via is None:
        return None
    instruction = decoded.instruction
    names = [operand.name for operand in instruction.operands]
    operand = instruction.operands[names.index(via)]
    value = decoded.operands[names.index(via)]
    kind = operand.type.name
    if kind in ("LrIdx", "MultStageReg"):
        return frozenset({value})
    if kind == "LrdIdx":
        return frozenset(lrd_pair(value))
    if kind == "LcrIdx":
        if register == "lr":
            return frozenset({value}) if value < _LR_COUNT else frozenset()
        return frozenset({value - _LR_COUNT}) if value >= _LR_COUNT else frozenset()
    if kind == "LrOrReshapeMaskImmediate":
        if value >= RESHAPE_MASK_LR_OFFSET:
            return frozenset({value - RESHAPE_MASK_LR_OFFSET})
        return frozenset()
    return frozenset({value})


def _overlaps(a: frozenset[int] | None, b: frozenset[int] | None) -> bool:
    if a is None:
        return b is None or bool(b)
    if b is None:
        return bool(a)
    return bool(a & b)


def _label(slot: Slot, instance: int, mnemonic: str, count: int) -> str:
    return f"{slot}[{instance}].{mnemonic}" if count > 1 else f"{slot}.{mnemonic}"


def word_hazards(program: str, index: int, word: int) -> Iterator[Hazard]:
    """Every cross-slot hazard of one instruction word."""
    slots = codecs().word.unpack(word)
    counts = {slot: sum(1 for s, _ in slots if s is slot) for slot, _ in slots}
    writes = []  # (slot, label, register, indices)
    for (slot, instance), decoded in slots.items():
        label = _label(slot, instance, decoded.instruction.mnemonic, counts[slot])
        for access in decoded.instruction.writes():
            indices = _indices(decoded, access.register, access.via_operand)
            writes.append((slot, instance, label, access.register, indices))
    for (slot, instance), decoded in sorted(
        slots.items(), key=lambda item: (SLOT_EXECUTION_ORDER.index(item[0][0]), item[0][1])
    ):
        instruction = decoded.instruction
        reader = _label(slot, instance, instruction.mnemonic, counts[slot])
        for access in instruction.reads():
            indices = _indices(decoded, access.register, access.via_operand)
            if indices is not None and not indices:
                continue
            writers = tuple(
                label
                for w_slot, w_instance, label, register, w_indices in writes
                if register == access.register
                and (w_slot, w_instance) != (slot, instance)
                and _overlaps(indices, w_indices)
            )
            if not writers or access.view is None:
                continue
            yield Hazard(
                program=program,
                word=index,
                reader=reader,
                register=access.register,
                via_operand=access.via_operand,
                indices=None if indices is None else tuple(sorted(indices)),
                writers=writers,
                declared=access.view,
                contract=contract_view(slot, access.register),
                fixed_by=_FIXED_BY.get(
                    (slot, instruction.mnemonic, access.register, access.via_operand), ""
                ),
            )


def corpus_hazards(root: Path) -> list[Hazard]:
    """Every cross-slot hazard in the program corpus under ``root``."""
    hazards: list[Hazard] = []
    for entry in asm_corpus.discover(root):
        with warnings.catch_warnings():
            # Emulator-only slots warn on every assembly; the audit covers them too.
            warnings.simplefilter("ignore")
            words = asm_corpus.assemble(root, entry).words
        for index, word in enumerate(words):
            hazards.extend(word_hazards(entry.key, index, word))
    return hazards


def report(hazards: list[Hazard], *, show_all: bool) -> str:
    shown = hazards if show_all else [h for h in hazards if h.deviates]
    lines = [h.describe() for h in shown]
    bundles = {(h.program, h.word) for h in shown}
    programs = {h.program for h in shown}
    what = "cross-slot hazard(s)" if show_all else "hazard(s) where the declared view deviates"
    lines.append(f"{len(shown)} {what} in {len(bundles)} bundle(s) of {len(programs)} program(s)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--all", action="store_true", help="list every same-word read after write")
    parser.add_argument("root", nargs="?", help="workspace root (default: the Bazel workspace)")
    args = parser.parse_args(argv)
    root = args.root or os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not root:
        print("Run this through Bazel: bazel run //tools:audit_cross_slot_hazards", file=sys.stderr)
        return 2
    print(report(corpus_hazards(Path(root)), show_all=args.all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
