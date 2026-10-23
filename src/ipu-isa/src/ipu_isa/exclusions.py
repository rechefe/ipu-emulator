"""Instructions that may not share one instruction word (ADR 0007).

Each slot holds one instruction per word, but two instructions in different
slots can still compete for one resource. The table is data: the assembler
reports a word that holds both sides of a row as an error, and the editor's
"fit its slots" check reads the same rows. No runtime fault exists for these
pairs; a program that has one never assembles.

A side is a slot and an ``fnmatch`` pattern over its mnemonics (upper case),
so ``LD.*`` names every load instruction but ``NOP``.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from dataclasses import dataclass

from ipu_isa.slots import Slot

__all__ = ["WORD_EXCLUSIONS", "WordExclusion", "word_conflicts"]


@dataclass(frozen=True, slots=True)
class WordExclusion:
    """Two sets of instructions that may not share a word, and why."""

    first: tuple[Slot, str]  # (slot, mnemonic pattern)
    second: tuple[Slot, str]
    reason: str

    def matches(self, slot: Slot, mnemonic: str, side: int) -> bool:
        """True when ``mnemonic`` in ``slot`` is on side 0 (``first``) or 1 (``second``)."""
        want_slot, pattern = self.first if side == 0 else self.second
        return slot is want_slot and fnmatch.fnmatchcase(mnemonic.upper(), pattern)


#: Every exclusion, in the order the assembler reports them.
WORD_EXCLUSIONS: tuple[WordExclusion, ...] = (
    WordExclusion(
        first=(Slot.AAQ, "LD.LUT"),
        second=(Slot.LOAD, "LD.*"),
        reason="LD.LUT and the LOAD slot's loads share the XMEM read port",
    ),
    WordExclusion(
        first=(Slot.AAQ, "ACTIVATE.QUANTIZE"),
        second=(Slot.AAQ_EXACT, "ACTIVATE.EXACT"),
        reason="ACTIVATE.QUANTIZE and ACTIVATE.EXACT both store the activated R_ACC",
    ),
)


def word_conflicts(
    word: Iterable[tuple[Slot, str]],
    exclusions: Iterable[WordExclusion] = WORD_EXCLUSIONS,
) -> tuple[WordExclusion, ...]:
    """The exclusions one word breaks.

    Args:
        word: ``(slot, mnemonic)`` of each instruction in the word.
        exclusions: The table to check (the repository's by default).

    Returns:
        Each broken row once, in table order.
    """
    members = [(Slot(slot), mnemonic) for slot, mnemonic in word]
    return tuple(
        row
        for row in exclusions
        if any(row.matches(s, m, 0) for s, m in members)
        and any(row.matches(s, m, 1) for s, m in members)
    )
