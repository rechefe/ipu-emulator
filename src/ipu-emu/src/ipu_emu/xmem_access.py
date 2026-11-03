"""Every XMEM row access a program makes, with the registers that addressed it.

Every instruction that reads or writes an XMEM row (``LD.MULT``,
``LD.MULT.CYCLIC``, ``LD.MASK``, ``LD.LUT``, ``ACTIVATE.QUANTIZE``,
``ACTIVATE.EXACT``) addresses it as ``table(CR[base]) + LR[offset]``
(``docs/content/requirements/memory-access.md``). An :class:`XmemAccess`
names both registers and their values, so a tool can tell which table an
access belongs to (its base CR) and where in the table it lands (the
offset): what the Cache Unit sees, which a byte address alone does not say.

:class:`XmemAccessObserver` is a :class:`~ipu_emu.cycle.CycleObserver`: it
reads only the public :class:`~ipu_emu.issue.IssueFacts` of each op.
``ipu_emu.reference.ReferenceModel.observe_xmem_accesses`` attaches one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

from ipu_isa.view import isa

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState
    from ipu_emu.issue import IssueFacts

__all__ = ["ADDRESS_OPERANDS", "XmemAccess", "XmemAccessObserver", "address_operands"]

#: The (base CR, offset LR) operand names an XMEM instruction adds into a row:
#: the LOAD slot's, then the AaQ slot's (``table_id`` + ``table_offset``).
ADDRESS_OPERANDS: tuple[tuple[str, str], ...] = (("base", "offset"), ("table_id", "table_offset"))

#: What an access does to its row, by the run statistic the instruction bumps.
_KIND_OF_STAT: Mapping[str, Literal["read", "write"]] = {
    "xmem_reads": "read",
    "xmem_writes": "write",
}


@dataclass(frozen=True)
class XmemAccess:
    """One XMEM row that an instruction read or wrote.

    Attributes:
        cycle: The words retired before this one since the observer was
            attached.
        pc: The word's program counter.
        mnemonic: The instruction (``LD.MULT``, ``ACTIVATE.EXACT``, ...).
        kind: ``read`` or ``write``.
        cr: The base CR's number (``CR0`` .. ``CR15``).
        base: Its value, as a signed 32-bit int: a flat row, or, when a
            table set is bound, a table ID.
        lr: The offset LR's number.
        offset: Its value, as a signed 32-bit int.
        storage_row: The first XMEM row of the table that ``base`` names, or
            ``None`` when no table set is bound (``base`` is then a row) or the
            ID is not bound (the access faults F-6).
    """

    cycle: int
    pc: int
    mnemonic: str
    kind: Literal["read", "write"]
    cr: int
    base: int
    lr: int
    offset: int
    storage_row: int | None = None

    @property
    def table_id(self) -> int | None:
        """The table ID in ``base``, or ``None`` when no table set is bound."""
        return None if self.storage_row is None else self.base

    @property
    def row(self) -> int:
        """The row addressed: the table's first row (or ``base``, flat) plus ``offset``."""
        first = self.base if self.storage_row is None else self.storage_row
        return first + self.offset


def _signed32(value: int) -> int:
    value &= 0xFFFF_FFFF
    return value - (1 << 32) if value >= 1 << 31 else value


@cache
def address_operands(mnemonic: str) -> tuple[str, str] | None:
    """The (base CR, offset LR) operand names of an instruction, if it has a pair.

    Args:
        mnemonic: The instruction, as the ISA spells it.

    Returns:
        One of :data:`ADDRESS_OPERANDS`, or ``None`` when the instruction
        has no such pair.
    """
    view = isa()
    try:
        slot = view.find_slot(mnemonic)
    except (KeyError, ValueError):  # no such instruction, or NOP (in every slot)
        return None
    names = {op.name for op in view.instruction(slot, mnemonic).operands}
    for pair in ADDRESS_OPERANDS:
        if set(pair) <= names:
            return pair
    return None


class XmemAccessObserver:
    """Calls ``callback(access)`` for each XMEM row access, as its op issues.

    An access is reported before its handler runs, so a row outside XMEM
    (fault F-6) is reported too. Host accesses never pass through a cycle
    and are not reported.

    Args:
        callback: Called with each :class:`XmemAccess`.
        state: The state whose bound table set, if any, resolves ``base``.
    """

    def __init__(self, callback: Callable[[XmemAccess], None], state: IpuState) -> None:
        self._callback = callback
        self._state = state
        self._pc = 0
        self._cycle = -1

    def begin(self, pc: int) -> None:
        """A word starts."""
        self._pc = pc
        self._cycle += 1

    def issued(self, facts: IssueFacts) -> None:
        """Report the op's access, if it reads or writes a row."""
        stat = facts.plan.stat
        if stat not in _KIND_OF_STAT:
            return
        pair = address_operands(facts.plan.inst_name)
        if pair is None:
            return
        base, offset = pair
        base_value = _signed32(facts.operands[base])
        tables = self._state.table_map
        entry = None if tables is None else tables.get(base_value)
        self._callback(
            XmemAccess(
                cycle=self._cycle,
                pc=self._pc,
                mnemonic=facts.plan.inst_name,
                kind=_KIND_OF_STAT[stat],
                cr=facts.encoded[base],
                base=base_value,
                lr=facts.encoded[offset],
                offset=_signed32(facts.operands[offset]),
                storage_row=None if entry is None else entry[0],
            )
        )

    def retired(self, facts: IssueFacts, result: Any) -> None:
        """Nothing: the access was reported when it issued."""

    def end(self) -> None:
        """Nothing at the end of a word."""
