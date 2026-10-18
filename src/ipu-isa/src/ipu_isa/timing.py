"""The timing contract: which value a read sees within one cycle (E03-14, Q32).

One instruction word issues every slot in the same cycle, in the fixed order
:data:`SLOT_EXECUTION_ORDER`. A register read sees one of two values
(GLOSSARY.md):

* **cycle start** -- the value the register held when the word began;
* **forwarded** -- the value after the writes of earlier slots of the same word.

The contract (decision Q32, ADR 0002) is four rules:

1. the Control stage (LR, COND) reads LRs at cycle start; every other stage
   reads LRs after the LR writes of the same word;
2. MULT reads R0, R1, R_CYCLIC and R_MASK at cycle start;
3. MULT_RES is forwarded to ACC, AGG and RESHAPE; R_ACC is forwarded to AAQ;
4. LR sub-slots read cycle-start values.

Any other read is forwarded when every slot that writes the register comes
earlier in the word, and cycle start otherwise (:func:`contract_view`).

Every instruction declares each register it reads or writes in
``instruction_spec.py``: as an operand with ``read`` or ``write``, or in its
``implicit_reads`` / ``implicit_writes``. The declarations describe the
emulator as it is today. Where today differs from the contract, the
difference is listed in :data:`KNOWN_DEVIATIONS` with the E04 issue that
removes it; :func:`validate_timing` (run by ``load_isa``) fails on any
difference that is not listed and on any listed one that has gone.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ipu_isa import schema
from ipu_isa.slots import Slot

__all__ = [
    "CONTROL_SLOTS",
    "KNOWN_DEVIATIONS",
    "OPERAND_REGISTERS",
    "SLOT_EXECUTION_ORDER",
    "Deviation",
    "RegisterAccess",
    "View",
    "contract_rule",
    "contract_view",
    "declared_reads",
    "declared_writes",
    "deviations",
    "register_writers",
    "validate_timing",
]


class View(str, Enum):
    """Which value of a register a read sees (the authored ``read`` / ``view`` values).

    A ``str`` mixin rather than ``StrEnum`` (Python 3.11), as for ``Slot``.
    """

    CYCLE_START = "cycle_start"
    FORWARDED = "forwarded"

    def __str__(self) -> str:
        return self.value


#: The order the slots of one word execute in. BREAK is evaluated first, before
#: any write; LR's sub-slots all read before any of them writes (rule 4).
SLOT_EXECUTION_ORDER: tuple[Slot, ...] = (
    Slot.BREAK,
    Slot.LR,
    Slot.LOAD,
    Slot.MULT,
    Slot.ACC,
    Slot.AAQ,
    Slot.STORE,
    Slot.ACC_STORE,
    Slot.COND,
)

#: The Control stage: every read here is at cycle start. BREAK is evaluated
#: before any write, so it is a Control slot as well.
CONTROL_SLOTS: frozenset[Slot] = frozenset({Slot.LR, Slot.COND, Slot.BREAK})

#: The registers a register operand type reads or writes. An ``LcrIdx``
#: operand selects an LR or a CR, so it names both.
OPERAND_REGISTERS: Mapping[str, tuple[str, ...]] = {
    "MultStageReg": ("r",),
    "LrIdx": ("lr",),
    "LrdIdx": ("lr",),
    "CrIdx": ("cr",),
    "LcrIdx": ("lr", "cr"),
}


@dataclass(frozen=True, slots=True)
class RegisterAccess:
    """One register an instruction reads or writes."""

    register: str  # REGISTER_DEFINITIONS name: "lr", "r", "r_cyclic", "r_mask", "mult_res", ...
    via_operand: str | None = None  # operand that selects the index, e.g. "dest_slot"
    view: View | None = None  # reads only


@dataclass(frozen=True, slots=True)
class Deviation:
    """A read whose view today differs from the contract."""

    slot: Slot
    mnemonic: str
    register: str
    via_operand: str | None
    today: View
    contract: View
    fixed_by: str  # the E04 issue that changes the emulator to the contract


def _execution_index(slot: Slot) -> int:
    return SLOT_EXECUTION_ORDER.index(slot)


def _precedes(producer: Slot, reader: Slot) -> bool:
    return _execution_index(producer) < _execution_index(reader)


# ---------------------------------------------------------------------------
# Declarations, read from the authoring dicts
# ---------------------------------------------------------------------------


def declared_reads(inst_def: Mapping[str, Any]) -> tuple[RegisterAccess, ...]:
    """Every register ``inst_def`` reads: its ``read`` operands, then its ``implicit_reads``."""
    reads = [
        RegisterAccess(register, operand["name"], View(operand["read"]))
        for operand in inst_def["operands"]
        if "read" in operand
        for register in OPERAND_REGISTERS[operand["type"]]
    ]
    reads += [
        RegisterAccess(entry["register"], entry.get("via_operand"), View(entry["view"]))
        for entry in inst_def.get("implicit_reads", ())
    ]
    return tuple(reads)


def declared_writes(inst_def: Mapping[str, Any]) -> tuple[RegisterAccess, ...]:
    """Every register ``inst_def`` writes: its ``write`` operands, then its ``implicit_writes``."""
    writes = [
        RegisterAccess(register, operand["name"])
        for operand in inst_def["operands"]
        if operand.get("write")
        for register in OPERAND_REGISTERS[operand["type"]]
    ]
    writes += [
        RegisterAccess(entry["register"], entry.get("via_operand"))
        for entry in inst_def.get("implicit_writes", ())
    ]
    return tuple(writes)


def _instructions(
    instruction_spec: Mapping[str, Mapping[str, Any]],
) -> Iterator[tuple[Slot, str, Mapping[str, Any]]]:
    for slot_name, instructions in instruction_spec.items():
        for mnemonic, inst_def in instructions.items():
            yield Slot(slot_name), mnemonic, inst_def


def register_writers(
    instruction_spec: Mapping[str, Mapping[str, Any]],
) -> dict[str, frozenset[Slot]]:
    """For each register, the slots with an instruction that writes it."""
    writers: dict[str, set[Slot]] = {}
    for slot, _, inst_def in _instructions(instruction_spec):
        for access in declared_writes(inst_def):
            writers.setdefault(access.register, set()).add(slot)
    return {register: frozenset(slots) for register, slots in writers.items()}


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

#: Rule 2: what MULT reads at cycle start ("r" is R0 and R1).
_MULT_CYCLE_START = frozenset({"r", "r_cyclic", "r_mask"})
#: Rule 3: what is forwarded, to which slot.
_FORWARDED_TO = {("mult_res", Slot.ACC), ("r_acc", Slot.AAQ)}


def _classify(
    slot: Slot, register: str, writers: Mapping[str, Iterable[Slot]]
) -> tuple[View, int | None]:
    """The contract's view of one read, and the number of the rule that gives it."""
    if slot is Slot.LR:  # rule 4 (and rule 1 for LRs)
        return View.CYCLE_START, 4
    if slot in CONTROL_SLOTS:  # rule 1; a Control read of anything else is at cycle start too
        return View.CYCLE_START, 1 if register == "lr" else None
    if register == "lr":  # rule 1
        return View.FORWARDED, 1
    if slot is Slot.MULT and register in _MULT_CYCLE_START:  # rule 2
        return View.CYCLE_START, 2
    if (register, slot) in _FORWARDED_TO:  # rule 3
        return View.FORWARDED, 3
    producers = tuple(writers.get(register, ()))
    if producers and all(_precedes(producer, slot) for producer in producers):
        return View.FORWARDED, None
    return View.CYCLE_START, None


def contract_view(slot: Slot, register: str, writers: Mapping[str, Iterable[Slot]]) -> View:
    """The view the contract gives a read of ``register`` by an instruction in ``slot``.

    Args:
        slot: The reading instruction's slot.
        register: The register read.
        writers: Each register's writing slots (:func:`register_writers` of the
            spec being checked).

    Returns:
        ``View.FORWARDED`` or ``View.CYCLE_START``.
    """
    return _classify(slot, register, writers)[0]


def contract_rule(slot: Slot, register: str) -> int | None:
    """The number (1 to 4, as in the module docstring) of the rule that fixes a read's view.

    ``None`` when no numbered rule names the read: its view then follows the
    slot order (forwarded when every writer runs earlier in the word, cycle start
    otherwise; a CR is never written in a run). The docs label each read with it.
    """
    return _classify(slot, register, {})[1]


# ---------------------------------------------------------------------------
# Where the emulator differs from the contract today
# ---------------------------------------------------------------------------


#: Every read whose declared view (today's emulator) differs from
#: :func:`contract_view`, with the E04 issue that removes it. The list can only
#: shrink: :func:`validate_timing` rejects a difference missing from it and a
#: row whose difference has gone. Empty since E04-27 removed the last rows
#: (MULT read R_MASK forwarded): the emulator follows the contract everywhere.
KNOWN_DEVIATIONS: tuple[Deviation, ...] = ()


def deviations(
    instruction_spec: Mapping[str, Mapping[str, Any]],
) -> tuple[Deviation, ...]:
    """Every declared read whose view differs from the contract, in spec order.

    ``fixed_by`` is copied from :data:`KNOWN_DEVIATIONS` where the read is
    listed there, and empty otherwise.
    """
    writers = register_writers(instruction_spec)
    known = {_deviation_key(d): d.fixed_by for d in KNOWN_DEVIATIONS}
    found = []
    for slot, mnemonic, inst_def in _instructions(instruction_spec):
        for access in declared_reads(inst_def):
            contract = contract_view(slot, access.register, writers)
            if access.view is not None and access.view is not contract:
                key = (slot, mnemonic, access.register, access.via_operand)
                found.append(
                    Deviation(
                        slot,
                        mnemonic,
                        access.register,
                        access.via_operand,
                        access.view,
                        contract,
                        known.get(key, ""),
                    )
                )
    return tuple(found)


def _deviation_key(deviation: Deviation) -> tuple[Slot, str, str, str | None]:
    return (deviation.slot, deviation.mnemonic, deviation.register, deviation.via_operand)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _describe(slot: Slot, mnemonic: str, access: RegisterAccess) -> str:
    via = f" via {access.via_operand!r}" if access.via_operand else ""
    return f"{slot}.{mnemonic}: {access.register}{via}"


def _check_slot_order(slots: Iterable[Slot]) -> None:
    if sorted(SLOT_EXECUTION_ORDER) != sorted(slots) or len(set(SLOT_EXECUTION_ORDER)) != len(
        SLOT_EXECUTION_ORDER
    ):
        raise schema.SpecError(
            f"SLOT_EXECUTION_ORDER: {[str(s) for s in SLOT_EXECUTION_ORDER]} must hold every "
            f"slot exactly once ({sorted(str(s) for s in slots)})"
        )


def _check_read(
    slot: Slot, mnemonic: str, access: RegisterAccess, writers: Mapping[str, frozenset[Slot]]
) -> None:
    """Rules 1 to 3 of the validator for one declared read."""
    where = _describe(slot, mnemonic, access)
    if access.view is not View.FORWARDED:
        return
    if slot in CONTROL_SLOTS:
        raise schema.SpecError(
            f"{where}: a {str(slot).upper()} (Control slot) read is at cycle start, not forwarded"
        )
    if access.register == "cr":
        raise schema.SpecError(
            f"{where}: a CR read is at cycle start (nothing writes a CR during a run)"
        )
    producers = writers.get(access.register, frozenset())
    late = sorted(str(p) for p in producers if not _precedes(p, slot))
    if not producers or late:
        reason = f"written by {late}, which do not run before {slot}" if late else "never written"
        raise schema.SpecError(
            f"{where}: a forwarded read needs every writer to run earlier in "
            f"SLOT_EXECUTION_ORDER; it is {reason}, so read it at cycle start"
        )


def _check_lr_writes(slot: Slot, mnemonic: str, inst_def: Mapping[str, Any]) -> None:
    """Every LR-slot instruction but NOP writes one LR, through a ``write`` operand
    (the emulator's LR write-conflict check finds the target there)."""
    where = f"{slot}.{mnemonic}"
    if any(entry["register"] == "lr" for entry in inst_def.get("implicit_writes", ())):
        raise schema.SpecError(
            f"{where}.implicit_writes: an LR-slot instruction declares its LR target "
            "with 'write': True on the operand"
        )
    targets = [op["name"] for op in inst_def["operands"] if op.get("write")]
    if mnemonic.upper() == "NOP":
        return
    if len(targets) != 1:
        raise schema.SpecError(
            f"{where}: an LR-slot instruction writes one LR, declared with 'write': True "
            f"on exactly one operand (found {targets})"
        )


def _check_deviations(
    instruction_spec: Mapping[str, Mapping[str, Any]], known: Iterable[Deviation]
) -> None:
    """The differences from the contract are exactly the listed ones."""
    actual = {_deviation_key(d): d for d in deviations(instruction_spec)}
    listed = {_deviation_key(d): d for d in known}
    new = [d for key, d in actual.items() if key not in listed]
    if new:
        rows = "; ".join(
            f"{d.slot}.{d.mnemonic} {d.register} via {d.via_operand!r}: "
            f"declared {d.today}, contract {d.contract}"
            for d in new
        )
        raise schema.SpecError(
            f"timing: read(s) differ from the contract and are not in KNOWN_DEVIATIONS: {rows}"
        )
    stale = [d for key, d in listed.items() if key not in actual]
    if stale:
        rows = "; ".join(
            f"{d.slot}.{d.mnemonic} {d.register} via {d.via_operand!r} ({d.fixed_by})"
            for d in stale
        )
        raise schema.SpecError(
            f"KNOWN_DEVIATIONS: row(s) no longer differ from the contract; delete them: {rows}"
        )
    for key, row in listed.items():
        if (row.today, row.contract) != (actual[key].today, actual[key].contract):
            raise schema.SpecError(
                f"KNOWN_DEVIATIONS: {row.slot}.{row.mnemonic} {row.register}: listed as "
                f"{row.today} -> {row.contract}, declared {actual[key].today} -> "
                f"{actual[key].contract}"
            )


def validate_timing(
    instruction_spec: Mapping[str, Mapping[str, Any]],
    slots: Iterable[Slot],
    known_deviations: Iterable[Deviation] = KNOWN_DEVIATIONS,
) -> None:
    """Check the declared register accesses against the timing contract.

    * a forwarded read is legal only if every slot that writes the register
      runs earlier in :data:`SLOT_EXECUTION_ORDER` (a same-slot producer is
      read at cycle start);
    * CR reads are at cycle start (nothing writes a CR during a run);
    * every read in a Control slot is at cycle start;
    * every LR-slot instruction that writes an LR declares it with ``write``;
    * the reads that differ from :func:`contract_view` are exactly
      ``known_deviations``.

    Expects a spec that already passed the schema check. Pure; raises
    :class:`ipu_isa.schema.SpecError`.
    """
    _check_slot_order(slots)
    writers = register_writers(instruction_spec)
    for slot, mnemonic, inst_def in _instructions(instruction_spec):
        for access in declared_reads(inst_def):
            _check_read(slot, mnemonic, access, writers)
        if slot is Slot.LR:
            _check_lr_writes(slot, mnemonic, inst_def)
    _check_deviations(instruction_spec, known_deviations)
