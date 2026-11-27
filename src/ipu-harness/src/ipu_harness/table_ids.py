"""Table IDs: what a tensor's CR holds.

A CR that an XMEM instruction names as its ``base`` or ``table_id`` holds a
bare table ID, 0-15, never a row. The hardware maps an ID to the banks that hold
the table, and the access is row ``storage(ID) + LR[offset]`` of that table. The
harness plays the hardware's part: :class:`TableBinder` gives every tensor
that ``[abi]`` (or a stage's table) binds to a CR an ID, :func:`abi.register_writes
<ipu_harness.abi.register_writes>` writes it, and the runner binds the set of
IDs to their storage rows on the reference model (``bind_tables``) just before
the program starts, so an ID that is not bound, or an offset outside its table,
faults F-6.

A table is *free* or *pinned*:

* A free table's CR is written at launch, and the binder chooses its ID: the
  lowest not yet taken, in ``[abi]`` order, or the one a chain hands over
  (below). A program must work whatever the ID is, so it may not read the CR as
  a value (:func:`ipu_harness.table_bases.check_table_bases` refuses it).
* A pinned table is named by a hardwired CR: a tensor bound to ``CR0`` is table
  0 (``CR1``: table 1). Nothing is written, and the ID is not the binder's to
  change.

One tensor has one ID for the whole run of a kernel: in a multi-stage kernel the
binder keeps it across the stages, whichever CR each stage binds it to
(``depthwise_conv_stride2_*``: ``y`` is ``CR2`` in stage 1 and ``CR3`` in
stage 2, one ID). Across the steps of a chain a buffer keeps the ID its
producer gave it when the consumer can have it (:attr:`TableBinder.hints`), or
is *retagged*: the consumer's table gets another ID, the storage never moves (the
hardware retags banks). A buffer a kernel wrote is never handed to a pinned
table (``CR0``): the pinned ID cannot be retagged, so the host copies it
(:meth:`ipu_harness.chain.Chain.write`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from ipu_emu.errors import ConfigurationError
from ipu_emu.geometry import EIGHT_BIT_ROWS
from ipu_harness.params import CasePlan
from ipu_harness.placement import Placement
from ipu_harness.tables import declares_tables
from ipu_isa.formats import ElementFormat, addressable_rows

__all__ = ["MAX_TABLES", "PINNED", "TableBinder", "TableSet"]

#: IDs a program has: a table ID is 4 bits (the Cache Unit spec).
MAX_TABLES = 16

#: The hardwired CRs a table may be pinned to, and the ID each one reads.
PINNED: Mapping[str, int] = MappingProxyType({"CR0": 0, "CR1": 1})


def _model_rows(mode: ElementFormat) -> int:
    """The XMEM rows the reference model lets a table reach in ``mode``: NX8
    reaches the first ``EIGHT_BIT_ROWS`` (the model's NX8 XMEM is smaller than
    the byte address space), FP32 all of it."""
    rows = addressable_rows(mode)
    return min(rows, EIGHT_BIT_ROWS) if mode is not ElementFormat.FP32 else rows


@dataclass(frozen=True)
class TableSet:
    """The tables one program runs with.

    Attributes:
        ids: Each bound tensor's ID.
        storage: Each ID's ``(first_row, rows)`` in XMEM: the argument of
            ``ReferenceModel.bind_tables``. A table's rows are its region's and
            its declared overhead rows (a loop's last prefetch), and a program
            that reads past them faults F-6. A kernel that declares no tables
            has no sizes to hold: its tables reach the end of XMEM.
        pinned: The IDs that are pinned (their CR is hardwired).
    """

    ids: Mapping[str, int]
    storage: Mapping[int, tuple[int, int]]
    pinned: frozenset[int] = frozenset()


@dataclass
class TableBinder:
    """Assigns table IDs to the tensors of one kernel run.

    One binder serves every stage of a multi-stage kernel: :meth:`bind` is
    called with each stage's ``tensor -> CR`` table, and a tensor keeps the ID
    it was first given.

    Args:
        plan: The checked case.
        placement: Where its regions are.
        hints: The ID each tensor would like, by name (a chain: the ID the
            producer gave a resident input). Kept when it is free and not
            another tensor's pinned ID, else the tensor takes the next free ID
            (a retag, listed in :attr:`retagged`).
        kernel_written: Inputs that an earlier kernel wrote in XMEM (a chain's
            kernel-written residents): refused on a pinned table.
        order: The IDs free tensors draw from, first to last; default ``0..15``.
            A test passes a permutation to show no program relies on an ID's
            value.

    Attributes:
        ids: Every ID given so far, by tensor.
        retagged: Tensors whose hint was not kept, as ``(hint, ID)``.
    """

    plan: CasePlan
    placement: Placement
    hints: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    kernel_written: frozenset[str] = frozenset()
    order: Sequence[int] = tuple(range(MAX_TABLES))
    ids: dict[str, int] = field(default_factory=dict)
    retagged: dict[str, tuple[int, int]] = field(default_factory=dict)

    def bind(self, bindings: Mapping[str, str]) -> TableSet:
        """The table set of one program: its tensors' IDs and storage.

        Args:
            bindings: Names to registers, as the program is assembled with
                (the kernel's ``[abi]`` or a stage's table, registers in upper
                case). Names that are not tensors of the case are skipped.

        Returns:
            The set. A tensor bound before keeps its ID.

        Raises:
            ConfigurationError: A tensor is bound to ``CR0`` after it was given
                another ID; more than 16 tables; a kernel-written buffer on a
                pinned table; two tensors share a CR but not their rows.
        """
        tensors = [(n, r.upper()) for n, r in bindings.items() if n in self.placement.regions]
        by_register: dict[str, list[str]] = {}
        for name, register in tensors:
            by_register.setdefault(register, []).append(name)
        pinned = self._pin(by_register)
        taken = {i for n, i in self.ids.items() if n not in pinned}
        reserved = set(pinned.values())
        for register, names in by_register.items():
            if register not in PINNED:
                self._free(register, names, taken, reserved)
        ids = {name: self.ids[name] for name, _ in tensors}
        return TableSet(
            MappingProxyType(ids),
            MappingProxyType(self._storage(ids)),
            frozenset(pinned.values()),
        )

    def _pin(self, by_register: Mapping[str, Sequence[str]]) -> dict[str, int]:
        """Give the tensors on hardwired CRs their pinned IDs."""
        manifest = self.plan.manifest
        pinned: dict[str, int] = {}
        for register, names in by_register.items():
            if register not in PINNED:
                continue
            table_id = PINNED[register]
            for name in names:
                if name in self.kernel_written:
                    raise ConfigurationError(
                        f"{manifest.name}: input {name} was written by an earlier kernel and "
                        f"{register} is a pinned table: a buffer is never handed over "
                        "through a pinned table (copy it with Chain.write)"
                    )
                if self.ids.get(name, table_id) != table_id:
                    raise ConfigurationError(
                        f"{manifest.name}: {name} has table ID {self.ids[name]} and this "
                        f"program binds it to {register}, the pinned table {table_id}"
                    )
                owner = next((n for n, i in self.ids.items() if i == table_id and n != name), None)
                if owner is not None:
                    raise ConfigurationError(
                        f"{manifest.name}: {name} is the pinned table {table_id}, which "
                        f"{owner} already has"
                    )
                pinned[name] = self.ids[name] = table_id
        return pinned

    def _free(
        self, register: str, names: Sequence[str], taken: set[int], reserved: set[int]
    ) -> None:
        """Give the free tensors on one CR their (common) ID."""
        manifest = self.plan.manifest
        regions = self.placement.regions
        first = names[0]
        for other in names[1:]:
            if regions[other].base_row != regions[first].base_row:
                raise ConfigurationError(
                    f"{manifest.name}: {first} and {other} both bind {register} but sit at "
                    f"rows {regions[first].base_row} and {regions[other].base_row}"
                )
        known = {self.ids[n] for n in names if n in self.ids}
        if len(known) > 1:
            raise ConfigurationError(
                f"{manifest.name}: {', '.join(names)} share {register} but have the IDs "
                f"{sorted(known)}"
            )
        if known:
            chosen = known.pop()
        else:
            chosen = self._next(first, taken | reserved)
        for name in names:
            self.ids[name] = chosen
        taken.add(chosen)

    def _next(self, name: str, unavailable: set[int]) -> int:
        """The ID ``name`` takes: its hint when free, else the first free one."""
        hint = self.hints.get(name)
        if hint is not None and hint not in unavailable and hint not in self._others(name):
            return hint
        for candidate in self.order:
            if candidate not in unavailable and candidate not in self._others(name):
                if hint is not None:
                    self.retagged[name] = (hint, candidate)
                return candidate
        raise ConfigurationError(
            f"{self.plan.manifest.name}: {len(set(self.ids.values())) + 1} tables; "
            f"a program has {MAX_TABLES} IDs"
        )

    def _others(self, name: str) -> set[int]:
        return {i for n, i in self.ids.items() if n != name}

    def _storage(self, ids: Mapping[str, int]) -> dict[int, tuple[int, int]]:
        capacity = _model_rows(self.plan.format.mode)
        bounded = declares_tables(self.plan.manifest)
        out: dict[int, tuple[int, int]] = {}
        for name, table_id in ids.items():
            region = self.placement.regions[name]
            # A kernel that declares no tables has no sizes to hold its program to: its
            # tables reach the end of XMEM, so a run can still measure them (--tables).
            rows = region.rows + region.overhead_rows if bounded else capacity
            rows = min(rows, capacity - region.base_row)
            first, count = out.get(table_id, (region.base_row, rows))
            out[table_id] = (first, max(count, rows))
        return out
