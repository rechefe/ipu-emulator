"""The issue step: one slot op in, one immutable :class:`IssueFacts` record out.

Issuing a slot op does everything that happens before its handler runs, once:
the read operands are resolved, the slot's counter is bumped, a multiply is
probed for a MAC identity (a multiply by 1.0) and the op is attributed to at
most one declared ISA alias. The result is an :class:`IssueFacts` record. The
handler, the statistics and the observers all read that record; nothing about
the op in flight is kept in mutable fields on ``Ipu``.

:meth:`Issuer.execute` runs the handler with the record's resolved operands as
keyword arguments named after the spec operands (the CLAUDE.md invariant), and
brackets it with the cycle observers' ``issued`` and ``retired`` hooks
(:class:`~ipu_emu.cycle.CycleObserver`). While it runs, the record is
:attr:`Issuer.current` (``Cycle.current``), which is where the multiply
handlers read the identity verdict for their lane counts and their lane mask.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from ipu_emu.geometry import effective_mult_mask
from ipu_isa.constants import decode_dstructure

if TYPE_CHECKING:
    from ipu_emu.cycle import CycleObserver, IssuedOp
    from ipu_emu.ipu import Ipu, SlotPlan
    from ipu_emu.resolver import OperandResolver


@dataclass(frozen=True, slots=True)
class MultIdentity:
    """Whether a multiply is a declared MAC identity, and by which operand.

    Attributes:
        cr: The scalar multiplicand is a constant 1.0 taken from a CR.
        vector: A vector multiplicand is a declared ONES load.
    """

    cr: bool
    vector: bool

    @property
    def any(self) -> bool:
        """Whether the multiply is an identity by either operand."""
        return self.cr or self.vector

    @staticmethod
    def of(cr: bool, vector: bool) -> MultIdentity:
        """The shared instance for a verdict (there are four), so issue allocates none."""
        return _MULT_IDENTITIES[cr, vector]


_MULT_IDENTITIES = {
    (cr, vector): MultIdentity(cr=cr, vector=vector)
    for cr in (False, True)
    for vector in (False, True)
}


class IssueFacts(NamedTuple):
    """What issuing one slot op established. Immutable.

    A named tuple rather than a frozen dataclass: one is built per issued op,
    nine per cycle, and a named tuple is about three times cheaper to build
    (the cycle module's ``IssuedOp`` and ``Schedule`` make the same choice).

    Attributes:
        slot: The slot type (``"lr"`` for every LR sub-slot).
        plan: What dispatch needs for this (slot, opcode).
        encoded: The raw operand fields from the word, by operand name. Shared
            with the word's schedule, so never mutated.
        operands: The resolved values passed to the handler, by operand name.
        mult_identity: The identity verdict of a multiply; ``None`` for any op
            that is not a MULT (the mult slot's NOP included).
        alias: The declared ISA alias the op was attributed to, if any.
        lr_writes: The LRs an LR sub-slot writes, from the word's schedule
            (none, one, or an LRD pair); empty for every other slot.
        mask: The lane mask a masking MULT applies, at issue
            (:func:`~ipu_emu.geometry.effective_mult_mask`); ``None`` for any
            other op.
    """

    slot: str
    plan: SlotPlan
    encoded: Mapping[str, int]
    operands: Mapping[str, Any]
    mult_identity: MultIdentity | None
    alias: str | None
    lr_writes: tuple[int, ...]
    mask: int | None


_IssueFacts = IssueFacts
_new_facts = tuple.__new__


class Issuer:
    """Issues slot ops of one :class:`~ipu_emu.ipu.Ipu` and executes the results.

    Args:
        ipu: The engine whose handlers, snapshot and statistics the issuer
            uses. Held weakly: the engine owns its issuer.
        resolver: How read operands become register values
            (:mod:`ipu_emu.resolver`).
    """

    __slots__ = ("_ipu", "_resolve", "current", "observers")

    def __init__(self, ipu: Ipu, resolver: OperandResolver) -> None:
        self._ipu = weakref.ref(ipu)
        self._resolve = resolver.resolve
        #: The record of the op whose handler is running (or ran last); ``None``
        #: before the first execute.
        self.current: IssueFacts | None = None
        #: The observers told of every op executed, set by the cycle at the
        #: start of each cycle (``Cycle.observers``); empty outside one.
        self.observers: tuple[CycleObserver, ...] = ()

    def issue(self, op: IssuedOp, inst: Mapping[str, int]) -> IssueFacts:
        """Resolve one op's reads, bump its counter, probe identity, attribute an alias.

        Args:
            op: The slot op, from the word's schedule. Its operands are never
                mutated: the resolved values go into a copy.
            inst: The whole word, for alias constraints on co-issued slots.

        Returns:
            The op's fact record, for :meth:`execute`.
        """
        ipu = self._ipu()
        plan = op.plan
        ipu._view.plan = plan
        operands = (
            self._resolve(plan, op.operands, ipu._snapshot_buffer)
            if plan.reads
            else dict(op.operands)
        )
        if plan.stat is not None:
            stats = ipu.state.stats
            setattr(stats, plan.stat, getattr(stats, plan.stat) + 1)
        if plan.is_mult or plan.alias_checks:
            return self._probe(ipu, op, inst, operands)
        return _new_facts(
            _IssueFacts, (op.slot, plan, op.operands, operands, None, None, op.lr_writes, None)
        )

    @staticmethod
    def _probe(
        ipu: Ipu, op: IssuedOp, inst: Mapping[str, int], operands: dict[str, Any]
    ) -> IssueFacts:
        """The identity verdict and alias of an op whose reads are resolved: its record."""
        plan = op.plan
        encoded = op.operands
        identity = mask = None
        if plan.mask_cr_operand is not None:
            # Read now, before the handler, which applies this same mask, in
            # the views the instruction declares (ipu._view's plan is this op's).
            view = ipu._view
            mask = effective_mult_mask(
                view.source("r_mask"),
                decode_dstructure(view.cr(operands[plan.mask_cr_operand], plan.mask_cr_operand)),
                operands["mask_offset"],
                operands["mask_shift"],
            )
        if plan.is_mult:
            # MAC accounting. A multiply by 1.0 is how kernels emulate the
            # vector move/add/subtract the ISA lacks, so it occupies the
            # multiplier while retiring nothing.
            identity = _MULT_IDENTITIES[
                bool(plan.scalar_kind)
                and ipu._mult_scalar_is_one(
                    plan.scalar_kind, operands[plan.scalar_operand], plan.scalar_operand
                ),
                ipu._mult_vector_is_one(plan.inst_name, operands, encoded),
            ]
            if identity.cr or identity.vector:
                ipu.state.stats.mult_identity_cycles += 1
        alias = ipu._record_alias(plan, operands, inst, identity) if plan.alias_checks else None
        return IssueFacts(op.slot, plan, encoded, operands, identity, alias, op.lr_writes, mask)

    def dispatch(self, op: IssuedOp, inst: Mapping[str, int]) -> Any:
        """:meth:`issue` then :meth:`execute`, fused for the cycle's hot path.

        The same steps in the same order; only the call overhead is saved.

        Args:
            op: The slot op, from the word's schedule.
            inst: The whole word.

        Returns:
            The handler's return value.
        """
        ipu = self._ipu()
        plan = op.plan
        ipu._view.plan = plan
        encoded = op.operands
        operands = (
            self._resolve(plan, encoded, ipu._snapshot_buffer) if plan.reads else dict(encoded)
        )
        stat = plan.stat
        if stat is not None:
            stats = ipu.state.stats
            setattr(stats, stat, getattr(stats, stat) + 1)
        if plan.is_mult or plan.alias_checks:
            # The identity probe and alias attribution are written once, in
            # _probe, for both paths.
            facts = self._probe(ipu, op, inst, operands)
        else:
            facts = _new_facts(
                _IssueFacts, (op.slot, plan, encoded, operands, None, None, op.lr_writes, None)
            )
        self.current = facts
        observers = self.observers
        if observers:
            return self._observed(ipu, facts, observers)
        return plan.fn(ipu, **operands)

    @staticmethod
    def _observed(ipu: Ipu, facts: IssueFacts, observers: tuple[CycleObserver, ...]) -> Any:
        """Run the op's handler between every observer's ``issued`` and ``retired``."""
        for observer in observers:
            observer.issued(facts)
        result = facts.plan.fn(ipu, **facts.operands)
        for observer in observers:
            observer.retired(facts, result)
        return result

    def dispatch_all(self, ops: Iterable[IssuedOp], inst: Mapping[str, int]) -> None:
        """Issue every op first, then execute them in order (the LR sub-slots).

        So every op's reads are resolved before any of them writes: a read
        that raises leaves nothing written.

        Args:
            ops: The ops, in execution order.
            inst: The whole word.
        """
        ipu = self._ipu()
        view = ipu._view
        resolve = self._resolve
        issued = []
        for op in ops:
            plan = op.plan
            if plan.stat is not None or plan.is_mult or plan.alias_checks:
                issued.append(self.issue(op, inst))
                continue
            # The LR sub-slots' hot path: just the reads (as issue() does).
            view.plan = plan
            encoded = op.operands
            operands = resolve(plan, encoded, ipu._snapshot_buffer) if plan.reads else dict(encoded)
            issued.append(
                _new_facts(
                    _IssueFacts,
                    (op.slot, plan, encoded, operands, None, None, op.lr_writes, None),
                )
            )
        observers = self.observers
        for facts in issued:
            self.current = facts
            view.plan = facts.plan
            if observers:
                self._observed(ipu, facts, observers)
            else:
                facts.plan.fn(ipu, **facts.operands)

    def execute(self, facts: IssueFacts) -> Any:
        """Run the op's handler with its resolved operands, between the observers' hooks.

        Args:
            facts: A record from :meth:`issue`.

        Returns:
            The handler's return value (a ``CycleOutcome`` for BREAK, else ``None``).
        """
        ipu = self._ipu()
        self.current = facts
        ipu._view.plan = facts.plan
        if self.observers:
            return self._observed(ipu, facts, self.observers)
        return facts.plan.fn(ipu, **facts.operands)
