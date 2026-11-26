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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import numpy as np
import numpy.typing as npt

from ipu_emu.geometry import effective_mult_mask
from ipu_emu.slots import mult

if TYPE_CHECKING:
    from ipu_emu.cycle import CycleObserver, IssuedOp
    from ipu_emu.cycle_view import CycleView
    from ipu_emu.ipu import Ipu, SlotPlan, _AliasCheck
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


_MULT_IDENTITIES: Final = {
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


_IssueFacts: Final = IssueFacts
_new_facts: Final = tuple.__new__


class Issuer:
    """Issues slot ops of one :class:`~ipu_emu.ipu.Ipu` and executes the results.

    Args:
        ipu: The engine whose handlers, snapshot and statistics the issuer
            uses. Held weakly: the engine owns its issuer.
        resolver: How read operands become register values
            (:mod:`ipu_emu.resolver`).
    """

    __slots__ = (
        "_ipu",
        "_resolve",
        "_view",
        "current",
        "mult_product",
        "observers",
        "prior_product",
    )

    def __init__(self, ipu: Ipu, resolver: OperandResolver) -> None:
        # The engine, held until close() (Ipu.close): an engine and its
        # issuer refer to each other.
        self._ipu: Ipu | None = ipu
        self._resolve = resolver.resolve
        # The engine's view, which it creates first and never replaces: held
        # so each op sets its plan and reads the snapshot without two
        # property calls on the engine (``ipu.view``, ``ipu.snapshot_buffer``,
        # whose buffer ``Ipu.bind_snapshot`` gives the view too).
        self._view: CycleView = ipu.view
        #: The record of the op whose handler is running (or ran last); ``None``
        #: before the first execute.
        self.current: IssueFacts | None = None
        #: The MULT -> ACC hand-over of one word (the fused path): the float32
        #: lanes a MULT handler stored in MULT_RES (``mult_product``), which
        #: become ``prior_product`` for the next op dispatched and are then
        #: dropped. Every other way an op runs drops both.
        self.mult_product: npt.NDArray[np.float32] | None = None
        self.prior_product: npt.NDArray[np.float32] | None = None
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
        ipu = self._ipu
        assert ipu is not None
        plan = op.plan
        self._view.plan = plan
        encoded = op.operands
        assert encoded is not None
        operands = (
            self._resolve(plan, encoded, self._view._snapshot)  # type: ignore[arg-type]
            if plan.reads
            else dict(encoded)
        )
        if plan.stat is not None:
            stats = ipu.state.stats
            setattr(stats, plan.stat, getattr(stats, plan.stat) + 1)
        if plan.is_mult or plan.alias_checks:
            return self._probe(ipu, op, inst, operands)
        return _new_facts(
            _IssueFacts, (op.slot, plan, encoded, operands, None, None, op.lr_writes, None)
        )

    @staticmethod
    def _probe(
        ipu: Ipu, op: IssuedOp, inst: Mapping[str, int], operands: dict[str, Any]
    ) -> IssueFacts:
        """The identity verdict and alias of an op whose reads are resolved: its record."""
        plan = op.plan
        encoded = op.operands
        assert encoded is not None
        identity: MultIdentity | None = None
        mask: int | None = None
        if plan.mask_cr_operand is not None:
            # Read now, before the handler, which applies this same mask, in
            # the views the instruction declares (ipu.view's plan is this op's).
            view = ipu.view
            mask = effective_mult_mask(
                view.source("r_mask"),
                view.ds(operands[plan.mask_cr_operand], plan.mask_cr_operand),
                operands["mask_offset"],
                operands["mask_shift"],
            )
        if plan.is_mult:
            # MAC accounting. A multiply by 1.0 is how kernels emulate the
            # vector move/add/subtract the ISA lacks, so it occupies the
            # multiplier while retiring nothing.
            identity = _MULT_IDENTITIES[
                bool(plan.scalar_kind)
                and mult.scalar_is_one(
                    ipu,
                    plan.scalar_kind,
                    operands[plan.scalar_operand],  # type: ignore[index]
                    plan.scalar_operand,  # type: ignore[arg-type]
                ),
                mult.vector_is_one(ipu, plan.inst_name, operands, encoded),
            ]
            if identity.cr or identity.vector:
                ipu.state.stats.mult_identity_cycles += 1
        alias = _record_alias(ipu, plan, operands, inst, identity) if plan.alias_checks else None
        return IssueFacts(op.slot, plan, encoded, operands, identity, alias, op.lr_writes, mask)

    def operands_of(self, op: IssuedOp) -> dict[str, Any]:
        """*op*'s operands as its handler would get them now, with nothing
        issued: no counter, no observer (the precise-fault check, E04-20).
        Also points the cycle view at *op*'s plan, as issuing does."""
        ipu = self._ipu
        assert ipu is not None
        plan = op.plan
        self._view.plan = plan
        encoded = op.operands
        assert encoded is not None
        if plan.reads:
            return self._resolve(plan, encoded, self._view._snapshot)  # type: ignore[arg-type]
        return dict(encoded)

    def dispatch(self, op: IssuedOp, inst: Mapping[str, int]) -> Any:
        """:meth:`issue` then :meth:`execute`, fused for the cycle's hot path.

        The same steps in the same order; only the call overhead is saved.

        Args:
            op: The slot op, from the word's schedule.
            inst: The whole word.

        Returns:
            The handler's return value.
        """
        ipu = self._ipu
        assert ipu is not None
        # The previous op's MULT product reaches this op only (see mult_product).
        self.prior_product = self.mult_product
        self.mult_product = None
        plan = op.plan
        self._view.plan = plan
        encoded = op.operands
        assert encoded is not None
        operands = (
            self._resolve(plan, encoded, self._view._snapshot)  # type: ignore[arg-type]
            if plan.reads
            else dict(encoded)
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
            self.prior_product = None  # observed ops take the general path
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
        ipu = self._ipu
        assert ipu is not None
        self.mult_product = self.prior_product = None
        view = self._view
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
            assert encoded is not None
            operands = (
                resolve(plan, encoded, self._view._snapshot)  # type: ignore[arg-type]
                if plan.reads
                else dict(encoded)
            )
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
        self.mult_product = self.prior_product = None
        ipu = self._ipu
        assert ipu is not None
        self.current = facts
        self._view.plan = facts.plan
        if self.observers:
            return self._observed(ipu, facts, self.observers)
        return facts.plan.fn(ipu, **facts.operands)

    def close(self) -> None:
        """Let go of the engine (see ``Ipu.close``) and the op records: a
        closed issuer cannot dispatch another op."""
        self._ipu = None
        self.current = None
        self.observers = ()
        self.mult_product = self.prior_product = None


_FIRST_HARDWARE_ONLY: dict[int, tuple[SlotPlan, _AliasCheck | None]] = {}


def _first_hardware_only(plan: SlotPlan) -> _AliasCheck | None:
    """The first of *plan*'s alias checks that is hardware-only, or ``None`` (cached per plan)."""
    entry = _FIRST_HARDWARE_ONLY.get(id(plan))
    if entry is None or entry[0] is not plan:
        first = next((check for check in plan.alias_checks if check.hardware_only), None)
        entry = (plan, first)  # holds the plan, so its id is not reused
        _FIRST_HARDWARE_ONLY[id(plan)] = entry
    return entry[1]


def _record_alias(
    ipu: Ipu,
    plan: SlotPlan,
    kwargs: Mapping[str, Any],
    inst: Mapping[str, int],
    identity: MultIdentity | None,
) -> str | None:
    """Attribute this instruction to at most one declared ISA alias.

    First match wins, so hits partition ``mult_identity_cycles`` instead of
    double-counting idioms whose constraints overlap.

    Args:
        ipu: The engine.
        plan: The instruction's plan.
        kwargs: Its resolved operands.
        inst: The whole word, for co-issued slot constraints.
        identity: Its identity verdict (``None`` unless it is a MULT).

    Returns:
        The alias ID counted, or ``None``.
    """
    if identity is None or not (identity.vector or identity.cr):
        # Every check that is not hardware-only needs one of the two verdicts,
        # so the first hardware-only check (which always matches) is the hit.
        first = _first_hardware_only(plan)
        if first is None:
            return None
        ipu.state.stats.alias_hits[first.alias_id] += 1
        return first.alias_id
    for check in plan.alias_checks:
        if not check.hardware_only:
            if identity is None or not (identity.vector if check.vector_only else identity.cr):
                continue
            if check.masked_window and not mult.alias_masked_window(ipu, kwargs):
                continue
            if check.imm_operand is not None and (kwargs[check.imm_operand] == check.imm_ne):
                continue
            if check.co_field is not None and (inst[check.co_field] not in check.co_opcodes):
                continue
        ipu.state.stats.alias_hits[check.alias_id] += 1
        return check.alias_id
    return None
