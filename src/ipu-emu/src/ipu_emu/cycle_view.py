"""How a handler reads registers: :class:`CycleView`, through the declared view.

The ISA timing contract (``ipu_isa.timing``, Q32) declares, for every register
an instruction reads, which value the read sees: the cycle-start snapshot or
the forwarded (live) one. :class:`CycleView` serves each handler read from the
register file the declaration names (E04-12), and decodes the DS and QF halves
an operand selects (ADR 0007). :mod:`ipu_emu.cycle` owns the instance the
handlers of one engine read through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipu_emu.cr_contents import ds_of, half_of, qf_of
from ipu_emu.errors import EmulatorError, UndeclaredRead
from ipu_emu.geometry import WIDE_REGISTERS

if TYPE_CHECKING:
    from ipu_emu.ipu import SlotPlan
    from ipu_emu.ipu_state import IpuState
    from ipu_emu.regfile import RegFile
    from ipu_isa.cr_halves import DataStructure
    from ipu_isa.nx8 import Nx8Format

__all__ = ["CycleView"]


#: Emulator-only storage that holds a declared register in a 4-byte element
#: format: a read of it has the view declared for that register.
_DECLARED_AS = {wide: declared for declared, wide in WIDE_REGISTERS.items()}


class CycleView:
    """How a handler reads registers: through the view its instruction declares.

    The ISA timing contract (``ipu_isa.timing``, Q32) declares, for every
    register an instruction reads, which value the read sees: the cycle-start
    snapshot or the forwarded (live) one. The declaration is the instruction's
    ``read`` operands and ``implicit_reads``; :attr:`SlotPlan.views
    <ipu_emu.ipu.SlotPlan>` holds it per plan. A handler names the register
    (and, where it matters, the operand that selects it) and this class picks
    the register file, so changing when a read happens is an edit to the spec,
    not to a handler.

    The issuer sets :attr:`plan` before it issues each op and again before its
    handler runs; :class:`Cycle` sets :attr:`snapshot` when it takes the
    cycle's snapshot. Writes never go through here: handlers write the live
    file (``RegFile.mutable`` and the ``set_*`` accessors).

    Args:
        state: The machine state whose live register file is read.
    """

    __slots__ = ("_snapshot", "_state", "plan")

    def __init__(self, state: IpuState) -> None:
        self._state = state
        self._snapshot: RegFile | None = None
        #: The plan of the op being issued or executed; ``None`` before the first.
        self.plan: SlotPlan | None = None

    @property
    def snapshot(self) -> RegFile:
        """The cycle-start register file (read only by handlers).

        Raises:
            EmulatorError: No cycle has taken a snapshot yet.
        """
        snapshot = self._snapshot
        if snapshot is None:
            raise EmulatorError("no register snapshot: a handler ran outside a cycle")
        return snapshot

    @snapshot.setter
    def snapshot(self, buffer: RegFile) -> None:
        self._snapshot = buffer

    @property
    def live(self) -> RegFile:
        """The register file this cycle writes (the forwarded view)."""
        return self._state.regfile

    def source(self, register: str, via: str | None = None) -> RegFile:
        """The register file a read of *register* comes from: snapshot or live.

        Args:
            register: A ``REGISTER_DEFINITIONS`` name (``"r_mask"``, ``"lr"``,
                ...); a wide-vector debug register reads as the register it
                stands in for.
            via: The operand that selects the register's index, as declared
                (``"src"``, ``"reshape_mask"``); ``None`` when the
                instruction's declared reads of *register* all share one view.

        Returns:
            The cycle-start snapshot or the live register file. Only its read
            methods may be called.

        Raises:
            UndeclaredRead: The instruction does not declare this read.
            EmulatorError: No op is in flight (a handler called outside a
                cycle), or no cycle has taken a snapshot.
        """
        plan = self.plan
        if plan is None:
            raise EmulatorError(
                f"read of {register} with no op issued: a handler ran outside a cycle"
            )
        try:
            from_snapshot = plan.views[_DECLARED_AS.get(register, register), via]
        except KeyError:
            raise self._undeclared(plan, register, via) from None
        return self.snapshot if from_snapshot else self._state.regfile

    def read(self, register: str, via: str | None = None) -> memoryview:
        """Register *register*'s bytes, read-only, in its declared view.

        Args:
            register: See :meth:`source`.
            via: See :meth:`source`.

        Returns:
            A read-only view of the register's bytes; constant-ones provenance
            is kept.
        """
        return self.source(register, via).view(register)

    def lr(self, index: int, via: str | None = None) -> int:
        """LR *index* in the view declared for ``("lr", via)``."""
        return self.source("lr", via).get_lr(index)

    def cr(self, index: int, via: str | None = None) -> int:
        """CR *index* in the view declared for ``("cr", via)``."""
        return self.source("cr", via).get_cr(index)

    def half(self, code: int, via: str) -> int:
        """The 16-bit CR half a ``HalfCrIdx`` operand *code* names, read as
        :meth:`cr` reads the CR (ADR 0007)."""
        return half_of(code, self.cr(code >> 1, via))

    def ds(self, code: int, via: str = "ds") -> DataStructure:
        """The DS the half-CR operand *code* (``via``) holds.

        Raises:
            CrContentError: The half is no valid DS.
        """
        return ds_of(self.half(code, via))

    def qf(self, code: int, via: str = "qf") -> Nx8Format:
        """The NX8 format the half-CR operand *code* (``via``) names.

        Raises:
            CrContentError: The half is no valid QF.
        """
        return qf_of(self.half(code, via))

    @staticmethod
    def _undeclared(plan: SlotPlan, register: str, via: str | None) -> UndeclaredRead:
        where = f" via {via!r}" if via is not None else ""
        declared = sorted(f"{reg} via {op!r}" if op else reg for reg, op in plan.views)
        return UndeclaredRead(
            f"{plan.inst_name} reads {register}{where}, which it does not declare "
            f"(declared reads: {declared}); add it to the instruction's read "
            "operands or implicit_reads in instruction_spec.py"
        )
